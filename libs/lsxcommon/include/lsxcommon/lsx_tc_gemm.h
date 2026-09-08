// lsx_tc_gemm.h - Blackwell tensor-core NVFP4 W4A4 fast path (audit Phase 3,
// F-01). Compiled EXCLUSIVELY for compute_120a/sm_120a + compute_121a/sm_121a:
// the kind::mxf4nvf4 PTX qualifier does not assemble on plain sm_120 targets
// (PTX ISA 9.3). All entry points are stream-ordered and allocate nothing.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon {
namespace lsx_tc {

// Quantize FP16 activations to NVFP4 (per-16 E4M3 power-of-2 scale +
// E2M1 nibble pairs), matching the repo's canonical weight-side codec
// (schema.h float_to_e4m3_p2 / nvfp4_codec.h decode). Output layouts:
// d_packed [seq, K/2] row-major nibble pairs (low nibble = even k),
// d_scales [seq, K/16] row-major, one byte per 16-element block.
// Buffers must be preallocated by the caller (device memory).
cudaError_t LsxTcQuantizeActsNvfp4(cudaStream_t stream, const __half *x,
                                   int seq, int K, uint8_t *d_packed,
                                   uint8_t *d_scales);

// BF16 twin of LsxTcQuantizeActsNvfp4: identical codec + output layout.
// Kimi hidden states are bf16; quantizing bf16->fp16->nvfp4 would add a
// pointless host-visible pass.
cudaError_t LsxTcQuantizeActsNvfp4Bf16(cudaStream_t stream,
                                       const __nv_bfloat16 *x, int seq,
                                       int K, uint8_t *d_packed,
                                       uint8_t *d_scales);
// Tensor-core NVFP4 GEMM: y = (quant(x) @ dequant(w)) * w_global_scale.
// W4A4: activations are quantized on the fly into stream-ordered scratch
// buffers; weights are consumed in the EXISTING repo layout ([N,K] nibbles +
// [N,K/16] pow2-E4M3 scales) -- no repack. Requires sm_120a/sm_121a device
// (checked at runtime; returns cudaErrorNotSupported otherwise). Numerics:
// tolerance-class (hardware accumulation order unspecified), NOT bit-exact
// vs the scalar path.
cudaError_t RunLsxTcNvfp4Gemm(cudaStream_t stream, const __half *x_fp16,
                               const uint8_t *w_packed,
                               const uint8_t *w_scales, float w_global_scale,
                               __half *y, int seq, int K, int N,
                               bool acc = false,
                               const float *seg_wgs_dev = nullptr,
                               const int *seg_bounds_dev = nullptr);

// TC GEMM over caller-owned quantized activations: skips the internal
// ActQuant pass + scratch allocs. Producer kernels that emit both fp16 and
// NVFP4 forms use this to avoid a separate quant launch per GEMM.
cudaError_t RunLsxTcGemmPreQuant(cudaStream_t stream, const uint8_t *ap,
                                  const uint8_t *as, const uint8_t *w_packed,
                                  const uint8_t *w_scales, float w_global_scale,
                                  __half *y, int seq, int K, int N,
                                  bool acc = false,
                                  const float *seg_wgs_dev = nullptr,
                                  const int *seg_bounds_dev = nullptr);

// Builds (once, cached) the N-concatenation of four NVFP4 weight matrices
// sharing one input K, so several projections can run as a single GEMM.
// All segments must have identical K (k_half = K/2, k_chunks = K/16).
// Returns cached device buffers; they live for the process lifetime.
cudaError_t LsxTcBuildFusedGemmWeights(cudaStream_t stream, const uint8_t *p1,
                                  const uint8_t *s1, int n1, const uint8_t *p2,
                                  const uint8_t *s2, int n2, const uint8_t *p3,
                                  const uint8_t *s3, int n3, const uint8_t *p4,
                                  const uint8_t *s4, int n4, int k_half,
                                  int k_chunks, float g1, float g2, float g3,
                                  float g4, const uint8_t **packed_out,
                                  const uint8_t **scales_out,
                                  const float **wgs_out,
                                  const int **bounds_out, int *n_total);

// B3 capability probe: encodes a 2D CUtensorMap over a synthetic weight-like
// buffer, performs ONE cp.async.bulk.tensor.2d into dynamic shared memory
// behind an mbarrier and verifies the tile contents. Returns cudaSuccess iff
// TMA bulk-tensor staging works on this device. Stream-ordered, allocates
// nothing persistent.
cudaError_t LsxProbeTmaBulkCopy(cudaStream_t stream);

// One slot of the batched routed-expert TC chain: cache slot + per-segment
// global scales. Layout-identical to qwen_moe::QwenMoeGvSlotDev (pinned by
// static_assert in qwen_moe_inference.cu) so the existing device gather
// kernel output can be consumed directly.
struct alignas(8) LsxTcBatchSlot {
    int32_t dev_slot;
    float gs1, gs3, gs2;
    float coeff;
};

// Host mirror of TcGemmKernel's split-K selection rule (seq==1 decode):
// same inputs -> same split_k -> same slice math. Pure arithmetic.
int LsxTcBatchSplitK(int K, int N);

// sm_12x runtime probe (cached): mxf4nvf4 TC path availability.
bool LsxTcRuntimeSupported();

// Persistent fused-GEMM output scratch keyed by the cached packed-weight
// pointer (process-lifetime; the LsxTcBuildFusedGemmWeights cache guarantees key
// stability). Returns nullptr on allocation failure.
__half *LsxTcFusedOutputScratch(const void *packed_key, int n_half);

// Batched seq==1 TC GEMM over routed-expert cache slots (gate|up fused
// slab, [N=2I, K=H] per slot): grid = (ceil(N/64), 1, split_k * k_slots).
// Per-slot weight base = pack_base + dev_slot*pack_stride (+w_off);
// scales analogously (+s_off). y = y + slot*N (per-slot rows).
// Bit-exact contract: identical math per slot as RunLsxTcGemmPreQuant
// against the same pointers (pinned by test_qwen_moe_kernels
// "[qwen-moe-tc-batch]"). splitk_partials: caller buffer of
// split_k * k_slots * N floats (nullptr OK when LsxTcBatchSplitK == 1).
cudaError_t RunLsxTcGemmBatch13(cudaStream_t stream, const uint8_t *ap,
                                 const uint8_t *as, const uint8_t *pack_base,
                                 const uint8_t *scale_base,
                                 int64_t pack_stride, int64_t scale_stride,
                                 const LsxTcBatchSlot *slots_dev,
                                 int k_slots, int K, int N, __half *y,
                                 float *splitk_partials);

// Batched down projection: per-slot w2 slab lives at pack + w13_bytes,
// scales at scale + s13_bytes; scalar global scale gs2 from the table.
cudaError_t RunLsxTcGemmBatchDown(cudaStream_t stream, const uint8_t *ap,
                                   const uint8_t *as,
                                   const uint8_t *pack_base,
                                   const uint8_t *scale_base,
                                   int64_t pack_stride, int64_t scale_stride,
                                   const LsxTcBatchSlot *slots_dev,
                                   int k_slots, int64_t w13_bytes,
                                   int64_t s13_bytes, int K, int N,
                                   __half *y, float *splitk_partials);

} // namespace lsx_tc
} // namespace lsxcommon
