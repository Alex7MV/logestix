// qwen_tc_gemm.h - Blackwell tensor-core NVFP4 W4A4 fast path (audit Phase 3,
// F-01). Compiled EXCLUSIVELY for compute_120a/sm_120a + compute_121a/sm_121a:
// the kind::mxf4nvf4 PTX qualifier does not assemble on plain sm_120 targets
// (PTX ISA 9.3). All entry points are stream-ordered and allocate nothing.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon {
namespace qwen {
namespace tc {

// Quantize FP16 activations to NVFP4 (per-16 E4M3 power-of-2 scale +
// E2M1 nibble pairs), matching the repo's canonical weight-side codec
// (schema.h float_to_e4m3_p2 / nvfp4_codec.h decode). Output layouts:
// d_packed [seq, K/2] row-major nibble pairs (low nibble = even k),
// d_scales [seq, K/16] row-major, one byte per 16-element block.
// Buffers must be preallocated by the caller (device memory).
cudaError_t QuantizeActsNvfp4(cudaStream_t stream, const __half *x, int seq,
                              int K, uint8_t *d_packed, uint8_t *d_scales);

// Tensor-core NVFP4 GEMM: y = (quant(x) @ dequant(w)) * w_global_scale.
// W4A4: activations are quantized on the fly into stream-ordered scratch
// buffers; weights are consumed in the EXISTING repo layout ([N,K] nibbles +
// [N,K/16] pow2-E4M3 scales) -- no repack. Requires sm_120a/sm_121a device
// (checked at runtime; returns cudaErrorNotSupported otherwise). Numerics:
// tolerance-class (hardware accumulation order unspecified), NOT bit-exact
// vs the scalar path.
cudaError_t RunQwenTcNvfp4Gemm(cudaStream_t stream, const __half *x_fp16,
                               const uint8_t *w_packed,
                               const uint8_t *w_scales, float w_global_scale,
                               __half *y, int seq, int K, int N,
                               bool acc = false,
                               const float *seg_wgs_dev = nullptr,
                               const int *seg_bounds_dev = nullptr);

// Builds (once, cached) the N-concatenation of four NVFP4 weight matrices
// sharing one input K, so several projections can run as a single GEMM.
// All segments must have identical K (k_half = K/2, k_chunks = K/16).
// Returns cached device buffers; they live for the process lifetime.
cudaError_t BuildFusedGemmWeights(cudaStream_t stream, const uint8_t *p1,
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
cudaError_t ProbeTmaBulkCopy(cudaStream_t stream);

} // namespace tc
} // namespace qwen
} // namespace lsxcommon
