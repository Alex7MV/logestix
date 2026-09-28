#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/qwen_moe_expert_cache.h"

namespace lsxcommon::qwen_moe {

// One routed slot: selected expert id + renormalized routing weight.
// Same layout contract as GlmRoutedSlot (8 bytes, one grid slot per token/k).
struct alignas(8) QwenMoeRoutedSlot {
    int32_t eid;
    float coeff;
};

// HF-parity MoE router (modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter):
// fp32 logits = router_w[e] . x[t] -> softmax over ALL experts ->
// top-K probabilities (lower-index tie-break) -> renormalize by their sum.
//
// Launch config: <<<seq, 128, E * sizeof(float), stream>>>.
// Shared-memory RMW isolation: selection/writes happen on thread 0 only.
//
// Determinism: same input bits -> same output bits on repeated runs.
cudaError_t RunQwenMoeRouterTopK(cudaStream_t stream,
                                 const __nv_bfloat16 *router_w,  // [E,H]
                                 const __half *normed,           // [seq,H]
                                 int seq, int E, int K, int H,
                                 QwenMoeRoutedSlot *d_slots);    // [seq*K]

// Forward decls from the loader side (pinned store + LRU device cache).
struct QwenMoeExpertStore;
struct QwenMoePinnedWeights;

// Per-slot device table for the batched routed GEMV chain (built host-side
// in the handshake flow or on device by the no-handshake gather kernel;
// 8-byte-aligned 24B payload).
struct alignas(8) QwenMoeGvSlotDev {
    int32_t dev_slot;
    float gs1, gs3, gs2;
    float coeff;
};

// Processes one token's K routed slots (ascending-k deterministic fp32
// accumulation): d_acc += coeff_k * FFN_eidk(x). Stages experts H2D on cache
// misses via EnsureQwenMoeExpertResident. Scratch: d_yg/d_yu [I], d_y2 [H].
//
// Copy-stream mode: when `copy_stream` != nullptr, cache-miss H2D staging is
// enqueued there instead of `stream`; `ev_uploads` (K caller-owned disabled-
// timing events) fence each upload and `ev_front` must be recorded-or-null —
// the implementation records it on `stream` before staging so the copy stream
// never overwrites a slot still being read by previously enqueued compute.
// Numerics are unchanged (same kernels, same accumulation order).
cudaError_t RunQwenMoeRoutedFFN(cudaStream_t stream,
                                QwenMoeExpertStore &store,
                                QwenMoeExpertDeviceCache &cache, int layer,
                                const QwenMoeRoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error,
                                cudaStream_t copy_stream = nullptr,
                                cudaEvent_t *ev_uploads = nullptr,
                                cudaEvent_t ev_front = nullptr,
                                const uint8_t *xq = nullptr,
                                const uint8_t *xqs = nullptr,
                                uint8_t *d_qpack_scratch = nullptr,
                                uint8_t *d_qscales_scratch = nullptr,
                                __half *ygu_batch = nullptr,
                                __half *hbuf_batch = nullptr,
                                __half *ypart_batch = nullptr,
                                QwenMoeGvSlotDev *d_gemv_tbl = nullptr);

// Casts the routed-accumulation buffer to FP16.
cudaError_t RunQwenMoeAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n);

// Test/parity helper: seeds the MoE fp32 accumulator from the fp16 residual
// (acc[i] = f32(resid[i])) -- the exact qwen_moe_init_acc_kernel math that
// the driver runs before the legacy routed chain.
cudaError_t RunQwenMoeAccInitForTest(cudaStream_t stream,
                                     const __half *d_resid, float *d_acc,
                                     int n);

// Batched routed-expert chain for seq==1 GEMV mode: replaces the per-slot
// {w1,w3,w2} launch loop (K x 5 kernels/layer) with 4 launches total:
// w13 GEMV batch ([2I,K]*x per slot), elementwise swiglu, down GEMV batch
// ([H,I]*h per slot -> per-slot partials), ordered coeff reduce
// acc[r] += sum_s coeff_s * ypart[s][r] (ascending s == legacy add order).
//
// Device addressing mirrors the STAGED FUSED cache slot layout: per-slot
// packed base = d_pack + dev_slot*pack_stride with [w13][w2] packed and
// [s13][s2] scales separated by w13_bytes / s13_bytes respectively.
// Requires H % 32 == 0 && I % 32 == 0 (GEMV row-alignment contract).
// Scratch ygu/hbuf/ypart are caller-owned [Ktop*...] buffers. Deterministic:
// fixed partitioning + fixed-order reduce.
cudaError_t RunQwenMoeRoutedGemmVBatch(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const QwenMoeGvSlotDev *slots_host_tbl, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

// Device-table variant of the batched chain (round-20c/20e): launches the
// four GEMV-batch kernels directly against an already-resident device slot
// table. External linkage so the graph-capture transparency test can pin
// launch semantics; the decode driver uses it for both the chain and the
// per-layer graph path.
cudaError_t RunQwenMoeRoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const QwenMoeGvSlotDev *slots_dev, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

// Per-expert static device meta for no-handshake decode: definition lives
// in qwen_moe_load.h (owned by QwenMoeExpertStore).

// Composes the batch GEMV slot table from device-side routing results:
// tbl[k] = {meta[d_slots[k].eid].*, coeff = d_slots[k].coeff}. Used by the
// no-handshake decode path where the host never reads back router output.
// Launch config: <<<1, 32>>>, K <= 16.
cudaError_t RunQwenMoeGemvGatherTable(
    cudaStream_t stream,
    const QwenMoeRoutedSlot *d_slots, int k_slots,
    const QwenMoeExpertMetaDev *d_meta,   // [num_experts] slice of the layer
    QwenMoeGvSlotDev *d_tbl_out);         // [k_slots]

// True once every routed expert is resident and mirrored to device meta
// (full-residency steady state): unlocks the router-readback-free decode
// path. One-time lazy upload of the mirror on first readiness.
bool QwenMoeGemvNohandshake(const QwenMoeExpertStore &store);

// Batched decode swiglu-quant: per slot s, gate = ygu[s*2I + j],
// up = ygu[s*2I + I + j]; h[s] = silu(gate)*up (FP16) + NVFP4 encode of h.
// Bit-identical to per-slot RunQwenSwigluQuant (same 16-group math).
cudaError_t RunQwenMoeSwigluQuantBatch(cudaStream_t stream,
                                       const __half *ygu, int k_slots, int I,
                                       __half *h, uint8_t *packed,
                                       uint8_t *scales);

// Batched TC routed-expert chain (no host readback): gather (device table)
// -> w13 batch TC GEMM -> batched swiglu-quant -> w2 batch TC GEMM ->
// ascending coeff reduce. d_acc = f32(d_resid) + sum_k coeff_k * FFN_k(x)
// -- the identical fp32 add order as the legacy chain (acc-init folded into
// the reduce; legacy: acc = f32(resid) then += coeff_k * y_k, k ascending).
// Requires TC fast_math, full residency mirror (QwenMoeGemvNohandshake),
// I % 16 == 0. splitk13: caller buffer for the w13 split-K partials
// (QwenTcBatchSplitK(H, 2*I) * K * 2I floats; nullptr OK when the helper
// returns 1). d_tbl: [16] reused gather output (QwenMoeGvSlotDev).
cudaError_t RunQwenMoeRoutedTcBatch(
    cudaStream_t stream, QwenMoeExpertStore &store,
    const QwenMoeExpertDeviceCache &cache, int layer,
    const QwenMoeRoutedSlot *d_routes, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, const __half *d_resid, float *d_acc,
    __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13,
    QwenMoeGvSlotDev *d_tbl);

// Device-table variant of the batched TC routed chain for streaming
// (partial-residency) models whose router-readback handshake already knows
// the per-slot dev_slot and per-segment global scales on the host: skips the
// device gather and runs w13 batch TC GEMM -> batched swiglu-quant -> w2
// batch TC GEMM -> ascending coeff reduce against a caller-supplied table
// (layout-identical to lsx_tc::LsxTcBatchSlot). Same numerics as
// RunQwenMoeRoutedTcBatch (bit-exact per slot). The caller must have staged
// every routed slot and zero-seeded d_acc.
cudaError_t RunQwenMoeRoutedTcBatchTbl(
    cudaStream_t stream, const QwenMoeExpertDeviceCache &cache,
    const QwenMoeGvSlotDev *d_tbl, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, int64_t w13_bytes, int64_t s13_bytes,
    float *d_acc, __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13);

// Host-side EOS truncation (decode loop runs to max_new without per-step
// EOS readback). Keeps the EOS token (legacy break semantics), returns
// true when a cut happened. eos_id < 0 disables.
bool TruncateTokensAtEos(std::vector<int32_t> &tokens, int eos_id);

// MoE FFN epilogue (HF parity): out = routed_acc + sigmoid(gate.x) * shared.
// d_routed_acc: [seq,H] fp32; scratch_i [seq,SI], scratch_h [seq,H];
// d_gate_val [seq] fp32 workspace. sw1=gate_proj [SI,H], sw3=up_proj,
// sw2=down_proj [H,SI].
cudaError_t RunQwenMoeSharedCombine(
    cudaStream_t stream, const lsxcommon::qwen::QwenNvfp4Weight &sw1,
    const lsxcommon::qwen::QwenNvfp4Weight &sw2,
    const lsxcommon::qwen::QwenNvfp4Weight &sw3,
    const __half *d_shared_gate, const __half *d_x, int seq, int H, int SI,
    const float *d_routed_acc, __half *d_scratch_i, __half *d_scratch_h,
    float *d_gate_val, __half *d_out, const uint8_t *xq = nullptr,
    const uint8_t *xqs = nullptr);

// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct QwenMoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch);
cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes);
void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch);

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr); mirrors RunQwenGeneration's per-token
// handshake and final-logits publication.
cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    QwenMoeExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::QwenKvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace lsxcommon::qwen_moe
