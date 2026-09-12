#pragma once

#include "lsxcommon/kimi_moe_expert_cache.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/lsx_tc_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace lsxcommon {
namespace kimi {

// Dequantizes compressed-tensors INT4 group-32 weights to BF16 bits.
// d_packed: int32 words, 8 nibbles each (low nibble = element first),
//   row-major [rows, cols/8].
// d_scales_bf16: BF16 bits [rows, cols/32], one scale per group of 32 along
//   the input (cols) dimension.
// d_out_bf16: BF16 bits [rows, cols]. Stream-async; no synchronization.
cudaError_t KimiInt4DequantToBf16(const int *d_packed,
                                  const uint16_t *d_scales_bf16,
                                  int rows, int cols,
                                  uint16_t *d_out_bf16,
                                  cudaStream_t stream);

// Kimi noaux_tc router: sigmoid scoring + e_score_correction_bias selection,
// top-k with lower-id tie-break, norm_topk_prob renormalization and
// routed-scaling. d_scores/d_bias: FP32 [n_routed]. Outputs: d_out_ids int
// [top_k], d_out_weights FP32 [top_k]. Stream-async.
//
// The host variant (KimiRouterTopKHost) computes the identical math on CPU
// for tests/fallbacks; both share the tie-break rule so ids match exactly.
void KimiRouterTopKHost(const float *scores, const float *bias, int n_routed,
                        int top_k, int *out_ids, float *out_weights);

cudaError_t KimiRouterTopK(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_out_ids, float *d_out_weights,
                           cudaStream_t stream);

// ── Split-phase router handshake (F-11) ────────────────────────────────────
// Begin: top-k kernel + async D2H into the caller's PINNED h_ids/h_w +
// event record on the stream. Finish: busy-poll that event (event-scoped
// fence; never drains unrelated stream work). Bitwise-identical results to
// the fused KimiRouterTopK + cudaStreamSynchronize path.
cudaError_t KimiRouteBegin(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_top_ids, float *d_top_w,
                           int *h_ids, float *h_w,
                           cudaEvent_t ev, cudaStream_t stream);
cudaError_t KimiRouteFinish(cudaEvent_t ev);

// Routed-expert NVFP4 FFN for ONE token: dequantizes the expert's three
// projections from the mmap'd entry, applies gate/up GEMMs + SwiGLU + down
// GEMM and accumulates `expert_weight * y` into d_out (caller pre-zeroes or
// seeds d_out). d_staging must hold >= RunKimiExpertFfnStagingBytes bytes.
// Staging layout is internal; sizes are derived from kKimi* geometry.
cudaError_t RunKimiExpertFfn(cublasHandle_t cublas_h, cudaStream_t stream,
                             const KimiExpertEntry &expert,
                             float expert_weight, const __nv_bfloat16 *d_x,
                             __nv_bfloat16 *d_out, void *d_staging,
                             size_t staging_bytes);

// Resident-cache variant: identical math, weights read from the expert's
// device cache slot (KimiMoeSlotView) instead of per-call staging.
// d_scratch must hold >= KimiExpertFfnStagingBytes bytes (dequant + acts).
cudaError_t RunKimiExpertFfnCached(cublasHandle_t cublas_h,
                                   cudaStream_t stream,
                                   const KimiExpertEntry &expert,
                                   float expert_weight,
                                   const __nv_bfloat16 *d_x,
                                   __nv_bfloat16 *d_out, void *d_scratch,
                                   size_t scratch_bytes,
                                   const KimiMoeSlotView &view);

// Upper bound of the staging requirement of RunKimiExpertFfn.
size_t KimiExpertFfnStagingBytes();

// ── Batched TC routed chain (fast-math decode path) ─────────────────────────
// One bf16 hidden row -> top-K expert FFN through the neutral lsx_tc
// batched mxf4nvf4 kernels. d_tbl: [K] slot table (dev_slot + gs1/gs3/gs2
// + coeff), built host-side by the generation loop. Writes the pure
// expert sum coeff_k * FFN_k(x) into d_out (ascending-k order; the call
// site adds the residual). Tolerance-class vs the legacy dequant->BF16
// GEMV chain (act-quant + split-K); run-to-run deterministic.
// d_scratch: caller-allocated KimiTcBatchScratchBytes() buffer.
// ev13/ev2 (nullable): split-fence upload events — the w13 GEMM waits ev13,
// the w2 GEMM waits ev2; null waits degenerate to no-op (uploads already
// fenced by the caller).
cudaError_t RunKimiRoutedTcBatch(cublasHandle_t cublas_h, cudaStream_t stream,
                                 const uint8_t *pack_base,
                                 const uint8_t *scale_base,
                                 int64_t pack_stride, int64_t scale_stride,
                                 const lsxcommon::lsx_tc::LsxTcBatchSlot *d_tbl,
                                 int K, const __nv_bfloat16 *d_x,
                                 __nv_bfloat16 *d_out, void *d_scratch,
                                 size_t scratch_bytes,
                                 cudaEvent_t ev13 = nullptr,
                                 cudaEvent_t ev2 = nullptr);

// Upper bound of the RunKimiRoutedTcBatch scratch requirement (K=8).
size_t KimiTcBatchScratchBytes();

struct KimiGenerationResult {
    std::vector<int> tokens;      // generated tokens only (never the prompt)
    float first_maxv = 0.0f;
    bool ok = false;
    bool eos_terminated = false;
    int kv_slots_used = 0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::string error_msg;
    std::vector<float> final_logits;
};

// Host-side EOS application (device-side decode, F-11): cuts tokens at the
// first occurrence of eos_id and reports whether the cut happened. eos_id
// < 0 disables the cut. Pure host function, unit-testable without CUDA.
struct KimiEosCut {
    std::vector<int> tokens;
    bool eos_terminated = false;
};
KimiEosCut TruncateAtEos(const std::vector<int> &tokens, int eos_id);

// Greedy full generation: embed -> 61-layer stack (MLA + MoE) per step ->
// final norm + lm_head + argmax. Prefill covers the whole prompt in one pass;
// every decoded step runs seq=1. Entire loop async on `stream`; the only
// synchronization happens when the caller drains results.
cudaError_t RunKimiGeneration(cublasHandle_t cublas_h, cudaStream_t stream,
                              const KimiPinnedWeights &pw, KimiExpertDb &db,
                              const std::vector<int> &prompt_ids,
                              int max_gen_tokens, KimiGenerationResult &gen,
                              KimiExpertCacheState *cache, int eos_id,
                              bool fast_math = false);

} // namespace kimi
} // namespace lsxcommon
