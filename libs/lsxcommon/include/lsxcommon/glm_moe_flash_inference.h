#pragma once

// GLM-5.3-Flash inference kernels + generation driver.
// KDA (linear_attention) layers: causal conv1d + gated delta rule with
// per-channel decay; DSA (deepseek_sparse_attention) layers: MLA + indexer
// (ported as NEW functions from the GLM-5.2 kernel semantics); MoE: sigmoid
// noaux_tc top-8 router + NVFP4 routed/shared experts; hyper-connections
// via the shared cached GatedResidual primitives.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "lsxcommon/glm_moe_flash_load.h"
#include "lsxcommon/qwen_inference.h"     // QwenNvfp4Weight, QwenGenerationResult
#include "lsxcommon/qwen_moe_inference.h"  // QwenMoeRoutedSlot
#include "lsxcommon/sampling.h"

#ifndef LSXCOMMON_GLM_FLASH_INFERENCE_H
#define LSXCOMMON_GLM_FLASH_INFERENCE_H

namespace lsxcommon {
namespace glm_moe_flash {

// ── Test twins (raw F32 buffers, no FP16, no NVFP4) for bit-parity ────────

// Depthwise causal conv window: x [K, C] rows oldest..newest, w [C, K],
// y [C]. y[c] = sum_k w[c*K+k] * x[k*C+c].
cudaError_t RunFlashConv1dF32(cudaStream_t stream, const float *x,
                              const float *w, float *y, int channels,
                              int kernel);

// Gated delta rule with per-channel decay (test twin). q/k/v: [seq, H, D]
// FP32; beta: [seq, H]; gdecay: [seq, H*D]; state_out: [H, D, D];
// out: [seq, H, D]. Mirrors FlashGatedDeltaHost exactly.
cudaError_t RunFlashGatedDeltaF32(cudaStream_t stream, const float *q,
                                  const float *k, const float *v,
                                  const float *beta, const float *gdecay,
                                  int seq, int heads, int dim,
                                  float *state_out, float *out);

// Router top-8: gate BF16 bits [E, H], bias F32 [E], x BF16 bits [tok, H].
// Slots: [tok*8] expert + coeff (renormed, scaled by 2.5).
cudaError_t RunFlashRouterTop8(cudaStream_t stream, const uint8_t *d_gate,
                               const float *d_bias, const uint8_t *d_x,
                               int tok, int hidden, int experts,
                               qwen_moe::QwenMoeRoutedSlot *d_slots);

// ── KDA decode-path state + kernels ──────────────────────────────────────
// Per-layer recurrent state: S[heads, dim, dim] FP32 + short-conv history
// window [K-1 rows, 3*8192] FP16 (q/k/v channels interleaved by projection).
struct FlashKdaState {
    float *d_state = nullptr;    // [heads*dim*dim] (heads=64, dim=128)
    __half *d_conv_hist = nullptr;  // [(K-1), 3*8192] oldest..newest
    int64_t capacity_ctx = 0;
};

cudaError_t FlashKdaStateAlloc(FlashKdaState &st, int64_t max_ctx);
void FlashKdaStateFree(FlashKdaState &st);

// Single-token KDA decode for one layer (seq == 1):
//   d_x: [H] current token hidden (post hc mixing, FP16).
//   d_out: [H] FP16 layer attention output (pre o_norm? no: post o_proj).
// Pipeline: q/k/v GEMM (FP16 cuBLAS via RunExpHeadGemm) -> depthwise causal
// conv1d (window = history + current) -> l2norm q/k per head -> gating prep
// (beta = sigmoid(b_proj@x), decay = exp(-exp(A_log)*softplus(dt+dt_bias))
// per (h,l), dt = f_b(f_a(x))) -> per-channel-decay delta step (state seed
// from d_state, write-back) -> RMSNormGated(core, gate, o_norm) -> o GEMM.
// All on `stream`, no host sync.
cudaError_t RunFlashKdaDecode(cudaStream_t stream, FlashKdaState &st,
                              const FlashLayerWeights &lw, int hidden,
                              int heads, int dim, int conv_kernel,
                              const __half *d_x, __half *d_out,
                              void *d_scratch /* caller workspace */);

// Router top-8 with FP16 activations (gate stays BF16 bits on device).
cudaError_t RunFlashRouterTop8F16(cudaStream_t stream, const uint8_t *d_gate,
                                  const float *d_bias, const __half *d_x,
                                  int tok, int hidden, int experts,
                                  qwen_moe::QwenMoeRoutedSlot *d_slots);

// Test probe: flash_expf_glibc(0/1).
cudaError_t RunFlashExp0Probe(cudaStream_t stream, float *d_out);

// Test wrapper: MLA attention kernel over a single-token cache.
cudaError_t RunFlashMlaAttnTest(cudaStream_t stream, const __half *d_q,
                                const __half *d_k, const __half *d_v,
                                __half *d_out, int heads, int dim,
                                float scale, size_t smem);

// ── Fused decode test twins (FP16 bit-parity harness) ────────────────────
// Depthwise causal conv decode with optional fused post-conv SiLU (the
// production decode path). hist [K-1, channels], x [channels], w [channels,
// K], y [channels], all FP16.
cudaError_t RunFlashConvDecodeTest(cudaStream_t stream, const __half *hist,
                                   const __half *x, const __half *w, __half *y,
                                   int channels, int kernel, bool apply_silu);

// In-place SiLU over n FP16 elements (reference for the conv+silu parity).
cudaError_t RunFlashSiluTest(cudaStream_t stream, __half *x, int n);

// In-place gate activation (sigmoid default, silu opt-in) over n FP16 elems.
cudaError_t RunFlashGateActTest(cudaStream_t stream, __half *g, int n,
                                bool silu);

// RMSNormGated per head with optional fused on-the-fly gate activation.
cudaError_t RunFlashRmsnormGatedTest(cudaStream_t stream, const float *core,
                                     const __half *gate, const __half *norm_w,
                                     __half *out, int heads, int dim, float eps,
                                     bool fuse_gate_act, bool gate_silu);

// L2-normalize q rows in place (single-plane reference path).
cudaError_t RunFlashL2normTest(cudaStream_t stream, __half *row, int dim,
                               float extra_scale, int heads);

// Fused q/k L2-normalize (one launch; q rows use q_scale, k rows use 1.0).
cudaError_t RunFlashL2normQKTest(cudaStream_t stream, __half *q, __half *k,
                                 int dim, float q_scale, int heads);

// ── MoE: clamped-swiglu routed chain (flash-local; GLM semantics) ────────
// Per-slot chain over LRU-resident experts (staging via
// EnsureQwenMoeExpertResident): y_g = w1@x, y_u = w3@x (NVFP4 GEMM),
// inter = clamp10(silu(y_g)) * clamp10(y_u), y2 = w2@inter, acc += c*y2.
//
// Perf path: when `batch_ygu`/`batch_hbuf`/`batch_ypart`/`d_gemv_tbl` are
// supplied, the K slots run through the shared batched routed-GEMV chain
// (qwen_moe::RunQwenMoeRoutedGemmVBatchCore) -- 4 launches per layer instead
// of K*4, same slot-major weight streaming. Nulls fall back to the legacy
// per-slot chain (also the test/ad-hoc path).
cudaError_t RunFlashRoutedFFN(
    cudaStream_t stream, qwen_moe::QwenMoeExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer, int moe_layer,
    const qwen_moe::QwenMoeRoutedSlot *h_slots, int k_top,
    const __half *d_x, int hidden, int moe_inter, float *d_acc,
    __half *d_yg, __half *d_yu, __half *d_y2, std::string &error,
    cudaStream_t h2d_stream, cudaEvent_t *copy_ev, cudaEvent_t *slot_free_ev,
    __half *batch_ygu, __half *batch_hbuf, __half *batch_ypart,
    qwen_moe::QwenMoeGvSlotDev *d_gemv_tbl);

// Shared expert: same clamped swiglu semantics; out += result.
cudaError_t RunFlashSharedFFN(cudaStream_t stream,
                              const FlashLayerWeights &lw,
                              const __half *d_x, int hidden, int shared_inter,
                              __half *d_yg, __half *d_yu, __half *d_y2,
                              __half *d_out);

// ── Generation driver ─────────────────────────────────────────────────────
// Single-stream incremental decode over the GLM-5.3-Flash layer stack:
// hc GatedResidual (cached) -> KDA/DSA attention branch -> hc inject scatter
// -> hc FFN branch (MoE: router top-8 + LRU-resident NVFP4 routed experts +
// shared expert; dense layers 0..2: NVFP4 MLP) -> hc inject scatter.
cudaError_t RunGlmFlashGeneration(
    cudaStream_t stream, const FlashPinnedWeights &pw,
    qwen_moe::QwenMoeExpertStore &store, gpu::ExpertDeviceCache &cache,
    const std::vector<int32_t> &prompt, int max_new,
    const std::vector<int> &eos_ids, qwen::QwenGenerationResult &result,
    sample::ISampler &sampler);

} // namespace glm_moe_flash
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_FLASH_INFERENCE_H
