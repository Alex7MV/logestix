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

// ── MoE: clamped-swiglu routed chain (flash-local; GLM semantics) ────────
// Per-slot chain over LRU-resident experts (staging via
// EnsureQwenMoeExpertResident): y_g = w1@x, y_u = w3@x (NVFP4 GEMM),
// inter = clamp10(silu(y_g)) * clamp10(y_u), y2 = w2@inter, acc += c*y2.
cudaError_t RunFlashRoutedFFN(
    cudaStream_t stream, qwen_moe::QwenMoeExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer, int moe_layer,
    const qwen_moe::QwenMoeRoutedSlot *h_slots, int k_top,
    const __half *d_x, int hidden, int moe_inter, float *d_acc,
    __half *d_yg, __half *d_yu, __half *d_y2, std::string &error);

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
