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
