#pragma once

// GigaChat4-37B-A4B DSA lightning indexer (decode step).
//
// The eight full-attention layers carry an indexer that scores every cached key
// for the current query and selects the top-`index_topk` (2048) of them; the MLA
// softmax then attends to that selection only. Semantics transcribed from the
// shipped modeling_gigachat4.py `GigaChat4DSAIndexer`:
//
//   q       = wq_b(alpha_q * q_latent)            [heads, head_dim]
//   k       = k_norm(wk(x))                       [head_dim]   (plain RMSNorm)
//   rope    = NON-interleaved, on the first `rope` lanes of q and k
//   weights = weights_proj(x) * heads^-0.5         [heads]
//   score_t = head_dim^-0.5 * sum_h weights_h * relu(q_h . k_t)
//
// `k_t` is the cached, POST-rope index key of position t. Selection is exact and
// deterministic: it returns min(topk, n) indices in ascending order, every
// strictly-greater score first and then the lowest-index tie fill, so a tie can
// never crowd out a strictly-greater entry and two runs always agree.
//
// All activations are F32 at the kernel boundary; the NVFP4/GEMV projections
// convert at their own edge.

#include <cstdint>

#include <cuda_runtime.h>

#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon::gigachat {

// DSA lightning-indexer weights of one full-attention layer.
struct GcDsaIndexerWeights {
    lsx_nvfp4::LsxNvfp4Weight wq_b;           // NVFP4 [heads*head_dim, q_lora]
    const uint8_t *d_wk = nullptr;            // BF16 [head_dim, hidden]
    const uint8_t *d_weights_proj = nullptr;  // BF16 [heads, hidden]
    const uint8_t *d_k_norm = nullptr;        // BF16 [head_dim]
    void Free();
};

struct GcDsaIndexerDims {
    int hidden = 0;
    int q_lora = 0;
    int heads = 0;
    int head_dim = 0;
    int rope = 0;
    int topk = 0;
    // q_latent arrives unscaled; the step applies alpha_q itself.
    float latent_scale = 1.0f;
};

// Per-layer scratch, allocated once (all carves out of one device block).
struct GcDsaIndexerScratch {
    void *d_base = nullptr;
    int64_t bytes = 0;
    void *q_h = nullptr;      // __half [q_lora]
    void *k_h = nullptr;      // __half [head_dim]
    void *x_h = nullptr;      // __half [hidden]
    void *q_f = nullptr;      // float [heads * head_dim]
    void *k_f = nullptr;      // float [head_dim]
    void *w_f = nullptr;      // float [heads]
    void *k_normed = nullptr; // float [head_dim]
    void *scores = nullptr;   // float [capacity]

    bool Alloc(const GcDsaIndexerDims &d, int capacity);
    void Free();
};

// Non-interleaved RoPE (rotate_half over the half-split of the first `rope`
// lanes) on q [heads, head_dim] and the single k vector. Table row `pos`.
cudaError_t RunGigaChatDsaIndexerRope(float *q, float *k, int heads, int head_dim,
                                    int rope, const float *cos_t,
                                    const float *sin_t, int pos,
                                    cudaStream_t stream);

// Plain RMSNorm (weight only, no gate) over `n` floats.
cudaError_t RunGigaChatDsaIndexerKNorm(float *k, const uint8_t *w_bf16, int n,
                                     float eps, cudaStream_t stream);

// Index scores over the cached keys: one thread per position in [0, n).
cudaError_t RunGigaChatDsaIndexerScores(const float *q, const float *k_cache,
                                      const float *weights, int heads,
                                      int head_dim, float scale, int n,
                                      float *scores, cudaStream_t stream);

// Exact top-k over `n` scores. Writes `min(k, n)` ascending indices and their
// count. Deterministic: block-wide binary search for the k-th largest value,
// then one ascending scan (thread 0) for the strictly-greater entries plus the
// lowest-index tie fill.
cudaError_t RunGigaChatDsaTopkSelect(const float *scores, int n, int k,
                                   int32_t *out_idx, int *d_count,
                                   cudaStream_t stream);

// One decode token: projects q/k/weights, appends k to the layer's index cache
// (post-rope, [capacity, head_dim] F32), scores the causal prefix [0, pos] and
// selects. `q_latent_scaled` is alpha_q * q_latent (post q_a_layernorm);
// `hidden_bf16` is the pre-attention normed hidden state.
cudaError_t RunGigaChatDsaIndexerStep(const float *q_latent_scaled,
                                    const uint8_t *hidden_bf16,
                                    const GcDsaIndexerWeights &w,
                                    const GcDsaIndexerDims &d,
                                    const float *cos_t, const float *sin_t,
                                    int pos, int capacity, float *k_cache,
                                    GcDsaIndexerScratch &sc, int32_t *out_idx,
                                    int *out_count, cudaStream_t stream);

} // namespace lsxcommon::gigachat
