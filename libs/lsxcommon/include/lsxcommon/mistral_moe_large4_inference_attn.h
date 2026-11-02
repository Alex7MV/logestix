#pragma once

// Mistral-Large-4 MLA attention module (Model Ingestion Protocol stage 5).
//
// The cache holds the COMPRESSED latent (the reference's own rule: "cache read
// and write is performed while the latent KV is still compressed"), so a
// 2048-token context costs ~150 MB instead of the ~8 GB an expanded K/V cache
// would need. Attention is the mathematically equivalent absorbed form:
//   q_abs[h] = q_nope[h] @ W_k,h                (512 per head)
//   score_t  = q_abs[h] . latent_t + q_rope[h] . k_pe_t
//   ctx[h]   = sum_t p_t latent_t ;  out[h] = W_v,h @ ctx[h]
//
// Cross-layer KV sharing: layers whose `kv_source` is an earlier layer have no
// kv_a projection and read the source layer's cache. DSA: a `full` indexer
// layer computes the top-k mask for the current query and stores it; a `shared`
// indexer layer reuses the most recent full layer's mask. At or below
// `index_topk` positions every index is selected (identity mask).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "lsxcommon/mistral_moe_large4_kernels.h"
#include "lsxcommon/mistral_moe_large4_load.h"

namespace lsxcommon {
namespace mistral_large4 {

struct Large4AttnState {
    int capacity = 0;
    int length = 0;  // positions written into every cache
    // Per layer: latent cache [capacity][kv_lora] BF16 (+ k_pe [capacity][64]
    // on RoPE layers). Null entries for non-owning (shared-KV) layers.
    std::vector<__nv_bfloat16 *> latent;
    std::vector<__nv_bfloat16 *> k_pe;
    // Per layer: indexer key cache [capacity][index_head_dim] BF16 (full
    // indexer layers only).
    std::vector<__nv_bfloat16 *> indexer_k;
    // DSA mask scratch, valid after every layer (the most recent full
    // indexer's selection).
    int32_t *d_topk = nullptr;
    int *d_topk_count = nullptr;
    // Scratch.
    float *d_q = nullptr;            // [heads * qk_head]
    float *d_q_abs = nullptr;        // [heads * kv_lora]
    float *d_ctx = nullptr;          // [heads * kv_lora]
    float *d_scores = nullptr;       // [heads * capacity]
    float *d_kv = nullptr;           // [kv_lora + qk_rope]
    float *d_attn = nullptr;         // [heads * v_head]
    __half *d_q16 = nullptr;         // [heads * qk_head]
    __half *d_x16 = nullptr;         // [hidden]
    __half *d_attn16 = nullptr;      // [heads * v_head]
    __half *d_out16 = nullptr;       // [hidden] (wo output)
    float *d_index_q = nullptr;      // [index_heads * index_head_dim]
    float *d_index_w = nullptr;      // [index_heads]
    float *d_index_scores = nullptr; // [capacity]
    __half *d_index_q16 = nullptr;   // [index_heads * index_head_dim]
    __half *d_index_x16 = nullptr;   // [hidden]
    __nv_bfloat16 *d_index_norm = nullptr;  // [hidden]
    float *d_index_norm32 = nullptr;        // [hidden]
    __nv_bfloat16 *d_index_kvec = nullptr;  // [index_head_dim]
    float *d_index_k32 = nullptr;           // [index_head_dim]
    float *d_norm = nullptr;  // [hidden] F32 (router input + norm scratch)
    float *d_proj = nullptr;  // [hidden] F32 (wo / down output)
    float *d_cos = nullptr;
    float *d_sin = nullptr;

    void Free();
};

cudaError_t AllocateLarge4AttnState(Large4AttnState &s, const Large4Dims &dims,
                                    const std::vector<Large4LayerWeights> &layers,
                                    int capacity, const std::vector<float> &cos_t,
                                    const std::vector<float> &sin_t,
                                    cudaStream_t stream);

void FreeLarge4AttnState(Large4AttnState &s);

// One layer's attention. `normed` is the post-input_norm BF16 activation;
// `normed32` the same values in F32 (router / LayerNorm input). Writes the
// post-`wo` F32 vector into `out` (NOT added to the residual).
cudaError_t Large4AttentionStep(Large4AttnState &s, const Large4Dims &dims,
                                const Large4LayerWeights &lw, int layer,
                                const __nv_bfloat16 *normed,
                                const float *normed32, int pos, float *out,
                                cudaStream_t stream);

} // namespace mistral_large4
} // namespace lsxcommon
