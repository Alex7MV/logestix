#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_moe_inference.h"

namespace lsxcommon {
namespace mistral_large4 {

// One resident BF16 matrix (row-major). Used for the attention KV projections,
// the indexer key/weights projections and the embeddings/head, which the
// checkpoint itself keeps out of its FP8 quantization set.
struct Large4Bf16Weight {
    const __nv_bfloat16 *data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t bytes = 0;
    bool valid() const { return data != nullptr; }
};

struct Large4LayerWeights {
    // NVFP4 projections (quantized from the source FP8).
    lsx_nvfp4::LsxNvfp4Weight wq;   // [heads*qk_head, H]
    lsx_nvfp4::LsxNvfp4Weight wo;   // [H, heads*v_head]
    lsx_nvfp4::LsxNvfp4Weight indexer_wq_b;  // [index_heads*index_head_dim, H]

    // BF16 attention edges (present only when this layer owns its KV cache).
    Large4Bf16Weight wkv_a;  // [kv_lora (+ qk_rope), H]
    Large4Bf16Weight wkv_b;  // [heads*((qk_nope|qk_head)+v_head), kv_lora]

    // BF16 indexer edges (present only on `full` indexer layers).
    Large4Bf16Weight indexer_wk;            // [index_head_dim, H]
    Large4Bf16Weight indexer_weights_proj;  // [index_heads, H]

    // Norms / scalars (BF16).
    const __nv_bfloat16 *input_ln = nullptr;    // [H]
    const __nv_bfloat16 *post_ln = nullptr;     // [H]
    const __nv_bfloat16 *kv_a_norm = nullptr;   // [kv_lora]
    const __nv_bfloat16 *ssmax_s = nullptr;     // [heads]
    const __nv_bfloat16 *ssmax_b = nullptr;     // [heads]
    const __nv_bfloat16 *idx_q_norm = nullptr;      // [H] weight
    const __nv_bfloat16 *idx_q_norm_bias = nullptr; // [H]
    const __nv_bfloat16 *idx_k_norm = nullptr;      // [index_head_dim]
    const __nv_bfloat16 *idx_k_norm_bias = nullptr; // [index_head_dim]
    const __nv_bfloat16 *idx_ssmax_s = nullptr;     // [index_heads]
    const __nv_bfloat16 *idx_ssmax_b = nullptr;     // [index_heads]

    // Router (BF16 [E, H]) and the MLP branch for this layer.
    const __nv_bfloat16 *router = nullptr;
    int64_t router_bytes = 0;
    // MoE layers: one gated shared expert (NVFP4, I = moe_intermediate).
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;
    // Dense layer(s): the plain MLP (NVFP4, I = dense_intermediate).
    lsx_nvfp4::LsxNvfp4Weight dense_gate, dense_up, dense_down;

    // Topology flags (from the converted index, cross-checked against the
    // passport): whether the layer owns a latent KV cache, runs its own
    // indexer, uses RoPE, is sparse, and which layer it reads KV from.
    bool uses_rope = false;
    bool owns_kv = false;
    bool full_indexer = false;
    bool is_moe = false;
    int kv_source = -1;

    void Free();
};

struct Large4Dims {
    int hidden = 0;
    int vocab = 0;
    int layers = 0;
    int heads = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int qk_head = 0;
    int v_head = 0;
    int experts = 0;
    int top_k = 0;
    int moe_inter = 0;
    int dense_inter = 0;
    int index_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    float rms_eps = 0.0f;
    float attn_scale = 0.0f;
    float logit_softcap = 0.0f;
};

struct Large4PinnedWeights {
    std::string model_path;
    Large4Dims dims;
    Large4Bf16Weight embed;
    Large4Bf16Weight head;
    const __nv_bfloat16 *final_norm = nullptr;
    const __nv_bfloat16 *lerope_alpha = nullptr;  // [qk_rope/2]
    std::vector<float> lerope_alpha_host;  // host mirror (rope table build)
    std::vector<Large4LayerWeights> layers;
    // Neutral host expert store (mmap views + LRU): 71 x 128 experts streamed
    // through a device slot cache. The dense layer 0 is skipped
    // (first_moe_layer).
    lsxcommon::moe::MoEExpertRuntime expert_store;
    std::size_t resident_bytes = 0;

    void Free();
};

cudaError_t LoadMistralLarge4Weights(const std::string &model_path,
                                     Large4PinnedWeights &pw,
                                     cudaStream_t stream, std::string &error);

void FreeMistralLarge4Weights(Large4PinnedWeights &pw);

} // namespace mistral_large4
} // namespace lsxcommon
