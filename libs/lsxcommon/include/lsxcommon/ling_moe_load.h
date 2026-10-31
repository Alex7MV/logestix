#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::ling {

struct LingIndex {
    std::string format;
    int num_layers = 0;
    int hidden = 0;
    int vocab = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int qk_head_dim = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int conv_kernel = 0;
    int num_experts = 0;
    int top_k = 0;
    int n_group = 0;
    int topk_group = 0;
    int moe_intermediate = 0;
    int shared_intermediate = 0;
    int dense_intermediate = 0;
    int first_moe_layer = 0;
    int total_experts = 0;
    double rms_eps = 0.0;
    double rope_theta = 0.0;
    double routed_scale = 0.0;
    int eos_token_id = -1;
    int pad_token_id = -1;
    std::vector<float> expert_swiglu_limit;
    std::vector<float> shared_swiglu_limit;
    std::vector<int> mla_layers;
    std::vector<int> kda_layers;
    std::vector<int> moe_layers;
    std::vector<int> dense_layers;
    VramProfile vram;
    std::string error_msg;
};

LingIndex LoadLingIndex(const std::string &model_path);

struct LingLayerWeights {
    bool is_mla = false;
    bool is_moe = false;
    float expert_swiglu_limit = 0.0f;
    float shared_swiglu_limit = 0.0f;

    // Plain RMSNorm weights (FP16, weight form).
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;

    // KDA projection set.
    lsx_nvfp4::LsxNvfp4Weight kda_q, kda_k, kda_v, kda_o, kda_f, kda_g;
    __half *d_conv_q = nullptr;   // [heads*head_dim, conv]
    __half *d_conv_k = nullptr;
    __half *d_conv_v = nullptr;
    float *d_a_log = nullptr;     // [heads]
    float *d_dt_bias = nullptr;   // [heads*head_dim]
    __half *d_kda_b = nullptr;    // [heads, hidden]
    __half *d_kda_o_norm = nullptr;  // [head_dim]

    // Gated MLA.
    lsx_nvfp4::LsxNvfp4Weight mla_q_a, mla_q_b, mla_kv_a, mla_kv_b, mla_dense;
    __half *d_q_a_norm = nullptr;   // [q_lora]
    __half *d_kv_a_norm = nullptr;  // [kv_lora]
    __half *d_mla_g = nullptr;      // [heads, hidden]

    // Dense SwiGLU prefix.
    lsx_nvfp4::LsxNvfp4Weight dense_gate, dense_up, dense_down;

    // MoE.
    __half *d_router = nullptr;   // [experts, hidden] BF16 bits (FP32 dot)
    float *d_router_bias = nullptr;  // [experts]
    std::int64_t router_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;

    bool loaded = false;
};

struct LingPinnedWeights {
    LingIndex top;
    __half *d_embed = nullptr;  // [V, H]
    __half *d_head = nullptr;   // [V, H]
    __half *d_final_norm = nullptr;
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    std::vector<LingLayerWeights> layers;
    moe::MoEExpertRuntime expert_store;
    std::uint64_t resident_bytes = 0;
    void Free();
};

void FreeLingPinnedWeights(LingPinnedWeights &pw);

bool LoadLingDenseWeights(const std::string &model_path, LingPinnedWeights &pw,
                          cudaStream_t stream, std::string &error);

KvCacheGeometry LingKvGeometry(const LingIndex &top, int64_t capacity);

} // namespace lsxcommon::ling
