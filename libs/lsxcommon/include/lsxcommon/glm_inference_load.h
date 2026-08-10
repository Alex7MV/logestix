#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace glm {

// GLM-5.2 geometry (from config.json, verified against the real model).
constexpr int kGlmHidden = 6144;
constexpr int kGlmNumLayers = 79;
constexpr int kGlmNumHeads = 64;
constexpr int kGlmHeadDim = 192;         // qk_nope(192) + qk_rope(64) = 256 per kv head
constexpr int kGlmQkNope = 192;
constexpr int kGlmQkRope = 64;
constexpr int kGlmHeadV = 256;           // v_head
constexpr int kGlmQLoraRank = 2048;
constexpr int kGlmKVLoraRank = 512;
constexpr int kGlmMoeIntermediate = 2048;
constexpr int kGlmNumRouted = 256;
constexpr int kGlmNumShared = 1;
constexpr int kGlmIndexNHeads = 32;
constexpr int kGlmIndexHeadDim = 128;
constexpr float kGlmRmsEps = 1e-5f;

// Per-layer pinned weights for one GLM-5.2 layer. All weights are stored as
// raw BF16 in the logestix layout and uploaded as FP16 here (host-side
// BF16→FP16 conversion, mirroring the DeepSeek loader). Dense layers (0-2)
// fill the mlp_* fields, MoE layers fill the router fields; indexer layers
// additionally fill idx_*; the MTP block (layer 78) fills eh_*/enorm/hnorm/
// shared_head_norm. Pointers stay null when the layer file lacks the tensor.
struct GlmLayerWeights {
    // Layer norms
    __half *d_input_norm = nullptr;         // input_layernorm.weight
    __half *d_post_attn_norm = nullptr;     // post_attention_layernorm.weight

    // MLA attention (raw BF16 → FP16)
    __half *d_q_a_proj = nullptr;           // self_attn.q_a_proj.weight
    __half *d_q_a_norm = nullptr;           // self_attn.q_a_layernorm.weight
    __half *d_q_b_proj = nullptr;           // self_attn.q_b_proj.weight
    __half *d_kv_a_proj = nullptr;          // self_attn.kv_a_proj_with_mqa.weight
    __half *d_kv_a_norm = nullptr;          // self_attn.kv_a_layernorm.weight
    __half *d_kv_b_proj = nullptr;          // self_attn.kv_b_proj.weight
    __half *d_o_proj = nullptr;             // self_attn.o_proj.weight
    int64_t q_a_rows = 0, q_a_cols = 0;
    int64_t q_b_rows = 0, q_b_cols = 0;
    int64_t kv_a_rows = 0, kv_a_cols = 0;
    int64_t kv_b_rows = 0, kv_b_cols = 0;
    int64_t o_rows = 0, o_cols = 0;

    // Router (MoE layers 3-78): mlp.gate.weight (BF16) + e_score_correction_bias (F32)
    __half *d_router_gate = nullptr;        // mlp.gate.weight
    float  *h_gate_bias = nullptr;          // mlp.gate.e_score_correction_bias (host pinned)
    int     gate_bias_n = 0;

    // Dense MLP (layers 0-2): mlp.{gate,down,up}_proj.weight
    __half *d_mlp_gate = nullptr;
    __half *d_mlp_down = nullptr;
    __half *d_mlp_up = nullptr;

    // Indexer (indexer layers)
    __half *d_idx_wq_b = nullptr;           // self_attn.indexer.wq_b.weight
    __half *d_idx_wk = nullptr;             // self_attn.indexer.wk.weight
    __half *d_idx_weights_proj = nullptr;   // self_attn.indexer.weights_proj.weight
    __half *d_idx_k_norm = nullptr;         // self_attn.indexer.k_norm.weight
    float  *d_idx_k_norm_bias = nullptr;    // self_attn.indexer.k_norm.bias (F32)

    // MTP block (layer 78)
    __half *d_eh_proj = nullptr;            // eh_proj.weight
    __half *d_enorm = nullptr;              // enorm.weight
    __half *d_hnorm = nullptr;              // hnorm.weight
    __half *d_shared_head_norm = nullptr;   // shared_head.norm.weight

    void Free();
};

struct GlmPinnedWeights {
    __half *d_embed = nullptr;              // model.embed_tokens.weight [vocab, hidden]
    __half *d_head = nullptr;               // lm_head.weight [vocab, hidden]
    __half *d_final_norm = nullptr;         // model.norm.weight [hidden]
    int vocab_size = 0;
    int hidden = 0;
    std::vector<GlmLayerWeights> layers;    // indexed by layer 0..num_layers-1

    void Free();
};

// Reads <model_path>/logestix/pinned/{global,layer_<n>}.arrow for a
// GLM-5.2 converted model and uploads every pinned tensor to device memory as
// FP16 (host-side BF16→FP16 conversion). Per-layer files that don't exist are
// skipped (heterogeneous GLM layer grid). Sizes GlmPinnedWeights.layers from
// the extended topology (glm::load_topology). Returns cudaErrorInvalidValue on
// a missing pinned dir / unreadable global.arrow.
//
// When layer_filter is non-null, only the layer files whose index appears in
// the list are uploaded (global tensors are always uploaded). This lets a
// partial forward pass (e.g. the dense layers 0-2) avoid uploading the full
// 32 GiB pinned set.
cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream,
                                 const std::vector<int> *layer_filter);

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream);

} // namespace glm
} // namespace lsxcommon