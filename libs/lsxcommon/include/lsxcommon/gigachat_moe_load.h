#pragma once

// Loader-stage passport for GigaChat3.5-432B-A28B-Reasoning. Parses
// <model_path>/logestix/model.index.json and validates it against the
// compile-time frozen geometry (src/gigachat_moe_constants.h) before any
// weight is touched.
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/gigachat_moe_37b_inference_indexer.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::gigachat {

struct GigaChatIndex {
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
    std::string model;
    int num_layers = 0;
    std::vector<std::string> layer_types;  // "full_attention" | "linear_attention"
    std::vector<int> full_attention_layers;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int first_k_dense_replace = 0;
    double rms_norm_eps = 0.0;
    double router_scaling_factor = 0.0;
    double swiglu_limit = 0.0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int mtp_tensors = 0;
    int total_pinned_tensors = 0;
    VramProfile vram;
};

// Reads <model_path>/logestix/model.index.json. Returns false and fills
// `error` on a missing/unparseable index or an unknown format tag.
bool LoadGigaChatIndex(const std::string &model_path, GigaChatIndex &out,
                       std::string &error);

// Cross-checks `idx` against the frozen passport. Honors
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 for synthetic fixtures. Returns false and
// fills `error` on the first mismatch.
bool ValidateGigaChatGeometry(const GigaChatIndex &idx, std::string &error);

// ── Device weight graph ─────────────────────────────────────────────────────

// One ZeroCenteredGatedNorm triple (weight + gate_up [rank, dim] + gate_down
// [dim, rank]). All three stay BF16: the norm kernels consume raw BF16 bits.
struct GcGatedNorm {
    const uint8_t *d_weight = nullptr;     // BF16 [dim]
    const uint8_t *d_gate_up = nullptr;    // BF16 [rank, dim]
    const uint8_t *d_gate_down = nullptr;  // BF16 [dim, rank]
    int dim = 0;
    int rank = 0;
    void Free();
};

// Per-layer resident weights. Fields not used by the layer type stay null.
struct GcLayerWeights {
    int layer = 0;
    bool full_attention = false;
    bool moe = false;

    // Every layer carries four gated norms.
    GcGatedNorm input_norm;
    GcGatedNorm post_attn_norm;
    GcGatedNorm post_self_attn_norm;
    GcGatedNorm post_ffw_norm;

    // MLA (full_attention layers).
    GcGatedNorm q_a_norm;
    GcGatedNorm kv_a_norm;
    lsx_nvfp4::LsxNvfp4Weight q_a, q_b, kv_a, kv_b, o_proj, attn_gate;
    // DSA lightning indexer of the full-attention layers. Empty (null planes)
    // for a checkpoint that has none (GigaChat3.5); the pipeline keys the DSA
    // step off `dims.index_topk`, not off this member.
    GcDsaIndexerWeights indexer;

    // GatedDeltaNet (linear_attention layers). in_proj_ba stays BF16 (the only
    // non-NVFP4 projection, dims are tiny); conv/A_log/dt_bias stay F32.
    lsx_nvfp4::LsxNvfp4Weight gdn_in_qkvz, gdn_out_proj;
    const uint8_t *d_conv1d = nullptr;     // F32 [conv_dim, 1, K]
    int conv_dim = 0;
    int conv_kernel = 0;
    const uint8_t *d_A_log = nullptr;      // F32 [value_heads]
    const uint8_t *d_dt_bias = nullptr;    // F32 [value_heads]
    int gdn_value_heads = 0;
    const uint8_t *d_gdn_in_ba = nullptr;  // BF16 [ba_dim, hidden]
    int gdn_ba_rows = 0;
    const uint8_t *d_gdn_norm = nullptr;   // BF16 [value_dim]
    int gdn_norm_dim = 0;

    // MoE (layers >= first_k_dense_replace).
    const uint8_t *d_router = nullptr;       // BF16 [num_experts, hidden]
    const uint8_t *d_router_bias = nullptr;  // F32 [num_experts] (may be null)
    int num_experts = 0;
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;

    // Dense MLP (layers < first_k_dense_replace).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;

    void Free();
};

// Resident (dense) device graph. Routed experts are NOT part of it: they
// stream through GcExpertCache (gigachat_moe_expert_cache.h).
struct GcWeights {
    // The embedding table stays in pinned host memory: decode reads exactly one
    // row per token (hidden*2 bytes), so keeping the 2*[vocab,hidden] BF16 table
    // device-resident wastes ~1.8 GiB that the expert cache (the decode wall)
    // can use. Step stages the selected row H2D. `d_embed` is always null.
    const uint8_t *d_embed = nullptr;        // unused (host-resident)
    const uint8_t *h_embed = nullptr;        // pinned host BF16 [vocab, hidden]
    void *h_embed_pinned = nullptr;          // cudaHostAlloc owner
    const uint8_t *d_lm_head = nullptr;      // BF16 [vocab, hidden]
    GcGatedNorm final_norm;                  // model.norm triple
    int hidden = 0;
    int vocab = 0;
    std::vector<GcLayerWeights> layers;  // [num_layers]
    std::uint64_t resident_bytes = 0;    // dense only
    void Free();
};

// Uploads the converted dense weights under <model_path>/logestix/ to the
// device. Strict: returns false and fills `error` on any missing/malformed
// tensor. Routed experts are handled by InitGigaChatExpertCache.
bool LoadGigaChatWeights(const std::string &model_path, const GigaChatIndex &idx,
                         GcWeights &w, cudaStream_t stream, std::string &error);

} // namespace lsxcommon::gigachat
