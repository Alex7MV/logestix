#pragma once

#include <arrow/api.h>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace glm {

// Category of a single GLM-5.2 safetensors tensor name relative to the
// logestix conversion pipeline.
enum class TensorKind {
    Unknown,
    RoutedExpert,   // layers.N.mlp.experts.E.{gate,down,up}_proj.weight
    SharedExpert,   // layers.N.mlp.shared_experts.{gate,down,up}_proj.weight
    Router,         // layers.N.mlp.gate.weight / .e_score_correction_bias
    DenseMlp,       // layers.N.mlp.{gate,down,up}_proj.weight (layers 0-2)
    Attention,      // layers.N.self_attn.{q_a,q_b,kv_a,kv_b,o}_proj / norms
    Indexer,        // layers.N.self_attn.indexer.*
    LayerNorm,      // layers.N.{input_layernorm,post_attention_layernorm} /
                    // MTP enorm/hnorm/shared_head.norm
    Global,         // embed_tokens / lm_head / model.norm
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;      // layer index, when applicable
    int expert = -1;     // expert index, RoutedExpert only
    int weight_id = 0;   // {gate,down,up}_proj -> {1,2,3}
    std::string base;    // trailing component after <layer>., e.g. "q_b_proj"
};

// Classifies a full safetensors tensor name into a GLM pipeline category.
ParsedTensor classify(const std::string &name);

// Topology of a converted GLM-5.2 model, read from the extended
// `model.index.json` (format "logestix-glm"). Unlike DeepSeek's uniform
// grid, GLM has heterogeneous layers: dense (0-2), MoE (3-78), and indexer
// layers (0,1,2,6,...,74,78), so moe/dense/indexer are explicit lists.
struct Topology {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    long long total_experts = 0;
    std::vector<long long> moe_layers;
    std::vector<long long> dense_layers;
    std::vector<long long> indexer_layers;
    VramProfile vram;   // conversion-time VRAM requirements (mandatory block)
    std::string error_msg;

    bool is_moe_layer(long long l) const {
        return std::binary_search(moe_layers.begin(), moe_layers.end(), l);
    }
    bool is_dense_layer(long long l) const {
        return std::binary_search(dense_layers.begin(), dense_layers.end(), l);
    }
    bool is_indexer_layer(long long l) const {
        return std::binary_search(indexer_layers.begin(), indexer_layers.end(), l);
    }
};

// Parses <index_json_path> into a Topology. On failure sets error_msg.
Topology load_topology(const std::string &index_json_path);

// Converts a GLM-5.2 safetensors model directory into the logestix layout:
//   <output>/model.index.json                  — "logestix-glm" extended index
//   <output>/pinned/global.arrow               — embed/lm_head/model.norm/etc
//   <output>/pinned/layer_NNN.arrow            — per-layer attn/router/indexer/norm
//   <output>/pinned/shared_experts.arrow       — NVFP4-packed shared experts
//   <output>/experts/layer_NNN/exp_MMM.arrow   — NVFP4-packed routed experts
// Source weights are raw BF16 (no paired scales); routed/shared experts are
// NVFP4-quantized on the GPU via PackExpertNVFP4 (converter.h).
arrow::Status ConvertGlmModel(
    const std::string &input_path,
    const std::filesystem::path &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory and rewrites the "vram" block of <output_root>/model.index.json.
arrow::Status RecomputeGlmVram(
    const std::string &input_path,
    const std::filesystem::path &output_root);

} // namespace glm
} // namespace lsxcommon