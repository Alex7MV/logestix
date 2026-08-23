#pragma once

#include <arrow/api.h>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace lsxcommon {
namespace kimi {

// Category of a single Kimi-K2.7-Code safetensors tensor name relative to the
// logestix conversion pipeline.
enum class KimiTensorKind {
    Unknown,
    RoutedExpert,   // layers.N.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale}
    SharedExpert,   // layers.N.mlp.shared_experts.{gate,up,down}_proj.weight
    Router,         // layers.N.mlp.gate.weight / .e_score_correction_bias
    DenseMlp,       // layers.0.mlp.{gate,up,down}_proj.weight
    Attention,      // layers.N.self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj / norms
    LayerNorm,      // layers.N.{input_layernorm,post_attention_layernorm}
    Global,         // embed_tokens / lm_head / model.norm
};

struct KimiParsedTensor {
    KimiTensorKind kind = KimiTensorKind::Unknown;
    int layer = -1;      // layer index, when applicable
    int expert = -1;     // expert index, RoutedExpert/SharedExpert only
    int weight_id = 0;   // {gate,up,down}_proj -> {1,2,3}
    std::string base;    // trailing component after <layer>.
};

// Classifies a full safetensors tensor name into a Kimi pipeline category.
// The "language_model." checkpoint prefix is stripped internally;
// vision_tower.* / mm_projector.* classify as Unknown (stripped fat).
KimiParsedTensor kimi_classify(const std::string &name);

// Topology of a converted Kimi-K2.7-Code model, derived from the source
// safetensors index: dense layer 0 only, MoE layers 1..60.
struct KimiTopology {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    std::vector<long long> moe_layers;
    std::vector<long long> dense_layers;
    std::string error_msg;

    bool is_moe_layer(long long l) const {
        return std::binary_search(moe_layers.begin(), moe_layers.end(), l);
    }
    bool is_dense_layer(long long l) const {
        return std::binary_search(dense_layers.begin(), dense_layers.end(), l);
    }
};

// Derives the topology from the source model.safetensors.index.json at
// <index_json_path> (weight_map key scan through kimi_classify). On failure
// sets error_msg.
KimiTopology kimi_load_topology(const std::string &index_json_path);

// Converts a Kimi-K2.7-Code safetensors model directory into the logestix
// layout ("logestix-kimi" extended index):
//   <output>/logestix-kimi/model.index.json
//   <output>/logestix-kimi/pinned/global.arrow
//   <output>/logestix-kimi/pinned/layer_NNN.arrow
//   <output>/logestix-kimi/pinned/shared_experts.arrow
//   <output>/logestix-kimi/experts/layer_NNN/exp_MMM.arrow
// Source routed experts are compressed-tensors INT4 group-32 packed; they are
// dequantized on the GPU (KimiInt4DequantToBf16) and re-quantized to NVFP4.
arrow::Status ConvertKimiModel(
    const std::string &input_path,
    const std::filesystem::path &output_root);

} // namespace kimi
} // namespace lsxcommon
