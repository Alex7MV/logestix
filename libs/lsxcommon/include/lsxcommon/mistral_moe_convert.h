#pragma once

#include <arrow/api.h>

#include <filesystem>
#include <string>
#include <vector>

namespace lsxcommon {
namespace mistral_moe {

// Category of a single Mistral-Small-4-119B safetensors tensor name relative to
// the logestix conversion pipeline. The checkpoint ships the Mistral
// "consolidated" layout (no `model.` prefix; `layers.N.…`, `tok_embeddings`,
// `output`, `norm`).
enum class MistralTensorKind {
    Unknown,
    RoutedExpert,    // layers.N.experts.E.w{1,2,3}.{weight_packed,weight_scale,
                     //                        weight_global_scale,input_global_scale}
    SharedExpert,    // layers.N.shared_experts.w{1,2,3}.*
    Router,          // layers.N.gate.weight
    Attention,       // layers.N.attention.{wq_a,wq_b,wkv_a_with_mqa,wkv_b,wo}
    LayerNorm,       // layers.N.{attention_norm,ffn_norm} + MLA inner norms
    Global,          // tok_embeddings / output / norm
    Vision,          // vision_encoder / vision_language_adapter / patch_merger
};

struct MistralParsedTensor {
    MistralTensorKind kind = MistralTensorKind::Unknown;
    int layer = -1;      // layer index, when applicable
    int expert = -1;     // expert index, RoutedExpert only
    int weight_id = 0;   // w1/w2/w3 -> {1,2,3}
    std::string base;    // trailing component after <layer>.
    // Companion suffix for expert/attention tensors:
    // "" (the weight itself), "scale", "global_scale", "input_global_scale".
    std::string companion;
};

// Classifies a full safetensors tensor name into a Mistral pipeline category.
// Vision/audio tower tensors classify as Vision (stripped); anything else
// outside the text decoder is Unknown.
MistralParsedTensor mistral_classify(const std::string &name);

// Topology of a converted Mistral model, derived from the source safetensors
// index: every one of the 36 layers is MoE (first_k_dense_replace = 0).
struct MistralTopology {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    std::vector<long long> moe_layers;
    std::vector<long long> dense_layers;
    std::string format;
    std::string error_msg;

    bool is_moe_layer(long long l) const
    {
        return std::binary_search(moe_layers.begin(), moe_layers.end(), l);
    }
};

// Derives the topology from the source consolidated index at
// <index_json_path> (weight_map key scan through mistral_classify). On failure
// sets error_msg.
MistralTopology mistral_load_topology(const std::string &index_json_path);

// Converts a Mistral-Small-4-119B-2603-NVFP4 checkpoint directory into the
// logestix layout:
//   <output>/logestix/model.index.json
//   <output>/logestix/pinned/global.arrow        (embed/head/final norm, BF16)
//   <output>/logestix/pinned/layer_N.arrow       (attention/norms/gate, BF16)
//   <output>/logestix/pinned/shared_experts.arrow(NVFP4, one record per layer)
//   <output>/logestix/experts/layer_N/exp_M.arrow(NVFP4, one record per expert)
//   <output>/logestix/pinned/tokenizer_*.arrow
// The routed and shared experts are ALREADY NVFP4 (E2M1 packed + per-16 E4M3
// scales + a per-tensor F32 global scale), so they are repacked verbatim into
// the repo layout (no decode→requantize). Attention, the router, the norms,
// the embeddings and the head stay BF16 exactly as the checkpoint ships them
// (the checkpoint's own quantization ignore-list excludes them).
arrow::Status ConvertMistralModel(
    const std::string &input_path,
    const std::filesystem::path &output_root);

} // namespace mistral_moe
} // namespace lsxcommon
