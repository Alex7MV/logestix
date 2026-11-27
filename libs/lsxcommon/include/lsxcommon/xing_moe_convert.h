#pragma once

#include <arrow/api.h>

#include <filesystem>
#include <string>
#include <vector>

namespace lsxcommon {
namespace xing {

// Category of a single Xing4.0-29B-A4B safetensors tensor name relative to the
// logestix conversion pipeline.
enum class XingTensorKind {
    Unknown,
    RoutedExpert,    // layers.N.mlp.experts.E.{gate,up,down}_proj.weight
    SharedExpert,    // layers.N.mlp.shared_experts.{gate,up,down}_proj.weight
    Router,          // layers.N.mlp.gate.weight + .e_score_correction_bias
    DenseMlp,        // layers.{0,1}.mlp.{gate,up,down}_proj.weight
    Attention,       // layers.N.self_attn.{q_a,q_b,kv_a_proj_with_mqa,kv_b,o}_proj
    HyperConnection, // layers.N.{attn_hc,ffn_hc}.{hc_fn,hc_base,hc_scale}
    LayerNorm,       // layers.N.{input,post_attention}_layernorm + MLA inner norms
    Global,          // embed_tokens / lm_head / model.norm
    Mtp,             // every tensor of the stored layers.40 speculation block
};

struct XingParsedTensor {
    XingTensorKind kind = XingTensorKind::Unknown;
    int layer = -1;      // layer index, when applicable
    int expert = -1;     // expert index, RoutedExpert only
    int weight_id = 0;   // {gate,up,down}_proj -> {1,2,3}
    std::string base;    // trailing component after <layer>.
};

// Classifies a full safetensors tensor name into a Xing pipeline category.
// Anything outside the text decoder (vision towers, mm_projector) classifies
// as Unknown and is stripped. Every tensor of the stored MTP block (layer >= 
// num_hidden_layers) classifies as Mtp.
XingParsedTensor xing_classify(const std::string &name);

// Topology of a converted Xing4.0-29B-A4B model, derived from the source
// safetensors index: dense layers 0..1, MoE layers 2..39, MTP block 40.
struct XingTopology {
    long long num_layers = 0;
    long long experts_per_layer = 0;
    long long hc_mult = 0;      // mHC stream count (frozen 4)
    std::vector<long long> moe_layers;
    std::vector<long long> dense_layers;
    std::vector<long long> mtp_layers;
    // Converted-index format tag ("logestix-xing-nvfp4"); empty for HF source
    // indexes (the converter, not the loader, consumes those).
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
    std::string error_msg;

    bool is_moe_layer(long long l) const
    {
        return std::binary_search(moe_layers.begin(), moe_layers.end(), l);
    }
    bool is_dense_layer(long long l) const
    {
        return std::binary_search(dense_layers.begin(), dense_layers.end(), l);
    }
};

// Derives the topology from the source model.safetensors.index.json at
// <index_json_path> (weight_map key scan through xing_classify). On failure
// sets error_msg.
XingTopology xing_load_topology(const std::string &index_json_path);

// Converts a Xing4.0-29B-A4B safetensors model directory into the logestix
// layout:
//   <output>/logestix/model.index.json
//   <output>/logestix/pinned/global.arrow
//   <output>/logestix/pinned/layer_NNN.arrow
//   <output>/logestix/experts/layer_NNN/exp_MMM.arrow
//   <output>/logestix/mtp/raw.arrow
// Dense linear matrices (MLA projections, dense MLP, routed + shared experts,
// embed/lm_head) become NVFP4 packed nibbles + per-block E4M3 scales. Norms,
// router bias/scales and the mHC mixing parameters stay BF16/F32.
arrow::Status ConvertXingModel(
    const std::string &input_path,
    const std::filesystem::path &output_root);

} // namespace xing
} // namespace lsxcommon