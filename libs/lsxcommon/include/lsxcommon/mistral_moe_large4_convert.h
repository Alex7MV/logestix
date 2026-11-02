#pragma once

#include <arrow/api.h>

#include <filesystem>
#include <string>
#include <vector>

namespace lsxcommon {
namespace mistral_large4 {

// Category of a Mistral-Large-4-1T-A52B safetensors tensor. The checkpoint uses
// the Mistral "consolidated" layout (no `model.` prefix; `layers.N.*`,
// `tok_embeddings`, `output`, `norm`, `lerope.alpha`).
enum class Large4TensorKind {
    Unknown,
    RoutedExpert,  // layers.N.experts.E.w{1,2,3}.{weight,qscale_weight}
    SharedExpert,  // layers.N.shared_experts.w{1,2,3}.{weight,qscale_weight}
    DenseMlp,      // layers.N.feed_forward.w{1,2,3}.{weight,qscale_weight}
    Router,        // layers.N.gate.weight (BF16, never quantized)
    AttnNvfp4,     // attention.{wq,wo}.{weight,qscale_weight}
    IndexerNvfp4,  // attention.indexer.wq_b.{weight,qscale_weight}
    AttnRaw,       // attention.{wkv_a_with_mqa,wkv_b}.weight (BF16)
    LayerRaw,      // norms, ssmax, indexer wk/weights_proj/norms (BF16)
    Global,        // tok_embeddings / output / norm / lerope.alpha
    Ballast,       // vision tower / adapter / patch merger / mtp_dspark
};

struct Large4ParsedTensor {
    Large4TensorKind kind = Large4TensorKind::Unknown;
    int layer = -1;
    int expert = -1;
    int weight_id = 0;   // w1/w2/w3 -> 1/2/3
    std::string base;    // canonical component name written into the tree
    // "" (the weight itself) or "scale" (its F32 per-128x128 tile scale).
    std::string companion;
};

Large4ParsedTensor mistral_large4_classify(const std::string &name);

// Parsed text_config of a Mistral-Large-4 checkpoint.
struct Large4Topology {
    int hidden = 0;
    int layers = 0;
    int vocab = 0;
    int heads = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int dense_intermediate = 0;
    int index_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    int context_length = 0;
    std::vector<int> rope_layers;
    std::vector<int> nope_layers;
    std::vector<int> kv_source;
    std::vector<int> indexer_full;
    std::vector<int> moe_layers;
    std::vector<int> dense_layers;
    std::string error_msg;
};

Large4Topology mistral_large4_load_topology(const std::string &config_json_path);

// Converts a Mistral-Large-4-1T-A52B checkpoint directory into the logestix
// layout:
//   <output>/logestix/model.index.json
//   <output>/logestix/pinned/global.arrow          (embed/head/norm/lerope)
//   <output>/logestix/pinned/layer_N.arrow         (BF16 edges + norms + router)
//   <output>/logestix/pinned/layer_N_nvfp4.arrow   (NVFP4 projections, shared
//                                                   and dense MLP)
//   <output>/logestix/experts/layer_N/exp_M.arrow  (NVFP4 routed experts)
//   <output>/logestix/pinned/tokenizer_*.arrow
// FP8 E4M3 tensors with F32 per-128x128 tile scales are dequantized to BF16 on
// the GPU and re-quantized to NVFP4; BF16 tensors on the checkpoint's own
// ignore list (norms, kv_a/kv_b, indexer wk/weights_proj, router, embeddings,
// head) are copied verbatim.
arrow::Status ConvertMistralLarge4Model(
    const std::string &input_path,
    const std::filesystem::path &output_root);

} // namespace mistral_large4
} // namespace lsxcommon
