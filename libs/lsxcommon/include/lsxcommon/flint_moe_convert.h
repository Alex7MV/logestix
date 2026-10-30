#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::flint {

// Tensor kinds of the RecursiveSI/flint checkpoint.
// Every projection is NVFP4-quantized. Norms, the per-head sinks, the router
// gate and the embed/head edges stay high precision: the sink is a softmax
// logit, the router moves the top-k SELECTION, and the head multiplies every
// logit.
enum class FlintTensorKind {
    Unknown,
    Embed,             // model.embed_tokens.weight
    LmHead,            // lm_head.weight
    FinalNorm,         // model.norm.weight
    InputNorm,         // layers.N.input_layernorm.weight
    PostNorm,          // layers.N.post_attention_layernorm.weight
    AttnQ,             // layers.N.self_attn.q_proj.weight
    AttnK,             // layers.N.self_attn.k_proj.weight
    AttnV,             // layers.N.self_attn.v_proj.weight
    AttnO,             // layers.N.self_attn.o_proj.weight
    AttnQNorm,         // layers.N.self_attn.q_norm.weight   (per head)
    AttnKNorm,         // layers.N.self_attn.k_norm.weight
    SinkLogit,         // layers.N.self_attn.softmax_offset (per head, BF16)
    DenseMlpGate,      // layers.N.mlp.gate_proj.weight   (dense prefix layers)
    DenseMlpUp,        // layers.N.mlp.up_proj.weight
    DenseMlpDown,      // layers.N.mlp.down_proj.weight
    RouterGate,        // layers.N.mlp.gate.weight            (BF16)
    ExpertGate,        // layers.N.mlp.experts.E.gate_proj.weight
    ExpertDown,        // layers.N.mlp.experts.E.down_proj.weight
    ExpertUp,          // layers.N.mlp.experts.E.up_proj.weight
    SharedGateProj,    // layers.N.mlp.shared_expert.gate_proj.weight
    SharedUpProj,      // layers.N.mlp.shared_expert.up_proj.weight
    SharedDownProj,    // layers.N.mlp.shared_expert.down_proj.weight
};

struct ParsedFlintTensor {
    FlintTensorKind kind = FlintTensorKind::Unknown;
    int layer = -1;
    int expert = -1;   // >= 0 for ExpertGate/Down/Up
};

// Classifies an HF Flint tensor name (unrecognised names are an error in the
// converter, never a silent skip).
ParsedFlintTensor classify_flint_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter.
bool flint_tensor_is_nvfp4(FlintTensorKind kind);

// True when `kind` is a routed-expert payload.
bool flint_tensor_is_routed_expert(FlintTensorKind kind);

struct FlintTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int dense_intermediate_size = 0;
    int moe_intermediate_size = 0;
    int shared_intermediate_size = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int sliding_window = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;
    int eos_token_id = -1;
    int bos_token_id = -1;
    int pad_token_id = -1;
    // Per-layer dispatch, derived from the config's own lists.
    std::vector<std::string> layer_types;   // "full_attention"|"sliding_attention"
    std::vector<int> no_rope_layers;
    std::vector<int> mlp_only_layers;       // dense-MLP layer ids
    std::vector<int> eos_ids;               // config.json eos_token_id list

    int total_pinned_tensors = 0;
    int skipped_ballast_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses <input_dir>/config.json via simdjson. On any structural problem the
// returned struct carries a non-empty error_msg.
FlintTopology ParseFlintConfig(const std::string &input_dir);

// Converts a Flint model directory (safetensors shards + config.json) into the
// unified <output_root>/logestix arrow layout.
arrow::Status ConvertFlintModel(const std::string &input_dir,
                                const std::string &output_root);

} // namespace lsxcommon::flint
