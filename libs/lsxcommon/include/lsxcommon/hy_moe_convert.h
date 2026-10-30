#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::hy {

// Tensor kinds of the Hy-Memory-30B-A3B (hy_v3) checkpoint.
// Every linear projection is NVFP4-quantized; norms, the router gate and the
// routing bias stay high precision (a 4-bit router does not blur an answer, it
// swaps the expert set).
enum class HyTensorKind {
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
    AttnQNorm,         // layers.N.self_attn.q_norm.weight      (per head, plain form)
    AttnKNorm,         // layers.N.self_attn.k_norm.weight
    DenseMlpGate,      // layers.N.mlp.gate_proj.weight   (dense prefix layers)
    DenseMlpUp,        // layers.N.mlp.up_proj.weight
    DenseMlpDown,      // layers.N.mlp.down_proj.weight
    RouterGate,        // layers.N.mlp.router.gate.weight            (BF16)
    RouterBias,        // layers.N.mlp.expert_bias                   (F32)
    ExpertGate,        // layers.N.mlp.experts.E.gate_proj.weight
    ExpertDown,        // layers.N.mlp.experts.E.down_proj.weight
    ExpertUp,          // layers.N.mlp.experts.E.up_proj.weight
    SharedGateProj,    // layers.N.mlp.shared_mlp.gate_proj.weight
    SharedUpProj,      // layers.N.mlp.shared_mlp.up_proj.weight
    SharedDownProj,    // layers.N.mlp.shared_mlp.down_proj.weight
};

struct ParsedHyTensor {
    HyTensorKind kind = HyTensorKind::Unknown;
    int layer = -1;
    int expert = -1;   // >= 0 for ExpertGate/Down/Up
};

// Classifies an HF hy_v3 tensor name.
ParsedHyTensor classify_hy_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter.
bool hy_tensor_is_nvfp4(HyTensorKind kind);

// True when `kind` is a routed-expert payload (host expert store, not pinned).
bool hy_tensor_is_routed_expert(HyTensorKind kind);

struct HyTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int dense_intermediate_size = 0;
    int moe_intermediate_size = 0;
    int shared_intermediate_size = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    double rms_norm_eps = 1e-5;
    double rope_theta = 11158840.0;
    double router_scaling_factor = 2.826;
    int eos_token_id = -1;
    int bos_token_id = -1;
    int pad_token_id = -1;
    // Per-layer dispatch, derived from mlp_layer_types.
    std::vector<std::string> mlp_types;    // "dense" | "sparse"

    int total_pinned_tensors = 0;
    int skipped_ballast_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses <input_dir>/config.json via simdjson. Token ids fall back to
// generation_config.json when the config omits them. On any structural
// problem the returned struct carries a non-empty error_msg.
HyTopology ParseHyConfig(const std::string &input_dir);

// Converts a Hy-Memory-30B-A3B model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout.
arrow::Status ConvertHyMoeModel(const std::string &input_dir,
                                const std::string &output_root);

} // namespace lsxcommon::hy
