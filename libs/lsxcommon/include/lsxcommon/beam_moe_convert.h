#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::beam {

// Tensor kinds of the reflection/Beam-1.0-501B-A23B checkpoint.
// Projections are NVFP4-quantized. Norms, q/k head norms, the attention gate
// bias, the final norm and the embed/head edges stay high precision; the
// router weight and its correction bias stay F32 because they move the top-k
// SELECTION (an NVFP4 router swaps the expert set, it does not blur an output).
enum class BeamTensorKind {
    Unknown,
    Embed,             // model.embed_tokens.weight
    LmHead,            // lm_head.weight
    FinalNorm,         // model.norm.weight
    InputNorm,         // layers.N.input_layernorm.weight
    PostAttnNorm,      // layers.N.post_attention_layernorm.weight
    PreFfnNorm,        // layers.N.pre_feedforward_layernorm.weight
    PostFfnNorm,       // layers.N.post_feedforward_layernorm.weight
    AttnQ,             // layers.N.self_attn.q_proj.weight
    AttnK,             // layers.N.self_attn.k_proj.weight
    AttnV,             // layers.N.self_attn.v_proj.weight
    AttnO,             // layers.N.self_attn.o_proj.weight
    AttnGate,          // layers.N.self_attn.gate_proj.weight
    AttnGateBias,      // layers.N.self_attn.gate_bias        [NH]
    AttnQNorm,         // layers.N.self_attn.q_norm.weight    [HD]
    AttnKNorm,         // layers.N.self_attn.k_norm.weight    [HD]
    DenseMlpGate,      // layers.N.mlp.gate_proj.weight   (dense prefix layers)
    DenseMlpUp,        // layers.N.mlp.up_proj.weight
    DenseMlpDown,      // layers.N.mlp.down_proj.weight
    RouterGate,        // layers.N.mlp.gate.weight             F32 [E,H]
    RouterBias,        // layers.N.mlp.gate.e_score_correction_bias F32 [E]
    ExpertGate,        // layers.N.mlp.experts.E.gate_proj.weight
    ExpertDown,        // layers.N.mlp.experts.E.down_proj.weight
    ExpertUp,          // layers.N.mlp.experts.E.up_proj.weight
};

struct ParsedBeamTensor {
    BeamTensorKind kind = BeamTensorKind::Unknown;
    int layer = -1;
    int expert = -1;   // >= 0 for ExpertGate/Down/Up
};

// Classifies an HF Beam tensor name (unrecognised names are an error in the
// converter, never a silent skip).
ParsedBeamTensor classify_beam_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter.
bool beam_tensor_is_nvfp4(BeamTensorKind kind);

// True when `kind` is a routed-expert payload.
bool beam_tensor_is_routed_expert(BeamTensorKind kind);

struct BeamTopology {
    int num_layers = 0;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int dense_intermediate_size = 0;
    int moe_intermediate_size = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int n_group = 0;
    int topk_group = 0;
    double routed_scaling_factor = 0.0;
    double final_logit_softcapping = 0.0;
    int sliding_window = 0;
    double rms_norm_eps = 0.0;
    double rope_theta = 0.0;
    int eos_token_id = -1;
    int eos_token_id_alt = -1;
    int bos_token_id = -1;
    int pad_token_id = -1;
    // Per-layer dispatch, derived from the config's own lists.
    std::vector<std::string> layer_types;     // "full_attention"|"sliding_attention"
    std::vector<std::string> mlp_layer_types; // "dense"|"sparse"
    std::vector<int> eos_ids;                 // config.json eos_token_id list

    int total_pinned_tensors = 0;
    int skipped_ballast_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses <input_dir>/config.json via simdjson. On any structural problem the
// returned struct carries a non-empty error_msg.
BeamTopology ParseBeamConfig(const std::string &input_dir);

// Converts a Beam model directory (safetensors shards + config.json) into the
// unified <output_root>/logestix arrow layout.
arrow::Status ConvertBeamModel(const std::string &input_dir,
                               const std::string &output_root);

} // namespace lsxcommon::beam
