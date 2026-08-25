#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::qwen_moe {

// Tensor kinds of the Qwen3.5-35B-A3B (qwen3_5_moe) checkpoint. Attention and
// norm kinds mirror qwen::TensorKind so dense-part records stay compatible
// with the qwen loader; MoE kinds are new.
enum class MoeTensorKind {
    Unknown,
    Embed,            // model.language_model.embed_tokens.weight
    LmHead,           // lm_head.weight
    FinalNorm,        // model.language_model.norm.weight
    LinearAttnQkv,    // layers.N.linear_attn.in_proj_qkv.weight
    LinearAttnZ,      // layers.N.linear_attn.in_proj_z.weight
    LinearAttnB,      // layers.N.linear_attn.in_proj_b.weight
    LinearAttnA,      // layers.N.linear_attn.in_proj_a.weight
    LinearAttnOut,    // layers.N.linear_attn.out_proj.weight
    LinearAttnConv,   // layers.N.linear_attn.conv1d.weight
    LinearAttnNorm,   // layers.N.linear_attn.norm.weight
    LinearAttnALog,   // layers.N.linear_attn.A_log
    LinearAttnDt,     // layers.N.linear_attn.dt_bias
    FullAttnQ,        // layers.N.self_attn.q_proj.weight (gate-doubled rows)
    FullAttnK,        // layers.N.self_attn.k_proj.weight
    FullAttnV,        // layers.N.self_attn.v_proj.weight
    FullAttnO,        // layers.N.self_attn.o_proj.weight
    FullAttnQNorm,    // layers.N.self_attn.q_norm.weight
    FullAttnKNorm,    // layers.N.self_attn.k_norm.weight
    InputNorm,        // layers.N.input_layernorm.weight
    PostNorm,         // layers.N.post_attention_layernorm.weight
    RouterGate,       // layers.N.mlp.gate.weight [E,H] — stays BF16
    FusedExpertsGateUp,   // layers.N.mlp.experts.gate_up_proj [E,2I,H]
    FusedExpertsDown,     // layers.N.mlp.experts.down_proj [E,H,I]
    SharedExpertGateProj, // layers.N.mlp.shared_expert.gate_proj.weight
    SharedExpertUpProj,   // layers.N.mlp.shared_expert.up_proj.weight
    SharedExpertDownProj, // layers.N.mlp.shared_expert.down_proj.weight
    SharedExpertGate,     // layers.N.mlp.shared_expert_gate.weight [1,H]
    Mtp,              // mtp.* — converted+stored, never applied at inference
    Visual,           // visual/model.visual.* — skipped and counted
};

struct ParsedMoeTensor {
    MoeTensorKind kind = MoeTensorKind::Unknown;
    int layer = -1;
};

// Classifies an HF Qwen3.5-MoE tensor name.
ParsedMoeTensor classify_qwen_moe_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter. Norms, router gate
// matrix and shared-expert scalar gate always stay high precision.
bool qwen_moe_tensor_is_nvfp4(MoeTensorKind kind);

struct QwenMoeTopology {
    // Dense fields (names mirror qwen::QwenTopology for kernel reuse).
    int num_layers = 0;
    std::vector<std::string> layer_types;
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 4;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 10000000.0;

    // MoE fields.
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int eos_token_id = -1;

    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses text_config from <input_dir>/config.json via simdjson. Falls back to
// flat root keys when text_config is absent. On any structural problem the
// returned struct carries a non-empty error_msg.
QwenMoeTopology ParseQwenMoeConfig(const std::string &input_dir);

// Converts a Qwen3.5-35B-A3B model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout (vault convention).
arrow::Status ConvertQwen35MoeModel(const std::string &input_dir,
                                    const std::string &output_root);

} // namespace lsxcommon::qwen_moe
