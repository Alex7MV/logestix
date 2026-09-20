#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::ornith {

// Tensor kinds of the Ornith-1.5-35B-A3B (qwen3_5_moe) MLX export.
// MLX layout: language_model.model.* prefix, mlp.switch_mlp.* stacked experts,
// linear_attn.conv1d.weight [C, K, 1].
enum class OrnithMoeTensorKind {
    Unknown,
    Embed,               // language_model.model.embed_tokens.weight
    LmHead,              // language_model.lm_head.weight
    FinalNorm,           // language_model.model.norm.weight
    InputNorm,
    PostNorm,
    FullAttnQ,
    FullAttnK,
    FullAttnV,
    FullAttnO,
    FullAttnQNorm,
    FullAttnKNorm,
    LinearAttnQkv,
    LinearAttnZ,
    LinearAttnB,
    LinearAttnA,
    LinearAttnOut,
    LinearAttnConv,      // [C, K, 1] (normalized to [C, 1, K] on write)
    LinearAttnNorm,
    LinearAttnALog,
    LinearAttnDt,
    RouterGate,          // mlp.gate.weight [E,H] (BF16)
    SwitchGateProj,      // mlp.switch_mlp.gate_proj.weight [E,I,H] -> w1
    SwitchUpProj,        // mlp.switch_mlp.up_proj.weight   [E,I,H] -> w3
    SwitchDownProj,      // mlp.switch_mlp.down_proj.weight [E,H,I] -> w2
    SharedExpertGateProj,
    SharedExpertUpProj,
    SharedExpertDownProj,
    SharedExpertGate,
    Mtp,                 // mtp.* (absent here; recognized for other MLX exports)
    Visual,              // model.visual.* / visual.* — skipped and counted
};

struct ParsedOrnithTensor {
    OrnithMoeTensorKind kind = OrnithMoeTensorKind::Unknown;
    int layer = -1;
};

// Classifies an MLX qwen3_5_moe tensor name.
ParsedOrnithTensor classify_ornith_moe_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter. Norms, router gate
// matrix and shared-expert scalar gate always stay high precision.
bool ornith_moe_tensor_is_nvfp4(OrnithMoeTensorKind kind);

struct OrnithMoeTopology {
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
    int bos_token_id = -1;

    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses text_config from <input_dir>/config.json via simdjson. Falls back to
// flat root keys when text_config is absent. On any structural problem the
// returned struct carries a non-empty error_msg.
OrnithMoeTopology ParseOrnithMoeConfig(const std::string &input_dir);

// Frozen-geometry gate against lsxcommon::ornith::frozen. Exported so the
// convert test exercises it without the synthetic-geometry escape hatch.
bool CheckOrnithFrozenGeometry(const OrnithMoeTopology &top, std::string &why);

// Converts an MLX Ornith-1.5-35B-A3B model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout.
// Format tag: logestix-ornithmoe.
arrow::Status ConvertOrnithMoeModel(const std::string &input_dir,
                                    const std::string &output_root);

} // namespace lsxcommon::ornith
