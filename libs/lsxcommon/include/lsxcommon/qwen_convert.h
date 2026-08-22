#pragma once

#include <arrow/api.h>
#include <optional>
#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::qwen {

enum class TensorKind {
    Unknown,
    Embed,           // model.language_model.embed_tokens.weight
    LmHead,          // lm_head.weight
    FinalNorm,       // model.language_model.norm.weight
    LinearAttnQkv,   // layers.N.linear_attn.in_proj_qkv.weight
    LinearAttnZ,     // layers.N.linear_attn.in_proj_z.weight
    LinearAttnB,     // layers.N.linear_attn.in_proj_b.weight
    LinearAttnA,     // layers.N.linear_attn.in_proj_a.weight
    LinearAttnOut,   // layers.N.linear_attn.out_proj.weight
    LinearAttnConv,  // layers.N.linear_attn.conv1d.weight
    LinearAttnNorm,  // layers.N.linear_attn.norm.weight
    LinearAttnA_log, // layers.N.linear_attn.A_log
    LinearAttnDt,    // layers.N.linear_attn.dt_bias
    FullAttnQ,       // layers.N.self_attn.q_proj.weight
    FullAttnK,       // layers.N.self_attn.k_proj.weight
    FullAttnV,       // layers.N.self_attn.v_proj.weight
    FullAttnO,       // layers.N.self_attn.o_proj.weight
    FullAttnQNorm,   // layers.N.self_attn.q_norm.weight
    FullAttnKNorm,   // layers.N.self_attn.k_norm.weight
    MlpGate,         // layers.N.mlp.gate_proj.weight
    MlpUp,           // layers.N.mlp.up_proj.weight
    MlpDown,         // layers.N.mlp.down_proj.weight
    InputNorm,       // layers.N.input_layernorm.weight
    PostNorm,        // layers.N.post_attention_layernorm.weight
    MTP,             // mtp.*
    Visual,          // visual.* (or model.visual.*)
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies an HF Qwen3.5 tensor name.
ParsedTensor classify_qwen_tensor(const std::string &name);

// True when `kind` should be NVFP4-quantized in the converter.
bool qwen_tensor_is_nvfp4(TensorKind kind);

struct QwenTopology {
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
    int linear_conv_kernel_dim = 0;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 10000000.0;
    int total_pinned_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when index.json is invalid
};

QwenTopology load_qwen_topology(const std::string &index_path);

arrow::Status WriteQwenModelIndex(
    const std::string &output_root,
    const QwenTopology &top);

// Converts a Qwen3.5 model directory (safetensors shards + config.json) into
// the logestix-qwen arrow layout under output_root.
arrow::Status ConvertQwenModel(
    const std::string &input_path,
    const std::string &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory and rewrites the "vram" block of the model.index.json under
// output_root/logestix.
arrow::Status RecomputeQwenVram(const std::string &input_path,
                                const std::string &output_root);

} // namespace lsxcommon::qwen