#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::alice {

// Tensor kinds of the AliceAI-Foundation-80B-A3B-Base checkpoint
// (model_type=alice_ai). KDA = linear attention; Fa = gated full attention;
// AttnRes = block attention-residual depth mixer.
enum class AliceTensorKind {
    Unknown,
    Embed,             // model.embed_tokens.weight
    LmHead,            // lm_head.weight
    FinalNorm,         // model.norm.weight
    AttnResFinalProj,  // model.attnres_final.res_proj.weight
    AttnResFinalNorm,  // model.attnres_final.res_norm_weight
    InputNorm,         // layers.N.input_layernorm.weight
    PostNorm,          // layers.N.post_attention_layernorm.weight
    MlpResProj,        // layers.N.mlp_res_proj.weight
    MlpResNorm,        // layers.N.mlp_res_norm_weight
    AttnResProj,       // layers.N.attn_res_proj.weight (layers >= 1)
    AttnResNorm,       // layers.N.attn_res_norm_weight (layers >= 1)
    KdaQ,              // layers.N.linear_attn.q_proj.weight
    KdaK,              // layers.N.linear_attn.k_proj.weight
    KdaV,              // layers.N.linear_attn.v_proj.weight
    KdaO,              // layers.N.linear_attn.o_proj.weight
    KdaFA,             // layers.N.linear_attn.f_a_proj.weight
    KdaFB,             // layers.N.linear_attn.f_b_proj.weight
    KdaGA,             // layers.N.linear_attn.g_a_proj.weight
    KdaGB,             // layers.N.linear_attn.g_b_proj.weight
    KdaB,              // layers.N.linear_attn.b_proj.weight
    KdaQConv,          // layers.N.linear_attn.q_conv1d.weight
    KdaKConv,          // layers.N.linear_attn.k_conv1d.weight
    KdaVConv,          // layers.N.linear_attn.v_conv1d.weight
    KdaALog,           // layers.N.linear_attn.a_log_bias
    KdaDt,             // layers.N.linear_attn.dt_bias
    KdaONorm,          // layers.N.linear_attn.o_norm.weight
    FaQ,               // layers.N.self_attn.q_proj.weight (2x: query + gate)
    FaK,               // layers.N.self_attn.k_proj.weight
    FaV,               // layers.N.self_attn.v_proj.weight
    FaO,               // layers.N.self_attn.o_proj.weight
    FaQNorm,           // layers.N.self_attn.q_norm.weight
    FaKNorm,           // layers.N.self_attn.k_norm.weight
    RouterGate,        // layers.N.mlp.gate.weight
    RouterBias,        // layers.N.mlp.gate.e_score_correction_bias
    SharedGateProj,    // layers.N.mlp.shared_expert.gate_proj.weight
    SharedUpProj,      // layers.N.mlp.shared_expert.up_proj.weight
    SharedDownProj,    // layers.N.mlp.shared_expert.down_proj.weight
    SharedGate,        // layers.N.mlp.shared_expert_gate.weight
    FusedExpertsGateUp,  // layers.N.mlp.experts.gate_up_proj [E,2I,H]
    FusedExpertsDown,    // layers.N.mlp.experts.down_proj [E,H,I]
    Mtp,                 // mtp.* subtree (stored + counted, never applied)
};

struct ParsedAliceTensor {
    AliceTensorKind kind = AliceTensorKind::Unknown;
    int layer = -1;
};

// Parses config.json (flat alice_ai layout) into the topology. Fills
// `error_msg` and leaves fields at defaults on failure.
struct AliceTopology {
    int hidden_size = 0;
    int num_layers = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 4;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int block_attn_res_block_size = 4;
    int mtp_num_hidden_layers = 0;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 1e6;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = -1;
    int total_pinned_tensors = 0;
    std::vector<std::string> layer_types;
    std::string error_msg;
};

AliceTopology ParseAliceConfig(const std::string &input_dir);
ParsedAliceTensor classify_alice_tensor(const std::string &name);
bool alice_tensor_is_nvfp4(AliceTensorKind kind);

// Converts the BF16 checkpoint into the logestix-alice-nvfp4 layout under
// <output_root>/logestix.
arrow::Status ConvertAliceModel(const std::string &input_dir,
                                const std::string &output_root);

} // namespace lsxcommon::alice
