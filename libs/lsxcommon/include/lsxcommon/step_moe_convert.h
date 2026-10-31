#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::step {

// Tensor kinds of the Step-5-Preview checkpoint (step5_vl envelope, step5 text
// backbone). Every attention/dense/share-expert projection is BF16 -> NVFP4;
// the routed experts are FUSED F8_E4M3 stacks [E, N, K] with a 128x128 F32
// scale grid, split per expert and re-quantized to NVFP4. Norms, the router
// gate/bias and the DSA indexer stay high precision.
enum class StepTensorKind {
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
    AttnG,             // layers.N.self_attn.g_proj.weight   (per-head gate)
    AttnQNorm,         // layers.N.self_attn.q_norm.weight
    AttnKNorm,         // layers.N.self_attn.k_norm.weight
    DenseMlpGate,      // layers.N.mlp.gate_proj.weight     (layers 0..2)
    DenseMlpUp,        // layers.N.mlp.up_proj.weight
    DenseMlpDown,      // layers.N.mlp.down_proj.weight
    RouterGate,        // layers.N.moe.gate.weight          (BF16)
    RouterBias,        // layers.N.moe.router_bias          (F32)
    ExpertGate,        // layers.N.moe.gate_proj.weight     (F8_E4M3 + scale)
    ExpertDown,        // layers.N.moe.down_proj.weight
    ExpertUp,          // layers.N.moe.up_proj.weight
    ExpertGateScale,   // layers.N.moe.gate_proj.weight_scale_inv
    ExpertDownScale,   // layers.N.moe.down_proj.weight_scale_inv
    ExpertUpScale,     // layers.N.moe.up_proj.weight_scale_inv
    SharedGateProj,    // layers.N.share_expert.gate_proj.weight
    SharedUpProj,      // layers.N.share_expert.up_proj.weight
    SharedDownProj,    // layers.N.share_expert.down_proj.weight
    IndexQ,            // layers.N.self_attn.sparse_indexer_q.weight
    IndexK,            // layers.N.self_attn.sparse_indexer_k.weight
    IndexZ,            // layers.N.self_attn.sparse_indexer_z.weight
    IndexW,            // layers.N.self_attn.sparse_indexer_w.weight
    IndexQNorm,        // layers.N.self_attn.sparse_indexer_q_norm.weight
    IndexKNorm,        // layers.N.self_attn.sparse_indexer_k_norm.weight
    IndexKNormBias,    // layers.N.self_attn.sparse_indexer_k_norm.bias
    IndexSsmax,        // layers.N.self_attn.ssmax_s
    VisionBallast,     // vision_model.* / vit_large_projector.* / vit
    MtpBallast,        // layers 92..94 (enorm/hnorm/eh_proj/shared_head/...)
};

struct ParsedStepTensor {
    StepTensorKind kind = StepTensorKind::Unknown;
    int layer = -1;
};

// Classifies an HF Step-5-Preview tensor name.
ParsedStepTensor classify_step_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter.
bool step_tensor_is_nvfp4(StepTensorKind kind);

// True when `kind` is a routed-expert payload or its scale grid.
bool step_tensor_is_routed_expert(StepTensorKind kind);

struct StepTopology {
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
    int first_moe_layer = 0;
    int num_mtp_layers = 0;
    int sliding_window = 0;
    double rms_norm_eps = 1e-5;
    double router_scaling_factor = 3.0;
    int eos_token_id = -1;
    int eos_token_id2 = -1;
    int bos_token_id = -1;
    int pad_token_id = -1;
    // Per-layer dispatch, derived from `moe_layer_list`.
    std::vector<std::string> mlp_types;    // "dense" | "sparse"
    std::vector<std::string> layer_types;  // "sliding_attention" | "full_attention"

    int total_pinned_tensors = 0;
    int skipped_ballast_tensors = 0;
    int skipped_mtp_tensors = 0;
    int indexer_tensors = 0;
    VramProfile vram;
    std::string error_msg;
};

// Parses <input_dir>/config.json (the `text_config` block) via simdjson.
StepTopology ParseStepConfig(const std::string &input_dir);

// Converts a Step-5-Preview model directory into <output_root>/logestix.
arrow::Status ConvertStepMoeModel(const std::string &input_dir,
                                  const std::string &output_root);

} // namespace lsxcommon::step
