#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::minimax {

// Tensor kinds of the MiniMax-M3 (minimax_m3_vl text_config) checkpoint.
// Every linear projection is NVFP4-quantized; norms, the F32 router gate and
// the F32 routing bias stay high precision (a 4-bit router does not blur an
// answer, it swaps the expert set).
enum class MinimaxTensorKind {
    Unknown,
    Embed,             // language_model.model.embed_tokens.weight
    LmHead,            // language_model.lm_head.weight
    FinalNorm,         // language_model.model.norm.weight
    InputNorm,         // layers.N.input_layernorm.weight
    PostNorm,          // layers.N.post_attention_layernorm.weight
    AttnQ,             // layers.N.self_attn.q_proj.weight
    AttnK,             // layers.N.self_attn.k_proj.weight
    AttnV,             // layers.N.self_attn.v_proj.weight
    AttnO,             // layers.N.self_attn.o_proj.weight
    AttnQNorm,         // layers.N.self_attn.q_norm.weight      (per head, gemma form)
    AttnKNorm,         // layers.N.self_attn.k_norm.weight
    IndexQ,            // layers.N.self_attn.index_q_proj.weight
    IndexK,            // layers.N.self_attn.index_k_proj.weight
    IndexQNorm,        // layers.N.self_attn.index_q_norm.weight
    IndexKNorm,        // layers.N.self_attn.index_k_norm.weight
    DenseMlpGate,      // layers.N.mlp.gate_proj.weight   (dense prefix layers)
    DenseMlpUp,        // layers.N.mlp.up_proj.weight
    DenseMlpDown,      // layers.N.mlp.down_proj.weight
    RouterGate,        // layers.N.block_sparse_moe.gate.weight          (F32)
    RouterBias,        // layers.N.block_sparse_moe.e_score_correction_bias (F32)
    ExpertW1,          // layers.N.block_sparse_moe.experts.E.w1.weight  (gate)
    ExpertW2,          // layers.N.block_sparse_moe.experts.E.w2.weight  (down)
    ExpertW3,          // layers.N.block_sparse_moe.experts.E.w3.weight  (up)
    SharedGateProj,    // layers.N.block_sparse_moe.shared_experts.gate_proj.weight
    SharedUpProj,      // layers.N.block_sparse_moe.shared_experts.up_proj.weight
    SharedDownProj,    // layers.N.block_sparse_moe.shared_experts.down_proj.weight
    Visual,            // vision_tower.* / multi_modal_projector.* / patch_merge_mlp.*
};

struct ParsedMinimaxTensor {
    MinimaxTensorKind kind = MinimaxTensorKind::Unknown;
    int layer = -1;
    int expert = -1;   // >= 0 for ExpertW1/W2/W3
};

// Classifies an HF MiniMax-M3 tensor name.
ParsedMinimaxTensor classify_minimax_tensor(const std::string &name);

// True when `kind` is NVFP4-quantized by the converter.
bool minimax_tensor_is_nvfp4(MinimaxTensorKind kind);

// True when `kind` is a routed-expert payload (host mmap store, not pinned).
bool minimax_tensor_is_routed_expert(MinimaxTensorKind kind);

struct MinimaxTopology {
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
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_block_size = 0;
    int index_topk_blocks = 0;
    int index_local_blocks = 0;
    double rms_norm_eps = 1e-6;
    double rope_theta = 5000000.0;
    double routed_scaling_factor = 2.0;
    double swiglu_alpha = 1.702;
    double swiglu_limit = 7.0;
    int eos_token_id = -1;
    int bos_token_id = -1;
    int pad_token_id = -1;
    // Per-layer dispatch, derived from moe_layer_freq / sparse_attention_freq.
    std::vector<std::string> mlp_types;    // "dense" | "sparse"
    std::vector<std::string> layer_types;  // "dense_attention" | "sparse_attention"

    int total_pinned_tensors = 0;
    int skipped_visual_tensors = 0;
    VramProfile vram;        // conversion-time VRAM requirements (mandatory)
    std::string error_msg;   // non-empty when config/index is invalid
};

// Parses text_config from <input_dir>/config.json via simdjson; bos/eos/pad
// fall back to generation_config.json. Falls back to flat root keys when
// text_config is absent. On any structural problem the returned struct carries
// a non-empty error_msg.
MinimaxTopology ParseMinimaxConfig(const std::string &input_dir);

// Converts a MiniMax-M3 model directory (safetensors shards + config.json)
// into the unified <output_root>/logestix arrow layout (vault convention).
arrow::Status ConvertMinimaxMoeModel(const std::string &input_dir,
                                     const std::string &output_root);

} // namespace lsxcommon::minimax
