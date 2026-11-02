#pragma once

#include <arrow/api.h>

#include <string>
#include <vector>

#include "lsxcommon/vram_estimate.h"

namespace lsxcommon::kimi_k3 {

// Tensor kinds of the released Kimi-K3 checkpoint
// (KimiK3ForConditionalGeneration -> KimiLinearForCausalLM, model_type
// `kimi_k3` / `kimi_linear`). Names are matched after the leading
// `language_model.` scope.
enum class KimiK3TensorKind {
    Unknown,
    // globals
    Embed,               // model.embed_tokens.weight
    LmHead,              // lm_head.weight
    FinalNorm,           // model.norm.weight
    OutputAttnResProj,   // model.output_attn_res_proj.weight
    OutputAttnResNorm,   // model.output_attn_res_norm.weight
    // per-layer common
    InputNorm,           // layers.N.input_layernorm.weight
    PostNorm,            // layers.N.post_attention_layernorm.weight
    AttnResProj,         // layers.N.self_attention_res_proj.weight
    AttnResNorm,         // layers.N.self_attention_res_norm.weight
    MlpResProj,          // layers.N.mlp_res_proj.weight
    MlpResNorm,          // layers.N.mlp_res_norm.weight
    // gated MLA (layers 3,7,...,91,92 0-based)
    MlaQA,               // self_attn.q_a_proj.weight
    MlaQANorm,           // self_attn.q_a_layernorm.weight
    MlaQB,               // self_attn.q_b_proj.weight
    MlaKvA,              // self_attn.kv_a_proj_with_mqa.weight
    MlaKvANorm,          // self_attn.kv_a_layernorm.weight
    MlaKvB,              // self_attn.kv_b_proj.weight
    MlaO,                // self_attn.o_proj.weight
    MlaG,                // self_attn.g_proj.weight (output gate)
    // KDA (all other layers)
    KdaQ,                // self_attn.q_proj.weight
    KdaK,                // self_attn.k_proj.weight
    KdaV,                // self_attn.v_proj.weight
    KdaO,                // self_attn.o_proj.weight
    KdaG,                // self_attn.g_proj.weight (full-rank gate)
    KdaFA,               // self_attn.f_a_proj.weight (decay rank)
    KdaFB,               // self_attn.f_b_proj.weight (decay)
    KdaB,                // self_attn.b_proj.weight (beta)
    KdaQConv,            // self_attn.q_conv1d.weight (F32)
    KdaKConv,            // self_attn.k_conv1d.weight (F32)
    KdaVConv,            // self_attn.v_conv1d.weight (F32)
    KdaALog,             // self_attn.A_log (F32, per channel)
    KdaDt,               // self_attn.dt_bias (F32)
    KdaONorm,            // self_attn.o_norm.weight (F32)
    // dense MLP (layer 0)
    DenseGate,           // mlp.gate_proj.weight
    DenseUp,             // mlp.up_proj.weight
    DenseDown,           // mlp.down_proj.weight
    // Stable LatentMoE
    RouterGate,          // block_sparse_moe.gate.weight
    RouterBias,          // block_sparse_moe.gate.e_score_correction_bias
    LatentDown,          // block_sparse_moe.routed_expert_down_proj.weight
    LatentUp,            // block_sparse_moe.routed_expert_up_proj.weight
    LatentNorm,          // block_sparse_moe.routed_expert_norm.weight
    SharedGate,          // block_sparse_moe.shared_experts.gate_proj.weight
    SharedUp,            // block_sparse_moe.shared_experts.up_proj.weight
    SharedDown,          // block_sparse_moe.shared_experts.down_proj.weight
    ExpertW1Packed,      // block_sparse_moe.experts.E.w1.weight_packed
    ExpertW1Scale,       // block_sparse_moe.experts.E.w1.weight_scale
    ExpertW2Packed,      // block_sparse_moe.experts.E.w2.weight_packed
    ExpertW2Scale,       // block_sparse_moe.experts.E.w2.weight_scale
    ExpertW3Packed,      // block_sparse_moe.experts.E.w3.weight_packed
    ExpertW3Scale,       // block_sparse_moe.experts.E.w3.weight_scale
    // ballast (stripped + counted)
    VisionTower,         // vision_tower.*
    Projector,           // mm_projector.*
};

struct ParsedKimiK3Tensor {
    KimiK3TensorKind kind = KimiK3TensorKind::Unknown;
    int layer = -1;
    int expert = -1;
};

// Parses config.json (KimiK3 wrapper -> text_config) into the topology.
// Fills `error_msg` and leaves fields at defaults on failure.
struct KimiK3ConvertTopology {
    int hidden_size = 0;
    int num_layers = 0;
    int vocab_size = 0;
    int dense_layers = 1;
    int attn_res_block = 0;
    int dense_intermediate = 0;
    // MLA
    int mla_heads = 0;
    int kv_heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    bool mla_nope = true;
    bool mla_output_gate = true;
    // KDA
    int kda_heads = 0;
    int kda_head_dim = 0;
    int conv_kernel = 4;
    bool kda_full_rank_gate = true;
    double kda_gate_lower_bound = -5.0;
    // MoE
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int latent_dim = 0;
    bool latent_norm = true;
    int shared_experts = 0;
    int shared_intermediate = 0;
    bool moe_renormalize = true;
    int expert_group = 1;
    int topk_group = 1;
    double route_scale = 1.0;
    // numerics
    double rms_norm_eps = 1e-5;
    double situ_beta = 1.0;
    double situ_linear_beta = 0.0;
    // ids
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    // layer split (0-based)
    std::vector<int> mla_layers;
    std::vector<int> kda_layers;
    // counters
    int skipped_visual_tensors = 0;
    int skipped_projector_tensors = 0;
    int total_pinned_tensors = 0;
    std::string error_msg;
};

KimiK3ConvertTopology ParseKimiK3Config(const std::string &input_dir);
ParsedKimiK3Tensor ClassifyKimiK3Tensor(const std::string &name);

// True when the tensor is uploaded to the device for the whole session.
bool KimiK3TensorIsPinnedNvfp4(KimiK3TensorKind kind);
// True when the tensor lives in the mmap-streamed latent/shared MoE plane.
bool KimiK3TensorIsStreamed(KimiK3TensorKind kind);

// Converts the checkpoint into the logestix-kimi-k3-nvfp4 layout under
// <output_root>/logestix. Routed experts are repacked exactly (MXFP4 E2M1
// nibbles verbatim, E8M0/32 -> two identical pow2-E4M3/16 scales with a
// per-tensor power-of-two global scale); everything else is BF16 -> NVFP4 or
// raw passthrough.
arrow::Status ConvertKimiK3Model(const std::string &input_dir,
                                 const std::string &output_root);

}  // namespace lsxcommon::kimi_k3
