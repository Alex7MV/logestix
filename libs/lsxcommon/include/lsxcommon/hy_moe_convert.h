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
    // Every stop id the checkpoint declares (`generation_config.json` may
    // carry an ARRAY). Hy-Memory stops on the single <eos>; Hy-SimulMT also
    // stops on <|hy_EOT|> (120008), so the generation loop must honour all.
    std::vector<int> eos_ids;
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

// Frozen-geometry expectations a caller checks the parsed config against.
// This is a neutral POD shared by the converter and the loader so a sibling
// variant (Hy-SimulMT) can reuse both without borrowing another model's
// `frozen` namespace.
struct HyPassportGeometry {
    int layers = 0;
    int hidden = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int vocab = 0;
    int dense_intermediate = 0;
    int moe_intermediate = 0;
    int shared_intermediate = 0;
    int num_experts = 0;
    int top_k = 0;
    int first_moe_layer = 0;
};

// The frozen passport as a geometry struct (one definition shared by the
// converter and the loader; dedup Ф5).
HyPassportGeometry HyMemoryPassportGeometry();

// Model identity + policy for one conversion. The default is Hy-Memory's;
// Hy-SimulMT supplies its own identity and audio-ballast policy.
struct HyConvertSpec {
    HyPassportGeometry geometry;
    std::string model_type = "hy-memory-30b-a3b";
    std::string model_name = "Hy-Memory-30B-A3B";
    // No `format_tag`: the index declares `layout` now (design §2.4.3), and a
    // field every writer set and no reader consumed is how the per-model tags
    // rotted to 11-of-18 dead in the first place.
    // Hy-SimulMT carries a speech encoder (`audio_tower.*`) beside the text
    // backbone. When set, those tensors are stripped AND counted.
    bool allow_audio_ballast = false;
};

// Converts a Hy-Memory-30B-A3B model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout.
arrow::Status ConvertHyMoeModel(const std::string &input_dir,
                                const std::string &output_root);

// Shared implementation: the same pipeline with an explicit identity/policy.
arrow::Status ConvertHyMoeModelWithSpec(const std::string &input_dir,
                                        const std::string &output_root,
                                        const HyConvertSpec &spec);

} // namespace lsxcommon::hy
