#pragma once

#include <arrow/api.h>
#include <optional>
#include <string>
#include <vector>

#include "lsxcommon/phi_dense_constants.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace phi {

// Structural classification of an HF phi tensor name. The real phi-4
// checkpoint stores the attention and MLP projections PRE-FUSED
// (`qkv_proj` = q|k|v, `gate_up_proj` = gate|up), so there is no separate
// q/k/v or gate/up taxonomy.
enum class TensorKind {
    Unknown,
    Embed,       // model.embed_tokens.weight
    LmHead,      // lm_head.weight
    FinalNorm,   // model.norm.weight
    InputNorm,   // model.layers.N.input_layernorm.weight
    PostNorm,    // model.layers.N.post_attention_layernorm.weight
    AttnQKV,     // model.layers.N.self_attn.qkv_proj.weight (fused q|k|v)
    AttnO,       // model.layers.N.self_attn.o_proj.weight
    MlpGateUp,   // model.layers.N.mlp.gate_up_proj.weight (fused gate|up)
    MlpDown,     // model.layers.N.mlp.down_proj.weight
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies an HF phi tensor name. Unknown kinds are ignored (the phi
// checkpoint carries no multimodal/MTP ballast, but unknown names are still
// skipped rather than failing the conversion).
ParsedTensor classify_phi_tensor(const std::string &name);

// True when `kind` is quantized to NVFP4 by the converter (all dense linears,
// including embed/lm_head). Norms stay raw (FP16).
bool phi_tensor_is_nvfp4(TensorKind kind);

PhiTopology load_phi_topology(const std::string &index_path);

arrow::Status WritePhiModelIndex(
    const std::string &output_root,
    const PhiTopology &top);

// Converts a phi model directory (safetensors shards + config.json) into the
// logestix-phi arrow layout under output_root.
arrow::Status ConvertPhiModel(
    const std::string &input_path,
    const std::string &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory and rewrites the "vram" block of the model.index.json under
// output_root/logestix.
arrow::Status RecomputePhiVram(
    const std::string &input_path,
    const std::string &output_root);

} // namespace phi
} // namespace lsxcommon
