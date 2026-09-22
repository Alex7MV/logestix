#pragma once

#include <arrow/api.h>
#include <optional>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace granite {

// Structural classification of an HF Granite tensor name.
enum class TensorKind {
    Unknown,
    Embed,       // model.embed_tokens.weight
    LmHead,      // lm_head.weight
    FinalNorm,   // model.norm.weight
    InputNorm,   // model.layers.N.input_layernorm.weight
    PostNorm,    // model.layers.N.post_attention_layernorm.weight
    AttnQ,       // model.layers.N.self_attn.q_proj.weight
    AttnK,       // model.layers.N.self_attn.k_proj.weight
    AttnV,       // model.layers.N.self_attn.v_proj.weight
    AttnO,       // model.layers.N.self_attn.o_proj.weight
    MlpGate,     // model.layers.N.mlp.gate_proj.weight
    MlpUp,       // model.layers.N.mlp.up_proj.weight
    MlpDown,     // model.layers.N.mlp.down_proj.weight
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies an HF Granite tensor name. Unknown kinds are ignored (the
// Granite checkpoint carries no multimodal/MTP ballast, but unknown names are
// still skipped rather than failing the conversion).
ParsedTensor classify_granite_tensor(const std::string &name);

// True when `kind` is quantized to NVFP4 by the converter (all dense linears,
// including embed/lm_head). Norms stay raw (FP16).
bool granite_tensor_is_nvfp4(TensorKind kind);

GraniteTopology load_granite_topology(const std::string &index_path);

arrow::Status WriteGraniteModelIndex(
    const std::string &output_root,
    const GraniteTopology &top);

// Converts a Granite model directory (safetensors shards + config.json) into
// the logestix-granite arrow layout under output_root.
arrow::Status ConvertGraniteModel(
    const std::string &input_path,
    const std::string &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory and rewrites the "vram" block of the model.index.json under
// output_root/logestix.
arrow::Status RecomputeGraniteVram(
    const std::string &input_path,
    const std::string &output_root);

} // namespace granite
} // namespace lsxcommon
