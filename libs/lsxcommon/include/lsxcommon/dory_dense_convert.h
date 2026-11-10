#pragma once

#include <arrow/api.h>
#include <string>
#include <vector>

#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace dory {

// Structural classification of a Dory (`DoryForCausalLM`) tensor name. Every
// physical layer stores EITHER an attention mixer (q/k/v/o/g + sqk/s_gate +
// attn_alpha) OR an mlp mixer (gate/up/down + suv_gate/suv_up + mlp_alpha),
// plus the two global edges (embeddings / lm_head) and the lm_head per-vocab
// logit scale. A name outside this set is a converter error — the checkpoint
// has no ballast to skip.
enum class TensorKind {
    Unknown,
    Embed,       // backbone.embeddings.weight
    LmHead,      // lm_head.weight
    LogitScale,  // lm_head.logit_scale            (per-vocabulary scale)
    MixerQ,      // backbone.layers.N.mixer.q_proj.weight
    MixerK,      // backbone.layers.N.mixer.k_proj.weight
    MixerV,      // backbone.layers.N.mixer.v_proj.weight
    MixerO,      // backbone.layers.N.mixer.o_proj.weight
    MixerG,      // backbone.layers.N.mixer.g_proj.weight   (attention gate)
    MixerGate,   // backbone.layers.N.mixer.gate_proj.weight
    MixerUp,     // backbone.layers.N.mixer.up_proj.weight
    MixerDown,   // backbone.layers.N.mixer.down_proj.weight
    MixerSqk,    // backbone.layers.N.mixer.sqk      (nGPT q scale, [QSize])
    MixerSGate,  // backbone.layers.N.mixer.s_gate   (nGPT gate scale)
    MixerSuvGate,// backbone.layers.N.mixer.suv_gate (nGPT mlp gate scale)
    MixerSuvUp,  // backbone.layers.N.mixer.suv_up
    AttnAlpha,   // backbone.layers.N.attn_alpha     (nGPT residual, [hidden])
    MlpAlpha,    // backbone.layers.N.mlp_alpha
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies a Dory tensor name. Unknown kinds are REPORTED, never skipped:
// `ConvertDoryModel` fails on one (the checkpoint carries nothing to strip).
ParsedTensor classify_dory_tensor(const std::string &name);

// True when `kind` is quantized to NVFP4 by the converter (every projection,
// dense or not). The embeddings and the LM head stay FP16 (edges), and the
// nGPT per-dimension vectors / residual alphas / logit scale stay raw BF16:
// they are the normalization reference of the architecture, not linear
// weights, and 4-bit error there is amplified by the slerp and the sigmoid
// gate before it ever reaches a projection.
bool dory_tensor_is_nvfp4(TensorKind kind);

// True for a fused NVFP4 record name (the converter fuses q|k|v and gate|up so
// the decode issues one GEMV per mixer input).
bool dory_fused_kind(TensorKind kind);

DoryTopology load_dory_topology(const std::string &index_path);

arrow::Status WriteDoryModelIndex(const std::string &output_root,
                                  const DoryTopology &top);

// Converts a Dory model directory (safetensors shards + config.json) into the
// logestix-dory-nvfp4 arrow layout under <output_root>/logestix/.
arrow::Status ConvertDoryModel(const std::string &input_path,
                               const std::string &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory and rewrites the "vram" block of the model.index.json.
arrow::Status RecomputeDoryVram(const std::string &input_path,
                                const std::string &output_root);

} // namespace dory
} // namespace lsxcommon
