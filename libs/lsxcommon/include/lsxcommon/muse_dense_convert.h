#pragma once

// Muse-Glimmer-30B (dense `muse` family) converter — Model Ingestion Protocol
// steps 1/2.
//
// `MuseGlimmerForConditionalGeneration`: a 52-layer dense decoder with gated
// GQA attention (NoPE full layers), centered norms, SwiGLU MLP and softcapped
// logits, plus a vision tower that is ballast for this text-only runtime.
// It owns its converter and format tag; it reuses no other family's topology
// or constants (model-independence invariant).

#include <cstdint>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "lsxcommon/muse_dense_constants.h"

namespace lsxcommon {
namespace muse_dense {

enum class TensorKind {
    Unknown,
    Visual,       // vision tower / adapter / projection: counted, never applied
    Embed,        // model.language_model.embed_tokens.weight (FP16, normed)
    LmHead,       // lm_head.weight (FP16, untied)
    FinalNorm,    // model.language_model.norm.weight (plain x*w)
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    AttnGate,     // self_attn.gate_proj — the attention OUTPUT gate
    MlpGate,
    MlpUp,
    MlpDown,
    InputNorm,    // centered, rms_norm_eps
    PostAttnNorm, // centered, post_norm_eps
    PreFfnNorm,   // centered, rms_norm_eps
    PostFfnNorm,  // centered, post_norm_eps
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies one safetensors tensor name. An unrecognised name is TensorKind::
// Unknown on purpose: the pipeline then errors instead of silently dropping it.
ParsedTensor classify_muse_dense_tensor(const std::string &name);

// True for the tensors emitted as NVFP4 (packed nibbles + E4M3 block scales +
// an F32 global scale): the eight linear projections of a layer. embed/lm_head
// stay FP16 (untied head, exact greedy); norms are never quantized.
bool muse_dense_tensor_is_nvfp4(TensorKind kind);

// Parses text_config out of <config_path>. config.json is mandatory; a missing
// or malformed file lands in error_msg. The sibling generation_config.json is
// merged for the full eos set and the pad id when present.
MuseDenseTopology load_muse_dense_topology(const std::string &config_path);

} // namespace muse_dense
} // namespace lsxcommon