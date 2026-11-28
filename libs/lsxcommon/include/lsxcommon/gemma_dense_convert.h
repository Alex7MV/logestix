#pragma once

// Gemma-4-31B-it (DENSE family) converter — Model Ingestion Protocol step 1/2.
//
// The dense member of the Gemma-4 family: same decoder as gemma-4-26B-A4B-it
// (5:1 sliding/global attention, per-head QK-norm with V tied to K on the
// global layers, four norms per layer, GeGLU MLP, tied embed/lm_head, softcapped
// logits) but WITHOUT routed experts. It therefore owns its converter and its
// format tag; it never reuses the MoE converter's topology or constants.

#include <cstdint>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "lsxcommon/gemma_dense_constants.h"

namespace lsxcommon {
namespace gemma_dense {

// Format tag written to <root>/logestix/model.index.json.
inline constexpr const char *kFormatTag = "logestix-gemma-dense";

enum class TensorKind {
    Unknown,
    Visual,     // multimodal ballast: skipped (and counted), never applied
    Embed,      // tied embed_tokens / lm_head table
    FinalNorm,
    AttnQ,
    AttnK,
    AttnV,      // absent on full-attention layers (attention_k_eq_v)
    AttnO,
    AttnQNorm,
    AttnKNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    InputNorm,
    PostNorm,
    PreFFN,
    PostFFN,
    LayerScalar,
};

struct ParsedTensor {
    TensorKind kind = TensorKind::Unknown;
    int layer = -1;
};

// Classifies one safetensors tensor name. An unrecognised name is TensorKind::
// Unknown on purpose: the pipeline then errors instead of silently dropping it.
ParsedTensor classify_gemma_dense_tensor(const std::string &name);

// True for the tensors emitted as NVFP4 (packed nibbles + E4M3 block scales +
// an F32 global scale). Norms, the per-layer scalar and the vision ballast stay
// high precision.
bool gemma_dense_tensor_is_nvfp4(TensorKind kind);

// Conversion knobs that differ BETWEEN dense Gemma-4 sizes. They are a struct
// rather than a compile-time constant because ONE converter serves every member
// and the right answer is a property of the checkpoint, not of the code:
//
//   gemma-4-12b-it   residual_plane = true    12.0 GB, fits a 32 GB card
//   gemma-4-31b-it   residual_plane = false   4-bit keeps 16.5 GiB loadable
//                                             (with the plane it is 33.7 GiB)
//
// The default is `false` — 4-bit only — deliberately: forgetting to declare the
// plane costs precision, which the FSI parity gate reports, and can never
// produce a tree that does not fit the card. The opposite default would fail
// silently in the expensive direction.
struct GemmaDenseConvertOptions {
    // Second NVFP4 plane per projection: the residual `w - dequant(base)`,
    // decoded as one more GEMV with acc=true (~8 effective bits for 2x the
    // projection bytes). Measured on the 12B: host relL2 0.0926 -> 0.0049 and
    // FSI 1/3 -> 2/3.
    bool residual_plane = false;
};

// Parses text_config out of <config_path>. config.json is mandatory; a missing
// or malformed file lands in error_msg. The sibling generation_config.json is
// merged for the full eos set when present.
GemmaDenseTopology load_gemma_dense_topology(const std::string &config_path);

// Writes <output_root>/model.index.json (the geometry the loader and the runtime
// gates read back). @p residual_plane is RECORDED, not assumed: an index that
// claims a plane the tree does not hold (or hides one it does) is exactly the
// kind of disagreement the VRAM estimate and a later re-conversion act on.
arrow::Status WriteGemmaDenseModelIndex(const std::string &output_root,
                                       const GemmaDenseTopology &top,
                                       bool residual_plane);

// Full conversion pipeline: safetensors -> pinned layout
//   pinned/global.arrow         raw BF16 (final norm)
//   pinned/global_nvfp4.arrow   the tied NVFP4 embed/lm_head table
//   pinned/layer_N.arrow        NVFP4 projection triples + raw norms/scalar
//   pinned/tokenizer*           stage-4 token bridge
//   logestix/model.index.json   structural manifest (vision ballast counted)
// @p opts carries the per-size conversion choices (see
// GemmaDenseConvertOptions above); the caller takes them from the model's
// manifest row, so a re-conversion of either sibling produces the tree that row
// describes.
arrow::Status ConvertGemmaDenseModel(
    const std::string &input_path, const std::string &output_root,
    const GemmaDenseConvertOptions &opts = GemmaDenseConvertOptions{});

} // namespace gemma_dense
} // namespace lsxcommon
