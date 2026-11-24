#pragma once

#include <arrow/status.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Hy4-flash (tencent/Hy4-flash) ingestion — stage 1/2 of the Model Ingestion
// Protocol: read the vendor safetensors tree, classify every tensor, quantize
// the linear matrices to NVFP4 and write the logestix artifact layout.
//
// Tensors deliberately NOT written (counted in the index `deferred` block, never
// silently dropped):
//   * Engram n-gram tables (`model.layers.{1,11,21,31}.engram_layer.*`,
//     ~126.7 GB BF16) — the checkpoint publishes the tensors and the
//     hyperparameters but no code, and the hash semantics exist in no public
//     implementation. Applying a guess would produce "answers, but not the
//     model's answers".
//   * The ViT tower and its projector (`vit.*`).
//   * The 5-layer DFly2 speculative drafter (`model.dfly_layers.*`).
namespace lsxcommon {
namespace hy4 {

// One bucket per source tensor. `kUnknown` is a hard conversion error: a tensor
// this classifier does not recognise is a converter that silently ships a
// truncated model, which is how a family ends up decoding garbage.
enum class TensorKind {
    kGlobalRaw,      // embed / lm_head / final norm / hc_head (source precision)
    kLayerRaw,       // norms, iHC params, sinks, router gate+bias, indexer F32
    kDenseNvfp4,     // 2-D linear projection -> NVFP4
    kExpertGateUp,   // 3-D [E, 2*I, latent] routed gate_up
    kExpertDown,     // 3-D [E, latent, I] routed down
    kCutVision,
    kCutDfly,
    kCutEngram,
    kUnknown,
};

struct ParsedTensor {
    TensorKind kind = TensorKind::kUnknown;
    int layer = -1;
};

// Name -> bucket. Layer index is -1 for globals and for cuts that are not
// layer-scoped (the ViT tower, the DFly2 drafter).
ParsedTensor ParseHy4Tensor(std::string_view name);

// True when the kind is a dense matrix that must be quantized.
inline bool IsDenseNvfp4(TensorKind k) { return k == TensorKind::kDenseNvfp4; }
inline bool IsRoutedExpert(TensorKind k)
{
    return k == TensorKind::kExpertGateUp || k == TensorKind::kExpertDown;
}
inline bool IsCut(TensorKind k)
{
    return k == TensorKind::kCutVision || k == TensorKind::kCutDfly ||
           k == TensorKind::kCutEngram;
}

// Per-layer classification the index records (and the loader cross-checks).
struct LayerShape {
    bool is_dsa = false;     // sparse IndexPool attention
    bool is_moe = false;     // LatentMoE instead of the dense FFN
    bool has_engram = false; // engram tables were cut from this layer
    int heads = 0;
};

struct DeferredCounts {
    int64_t engram = 0;
    int64_t vision = 0;
    int64_t dfly = 0;
};

// The converter. `input_path` is the vendor tree, `output_path` the tree the
// artifact lands under (`<output_path>/logestix/...`).
arrow::Status ConvertHy4FlashModel(const std::string &input_path,
                                   const std::string &output_path);

// Reads config.json and derives the per-layer shape table without touching any
// weights. Exposed for the classifier test and the loader's cross-check.
arrow::Status ReadHy4LayerShapes(const std::string &input_path,
                                 std::vector<LayerShape> &layers,
                                 std::string &why);

} // namespace hy4
} // namespace lsxcommon
