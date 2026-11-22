#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "lsxcommon/vram_estimate.h"

#include "deepseek_moe_v41_constants.h"

namespace lsxcommon {
namespace deepseek_v41 {

// Sole supported index format for this contour (anti-cross-load gate).
inline constexpr char kIndexFormat[] = "logestix-v41-nvfp4";

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    lsxcommon::VramProfile vram;
    std::string format;
    std::string model;
    std::array<LayerMeta, kLayers> layers{};
    std::string error_msg;
};

// Reads <model_path>/logestix/model.index.json (vault manifest fallback,
// same policy as every other family). Fills error_msg on any failure and
// rejects an index whose `format` is not kIndexFormat.
ModelTopology LoadTopology(const std::string &model_path);

// Frozen-passport cross-check: layer/expert counts must equal the compile-time
// constants. @p error receives a human-readable reason on mismatch.
bool ValidateGeometry(const ModelTopology &topo, std::string *error);

// The fold `gpu::detect_model_type` applies to a card name before the factory
// lookup: spaces become '-', A-Z becomes a-z (dots and digits are kept).
// "DeepSeek-V4.1-Flash-1017" -> "deepseek-v4.1-flash-1017".
std::string NormalizeCardName(std::string_view card);

// True when the converted tree names @p model_type, i.e. when path
// auto-detection would resolve THIS tree to that registry id. One architecture
// serves two checkpoints (V4.1-Flash and its 1017 update), so the index's
// `model` field is the only thing that tells the trees apart and a tree that
// names a sibling must not load here. An index that names no model is accepted;
// the format and frozen-geometry gates still apply.
bool MatchesModelType(const ModelTopology &topo, std::string_view model_type);

inline bool IsSupportedFormat(const ModelTopology &topo)
{
    return topo.format == kIndexFormat;
}

} // namespace deepseek_v41
} // namespace lsxcommon
