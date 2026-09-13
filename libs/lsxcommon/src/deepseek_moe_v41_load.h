#pragma once

#include <array>
#include <cstdint>
#include <string>

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

inline bool IsSupportedFormat(const ModelTopology &topo)
{
    return topo.format == kIndexFormat;
}

} // namespace deepseek_v41
} // namespace lsxcommon
