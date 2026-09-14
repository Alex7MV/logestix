// deepseek_moe_v41_convert.h — DeepSeek-V4.1-Flash ingestion converter
// (autonomous contour). Produces the `logestix-v41-nvfp4` layout under
// <output>/logestix/:
//   model.index.json, pinned/global.arrow, pinned/layer_NNN.arrow,
//   experts/layer_NNN/exp_E.arrow
// Dense FP8 (32x32 E8M0 tiles) and norms/biases pass through at source
// precision; routed experts (I8 E2M1 + E8M0/32) are repacked to the neutral
// NVFP4 contour (pow2-E4M3 per-16 + per-tensor global_scale). Vision/aligner,
// engram and DSpark MTP tensors are excluded with a counted record.
#pragma once

#include <string>

#include <arrow/status.h>

namespace lsxcommon {
namespace deepseek_v41 {

arrow::Status ConvertDeepSeekV41Model(const std::string &input_path,
                                      const std::string &output_path);

} // namespace deepseek_v41
} // namespace lsxcommon
