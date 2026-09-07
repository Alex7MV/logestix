#pragma once

#include <arrow/api.h>
#include <string>

namespace lsxcommon::qwen_moe_36 {

// Converts a Qwen3.6-35B-A3B model directory (safetensors shards +
// config.json) into the unified <output_root>/logestix arrow layout.
// Shares the conversion body with Qwen3.5-35B-A3B
// (ConvertQwenMoeModelCommon) and carries its own frozen-geometry passport
// (qwen_moe_36_constants.h) plus index metadata (logestix-qwen36moe).
arrow::Status ConvertQwen36MoeModel(const std::string &input_dir,
                                    const std::string &output_root);

} // namespace lsxcommon::qwen_moe_36
