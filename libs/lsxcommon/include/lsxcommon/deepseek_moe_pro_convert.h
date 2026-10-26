#pragma once

// DeepSeek-V4-Pro-0813 ingestion (Model Ingestion Protocol stage 1).
//
// Source: HF safetensors (DeepseekV4ForCausalLM, FP8 dense + fp4-packed
// routed experts) -> <output>/logestix (logestix-v1-nvfp4).
//
// Differences from the generic DeepSeek converter:
//   * frozen V4-Pro passport gate (61 layers / 7168 hidden / 129280 vocab /
//     384 routed experts) unless LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1,
//   * the DSpark/MTP speculative stage (`mtp.0..2.*`) is stripped and COUNTED
//     in model.index.json's "deferred" block,
//   * the indexed model name is pinned to "DeepSeek-V4-Pro-0813" so a renamed
//     download still resolves to the registered kType.
#include <arrow/api.h>

#include <string>

namespace lsxcommon {
namespace deepseek_pro {

arrow::Status ConvertDeepSeekProModel(const std::string &input_path,
                                      const std::string &output_path);

} // namespace deepseek_pro
} // namespace lsxcommon
