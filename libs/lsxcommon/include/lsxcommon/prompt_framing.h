#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace prompt {

// Frames a raw generation prompt into model-specific input token ids.
// For model_type == "glm-5.2" the GLM chat-template framing for the
// non-thinking generation mode is applied:
//   [gMASK]<sop><|user|>{prompt}<|assistant|><think></think>
// (falls back to the raw BPE encode when any of those special tokens is
// missing from the tokenizer vocab). Any other model type is encoded raw.
// Returns true and fills out_ids on success; false when the tokenizer
// rejects the prompt (or it is empty/blank).
bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids);

} // namespace prompt
} // namespace lsxcommon