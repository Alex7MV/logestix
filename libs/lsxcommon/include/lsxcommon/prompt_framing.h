#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace prompt {

// Ordered chat-template sentinel descriptor for one model. Sentinels are
// resolved via arrow_tokenizer_lookup at encode time; a model with no entry
// (or an incomplete sentinel set) encodes raw.
struct FramingDescriptor {
    std::string_view model_type;
    std::string_view leading[3];
    std::string_view trailing[3];
    std::size_t leading_count = 0;
    std::size_t trailing_count = 0;
};

// Compiled static registry keyed by model_type. GLM-5.2 is the only entry
// today. Returns nullptr for models without framing.
const FramingDescriptor *LookupFraming(std::string_view model_type);

// Frames a raw generation prompt into model-specific input token ids.
// For model_type == "glm-5.2" the GLM chat-template framing for the
// non-thinking generation mode is applied:
//   [gMASK]<sop><|user|>{prompt}<|assistant|> thinking response
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