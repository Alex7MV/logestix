#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "lsxcommon/tokenizer.h"

namespace lsxcommon {
namespace prompt {

// Ordered chat-template sentinel descriptor for one model family.
//
// OWNERSHIP: each framed model owns its descriptor next to its model code
// (GlmV5Model -> glm::kFraming, GemmaBaseModel -> gemma::kFraming) and
// exposes it through IModel::BuildPromptInputIds. There is deliberately no
// central model_type -> framing registry: a new model cannot silently miss a
// framing entry, and BOS policy cannot leak across families.
//
// Sentinels are resolved via arrow_tokenizer_lookup at encode time; when any
// sentinel is missing from the tokenizer vocab the framing falls back to the
// raw BPE encode.
struct FramingDescriptor {
    std::string_view leading[16];
    std::string_view trailing[16];
    std::size_t leading_count = 0;
    std::size_t trailing_count = 0;
    // Model-dependent BOS policy (HF chat templates that start with
    // {{ bos_token }}): when true, the encoder's auto-prepended bos_id is
    // relocated out of the framed body to position 0. Models without a BOS
    // token keep false; their framed output then contains no BOS at all.
    bool leading_bos = false;
};

// Raw BPE encode path (the fallback for unframed models). The tokenizer
// auto-prepends bos_id when it defines one. Returns false on blank prompt or
// encoder failure.
bool EncodeRawPrompt(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                     std::vector<int32_t> &out_ids);

// Apply a model-owned framing descriptor:
//   [bos?][leading sentinels][encoded prompt][trailing sentinels]
// Falls back to EncodeRawPrompt when any sentinel is unresolvable.
bool ApplyFraming(const FramingDescriptor &frame, const ArrowTokenizerDb &tok,
                  std::string_view raw_prompt, std::vector<int32_t> &out_ids);

} // namespace prompt

namespace glm {

// GLM-5.2 chat framing, owned by GlmV5Model:
//   [gMASK]<sop><|user|>{prompt}<|assistant|><think></think>
extern const prompt::FramingDescriptor kFraming;
bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids);

} // namespace glm

namespace gemma {

// Gemma-4 canonical chat framing, owned by GemmaBaseModel (BOS-led):
//   <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
extern const prompt::FramingDescriptor kFraming;
bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids);

} // namespace gemma

} // namespace lsxcommon
