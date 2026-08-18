#include "lsxcommon/prompt_framing.h"

#include <array>

namespace lsxcommon {
namespace prompt {
namespace {

constexpr FramingDescriptor kGlmFraming = {
    "glm-5.2",
    {"[gMASK]", "<sop>", "<|user|>"},
    {"<|assistant|>", " thinking", " response"},
    3, 3,
};

constexpr std::array<FramingDescriptor, 1> kFramings = {kGlmFraming};

bool IsBlank(std::string_view s)
{
    for (char c : s)
        if (static_cast<unsigned char>(c) > 0x20) return false;
    return true;
}

} // namespace

const FramingDescriptor *LookupFraming(std::string_view model_type)
{
    for (const auto &f : kFramings)
        if (f.model_type == model_type) return &f;
    return nullptr;
}

bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    const FramingDescriptor *frame = LookupFraming(model_type);
    int32_t sentinel[6];
    bool complete = frame != nullptr;
    if (frame) {
        for (size_t i = 0; i < frame->leading_count; ++i)
            sentinel[i] = arrow_tokenizer_lookup(tok, frame->leading[i].data(),
                                                 frame->leading[i].size());
        for (size_t i = 0; i < frame->trailing_count; ++i)
            sentinel[frame->leading_count + i] =
                arrow_tokenizer_lookup(tok, frame->trailing[i].data(),
                                       frame->trailing[i].size());
        for (size_t i = 0; i < frame->leading_count + frame->trailing_count; ++i)
            if (sentinel[i] < 0) { complete = false; break; }
    }
    const bool use_frame = complete;

    if (IsBlank(raw_prompt)) return false;

    const size_t nlead = use_frame ? frame->leading_count : 0;
    const size_t ntrail = use_frame ? frame->trailing_count : 0;
    const size_t nspec = nlead + ntrail;

    for (size_t cap = 256; cap <= 8192; cap *= 2) {
        std::vector<int32_t> ids(cap, 0);
        size_t enc_cap = cap - nspec;
        const int32_t n = arrow_tokenizer_encode(
            tok, raw_prompt.data(), raw_prompt.size(), ids.data() + nlead,
            &enc_cap);
        if (n == -2) continue;
        if (n <= 0) return false;
        if (use_frame) {
            for (size_t i = 0; i < nlead; ++i) ids[i] = sentinel[i];
            for (size_t i = 0; i < ntrail; ++i)
                ids[nlead + static_cast<size_t>(n) + i] =
                    sentinel[nlead + i];
        }
        ids.resize(static_cast<size_t>(n) + nspec);
        out_ids = std::move(ids);
        return true;
    }
    return false;
}

} // namespace prompt
} // namespace lsxcommon