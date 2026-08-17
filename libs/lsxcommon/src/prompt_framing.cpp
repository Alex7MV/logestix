#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {
namespace prompt {
namespace {

constexpr std::string_view kGlmType = "glm-5.2";

struct GlmSpecials {
    int32_t gmask = -1, sop = -1, user = -1, asst = -1, think = -1, endthink = -1;
    bool complete() const
    {
        return gmask >= 0 && sop >= 0 && user >= 0 && asst >= 0 &&
               think >= 0 && endthink >= 0;
    }
};

GlmSpecials LookupGlmSpecials(const ArrowTokenizerDb &tok)
{
    GlmSpecials s;
    s.gmask    = arrow_tokenizer_lookup(tok, "[gMASK]", 7);
    s.sop      = arrow_tokenizer_lookup(tok, "<sop>", 5);
    s.user     = arrow_tokenizer_lookup(tok, "<|user|>", 8);
    s.asst     = arrow_tokenizer_lookup(tok, "<|assistant|>", 13);
    s.think    = arrow_tokenizer_lookup(tok, "<think>", 7);
    s.endthink = arrow_tokenizer_lookup(tok, "</think>", 8);
    return s;
}

} // namespace

bool BuildPromptInputIds(std::string_view model_type,
                         const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    const bool glm = (model_type == kGlmType);
    const GlmSpecials specials = glm ? LookupGlmSpecials(tok) : GlmSpecials{};
    const bool frame = glm && specials.complete();

    // Reject empty/blank prompts up front: the raw BPE encode would otherwise
    // emit a bare BOS for tokenizers that define one (the synthetic fixtures
    // do), producing a "success" for no prompt at all.
    bool blank = true;
    for (char c : raw_prompt) {
        if (static_cast<unsigned char>(c) > 0x20) {
            blank = false;
            break;
        }
    }
    if (blank) return false;

    // Capacity-grow loop: 256 -> 8192 ids. GLM framing reserves 6 leading
    // ids for the chat-template sentinels; encode may return -2 (capacity too
    // small), in which case the buffer is doubled and the prompt re-encoded.
    for (size_t cap = 256; cap <= 8192; cap *= 2) {
        std::vector<int32_t> ids(cap, 0);
        const int32_t head = frame ? 3 : 0;
        size_t enc_cap = cap - (frame ? 6u : 0u);
        const int32_t n = arrow_tokenizer_encode(
            tok, raw_prompt.data(), raw_prompt.size(), ids.data() + head,
            &enc_cap);
        if (n == -2) continue;
        if (n <= 0) return false;   // empty or rejected prompt
        if (frame) {
            ids[0] = specials.gmask;
            ids[1] = specials.sop;
            ids[2] = specials.user;
            ids[3 + static_cast<size_t>(n)] = specials.asst;
            ids[4 + static_cast<size_t>(n)] = specials.think;
            ids[5 + static_cast<size_t>(n)] = specials.endthink;
        }
        ids.resize(static_cast<size_t>(n) + (frame ? 6u : 0u));
        out_ids = std::move(ids);
        return true;
    }
    return false;
}

} // namespace prompt
} // namespace lsxcommon