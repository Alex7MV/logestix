#include "lsxcommon/prompt_framing.h"

#include <vector>

namespace lsxcommon {
namespace prompt {
namespace {

constexpr size_t kMaxSentinels = 16;

bool IsBlank(std::string_view s)
{
    for (char c : s)
        if (static_cast<unsigned char>(c) > 0x20) return false;
    return true;
}

// Encodes the raw prompt into `body`. Returns the element count (>0) or 0 on
// encoder rejection. Capacity is grown geometrically like every other caller
// of arrow_tokenizer_encode.
std::vector<int32_t> EncodeBody(const ArrowTokenizerDb &tok,
                                std::string_view raw_prompt)
{
    for (size_t cap = 256; cap <= 8192; cap *= 2) {
        std::vector<int32_t> ids(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(
            tok, raw_prompt.data(), raw_prompt.size(), ids.data(), &enc_cap);
        if (n == -2) continue;
        if (n <= 0) return {};
        ids.resize(static_cast<size_t>(n));
        return ids;
    }
    return {};
}

} // namespace

bool EncodeRawPrompt(const ArrowTokenizerDb &tok, std::string_view raw_prompt,
                     std::vector<int32_t> &out_ids)
{
    if (IsBlank(raw_prompt)) return false;
    out_ids = EncodeBody(tok, raw_prompt);
    return !out_ids.empty();
}

bool ApplyFraming(const FramingDescriptor &frame, const ArrowTokenizerDb &tok,
                  std::string_view raw_prompt, std::vector<int32_t> &out_ids)
{
    if (IsBlank(raw_prompt)) return false;
    if (frame.leading_count > kMaxSentinels ||
        frame.trailing_count > kMaxSentinels)
        return false;

    int32_t sentinel[2 * kMaxSentinels];
    bool complete = true;
    for (size_t i = 0; i < frame.leading_count && complete; ++i) {
        sentinel[i] = arrow_tokenizer_lookup(
            tok, frame.leading[i].data(), frame.leading[i].size());
        if (sentinel[i] < 0) complete = false;
    }
    for (size_t i = 0; i < frame.trailing_count && complete; ++i) {
        sentinel[frame.leading_count + i] = arrow_tokenizer_lookup(
            tok, frame.trailing[i].data(), frame.trailing[i].size());
        if (sentinel[frame.leading_count + i] < 0) complete = false;
    }
    if (!complete) return EncodeRawPrompt(tok, raw_prompt, out_ids);

    const std::vector<int32_t> body = EncodeBody(tok, raw_prompt);
    if (body.empty()) return false;

    // Model-dependent BOS policy: relocate the encoder's auto-prepended BOS
    // from the body to position 0 when the template is BOS-led.
    size_t body_off =
        (frame.leading_bos && tok.bos_id >= 0 && !body.empty() &&
         body.front() == tok.bos_id)
            ? 1
            : 0;

    std::vector<int32_t> framed;
    framed.reserve(1 + frame.leading_count + (body.size() - body_off) +
                   frame.trailing_count);
    if (frame.leading_bos && tok.bos_id >= 0) framed.push_back(tok.bos_id);
    for (size_t i = 0; i < frame.leading_count; ++i)
        framed.push_back(sentinel[i]);
    for (size_t i = body_off; i < body.size(); ++i) framed.push_back(body[i]);
    for (size_t i = 0; i < frame.trailing_count; ++i)
        framed.push_back(sentinel[frame.leading_count + i]);
    out_ids = std::move(framed);
    return true;
}

} // namespace prompt

} // namespace lsxcommon
