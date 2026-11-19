#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace lsxcommon {
namespace atem {

// ATEM channel framing shared by the parity gates of a reasoning checkpoint
// (Muse-Glimmer): FSI, CODE and DET.
//
// Such a checkpoint marks channels with special tokens — e.g.
// `<|start|>assistant to=user<|message|>...<|eot|>` — so the decoder
// (`runtime::DecodeGeneratedText`, skip-special) drops the markers and the
// channel HEADERS survive as literal text. The model answers inside the answer
// channel, then loops back into the reasoning channel and re-emits the answer;
// the LAST answer channel can be cut mid-answer by the token budget. The answer
// is therefore the FIRST COMPLETE answer channel, i.e. one that is followed by
// another channel header. When no channel is complete (the budget ran out
// inside the only one), that channel — never the reasoning text before it — is
// the answer.
//
// `tools/muse_golden_oracle.py::select_answer_channel` is the capture-side twin
// of this rule: the golden's `expected` must be the answer this function
// extracts from the same text. The C++ side is pinned by
// tests/test_fsi_selfcheck.cpp, the Python side by
// `tools/fsi_capture.py --selftest`.
inline std::string SelectAnswerChannel(std::string_view text,
                                       std::string_view answer_channel,
                                       std::string_view reasoning_channel)
{
    if (answer_channel.empty()) return std::string(text);
    std::size_t pos = 0;
    std::string first_complete;
    while (pos < text.size()) {
        const std::size_t a = text.find(answer_channel, pos);
        if (a == std::string_view::npos) break;
        const std::size_t body = a + answer_channel.size();
        std::size_t end = text.size();
        const std::size_t next_a = text.find(answer_channel, body);
        if (next_a != std::string_view::npos) end = std::min(end, next_a);
        if (!reasoning_channel.empty()) {
            const std::size_t next_r = text.find(reasoning_channel, body);
            if (next_r != std::string_view::npos) end = std::min(end, next_r);
        }
        if (end < text.size() && first_complete.empty())
            first_complete = std::string(text.substr(body, end - body));
        pos = end;
    }
    if (!first_complete.empty()) return first_complete;
    const std::size_t last = text.rfind(answer_channel);
    if (last != std::string_view::npos)
        return std::string(text.substr(last + answer_channel.size()));
    return std::string(text);
}

}  // namespace atem
}  // namespace lsxcommon
