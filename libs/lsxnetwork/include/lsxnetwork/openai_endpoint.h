#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <simdjson.h>

namespace lsxnetwork {

/// Zero-alloc extraction result for an OpenAI `/v1/chat/completions` document.
///
/// Every string is a `string_view` into the simdjson-backed input buffer and
/// every token counter is written into caller-owned storage — no heap
/// allocations, no `std::string` instantiations on the parse path. (An on-demand
/// parser instance is created per call; the payload bytes are never copied.)
struct ChatCompletion {
    std::string_view id;
    std::string_view object;
    std::string_view role;
    std::string_view content;
    std::string_view finish_reason;
    std::int64_t prompt_tokens     = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens      = 0;
};

/// Maps raw incoming OpenAI chat-completion JSON onto a `ChatCompletion`
/// without allocating.
class OpenAiEndpoint {
public:
    /// @param data     buffer containing the JSON document.
    /// @param size     document length in bytes.
    /// @param capacity total readable bytes from @p data; must be >= size +
    ///                 FrameQueue::kSimdjsonPadding so simdjson can zero-copy
    ///                 read its trailing SIMD padding without re-allocating.
    /// @param out      receives the extracted fields.
    /// @return split-free simdjson error_code; simdjson::SUCCESS on success.
    simdjson::error_code Parse(const char* data, std::size_t size,
                               std::size_t capacity, ChatCompletion& out) const;
};

}  // namespace lsxnetwork