#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <simdjson.h>

namespace lsxnetwork {

/// Extraction result for an OpenAI `/v1/chat/completions` document.
///
/// ZERO-COPY at the queue boundary: the document bytes are never copied into a
/// middleman buffer — simdjson reads them straight out of the FrameQueue chunk.
/// String members are owned (`std::string`): on-demand `get_string()` returns
/// views that are invalidated when the parser dies (simdjson docs), so the
/// parser is destroyed inside Parse() and the values must be copied out while
/// it is alive. Allocation happens only for frames that actually carry a
/// string field, never for the ring handoff itself.
struct ChatCompletion {
    std::string id;
    std::string object;
    std::string model;          // top-level "model" (dispatch target)
    std::string role;
    std::string content;
    std::string finish_reason;
    std::int64_t prompt_tokens     = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens      = 0;
};

/// Maps raw incoming OpenAI chat-completion JSON onto a `ChatCompletion`.
///
/// Zero-copy at the queue boundary (no middleman buffer; simdjson reads the
/// FrameQueue chunk directly), but string members are heap-owned and allocated
/// per frame with a string field — not zero-allocating.
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