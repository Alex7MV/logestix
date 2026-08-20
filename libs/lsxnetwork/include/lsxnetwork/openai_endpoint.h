#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
    int max_tokens = 0;         // "max_tokens" (0 = model default 64)
};

/// Network-layer slice of lsxcommon::gpu::InferenceResult. lsxnetwork cannot
/// depend on lsxcommon, so the orchestrator maps InferenceResult -> this and
/// hands it to OpenAiEndpoint::WriteCompletion.
struct CompletionResponse {
    bool ok = false;
    std::string output_text;
    std::vector<int32_t> gen_tokens;
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double tokens_per_sec = 0.0;
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

    /// Serializes an OpenAI-style chat-completion reply. Echoes @p req's `id`
    /// and `model` so the response ring can correlate without a per-slot
    /// channel. Content is JSON-escaped (quotes/backslash/control chars).
    /// Returns false only on OOM (string growth) — never on valid input.
    bool WriteCompletion(const ChatCompletion& req,
                         const CompletionResponse& res,
                         std::string& out) const;
};

}  // namespace lsxnetwork