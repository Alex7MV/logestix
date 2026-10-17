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

/// Request-side message (content flattened to a string).
struct ChatToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;  // JSON string
};

struct ChatInputMessage {
    std::string role;
    std::string content;
    std::string tool_call_id;
    std::vector<ChatToolCall> tool_calls;
};

struct ChatToolDef {
    std::string name;
    std::string description;
    std::string parameters_json;
};

struct ChatRequest {
    std::string model;
    std::vector<ChatInputMessage> messages;
    std::vector<ChatToolDef> tools;
    std::string tool_choice;
    bool stream = false;
    int max_tokens = 0;
    std::vector<std::string> stop;
    bool include_usage = false;
};

struct LegacyCompletionRequest {
    std::string model;
    std::string prompt;
    bool stream = false;
    int max_tokens = 0;
    std::vector<std::string> stop;
    bool include_usage = false;
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

    /// Parses a `POST /v1/chat/completions` request body. String members are
    /// owned. Unknown fields (tools, ...) are ignored.
    ///
    /// SAMPLING IS NOT IMPLEMENTED, and this is not an accident — read the field
    /// as "accepted and served greedily", not as "understood":
    ///   * `temperature`, `top_p`, `top_k`, `n`, `seed`, `presence_penalty`,
    ///     `frequency_penalty` and `logprobs` are parsed as unknown and
    ///     DISCARDED. Every family decodes argmax (`sample::GreedySampler`), so
    ///     a request carrying `"temperature": 0.7` is served exactly as if the
    ///     field were absent.
    ///   * This is stable and deliberate, not a silent partial rollout: the
    ///     engine is a bit-exactness/parity machine (see OPEN_TASKS §E4), where
    ///     a sampling kernel would make generations non-reproducible and would
    ///     invalidate the DET/FSI/CODE goldens captured from the source
    ///     checkpoints. Greedy is the numeric class those gates pin.
    ///   * To honour sampling, follow the plan in OPEN_TASKS §E4: plumb the
    ///     fields into `ModelRequest` and `sample::ISampler` (a real sampling
    ///     CUDA kernel, not just a scalar), through the v2 fabric wire frame.
    simdjson::error_code ParseChatRequest(const char* data, std::size_t size,
                                          std::size_t capacity,
                                          ChatRequest& out) const;

    /// Parses a legacy `POST /v1/completions` request body.
    simdjson::error_code ParseLegacyCompletionRequest(
        const char* data, std::size_t size, std::size_t capacity,
        LegacyCompletionRequest& out) const;
};

/// Minimal JSON string escaping for response bodies.

/// Frames a payload as an SSE event (`data: <payload>\n\n`).
std::string SseFrame(const std::string& data);

std::string BuildChatCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens);
std::string BuildChatChunk(const std::string& id, const std::string& model,
                           const std::string& delta_content, bool first_delta,
                           const std::string& finish_reason, int prompt_tokens,
                           int completion_tokens, bool include_usage);

/// Renders tool calls as a compact OpenAI `tool_calls` JSON array (each
/// `function.arguments` is a JSON string).
std::string ToolCallsToJson(const std::vector<ChatToolCall>& calls);

/// Completion whose assistant message carries `tool_calls` and
/// `finish_reason:"tool_calls"`.
std::string BuildChatCompletionTools(const std::string& id,
                                     const std::string& model,
                                     const std::string& tool_calls_json,
                                     int prompt_tokens, int completion_tokens);

/// Streaming chunk carrying a `tool_calls` delta.
std::string BuildChatChunkTools(const std::string& id, const std::string& model,
                                const std::string& tool_calls_json,
                                bool first_delta, int prompt_tokens,
                                int completion_tokens, bool include_usage);
std::string BuildTextCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens);
std::string BuildTextChunk(const std::string& id, const std::string& model,
                           const std::string& delta_text,
                           const std::string& finish_reason);
std::string BuildModelsList(const std::vector<std::string>& ids);
std::string BuildModel(const std::string& id);
std::string BuildError(const std::string& message, const std::string& type);

}  // namespace lsxnetwork