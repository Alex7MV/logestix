# TaskMosaic OpenAI Bridge — Plan 4: Anthropic Messages Adapter

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline; AGENTS.md forbids subagents). Steps use checkbox (`- [ ]`).
>
> **Depends on:** Plans 1–3 on `feat/taskmosaic-openai-bridge`.

**Goal:** Serve the Anthropic Messages API so Claude Code can connect directly via `ANTHROPIC_BASE_URL`: `POST /v1/messages`, `/v1/messages/count_tokens`, `HEAD /api/hello`, Anthropic-shaped `GET /v1/models`, with the exact SSE event sequence.

**Architecture:** A host-only `anthropic_endpoint` module parses the Anthropic request into `lsxcommon::ChatConversation` (reusing the same model-side framing/tool machinery as OpenAI) and serializes Anthropic responses/events. The taskmosaic bridge adds Anthropic routes that reuse the existing backend/queue/cancel/stream plumbing; a dedicated `AnthropicStreamSource` emits the Anthropic SSE sequence.

**Tech Stack:** C++20, simdjson, catch2, cpp-httplib v0.58.0.

## Global Constraints

- Reuse `lsxcommon` framing/tools; no model-side changes.
- Anthropic `content` blocks: `text`, `tool_use`, `tool_result`; `system` is a string or block array.
- SSE must emit: `message_start`, `content_block_start`/`content_block_delta`/`content_block_stop`, `message_delta`, `message_stop`; `content-type: text/event-stream`.
- Errors: `{"type":"error","error":{"type","message"}}`; `retry-after` integer seconds on 429.
- spdlog only; labels per `test-lifecycle`; commit prefixes as before.

## File Structure

**Create**
- `libs/lsxnetwork/include/lsxnetwork/anthropic_endpoint.h` + `src/anthropic_endpoint.cpp`
- `tests/test_anthropic_endpoint.cpp`

**Modify**
- `libs/lsxnetwork/CMakeLists.txt` (add source)
- `apps/taskmosaic/src/chat_bridge.{h,cpp}` (routes + Anthropic stream source)
- `apps/taskmosaic/tests/test_taskmosaic_bridge.cpp` (Anthropic tests)
- `tests/CMakeLists.txt` (register the endpoint test)

---

### Task 1: Anthropic request parse + response/event builders

**Interfaces (`lsxnetwork`):**
```cpp
struct AnthropicRequest {
    std::string model;
    int max_tokens = 0;
    bool stream = false;
    std::string system;                    // flattened
    std::vector<ChatInputMessage> messages; // roles user/assistant/tool
    std::vector<ChatToolDef> tools;        // input_schema -> parameters_json
    std::string tool_choice;
};
simdjson::error_code ParseAnthropicRequest(const char*, size_t, size_t, AnthropicRequest&);

std::string AnthropicEvent(const std::string& event, const std::string& data); // "event: X\ndata: {...}\n\n"
std::string AnthropicMessageStart(const std::string& id, const std::string& model,
                                  const std::string& system, int input_tokens);
std::string AnthropicContentBlockStartText(int index);
std::string AnthropicContentBlockDeltaText(int index, const std::string& text);
std::string AnthropicContentBlockStop(int index);
std::string AnthropicToolUseBlockStart(int index, const std::string& id, const std::string& name);
std::string AnthropicToolUseDelta(int index, const std::string& partial_json);
std::string AnthropicMessageDelta(const std::string& stop_reason, int output_tokens);
std::string AnthropicMessageStop();
std::string BuildAnthropicMessage(const std::string& id, const std::string& model,
                                  const std::string& text, const std::string& stop_reason,
                                  int input_tokens, int output_tokens);
std::string BuildAnthropicToolMessage(const std::string& id, const std::string& model,
                                      const std::vector<ChatToolCall>& calls,
                                      const std::string& stop_reason,
                                      int input_tokens, int output_tokens);
std::string BuildAnthropicError(const std::string& message, const std::string& type);
std::string BuildAnthropicModelsList(const std::vector<std::string>& ids);
```

- [ ] **Step 1: Failing tests** — parse a request with `system` (string), a user
  text block, an assistant `tool_use` block and a user `tool_result` block, and
  `tools[].input_schema`; assert the flattened system, message roles/content, the
  tool_use→`ChatInputMessage::tool_calls` mapping, tool_result→`tool_call_id`, and
  that `input_schema` lands in `parameters_json`. Assert `AnthropicMessageStart`/
  `MessageDelta`/`MessageStop` framing and `AnthropicEvent` format.
- [ ] **Step 2: Run to fail**, then implement `anthropic_endpoint.cpp` with
  simdjson on-demand (mirror `openai_endpoint.cpp`; use `.raw_json()` for schemas).
- [ ] **Step 3: Run to pass**, register `test_anthropic_endpoint` (`fast`), commit
  `feat: Anthropic Messages codec`.

---

### Task 2: `/v1/messages` routes + Anthropic SSE

**Files:** `apps/taskmosaic/src/chat_bridge.*`; test bridge.

- [ ] **Step 1: Register** `HEAD /api/hello`, `GET /v1/models` (Anthropic shape),
  `POST /v1/messages/count_tokens`, `POST /v1/messages`.
  - `count_tokens`: parse, frame via `framer_.EncodeConversation`, return
    `{"input_tokens": N}`.
  - `messages` non-stream: run blocking; if `framer_.ParseToolCalls` finds calls →
    `BuildAnthropicToolMessage` (`stop_reason:"tool_use"`), else
    `BuildAnthropicMessage` (`stop_reason:"end_turn"`).
- [ ] **Step 2: Streaming** — `AnthropicStreamSource` emits `message_start`,
  then text deltas (`content_block_start`/`delta`/`stop`) live, or (when tools are
  requested) buffers and emits a `tool_use` block, then `message_delta` +
  `message_stop`. Client disconnect → backend `Cancel`.
- [ ] **Step 3: Tests** — non-stream message text; `count_tokens`; tool_use
  response; streaming event sequence contains `message_start` … `message_stop`.
- [ ] **Step 4: Commit** `feat: Anthropic /v1/messages adapter`.

---

### Task 3: Claude Code compatibility notes + docs

- [ ] **Step 1:** Document the exact Claude Code config:
  `ANTHROPIC_BASE_URL=http://<host>:<port>` (no `/v1`), `ANTHROPIC_AUTH_TOKEN`/
  `ANTHROPIC_API_KEY` arbitrary (not validated), model id from `/v1/models`.
- [ ] **Step 2:** Note `HEAD /api/hello`, `GET /v1/models?limit=1000`, and that
  `anthropic-version`/`anthropic-beta` are accepted and ignored.
- [ ] **Step 3:** commit `docs: document Claude Code / Anthropic connection`.

## Self-Review

| Spec §9 | Task |
|---|---|
| parse Anthropic request (system/blocks/tools) | 1 |
| messages/count_tokens/models/api_hello | 2 |
| SSE event sequence | 1, 2 |
| errors + retry-after | 1, 2 |
| Claude Code docs | 3 |

## Execution Handoff

Execute inline via superpowers:executing-plans. Host-only and testable end-to-end
with the fake backend (no GPU).
