# TaskMosaic OpenAI Bridge — Plan 3: Tool / Function Calling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline; AGENTS.md forbids subagents). Steps use checkbox (`- [ ]`).
>
> **Depends on:** Plan 1 (bridge) and Plan 2 (streaming) on `feat/taskmosaic-openai-bridge`.

**Goal:** Full OpenAI tool/function calling end-to-end: parse `tools`/`tool_choice`/assistant `tool_calls`/`role:"tool"`, render the conversation + tool schemas into model input ids using each family's native format (generic Hermes fallback), parse generated tokens back into `tool_calls`, and return them in both non-streaming and streaming responses with `finish_reason:"tool_calls"`.

**Architecture:** Host-only conversation/tool types in `lsxcommon/chat_conversation.h`. Two `IModel` virtuals with defaults: `BuildConversationInputIds` (native tool render; default = flatten + existing single-string framing) and `ParseToolCalls` (native parser; default = generic Hermes). `taskmosaic` builds a `ChatConversation` from the OpenAI request, calls the framer, and maps parsed calls back to OpenAI `tool_calls`. When tools are present the bridge uses buffered streaming (no live text deltas) to avoid leaking tool-call markup.

**Tech Stack:** C++20, simdjson, catch2; host-only additions (no `.cu` math changes).

## Global Constraints

- `.cu`/`.cuh` reachable headers must not include `<absl/container/*>`. `chat_conversation.h` uses only `<string>/<vector>`.
- No numeric change to generation; tool rendering only changes **input ids** and adds an output parser. All SelfCheck/FSI/CODE/DET/bit-exact gates stay green.
- Parse tool calls from **token ids** where the native format uses special tokens (DeepSeek/GLM/Kimi); `arrow_tokenizer_decode` drops specials. Generic Hermes parses decoded text.
- Per-family ownership: no central model-type→tool-format registry; each family overrides the virtuals next to its model code.
- spdlog only; labels per `test-lifecycle`; commit prefixes as before.

## File Structure

**Create**
- `libs/lsxcommon/include/lsxcommon/chat_conversation.h` (`ChatMessage`/`ToolCall`/`ToolSpec`/`ChatConversation`)
- `libs/lsxcommon/include/lsxcommon/tool_format.h` + `src/tool_format.cpp` (generic Hermes render + parse, JSON args codec)
- `tests/test_tool_format.cpp`

**Modify**
- `libs/lsxcommon/include/lsxcommon/model.h` (two virtuals with defaults)
- `libs/lsxcommon/src/model.cpp` (default implementations) — or inline in the header
- `libs/lsxcommon/include/lsxcommon/prompt_bridge.h` + `src/prompt_bridge.cpp`
  (`FrameConversationForModel`, `ParseToolCallsForModel`)
- `libs/lsxcommon/CMakeLists.txt` (add `tool_format.cpp`)
- `libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h` + `src/openai_endpoint.cpp`
  (request tools/messages; response tool_calls; chunk tool_calls)
- `apps/taskmosaic/src/chat_bridge.cpp` (conversation build, tool mapping, buffered streaming)
- `apps/taskmosaic/tests/test_taskmosaic_bridge.cpp` (tool round-trip)
- Per-family native overrides: `qwen_base_model.*`, `glm_moe_model.*`/`gemma_base_model.*`/`deepseek_moe_model.*`/`kimi_moe_model.*` as available.

---

### Task 1: Conversation/tool types

**Files:** create `chat_conversation.h`; test `tests/test_chat_conversation.cpp`.

**Interfaces:**
- Produces:
  ```cpp
  struct ToolCall { std::string id, name, arguments_json; };
  struct ToolSpec { std::string name, description, parameters_json; };
  struct ChatMessage { std::string role, content, tool_call_id; std::vector<ToolCall> tool_calls; };
  struct ChatConversation { std::vector<ChatMessage> messages; std::vector<ToolSpec> tools; };
  ```

- [ ] **Step 1: Header**

```cpp
#pragma once
#include <string>
#include <vector>
namespace lsxcommon {
struct ToolCall { std::string id; std::string name; std::string arguments_json; };
struct ToolSpec { std::string name; std::string description; std::string parameters_json; };
struct ChatMessage {
    std::string role;          // system|user|assistant|tool
    std::string content;
    std::string tool_call_id;  // role == "tool"
    std::vector<ToolCall> tool_calls;  // role == "assistant"
};
struct ChatConversation {
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec> tools;
};
}  // namespace lsxcommon
```

- [ ] **Step 2: trivial construct/round-trip test**, register target (`fast`), build, commit
  `feat: conversation and tool types`.

---

### Task 2: Generic Hermes tool format

**Files:** create `tool_format.h`/`tool_format.cpp`; test `tests/test_tool_format.cpp`.

**Interfaces:**
- Produces in `namespace lsxcommon::tools`:
  - `std::string RenderToolsBlock(const std::vector<ToolSpec>&)` — JSON array wrapped in `<tools>\n…\n</tools>`.
  - `std::string RenderConversation(const ChatConversation&, bool include_tools)` — Hermes transcript: `system` text (with tools block appended), then `user`/`assistant`/`tool` turns with role markers.
  - `std::vector<ToolCall> ParseToolCallsText(std::string_view text)` — extracts every `<tool_call>…</tool_call>` JSON object; tolerant of an object or a string `arguments`; assigns ids `call_0…`.

- [ ] **Step 1: Failing tests**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/tool_format.h>
using namespace lsxcommon;

TEST_CASE("tool_format: tools block is valid JSON in <tools>",
          "[tools][fast]") {
    std::vector<ToolSpec> ts{{"get_weather", "Get weather",
                              R"({"type":"object","properties":{"city":{"type":"string"}}})"}};
    const std::string b = tools::RenderToolsBlock(ts);
    CHECK(b.find("<tools>") == 0u);
    CHECK(b.find("get_weather") != std::string::npos);
    CHECK(b.rfind("</tools>") != std::string::npos);
}

TEST_CASE("tool_format: parses a Hermes tool call", "[tools][fast]") {
    const std::string text =
        "sure<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
    const auto calls = tools::ParseToolCallsText(text);
    REQUIRE(calls.size() == 1u);
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json == R"({"city":"Paris"})");
}

TEST_CASE("tool_format: parses multiple calls and string arguments",
          "[tools][fast]") {
    const std::string text =
        "<tool_call>{\"name\":\"a\",\"arguments\":\"{}\"}</tool_call>"
        "<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
    const auto calls = tools::ParseToolCallsText(text);
    REQUIRE(calls.size() == 2u);
    CHECK(calls[0].name == "a");
    CHECK(calls[0].arguments_json == "{}");
    CHECK(calls[1].name == "b");
    CHECK(calls[1].arguments_json == R"({"x":1})");
}

TEST_CASE("tool_format: no marker yields no calls", "[tools][fast]") {
    CHECK(tools::ParseToolCallsText("plain answer").empty());
}
```

- [ ] **Step 2: Run to fail**, then implement `tool_format.cpp` with simdjson for
  argument normalization (object → compact JSON, string → as-is) and a linear scan
  for the `<tool_call>`/`</tool_call>` markers. `RenderToolsBlock` emits a compact
  JSON array `[{"type":"function","function":{"name","description","parameters"}}]`.
- [ ] **Step 3: Run to pass**; register in `tests/CMakeLists.txt` (`fast`); commit
  `feat: generic Hermes tool format`.

---

### Task 3: `IModel` conversation/tool virtuals

**Files:** modify `model.h`; test `tests/test_generation_stream.cpp`/`test_model_engine.cpp` compile.

**Interfaces:**
- Produces (additive, with defaults so fakes/tests still compile):
  ```cpp
  virtual bool BuildConversationInputIds(const ArrowTokenizerDb& tok,
                                         const ChatConversation& conv,
                                         std::vector<int32_t>& out_ids) const;
  virtual bool ParseToolCalls(const ArrowTokenizerDb& tok,
                              const std::vector<int32_t>& gen_tokens,
                              std::vector<ToolCall>& out_calls) const;
  ```
- Defaults: `BuildConversationInputIds` = `tools::RenderConversation(conv,false)`
  then `BuildPromptInputIds`; `ParseToolCalls` = decode ids with the tokenizer and
  `tools::ParseToolCallsText`.

- [ ] **Step 1: Declarations + inline defaults in `model.h`** (host-only; include
  `chat_conversation.h` and `tool_format.h`). Decode helper uses
  `arrow_tokenizer_decode` with capacity retry (mirror `runtime::DecodeGeneratedText`).
- [ ] **Step 2: Build `test_model_engine` + a fake-model test asserting the default
  conversation path flattens with role markers and the default parser finds a
  Hermes call.** Commit `feat: IModel conversation and tool-call virtuals`.

---

### Task 4: Native Qwen/Hermes override

**Files:** modify `qwen_base_model.{h,cpp}` (and inherited qwen MoE/ornith if shared).

- [ ] **Step 1: Override `BuildConversationInputIds`** — build the Hermes transcript
  with the tools block (Task 2), then apply the existing framing sentinels via the
  family's `BuildPromptInputIds` on the rendered string.
- [ ] **Step 2: Override `ParseToolCalls`** — decode ids, `tools::ParseToolCallsText`.
- [ ] **Step 3: Host test** — a qwen framing test asserts the tools block appears in
  the encoded text (no GPU). Commit `feat: native qwen tool rendering`.

---

### Task 5: Native DeepSeek / GLM / Kimi (token-level)

**Files:** `deepseek_moe_model.*`, `glm_moe_model.*`, `kimi_moe_model.*` (+ their framing headers).

- [ ] **Step 1: Verify** each family's tool syntax and special-token ids against the
  model tokenizer/card (mandatory before coding; do NOT guess).
- [ ] **Step 2: Override** `BuildConversationInputIds` (native tool block placement)
  and `ParseToolCalls` (scan **token ids** for the native tool-call delimiters, then
  JSON-parse the argument bytes; fall back to `tools::ParseToolCallsText` when the
  native markers are absent).
- [ ] **Step 3: Host tests** per family with synthetic token id sequences (no GPU);
  commit per family.

---

### Task 6: OpenAI codec — tools in/out

**Files:** `openai_endpoint.{h,cpp}`; extend `tests/test_mosaic_network.cpp`.

**Interfaces:**
- Extend `ChatInputMessage` → `{role, content, tool_call_id, std::vector<ChatToolCall> tool_calls}`.
- Extend `ChatRequest` → `std::vector<ToolSpec> tools; std::string tool_choice;`.
- New builders: `BuildChatCompletionWithTools(...)`, and `BuildChatChunk` gains an
  optional `tool_calls_json` payload appended to `delta.tool_calls`.

- [ ] **Step 1: Failing tests** — parse a request with `tools` + assistant
  `tool_calls` + a `role:"tool"` message; build a completion whose message carries
  `tool_calls` and `finish_reason:"tool_calls"`; build a chunk with a `tool_calls`
  delta.
- [ ] **Step 2: Implement** parsing (reuse simdjson; iterate `tools[].function`,
  messages[].tool_calls) and serialization (arguments kept as a JSON **string**
  per the OpenAI wire format).
- [ ] **Step 3: Run to pass**; commit `feat: OpenAI tool calling codec`.

---

### Task 7: Bridge wiring

**Files:** `apps/taskmosaic/src/chat_bridge.cpp`, `lsx_framer.{h,cpp}`; test
`apps/taskmosaic/tests/test_taskmosaic_bridge.cpp`.

**Interfaces:**
- `PromptFramer` gains:
  ```cpp
  virtual bool EncodeConversation(std::string_view model_path,
                                  const lsxcommon::ChatConversation& conv,
                                  std::vector<std::int32_t>& out_ids) = 0;
  virtual bool ParseToolCalls(std::string_view model_path,
                              const std::vector<std::int32_t>& ids,
                              std::vector<lsxcommon::ToolCall>& out) = 0;
  ```
  `LsxPromptFramer` implements them via `FrameConversationForModel` /
  `ParseToolCallsForModel`.
- `FlattenMessages` is replaced by a `BuildConversation(ChatRequest)` that maps
  OpenAI messages/tools into a `ChatConversation`.

- [ ] **Step 1: Non-streaming tool path** — after generation, call
  `framer_.ParseToolCalls`; if non-empty build a `tool_calls` completion with
  `finish_reason:"tool_calls"`; else the existing text completion.
- [ ] **Step 2: Streaming** — when `tools` is non-empty and `tool_choice != "none"`,
  buffer tokens (do not emit live text); on `done`, parse and emit either one
  `delta.tool_calls` chunk or the buffered text, then the finish chunk + `[DONE]`.
- [ ] **Step 3: Test** — extend `FakeFramer` to render/parse a synthetic tool call;
  assert the bridge returns `tool_calls` (non-stream) and a `tool_calls` delta
  (stream). Commit `feat: bridge tool calling`.

---

### Task 8: Regression + docs

- [ ] **Step 1:** run the host suite (all `fast`/bridge tests) green.
- [ ] **Step 2:** update `AGENTS.md`/README with the tool-calling surface and the
  "buffered streaming when tools are present" note.
- [ ] **Step 3:** commit `docs: document bridge tool calling`.

## Self-Review

| Spec §6.2 | Task |
|---|---|
| conversation/tool types | 1 |
| generic Hermes render+parse | 2 |
| `IModel` overloads (defaults) | 3 |
| native per-family | 4, 5 |
| OpenAI request/response tools | 6 |
| bridge mapping + streaming deltas | 7 |
| tests/labels/docs | 1–8 |

**Placeholder scan:** Task 5 intentionally requires tokenizer verification before
coding (native syntax must not be guessed); the generic path (Tasks 1–3, 6–7) is
fully specified.

## Execution Handoff

Execute inline via superpowers:executing-plans. Tasks 1–3 and 6–8 are host-only and
immediately testable; Tasks 4–5 need per-family tokenizer verification. GPU only for
the end-to-end tool round-trip (optional `inference` test).
