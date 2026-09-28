# TaskMosaic OpenAI/Anthropic Bridge — design

**Status:** approved (design), 2026-09-28
**Branch:** `feat/taskmosaic-openai-bridge`
**Related:** `apps/taskmosaic/`, `apps/lsxfabric/`, `libs/lsxnetwork/`,
`libs/lsxcommon/`, `docs/superpowers/specs/2026-09-25-lsxfabric-binary-server-design.md`,
`AGENTS.md` §Resources/§Conventions/§absol, skills `running-a-model`,
`test-lifecycle`, `cuda-cmake-build-gotchas`, `nvfp4-blackwell-conventions`.

## 1. Problem

`lsxfabric --serve` runs an inference engine behind a ZeroMQ REP binary-token
protocol that returns a whole generated token stream per request. `taskmosaic`
is a client-only CLI. There is no HTTP surface, so no OpenAI-compatible or
Anthropic-compatible client (opencode, Claude Code, Cline, Roo, Continue,
Aider, …) can use a logestix model.

`lsxnetwork` already holds a `openai_endpoint` JSON codec, an in-memory
`HttpRouter` (canned SSE), an SPSC `FrameQueue` and a pinned `IoWorker`, but no
listening HTTP transport and no real inference in the path. The engine already
exposes a per-token `ModelRequest::token_callback` and a cooperative
`ModelRequest::abort_flag`, but only the gigachat and nemotron generation loops
invoke the callback; `InferenceEngine::Run` merely replays tokens after
`model.Infer` returns.

**Goal.** Turn `taskmosaic` into an OpenAI-compatible (and Anthropic Messages
compatible) HTTP/SSE server that bridges to `lsxfabric` as the GPU engine, with
real incremental token streaming, per-request cancellation, and enough
compatibility for agentic clients (tool/function calling included). Extend
`lsxfabric` and the model families where required.

## 2. Decisions (agreed)

1. **Topology.** `taskmosaic` is the HTTP bridge; `lsxfabric` owns the GPU and
   serves one model per process. The bridge tokenizes/frames on the host and
   forwards tokens over ZeroMQ. Enforced one GPU load at a time.
2. **Streaming.** Real incremental streaming. `lsxfabric` streams token batches
   as separate ZMQ messages while `model.Infer` runs.
3. **API surface.** `/v1/chat/completions`, `/v1/completions` (legacy),
   `/v1/models` + `/v1/models/{id}`, `/health`, `/ready`; plus Anthropic
   `/v1/messages`, `/v1/messages/count_tokens`, `HEAD /api/hello`, Anthropic
   shaped `GET /v1/models`.
4. **Multi-model.** Bridge config maps model id → lsxfabric endpoint (+ cancel
   endpoint) + converted-model path. Requests route by the `model` field;
   `/v1/models` lists configured ids.
5. **HTTP transport.** `cpp-httplib` via FetchContent (new pinned dependency).
6. **Concurrency.** One in-flight request per backend; bounded FIFO queue per
   backend (`queue_depth`); overflow → 429.
7. **Cancellation.** Client disconnect (or a matched `stop` sequence) aborts the
   in-flight generation via a `PUSH/PULL` cancel channel that sets the model's
   `abort_flag`.
8. **Wire protocol.** DEALER(client)/ROUTER(server) for request + stream, so
   every token batch is its own flushed ZMQ message; `PUSH/PULL` for cancel.
   (Multipart over REQ/REP is rejected: libzmq may batch `SNDMORE` parts until
   the final frame, so it cannot guarantee incremental delivery.)
9. **Chat flattening.** Multi-turn `messages` are rendered as a full transcript
   with role markers and fed through the family chat template.
10. **Family streaming coverage.** Thread the per-token callback and abort
    checks into **all** family generation loops.
11. **Tool calling.** OpenAI `tools`/`tool_calls` end-to-end; native per-family
    render+parse where a tool format is known (Qwen/Hermes, DeepSeek, GLM,
    Kimi), generic Hermes fallback otherwise.
12. **Anthropic.** A native Anthropic Messages adapter in the bridge (no
    external translator) so Claude Code can connect directly.
13. **Compatibility.** `stream_options.include_usage`, tolerant parsing of
    sampling params, real `stop`, correct `finish_reason`/`stop_reason`.

## 3. Architecture

```
OpenAI / Anthropic client
        │  HTTP + SSE (cpp-httplib, taskmosaic)
        ▼
taskmosaic bridge (CPU, host-only)
  ├─ config registry (id → backend + model path)
  ├─ ModelContext per id (tokenizer, framer, incremental detokenizer)
  ├─ per-backend bounded queue + one FabricStreamClient
  └─ format adapters: openai  |  anthropic
        │  ZMQ DEALER (request / stream)
        │  ZMQ PUSH   (cancel)
        ▼
lsxfabric (GPU, one model/process)
  ├─ ROUTER serve loop (main thread; Infer runs here)
  ├─ PULL cancel listener thread → sets abort_flag
  └─ token_callback → ROUTER token-batch messages
```

`taskmosaic` links `lsxcommon` (tokenizer/framing/detokenizer) and `lsxnetwork`
(wire codec/stream client/HTTP server wrapper). `lsxfabric` links the same and
keeps the engine lifecycle (`InferenceEngine::Session`) and per-GPU perf policy.

## 4. Wire protocol v2

Little-endian, explicit byte-wise encoding (no struct aliasing). Magic `"LXFB"`,
version `u16 = 2`. Supersedes the v1 REQ/REP codec; the v1 codec and its tests
are replaced.

**Request** (client → server, one message):

```
u32 magic;          // 'L' | 'X'<<8 | 'F'<<16 | 'B'<<24
u16 version;        // 2
u16 reserved;       // 0
u32 request_id;     // bridge-assigned, echoed on every stream frame
u32 max_gen;        // 0 => server CLI --max-gen default
u32 num_tokens;
i32 tokens[num_tokens];
```

**Stream** (server → client, each a separate ZMQ message):

```
token_batch:  u32 magic; u16 version; u16 type=1; u32 request_id;
              u32 count; i32 tokens[count];
done:         u32 magic; u16 version; u16 type=2; u32 request_id;
              u32 flags;        // bit0 = eos_terminated
              u32 total_tokens;
              f64 ttft_ms; f64 total_ms; f64 tokens_per_sec;
error:        u32 magic; u16 version; u16 type=3; u32 request_id;
              u32 len; bytes[ len ]   // UTF-8 message
```

**Cancel** (client → server on `PUSH`):

```
u32 magic 'LXFC'; u16 version=2; u16 reserved; u32 request_id;
```

Decoder validation (all failures → `error` frame, server keeps serving): magic,
version, `num_tokens`/`count`/`len` bounded by frame size, no trailing bytes.
Non-streaming OpenAI clients are handled entirely on the bridge: the server
always streams, the bridge accumulates until `done`.

## 5. lsxfabric changes

`apps/lsxfabric/src/fabric_server.{h,cpp}`:

- Bind a `ROUTER` on `--listen` and a `PULL` on `--listen-cancel`.
- Per request: assign/validate `request_id`, create a local
  `std::atomic<bool> abort{false}`, register `{id → &abort}` in a small
  mutex-guarded registry (only one in flight), capture the ROUTER identity.
- Build `ModelRequest{input_ids, max_gen = req.max_gen ?: default,
  abort_flag = &abort, token_callback = sender}` and call `model.Infer` **on the
  main thread** (so the ROUTER socket is used from one thread). `sender` encodes
  a `token_batch` message with the captured identity envelope and sends it via
  ROUTER. On return, send `done` (or `error`), unregister.
- Bypass `InferenceEngine::Run` replay (would double-call the callback); call
  `model.Infer` directly, as today. `Run` is left intact for other callers.
- Cancel listener thread owns the `PULL` socket: on `cancel{id}` sets the
  registered atomic. Cleared/joined on stop.

CLI: add `--listen-cancel` (default `tcp://*:5556`). `--serve` selection and
`ready`/`resolved_endpoint` synchronization keep working; `resolved_endpoint`
must also be reportable for the cancel socket.

## 6. Model families — streaming + tools

### 6.1 Per-token streaming (all families)

Add the token callback (and per-step abort check) to every generation loop:
`qwen::RunQwenGeneration`, `qwen_moe::RunQwenMoeGeneration`,
`qwen_moe_exp::RunExpGeneration`, `ornith::RunOrnithMoeGeneration`,
`gpu::RunInferenceModel` (deepseek v4 / v4.1), `glm::RunGlmRealGeneration`,
`glm::RunGlmGeneration`, `glm_moe_flash::RunGlmFlashGeneration`,
`gemma::RunGemmaGeneration`, `granite::RunGraniteGeneration`,
`kimi::RunKimiGeneration`; gigachat and nemotron already call the callback.

Rules:

- Only add the callback invocation and the abort probe — **no math change**.
  Bit-exactness must be preserved (existing `--bit-exact`, SelfCheck/FSI/CODE/DET
  gates stay green).
- Each `Infer` passes `req.token_callback` and `req.abort_flag` into its loop.
- `InferenceEngine::Run`'s post-hoc replay is retained for callers that still
  use it; the family loops are the single source of per-token events.

### 6.2 Tool calling (model side)

New host-only types in `libs/lsxcommon` (`chat_conversation.h`):

```cpp
struct ToolCall { std::string id, name, arguments_json; };
struct ToolSpec { std::string name, description, parameters_json; };

struct ChatMessage {
    std::string role;               // "system"|"user"|"assistant"|"tool"
    std::string content;
    std::string tool_call_id;       // role == "tool"
    std::vector<ToolCall> tool_calls;  // role == "assistant"
};
struct ChatConversation {
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec> tools;
};
```

`IModel` additions (default implementations keep the interface additive):

```cpp
// Renders a full conversation (incl. tool specs) into input ids. Default:
// flatten the transcript with role markers and call the single-string
// BuildPromptInputIds (no tools).
virtual bool BuildConversationInputIds(const ArrowTokenizerDb& tok,
                                       const ChatConversation& conv,
                                       std::vector<int32_t>& out_ids) const;

// Parses generated tokens into tool calls. Token-id based because
// arrow_tokenizer_decode drops special tokens, and DeepSeek/GLM/Kimi tool
// syntax is built from special tokens. Default: generic Hermes text parser
// over the decoded text.
virtual bool ParseToolCalls(const ArrowTokenizerDb& tok,
                            const std::vector<int32_t>& gen_tokens,
                            std::vector<ToolCall>& out_calls) const;
```

Ownership stays per family (consistent with `prompt::FramingDescriptor`, no
central registry):

- **Qwen / Ornith / Hermes-style:** tools as a JSON block in the system prompt
  (`<tools>…</tools>`); calls `<tool_call>{"name":…,"arguments":{…}}</tool_call>`.
- **DeepSeek (V4/V4.1):** native tool tokens + system tool block; exact token
  strings verified against the model tokenizer/card at implementation time.
- **GLM:** native `<|system|>` tool block and tool-call sentinels.
- **Kimi:** native tool block and call syntax.
- **Generic fallback (Gemma, Granite, Nemotron, GigaChat, others):**
  Hermes-style render + parse.

Bridge access: extend `prompt_bridge` with
`FrameConversationForModel(model_path, conv, out_ids)` and
`ParseToolCallsForModel(model_path, tok_ids, out_calls)`; both build a throwaway
`IModel` via the factory (no `Load`), like the existing `FramePromptForModel`.

## 7. taskmosaic bridge components

### 7.1 Config (`apps/taskmosaic/src/config.{h,cpp}`)

JSON file via `--config`, overridable with `--host`/`--port`; parsed with
simdjson into owned structs.

```json
{
  "listen": { "host": "127.0.0.1", "port": 8080 },
  "default_model": "granite-4.2-30b",
  "queue_depth": 4,
  "request_timeout_ms": 300000,
  "default_max_tokens": 64,
  "models": [
    {
      "id": "granite-4.2-30b",
      "model_path": "/vault/models/granite-4.2-30b",
      "backend": {
        "endpoint": "tcp://127.0.0.1:5555",
        "cancel_endpoint": "tcp://127.0.0.1:5556"
      }
    }
  ]
}
```

### 7.2 Model context (`model_context.{h,cpp}`)

Per model id: `ArrowTokenizerDb`, cached throwaway `IModel` framer, and an
`IncrementalDetokenizer`. Built at startup; a bad model path fails startup.

### 7.3 Backend + queue (`backend.{h,cpp}`)

One `FabricStreamClient` (DEALER + PUSH). Atomic admission counter bounded by
`queue_depth`; no slot → 429. A mutex serializes actual use of the connection
(one in flight). Tracks readiness (connected). `Cancel(id)` sends on PUSH.

### 7.4 Incremental detokenizer (`libs/lsxcommon/incremental_detokenizer.{h,cpp}`)

Feeds ids one at a time; decodes the full id list, diffs against the previously
emitted text, and emits only the new suffix up to the last complete UTF-8
sequence (holds back a trailing incomplete sequence). Special tokens are skipped
by the decoder, matching `DecodeGeneratedText` semantics.

### 7.5 Format adapters

- `openai_endpoint.{h,cpp}` (extended): full request parse (messages incl.
  tools/roles/tool_call_id, `stream`, `stream_options`, `max_tokens`, `stop`,
  unknown params); serializers for `chat.completion`, `chat.completion.chunk`,
  `text_completion` (+chunk), `model`, `list`, `error`, usage.
- `anthropic_endpoint.{h,cpp}` (new): Anthropic request parse (`system`,
  content blocks `text`/`tool_use`/`tool_result`/`image`, `tools` with
  `input_schema`, `max_tokens`) → `ChatConversation` + `ToolSpec`; serializers
  for the message body and the SSE event sequence.
- `chat_bridge.{h,cpp}`: shared orchestration (resolve model, admission,
  framing, submit, consume frames, stop handling, cancel, finish reason).
- SSE writers: `openai_sse` and `anthropic_sse` over a common sink abstraction.

### 7.6 taskmosaic entry point and the legacy client

`apps/taskmosaic/src/main.cpp` gains a server mode selected by `--config`
(`--host`/`--port` overrides); it loads the config, builds the model contexts,
registers HTTP routes, and serves until SIGINT/SIGTERM. The existing client-only
CLI (`--server --model --prompt`, using the v1 REQ `FabricClient`) is removed
with the v1 wire — `taskmosaic` becomes the bridge. The v1 REQ client/server
(`lsxnetwork/fabric_client.*` and the old REP body of `FabricServer`) are
replaced by the v2 stream client and ROUTER server; their tests are rewritten
for v2.

### 7.7 HTTP server (`libs/lsxnetwork/http_server.{h,cpp}`)

Thin wrapper over `cpp-httplib` (private dependency; no httplib type in the
public header — handlers are `std::function`). Static SPA requests delegate to
the existing `HttpRouter`. SSE via a chunked content provider; `sink.is_writable()`
detects client disconnect. Routes are registered by taskmosaic.

## 8. Endpoint mapping

### OpenAI

- `GET /health` → 200 `{"status":"ok"}` (no backend).
- `GET /ready` → 200, or 503 with per-backend state.
- `GET /v1/models` → `{"object":"list","data":[{id,object:"model",created,owned_by:"logestix"}]}`.
- `GET /v1/models/{id}` → single model or 404 `model_not_found`.
- `POST /v1/chat/completions`:
  - resolve model (config `default_model` when omitted) → 404 if unknown;
  - render `ChatConversation` (flatten + tools) → input ids;
  - submit; `stream:false` → accumulate, decode, single JSON with usage
    (`prompt_tokens` = input id count, `completion_tokens` = generated count);
  - `stream:true` → SSE `chat.completion.chunk` (first delta `role`, then
    content deltas, optional `tool_calls` delta, final `finish_reason`), then
    `data: [DONE]`; if `stream_options.include_usage` → a usage-only chunk;
  - client disconnect / `stop` match → `Cancel`.
- `POST /v1/completions` (legacy): `prompt` (string/array) used as the raw
  prompt (no chat template); `text_completion` / stream chunks with `text`.
- Errors: OpenAI object `{"error":{"message","type","param","code"}}`.

### Anthropic

- `HEAD /api/hello` → 200 (connection warming).
- `POST /v1/messages` (`? beta=true` ignored): parse → same backend path.
  - non-stream: `{id,type:"message",role:"assistant",model,
    content:[{type:"text",text}|{type:"tool_use",id,name,input}],
    stop_reason,stop_sequence,usage:{input_tokens,output_tokens}}`;
  - stream: `message_start` → (`content_block_start`/`content_block_delta`/
    `content_block_stop`)… → `message_delta`(stop_reason,usage) →
    `message_stop`, with `ping` during silent gaps; `text/event-stream`.
- `POST /v1/messages/count_tokens` → `{"input_tokens":N}` from the tokenizer.
- `GET /v1/models?limit=1000` → Anthropic shape
  `{"data":[{id,display_name,description,type:"model"}],"has_more":false}`,
  served directly (no redirect).
- Errors: `{"type":"error","error":{"type","message"}}`; `retry-after` integer
  seconds on 429.
- Document the exact `ANTHROPIC_BASE_URL` value to use (no `/v1` suffix) and the
  model id to select.

## 9. Compatibility details

- `stream_options.include_usage` → trailing usage chunk (`choices: []`).
- Accept and ignore `temperature`, `top_p`, `top_k`, `seed`,
  `presence_penalty`, `frequency_penalty`, `response_format`, `logprobs`,
  `parallel_tool_calls`; `n > 1` → return a single choice (no 400).
- `stop` (OpenAI) / `stop_sequences` (Anthropic): scan each detokenized delta;
  on the first match, truncate and `Cancel` the backend, `finish_reason:"stop"`
  / `stop_reason:"stop_sequence"`.
- `tool_choice`: `none` omits tools; `auto`/`required`/`{function}` inject tools
  (forcing is best-effort; text output is still returned if the model ignores
  it).
- `finish_reason`: `stop` | `length` | `tool_calls`; Anthropic `stop_reason`:
  `end_turn` | `max_tokens` | `tool_use` | `stop_sequence`.

## 10. Concurrency, backpressure, cancellation

- Admission: atomic per-backend counter, max `queue_depth`; overflow 429.
- Serialization: backend connection mutex (one in flight).
- Request timeout (`request_timeout_ms`) → 504 and `Cancel`.
- Slow HTTP client: the SSE sink blocks → the bridge stops receiving → ROUTER
  send backpressure throttles GPU decode (bounded buffering; ROUTER `ZMQ_SNDHWM`
  tunable).
- Disconnect: `sink.is_writable() == false` → `Cancel(request_id)`.
- Cancel channel is best-effort: if the backend already finished, the stray
  cancel is ignored (unknown id).

## 11. Error handling

- Malformed JSON → 400 in the client's format.
- Unknown model → 404.
- Queue full → 429 (+ `retry-after` for Anthropic).
- Backend infer failure / malformed reply → 500/502; timeout → 504.
- Mid-stream failure: OpenAI emits an SSE error JSON then `data: [DONE]`;
  Anthropic emits an `error` event and ends the stream.
- Oversized frames/inputs rejected by the decoder bounds; never allocate beyond
  the frame size.

## 12. Testing (TDD — tests first, red)

No test requires a GPU in the default suite; labels per `test-lifecycle`.

- **fast / codec:** v2 request/stream/cancel round-trips, malformed magic/
  version/truncation/trailing, bounds; `FabricStreamClient` over inproc
  ROUTER + PUSH cancel.
- **fast / lsxnetwork:** `HttpServer` wrapper on an ephemeral port (static via
  `HttpRouter`, SSE sink, disconnect).
- **fast / lsxcommon:** incremental detokenizer (UTF-8 boundaries, special
  skipping); conversation types; generic Hermes render/parse.
- **fast / taskmosaic:** config parse (defaults, missing/typed fields),
  message flatten + tool rendering, backend admission/429, both SSE writers.
- **fast / lsxfabric:** fake `IModel` emitting chunks with delays → incremental
  order, `done` timing, cancel mid-stream sets abort, malformed → error,
  graceful stop.
- **fast / integration (in-process):** httplib + fake DEALER/ROUTER backend →
  OpenAI stream/non-stream, tools, `/v1/models`, `/health`, 429; Anthropic
  non-stream/stream event sequence, `count_tokens`, Anthropic `/v1/models`.
- **inference:** per family, a streaming assert that `token_callback` receives
  tokens in order equal to `gen_tokens`; tool parse/render per family where the
  native format exists. Existing SelfCheck/FSI/CODE/DET and bit-exact gates must
  stay green (no math change).
- Guard tests: new CTest targets get labels in the same commit
  (`test_ctest_labels_guard`); no new family-prefixed stems
  (`test_code_naming_guard`); no model registration changes
  (`test_model_suite_guard`, `test_model_independence_guard`).

## 13. CMake / dependencies / conventions

- New dependency: `cpp-httplib` via FetchContent with a pinned `GIT_TAG`; follow
  the dependency-bump procedure (pin + AGENTS/README tables + after-fetch check,
  skill `cuda-cmake-build-gotchas`). Add a `CPPHTTPLIB_VERSION` guard alongside
  the existing pinned-version `STATIC_REQUIRE` set in `tests/test_core.cpp`
  (optional but preferred).
- httplib is host-only, linked PRIVATE to `lsxnetwork`; must not leak types
  through public headers. Windows: `WIN32_LEAN_AND_MEAN`, `NOMINMAX`.
- `.cu` files must not see httplib or tool/HTTP headers reachable from `.cu`
  (absol reachability rule). Host-only tool types go in host-only headers.
- Logging via spdlog only; `absl::` container/string policy for host-only code.

## 14. Risks

- **Callback threading across ~12 `.cu` loops** — largest risk; mitigated by
  adding only the callback/abort probe (no math change) and by the existing
  bit-exact/SelfCheck gates. Do it family by family behind `inference` tests.
- **Native tool formats** — token-level syntax must be verified against each
  model tokenizer/card before implementation; a wrong format silently degrades
  agent behavior. Generic fallback bounds the blast radius.
- **Tool-call streaming granularity** — emitting a single completed `tool_calls`
  delta is interoperable but not token-incremental; acceptable MVP, documented.
- **cpp-httplib + MSVC/Windows** — header-only, host-only; keep it out of NVCC
  TUs.
- **ZMQ ROUTER send backpressure** — a slow client stalls decode; bounded by
  `ZMQ_SNDHWM` and request timeout.
- **Scope** — large, multi-phase; the plan must phase streaming (per family),
  tools (per family), and the Anthropic adapter independently.

## 15. Out of scope

- Sampling control (the engine's own default sampler is used; params accepted
  but ignored).
- Auth/API-key validation, TLS, CORS (accept credentials headers, never reject).
- Multimodal image inputs (Anthropic `image` blocks parsed but not fed to the
  model).
- `/v1/embeddings`, `/v1/responses`, continuous batching, parallel inference
  per backend, prompt caching, `logprobs`.
- Replacing/removing the legacy `LsxOrchestrator` (left untouched).
