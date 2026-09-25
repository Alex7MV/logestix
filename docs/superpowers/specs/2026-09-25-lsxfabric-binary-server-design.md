# lsxfabric Binary-Token Inference Server + taskmosaic Client MVP

**Status:** approved (design), 2026-09-25
**Branch:** `feat/lsxfabric-binary-server`
**Related:** `libs/lsxnetwork/` (FrameQueue, ZMQ, OpenAI endpoint),
`libs/lsxcommon/include/lsxcommon/model.h` (`IModel`), `apps/lsxfabric/`,
`apps/taskmosaic/`.

## Problem

`lsxfabric` currently runs in one of two modes — `--validate` or
`--testprompt` — and then exits. There is no long-lived serving path, and no
way for another process to submit inference work to a running engine. The
engine already exposes everything a server needs: `IModel::Infer` consumes
raw `input_ids` (`ModelRequest::input_ids`) and `gpu::InferenceResult` carries
the generated `gen_tokens` plus timing/eos metadata. `lsxnetwork` already
provides a ZeroMQ context/socket wrapper, an SPSC ring, and an I/O worker.

`taskmosaic` is a stub whose `main()` only initializes `Core`. It must become
a client of a remote `lsxfabric`: convert text to model tokens, submit them,
and convert the returned tokens back to text.

The transport is deliberately binary — no JSON, no HTTP on the inference path.
Clients submit ready `input_ids` and receive generated token ids. Tokenizer
and chat-template handling live on the client, which has host-only access to
the same converted model directory.

## Decisions (agreed)

1. **Transport:** ZeroMQ `REP`/`REQ` (`libzmq` is already a dependency;
   `lsxnetwork` already wraps a ZMQ context/socket).
2. **Concurrency:** exactly one request in flight. The server is a single
   blocking receive/dispatch/send loop; ZMQ queues additional requests at the
   socket. This matches the one-GPU-load-at-a-time policy.
3. **Response shape:** one binary frame carrying the whole generated token
   stream. No token-by-token streaming.
4. **Input semantics:** incoming tokens ARE the model's `input_ids`; the
   server performs no chat-template or BOS framing.
5. **`max_gen`:** a per-request `u32`; `0` means "use the CLI `--max-gen`
   default".
6. **Launch:** new `--serve` mode with `--listen <endpoint>` (default
   `tcp://*:5555`). `--testprompt`/`--validate` keep their behavior; `--serve`
   takes precedence if combined.
7. **Response metadata:** tokens + `status` + `eos` + timings (`ttft_ms`,
   `total_ms`, `tokens_per_sec`).
8. **taskmosaic:** client-only MVP — text -> tokens, per-family chat framing,
   tokens -> text, plus the ZMQ round-trip. No local inference, no GPU.
9. **Codec + client home:** `libs/lsxnetwork` (network layer; links `libzmq`,
   no dependency on `lsxcommon`, so no cycle). Server stays app-local in
   `apps/lsxfabric`.

## Wire protocol

Little-endian, explicit byte-wise encoding (no struct aliasing, no padding).
Magic `"LXFB"`, version `u16 = 1`.

Request — 16-byte header + `4 * num_tokens` body:

```
u32 magic;        // 'L' | 'X'<<8 | 'F'<<16 | 'B'<<24
u16 version;      // 1
u16 reserved;     // 0
u32 max_gen;      // 0 = server default (CLI --max-gen)
u32 num_tokens;
i32 tokens[num_tokens];
```

Response — 40-byte header + body:

```
u32 magic;
u16 version;
u16 status;       // 0=OK, 1=bad-request, 2=infer-failed, 3=internal
u32 num_tokens;
u32 flags;        // bit0 = eos_terminated
f64 ttft_ms;
f64 total_ms;
f64 tokens_per_sec;
// status == 0 -> i32 tokens[num_tokens]
// status != 0 -> UTF-8 error message (possibly empty), num_tokens = 0
```

Decoder validation (all failures -> `status = 1` response, connection stays
open): magic, version, `num_tokens <= (frame_size - header) / 4` (no
over/underflow), no trailing bytes, and for responses the same shape checks
against `status`. Encoding requires no allocation on the server beyond the
request/response vectors already owned by the codec call site.

## Components

### `libs/lsxnetwork/include/lsxnetwork/fabric_wire.h` + `src/fabric_wire.cpp`

Pure codec. Types: `FabricRequestHeader`, `FabricResponseHeader`,
`FabricResponse` (`int status; bool eos; double ttft_ms, total_ms,
tokens_per_sec; std::vector<int32_t> tokens; std::string error;`). Functions:

- `std::vector<std::byte> EncodeFabricRequest(std::span<const int32_t> ids, uint32_t max_gen);`
- `enum class WireStatus { kOk, kBadMagic, kBadVersion, kTruncated, kTrailing };`
- `WireStatus DecodeFabricRequest(std::span<const std::byte>, FabricRequestView&);`
  where `FabricRequestView` is a non-owning `{uint32_t max_gen; std::span<const int32_t> tokens;}`.
- `std::vector<std::byte> EncodeFabricResponse(const FabricResponse&);`
- `WireStatus DecodeFabricResponse(std::span<const std::byte>, FabricResponse&);`

No ZMQ, no `lsxcommon` dependency; unit-testable as pure functions.

### `libs/lsxnetwork/include/lsxnetwork/fabric_client.h` + `src/fabric_client.cpp`

`class FabricClient` owns a `ZmqContext` and a `ZMQ_REQ` `ZmqSocket`.

- `int Connect(const char* endpoint)` — `ZmqSocket::Connect`.
- `bool Infer(std::span<const int32_t> input_ids, uint32_t max_gen,
  FabricResponse& out, int timeout_ms)` — encode, send, receive (arbitrary
  size via `zmq_msg`), decode. Transport failure or timeout returns `false`
  and sets `out.status = -1`.
- Blocking, one request in flight (REQ must alternate send/recv).

### `apps/lsxfabric/src/fabric_server.{h,cpp}`

`class FabricServer` — constructed with `lsxcommon::IModel& model` and
`uint32_t default_max_gen`.

- `int Run(const std::string& endpoint, std::atomic<bool>& stop)`:
  `ZmqContext` + `ZMQ_REP`, `Bind`, `ZMQ_RCVTIMEO` (~200 ms) so the loop can
  observe `stop`. On each frame: decode; malformed -> `status=1` reply;
  otherwise build `ModelRequest{input_ids, max_gen = req.max_gen ?: default}`,
  take `std::shared_lock(ResidencyMutex())`, set `abort_flag = &abort_`, call
  `model.Infer`; encode `gen_tokens`/stats/eos on success, `status=2` +
  `error_msg` on failure. `stop` -> return.
- No internal threads. The model is loaded once by `main.cpp`.

### `apps/lsxfabric/src/main.cpp` + `cli.{h,cpp}`

- `FabricCliArgs` gains `bool serve` and `std::string listen = "tcp://*:5555"`.
- `--serve` / `--listen` parsed; `--serve` selects the server path before
  `--testprompt`/`--validate`.
- Server path: create the model through `LsxModelFactory` (kAuto perf policy,
  as in `CreateCliModel`), `Load()`, run `FabricServer`, `Unload()` on exit.
  `Core::Initialize`/`Shutdown` and `InferenceEngine::Session` stay as today.

### `libs/lsxcommon` host-only bridge

- `bool LoadTokenizerArrows(const std::string& model_path, ArrowTokenizerDb& out)`
  — extracted from `ModelBase::LoadTokenizerDb` (same pinned-dir resolution and
  `load_tokenizer_metadata`); `ModelBase::LoadTokenizerDb` is refactored to
  call it.
- `bool FramePromptForModel(const std::string& model_path,
  std::string_view raw_prompt, std::vector<int32_t>& out_ids)` — creates the
  `IModel` via `LsxModelFactory::Create` **without** `Load()`, loads the
  tokenizer through the helper, and calls the virtual
  `IModel::BuildPromptInputIds(tok, prompt, out_ids)`. This keeps chat framing
  owned by each model (per `AGENTS.md`, no central type -> framing registry);
  the throwaway instance never touches GPU weights.

### `apps/taskmosaic/src/main.cpp`

Client-only MVP. Args: `--server <endpoint>` (required), `--model <path>`
(required, for tokenizer/framing), `--prompt <text>` or stdin, `--max-gen N`.
Flow: `LoadTokenizerArrows` -> `FramePromptForModel` -> `FabricClient::Infer`
-> `runtime::DecodeGeneratedText` -> print text plus status/timings. No
`IModel::Load`, no inference.

## Error handling

- Malformed request: `status=1`, server continues serving.
- `IModel::Infer` failure: `status=2` + `error_msg`.
- Unexpected exception in dispatch: caught in the server loop -> `status=3`.
- Client transport error/timeout: `Infer` returns `false`, `out.status=-1`.
- Oversized input: rejected by the decoder bound check (`status=1`), never
  allocating `num_tokens` beyond the frame size.

## Testing (TDD — tests first, red)

All new tests carry label `fast`; none require a GPU.

1. `tests/test_fabric_wire.cpp` — request/response round-trips; empty token
   stream; `max_gen` boundaries; malformed magic/version, truncated header,
   truncated body, `num_tokens` larger than the body, trailing bytes; error
   response (empty and non-empty message).
2. `tests/test_fabric_client.cpp` — a fake `ZMQ_REP` server thread built from
   `ZmqSocket` + the codec on `inproc://`; real `FabricClient` round-trip for
   OK and error responses; timeout path.
3. `apps/lsxfabric/tests/test_fabric_server.cpp` — a fake `IModel` (echoes the
   input, configurable delay/failure) + real `FabricServer` + real
   `FabricClient` on `inproc://`; asserts input ids reach `Infer`, per-request
   `max_gen` override vs default, `eos`/`status` mapping, infer-failure ->
   `status=2`, malformed -> `status=1`, graceful `stop`.
4. `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` — extended for
   `--serve`/`--listen` parsing.
5. Optional follow-up (not in this MVP): an `inference`-labeled end-to-end
   smoke against a real converted model.

## CMake / branch / commits

- `libs/lsxnetwork/CMakeLists.txt`: add `src/fabric_wire.cpp`,
  `src/fabric_client.cpp`.
- `apps/lsxfabric/CMakeLists.txt`: add `src/fabric_server.cpp`; link
  `lsxnetwork` explicitly.
- `apps/taskmosaic/CMakeLists.txt`: link `lsxnetwork` for `FabricClient`.
- New test executables in `tests/CMakeLists.txt` and
  `apps/lsxfabric/tests/CMakeLists.txt`, each `LABELS fast` in the same
  commit (per `test-lifecycle` and `test_ctest_labels_guard`).
- Guard tests unaffected: no family-prefixed stems added
  (`test_code_naming_guard`), no model registration changes
  (`test_model_suite_guard`, `test_model_independence_guard`).

Commit sequence (prefixes per `coding-conventions`): `test:` red codec
tests -> `feat:` fabric_wire -> `test:` red client tests -> `feat:`
fabric_client -> `test:` red server tests -> `feat:` FabricServer + CLI ->
`feat:` taskmosaic client MVP + host-only bridge -> `build:` CMake/labels ->
`docs:` this spec.

## Out of scope

- Streaming token responses.
- Multi-client `ROUTER` multiplexing / parallel inference.
- Server-side tokenization / detokenization endpoints.
- TLS, authentication, compression.
- Persistent request ids / cancellation beyond the `stop` abort flag.

## As-built deltas (2026-09-25)

- `FabricServer::Run` additionally takes optional `std::atomic<bool>* ready`
  and `std::string* resolved_endpoint` out-params: the former lets tests
  synchronize on a successful bind, the latter resolves a
  `tcp://127.0.0.1:*` wildcard (via `ZMQ_LAST_ENDPOINT`) so tests can use an
  ephemeral port without hardcoding one.
- The host-only lsxcommon bridge is split into
  `tokenizer_arrows.{h,cpp}` (`LoadTokenizerArrows`) and
  `prompt_bridge.{h,cpp}` (`FramePromptForModel`); `ModelBase::LoadTokenizerDb`
  now delegates to `LoadTokenizerArrows` (behavior-preserving refactor,
  regression-checked by `test_tokenizer` / `test_model_factory`).
- `lsxnetwork` owns `fabric_wire.{h,cpp}` and `fabric_client.{h,cpp}`;
  `FabricClient` receives via `zmq_msg` (arbitrary-size frames).
- taskmosaic CLI parsing lives in `apps/taskmosaic/src/cli.{h,cpp}` with a
  `fast`-labeled `test_taskmosaic_cli` target.

