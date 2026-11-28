# MCPSelfCheck — the MCP protocol-payload gate

Status: implemented 2026-11-28. Fourth member of the parity-gate family
(FSI, CODE, DET), after `docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`,
`docs/superpowers/specs/2026-09-25-code-selfcheck-design.md` and
`docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`.

## 1. What it proves, and why it is a separate gate

A converted model that will be driven by an agent harness has to emit
**wire-correct MCP payloads**: a `tools/call` request, a `resources/read`
request, and a JSON-RPC 2.0 error response. That is a functionality of its own,
not a variant of the arithmetic gate (FSI), the compiler-syntax gate (CODE) or
the determinism gate (DET).

Decoupling invariants — same table shape as the FSI spec, because the same
regressions are available:

| Dimension | SelfCheck | FSI | CODE | DET | **MCP** |
|---|---|---|---|---|---|
| Entry point | `IModel::SelfCheck` | `fsi::RunFsiSelfCheck` | `code::RunCodeSelfCheck` | `det::RunDetSelfCheck` | `mcp::RunMcpSelfCheck` |
| Input | hardcoded "The capital of France is" | golden `input_ids` | golden `input_ids` | golden `input_ids` | golden `input_ids` |
| Reference | in-test `"Paris"` | committed golden (model-defined) | committed golden | committed golden | committed golden (**protocol-defined**) |
| Criterion | top-1 text | answer text `==` golden | minified JSON `==` golden | 3 identical runs + answer | **JSON document equality** (§3) |
| Result | `InferenceResult.selfcheck_passed` | `fsi::FsiResult` | `code::CodeResult` | `det::DetResult` | `mcp::McpResult` |
| Numeric mode | `kAuto` | `kOff` | `kOff` | `kOff` | `kOff` |
| Env gate | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` | `LSX_CODE_GOLDEN_DIR` | `LSX_DET_GOLDEN_DIR` | `LSX_MCP_GOLDEN_DIR` |
| CLI | `--selfcheck` | `--fsi-selfcheck` | `--code-selfcheck` | `--det-selfcheck` | `--mcp-selfcheck` |
| Host unit test | `test_model_runtime.cpp` | `test_fsi_selfcheck.cpp` | `test_code_selfcheck.cpp` | `test_det_selfcheck.cpp` | `test_mcp_selfcheck.cpp` |
| Real-model test | `test_<family>_selfcheck` | `test_<family>_fsi_selfcheck` | `test_<family>_code_selfcheck` | `test_<family>_det_selfcheck` | `test_<family>_mcp_selfcheck` |
| Golden | — | `fixtures/fsi/<type>.json` | `fixtures/code/<type>.json` | `fixtures/det/<type>.json` | `fixtures/mcp/<type>.json` |

`IModel::SelfCheck`, `runtime::RunParisSelfCheck`, `selfcheck_passed` and
`LSX_SELFCHECK_EXPECT_HASH` are untouched, as are the FSI/CODE/DET harnesses.

## 2. The prompt set

`tools/mcp_prompts.json` (`format: logestix-mcp-prompts`) is the single source
of truth. C++ never embeds a prompt body: the golden carries the captured
`input_ids`.

| id | kind | anchor | payload |
|---|---|---|---|
| MCP-TEST-01 | `tool_call` | `tools/call` | `{"jsonrpc":"2.0","method":"tools/call","params":{"name":"fetch_secure_ledger","arguments":{"node_id":"NX-5090"}},"id":1}` |
| MCP-TEST-02 | `resource_read` | `resources/read` | `{"jsonrpc":"2.0","method":"resources/read","params":{"uri":"schema://fsi/basel_three.json"},"id":2}` |
| MCP-TEST-03 | `error_response` | `Invalid params` | `{"jsonrpc":"2.0","error":{"code":-32602,"message":"Invalid params"},"id":99}` |

The three cover the three shapes a client library has to construct or accept:
a request with nested arguments, a request with a URI parameter, and a standard
protocol error (`-32602 Invalid params`).

## 3. The criterion: JSON document equality, not byte equality

**This is the one deliberate departure from CODE, and it is forced by where the
golden comes from.**

FSI/CODE/DET goldens are *model-defined*: the expected answer is what the source
model itself said, so text equality is parity. An MCP payload is
*protocol-defined*: JSON-RPC 2.0 says a request is an object with `jsonrpc`,
`method`, `params` and `id`, and it says nothing about key order or whitespace.
Comparing bytes would test whether the model guessed the template's layout —
which nothing in the protocol requires, and which most models will lose, turning
the gate into a wall of debt that measures nothing.

So the harness:

1. extracts the first balanced `{...}` that contains the case's `anchor` and
   **minifies** it (whitespace removed outside string literals) — so a model
   that pretty-prints is not failed for it;
2. **canonicalizes** it: keys sorted, strings escaped canonically, numbers in
   shortest round-trip form;
3. compares the canonical form with the canonical form of the golden's
   `expected`.

Canonical equality still pins every field that matters, and it is strict about
the set: a missing key, an extra key, a wrong `id`, a wrong `-32602`, a string
that changed a character — all fail. It only frees the *order* of keys and
*insignificant whitespace*, which JSON itself declares insignificant.

`exact` is recorded per case: whether the extracted text equals the golden
`expected` byte for byte, i.e. whether the model reproduced the template's
layout. It is logged and reported, and it gates nothing, so "how close to the
template is this model" stays measurable without being a verdict.

## 4. The golden

```json
{ "format": "logestix-mcp-golden", "version": 1, "model_type": "<Type()>",
  "source": "prompt-template (JSON-RPC 2.0 payload from tools/mcp_prompts.json)",
  "runtime": "transformers X / torch Y",
  "prompts": [ { "id": "MCP-TEST-01", "kind": "tool_call", "anchor": "tools/call",
    "input_ids": [151644, 872, ...], "max_new_tokens": 512,
    "ignore_eos": false, "expected": "{...minified...}", "raw": "" } ] }
```

Filename is `<model_type>.json` verbatim, matching `IModel::Type()`.

**No forward pass is taken.** `tools/mcp_capture.py` runs the source
checkpoint's *tokenizer and chat template* only: the expected payload is fixed
by the protocol, so running every source checkpoint (one of them 708 B
parameters) to obtain a string the spec already determines would cost days and
prove nothing. The captured `input_ids` are still the model's own, framed by the
same `fsi_capture` helper the FSI gate uses, so no HF↔logestix chat-template
parity is assumed and the converted model is replayed on the prompt the source
would have seen.

Provenance is declared the way the suite guard requires: every `raw` is empty
and `source` starts with `prompt-template`. A golden that one day carries a real
capture must change both — `test_model_suite_guard` fails a mismatch in either
direction. (That guard scans `fixtures/{fsi,code,det,mcp}`, so the day a model's
MCP case is captured from a running source the honest `raw` and the honest
`source` have to move together.)

## 5. Layer placement and wiring

- `libs/lsxcommon/{include/lsxcommon,src}/mcp_selfcheck.{h,cpp}` — host-only,
  model-neutral: the test owns the model and passes a `ModelRequest`; the
  harness never loads weights.
- `tests/test_parity_gate.cpp` — the shared driver gains a fourth `LSX_GATE_KIND`
  (`mcp`); the model row still comes from `models/<org>/<model>.json`.
- `models/<org>/<model>.json` — stage `mcp` after `det`, target
  `test_<family>_mcp_selfcheck`, `form: parity`, `kind: mcp`. The generator
  (`tools/gen_model_manifest.py`) owns the ctest registration; the executable
  is built by `lsx_add_parity_gate` (`cmake/LsxTestHelpers.cmake`).
- `tests/CMakeLists.txt` — the host unit test plus one
  `lsx_add_parity_gate(... KIND mcp)` line per model (48 rows).
- `apps/lsxbenchmark` — `--mcp-selfcheck`, `LSX_MCP_GOLDEN_DIR`, `kOff`.
- Guards extended in the same commit: `test_model_suite_guard` (stage order,
  required stage set, `SKIP_RETURN_CODE 4`, golden-or-declared-debt, and the
  template-provenance scan).

`SKIP` (exit 4) is not a pass: absent artifact or absent golden skips, and the
message names the model and the directory that was searched.

## 6. Anti-goals

- Not a live MCP client. The gate never opens a socket: it validates the
  payload a model emits against the JSON-RPC 2.0 contract offline, so it runs on
  a box with no MCP server (which is what this machine is — the session had no
  MCP server configured).
- Not a replacement for the native tool round-trip (ingestion stage 4b): that
  proves the model's *own* renderer/parser, this proves the wire format it is
  asked to produce.
- No new ctest label and no new lifecycle concept: `mcp` is a stage inside the
  existing `inference` label, exactly like `fsi`/`code`/`det`.

## 7. Revision note (2026-11-28b) — what implementing it changed

Five things were decided during the build, none of which the first draft had:

1. **Rollout is two models, not 47.** The gate ships on
   `microsoft/phi-4/nvfp4` and `Qwen/Qwen3.5-9B/nvfp4` — the pair whose
   artifacts this branch verified end to end. That choice is expressed in the
   machinery, not in prose (the original pilot gate has since closed — `mcp` is
   now in `kRequired` and out of `OPTIONAL_STAGES`, see the 2026-11-28 rollout
   archive). A row that does not
   declare the stage is neither a gap nor coverage; a row that does declare it
   gets the target check, the skip code, and the golden-or-debt check like every
   other stage. The alternative — a `golden_debt` entry per un-rolled-out row —
   would have been 45 fabricated declarations saying "we cannot capture this"
   about a gate that simply has not been rolled out.
2. **An `exact` flag, not an `exact` verdict.** Measured on the two pilots, both
   reproduce the template byte for byte — phi-4 inside a ```json fence,
   qwen3.5-9b bare. Under byte equality the fence would have been a FAIL for
   phi-4 and a PASS for qwen3.5-9b, i.e. the gate would have been measuring
   markdown habits. It is recorded and it gates nothing.
3. **The extractor was wrong for MCP's shapes and was fixed for everyone.**
   `code_selfcheck`'s local extractor took the FIRST balanced `{...}` and then
   rejected it if the anchor was missing — so a model that opens with a stray
   `{}` or a one-key illustration had a correct payload reported as an empty
   extraction. The shared `selfcheck::ExtractMinifiedJson` scans candidates for
   the anchor (and takes an optional `accept` predicate, which is how MCP
   refuses a request object as the answer to an error prompt). CODE keeps its
   behaviour for every existing golden; the improvement is that it no longer
   loses a payload behind a decoy.
4. **`Preprocess`, `CutPromptEcho` and `EscapeForLog` moved to
   `lsxcommon::selfcheck`.** Three harnesses had byte-identical copies; MCP
   would have been the fourth. CODE and DET now call the shared bodies, and the
   DET bug report about a newline truncating the one-line verdict is now
   structurally impossible in CODE as well (it was logging `raw` unescaped).
5. **Two guards in the tree corrected me, and both corrections are the point.**
   `test_json_escape` forbids a second hex table anywhere: MCP had grown one for
   canonical string escaping (the tree already has `lsxnetwork::json::JsonEscape`)
   and `EscapeForLog` had grown another. Both now use the one implementation.
   `test_index_layout` exempted three golden readers by FILE NAME, so the fourth
   gate tripped it on the day it landed; the exemption is now the class
   (`*_selfcheck.cpp`), which is what the comment always said it meant.

Not built, and deliberately: a live MCP client (§6), `Content-Length`/SSE
transport framing, and `tools/list`-style result shapes. All three would need a
server or a session to observe, and this gate's whole point is that it needs
neither.
