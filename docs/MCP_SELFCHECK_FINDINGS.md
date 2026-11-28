# MCPSelfCheck — capture and parity findings

- **Date**: 2026-11-28
- **Scope**: MCPSelfCheck
  (`docs/superpowers/specs/2026-11-28-mcp-selfcheck-design.md`)
- **Status**: mechanism complete, and (operator directive, 2026-11-28) a
  **required lifecycle stage** with the same standing as SelfCheck/FSI/CODE/DET:
  all 48 manifest rows declare the stage and have a target, 39 goldens are
  captured, and 9 rows declare `golden_debt.mcp` because their source directory
  is an empty placeholder and no artifact exists. Three rows have run on real
  weights — `microsoft/phi-4/nvfp4`, `Qwen/Qwen3.5-9B/nvfp4` and
  `google/gemma-4-12B-it/nvfp4` — all 3/3 byte-exact. See "Rollout" below.

The gate is **protocol-defined**, and that is its one real departure from
FSI/CODE/DET: JSON-RPC 2.0 fixes the payload, so `expected` comes from
`tools/mcp_prompts.json` and no source model is executed. What a model-specific
golden still carries is the framed prompt — `input_ids` from that checkpoint's
own tokenizer and chat template, through the same `fsi_capture` helper the FSI
gate uses.

## The prompt set

| id | kind | anchor | payload |
|---|---|---|---|
| MCP-TEST-01 | `tool_call` | `tools/call` | tool invocation with a nested `arguments.node_id` |
| MCP-TEST-02 | `resource_read` | `resources/read` | a resource read bounded by a `schema://` URI |
| MCP-TEST-03 | `error_response` | `Invalid params` | standard error `-32602` on session id 99 |

## The criterion, and why not bytes

The harness extracts the first balanced object carrying the case's anchor,
minifies it, **canonicalizes** it (keys sorted, strings escaped canonically) and
compares documents. A missing key, an extra key, a wrong `id`, a wrong error
code, a wrong protocol version all fail; key order and insignificant whitespace
do not, because JSON declares them insignificant.

`exact` records whether the extracted text also matched the template byte for
byte. It is logged and it gates nothing, so "how close to the template is this
model" stays measurable without becoming a verdict.

**Measured, and this is the evidence the choice was right rather than convenient:**

| probe | result |
|---|---|
| phi-4 golden as committed | 3/3 pass, `exact=true` — the model reproduces the template byte for byte, markdown fence and all |
| qwen3.5-9b golden as committed | 3/3 pass, `exact=true`, and it emits the payload with **no fence at all** — the two models disagree on formatting and both are accepted |
| phi-4 golden with every `expected` re-keyed in reverse | 3/3 **pass**, `exact=false` on all three. Byte equality would have failed all three |
| phi-4 golden with `-32601` substituted for `-32602` | **FAILS** on MCP-TEST-03 with `diff='error.code: -32601 != -32602'` |

So the gate is neither vacuous (a one-token protocol error is caught and named)
nor a layout lottery (a legitimate re-keying is not).

## Capture feasibility

`tools/mcp_capture.py` needs a tokenizer and a chat template, not weights, so
the usual "the 708B source cannot run locally" problem does not arise. Two
obstacles were real and are now recorded rather than rediscovered:

- **Python environment.** `transformers` lives in the system `python3`
  (`~/.local/lib/python3.14/site-packages`), which is PEP-668 managed. The
  tokenizer conversion of a SentencePiece checkpoint needs `sentencepiece` and
  `tiktoken`, and `alice-ai`'s `tokenizer.model` additionally needs `protobuf`
  (without it transformers falls back to a TikToken extractor that cannot parse
  the file and reports `Error parsing line b'\x0e'`). Installed with
  `pip install --user --break-system-packages sentencepiece tiktoken protobuf`.
- **Empty placeholder directories.** The org-nested source store contains rows
  whose weights were never downloaded; `AutoTokenizer` then fails with a
  message about SentencePiece that has nothing to do with the real cause.
  `tools/mcp_capture_all.py` now reports "empty placeholder (weights not
  downloaded)" instead.

**Measured tally, verified per row rather than inferred from the log:** the first
sweep captured 37 of 47. Of the ten that failed, **nine have no weights on disk
at all** (`beam`, `flint`, `glm-5.2`, `kimi-k3`, `ling`, `mimo-v2.6-flash-rl`,
`minimax-m3`, `mistral-large-4-1t`, `step-5-preview`: their org-nested source
directory exists and is empty, which is why the tokenizer error was about
SentencePiece — there is no `tokenizer.model` to read). Only `alice` had a real
checkpoint and failed on the missing `protobuf`. After installing it, `alice`
captured cleanly.

So the honest statement is not "nine unmeasured" but: **every row that has
weights on disk has a working capture, 38 of 38** (37 + `alice`), and the nine
without are blocked on downloads, not on this tool. `tools/mcp_capture_all.py`
now says so explicitly instead of echoing a Python-package error.

## Rollout

The stage is wired in `models/<row>.json` → `model_targets.cmake` → ctest, and
the build half is one call per model:

```cmake
lsx_add_parity_gate(NAME test_phi_dense_mcp_selfcheck MODEL phi-4 KIND mcp)
lsx_add_parity_gate(NAME test_qwen_dense_mcp_selfcheck MODEL qwen3.5-9b KIND mcp)
```

`mcp` **is** in `test_model_suite_guard`'s `kRequired` and is no longer in
`gen_model_manifest.py`'s `OPTIONAL_STAGES` (operator directive, 2026-11-28: it
is a lifecycle gate, not a pilot). Every row declares the stage, every row has a
target from one `lsx_add_parity_gate(... KIND mcp)` line, and the golden-coverage
guard applies to it unchanged — an undeclared gap fails, a stale debt row fails.

The roll-out is recorded in
`docs/open-tasks/2026-11-28-mcp-gate-rollout-closed.md`, including the reason
each of the nine debt rows cannot be captured today (empty source-store
placeholder, no artifact) and the one-command re-capture once a checkpoint
returns.

## Anti-goals, restated because they are easy to erode

- **Not a live MCP client.** No socket is opened. The payload a model emits is
  validated against the JSON-RPC 2.0 contract offline. This machine has no MCP
  server configured (`list_mcp_resources` returned an empty set), and the gate
  is designed so that it never needs one.
- **Not the tool round-trip.** Ingestion stage 4b proves the model renders and
  parses its OWN chat template's tool syntax. This proves the wire format it is
  asked to produce. A model can pass one and fail the other, and both failures
  are informative.
- **Not a fourth copy of the extractor.** `Preprocess`, `CutPromptEcho`,
  `ExtractJson` and `EscapeForLog` moved into `lsxcommon::selfcheck`
  (`selfcheck_common.{h,cpp}`) and CODE/DET/MCP now share them. The JSON
  extractor also gained the anchor-across-candidates scan, which fixed a real
  latent defect: it used to take the FIRST balanced object and then fail the
  anchor test, so a correct payload behind a stray `{}` read as an empty
  extraction.
