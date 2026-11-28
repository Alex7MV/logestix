# Skill: mcp-selfcheck

# MCPSelfCheck (MCP / JSON-RPC 2.0 payload parity gate)

> **Registration:** the real-model target is `test_<family>_mcp_selfcheck` and its
> SOURCE is the shared `tests/test_parity_gate.cpp` compiled with
> `LSX_GATE_MODEL` / `LSX_GATE_KIND=mcp`. Do **not** create a per-family gate
> file. The model row (artifact dir, SKIP text) comes from
> `models/<org>/<model>.json`, and the build half is one call:
> `lsx_add_parity_gate(NAME … MODEL … KIND mcp)` (`cmake/LsxTestHelpers.cmake`).
>
> **Status: a required lifecycle stage** (operator directive, 2026-11-28), with
> the same standing as SelfCheck/FSI/CODE/DET. Every manifest row declares the
> stage and has a target; `mcp` is in `test_model_suite_guard`'s `kRequired` and
> no longer in the generator's `OPTIONAL_STAGES`. A row whose framed prompt could
> not be captured declares `golden_debt.mcp` — a measured, named gap, never
> silence. See `Rollout` below for how the roll-out was completed.

## Overview

MCPSelfCheck asks a converted model for the three JSON-RPC 2.0 payloads an agent
harness has to be able to emit — a `tools/call` request, a `resources/read`
request and a standard `-32602` error response — and compares them with a
committed golden.

Design: `docs/superpowers/specs/2026-11-28-mcp-selfcheck-design.md`.
Observed results: `docs/MCP_SELFCHECK_FINDINGS.md`.

**Core principle:** MCP is a separate functionality from SelfCheck, FSI, CODE and
DET. Do not touch `IModel::SelfCheck`, `runtime::RunParisSelfCheck`,
`selfcheck_passed`, `LSX_SELFCHECK_EXPECT_HASH`, or the other three harnesses'
prompt sets. The MCP harness depends only on the neutral `IModel` surface,
`ArrowTokenizerDb` and `runtime::DecodeGeneratedText`.

## The criterion — read this before changing anything

**The golden is protocol-defined, not model-defined.** JSON-RPC 2.0 fixes the
payload, so `expected` comes from `tools/mcp_prompts.json` and no source model is
run. That is the opposite of FSI/CODE/DET, and it is why:

- the comparison is **JSON document equality**, not text equality: extract the
  first balanced object carrying the case's anchor, minify, **canonicalize**
  (keys sorted), compare. Key order and insignificant whitespace are free;
  every key, every value and the protocol version are not;
- `exact` is recorded per case (did the model also match the template byte for
  byte) and gates nothing. Measured: phi-4 reproduces the template exactly,
  qwen3.5-9b emits the same payload with no code fence, and both pass;
- the failure verdict names the field: `diff='error.code: -32601 != -32602'`.
  If you ever find yourself comparing bytes here, you are measuring which
  models guessed the template's layout.

## The prompt set

`tools/mcp_prompts.json` (`format: logestix-mcp-prompts`) is the single source of
truth. C++ never embeds the bodies — the golden carries the captured `input_ids`.

| id | kind | anchor | shape |
|---|---|---|---|
| MCP-TEST-01 | `tool_call` | `tools/call` | request with nested `arguments` |
| MCP-TEST-02 | `resource_read` | `resources/read` | request with a `uri` |
| MCP-TEST-03 | `error_response` | `Invalid params` | `error.code = -32602`, `id = 99` |

A `kind` is also a SHAPE filter: the extractor will not accept a `method` object
as the answer to `error_response`. That is what stops a reasoning model's
illustrative request from being adopted as its error response.

## Workflow — adding a model

1. **Capture** the framed prompt (tokenizer only, no forward pass):
   ```bash
   python3 tools/mcp_capture.py --model <checkpoint-dir> \
       --type <model_type> --out tests/fixtures/mcp/<model_type>.json
   python3 tools/mcp_capture.py --selftest      # validates the prompt set
   ```
   Framing facts (chat-template kwargs, vendor system turns, DeepSeek release
   encoders) come from `tools/fsi_capture.py`'s `OVERRIDES` — one table for every
   gate. If a model needs a quirk, add it THERE, not in a second copy.
2. **Manifest**: add the stage + the override to `models/<org>/<model>.json`:
   ```json
   "stages": { "mcp": ["test_<family>_mcp_selfcheck"] },
   "target_overrides": {
     "test_<family>_mcp_selfcheck": { "form": "parity", "kind": "mcp" }
   }
   ```
3. **CMake**: one line, with the other rows in `tests/CMakeLists.txt`:
   ```cmake
   lsx_add_parity_gate(NAME test_<family>_mcp_selfcheck MODEL <type> KIND mcp)
   ```
   The `add_test`/properties half is emitted into `model_targets.cmake`; label
   `inference`, `SKIP_RETURN_CODE 4` and `RESOURCE_LOCK "gpu"` come from the
   manifest, so do not restate them here.
4. **Regenerate**: `python3 tools/gen_model_manifest.py --docs docs/MODELS.md --root .`
5. **Run**: `ctest --test-dir build/<preset> -R "_mcp_selfcheck" -j 1`.

A FAIL is a real finding. Investigate the model, do not loosen the criterion —
and do not delete the case.

## Reading a failure

| log field | meaning |
|---|---|
| `diff=''`, `passed=true`, `exact=true` | byte-identical to the template |
| `diff=''`, `passed=true`, `exact=false` | same document, different key order/formatting — **a pass** |
| `diff='error.code: … != …'` | wrong protocol field; the model is the finding |
| `diff='params: missing key node_id'` | wrong payload contents |
| `diff='no JSON-RPC tool_call payload carrying …'` | nothing matched kind + anchor; read `raw` (fence? truncation? budget?) |
| `error='…'` | inference failed; not a gate verdict |

`raw` is the whole decoded tail. The budget is 512 tokens because reasoning
models spend tokens before the payload; if `raw` ends mid-thought, raise
`max_new_tokens` in the prompt set capture, not the gate.

## Common mistakes

- Comparing bytes (or the minified text) instead of canonical documents.
- Letting `exact` become the verdict — it is a measurement, not a gate.
- Adding a per-family `test_<family>_mcp_selfcheck.cpp`. The driver is shared.
- Forgetting `"kind": "mcp"` in `target_overrides` (the target then builds
  without a kind and the driver refuses at compile time — the generator also
  rejects an unknown kind at validate time).
- Claiming the gate covers 47 models. It covers two; the manifest is the truth.
- Treating this gate as the native tool round-trip (ingestion stage 4b). That
  proves the model's own renderer/parser; this proves the wire format.

## Rollout

The roll-out is **complete** (2026-11-28). What used to be a two-row pilot is now
the required lifecycle stage, and the machinery says so rather than a comment:

- `tools/gen_model_manifest.py`: `OPTIONAL_STAGES = ("perf",)` — only `perf` is
  relaxed (a baseline exists once a model is measured). `mcp` is not.
- `tests/test_model_suite_guard.cpp`: `kRequired` contains `"mcp"`, so dropping
  the stage fails the same test as dropping `selfcheck`.
- the golden-coverage guard already listed `mcp` in `kGoldenStages`, so the
  two-way rule now applies to a required stage: an **undeclared** gap fails, and
  a **stale** debt row fails.

Current state: 48 rows declare the stage, 48 targets exist (one
`lsx_add_parity_gate ... KIND mcp` each), 39 goldens are captured and 9 rows
carry `golden_debt.mcp`. Those nine are the empty source-store placeholders
(`minimax-m3`, `flint`, `mimo-v2.6-flash-rl`, `ling`, `mistral-large-4-1t`,
`kimi-k3`, `beam`, `step-5-preview`, `glm-5.2`): no weights, no artifact, so the
`input_ids` cannot be produced. Re-capture is one command when a checkpoint
returns:

```bash
LSX_SOURCE_ROOT=<store> python3 tools/mcp_capture_all.py          # every row
LSX_SOURCE_ROOT=<store> python3 tools/mcp_capture_all.py <model>  # one row
```

Adding a model to the gate is therefore NOT a rollout decision any more — a new
manifest row without `stages.mcp` is a broken suite, exactly like one without
`selfcheck`. Capture the golden while the checkpoint is on disk; if it genuinely
cannot be captured, declare `golden_debt.mcp` with the reason in the same commit.

## Verification

```bash
ctest --test-dir build/linux-debug -R test_mcp_selfcheck          # host criterion
ctest --test-dir build/linux-debug -R "_mcp_selfcheck" -j 1       # real weights
ctest --test-dir build/linux-debug -R "test_model_suite_guard|test_manifest_sync|test_ctest_labels_guard"
python3 tools/mcp_capture.py --selftest
```

Green means: the host criterion test passes, every row that has a converted
artifact PASSes its real-weight target, every row without one SKIPs (exit 4, and
its message names the model and the directory), and the three guards are green.
