---
name: code-selfcheck
description: Use when capturing, adding, running or debugging CODESelfCheck in logestix — the compiler-syntactic-strictness parity gate (a separate Model Ingestion Protocol stage, not SelfCheck or FSISelfCheck). Covers tools/code_capture.py, the tests/fixtures/code golden schema (json/assignment/regex + body), exact-match extraction, the per-family test + CMake + suite-guard 'code' stage, lsxbenchmark --code-selfcheck and LSX_CODE_GOLDEN_DIR. Triggers - code_selfcheck, --code-selfcheck, code golden, syntactic strictness, code_capture.
---

# CODESelfCheck (compiler-syntactic-strictness parity gate)


> **Registration (2026-11-06):** the real-model target is still named
> `test_<family>_{fsi,code,det}_selfcheck` and still runs one per (model, stage),
> but its SOURCE is now the shared `tests/test_parity_gate.cpp`, compiled with
> `LSX_GATE_MODEL` / `LSX_GATE_KIND`. Do **not** create a per-family gate file:
> 105 of them were deleted because three shipped a SKIP naming a neighbour model,
> and the model row (vault dir, SKIP text, declared debt) now comes from
> `models/<org>/<model>.json`. See skill `ingesting-a-model` § "The manifest is the
> only registry you write by hand".

## Overview

CODESelfCheck is a **model-defined parity gate** and a first-class stage of the
Model Ingestion Protocol (step 8), parallel to FSISelfCheck but testing
*compiler-syntactic strictness* instead of financial formulas. It captures the
**original safetensors** model's own answer to a fixed 3-prompt code set,
freezes it as a committed golden, and requires the **converted** NVFP4 model to
reproduce the same **exactly normalized** answer *text* at `FastMathMode::kOff`.

**Core principle:** CODE is a separate functionality from `SelfCheck` **and** from
`fsi::RunFsiSelfCheck` — not a variant of either. Do not touch
`IModel::SelfCheck`, `runtime::RunParisSelfCheck`, `selfcheck_passed`,
`LSX_SELFCHECK_EXPECT_HASH` or `fsi::*`. It depends only on the neutral `IModel`
surface, `ArrowTokenizerDb` and `runtime::DecodeGeneratedText`.

The single source of truth for the design is
`docs/superpowers/specs/2026-09-27-code-selfcheck-design.md` (§3.4 decoupling
invariants, §4 extraction, §5 golden schema, §7 capture). Per-model capture
recipes and observed outcomes are in `docs/CODE_SELFCHECK_FINDINGS.md`.

## Relationship to the ingestion protocol

**REQUIRED SUB-SKILL:** `ingesting-a-model` (step 8, SelfCheck harness). CODE
extends step 8; the CODE test is a **new** target, never the existing
`test_<family>_selfcheck` or `test_<family>_fsi_selfcheck`.

Adding/adopting a model means extending `ingesting-a-model`'s step 8 with the
CODE row below, in the same commit as the model:

| Guard (in `tests/`) | What to extend | Rule |
|---|---|---|
| `test_model_suite_guard.cpp` | `Suites()`: add a `"code"` stage target per model | CODE lifecycle stage exists; `SKIP_RETURN_CODE 4` on `code` targets |
| `test_ctest_labels_guard.cpp` | label the new test `inference` | label in the same commit |
| `test_code_naming_guard.cpp` | — | `code_selfcheck.*` has no family stem (not scanned); test files are not scanned |

Do **not** add a new ctest label or lifecycle stage — `code` is a stage inside
the existing `inference` label (see `test-lifecycle`).

Full wiring is three files in the same commit: the golden, the test `.cpp`, and
the `tests/CMakeLists.txt` target.

## Decoupling invariants (CODE vs SelfCheck vs FSI)

These are invariants, not guidelines. Violating one is a design regression.

| Dimension | SelfCheck | FSISelfCheck | CODESelfCheck |
|---|---|---|---|
| Entry point | `IModel::SelfCheck` / `RunParisSelfCheck` | `fsi::RunFsiSelfCheck` | `code::RunCodeSelfCheck` |
| Input | hardcoded prompt | FSI golden `input_ids` | CODE golden `input_ids` |
| Reference | in-test literal | committed FSI golden | committed CODE golden |
| Criterion | top-1 / prefix (+logits hash) | **canonicalized** answer `==` golden | **exact** answer `==` golden |
| Result type | `gpu::InferenceResult.selfcheck_passed` | `fsi::FsiResult.all_passed` | `code::CodeResult.all_passed` |
| Numeric mode | `kAuto` | `kOff` | `kOff` |
| Env gates | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` | `LSX_CODE_GOLDEN_DIR` |
| CLI | `--selfcheck` | `--fsi-selfcheck` | `--code-selfcheck` |
| Host unit test | `tests/test_model_runtime.cpp` | `tests/test_fsi_selfcheck.cpp` | `tests/test_code_selfcheck.cpp` |
| Real-model test | `tests/test_<family>_selfcheck.cpp` | `tests/test_<family>_fsi_selfcheck.cpp` | `tests/test_<family>_code_selfcheck.cpp` |

**Exactness is the point.** Unlike FSI, CODE applies **no algebraic
canonicalization**: `m=(l+(h-l)/2)` fails against `m=l+((h-l)/2)`. A
syntactically different but equivalent answer is a failure by design.

## The CODE prompt set

`tools/code_prompts.json` (`format: logestix-code-prompts`) is the single source
of truth. The C++ side never embeds the bodies — the golden carries the captured
`input_ids` (and the `body` for the echo cut).

| id | kind | anchor | expected |
|---|---|---|---|
| CODE-TEST-01 | `json` | `root` | `{"root":{"nodes":{"leaf":true}}}` |
| CODE-TEST-02 | `assignment` | `m` | `m=l+((h-l)/2)` |
| CODE-TEST-03 | `regex` | `[0-9A-F]` | `\b[0-9A-F]{4}\b` |

## Workflow

1. **Capture** the golden from the source safetensors model.
2. **Commit** it under `tests/fixtures/code/<model_type>.json`.
3. **Add** `tests/test_<family>_code_selfcheck.cpp` + its `tests/CMakeLists.txt`
   target (label `inference`, `SKIP_RETURN_CODE 4`) and the suite-guard `code` row.
4. **Run** the target on the converted model; a failure is a real finding to
   investigate, not to mask.

### 1. Capture (`tools/code_capture.py`)

```bash
python tools/code_capture.py --selftest                 # no model; validates json/assignment/regex
python tools/code_capture.py --model <safetensors-dir|hf-id> \
    --type <model_type> --out tests/fixtures/code/<model_type>.json \
    [--device-map auto] [--max-new-tokens 512]
python tools/code_capture.py --model <dir> --type <type> --out <file> \
    --template-golden --max-new-tokens 512   # unrunnable source stopgap
```

- Greedy (`do_sample=False`), `--max-new-tokens` default 512, frames with
  `tokenizer.apply_chat_template(...)` when the tokenizer defines one.
- **If `tokenizer.chat_template` is unset, it loads `chat_template.jinja`** —
  required for granite/gemma/nemotron/GLM/Qwen; without it the raw body is
  encoded and the answer is malformed.
- Per-model quirks go in the `OVERRIDES` table keyed by `--type` (never
  implicit). Known ones: `granite-4.2-30b` →
  `{"enable_thinking": True, "low_effort": True}`;
  `qwen3.8-flash-next` → `{"loader": "qwen4_exp"}` (config allowlist + cache
  mapping + PLE shard fusion, same as the FSI capture; runs on CPU ~22 min).
- The golden `expected` is **model-defined**: the source model's own extracted
  answer, not the prompt template. A template difference is a warning, not a
  failure. Code-specific structural sanity must pass or capture aborts (json
  parses; extraction non-empty).
- Each case records the prompt **`body`** (used by the echo cut, see below).
- `input_ids` are captured **after** framing and fed verbatim by the verifier, so
  no HF↔logestix chat-template parity is assumed.

**When the source cannot run locally** (own FP8/FP4 stack, unknown arch, too big),
use `--template-golden`: frame each prompt with the tokenizer's chat template and
record the prompt's Expected Template as `expected` (no model run). This is a
deliberate deviation from the model-defined rule, forced by the unrunnable
source; it still guards the converted path (e.g. GLM-5.3-Flash). See
`docs/CODE_SELFCHECK_FINDINGS.md`.

### 2. Golden schema (`tests/fixtures/code/<model_type>.json`)

```json
{ "format": "logestix-code-golden", "version": 1, "model_type": "<Type()>",
  "source": "...", "runtime": "transformers X / torch Y",
  "prompts": [ { "id": "CODE-TEST-01", "kind": "json", "anchor": "root",
    "input_ids": [151644, 872], "max_new_tokens": 512,
    "body": "[CONTEXT]: CODE-Validation-Core-01 // ...",
    "expected": "{\"root\":{\"nodes\":{\"leaf\":true}}}", "raw": "..." } ] }
```

- Filename is `<model_type>.json` verbatim (dots allowed), matching `IModel::Type()`.
- `kind` ∈ {`json`,`assignment`,`regex`}; `anchor`, `input_ids`, `expected`
  non-empty; a malformed file is a parse error, never a silent skip.
- `body` (optional) is the prompt body used by `CutPromptEcho`; `ignore_eos`
  (optional) forces the full token budget — set it for reasoning models that
  never emit EOS.
- `raw` is diagnostic only.

### 3. Test + CMake + guard

- Copy the family's vault-path-helper **pattern** into a self-contained
  `tests/test_<family>_code_selfcheck.cpp`; call only the `code::` harness — its header and implementation are
  `libs/lsxcommon/include/lsxcommon/code_selfcheck.h` and
  `libs/lsxcommon/src/code_selfcheck.cpp` (host-only, model-neutral: the
  test owns the model and passes a `ModelRequest`; the harness never loads
  weights itself).
- Create the model with `cfg.fast_math_mode = kOff`.
- `SKIP` when the converted model **or** the golden is absent.
- CMake: `target_compile_definitions(... LSX_CODE_GOLDEN_DIR="${LSX_CODE_GOLDEN_DIR_FWD}")`,
  `add_test`, `set_tests_properties(... SKIP_RETURN_CODE 4)` and add the name to
  the `LABELS inference` list. Follow the existing CODE block in
  `tests/CMakeLists.txt`.
- `test_model_suite_guard.cpp::Suites()`: add the `"code"` stage entry.

### 4. Run

```bash
lsxbenchmark --model <vault-model-root> --code-selfcheck   # forces kOff
ctest --test-dir build/linux-debug -R "<family>_code_selfcheck" -j 1
LSX_CODE_GOLDEN_DIR=<dir> <test-binary>                    # ad-hoc golden
```

`lsxbenchmark` takes `--model <root>` (the vault model directory, not a type);
the type comes from the converted `model.index.json`. The CLI requires
`LSX_CODE_GOLDEN_DIR` (directory with `<model_type>.json`) and returns 0 iff
`all_passed`.

## Extraction contract

`ExtractCodeAnswer(kind, anchor, raw)` is pure host code
(`libs/lsxcommon/src/code_selfcheck.cpp`). Common preprocessing (same as FSI):
unfence ``` blocks (drop the language-tag line), then discard everything up to
and including the last `</think>`. Then per kind:

- **`json`** — take the first balanced `{...}` (tracking string literals/escapes)
  and minify whitespace outside strings; empty if unbalanced or the `anchor` is
  absent.
- **`assignment`** — first line where `anchor` is directly followed by `=`
  (no fallback: no `=` -> empty); take the maximal run over
  `[A-Za-z0-9_+\-*/().=]`, stopping at an adjacent identifier. Parentheses are
  preserved verbatim (no associativity rewriting).
- **`regex`** — first line containing `anchor`; expand left and right over the
  regex-token alphabet `[A-Za-z0-9\\\[\]{}()|.*+?^$/-]`, then strip whitespace.
  This captures the leading `\b` when the anchor is the inner `[0-9A-F]`.

### Prompt-echo cut (`CutPromptEcho`)

A fragile reasoning model can fail to emit a turn-end token and echo the prompt
after its answer, with no separator (e.g.
`\b[0-9A-F]{4}\b[CONTEXT]: ...`). When the golden carries `body`, the harness
cuts the decoded text at the body's **first 32 characters**, applied **after**
the `</think>` cut, before extraction. This is a harness bound, **not** a
relaxation of the exact comparison — a case with no `body` is unaffected.
Implemented in `RunCodeCase`; unit-tested in `tests/test_code_selfcheck.cpp`.

## Common mistakes

- Extending/changing `SelfCheck`, `RunParisSelfCheck` or `fsi::RunFsiSelfCheck`
  to "share" CODE work (violates every decoupling invariant).
- Forgetting `SKIP_RETURN_CODE 4` (a missing weight/golden then fails instead of
  skipping) or the `inference` label; forgetting the suite-guard `code` row.
- Reusing `test_<family>_selfcheck.cpp` / `test_<family>_fsi_selfcheck.cpp` or
  their targets for CODE.
- Encoding the prompt with the raw body because `chat_template.jinja` was not
  loaded (granite/gemma/nemotron/GLM/Qwen).
- Expecting the prompt template to equal the golden for a captured model — the
  golden is the source model's own answer (only `--template-golden` records the
  template).
- Running the verifier at `kAuto` or comparing logits/timings; CODE compares
  answer text at `kOff`.
- Setting the token budget too small for a reasoning model: an answer cut
  mid-reasoning looks like a divergence but is only truncation (raise
  `max_new_tokens`; GLM needed 512). Confirm against the full `raw`.
- Confusing a harness/extraction artifact with a model defect. A blank or
  over-long extraction (e.g. `[CONTEXT]` glued on) is usually the echo cut, not
  a kernel bug; an argmax flip on a rare token (granite `}}}` -> `}}`) is a real
  quantization divergence to investigate, not to loosen the gate for.

## Verification

```bash
ctest --preset linux-debug -R test_code_selfcheck                 # host unit tests (fast)
ctest --preset linux-debug-inference -R "_code_selfcheck"         # EVERY model: infra only
ctest --preset linux-debug -R "test_model_suite_guard|test_ctest_labels_guard"
```

Green means: host unit tests pass, the family target PASSes on the converted
model, and both guards are green. Findings (capture feasibility, per-model
recipes, resolved divergences) are recorded in
`docs/CODE_SELFCHECK_FINDINGS.md`.
