# Skill: det-selfcheck

# DETSelfCheck (determinism stress-test + near-tie golden gate)

## Overview

DETSelfCheck is a **model-defined parity gate** and a first-class stage of the
Model Ingestion Protocol (step 8), sitting beside FSISelfCheck and
CODESelfCheck. It captures the original safetensors model's own answer to a
fixed 3-prompt **near-tie / symmetric** set, freezes it as a committed golden,
and requires the converted NVFP4 model to satisfy two properties at
`FastMathMode::kOff`:

1. **Determinism stress-test** — repeated greedy decoding of the same prompt
   produces a **byte-identical generated token stream** (zero variance).
2. **Near-Tie filtering** — the ambiguous prompt resolves, greedily, to the
   golden canonical token.

**Core principle:** DET is a separate functionality from `SelfCheck`,
`FSISelfCheck` and `CODESelfCheck`. Do not touch `IModel::SelfCheck`,
`runtime::RunParisSelfCheck`, `fsi::RunFsiSelfCheck` or
`code::RunCodeSelfCheck`, and never read `selfcheck_passed` /
`LSX_SELFCHECK_EXPECT_HASH`. It depends only on the neutral `IModel` surface,
`ArrowTokenizerDb` and `runtime::DecodeGeneratedText`.

The single source of truth for the design is
`docs/superpowers/specs/2026-09-27-det-selfcheck-design.md`. Per-model capture
recipes and observed outcomes are in `docs/DET_SELFCHECK_FINDINGS.md`.

## Relationship to the ingestion protocol

**REQUIRED SUB-SKILL:** `ingesting-a-model` (step 8, SelfCheck harness). DET
extends step 8; the DET test is a **new** target, never an existing
`test_<family>_selfcheck`.

| Guard (in `tests/`) | What to extend | Rule |
|---|---|---|
| `test_model_suite_guard.cpp` | `Suites()`: add a `"det"` stage per model | lifecycle stage exists; `SKIP_RETURN_CODE 4` on `det` targets |
| `test_ctest_labels_guard.cpp` | label the new test `inference` | label in the same commit |
| `test_code_naming_guard.cpp` | — | `det_selfcheck.*` has no family stem (not scanned); test files are not scanned |

Do **not** add a new ctest label — `det` is a stage inside the existing
`inference` label.

Full wiring is three files in the same commit: the golden, the test `.cpp`, and
the `tests/CMakeLists.txt` target.

## Decoupling invariants (DET vs SelfCheck/FSI/CODE)

These are invariants, not guidelines.

| Dimension | SelfCheck | FSI | CODE | DET |
|---|---|---|---|---|
| Entry point | `IModel::SelfCheck` | `fsi::RunFsiSelfCheck` | `code::RunCodeSelfCheck` | `det::RunDetSelfCheck` |
| Criterion | top-1 text (+hash) | canonical answer `==` golden | normalized answer `==` golden | token-stream identity across R runs **and** answer `==` golden |
| Result type | `gpu::InferenceResult.selfcheck_passed` | `fsi::FsiResult` | `code::CodeResult` | `det::DetResult` |
| Numeric mode | `kAuto` | `kOff` | `kOff` | `kOff` |
| Env gates | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` | `LSX_CODE_GOLDEN_DIR` | `LSX_DET_GOLDEN_DIR` (+ `LSX_DET_STRESS_RUNS`) |
| CLI | `--selfcheck` | `--fsi-selfcheck` | `--code-selfcheck` | `--det-selfcheck` |
| Host unit test | `tests/test_model_runtime.cpp` | `tests/test_fsi_selfcheck.cpp` | `tests/test_code_selfcheck.cpp` | `tests/test_det_selfcheck.cpp` |
| Real-model test | `test_<family>_selfcheck.cpp` | `test_<family>_fsi_selfcheck.cpp` | `test_<family>_code_selfcheck.cpp` | `test_<family>_det_selfcheck.cpp` |
| ctest stage | `selfcheck` | `fsi` | `code` | `det` |

## The DET prompt set

`tools/det_prompts.json` (`format: logestix-det-prompts`) is the single source of
truth. The C++ side never embeds the bodies — the golden carries the captured
`input_ids`.

| id | title | canonical answer |
|---|---|---|
| DET-TEST-01 | PARADOXICAL-LOGIC-ANCHOR | `PARADOX_ANTINOMY` |
| DET-TEST-02 | SYNONYM-COLLAPSE-PRESSURE | `CATASTROPHIC` |
| DET-TEST-03 | QUANTUM-STATE-DETERMINISM | `BIT_ZERO` |

All cases use `kind: "token"`.

## Workflow

1. **Capture** the golden from the source safetensors model.
2. **Commit** it under `tests/fixtures/det/<model_type>.json`.
3. **Add** `tests/test_<family>_det_selfcheck.cpp` + its `tests/CMakeLists.txt`
   target (label `inference`, `SKIP_RETURN_CODE 4`) and the suite-guard `det`
   row.
4. **Run** the target on the converted model; a `deterministic=no` or a mismatch
   is a real finding to investigate, not to mask.

### 1. Capture (`tools/det_capture.py`)

```bash
python tools/det_capture.py --selftest                 # no model; validates the set
python tools/det_capture.py --model <safetensors-dir|hf-id> \
    --type <model_type> --out tests/fixtures/det/<model_type>.json
python tools/det_capture.py --template-golden \
    --model <dir> --type <t> --out <o>
```

- Greedy (`do_sample=False`), `max_new_tokens` default 512 (matching FSI), frames
  with `tokenizer.apply_chat_template(...)` when the tokenizer defines one.
  Reasoning models whose chat template forces a `<think>` preamble (e.g.
  GLM-5.3-Flash) need a large budget or the answer is never reached and the
  extractor latches onto the reasoning (see
  `docs/DET_SELFCHECK_FINDINGS.md`).
- **If `tokenizer.chat_template` is unset, it loads `chat_template.jinja`** —
  required for granite/gemma/nemotron/GLM/Qwen.
- Per-model quirks go in the `OVERRIDES` table keyed by `--type`: known ones are
  `granite-4.2-30b` → `{"enable_thinking": True, "low_effort": True}` and
  `qwen3.8-flash-next` → `{"loader": "qwen4_exp"}`.
- The golden `expected` is **model-defined** (the source model's own extracted
  token); a template difference is a warning, not a failure. Capture aborts on an
  empty extraction.
- `--template-golden`: tokenizer-only fallback for an unrunnable source; records
  the prompt Expected Template and marks the deviation in `source`.
- `input_ids` are captured after framing and fed verbatim by the verifier.

### 2. Golden schema (`tests/fixtures/det/<model_type>.json`)

```json
{ "format": "logestix-det-golden", "version": 1, "model_type": "<Type()>",
  "source": "...", "runtime": "transformers X / torch Y",
  "prompts": [ { "id": "DET-TEST-01", "kind": "token",
    "input_ids": [151644, 872, ...], "max_new_tokens": 64,
    "stress_runs": 3, "body": "[CONTEXT]: ...", "expected": "PARADOX_ANTINOMY",
    "raw": "..." } ] }
```

- Filename is `<model_type>.json` verbatim (dots allowed), matching
  `IModel::Type()`.
- `kind` must be `token`; `input_ids` and `expected` non-empty; a malformed file
  is a parse error, never a silent skip.
- `stress_runs` (optional, default 3, must be ≥ 1) is the repetition count.
- `body` (optional) feeds the prompt-echo cut; `ignore_eos` (optional) forces the
  full token budget.

### 3. Test + CMake + guard

- Copy the family's vault-path-helper **pattern** into a self-contained
  `tests/test_<family>_det_selfcheck.cpp`; call only the `det::` harness.
- Create the model with `cfg.fast_math_mode = kOff`.
- `SKIP` when the converted model **or** the golden is absent.
- CMake: `target_compile_definitions(... LSX_DET_GOLDEN_DIR="${LSX_DET_GOLDEN_DIR_FWD}")`,
  `add_test`, `set_tests_properties(... SKIP_RETURN_CODE 4)` and add the name to
  the `LABELS inference` list.
- `test_model_suite_guard.cpp::Suites()`: add the `"det"` stage entry.

### 4. Run

```bash
lsxbenchmark --model <vault-model-root> --det-selfcheck   # forces kOff
ctest --preset linux-debug-inference -R "_det_selfcheck"
LSX_DET_STRESS_RUNS=5 LSX_DET_GOLDEN_DIR=<dir> <test-binary>
```

`lsxbenchmark` takes `--model <root>` (the vault model directory); the type comes
from the converted `model.index.json`.

## Extraction contract

`ExtractDetAnswer(DetKind::kToken, raw)` is pure host code
(`libs/lsxcommon/src/det_selfcheck.cpp`):

- Unfence ``` blocks; discard everything before the last `</think>`.
- Return the **first standalone token** matching `[A-Z][A-Z0-9_]*` with length
  ≥ 2 (characters before/after are not `[A-Za-z0-9_]`), uppercased; empty if
  none.
- The harness applies `CutPromptEcho` (body's first 32 chars, after the
  `</think>` cut) before extraction when the golden carries `body`.

**Fragility to know:** an echoed prompt (`DET-TEST-01`) can yield the standalone
`DET` token; the `body` cut suppresses this. On a mismatch, inspect `raw` before
assuming a kernel bug.

## Common mistakes

- Extending/changing `SelfCheck`/FSI/CODE to "share" DET work (violates every
  decoupling invariant).
- Forgetting `SKIP_RETURN_CODE 4` or the `inference` label; forgetting the
  suite-guard `det` row.
- Encoding the prompt with the raw body because `chat_template.jinja` was not
  loaded.
- Running the verifier at `kAuto`; DET compares token streams at `kOff`.
- Treating a real nondeterminism/divergence as a gate artifact — investigate the
  runtime/model; do not loosen the gate.

## Verification

```bash
ctest --preset linux-debug -R test_det_selfcheck                 # host unit tests (fast)
ctest --preset linux-debug-inference -R "_det_selfcheck"         # all real-model DET
ctest --preset linux-debug -R "test_model_suite_guard|test_ctest_labels_guard"
python3 tools/det_capture.py --selftest
```

Green means: host unit tests pass, the family target PASSes on the converted
model, both guards are green, and the capture self-test passes. Findings are
recorded in `docs/DET_SELFCHECK_FINDINGS.md`.
