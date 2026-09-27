# CODESelfCheck (Compiler Syntactic Strictness Self-Check) Design

- **Date**: 2026-09-27
- **Status**: Implemented (TDD)
- **Branch**: `feat/code-selfcheck`
- **Related**: `docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`,
  `libs/lsxcommon/src/fsi_selfcheck.cpp`, Model Ingestion Protocol step 8
  (`ingesting-a-model`)

## 1. Purpose

Add a deterministic, model-parity gate keyed to a fixed **compiler-syntax**
prompt set, exactly parallel to FSISelfCheck but testing token-level syntactic
strictness (JSON minification, C assignment syntax, PCRE escaping) instead of
financial formulas.

**CODESelfCheck is a separate functionality from both `SelfCheck` and
`FSISelfCheck`.** It shares no API, state, golden, environment variable, result
type or ctest target with either. It depends only on the model-neutral
`IModel` surface (`Load`/`Infer`/`Unload`/`Type`/`Tokenizer`) and never calls
`IModel::SelfCheck`, `runtime::RunParisSelfCheck`, `fsi::RunFsiSelfCheck`, or
reads `selfcheck_passed` / `LSX_SELFCHECK_EXPECT_HASH`.

The gate is:

1. **Capture**: run the CODE prompt set on the original **safetensors** model
   (via HF transformers) and freeze the extracted answer for that model as its
   golden reference.
2. **Verify**: run the same prompts on the **converted** logestix (NVFP4) model
   and require the extracted answer to match the golden **byte-for-byte** after
   normalization.
3. **GPU-independent result**: the pass/fail criterion is the normalized answer
   *text*, never logits, hashes or timings.

## 2. Scope

In scope:

- The three-prompt CODE set (single source of truth).
- A host-only, model-neutral capture/verify contract in `lsxcommon`.
- A Python capture script (HF transformers, generic + per-model overrides).
- Committed per-model golden JSON under `tests/fixtures/code/`.
- Host-only unit tests (fake model, no GPU) and separate per-family real-model
  tests.
- A separate `lsxbenchmark --code-selfcheck` entry point.
- A `code` lifecycle stage in `test_model_suite_guard.cpp`.

Out of scope:

- Changing `SelfCheck` / `FSI` semantics or targets.
- Running raw safetensors models inside logestix.
- Any speed/perf behaviour.
- Adding a new ctest label (stays `inference`; `code` is a stage).

## 3. Contracts

### 3.1 CODE prompt set (single source of truth)

The set lives in `tools/code_prompts.json`, read by the Python capture script.
The C++ side does **not** embed the prompt bodies: the golden carries the
captured `input_ids`.

| id | title | kind | anchor | expected |
|---|---|---|---|---|
| CODE-TEST-01 | ABSTRACT-SYNTAX-TREE-NESTING | `json` | `root` | `{"root":{"nodes":{"leaf":true}}}` |
| CODE-TEST-02 | ALGORITHMIC-RECURSION-BLOCK | `assignment` | `m` | `m=l+((h-l)/2)` |
| CODE-TEST-03 | REGEX-ESCAPE-SEQUENCE | `regex` | `[0-9A-F]` | `\b[0-9A-F]{4}\b` |

### 3.2 Verify contract

- For each golden case: `model.Infer(ModelRequest{input_ids, max_gen_tokens})`
  with `input_ids`/`max_gen_tokens` from the golden (greedy, eos-terminated).
- Decode with `runtime::DecodeGeneratedText`.
- If the golden carries the prompt `body`, cut the decoded text at the body's
  first 32 characters (after the `</think>` cut) — a fragile reasoning model can
  fail to emit a turn-end token and echo the prompt after its answer; the cut
  stops that echo from bleeding into the extracted answer. No `body` -> no cut.
- Extract with `ExtractCodeAnswer(kind, anchor, decoded)`.
- Pass iff the extracted string equals the golden `expected` **exactly**.

### 3.3 GPU independence / numeric mode

- The pass criterion compares normalized text only.
- Verify runs at `FastMathMode::kOff`; the CLI forces `kOff` for
  `--code-selfcheck`.
- Extraction/harness are host-only and unit-tested without a GPU.

### 3.4 Decoupling invariants (CODESelfCheck independent)

These are invariants, not guidelines.

| Dimension | SelfCheck | FSISelfCheck | CODESelfCheck |
|---|---|---|---|
| Entry point | `IModel::SelfCheck`, `runtime::RunParisSelfCheck` | `fsi::RunFsiSelfCheck` | `code::RunCodeSelfCheck` |
| Input | hardcoded `"The capital of France is"` | FSI golden `input_ids` | CODE golden `input_ids` |
| Reference | in-test literal `"Paris"` | committed FSI golden JSON | committed CODE golden JSON |
| Criterion | top-1 / short-prefix text (+logits hash) | canonicalized answer `==` golden | normalized answer `==` golden (exact) |
| Result type | `gpu::InferenceResult.selfcheck_passed` | `fsi::FsiResult.all_passed` | `code::CodeResult.all_passed` |
| Numeric mode | `kAuto` | `kOff` | `kOff` |
| Env gates | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` | `LSX_CODE_GOLDEN_DIR` |
| CLI | `--selfcheck` | `--fsi-selfcheck` | `--code-selfcheck` |
| Host unit test | `tests/test_model_runtime.cpp` | `tests/test_fsi_selfcheck.cpp` | `tests/test_code_selfcheck.cpp` |
| Real-model test | `tests/test_<family>_selfcheck.cpp` | `tests/test_<family>_fsi_selfcheck.cpp` | `tests/test_<family>_code_selfcheck.cpp` |
| ctest stage | `selfcheck` | `fsi` | `code` |

Shared, intentionally: the neutral `IModel` inference surface, `ArrowTokenizerDb`
and `runtime::DecodeGeneratedText`. Nothing else.

## 4. Extraction / normalization contract

`ExtractCodeAnswer(CodeKind kind, std::string_view anchor, std::string_view raw)`
is a pure host function.

Common preprocessing (identical to FSI):

1. Trim ASCII whitespace.
2. If a fenced block exists, keep the first block's content; drop a leading
   language-tag line matching `^[A-Za-z0-9_+-]+$`.
3. Discard everything up to and including the last `</think>`.

The harness additionally applies `CutPromptEcho` before extraction when the
golden carries `body` (see 3.2): after step 3, if the body's first 32 characters
occur in the text, the text is truncated there. This is not part of
`ExtractCodeAnswer` (a pure function of `kind`/`anchor`/`raw`).

Then per kind:

- **`json`**: locate the first `{` in the text; scan to the matching `}` while
  tracking string literals and backslash escapes (so braces inside strings do
  not confuse depth); minify the substring by removing whitespace **outside**
  string literals. Empty if no brace or unbalanced. If `anchor` is non-empty the
  result must contain it, else empty.
- **`assignment`**: select the first line where `anchor` is directly followed
  (spaces allowed) by `=`; take the maximal run starting at `anchor` over
  `[A-Za-z0-9_+\-*/().=]` with the FSI truncation rule (stop at an adjacent
  identifier / depth-0 operand boundary). Parentheses are preserved verbatim (no
  associativity rewriting). Whitespace is stripped.
- **`regex`**: select the first line containing `anchor`; start at the anchor and
  expand left and right over the regex-token alphabet
  `[A-Za-z0-9\\\[\]{}()|.*+?^$/-]` (left expansion stops at whitespace or a
  non-token char), then strip whitespace. This captures the leading `\b` when
  the anchor is the inner `[0-9A-F]`.

Canonical outputs (the golden `expected` is produced by the same normalization,
so a correctly-trained model reproduces it verbatim):

| id | extracted |
|---|---|
| CODE-TEST-01 | `{"root":{"nodes":{"leaf":true}}}` |
| CODE-TEST-02 | `m=l+((h-l)/2)` |
| CODE-TEST-03 | `\b[0-9A-F]{4}\b` |

## 5. Golden format (`tests/fixtures/code/<model_type>.json`)

```json
{
  "format": "logestix-code-golden",
  "version": 1,
  "model_type": "granite-4.2-30b",
  "source": "ibm-granite/granite-4.2-30b",
  "runtime": "transformers 5.15.1 / torch 2.13.0",
  "prompts": [
    {
      "id": "CODE-TEST-01",
      "kind": "json",
      "anchor": "root",
      "input_ids": [151644, 872],
      "max_new_tokens": 512,
      "body": "[CONTEXT]: CODE-Validation-Core-01 // ...",
      "expected": "{\"root\":{\"nodes\":{\"leaf\":true}}}",
      "raw": "..."
    }
  ]
}
```

Rules: one file per `IModel::Type()`, filename `<model_type>.json` verbatim;
`input_ids` non-empty, `expected` non-empty, `kind` ∈ {json, assignment, regex},
`anchor` non-empty; malformed file is a parse error (never a silent skip).
`body` (optional) is the prompt body used by `CutPromptEcho`; `ignore_eos`
(optional) forces the full token budget for reasoning models.

## 6. C++ API

New model-neutral files (no family stem; not scanned by the naming guard):

- `libs/lsxcommon/include/lsxcommon/code_selfcheck.h`
- `libs/lsxcommon/src/code_selfcheck.cpp`

```cpp
namespace lsxcommon {
namespace code {

enum class CodeKind { kJson, kAssignment, kRegex };

struct CodeGoldenCase {
    std::string id;
    CodeKind kind = CodeKind::kJson;
    std::string anchor;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 128;
    bool ignore_eos = false;
    std::string expected;
    std::string raw;        // diagnostic
};

struct CodeGolden {
    std::string model_type;
    std::vector<CodeGoldenCase> cases;
};

struct CodeCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;
    bool passed = false;
    std::string error;
};

struct CodeResult {
    std::vector<CodeCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractCodeAnswer(CodeKind kind, std::string_view anchor,
                              std::string_view raw);

bool ParseCodeGolden(std::string_view json, CodeGolden &out, std::string *error);
bool LoadCodeGoldenFile(const std::string &path, CodeGolden &out,
                        std::string *error);

using CodeDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;
CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden);
CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden,
                            const CodeDecodeFn &decode);

}  // namespace code
}  // namespace lsxcommon
```

### 6.1 CLI

- `libs/lsxcommon/src/model_runtime.cpp`:
  `int RunCliCodeSelfCheck(const GenerationCliArgs &args)` — Create -> Load ->
  load golden for `model.Type()` from `LSX_CODE_GOLDEN_DIR` -> `RunCodeSelfCheck`
  -> print per-case pass/fail -> Unload. Returns 0 iff `all_passed`.
- `apps/lsxbenchmark`: add `--code-selfcheck` (mutually exclusive with
  `--selfcheck`/`--fsi-selfcheck`/`--bench`); forces `FastMathMode::kOff`.

## 7. Python capture script

`tools/code_capture.py` (operator-run, not part of the CMake build):

```
python tools/code_capture.py --selftest
python tools/code_capture.py --model <safetensors dir or HF id> \
    --type <model_type> --out tests/fixtures/code/<model_type>.json
```

- `--selftest`: no model. Validates the three templates and the Python
  extraction mirror: CODE-01 parses as the expected nested object, CODE-02
  evaluates to the midpoint, CODE-03 compiles and matches `"ABCD"` but not
  `"abcd"`.
- Per-model override table keyed by `--type` (chat template, dtype, device,
  loader, `enable_thinking`, `max_new_tokens`), same shape as `fsi_capture.py`.
- Capture runs greedily, decodes the continuation, mirrors `extract()` and
  requires a non-empty, balanced result before writing.

## 8. Test integration

### 8.1 Host-only unit tests (TDD, no GPU)

`tests/test_code_selfcheck.cpp`, model-neutral, label `fast`:

- `ExtractCodeAnswer`: json nested/minified/fenced/trailing-prose; assignment
  with spaces/extra parens/fences; regex with leading `\b`, fences, trailing
  period; degenerate (empty, no anchor).
- `ParseCodeGolden` / `LoadCodeGoldenFile`: round-trip, malformed, missing
  fields, unknown kind.
- `RunCodeSelfCheck` against a fake `IModel`: all-pass, one mismatch, inference
  error propagation, request forwarding.

### 8.2 Per-family real-model tests

Each family gets its own `tests/test_<family>_code_selfcheck.cpp`, `add_test`
target with label `inference` and `SKIP_RETURN_CODE 4`, copying the family's
vault-path-helper pattern and creating the model with `fast_math_mode = kOff`.
`SKIP` when the converted model or golden is absent. The suite guard
`Suites()` gains a `"code"` stage per model.

Families: glm-5.2, glm-5.3-flash, deepseek-v4-flash, deepseek-v4.1-flash,
qwen3.5-9b, qwen3.8-27b, qwen3.5-35b-a3b, qwen3.6-35b-a3b, ornith-1.5-35b-a3b,
qwen3.8-flash-next, gemma-4-26b-a4b-it, kimi-k2.7-code,
nemotron-3.5-lightning-30b-a3b, gigachat3.5-432b-a28b-reasoning, granite-4.2-30b.

### 8.3 TDD ordering

1. Failing host unit tests for `ExtractCodeAnswer`.
2. Implement extraction; green.
3. Failing golden parse tests; implement; green.
4. Failing fake-model harness tests; implement; green.
5. `tools/code_capture.py --selftest` implemented and run.
6. Pilot: capture goldens for the runnable families; per-family test targets +
   suite-guard `code` stage; run green on the converted models.
7. CLI `--code-selfcheck` + `RunCliCodeSelfCheck`.

## 9. Risks

- **Exact-match brittleness**: unlike FSI, no arithmetic canonicalization is
  applied, so an association-only rewrite (`m=l+(h-l)/2`) is a gate failure.
  This is intended ("syntactic strictness"): the prompt requests an exact
  unspaced form; a failure is a real finding to investigate.
- **Captured-model coverage**: 432B+ sources cannot be captured locally; those
  goldens are absent and the test SKIPs (never fails).
- **Chat-template mismatch**: removed by capturing `input_ids` verbatim.
- **Golden staleness**: recapture if a model is reconverted from a different
  source revision; the golden records `source`/`runtime`.

## 10. Resources

- Model vault: `/mnt/llm_models/models` (path access via
  `lsxcommon::path_registry`, `LSX_MODEL_ROOT` override).
- Capture runtime: `python3`, `transformers 5.15.1` / `torch 2.13.0`, RTX 5090
  32 GB + host RAM (device offload for >32 GB checkpoints).

## 11. Definition of done

- [ ] `ExtractCodeAnswer` + golden parse + fake-model harness unit-tested
      (label `fast`), green.
- [ ] `tools/code_prompts.json`, `tools/code_capture.py` with `--selftest` pass.
- [ ] Per-family `test_<family>_code_selfcheck` target + suite-guard `code` stage.
- [ ] Pilot goldens committed for every locally-runnable family; test green on
      the converted model.
- [ ] `lsxbenchmark --code-selfcheck` runs and returns 0 on a passing model.
- [ ] `SelfCheck` / FSI targets untouched (decoupling invariants); guards green;
      `AGENTS.md` updated.
