# FSISelfCheck (Financial Services & Infrastructure Self-Check) Design

- **Date**: 2026-09-26
- **Status**: Draft for review
- **Branch**: `feat/fsi-selfcheck`
- **Related**: `docs/superpowers/specs/2026-08-06-selfcheck-design.md`,
  `libs/lsxcommon/src/model_runtime.cpp` (`RunParisSelfCheck`),
  Model Ingestion Protocol step 8 (`ingesting-a-model`)

## 1. Purpose

Add a deterministic, model-parity gate to the Model Ingestion Protocol,
analogous to `SelfCheck` ("The capital of France is" -> "Paris") but keyed to a
fixed financial-services prompt set.

The gate is:

1. **Capture**: run the FSI prompt set on the original **safetensors** model
   (BF16/FP16, via HF transformers) and freeze the extracted answer for that
   model as its golden reference.
2. **Verify**: run the same prompts on the **converted** logestix (NVFP4) model
   and require that the extracted answer matches the golden.
3. **GPU-independent result**: the pass/fail criterion is the normalized answer
   *text*, never logits, hashes or timings. The answer must be reproducible on
   any host/GPU and always equal the model's reference.

The source safetensors model is always available in the vault during and after
adaptation, so capture is an always-runnable, first-class step; the golden is
committed only for reproducibility and CI.

## 2. Scope

In scope:

- The three-prompt FSI set (single source of truth).
- A host-only, model-neutral capture/verify contract in `lsxcommon`.
- A Python capture script (HF transformers, generic + per-model overrides).
- Committed per-model golden JSON under `tests/fixtures/fsi/`.
- Host-only unit tests (fake model, no GPU) and per-family real-model tests.
- `lsxbenchmark --fsi-selfcheck` CLI parity with `--selfcheck`.

Out of scope:

- Changing `SelfCheck` / `RunParisSelfCheck` semantics.
- Running raw safetensors models inside logestix (logestix runs converted
  artifacts only).
- Any speed/perf behaviour.
- Adding a new ctest label or lifecycle stage.

## 3. Contracts

### 3.1 FSI prompt set (single source of truth)

The set lives in `tools/fsi_prompts.json`, read by the Python capture script.
The C++ side does **not** embed the prompt bodies: the golden carries the
captured `input_ids`, so prompt text/tokenization lives only where it is used.

```json
{
  "format": "logestix-fsi-prompts",
  "version": 1,
  "prompts": [
    { "id": "FSI-TEST-01", "title": "CONTINUOUS-COMPOUNDING-STRUCT",
      "kind": "arithmetic", "anchor": "PV",
      "expected": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))", "body": "..." },
    { "id": "FSI-TEST-02", "title": "ARBITRAGE-ALGORITHMIC-PATH",
      "kind": "arithmetic", "anchor": "Balance_Alpha",
      "expected": "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
      "body": "..." },
    { "id": "FSI-TEST-03", "title": "BASEL-THRESHOLD-LOGIC",
      "kind": "comparison", "anchor": "Drawdown",
      "expected": "Drawdown>Threshold", "body": "..." }
  ]
}
```

Prompt bodies (verbatim, inserted into `body`):

**FSI-TEST-01**
```
[CONTEXT]: FSI-Validation-Core-01 // Financial Structure
Define the mathematical execution block for the Present Value (PV) of an annuity-due asset under continuous compounding.
Parameters: CF = 2500000; n = 7; r = 0.0835.
Task: Format the target PV calculation strictly as a single-line Python-compatible expression using exp() and the exact variable names provided.
Output Specification: Output ONLY the raw formula string. Do not use markdown blocks, do not include spaces inside the formula, and do not emit explanatory text.
Expected Template: PV=CF*((1-exp(-r*n))/(1-exp(-r)))
```

**FSI-TEST-02**
```
[CONTEXT]: FSI-Validation-Core-02 // Deterministic Routing Loop
Map the structural sequence for a 3-leg currency clearing loop with fixed slippage mitigation.
Initial_State: Balance_Alpha = 100000000.
Legs: Rate_A=1.083541; Slip_A=0.0005; Rate_B=83.152402; Slip_B=0.0005; Rate_C=0.011104; Slip_C=0.0005.
Task: Formulate the continuous multi-leg balance mutation sequence in a single line of standard arithmetic operators without spaces.
Output Specification: Output ONLY the final math equation string solving for the terminal state of Balance_Alpha using the provided variable names. No text, no markdown blocks.
Expected Template: Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C
```

**FSI-TEST-03**
```
[CONTEXT]: FSI-Validation-Core-03 // Boundary Decision Check
Construct the evaluation logic for a Basel-compliant threshold boundary breach.
Variables: Drawdown = 14.999991; Threshold = 15.000000.
Task: Formulate the exact conditional logical statement that checks if Drawdown strictly exceeds Threshold.
Output Specification: Output strictly the standard boolean comparison string without spaces. Do not evaluate numerically, do not include code blocks or text.
Expected Template: Drawdown>Threshold
```

### 3.2 Capture contract (Python, original safetensors)

- Load with `transformers`:
  `AutoTokenizer.from_pretrained(src, trust_remote_code=True)` and
  `AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, torch_dtype=...)`.
- Frame each prompt exactly as the model expects: use
  `tokenizer.apply_chat_template(...)` when the tokenizer defines a chat
  template and the model is instruction-tuned, otherwise raw encode. The choice
  is explicit per model in the script's override table, never implicit.
- Generate greedily (`do_sample=False`), fixed `max_new_tokens` (default 128),
  eos-terminated, single sequence (batch 1), deterministic seed irrelevant.
- Decode only the generated continuation and run the **same** extraction as
  C++ (Python mirror of `ExtractFsiFormula`).
- Validate before writing: the extracted answer must equal the prompt's
  `expected` after normalization. A mismatch aborts capture (the reference is
  not accepted) and prints the raw output for review.
- Write the golden JSON (schema below) with `input_ids` captured **after**
  framing, so the C++ verify path re-uses them verbatim and no chat-template /
  tokenizer parity is assumed between HF and logestix. The captured ids include
  every special token the model's own tokenizer added.

### 3.3 Verify contract (logestix, converted model)

- For each golden case: `model.Infer(ModelRequest{input_ids, max_gen_tokens})`
  with `input_ids` and `max_gen_tokens` taken from the golden.
- Decode the generated tokens with `runtime::DecodeGeneratedText`.
- Extract with `ExtractFsiFormula(kind, anchor, decoded)`.
- Pass iff the extracted string equals the golden `expected`.
- Greedy decoding, eos-terminated (`ignore_eos = false`), no `ignore_eos`
  fixed-length override.

### 3.4 GPU independence / numeric mode

- The pass criterion compares normalized text only; no `final_logits`, no
  `LSX_SELFCHECK_EXPECT_HASH`, no timing.
- Verify runs at `FastMathMode::kOff` (bit-exact scalar path) so the decoded
  answer is reproducible across hosts and GPUs. The test creates the model with
  `ModelInitConfig::fast_math_mode = kOff`; the CLI forces `kOff` for
  `--fsi-selfcheck` regardless of flags.
- The extraction/normalization function and the harness are host-only and
  unit-tested without a GPU (fake `IModel`).

## 4. Extraction / normalization contract

`ExtractFsiFormula(FsiKind kind, std::string_view anchor, std::string_view raw)`
is a pure host function. Algorithm:

1. Trim ASCII whitespace from `raw`.
2. If the string contains a fenced block (``` ``` ```), keep the content of the
   first block; drop a leading language tag line matching `^[A-Za-z0-9_+-]+$`.
3. Select the first line containing `anchor`.
4. Kind `arithmetic`:
   - Skip to the first `=` after `anchor`, take `anchor` + `=`.
   - Consume the expression from there:
     - if a `(` appears, scan tracking paren depth and stop when depth first
       returns to 0 (drops trailing prose deterministically);
     - otherwise consume the maximal run of `[A-Za-z0-9_+\-*/().]`.
5. Kind `comparison`:
   - Take `anchor`, then the operator run `[<>=!]+`, then one identifier
     `[A-Za-z_][A-Za-z0-9_]*` (e.g. `Drawdown>Threshold`).
6. Remove all internal whitespace; strip trailing `.`, `,`, `;`.
7. Return the normalized result.

Canonical outputs:

| id | extracted |
|---|---|
| FSI-TEST-01 | `PV=CF*((1-exp(-r*n))/(1-exp(-r)))` |
| FSI-TEST-02 | `Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C` |
| FSI-TEST-03 | `Drawdown>Threshold` |

The Python capture script implements the identical algorithm, so the golden
`expected` is produced by the same normalization that the verify path applies.

## 5. Golden format (`tests/fixtures/fsi/<model_type>.json`)

```json
{
  "format": "logestix-fsi-golden",
  "version": 1,
  "model_type": "qwen3.6-35b-a3b",
  "source": "Qwen/Qwen3.6-35B-A3B",
  "runtime": "transformers 4.x / torch 2.x",
  "prompts": [
    {
      "id": "FSI-TEST-01",
      "kind": "arithmetic",
      "anchor": "PV",
      "input_ids": [151644, 872, ...],
      "max_new_tokens": 128,
      "expected": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))",
      "raw": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))"
    }
  ]
}
```

Rules:

- One file per `IModel::Type()`; the filename is `<model_type>.json` verbatim
  (dots allowed, matching the registry strings).
- `input_ids` non-empty, `expected` non-empty, `kind` ∈ {arithmetic,
  comparison}, `anchor` non-empty; a malformed file is a parse error and the
  case fails (never silently skips).
- `raw` is diagnostic only.

## 6. C++ API

New model-neutral files (no family stem; not scanned by the naming guard):

- `libs/lsxcommon/include/lsxcommon/fsi_selfcheck.h`
- `libs/lsxcommon/src/fsi_selfcheck.cpp`

```cpp
namespace lsxcommon {
namespace fsi {

enum class FsiKind { kArithmetic, kComparison };

struct FsiGoldenCase {
    std::string id;
    FsiKind kind = FsiKind::kArithmetic;
    std::string anchor;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 128;
    std::string expected;   // normalized golden answer
    std::string raw;        // diagnostic
};

struct FsiGolden {
    std::string model_type;
    std::vector<FsiGoldenCase> cases;
};

struct FsiCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    bool passed = false;
    std::string error;      // inference/decode error for this case, if any
};

struct FsiResult {
    std::vector<FsiCaseResult> cases;
    bool all_passed = false;
    std::string error;      // golden parse / model-level error
};

// Pure host function; unit-tested without CUDA.
std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw);

// simdjson (already a PRIVATE dependency of lsxcommon).
bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error);
std::string LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                              std::string *error);

// Feeds golden input_ids to the model, decodes, extracts, compares.
FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden);

}  // namespace fsi
}  // namespace lsxcommon
```

`RunFsiSelfCheck` is model-neutral: it uses only `IModel::Infer`, `Tokenizer`
and `runtime::DecodeGeneratedText`. It does not read prompt bodies and does not
re-tokenize.

### 6.1 CLI

- `libs/lsxcommon/src/model_runtime.cpp`:
  `int RunCliFsiSelfCheck(const GenerationCliArgs &args)` — Create -> Load ->
  load golden for `model.Type()` -> `RunFsiSelfCheck` -> print per-case
  pass/fail -> Unload. Returns 0 iff `all_passed`.
- `apps/lsxbenchmark`: add `--fsi-selfcheck` (mutually exclusive with
  `--selfcheck`/`--bench`); it forces `FastMathMode::kOff`.
- Golden directory resolution: env `LSX_FSI_GOLDEN_DIR` (default
  `tests/fixtures/fsi` relative to CWD); tests pass an explicit path.

## 7. Python capture script

`tools/fsi_capture.py` (not part of the CMake build; operator-run, like
`tools/generate_unicode_tables.ps1`):

```
python tools/fsi_capture.py --selftest
python tools/fsi_capture.py --model <safetensors dir or HF id> \
    --type <model_type> --out tests/fixtures/fsi/<model_type>.json
```

- `--selftest`: no model required. Validates the three templates:
  FSI-TEST-01 / FSI-TEST-02 parse and evaluate correctly with the given
  parameters; FSI-TEST-03 evaluates to `False`. Also unit-checks the Python
  extraction mirror against the canonical outputs in section 4.
- `--capture` (default when `--model` given): runs the model as in 3.2,
  asserts the extracted answer equals `expected`, writes the golden JSON.
- Per-model override table: `chat_template` on/off, `torch_dtype`,
  `trust_remote_code`, `max_new_tokens`, and any tokenizer quirk. Keyed by
  `--type`; absent entry -> generic defaults.

## 8. Test integration

### 8.1 Host-only unit tests (TDD, no GPU)

New `tests/test_fsi_selfcheck.cpp`, model-neutral, label `fast`:

- `ExtractFsiFormula`: arithmetic with fences, trailing prose, extra spaces,
  balanced-paren truncation; comparison with spaces and trailing period;
  degenerate inputs (empty, no anchor).
- `ParseFsiGolden` / `LoadFsiGoldenFile`: round-trip, malformed JSON, missing
  fields, unknown `kind`.
- `RunFsiSelfCheck` against a fake `IModel` returning canned generated token
  ids: all-pass, one mismatch, inference error propagation, short answer.

### 8.2 Per-family real-model tests

Add one `TEST_CASE("... FSI SelfCheck ...")` to each existing
`tests/test_<family>_selfcheck.cpp` (same binary/target, so label `inference`
and `SKIP_RETURN_CODE 4` already hold, and `test_model_suite_guard` needs no
stage change):

- Reuse the file's existing vault-path helper.
- `SKIP` when the converted model is absent **or** the golden JSON is absent.
- Create the model with `fast_math_mode = kOff`.
- `REQUIRE(result.all_passed)` and, on failure, `INFO` the per-case
  expected/extracted strings for diagnosis.
- Golden path via a forwarded `LSX_FSI_GOLDEN_DIR` compile definition
  (same CMake pattern as `LSX_SOURCE_DIR` / `TOKENIZER_TEST_DATA_DIR`), with an
  env override for ad-hoc runs.

Families: glm-5.2, glm-5.3-flash, deepseek-v4-flash, deepseek-v4.1-flash,
qwen3.5-9b, qwen3.8-27b, qwen3.5-35b-a3b, qwen3.6-35b-a3b, ornith-1.5-35b-a3b,
qwen3.8-flash-next, gemma-4-26b-a4b-it, kimi-k2.7-code,
nemotron-3.5-lightning-30b-a3b, gigachat3.5-432b-a28b-reasoning,
granite-4.2-30b.

### 8.3 TDD ordering

1. Failing host unit tests for `ExtractFsiFormula`.
2. Implement extraction; green.
3. Failing JSON parse tests; implement; green.
4. Failing fake-model `RunFsiSelfCheck` tests; implement; green.
5. `tools/fsi_capture.py --selftest` (formula checks) implemented and run.
6. Pilot: capture the Qwen3.6-35B-A3B golden; add its per-family TEST_CASE;
   run green on the real converted model.
7. Roll the per-family TEST_CASE + golden out to the remaining families.
8. CLI `--fsi-selfcheck` + `RunCliFsiSelfCheck`.

### 8.4 Guards / docs

- No new family/variant/role token -> `test_code_naming_guard` untouched.
- No new ctest label -> `test_ctest_labels_guard` untouched.
- `test_model_suite_guard` untouched (FSI lives in the existing selfcheck
  target).
- `AGENTS.md` Structure section gains `fsi_selfcheck` and `tools/fsi_capture.py`
  / `tests/fixtures/fsi/` on landing.

## 9. Risks

- **NVFP4 drift on long formulas**: FSI-02 is ~60 characters. If the converted
  model diverges, the extracted answer differs and the gate fails. Mitigation:
  `kOff` bit-exact verify; if a family still drifts, the failure is real and
  must be investigated (that is the gate's purpose), not masked.
- **Chat-template mismatch**: removed by capturing `input_ids` in the golden and
  feeding them verbatim.
- **Python/HF availability per family**: capture needs the model's code and a
  compatible transformers version; the override table isolates quirks. Absent
  golden -> test SKIPs, never fails.
- **Golden staleness**: if a model is reconverted from a different source
  revision, recapture. The golden records `source`/`runtime` for traceability.

## 10. Resources

- **Model vault**: `/mnt/llm_models/models` (source safetensors and converted
  `logestix/` artifacts). Path access stays through `lsxcommon::path_registry`
  (`LSX_MODEL_ROOT` override); no hardcoded vault paths in code.
- **NVIDIA/CUDA reference docs** (MCP dump): `/home/alexmv2025/projects/cuda/docs`
  (`NVIDIA_DOCS_NVFP4_MEMORY.md`, `NVIDIA_DOCS_HOPPER.md`,
  `NVIDIA_DOCS_AMPERE.md`, `NVIDIA_DOCS_RUBIN.md`). FSISelfCheck is host-side
  and touches no NVFP4 kernel; these docs are the mandatory reference only if a
  kernel edit becomes necessary (e.g. a numeric-determinism fix exposed by an
  FSI failure).
- **Capture runtime** (verified available): `python3` with `transformers 5.15.1`
  / `torch 2.13.0`; host has 750 GiB RAM and an RTX 5090 32 GB, so a pilot
  capture can offload a BF16 model to host RAM.

## 11. Definition of done

- [ ] `ExtractFsiFormula` + golden parse + fake-model harness unit-tested
      (label `fast`), green.
- [ ] `tools/fsi_prompts.json`, `tools/fsi_capture.py` with `--selftest` pass.
- [ ] Pilot golden + per-family TEST_CASE green on the real converted model.
- [ ] Per-family TEST_CASE + golden present for every ingested family.
- [ ] `lsxbenchmark --fsi-selfcheck` runs and returns 0 on a passing model.
- [ ] All four guard tests green; `AGENTS.md` updated.
