# DETSelfCheck (Determinism Stress-Test + Near-Tie Filtering) Design

- **Date**: 2026-09-27
- **Status**: Approved (design) — implementation via TDD
- **Branch**: `feat/det-selfcheck`
- **Related**: `docs/superpowers/specs/2026-09-26-fsi-selfcheck-design.md`,
  `docs/superpowers/specs/2026-09-27-code-selfcheck-design.md`,
  `libs/lsxcommon/src/{fsi,code}_selfcheck.cpp`, Model Ingestion Protocol step 8
  (`ingesting-a-model`)

## 1. Purpose

Add a deterministic, model-parity gate keyed to a fixed **near-tie / symmetric**
prompt set, exactly parallel to FSISelfCheck and CODESelfCheck but testing two
properties the other gates do not:

1. **Determinism stress-test** — repeated greedy decoding of the same prompt
   must produce a **byte-identical generated token stream** (zero variance).
2. **Near-Tie filtering** — the prompts are constructed so the sampler faces an
   ambiguous / symmetric decision; greedy decoding must resolve the tie to the
   golden canonical token.

**DETSelfCheck is a separate functionality from `SelfCheck`, `FSISelfCheck` and
`CODESelfCheck`.** It shares no API, state, golden, environment variable, result
type or ctest target with any of them. It depends only on the model-neutral
`IModel` surface (`Load`/`Infer`/`Unload`/`Type`/`Tokenizer`) and never calls
`IModel::SelfCheck`, `runtime::RunParisSelfCheck`, `fsi::RunFsiSelfCheck`,
`code::RunCodeSelfCheck`, or reads `selfcheck_passed` / `LSX_SELFCHECK_EXPECT_HASH`.

The gate is:

1. **Capture**: run the DET prompt set on the original **safetensors** model
   (via HF transformers), freeze the model's own extracted answer as its golden.
2. **Verify**: run the same prompts R times on the **converted** logestix
   (NVFP4) model at `FastMathMode::kOff`; require (a) all R generated token
   streams identical and (b) the extracted answer `==` golden `expected`.
3. **GPU-independent result**: the criterion is token-stream identity plus
   normalized answer text — never logits, hashes, timings or a numeric margin.

## 2. Scope

In scope:

- The three-prompt DET set (single source of truth).
- A host-only, model-neutral capture/verify contract in `lsxcommon`.
- A Python capture script (HF transformers, generic + per-model overrides).
- Committed per-model golden JSON under `tests/fixtures/det/`.
- Host-only unit tests (fake model, no GPU) and per-family real-model tests.
- A separate `lsxbenchmark --det-selfcheck` entry point.
- A `det` lifecycle stage in `test_model_suite_guard.cpp` for all 15 families.

Out of scope:

- Changing `SelfCheck` / FSI / CODE semantics or targets.
- Any logit-margin threshold policy, top-2 margin assertion, or token-probability
  comparison (explicitly rejected: it would bind the golden to a vocabulary and
  numeric class).
- Determinism across `FastMathMode` classes (kAuto is not contractually
  bit-identical to kOff).
- Running raw safetensors models inside logestix.
- Any speed/perf behaviour.
- Adding a new ctest label (stays `inference`; `det` is a stage).

## 3. Contracts

### 3.1 DET prompt set (single source of truth)

The set lives in `tools/det_prompts.json` (`format: logestix-det-prompts`,
`version: 1`), read by the Python capture script. The C++ side does **not**
embed the prompt bodies: the golden carries the captured `input_ids`.

All three cases use `kind: "token"` (no arithmetic/json parsing; the answer is a
single canonical capitalized token).

| id | title | expected |
|---|---|---|
| DET-TEST-01 | PARADOXICAL-LOGIC-ANCHOR | `PARADOX_ANTINOMY` |
| DET-TEST-02 | LEXICAL-INDEX-CLAMP | `CATASTROPHIC` |
| DET-TEST-03 | BINARY-REGISTER-STRING | `BIT_ZERO` |

Bodies (verbatim):

- DET-TEST-01
  `[CONTEXT]: DET-Validation-Core-01 // Self-Referential Resolution`
  `Resolve the standard Epimenides boundary paradox under absolute zero-variance constraints.`
  `Statement: "This explicit statement is completely false."`
  `Task: Determine the structural truth-value behavior condition of the statement above.`
  `Output Specification: Output strictly one capitalized token: PARADOX_ANTINOMY. Do not evaluate textually, do not include whitespace or formatting tags.`
  `Expected Template: PARADOX_ANTINOMY`
- DET-TEST-02
  `[CONTEXT]: DET-Validation-Core-02 // Literal Alphabetical Sort`
  `Select exactly one word from the provided array: [CATASTROPHIC, CRITICAL, FATAL, TERMINAL].`
  `Constraint: Choose the specific word that strictly begins with the character 'C' and ends with the character 'C'.`
  `Output Specification: Output ONLY the single matching uppercase word payload without spaces or markdown blocks.`
  `Expected Template: CATASTROPHIC`
- DET-TEST-03
  `[CONTEXT]: DET-Validation-Core-03 // Superposition State Anchor`
  `Evaluate a simulated hardware register state holding a default static zero condition.`
  `Task: Force the output token to represent this state strictly as a two-token alphabetical word connected by an underscore.`
  `Output Specification: Output exactly the uppercase characters pattern provided in the template below. Do not output raw single digits, omit spaces.`
  `Expected Template: BIT_ZERO`

The golden `expected` is **template-defined** (the prompt's canonical Expected
Template); the source run only supplies `input_ids` and logs an agreement warning.
DET-TEST-02 and DET-TEST-03 were reworded after the first run (the original
"DET-TEST-02 highest alphabetical sort priority" was ambiguous and DET-TEST-03 made
models emit the raw bit `0`) — see `docs/DET_SELFCHECK_FINDINGS.md`.

### 3.2 Verify contract

For each golden case:

- Run `model.Infer(ModelRequest{input_ids, max_gen_tokens, ignore_eos})` **R
  times** in the same loaded session, with `input_ids`/`max_gen_tokens` from the
  golden (greedy, eos-terminated unless `ignore_eos`).
- `R = golden.stress_runs` (capture writes 3); an explicit override (CLI/test)
  takes precedence, and the effective R is floored at 3.
- `deterministic` iff every run's `InferenceResult.gen_tokens` equals run 0's,
  element for element.
- Decode run 0 with `runtime::DecodeGeneratedText` for diagnostics and
  extraction.
- If the golden carries `body`, cut the decoded text at the body's first 32
  characters (after the `</think>` cut) — a fragile reasoning model can fail to
  emit a turn-end token and echo the prompt after its answer.
- `extracted = ExtractDetAnswer(DetKind::kToken, decoded)`.
- `passed = deterministic && (extracted == golden.expected)`.

### 3.3 GPU independence / numeric mode

- The pass criterion compares token-id streams and normalized text only.
- Verify runs at `FastMathMode::kOff`; the CLI forces `kOff` for
  `--det-selfcheck`.
- Extraction/harness are host-only and unit-tested without a GPU.

### 3.4 Decoupling invariants (DETSelfCheck independent)

These are invariants, not guidelines.

| Dimension | SelfCheck | FSISelfCheck | CODESelfCheck | DETSelfCheck |
|---|---|---|---|---|
| Entry point | `IModel::SelfCheck`, `runtime::RunParisSelfCheck` | `fsi::RunFsiSelfCheck` | `code::RunCodeSelfCheck` | `det::RunDetSelfCheck` |
| Input | hardcoded `"The capital of France is"` | FSI golden `input_ids` | CODE golden `input_ids` | DET golden `input_ids` |
| Reference | in-test literal `"Paris"` | committed FSI golden | committed CODE golden | committed DET golden |
| Criterion | top-1 / short-prefix text (+logits hash) | canonicalized answer `==` golden | normalized answer `==` golden (exact) | token-stream identity across R runs **and** answer `==` golden (exact) |
| Result type | `gpu::InferenceResult.selfcheck_passed` | `fsi::FsiResult.all_passed` | `code::CodeResult.all_passed` | `det::DetResult.all_passed` |
| Numeric mode | `kAuto` | `kOff` | `kOff` | `kOff` |
| Env gates | `LSX_SELFCHECK_EXPECT_HASH` | `LSX_FSI_GOLDEN_DIR` | `LSX_CODE_GOLDEN_DIR` | `LSX_DET_GOLDEN_DIR` (+ `LSX_DET_STRESS_RUNS`) |
| CLI | `--selfcheck` | `--fsi-selfcheck` | `--code-selfcheck` | `--det-selfcheck` |
| Host unit test | `tests/test_model_runtime.cpp` | `tests/test_fsi_selfcheck.cpp` | `tests/test_code_selfcheck.cpp` | `tests/test_det_selfcheck.cpp` |
| Real-model test | `tests/test_<family>_selfcheck.cpp` | `tests/test_<family>_fsi_selfcheck.cpp` | `tests/test_<family>_code_selfcheck.cpp` | `tests/test_<family>_det_selfcheck.cpp` |
| ctest stage | `selfcheck` | `fsi` | `code` | `det` |

Shared, intentionally: the neutral `IModel` inference surface, `ArrowTokenizerDb`
and `runtime::DecodeGeneratedText`. Nothing else.

## 4. Extraction / normalization contract

`ExtractDetAnswer(DetKind kind, std::string_view raw)` is a pure host function.

Common preprocessing (identical to FSI/CODE):

1. Trim ASCII whitespace.
2. If a fenced block exists, keep the first block's content; drop a leading
   language-tag line matching `^[A-Za-z0-9_+-]+$`.
3. Discard everything up to and including the last `</think>`.

The harness additionally applies `CutPromptEcho` before extraction when the
golden carries `body` (see 3.2): after step 3, if the body's first 32 characters
occur in the text, the text is truncated there. This is not part of
`ExtractDetAnswer` (a pure function of `kind`/`raw`).

Then for the single `kToken` kind:

- Scan the answer region for the first standalone token matching
  `[A-Z][A-Z0-9_]*` with length ≥ 2, uppercased.
- "Standalone" means the character before is not `[A-Za-z0-9_]` and the
  character after is not `[A-Za-z0-9_]` (so `PARADOX_ANTINOMY` matches whole,
  and words inside sentences are still eligible when fully uppercase).
- Return empty if none is found.

Canonical outputs (the golden `expected` is produced by the capture model's own
answer, so a correctly-trained model reproduces its token verbatim):

| id | expected |
|---|---|
| DET-TEST-01 | `PARADOX_ANTINOMY` |
| DET-TEST-02 | `CATASTROPHIC` |
| DET-TEST-03 | `BIT_ZERO` |

## 5. Golden format (`tests/fixtures/det/<model_type>.json`)

```json
{
  "format": "logestix-det-golden",
  "version": 1,
  "model_type": "granite-4.2-30b",
  "source": "ibm-granite/granite-4.2-30b",
  "runtime": "transformers 5.15.1 / torch 2.13.0",
  "prompts": [
    {
      "id": "DET-TEST-01",
      "kind": "token",
      "input_ids": [151644, 872],
      "max_new_tokens": 512,
      "stress_runs": 3,
      "body": "[CONTEXT]: DET-Validation-Core-01 // ...",
      "expected": "PARADOX_ANTINOMY",
      "raw": "..."
    }
  ]
}
```

Rules: one file per `IModel::Type()`, filename `<model_type>.json` verbatim;
`input_ids` non-empty, `expected` non-empty, `kind` ∈ {`token`}; malformed file
is a parse error (never a silent skip). `body` (optional) is the prompt body
used by `CutPromptEcho`; `ignore_eos` (optional) forces the full token budget for
reasoning models; `stress_runs` (optional, default 3) is the determinism
repetition count.

## 6. C++ API

New model-neutral files (no family stem; not scanned by the naming guard):

- `libs/lsxcommon/include/lsxcommon/det_selfcheck.h`
- `libs/lsxcommon/src/det_selfcheck.cpp`

```cpp
namespace lsxcommon {
namespace det {

enum class DetKind { kToken };

struct DetGoldenCase {
    std::string id;
    DetKind kind = DetKind::kToken;
    std::vector<int32_t> input_ids;
    int max_new_tokens = 64;
    bool ignore_eos = false;
    int stress_runs = 3;
    std::string expected;
    std::string raw;   // diagnostic
    std::string body;  // optional prompt-echo cut
};

struct DetGolden {
    std::string model_type;
    std::vector<DetGoldenCase> cases;
};

struct DetCaseResult {
    std::string id;
    std::string expected;
    std::string extracted;
    std::string raw;            // decoded run-0 text (diagnostic)
    int runs = 0;               // effective R
    bool deterministic = false; // all runs' gen_tokens identical
    bool passed = false;
    std::string error;
};

struct DetResult {
    std::vector<DetCaseResult> cases;
    bool all_passed = false;
    std::string error;
};

std::string ExtractDetAnswer(DetKind kind, std::string_view raw);

bool ParseDetGolden(std::string_view json, DetGolden &out, std::string *error);
bool LoadDetGoldenFile(const std::string &path, DetGolden &out,
                       std::string *error);

using DetDecodeFn = std::function<std::string(const std::vector<int32_t> &)>;

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden);
DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden,
                          const DetDecodeFn &decode, int stress_runs_override = 0);

}  // namespace det
}  // namespace lsxcommon
```

### 6.1 CLI

- `libs/lsxcommon/src/model_runtime.cpp`:
  `int RunCliDetSelfCheck(const GenerationCliArgs &args)` — Create -> Load ->
  load golden for `model.Type()` from `LSX_DET_GOLDEN_DIR` -> resolve
  `LSX_DET_STRESS_RUNS` override -> `RunDetSelfCheck` -> print per-case
  determinism/extracted/pass -> Unload. Returns 0 iff `all_passed`.
- `apps/lsxbenchmark`: add `--det-selfcheck` (mutually exclusive with
  `--selfcheck`/`--fsi-selfcheck`/`--code-selfcheck`/`--bench`); forces
  `FastMathMode::kOff`.

## 7. Python capture script

`tools/det_capture.py` (operator-run, not part of the CMake build):

```
python tools/det_capture.py --selftest
python tools/det_capture.py --model <safetensors dir or HF id> \
    --type <model_type> --out tests/fixtures/det/<model_type>.json
python tools/det_capture.py --model <dir> --type <t> --out <o> --template-golden
```

- `--selftest`: no model. Validates the three templates and the Python
  extraction mirror (each Expected Template extracts to itself; a nonsensical
  string yields empty).
- Per-model override table keyed by `--type` (chat template, dtype, device,
  loader, `enable_thinking`, `max_new_tokens`), same shape as `code_capture.py`.
- Capture runs greedily, decodes the continuation, mirrors `extract()`, requires
  a non-empty result before writing, and records `stress_runs` (default 3).
  `max_new_tokens` defaults to 512 (matching `fsi_capture.py`): reasoning models
  whose chat template forces a `<think>` preamble (e.g. GLM-5.3-Flash) need a
  budget well above their reasoning length or the answer is never reached
  (`docs/DET_SELFCHECK_FINDINGS.md`).
- `--template-golden`: for an unrunnable source, frame with the tokenizer's chat
  template + `tokenizer.json`, record the prompt Expected Template as `expected`,
  and mark the deviation in `source`.

## 8. Test integration

### 8.1 Host-only unit tests (TDD, no GPU)

`tests/test_det_selfcheck.cpp`, model-neutral, label `fast`:

- `ExtractDetAnswer`: first uppercase token; unfenced; after `</think>`;
  trailing prose/period; lowercase prose ignored; short tokens (<2) ignored;
  degenerate (empty, no token).
- `ParseDetGolden` / `LoadDetGoldenFile`: round-trip, malformed, missing fields,
  unknown kind, `stress_runs` default and explicit.
- `RunDetSelfCheck` against a fake `IModel`: all-runs-identical + expected ->
  pass; a nondeterministic fake (different `gen_tokens` on the 2nd call) ->
  `deterministic == false` and fail; answer mismatch -> fail; inference error
  propagation; `stress_runs` override/floor and request forwarding.

### 8.2 Per-family real-model tests

Each family gets its own `tests/test_<family>_det_selfcheck.cpp`, `add_test`
target with label `inference`, `SKIP_RETURN_CODE 4`, and a
`LSX_DET_GOLDEN_DIR="${LSX_DET_GOLDEN_DIR_FWD}"` definition, copying the family's
vault-path-helper pattern and creating the model with `fast_math_mode = kOff`.
`SKIP` when the converted model or golden is absent. The suite guard `Suites()`
gains a `"det"` stage per model.

Families: glm-5.2, glm-5.3-flash, deepseek-v4-flash, deepseek-v4.1-flash,
qwen3.5-9b, qwen3.8-27b, qwen3.5-35b-a3b, qwen3.6-35b-a3b, ornith-1.5-35b-a3b,
qwen3.8-flash-next, gemma-4-26b-a4b-it, kimi-k2.7-code,
nemotron-3.5-lightning-30b-a3b, gigachat3.5-432b-a28b-reasoning, granite-4.2-30b.

### 8.3 TDD ordering

1. Failing host unit tests for `ExtractDetAnswer`.
2. Implement extraction; green.
3. Failing golden parse tests; implement; green.
4. Failing fake-model harness tests (determinism, R override, error); implement;
   green.
5. `tools/det_prompts.json` + `tools/det_capture.py --selftest` implemented and
   run.
6. CLI `--det-selfcheck` + `RunCliDetSelfCheck` + CLI unit test.
7. Pilot: capture goldens for the runnable families; per-family test targets +
   suite-guard `det` stage; run green on the converted models.
8. Docs (`AGENTS.md`, this spec, `DET_SELFCHECK_FINDINGS.md`, skill) and guards.

## 9. Risks

- **Runtime cost**: R× the inferences of FSI per model. Loading dominates in
  practice; `stress_runs` is per-golden and runtime-overridable.
- **Runtime nondeterminism is a real finding**: if a converted model is not
  bit-deterministic at `kOff`, DET fails intentionally. Investigate the kernel
  or cache path; do not loosen the gate.
- **Extraction fragility**: a model that echoes the prompt after a missing
  turn-end token is handled by the `body` cut; otherwise a reasoning line that
  contains the token could shadow the answer — inspect `raw` on failure.
- **Captured-model coverage**: sources that cannot run locally use the
  template-golden stopgap (marked in `source`) or are absent (test SKIPs, never
  fails).
- **Golden staleness**: recapture if a model is reconverted from a different
  source revision; the golden records `source`/`runtime`.

## 10. Resources

- Model vault: `/mnt/llm_models/models` (path access via
  `lsxcommon::path_registry`, `LSX_MODEL_ROOT` override).
- Capture runtime: `python3`, `transformers 5.15.1` / `torch 2.13.0`, RTX 5090
  32 GB + host RAM (device offload for >32 GB checkpoints).

## 11. Definition of done

- [ ] `ExtractDetAnswer` + golden parse + fake-model harness (determinism, R
      override, error) unit-tested (label `fast`), green.
- [ ] `tools/det_prompts.json`, `tools/det_capture.py` with `--selftest` pass.
- [ ] Per-family `test_<family>_det_selfcheck` target + suite-guard `det` stage
      for all 15 families; guards green.
- [ ] Pilot goldens committed for every locally-runnable family; test green on
      the converted model (SKIP otherwise).
- [ ] `lsxbenchmark --det-selfcheck` runs and returns 0 on a passing model.
- [ ] `SelfCheck` / FSI / CODE targets untouched (decoupling invariants);
      `AGENTS.md` updated.
