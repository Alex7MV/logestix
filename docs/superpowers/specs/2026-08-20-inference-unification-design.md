# Inference Unification & Test Actualization — Design Spec

**Date:** 2026-08-20
**Status:** Approved (pending implementation plan)
**Scope:** `libs/lsxcommon` model layer + `tests/`

## 1. Problem Statement

Four models are registered in `LsxModelFactory` (`libs/lsxcommon/src/model_registry.cpp`):
`deepseek-v4-flash`, `glm-5.2`, `qwen3.5-9b`, `qwen3.8-27b`. An inference audit
surfaced duplication and inconsistency across their `IModel` implementations:

1. **`ResidentWeightBytes()` divergence** — GLM (`glm_v5_model.h:24`) and Qwen
   (`qwen_base_model.h:29`) override it to return pinned-weight bytes; DeepSeek
   does not override → returns default `0`.
2. **Triplicated `Load()` preamble** — `DeepSeekV4FlashModel::Load`
   (`deepseek_v4_flash_model.cpp:29-39`), `GlmV5Model::Load`
   (`glm_v5_model.cpp:21-31`), `QwenBaseModel::Load`
   (`qwen_base_model.cpp:21-52`) repeat the same guard sequence (`loaded_`
   early-out, empty `model_path` error, `LoadTokenizerDb`) with only the log tag
   differing.
3. **Triplicated `Inspect()` body** — all three `Inspect()` implementations are
   identical `return lsxcommon::scan_directory(cfg_.model_path);`.
4. **`num_layers` inconsistency** — GLM `Infer` sets `r.num_layers`
   (`glm_v5_model.cpp:113`), DeepSeek pipeline sets it; Qwen `Infer`
   (`qwen_base_model.cpp:101-123`) leaves it `0`.
5. **Stale test** — `test_glm_model.cpp:48-58` asserts `Infer`/`SelfCheck`
   FAIL with comment "Inference pipeline is not yet implemented for GLM".
   Contradicts the now-implemented GLM inference (real-model selfcheck/prompt
   tests exist).
6. **Missing deterministic SelfCheck tests** — only `test_glm_selfcheck.cpp`
   has a 3-cycle bit-exact `"Paris"` SelfCheck. DeepSeek and Qwen both implement
   working `SelfCheck` via the shared `runtime::RunParisSelfCheck`
   (`model_runtime.cpp:47-84`) but neither has a deterministic selfcheck test
   (DeepSeek's `"Paris"` output is only implicitly proven by
   `test_deepseek_prompt_output.cpp`; Qwen is unproven).

## 2. Goals (non-goals)

**Goals**

- Zero-behaviour-change refactor: unify duplicated `IModel` plumbing in
  `ModelBase` with protected helpers while preserving each model's distinct
  load semantics (GLM pins per-`Infer`, DeepSeek loads per-call, Qwen pins in
  `Load`).
- Consistent `ResidentWeightBytes()` and `num_layers` across all models.
- Actualize tests: fix the stale GLM test, add deterministic real-model
  SelfCheck tests for Qwen and DeepSeek (SKIP when weights absent).

**Non-goals**

- No change to inference algorithm/kernel behaviour, no new features.
- No migration of GLM/DeepSeek to pin-in-`Load` semantics (out of scope).
- `test_model_factory.cpp` stays adaptive: only tests models actually present
  locally (already does this via `TestModelPathUsable()` / `SKIP`). No hard
  requirement to enumerate all 4 registrations.

## 3. Approach: protected helpers in `ModelBase`

Chosen approach (user-selected): **helpers in `ModelBase`** over full default
methods in base, because `Load()` genuinely differs between models (weight-pin
timing). This is a low-risk, non-breaking refactor.

### 3.1 `ModelBase` additions (`libs/lsxcommon/include/lsxcommon/model_base.h`)

- `protected`:
  - `bool BeginLoad(const std::string &model_path)` — `loaded_` early-out
    `return true`; empty `model_path` → `spdlog::error` + `return false`;
    `LoadTokenizerDb(model_path)` failure → `false`; else `true`.
  - `void FinishLoad()` — sets `loaded_ = true;`.
  - `std::uint64_t resident_bytes_ = 0;`
- `public`:
  - `SafetensorsModel Inspect() override` — `return scan_directory(cfg_.model_path);`
    (currently duplicated in 3 subclasses).
  - `std::uint64_t ResidentWeightBytes() const override { return resident_bytes_; }`
- Remove `Inspect()`/`ResidentWeightBytes()` overrides from the three
  subclasses that currently carry identical/byte-returning implementations.

### 3.2 Subclass `Load()` rewrites

Each subclass keeps its distinct middle, replacing the common guard with the
helper. The `spdlog::error` tag moves into the model as an optional
`BeginLoad` logging note, or `BeginLoad` logs with the generic `[model]` tag —
**decide during implementation**; functionally equivalent because the log tag
is non-contractual.

- `DeepSeekV4FlashModel::Load` → `BeginLoad` + `FinishLoad`.
- `GlmV5Model::Load` → `BeginLoad` + `FinishLoad`.
- `QwenBaseModel::Load` → `BeginLoad` (after setting `tokenizer_optional_ =
  true` **before** `LoadTokenizerDb` — the current order at
  `qwen_base_model.cpp:28` must be preserved) + weight-pin + geometry +
  `FinishLoad`; set `resident_bytes_ = pw_.resident_bytes` after pin.

### 3.3 `ResidentWeightBytes` migration

- `GlmV5Model` / `QwenBaseModel`: drop the override; assign `resident_bytes_`
  after each weight pin. GLM pins in `Infer` (weights `pw_.Free()`d at cleanup),
  so `resident_bytes_` latest-write semantics apply and may reflect zero between
  infers — mirror previous override behaviour which read `pw_.resident_bytes`
  live. **Implementation detail:** to preserve exact live-read semantics,
  keep a `const` accessor; simplest faithful option is GLM/Qwen keep assigning
  `resident_bytes_` in `Unload()` too. Decide in plan; behavioural equivalence
  with today is the requirement.
- `DeepSeekV4FlashModel`: no override today → stays `0` via base default.

### 3.4 `num_layers` consistency

`QwenBaseModel::Infer` sets `r.num_layers` from `pw_.top` geometry (inspect
`QwenTopology` field names in `lsxcommon/qwen_inference_load.h` during
implementation; `ValidateGeometry(pw_.top)` is already the source). GLM stays
`glm::kGlmNumLayers`; DeepSeek unchanged.

## 4. Test actualization

### 4.1 Fix stale `test_glm_model.cpp` (`tests/test_glm_model.cpp:48-58`)

Rewrite the guard-assertions section: model loaded with synthetic dir has no
real pinned weights, so `Infer` fails only on invalid requests. Replace the
"not implemented" block with guard-semantics expectations consistent with
`ValidateInferRequest`:

- `max_gen_tokens = 0` → `Infer` returns `!r.ok` with `r.error_msg` set
  ("max_gen_tokens must be positive").
- `input_ids` empty with positive `max_gen_tokens` → `!r.ok`,
  "empty input_ids".
- `SelfCheck` with a 0-token request → `!sc.ok` (guard propagation).

Do NOT assert real-generation failure. `CHECK` moves to `REQUIRE`-grade guard
checks where deterministic.

### 4.2 New `tests/test_qwen_selfcheck.cpp`

Mirror `test_glm_selfcheck.cpp` structure:

- Real-model test, `SKIP` when `LSX_TEST_QWEN_MODEL_PATH` / default Qwen dir's
  `logestix/pinned` absent, CTest `SKIP_RETURN_CODE 4`.
- Prompt: `"The capital of France is"` framed per the Qwen tokenizer
  (copy the GLM chat-template encoding pattern — Qwen also uses a jinja chat
  template with `<|user|>`/`<|assistant|>` regions).
- Contract: `SelfCheck` passes with `first_gen_text == "Paris"`,
  `selfcheck_passed == true`, non-empty `final_logits`, and ≥3 stress cycles
  bit-exact `final_logits` (`LSX_TEST_QWEN_SELFCHECK_CYCLES` env override,
  default 3).
- Negative start: `REQUIRE(sc0.ok)`.

### 4.3 New `tests/test_deepseek_selfcheck.cpp`

Same structure, DeepSeek framing (raw BPE encode, no chat template — see
`test_deepseek_prompt_output.cpp:39-49`).

- `LSX_TEST_DEEPSEEK_MODEL_PATH` env / default `DeepSeek-V4-Flash` dir.
- Contract identical to Qwen selfcheck (Paris, 3 cycles bit-exact,
  `LSX_TEST_DEEPSEEK_SELFCHECK_CYCLES`).

### 4.4 CMake wiring

Add the two new test executables to `tests/CMakeLists.txt` following the
existing per-executable pattern (copy `test_glm_selfcheck` / `test_deepseek_*`
wiring; `POST_BUILD` arrow.dll copy applies).

## 5. Error handling

No new error paths. `BeginLoad` reuses existing failure messages
("empty model_path", tokenizer failure) so CLI/app error handling is
unaffected. ValidateInferRequest guard messages unchanged (locked by
`test_model_runtime.cpp`).

## 6. Verification

- Build: `cmake --preset windows-debug && cmake --build --preset windows-debug`
  (and Linux preset on demand).
- `ctest -R glm -R qwen -R deepseek -R model` subsets plus full suite;
  real-model SelfCheck tests SKIP without weights, exit code discipline intact.
- Confirm zero-behaviour-change: existing model tests
  (`test_deepseek_prompt_output`, `test_glm_prompt_output`, `test_model_runtime`,
  `test_model_engine`, `test_model_factory`) pass unchanged.
- Confirm `ResidentWeightBytes` equivalence via `test_model_runtime.cpp`
  (default 0) and a factory/model test for GLM/Qwen real-byte path when models
  are present.

## 7. Out of scope / open items for plan

- Whether `BeginLoad` logs `[model]` or a per-model tag.
- Exact `QwenTopology` field providing layer count for `num_layers`.
- Whether GLM/Qwen also refresh `resident_bytes_` in `Unload`.