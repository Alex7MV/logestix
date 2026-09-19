# Test categories: fast / inference / perf

Date: 2026-09-10
Status: approved (design), implemented per the TDD plan

> **Addendum (2026-09-10, during implementation):** a fourth category
> `convert` was added — model converters, run when the converter is edited (router),
> when debugging a model, or explicitly (`ctest -L convert`). 9 `*_convert`
> tests moved from `fast` to `convert`, and `test_nvfp4_pack` moved from `inference`
> (the converters' GPU quantize contract). Router: files
> `libs/lsxcommon/*(converter|_convert|tokenizer_convert)` → convert,
> `safetensors_gpu.cu` (the quantization engine) → convert+inference; the
> `tests/` rule raises convert+inference. Presets: +5 `<preset>-convert`.
> Windows convert/inference sweeps run with the filter `^(test_qwen|test_gemma)`.
> Final numbers: fast 33 (+4 app), convert 10, inference 77, perf 6.

## Problem

109 `add_test` entries in `tests/CMakeLists.txt` (125 registered tests
counting foreach loops) run in one monolithic
`ctest --preset <preset>` pass — hours on a machine with checkpoints. Filtering
exists only by name (`ctest -R`); there are no categories.

## Goals

1. Split tests into three categories:
   - `fast` — run on every build (host-only, minutes);
   - `inference` — the inference contour (CUDA kernels, loaders, models,
     selfchecks); run automatically when the inference
     contour changed (git diff);
   - `perf` — perf benchmarks on real weights; only via a separate command,
     never part of the automated run.
2. The base `ctest --preset <preset>` runs only `fast`.
3. Autodetect "the inference contour changed" — via `git diff`.
4. Unify file naming in `tests/` (everything `test_*`).
5. Category discipline is protected by a guard test (TDD).

## Non-mechanisms (what we do not do)

- No physical separation of files into directories (`tests/fast/` etc.) —
  labels solve the task without moving 100+ files.
- No renaming of the 125 test targets — prefixes in names
  (`fast_*`/`inf_*`) would break the `ctest -R glm` recipes from AGENTS.md.
- CI logic is out of scope (there is no CI config in the repo); the router works locally
  on both OSes.

## Design

### 1. Categories = CTest LABELS

Each test gets exactly one label: `fast` | `inference` | `perf`.
Assignment — via three canonical blocks at the end of `tests/CMakeLists.txt`
(after all `add_test`), in a regular form convenient for static
parsing by the guard test:

```cmake
# ── Test category labels (fast | inference | perf) ────────────────────────
set_tests_properties(<fast tests...>     PROPERTIES LABELS fast)
set_tests_properties(<inference tests...> PROPERTIES LABELS inference)
set_tests_properties(<perf tests...>      PROPERTIES LABELS perf)
```

Existing properties (`SKIP_RETURN_CODE 4`, `ENVIRONMENT`) are not touched.
The `${qt}` names from foreach loops are listed out explicitly in the blocks.

### 2. testPresets

Three testPreset variants are added to each of the 5 base presets
(15 total):

- `<preset>-fast` — `filter.include.label = "^fast$"` (matches the base
  preset in meaning; the base presets also switch to `^fast$`);
- `<preset>-inference` — `label = "^inference$"`;
- `<preset>-perf` — `label = "^perf$"`.

The base testPresets (`linux-debug` etc.) get the same `^fast$` filter —
the default run becomes fast. The full run is available manually:
`ctest -L "^(fast|inference|perf)$"`.

Windows restriction (qwen|gemma models only): presets are pure label filters;
the `ctest -R "qwen|gemma"` rule for the inference run on Windows is applied by the
router script (see below) and remains documented practice in AGENTS.md.

### 3. Router `tools/test_router.cmake`

CMake script mode (cross-platform, no compilation):

```
cmake -P tools/test_router.cmake -- --preset linux-debug [--base origin/main] [--all] [--list]
```

Algorithm:

1. `git diff --name-only <base>...HEAD` (base defaults to the merge-base with
   `origin/main`; no git/no diff → `fast` only).
2. Classify each changed file against an ordered rule table
   (first match wins); the labels accumulate into a set:
   - `\.cuh?$` anywhere → `inference`;
   - `libs/lsxcommon/...` (including its `CMakeLists.txt`) → `inference`, EXCEPT
     files matching the fast lexicon
     `(tokenizer|unicode|pretokenizer|path_registry|schema|converter|_convert|orchestrator)`
     (conservative default: an unknown new lsxcommon file → inference);
   - `libs/lsxnetwork/**`, `apps/**` → `fast`;
   - `tests/**` → `inference` (the test infra changed; the rule sits BEFORE the
     tooling rule so that `tests/CMakeLists.txt` is not swallowed by `\.txt$`);
   - root `CMakeLists.txt`/`CMakePresets.json`, `docs/**`, `*.md`
     (top-level), `tools/**`, `cmake/**` → `fast`;
   - everything else → `fast`.
3. Result: `ctest --preset <preset> -L "^(fast|inference)$"`.
   On WIN32 the inference run additionally gets `-R "^(test_qwen|test_gemma)"`.
4. Flags: `--all` — force `fast+inference`; `--list` — only show the file
   classification and the selected labels, do not run ctest.

The rule table lives at the top of the script as a list of `lsx_router_rule(<regex> <labels>)`
— explicit, reviewable, checked by the guard test. Perf never enters the automated run:
the router only knows `fast` and `inference`.

### 4. Guard test `tests/test_ctest_labels_guard.cpp` (TDD anchor)

A host-only Catch2 test modeled on `test_code_naming_guard` (it scans
`LSX_SOURCE_DIR`). It asserts:

1. Every test (all names from `add_test(NAME ...)` plus all elements of the
   foreach lists in `tests/CMakeLists.txt`) appears in exactly one
   `set_tests_properties(... PROPERTIES LABELS <cat>)` with a category from
   {fast, inference, perf}.
2. Every perf test has `SKIP_RETURN_CODE 4` in its
   `set_tests_properties`.
3. `CMakePresets.json`: all testPreset `label` filters reference only
   valid categories; for every configurePreset the `-fast`,
   `-inference`, `-perf` variants exist.
4. `tools/test_router.cmake`: all `lsx_router_rule` labels ∈ {fast, inference};
   the `perf` label in the rules is forbidden.
5. `tests/` naming: every file matches `^test_.+\.(cpp|cu|h)$`.

Plus a runtime pin: a mini-parser (extracted into a testable function) on a
synthetic CMake snippet — a duplicate label, an unknown label, a missed test → FAILURE.
This makes the guard self-checking, not merely "a regex over the file".

### 5. Fixture name unification

5 `<family>_test_fixture.h` headers → `test_<family>_fixture.h`:

- `glm_moe_test_fixture.h` → `test_glm_moe_fixture.h`
- `deepseek_moe_test_fixture.h` → `test_deepseek_moe_fixture.h`
- `gemma_moe_test_fixture.h` → `test_gemma_moe_fixture.h`
- `qwen_dense_test_fixture.h` → `test_qwen_dense_fixture.h`
- `qwen_moe_test_fixture.h` → `test_qwen_moe_fixture.h`
- `qwen_moe_exp_test_fixture.h` → `test_qwen_moe_exp_fixture.h`
- `kimi_moe_test_fixture.h` → `test_kimi_moe_fixture.h`

(7 files; update ~30 `#include`s in the tests. A mechanical refactor without
changing behaviour; guard item 5 is green after that.)

### 6. Documentation

AGENTS.md, Test section: the category table, commands
(`ctest --preset <p>` = fast; `ctest -L inference`; `ctest -L perf`;
`cmake -P tools/test_router.cmake -- --preset <p>`), and the rule
"a new test must get a label in the same commit" (penalty — a red
guard in CI).

## Classification (full, 125 + 1 new)

### perf (6)

test_blackwell_perf_glm, test_blackwell_perf_dense,
test_blackwell_perf_moe_qwen, test_blackwell_perf_moe_gemma,
test_blackwell_perf_deepseek, test_kimi_moe_perf

### fast (41 + 1 new guard)

test_core, test_fast_load, test_geometry_guard, test_gpu_perf,
test_schema_views, test_deepseek_moe_loader, test_expert_store,
test_qwen_dense_convert, test_glm_moe_expert_cache, test_model_integrity,
test_qwen_dense_converter_integrity, test_glm_moe_convert,
test_glm_moe_flash_convert, test_kimi_moe_convert, test_model_factory,
test_generation_cli, test_glm_moe_prompt_framing, test_gemma_prompt_framing,
test_gemma_moe_convert, test_model_runtime, test_model_engine,
test_tokenizer, test_mosaic_network, test_mosaic_frontend_route,
test_token_translator, test_network_translator_weld, test_integrated_pipeline,
test_path_registry, test_vram_estimate, test_resource_governor,
test_hardware_profile, test_lifecycle_plan, test_runtime_tuner,
test_kimi_moe_tokenizer, test_code_naming_guard, test_absl_format_guard,
test_kimi_moe_expert_cache, test_qwen_moe_convert, test_qwen_moe_36_convert,
test_qwen_moe_exp_convert, test_model_independence_guard,
**test_ctest_labels_guard** (new)

### inference (78)

**CUDA kernels and model-neutral (.cu):** test_sampling, test_lsx_nvfp4_dispatch,
test_perf_knobs_apply, test_blackwell_tc_mma, test_lsx_kernels,
test_blackwell_micro_gemm, test_nvfp4_pack,
test_deepseek_moe_inference_kernels, test_deepseek_moe_moe,
test_deepseek_moe_arena, test_qwen_moe_kernels, test_qwen_moe_exp_kernels,
test_glm_moe_flash_kernels, test_kimi_moe_int4_dequant, test_kimi_moe_int4_real,
test_kimi_moe_router, test_kimi_moe_attention, test_kimi_moe_experts,
test_kimi_moe_tc_batch, test_glm_moe_load_fast, test_deepseek_moe_load_fast,
test_glm_moe_model_lifecycle

**Qwen dense:** test_qwen_dense_loader, test_qwen_dense_model,
test_qwen_dense_27b_model, test_qwen_dense_kv_cache, test_qwen_dense_kv_parity,
test_qwen_dense_selfcheck, test_qwen_dense_27b_selfcheck,
test_qwen_dense_norm, test_qwen_dense_rotary, test_qwen_dense_shuffle_reduce,
test_qwen_dense_mlp, test_qwen_dense_nvfp4, test_qwen_nvfp4_tile_gemm,
test_qwen_nvfp4_gemv, test_qwen_dense_full_attn,
test_qwen_dense_full_attn_parallel, test_qwen_dense_linear_attn,
test_qwen_dense_fused_mlp, test_qwen_dense_fused_linear_attn

**Qwen MoE (3.5/3.6/exp):** test_qwen_moe_loader, test_qwen_moe_model,
test_qwen_moe_selfcheck, test_qwen_moe_36_model, test_qwen_moe_36_selfcheck,
test_qwen_moe_exp_loader, test_qwen_moe_exp_model, test_qwen_moe_exp_selfcheck

**GLM-5.2:** test_glm_moe_loader, test_glm_moe_model, test_glm_moe_selfcheck,
test_glm_moe_inference, test_glm_moe_full_compute, test_glm_moe_prefill_seqp,
test_glm_moe_prompt_output, test_glm_moe_attention, test_glm_moe_dense_compute,
test_glm_moe_layer_compute, test_glm_moe_experts, test_glm_moe_moe_router,
test_glm_moe_moe_compute

**GLM-5.3-Flash:** test_glm_moe_flash_loader, test_glm_moe_flash_model,
test_glm_moe_flash_selfcheck

**DeepSeek:** test_deepseek_moe_prompt_output, test_deepseek_moe_selfcheck

**Gemma:** test_gemma_moe_loader, test_gemma_moe_model, test_gemma_moe_selfcheck,
test_gemma_moe_norm, test_gemma_moe_rotary, test_gemma_moe_attn,
test_gemma_moe_moe

**Kimi:** test_kimi_moe_loader, test_kimi_moe_model, test_kimi_moe_selfcheck,
test_kimi_moe_inference

Principle for borderline cases: a test that exercises the CUDA contour (even on a
synthetic fixture) is `inference`; host-only is `fast`.
`test_integrated_pipeline` and `test_model_engine` are host-only (mocks), → fast.
`test_expert_store`/`test_glm_moe_expert_cache`/`test_kimi_moe_expert_cache`
— host bookkeeping → fast.

## TDD plan (brief; details in the implementation plan)

1. **Cycle 1 — guard RED→GREEN.** Write `test_ctest_labels_guard.cpp`
   (items 1–4 above) → RED (no LABELS). Add the label blocks,
   testPresets, router → GREEN.
2. **Cycle 2 — naming RED→GREEN.** Add item 5 to the guard → RED
   (fixtures). Rename the 6 fixtures + includes → GREEN.
3. **Verification:** configure (the preset schema is validated by CMake),
   `ctest -L fast` green, `ctest -L perf` selects exactly 6, the router
   `--list` classifies a test diff correctly.
