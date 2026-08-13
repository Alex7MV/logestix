# DeepSeek Tests Separation Design

## Goal

Extract DeepSeek-specific tests into dedicated `test_deepseek_*.cpp` files so
GLM and DeepSeek suites are autonomous and each model can be tested
independently. GLM tests already live in `test_glm_*.cpp`; the DeepSeek tests
currently sit in generically-named files mixed with model-agnostic tests.

## Non-Goals

- Do NOT restructure GLM tests (`test_glm_*.cpp` already isolated).
- Do NOT split shared/agnostic files: `test_core.cpp` (dependency
  validation), `test_tokenizer.cpp` (shared tokenizer handling both formats),
  `test_mosaic_network.cpp` (network layer), `test_integrated_pipeline.cpp`
  (orchestrator exercising both branches by design).
- Do NOT touch `test_model_factory.cpp` — the factory is shared
  infrastructure; its tests stay as-is.

## Changes

### Renames (git mv, preserving history)

| Current file | New file | Content |
|---|---|---|
| `tests/test_loader.cpp` | `tests/test_deepseek_loader.cpp` | ExpertSchema, PackedExpert, E4M3 codec, Loader round-trip (DeepSeek expert path) |
| `tests/test_inference_kernels.cu` | `tests/test_deepseek_inference_kernels.cu` | E2M1/E8M0, act_quant, RMSNorm, SwiGLU, CSA Compressor, routed-expert FFN, NVFP4 codec (DeepSeek kernels) |

`tests/test_model_factory.cpp` is NOT renamed.

### CMake (`tests/CMakeLists.txt`)

Rename targets and `add_test` names to mirror the file renames:

- `test_loader` → `test_deepseek_loader`
  (executable, `target_link_libraries`, POST_BUILD arrow.dll copies, `add_test`)
- OBJECT library `test_inference_kernels_cu` → `test_deepseek_inference_kernels_cu`
  and executable `test_inference_kernels` → `test_deepseek_inference_kernels`.
  Preserve the entire MSVC/`/utf-8` workaround block verbatim:
  `CUDA_SEPARABLE_COMPILATION OFF`, `CUDA_ARCHITECTURES OFF`,
  `--extended-lambda`, `-std=c++20`, `--generate-code=arch=compute_120,code=sm_120/121`,
  `-Xcompiler=/utf-8` (Windows only), explicit include dirs, CCCL defines,
  `LINKER_LANGUAGE CXX`, POST_BUILD arrow.dll copies, `add_test`.

### Catch2 tags

Add `[deepseek]` to every TEST_CASE in the two renamed files, symmetric with
the existing `[glm]` tag. Existing tags (e.g. `[loader]`, `[inference]`) are
retained. This enables `ctest -R` filtering by model without running the full
suite.

### Model path autonomy (unchanged pattern)

- GLM real-model tests: `LSX_TEST_GLM_MODEL_PATH` env, default
  `/mnt/llm_models/models/GLM-5.2`, `SKIP` when `logestix/pinned` absent.
- DeepSeek real-model factory test: `LSX_TEST_MODEL_PATH` env, `SKIP` when
  index absent.
- The two renamed DeepSeek files contain only synthetic fixtures — no real
  model path required, so each suite runs standalone.

## Verification

1. Configure + build `linux-release-znver4` preset (full rebuild).
2. `ctest -R deepseek` → runs `test_deepseek_loader`,
   `test_deepseek_inference_kernels` (2 tests pass).
3. `ctest -R glm` → runs all 11 GLM executables independently.
4. Full `ctest` → all 22 tests still pass (same total count; two entries
   renamed).
5. `git status` shows pure renames (no content drift) + CMake/tag edits.
