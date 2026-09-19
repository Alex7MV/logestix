# Spec: Unify model-code file naming (family + type + role)

Date: 2026-08-27
Status: approved by the operator
Scope: **code** only (`libs/lsxcommon/src`, `libs/lsxcommon/include/lsxcommon`, `tests/`,
CMakeLists, AGENTS.md). Model data files (`.arrow`, `model.index.json`, conversion
directories) are NOT affected.

## 1. Problem

Model-code file names are inconsistent:

- the architecture-type marker (`moe`/`dense`) is almost never present: `glm_convert.cpp`
  is actually MoE, but that is not visible from the name; `qwen_convert.cpp` vs `qwen_moe_convert.cpp`;
- model versions are mixed with file names in different styles: `glm_v5_model`,
  `deepseek_v4_flash_model`, `qwen_3_8_27b_model`, `gemma_4_26b_a4b_model`;
- prefixes within one family differ: `gemma4_*` vs `gemma_4_26b_a4b_*` vs `gemma_inference.cu`;
- role suffixes are a grab bag: `_inference_load`, `load_internal`, empty (`glm_inference.cu`).

## 2. Target scheme

```
<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>
```

| Token | Meaning |
|---|---|
| `family` | lowercase family name: glm, qwen, gemma, kimi, deepseek |
| `type` | **required**: moe or dense |
| `variant` | only on a collision of two models of one type within a family |
| `role` | short vocabulary: convert, load, model, inference, kernels, … (the registry is extended in the guard test) |
| `subsystem` | optional split of a large kernel: attn, csa, head, moe, pipeline |

Model versions are removed from file names forever — they live in class names and `kType`
(`GlmV5Model::kType = "glm-5.2"`). Class/namespace names are NOT changed.

## 3. Rename map

The `.cpp`/`.h`/`.cu` pairs are renamed consistently.

### GLM (moe)

| Now | Becomes |
|---|---|
| `glm_convert.{cpp,h}` | `glm_moe_convert.{cpp,h}` |
| `glm_inference_load.{cpp,cu}` + `include/glm_inference_load.h` | `glm_moe_load.{cpp,cu}` + `glm_moe_load.h` |
| `src/glm_load_internal.h` | `glm_moe_load_internal.h` |
| `glm_v5_model.{cpp,h}` | `glm_moe_model.{cpp,h}` |
| `glm_inference.cu` | `glm_moe_inference.cu` |
| `glm_device_math.h` | `glm_moe_device_math.h` |
| `glm_expert_kernels.cuh` | `glm_moe_expert_kernels.cuh` |
| `include/glm_expert_cache.h` | `glm_moe_expert_cache.h` |

### Qwen dense

| Now | Becomes |
|---|---|
| `qwen_convert.{cpp,h}` | `qwen_dense_convert.{cpp,h}` |
| `qwen_v5_model.{cpp,h}` | `qwen_dense_model.{cpp,h}` |
| `qwen_3_8_27b_model.{cpp,h}` | `qwen_dense_27b_model.{cpp,h}` |

**Verification note (planning):** the line `qwen_inference.cu → qwen_dense_inference.cu`
was removed — grep showed `qwen_inference.h` inclusions from `qwen_moe_inference.{h,cu}`,
`qwen_3_5_moe_model.h` and the MoE tests, i.e. the file is cross-type → moved to the allowlist (item 4).

### Qwen MoE

| Now | Becomes |
|---|---|
| `qwen_moe_inference_load.{cpp,h}` | `qwen_moe_load.{cpp,h}` |
| `qwen_3_5_moe_model.{cpp,h}` | `qwen_moe_model.{cpp,h}` |

`qwen_moe_{convert,inference}.{cpp,cu}`, `qwen_moe_constants.h` already match the scheme.
`qwen_moe_inference.h` (include) stays.

### Gemma (moe)

| Now | Becomes |
|---|---|
| `gemma4_convert.{cpp,h}` | `gemma_moe_convert.{cpp,h}` |
| `gemma4_inference_load.{cpp,h}` | `gemma_moe_load.{cpp,h}` |
| `include/gemma4_inference.h` | `gemma_moe_inference.h` |
| `gemma_4_26b_a4b_model.{cpp,h}` | `gemma_moe_model.{cpp,h}` |
| `gemma_inference.cu` | `gemma_moe_inference.cu` |

### Kimi (moe)

| Now | Becomes |
|---|---|
| `kimi_convert.{cpp,h}` | `kimi_moe_convert.{cpp,h}` |
| `kimi_inference_load.{cpp,h}` | `kimi_moe_load.{cpp,h}` |
| `kimi_v5_model.{cpp,h}` | `kimi_moe_model.{cpp,h}` |
| `kimi_inference.cu` | `kimi_moe_inference.cu` |
| `kimi_inference_kernels.h` | `kimi_moe_kernels.h` |

### DeepSeek (moe)

| Now | Becomes |
|---|---|
| `deepseek_v4_flash_model.{cpp,h}` | `deepseek_moe_model.{cpp,h}` |
| `deepseek_inference_load.{cu,cuh}` | `deepseek_moe_load.{cu,cuh}` |
| `deepseek_inference_{attn,csa,head,moe,pipeline}.cu` | `deepseek_moe_inference_{attn,csa,head,moe,pipeline}.cu` |
| `deepseek_norm_kernels.{cu,h}` | `deepseek_moe_norm_kernels.{cu,h}` |

## 4. Files outside the scheme (guard-test allowlist)

Not renamed (shared infrastructure or cross-type files of a family):

- shared infrastructure: `lsx_kernels_*`, `safetensors_gpu.*`, `safetensors.*`, `bf16_convert.cu`,
  `nvfp4_*`, `model_*`, `converter.*`, `tokenizer*`, `pretokenizer.cpp`, `unicode*`,
  `schema.cpp`, `sampling.*`, `benchmark_gpu.*`, `path_registry.*`, `resource_governor.*`,
  `runtime_tuner.*`, `vram_estimate.*`, `lifecycle_strategy.*`, `generation_cli.*`,
  `prompt_framing.*`, `log.cpp`, `common.*`, `core.cpp`, `expert_store.*`, `fast_load.*`,
  `inference_gpu.h`, `diagnostics.h`, `shared_experts.*`, `byte_unicode.h` etc.;
- cross-type Qwen files: `qwen_base_model.{cpp,h}`, `qwen_inference_load.{cpp,h}`,
  `qwen_inference.{h,cu}`, `qwen_device_math.h`, `qwen_reduce.cuh`, `qwen_tc_gemm.{cu,h}`;
- Gemma base: `gemma_base_model.{cpp,h}`.

The membership of each "border" file is verified by grepping inclusions during the
implementation plan: if a file turns out to be used by exactly one model of one type,
it is moved from the allowlist under the scheme.

## 5. Tests

Tests mirror the scheme: `test_<family>_<type>_<topic>.cpp/.cu`, fixtures
`<family>_<type>_test_fixture.h`. Correspondences:

- `test_glm_*` → `test_glm_moe_*`; `glm_test_fixture.h` → `glm_moe_test_fixture.h`;
- `test_qwen_selfcheck`, `test_qwen_loader`, `test_qwen_model`, … → `test_qwen_dense_*`;
- `test_deepseek_*` → `test_deepseek_moe_*`; `test_kimi_*` → `test_kimi_moe_*`;
  `test_gemma4_*` → `test_gemma_moe_*`;
- the qwen MoE tests stay (`test_qwen_moe_*`);
- the membership of shared qwen-kernel tests (`test_qwen_nvfp4_*`, `test_qwen_rotary`, …)
  is determined by the TU actually tested: dense only → `_dense_`; both → stays
  in the allowlist without a marker;
- the `ctest -R "qwen|gemma"` filters from AGENTS.md keep working.

## 6. Continuous convention control (guard test)

A new host-only Catch2 test `tests/test_code_naming_guard.cpp`:

- recursively scans `libs/lsxcommon/src/**` and `libs/lsxcommon/include/lsxcommon/**`
  (the repository root is passed via the compile-time macro `LSX_SOURCE_DIR` from
  `tests/CMakeLists.txt`);
- a file whose stem begins with a known family must match the scheme regex from item 2 with
  the extensions `{cpp, h, cu, cuh}`; failure — a failure with exact diagnostics
  (file name, expected pattern, nearest correct variant);
- stems outside the known families and the listed exceptions must be in the allowlist
  given as a list of literals with a rationale in a comment;
- the family / role / subsystem / variant registries live as constants inside the test and
  are extended when a new model is added **in the same commit** as the new files.

Thus TDD also works for future models: creating a file outside the scheme fails CI
at the moment it is added.

## 7. Accompanying edits

1. All `#include "..."` across the tree (src, include, apps/, tests/) — a textual replacement of header names.
2. `libs/lsxcommon/CMakeLists.txt` — the source lists of the OBJECT/CUDA libraries.
3. `tests/CMakeLists.txt` — test targets, fixtures, the `LSX_SOURCE_DIR` macro.
4. AGENTS.md: the Structure section (paths), the Model Ingestion Protocol reference anchors, a new section
   "Model Code File Naming Convention" with the scheme, role vocabulary and a new-model checklist.

## 8. Work order (TDD)

1. Write the guard test against the current tree → red, the violation list matches the rename
   map (the test = an executable specification).
2. Mechanical rename of all files + edits to include/CMakeLists/AGENTS.md → guard green.
3. Full windows-debug rebuild + `ctest` (Windows: only the `ctest -R "qwen|gemma"` filter),
   confirming bit-identical behaviour — only file names change.
4. Commits: a single atomic `refactor:` (guard + rename together — CI never sees a red
   state); AGENTS.md may be a separate `docs:` commit after.

## 9. Definition of done

- In `src/`+`include/` no model file is left without a `moe`/`dense` marker (except the allowlist).
- The guard test is green and included in the standard ctest run.
- All target builds (windows-debug) pass; the model tests produce the same results.
- AGENTS.md contains the convention and updated paths.
