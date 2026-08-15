# Unify Model & Model-Test File Naming

Date: 2026-08-15
Status: Approved

## Problem

The multi-app monorepo hosts three model families (GLM-5.2, Qwen3.5-9B,
DeepSeek-V4-Flash). Their source and kernel files follow a *nearly* uniform
`<family>_<...>` prefix scheme, with one genuine inconsistency:

- GLM and Qwen expose their kernels under a **family-prefixed** name
  (`glm_inference.cu`, `qwen_inference.cu`).
- DeepSeek's kernels are **unprefixed** (`inference_load.cu`,
  `inference_norm.cu`, `inference_moe.cu`, `inference_head.cu`,
  `inference_attn.cu`, `inference_csa.cu`, `inference_pipeline.cu`,
  `inference_load.cuh`) plus the generic `norm_kernels.cu`/`.h` and
  `inference_csa.h` public headers.

This makes DeepSeek's files hard to distinguish from shared engine utility,
complicates discovery, and sets a bad precedent for adding new models.

## Goal

Unify all DeepSeek model-source files under the `deepseek_` family prefix so
that every model's files follow one consistent scheme, matching the pattern
already used by GLM (`glm_*`) and Qwen (`qwen_*`). This makes it easy to reason
about and to add new models in the future.

## Naming Scheme

- **Model class files**: `<family>_v5_model.cpp/.h` — GLM/Qwen already conform
  (`glm_v5_model.*`, `qwen_v5_model.*`). DeepSeek keeps its real generation tag
  `deepseek_v4_flash_model.cpp/.h` because its architecture generation is
  v4-flash (registry `type` = `deepseek-v4-flash`), not v5.
- **Kernel files**: `<family>_inference*.cu` and `<family>_norm_kernels.cu`.
- **Tests**: `test_<family>_<subsystem>` — already uniform across GLM, Qwen,
  and DeepSeek; no renames required.

## Non-Goal / Kept as Shared

- `inference_gpu.h` — core engine header exposing `gpu::InferenceResult`
  (`kType` in the shared `IModel` interface via `model.h:8`). Model-agnostic.
- `shared_experts.cu` — shared routed-expert / NVFP4 decode primitives.
- `model_factory.cpp`, `model_registry.cpp`, `lsx_orchestrator.cpp` — engine /
  orchestration, not per-model.

## Rename Set

### Sources (`libs/lsxcommon/src/`)

| From | To |
|---|---|
| `inference_load.cu` / `inference_load.cuh` | `deepseek_inference_load.cu` / `.cuh` |
| `inference_norm.cu` | `deepseek_inference_norm.cu` |
| `inference_moe.cu` | `deepseek_inference_moe.cu` |
| `inference_head.cu` | `deepseek_inference_head.cu` |
| `inference_attn.cu` | `deepseek_inference_attn.cu` |
| `inference_csa.cu` | `deepseek_inference_csa.cu` |
| `inference_pipeline.cu` | `deepseek_inference_pipeline.cu` |
| `norm_kernels.cu` | `deepseek_norm_kernels.cu` |

### Public headers (`libs/lsxcommon/include/lsxcommon/`)

| From | To |
|---|---|
| `inference_csa.h` | `deepseek_inference_csa.h` |
| `norm_kernels.h` | `deepseek_norm_kernels.h` |

## Build Plumbing (mandatory companion)

File renames are only valid when the build stays green:

- `libs/lsxcommon/CMakeLists.txt` — `lsxcommon_cu` source list (lines 78-86)
  updated to the new `.cu` names.
- Include sites updated to the new names:
  - the 7 DeepSeek `.cu` files (`inference_*.cu`) and `norm_kernels.cu`
  - `deepseek_v4_flash_model.cpp`
  - `model_factory.cpp` (`#include "inference_load.cuh"` →
    `"deepseek_inference_load.cuh"`)
  - `testmodel_gpu.cu` (`#include "inference_load.cuh"` → new name)
  - `tests/test_deepseek_moe.cu` (`#include "inference_load.cuh"` → new name)
  - `tests/test_deepseek_inference_kernels.cu` (any `inference_*.cuh` includes)

## Not Touched

- `inference_gpu.h`, `shared_experts.cu`, engine/orchestration files.
- GLM/Qwen/model files and all test executables (already uniform).
- Historical design/plan docs under `docs/superpowers/` (kept as-is).

## Docs

- `AGENTS.md:108` updated: `inference_*.cu + inference_load.cuh` →
  `deepseek_inference_*.cu + deepseek_inference_load.cuh`.

## Verification

- Configure + build succeeds on Windows (`cmake --preset windows-debug`);
  CUDA object libs (`lsxcommon_cu`) compile with the renamed sources.
- `ctest` DeepSeek tests still pass (`test_deepseek_inference_kernels`,
  `test_deepseek_moe`, `test_deepseek_loader`).
- `grep -R "inference_load\.cuh\|norm_kernels\|inference_csa\|inference_attn\.cu\|inference_moe\.cu\|inference_head\.cu\|inference_pipeline\.cu\|inference_norm\.cu"` returns no stale references outside `docs/`.
