# Unify Model & Model-Test File Naming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename DeepSeek-V4-Flash kernel/source files to carry the `deepseek_` family prefix, matching the uniform `glm_*`/`qwen_*` scheme, and update all build/include references so the project stays buildable.

**Architecture:** Pure file-rename refactor (no behavior change). Seven `inference_*.cu` sources + `inference_load.cuh` + `norm_kernels.cu` become `deepseek_inference_*.cu`/`.cuh` and `deepseek_norm_kernels.cu`; `inference_csa.h` and `norm_kernels.h` public headers are renamed to `deepseek_*`. The `lsxcommon_cu` CMake source list and every affected `#include` site are updated in lockstep. Shared engine files (`inference_gpu.h`, `shared_experts.cu`, model factory/orchestrator) are left untouched.

**Tech Stack:** C++20 / CUDA, CMake 3.25+ (CMakePresets), NVCC 13.2, Catch2 v3.

---

## File Structure

Files being **renamed** (via `git mv`, preserving history):

- `libs/lsxcommon/src/inference_load.cu` → `deepseek_inference_load.cu`
- `libs/lsxcommon/src/inference_load.cuh` → `deepseek_inference_load.cuh`
- `libs/lsxcommon/src/inference_norm.cu` → `deepseek_inference_norm.cu`
- `libs/lsxcommon/src/inference_moe.cu` → `deepseek_inference_moe.cu`
- `libs/lsxcommon/src/inference_head.cu` → `deepseek_inference_head.cu`
- `libs/lsxcommon/src/inference_attn.cu` → `deepseek_inference_attn.cu`
- `libs/lsxcommon/src/inference_csa.cu` → `deepseek_inference_csa.cu`
- `libs/lsxcommon/src/inference_pipeline.cu` → `deepseek_inference_pipeline.cu`
- `libs/lsxcommon/src/norm_kernels.cu` → `deepseek_norm_kernels.cu`
- `libs/lsxcommon/include/lsxcommon/inference_csa.h` → `deepseek_inference_csa.h`
- `libs/lsxcommon/include/lsxcommon/norm_kernels.h` → `deepseek_norm_kernels.h`

Files being **modified** (references only):

- `libs/lsxcommon/CMakeLists.txt` — `lsxcommon_cu` source list (lines 78-86)
- `libs/lsxcommon/src/inference_load.cu` — include
- `libs/lsxcommon/src/inference_moe.cu` — include
- `libs/lsxcommon/src/inference_attn.cu` — includes
- `libs/lsxcommon/src/inference_pipeline.cu` — includes
- `libs/lsxcommon/src/inference_csa.cu` — include
- `libs/lsxcommon/src/norm_kernels.cu` — include
- `libs/lsxcommon/src/model_factory.cpp` — include
- `libs/lsxcommon/src/benchmark_gpu.cu` — include
- `tests/test_deepseek_moe.cu` — include
- `AGENTS.md` — structure note at line 108

**Not touched:** `inference_gpu.h`, `shared_experts.cu`, `deepseek_v4_flash_model.cpp/.h`,
`model_*.cpp`, `lsx_orchestrator.cpp`, GLM/Qwen files, test executables
(`test_deepseek_loader`, `test_deepseek_moe`, `test_deepseek_inference_kernels`
names stay). `tests/test_deepseek_inference_kernels.cu` needs no change (it does
not include any renamed header).

Note: all source/header paths below are relative to the repo root
`D:\Projects\GitHub\logestix`. This is a Windows + CUDA build; all `git mv` and
`grep` commands run in PowerShell via the call operator / pwsh.

---

### Task 1: Rename the DeepSeek source files with `git mv`

**Files:**
- Rename: 8 `.cu`/`.cuh` sources under `libs/lsxcommon/src/`

- [ ] **Step 1: Rename the seven `inference_*.cu` sources**

Run (from repo root `D:\Projects\GitHub\logestix`):

```powershell
git mv libs/lsxcommon/src/inference_load.cu     libs/lsxcommon/src/deepseek_inference_load.cu
git mv libs/lsxcommon/src/inference_norm.cu     libs/lsxcommon/src/deepseek_inference_norm.cu
git mv libs/lsxcommon/src/inference_moe.cu      libs/lsxcommon/src/deepseek_inference_moe.cu
git mv libs/lsxcommon/src/inference_head.cu     libs/lsxcommon/src/deepseek_inference_head.cu
git mv libs/lsxcommon/src/inference_attn.cu     libs/lsxcommon/src/deepseek_inference_attn.cu
git mv libs/lsxcommon/src/inference_csa.cu      libs/lsxcommon/src/deepseek_inference_csa.cu
git mv libs/lsxcommon/src/inference_pipeline.cu libs/lsxcommon/src/deepseek_inference_pipeline.cu
```

Expected: no output (silent success). Verify:

```powershell
git status --short
```

Expected: 7 renamed entries shown.

- [ ] **Step 2: Rename `inference_load.cuh` and `norm_kernels.cu`**

```powershell
git mv libs/lsxcommon/src/inference_load.cuh libs/lsxcommon/src/deepseek_inference_load.cuh
git mv libs/lsxcommon/src/norm_kernels.cu    libs/lsxcommon/src/deepseek_norm_kernels.cu
```

- [ ] **Step 3: Rename the two public headers**

```powershell
git mv libs/lsxcommon/include/lsxcommon/inference_csa.h libs/lsxcommon/include/lsxcommon/deepseek_inference_csa.h
git mv libs/lsxcommon/include/lsxcommon/norm_kernels.h  libs/lsxcommon/include/lsxcommon/deepseek_norm_kernels.h
```

- [ ] **Step 4: Confirm no more `inference_*`/`norm_kernels` sources remain in `src/`**

Run:

```powershell
Get-ChildItem libs/lsxcommon/src -Filter "inference*" ; Get-ChildItem libs/lsxcommon/src -Filter "norm_kernels*"
```

Expected: only `deepseek_inference_*` matches appear; no bare `inference_*.cu`/`norm_kernels.cu`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: prefix DeepSeek kernel sources with deepseek_ family tag"
```

---

### Task 2: Update the `lsxcommon_cu` CMake source list

**Files:**
- Modify: `libs/lsxcommon/CMakeLists.txt:75-87`

- [ ] **Step 1: Update the `lsxcommon_cu` `add_library` source names**

In `libs/lsxcommon/CMakeLists.txt`, change the `lsxcommon_cu` object library
source list so the DeepSeek kernel entries use their new names. Replace:

```cmake
add_library(lsxcommon_cu OBJECT
    src/safetensors_gpu.cu
    src/benchmark_gpu.cu
    src/inference_load.cu
    src/inference_norm.cu
    src/inference_moe.cu
    src/inference_head.cu
    src/inference_attn.cu
    src/inference_csa.cu
    src/inference_pipeline.cu
    src/norm_kernels.cu
    src/shared_experts.cu
)
```

with:

```cmake
add_library(lsxcommon_cu OBJECT
    src/safetensors_gpu.cu
    src/benchmark_gpu.cu
    src/deepseek_inference_load.cu
    src/deepseek_inference_norm.cu
    src/deepseek_inference_moe.cu
    src/deepseek_inference_head.cu
    src/deepseek_inference_attn.cu
    src/deepseek_inference_csa.cu
    src/deepseek_inference_pipeline.cu
    src/deepseek_norm_kernels.cu
    src/shared_experts.cu
)
```

- [ ] **Step 2: Verify no stale source names remain in the file**

Run:

```powershell
Select-String -Path libs/lsxcommon/CMakeLists.txt -Pattern "inference_(load|norm|moe|head|attn|csa|pipeline)\.cu|norm_kernels\.cu"
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/CMakeLists.txt
git commit -m "build: update lsxcommon_cu source list for deepseek_ kernel renames"
```

---

### Task 3: Update includes inside the renamed DeepSeek kernel sources

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_moe.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_attn.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_csa.cu`
- Modify: `libs/lsxcommon/src/deepseek_norm_kernels.cu`

- [ ] **Step 1: `deepseek_inference_load.cu`**

In `libs/lsxcommon/src/deepseek_inference_load.cu`, change line 5:

```
#include "inference_load.cuh"
```

to:

```
#include "deepseek_inference_load.cuh"
```

- [ ] **Step 2: `deepseek_inference_moe.cu`**

In `libs/lsxcommon/src/deepseek_inference_moe.cu`, change line 4:

```
#include "inference_load.cuh"
```

to:

```
#include "deepseek_inference_load.cuh"
```

- [ ] **Step 3: `deepseek_inference_attn.cu`**

In `libs/lsxcommon/src/deepseek_inference_attn.cu`, change:
- line 9 `#include "lsxcommon/inference_csa.h"` →
  `#include "lsxcommon/deepseek_inference_csa.h"`
- line 11 `#include "lsxcommon/norm_kernels.h"` →
  `#include "lsxcommon/deepseek_norm_kernels.h"`
- line 12 `#include "inference_load.cuh"` → `#include "deepseek_inference_load.cuh"`

- [ ] **Step 4: `deepseek_inference_pipeline.cu`**

In `libs/lsxcommon/src/deepseek_inference_pipeline.cu`, change:
- line 2 `#include "lsxcommon/norm_kernels.h"` →
  `#include "lsxcommon/deepseek_norm_kernels.h"`
- line 5 `#include "lsxcommon/inference_csa.h"` →
  `#include "lsxcommon/deepseek_inference_csa.h"`
- line 6 `#include "inference_load.cuh"` → `#include "deepseek_inference_load.cuh"`

- [ ] **Step 5: `deepseek_inference_csa.cu`**

In `libs/lsxcommon/src/deepseek_inference_csa.cu`, change line 1:

```
#include "lsxcommon/inference_csa.h"
```

to:

```
#include "lsxcommon/deepseek_inference_csa.h"
```

- [ ] **Step 6: `deepseek_norm_kernels.cu`**

In `libs/lsxcommon/src/deepseek_norm_kernels.cu`, change line 1:

```
#include "lsxcommon/norm_kernels.h"
```

to:

```
#include "lsxcommon/deepseek_norm_kernels.h"
```

- [ ] **Step 7: Verify no stale kernel-header includes remain in `src/`**

Run:

```powershell
Select-String -Path libs/lsxcommon/src/*.cu,libs/lsxcommon/src/*.cuh,libs/lsxcommon/src/*.cpp -Pattern 'inference_load\.cuh|lsxcommon/inference_csa\.h|lsxcommon/norm_kernels\.h'
```

Expected: no matches in the 6 files above (note: `model_factory.cpp` and
`benchmark_gpu.cu` are updated in Task 4).

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/src
git commit -m "refactor: update includes inside renamed DeepSeek kernel sources"
```

---

### Task 4: Update remaining include sites (engine + tests)

**Files:**
- Modify: `libs/lsxcommon/src/model_factory.cpp`
- Modify: `libs/lsxcommon/src/benchmark_gpu.cu`
- Modify: `tests/test_deepseek_moe.cu`

- [ ] **Step 1: `model_factory.cpp`**

In `libs/lsxcommon/src/model_factory.cpp`, change line 7:

```
#include "inference_load.cuh"
```

to:

```
#include "deepseek_inference_load.cuh"
```

- [ ] **Step 2: `benchmark_gpu.cu`**

In `libs/lsxcommon/src/benchmark_gpu.cu`, change line 4:

```
#include "inference_load.cuh"
```

to:

```
#include "deepseek_inference_load.cuh"
```

- [ ] **Step 3: `tests/test_deepseek_moe.cu`**

In `tests/test_deepseek_moe.cu`, change line 23:

```
#include "inference_load.cuh"
```

to:

```
#include "deepseek_inference_load.cuh"
```

- [ ] **Step 4: Verify zero stale `inference_load.cuh` references remain outside `docs/` and `AGENTS.md`**

Run:

```powershell
Select-String -Path libs/lsxcommon/src/*.cu,libs/lsxcommon/src/*.cuh,libs/lsxcommon/src/*.cpp,tests/*.cu -Pattern 'inference_load\.cuh|lsxcommon/inference_csa\.h|lsxcommon/norm_kernels\.h'
```

Expected: **no matches** (all renamed references resolved).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/model_factory.cpp libs/lsxcommon/src/benchmark_gpu.cu tests/test_deepseek_moe.cu
git commit -m "refactor: update engine and test includes for deepseek_ kernel renames"
```

---

### Task 5: Update AGENTS.md structure note

**Files:**
- Modify: `AGENTS.md:108`

- [ ] **Step 1: Update the DeepSeek-kernel structure bullet**

In `AGENTS.md`, line 108 currently reads:

```
  - `libs/lsxcommon/src/inference_*.cu` + `inference_load.cuh` — DeepSeek-V4-Flash inference kernels: load, norm, moe, head, attn, csa, pipeline
```

Change it to:

```
  - `libs/lsxcommon/src/deepseek_inference_*.cu` + `deepseek_inference_load.cuh` + `deepseek_norm_kernels.cu` — DeepSeek-V4-Flash inference kernels: load, norm, moe, head, attn, csa, pipeline
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: update AGENTS.md for deepseek_ kernel file renames"
```

---

### Task 6: Configure and build to verify the rename

**Files:**
- Test: full project build (Windows VS 2026, CUDA)

- [ ] **Step 1: Configure with the windows-debug preset**

Run (may take several minutes; CUDA toolchain configure):

```powershell
cmake --preset windows-debug
```

Expected: configure succeeds with no errors.

- [ ] **Step 2: Build**

```powershell
cmake --build --preset windows-debug
```

Expected: build succeeds. Key confirmations:
- `lsxcommon_cu` object library compiles the 8 renamed `deepseek_*.cu` sources and `deepseek_norm_kernels.cu`
- no "cannot open source file" / "No such file" errors for the renamed headers
- `test_deepseek_moe` and `test_deepseek_inference_kernels` compile and link

- [ ] **Step 3: Run the DeepSeek test subset**

```powershell
ctest --preset windows-debug -R "deepseek"
```

Expected: all three DeepSeek tests pass (`test_deepseek_loader`,
`test_deepseek_moe`, `test_deepseek_inference_kernels`).

- [ ] **Step 4: Final stale-reference sweep**

Run:

```powershell
Select-String -Path libs/lsxcommon/src/*.cu,libs/lsxcommon/src/*.cuh,libs/lsxcommon/src/*.cpp,libs/lsxcommon/CMakeLists.txt,tests/*.cu,tests/*.cpp -Pattern 'inference_load\.cuh|lsxcommon/inference_csa\.h|lsxcommon/norm_kernels\.h|inference_(load|norm|moe|head|attn|csa|pipeline)\.cu|src/norm_kernels\.cu'
```

Expected: **no matches** (only historical `docs/` references may remain, which are intentionally untouched).

- [ ] **Step 5: Commit any incidental fixes**

If any manual fix was needed during build, commit it with a descriptive message.
Otherwise skip this step (no commit needed).

---

## Self-Review

**Spec coverage:**
- Rename set (7 sources + `inference_load.cuh` + `norm_kernels.cu` + 2 headers) → Tasks 1.
- CMake `lsxcommon_cu` source list update → Task 2.
- Include sites in the 7 kernels + `norm_kernels.cu` → Task 3.
- Engine/test include sites (`model_factory.cpp`, `benchmark_gpu.cu`, `test_deepseek_moe.cu`) → Task 4.
- AGENTS.md note → Task 5.
- Build + `ctest -R deepseek` verification → Task 6.
- `inference_gpu.h` / `shared_experts.cu` kept shared → explicitly excluded in File Structure.

**Placeholder scan:** All steps contain exact paths, commands, and replacement
strings. No TBD/TODO placeholders.

**Type consistency:** All `git mv` targets (Task 1) match the CMake names (Task
2) and include strings (Tasks 3-4) exactly:
`deepseek_inference_load/norm/moe/head/attn/csa/pipeline.{cu,cuh}`,
`deepseek_norm_kernels.{cu,h}`, `deepseek_inference_csa.h`.
