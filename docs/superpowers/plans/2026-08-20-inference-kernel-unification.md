# CUDA Inference Kernel Unification — Implementation Plan (§3.1 + §3.3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse duplicated NVFP4/E4M3 decode primitives into one shared device codec, and merge the GLM-internal twin kernels, preserving exact bit-level behavior.

**Architecture:** Add `lsxcommon/src/nvfp4_codec.h` (device-only `__constant__` E2M1 table + `__device__` E4M3 scale), reachable by all three CUDA object libs via `LSXCOMMON_CU_INCLUDES` (add `src`). Migrate the 4 local copies to it. Then unify GLM's duplicated GEMM/RMSNorm/SwiGLU kernels via an `extern __global__` declaration pattern (avoids ODR across the single `lsxcommon_glm_cu` object lib).

**Tech Stack:** CUDA/NVCC 13.3 (Blackwell CC 120/121), C++20, CMake. GLM targets compile with `-fmad=false` (bit-exact). No host-linking private device symbols → no ODR risk for the codec.

---
## File map

- Create: `libs/lsxcommon/src/nvfp4_codec.h`
- Modify: `libs/lsxcommon/src/shared_experts.cu` (use codec, drop local table/fn)
- Modify: `libs/lsxcommon/src/glm_inference_load.cu` (use codec; drop `kGlmE2M1Table`/`GlmE4M3P2ToFloat`; merge `GlmSwigluKernel`, `GlmMoERmsNormKernel`, `GlmGemmFp64Kernel` to extern-decl)
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (use codec, drop `qwen_e2m1_tab`/`qwen_e4m3_scale`)
- Modify: `libs/lsxcommon/src/glm_inference.cu` (merge `GlmRmsNormKernel`/`GlmSwigluClampKernel`/`GlmRealSwigluKernel`/`GlmGemmFp64Kernel` to canonical defs)
- Modify: `libs/lsxcommon/src/benchmark_gpu.cu` (use codec `E4M3ScaleToFloat`; keep `nvfp4_to_float` — it is consumed by `dequant_nvfp4_to_half_kernel`)
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src` to `LSXCOMMON_CU_INCLUDES`)
- Test: existing `test_glm_*`, `test_deepseek_inference_kernels.cu`, `test_qwen_model`, `test_deepseek_loader`

## Critical correctness notes (read first)

1. **E4M3 semantics differ subtly by guard.** `shared_experts.qwen`/benchmark use `if (e4==0) return 0`; GLM uses `if (e4m3==0) return 0`. On VALID power-of-two block scales (all three models store `value = 2^exp`, mantissa bits unused/zero) both are equal because `e4m3==0 ⟺ e4==0`. The unified function uses `e4==0` (matches 3 of 4 sites). Before committing Task 2, run the bit-exact GLM selfcheck to confirm no mantissa-set-block encodings exist in real checkpoints.
2. **`nvfp4_to_float` (benchmark_gpu.cu:33) is NOT dead** — it is used by `dequant_nvfp4_to_half_kernel` (line 67). Only its local copy of `e4m3_scale_to_float` is a duplicate and is removed. Do NOT delete `nvfp4_to_float` / the kernel / its `extern __global__` decls.
3. **`GlmGemmFp64Kernel` is defined twice** in the SAME object library (`glm_inference.cu:93`, `glm_inference_load.cu:223`) — byte-identical, both `lsxcommon::glm`. This currently links because nvcc only emits device code for kernels actually launched in their TU. Unify by declaring the canonical in `glm_inference.cu` and adding an `extern __global__` forward declaration in `glm_inference_load.cu` that calls it.
4. **`GlmSwigluKernel` (load.cu:86) and `GlmRealSwigluKernel` (inference.cu:937) are byte-identical** (no-clamp). Keep ONE canonical no-clamp kernel; `GlmSwigluClampKernel` stays separate (adds clamp).

---

## Task 1: Create the shared NVFP4 codec header

**Files:**
- Create: `libs/lsxcommon/src/nvfp4_codec.h`
- Modify: `libs/lsxcommon/CMakeLists.txt`

- [ ] **Step 1: Create `libs/lsxcommon/src/nvfp4_codec.h`**

```cpp
// nvfp4_codec.h — shared device-only NVFP4 decode primitives.
// Reachable by all three CUDA object libraries (lsxcommon_cu /
// lsxcommon_glm_cu / lsxcommon_qwen_cu). Device symbols only
// (__constant__ / __device__): no host linkage, so each NVCC TU gets its own
// const-bank copy — no ODR risk across the object-lib split.
//
// Bit-exactness: E4M3ScaleToFloat uses `e4==0 -> 0.0`, else (e4+120)<<23.
// All three prior sites (shared_experts, GLM, Qwen, benchmark) produced the
// same result for valid power-of-two block scales; see plan note #1.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace gpu {
namespace nvfp4 {

// E2M1 decode table (indexed by 4-bit nibble, sign bit 3).
__constant__ float kE2M1Table[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 power-of-2 block scale (only exponent bits used). e4==0 -> 0.0;
// else 2^(e4-7) reconstructed bit-wise as (e4+120)<<23.
__device__ __forceinline__ float E4M3ScaleToFloat(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    return __uint_as_float((e4 + 120u) << 23);
}

} // namespace nvfp4
} // namespace gpu
} // namespace lsxcommon
```

- [ ] **Step 2: Add `src` to CUDA include dirs**

Edit `libs/lsxcommon/CMakeLists.txt`. `LSXCOMMON_CU_INCLUDES` currently starts with:
```cmake
set(LSXCOMMON_CU_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
```
Add the src dir as the FIRST entry so `.cu` files can `#include "nvfp4_codec.h"`:
```cmake
set(LSXCOMMON_CU_INCLUDES
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
```
Note: this dir is already implicitly searched by NVCC for quoted `#include` from the same file location, but adding it makes the reachability explicit and robust for headers included from `include/`. It is PRIVATE to the three object libs and does not leak to consumers.

- [ ] **Step 3: Reconfigure + verify it compiles standalone**

Run: `cmake --preset windows-debug`
Expected: configure succeeds; no new errors.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/nvfp4_codec.h libs/lsxcommon/CMakeLists.txt
git commit -m "feat: shared NVFP4 device codec (E2M1 table + E4M3 scale)"
```

## Task 2: Migrate `shared_experts.cu` to the codec

**Files:**
- Modify: `libs/lsxcommon/src/shared_experts.cu`

`shared_experts.cu` is in `lsxcommon_cu` (default FMA, namespace `lsxcommon::gpu::` anonymous). Its E2M1/E4M3 use matches the codec exactly.

- [ ] **Step 1: Add include and alias**

At top of `shared_experts.cu`, after the existing includes:
```cpp
#include "nvfp4_codec.h"
```
Inside `namespace lsxcommon { namespace gpu { namespace {`, add an alias so existing call sites read cleanly:
```cpp
using nvfp4::kE2M1Table;
using nvfp4::E4M3ScaleToFloat;
```

- [ ] **Step 2: Delete the local constants/functions**

Remove the local copy of `kE2M1Table` (was lines 38-44) and `e4m3_scale_to_float` (was lines 46-52). Keep `f8e4m3_to_float`, `e8m0_scale_to_float`, `unpack_nibbles8`, and the kernels unchanged (they now resolve to the aliases).

- [ ] **Step 3: Build + run DeepSeek kernel-inference test (heavy, skip-note)**

Run: `cmake --build build/windows-debug --config Debug --target test_deepseek_inference_kernels` 2>&1 | select -Last 2
Expected: target builds. Then check that `test_deepseek_inference_kernels` is a CUDA unit test (4470 assertions) runnable without a model checkpoint — if so, run it; if it needs real weights, record the build-only verification and confirm via the loader test:
Run: `...\test_deepseek_loader.exe` → "All tests passed (64 assertions in 3 test cases)".

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/shared_experts.cu
git commit -m "refactor: shared_experts uses nvfp4_codec"
```

## Task 3: Migrate `qwen_inference.cu` to the codec

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference.cu`

`qwen_inference.cu` is in `lsxcommon_qwen_cu`, anonymous-ns `qwen_e2m1_tab`/`qwen_e4m3_scale`. `qwen_nvfp4_gemm_kernel` (line 352) and `qwen_out_gate_kernel` reference them.

- [ ] **Step 1: Include + local alias**

Add after the file's existing `#include`s:
```cpp
#include "nvfp4_codec.h"
```
Inside the anonymous namespace block that defines the old symbols, replace:
```cpp
__constant__ float qwen_e2m1_tab[16] = { ... };
__device__ __forceinline__ float qwen_e4m3_scale(uint8_t e4m3) { ... }
```
with:
```cpp
using nvfp4::kE2M1Table;
using nvfp4::E4M3ScaleToFloat;
```
(Keep the surrounding `namespace {` block; references to `qwen_e2m1_tab`/`qwen_e4m3_scale` may remain if you instead only add using-aliases named `qwen_e2m1_tab`/`qwen_e4m3_scale` — pick the lower-churn option: since the whole file is searchable, rename the 2-3 call sites to `kE2M1Table`/`E4M3ScaleToFloat` and use the alias above.)

- [ ] **Step 2: Build + run Qwen model test**

Run: `cmake --build build/windows-debug --config Debug --target test_qwen_model` 2>&1 | Select -Last 1
Then: `...\test_qwen_model.exe` → "All tests passed (13 assertions in 1 test case)".

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/qwen_inference.cu
git commit -m "refactor: qwen_inference uses nvfp4_codec"
```

## Task 4: Migrate `benchmark_gpu.cu` to the codec (remove duplicated scale)

**Files:**
- Modify: `libs/lsxcommon/src/benchmark_gpu.cu`

`benchmark_gpu.cu` is in `lsxcommon_cu`, namespace `lsxcommon::gpu` (non-anonymous). It keeps `nvfp4_to_float` (used by `dequant_nvfp4_to_half_kernel`) but its `e4m3_scale_to_float` is a duplicate to remove.

- [ ] **Step 1: Include + replace**

Add:
```cpp
#include "nvfp4_codec.h"
```
as `#include "lsxcommon/..."` resolves from the `src` include added in Task 1. Remove the local `e4m3_scale_to_float` (lines 44-49). Replace its use in `dequant_nvfp4_to_half_kernel` (line 66):
```cpp
    float bs = gpu::nvfp4::E4M3ScaleToFloat(scales[i >> 4]);
```
(Keep `nvfp4_to_float` and the kernel — do NOT delete; they remain referenced by the `extern __global__` declarations in pipeline/moe per their comment.)

- [ ] **Step 2: Build to confirm**

Run: `cmake --build build/windows-debug --config Debug --target lsxcommon_cu` 2>&1 | Select -Last 2
Expected: compiles clean.

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/src/benchmark_gpu.cu
git commit -m "refactor: benchmark_gpu uses shared E4M3ScaleToFloat"
```

## Task 5: Merge GLM twin kernels (GEMM, RMSNorm, SwiGLU)

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`

Both are in `lsxcommon_glm_cu` compiled with `-fmad=false`. Strategy: canonical definitions stay/land in `glm_inference.cu`; `glm_inference_load.cu` uses `extern __global__` forward declarations.

- [ ] **Step 1: `GlmGemmFp64Kernel` — single canonical definition**

`glm_inference.cu:93` already has the canonical. In `glm_inference_load.cu`, DELETE the local duplicate (lines 223-241) and add a forward declaration inside `namespace lsxcommon { namespace glm {` (before its use — locate the call site, e.g. the router path) :
```cpp
// Canonical definition lives in glm_inference.cu (bit-exact FP64-accum FP16
// GEMM, -fmad=false). Declared extern here; same object lib, no device-link
// duplicate once referenced from a single launch site per TU.
extern __global__ void GlmGemmFp64Kernel(const __half *__restrict__ w,
                                         const __half *__restrict__ x,
                                         __half *__restrict__ out,
                                         int m, int k, int n);
```
Keep the launch site as-is. Confirm `glm_inference.cu` still launches it (grep `<<<` for `GlmGemmFp64Kernel`).

- [ ] **Step 2: `GlmSwigluKernel` vs `GlmRealSwigluKernel` — one no-clamp kernel**

They are byte-identical (no-clamp). Keep the canonical in `glm_inference.cu`. Rename nothing; instead:
- In `glm_inference_load.cu`, DELETE `GlmSwigluKernel` (lines 86-95) and add extern:
```cpp
extern __global__ void GlmRealSwigluKernel(__half *__restrict__ gate,
                                           const __half *__restrict__ up, int n);
```
- Update the launch call in `glm_inference_load.cu` that used `GlmSwigluKernel` to `GlmRealSwigluKernel`.
- Keep `GlmSwigluClampKernel` (adds clamp) untouched.

- [ ] **Step 3: `GlmRmsNormKernel` vs `GlmMoERmsNormKernel` — one kernel**

`:GlmRmsNormKernel` (glm_inference.cu:35) and `GlmMoERmsNormKernel` (glm_inference_load.cu:114) perform identical sequential-FP32 scan + `GlmRsqrtfGlibc`, differing only in binding `xr`/`yr` locals (semantically the same). Keep the canonical `GlmRmsNormKernel` in glm_inference.cu and:
- Delete `GlmMoERmsNormKernel` (glm_inference_load.cu:114-140).
- Add extern:
```cpp
extern __global__ void GlmRmsNormKernel(const __half *__restrict__ x,
                                        const __half *__restrict__ w,
                                        __half *__restrict__ y,
                                        int dim, float eps);
```
- Update `GlmMoERmsNormKernel` launch sites in glm_inference_load.cu to `GlmRmsNormKernel`, passing dimensions so `dim` == per-row length. **Verify row count handling**: if the MoE path processes `rows>1` with one block per row using the same `dim`, the single-block-per-row `GlmRmsNormKernel` semantics are preserved (it already grid-strides by `blockIdx.x` per row). Confirm the launch uses `gridDim = rows`.

- [ ] **Step 4: Build GLM bit-exact tests + real selfcheck**

Run: `cmake --build build/windows-debug --config Debug --target test_glm_model --target test_glm_selfcheck` 2>&1 | Select -Last 2
Expected: builds. Run `test_glm_model.exe` → "All tests passed (16 assertions in 2 test cases)".
If a GLM checkpoint is present, run `test_glm_selfcheck.exe` (bit-exact `final_logits`, `"Paris"`); else confirm SKIP ("1 skipped"). This validates note #1 (E4M3 guard) on real data.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_inference.cu libs/lsxcommon/src/glm_inference_load.cu
git commit -m "refactor: unify GLM GEMM/RMSNorm/SwiGLU twin kernels via extern"
```

## Task 6: Full-suite verification

- [ ] **Step 1: Reconfigure + full build**

Run: `cmake --preset windows-debug && cmake --build --preset windows-debug`
Expected: clean build (only pre-existing Arrow warnings).

- [ ] **Step 2: Run affected real/synthetic tests**

```bash
ctest --preset windows-debug -R "glm_model|qwen_model|qwen_3_8_27b_model|deepseek_loader|model_runtime|model_engine|model_factory"
```
Expected: all PASS (real SelfChecks SKIP if the checkpoint is absent). Deterministic Qwen/DeepSeek SelfChecks (if checkpoints present) additionally confirm zero numeric drift:
```bash
...\test_qwen_selfcheck.exe    # 20 assertions, "Paris", 3-cycle bit-exact
...\test_deepseek_selfcheck.exe
```

- [ ] **Step 3: Confirm no codec duplicates remain**

```bash
rg -n "kE2M1Table|kGlmE2M1Table|qwen_e2m1_tab|e4m3_scale_to_float|GlmE4M3P2ToFloat|qwen_e4m3_scale" libs/lsxcommon/src
```
Expected: the ONLY matches are the alias `using` lines and the canonical `nvfp4_codec.h` symbols. No standalone duplicate table/function definitions remain.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: verify inference kernel unification (nvfp4 codec + GLM twins)"
```
(Only if Step 3/2 surfaced uncommitted changes.)