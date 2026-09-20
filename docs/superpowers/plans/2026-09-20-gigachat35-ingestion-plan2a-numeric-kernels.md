# GigaChat3.5 Ingestion — Plan 2a: Numeric Kernels (norms, gating, router, MLP)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **No subagents** (AGENTS.md). Execute inline.

**Goal:** Implement and device-parity-test the elementwise gating kernels of GigaChat3.5 — `ZeroCenteredGatedNorm`, the GatedDeltaNet output norm, the sigmoid router (top-8 + `e_score_correction_bias` + `norm_topk_prob` + `routed_scaling_factor`), the `swiglu_limit` MLP, and the gated-attention sigmoid gate — against a pinned host reference.

**Architecture:** A `__host__ __device__` reference header (`gigachat_moe_device_math.h`) is the single source of truth; host tests pin the formulas and CUDA kernels reproduce them bit-for-bit (`-fmad=false`). All matrix math uses the neutral NVFP4 layer (`LsxNvfp4Weight` + `RunLsxNvfp4Gemm*`); norms and biases stay BF16/FP32. No NVFP4 codec change in this plan.

**Tech Stack:** CUDA 13.4 (sm_120a/sm_121a), C++20, Catch2 v3.16.0, CCCL 3.4.2, abseil.

## Global Constraints

- CC 12.x: **48 warps/SM**, 100 KB smem/SM, 1536 threads/SM; no tcgen05 tensor-memory MMA — warp-level `mma.sync...mxf4nvf4.block_scale` is the only NVFP4 MMA route (docs `NVIDIA_DOCS_NVFP4_MEMORY.md` §B1/B4).
- One stream, no per-token `cudaDeviceSynchronize`; shared-memory RMW is thread-0-only behind `__syncthreads()`; reductions feeding logits are deterministic (shared partials + fixed sequential sum, **no `atomicAdd`**).
- NVFP4 only through `lsx_inference_nvfp4.h`; norms/biases never quantized.
- Numeric class: these kernels are **C-exact** (fixed accumulation order, device-independent launch shapes).
- `#include`-reachable from `.cu`: no `<absl/container/*>`.
- New tests: ctest label in the same commit; converter/selfcheck skip rules unchanged.
- Commit prefixes `feat:`/`test:`; English messages.

---

## File Structure

- `libs/lsxcommon/src/gigachat_moe_device_math.h` — `__host__ __device__` reference math (norms, gating, swiglu, router selection).
- `libs/lsxcommon/src/gigachat_moe_norm_kernels.cu` — `ZeroCenteredGatedNorm` + GDN output norm device kernels.
- `libs/lsxcommon/src/gigachat_moe_kernels.cu` — router, expert/dense `swiglu_limit` MLP, gated-attention sigmoid.
- `tests/test_gigachat_moe_math.cpp` — host reference assertions (extend).
- `tests/test_gigachat_moe_devkernels.cu` — device-vs-host parity for the new kernels (separate target `test_gigachat_moe_devkernels`; the existing `test_gigachat_moe_kernels.cpp` keeps the dequant parity test).

---

### Task 1: Device reference math header

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_device_math.h`
- Modify: `tests/test_gigachat_moe_math.cpp`

**Interfaces:**
- Produces (all `inline __host__ __device__`, namespace `lsxcommon::gigachat::dev`):
```cpp
float ZeroCenteredGatedNormElement(float x, float w, float gate);           // x_norm*(1+w)*sigmoid(gate)
float GdnOutputNormElement(float o, float w, float z, float eps, float scale);
float SwigluLimitElement(float g, float u, float limit);                    // silu(clamp(g,max=L))*clamp(u,-L,L)
float SigmoidGateElement(float y, float gate);                              // y*sigmoid(gate)
```

- [ ] **Step 1: Write the failing host test**

Append to `tests/test_gigachat_moe_math.cpp`:
```cpp
#include "gigachat_moe_device_math.h"

TEST_CASE("gigachat gating reference math", "[gigachat][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    // ZeroCenteredGatedNorm: x_norm * (1+w) * sigmoid(gate), gating weight 2.
    // x=3, w=0 -> x_norm=1; gate=0 -> sigmoid=0.5; y=1*1*2*0.5=1.
    CHECK(dev::ZeroCenteredGatedNormElement(3.0f, 0.0f, 0.0f) ==
          Catch::Approx(1.0f).margin(1e-6f));
    // GDN output norm: (1+w)=1, sigmoid(0)=0.5, scale=2 -> 1.
    CHECK(dev::GdnOutputNormElement(5.0f, 0.0f, 0.0f, 1e-6f, 2.0f) ==
          Catch::Approx(1.0f).margin(1e-6f));
    // SwiGLU limit: gate 20 -> clamp 10, silu(10)~9.9995; up 20 -> clamp 10.
    CHECK(dev::SwigluLimitElement(20.0f, 20.0f, 10.0f) ==
          Catch::Approx(99.995f).margin(1e-2f));
    CHECK(dev::SigmoidGateElement(8.0f, 0.0f) ==
          Catch::Approx(4.0f).margin(1e-6f));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_math && ./build/linux-debug/tests/test_gigachat_moe_math`
Expected: FAIL — `gigachat_moe_device_math.h` missing.

- [ ] **Step 3: Create the reference header**

Create `libs/lsxcommon/src/gigachat_moe_device_math.h`:
```cpp
#pragma once
// Pinned host/device reference for GigaChat3.5 gating math, transcribed from
// modeling_gigachat3_5.py (ZeroCenteredGatedNorm / GDN output norm / swiglu
// limit / gated attention). Single source of truth for the device kernels.
#include <cstdint>

#ifndef __CUDACC__
#include <cmath>
#define LSX_HD
#define LSX_DEV_INLINE
#define LSX_DEV_FUN inline
#else
#define LSX_HD __host__ __device__
#define LSX_DEV_INLINE __device__ __forceinline__
#define LSX_DEV_FUN __device__
#endif

namespace lsxcommon::gigachat::dev {

LSX_HD inline float Sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }
LSX_HD inline float Silu(float v) { return v * Sigmoid(v); }

// ZeroCenteredGatedNorm: `x_norm * (1+w) * gating_weight * sigmoid(gate)` where
// `gating_weight` = 2.0 (config layernorm_gating_weight). Callers pass the
// already-computed gate = W_down . silu(W_up . x_norm).
LSX_HD inline float ZeroCenteredGatedNormElement(float x_norm, float w,
                                                 float gate) {
    return x_norm * (1.0f + w) * 2.0f * Sigmoid(gate);
}

// GatedDeltaNet output norm (gated_rmsnorm_sigmoid_zero_centered):
// x_norm * (1+w) * sigmoid(z) * linear_sigmoid_gate_scale (2.0).
LSX_HD inline float GdnOutputNormElement(float x_norm, float w, float z,
                                         float /*eps*/, float scale) {
    return x_norm * (1.0f + w) * scale * Sigmoid(z);
}

// swiglu_limit: gate = clamp(gate, max=L), up = clamp(up, -L, L);
// returns silu(gate) * up. L <= 0 disables.
LSX_HD inline float SwigluLimitElement(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fmaxf(fminf(u, limit), -limit);
    }
    return Silu(g) * u;
}

// Gated attention: y * sigmoid(gate_proj(x)).
LSX_HD inline float SigmoidGateElement(float y, float gate) {
    return y * Sigmoid(gate);
}

} // namespace lsxcommon::gigachat::dev
```

- [ ] **Step 4: Run to green**

Run the math target; Expected: PASS. The existing dequant target is untouched.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_device_math.h tests/test_gigachat_moe_math.cpp
git commit -m "feat: gigachat device reference math"
```

---

### Task 2: ZeroCenteredGatedNorm device kernel

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_norm_kernels.cu` (declare in `gigachat_moe_device_math.h`? no — in a new `gigachat_moe_kernels.h`)
- Create: `libs/lsxcommon/src/gigachat_moe_kernels.h`
- Create: `tests/test_gigachat_moe_devkernels.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `.cu` to `lsxcommon_gigachat_cu`), `tests/CMakeLists.txt`

**Interfaces:**
- Produces:
```cpp
namespace lsxcommon::gigachat {
// Applies ZeroCenteredGatedNorm over `rows` rows of `dim`:
//   out[r][i] = zero_centered_rmsnorm(x[r])[i] * gate(r)[i]
// where gate = (Wdown @ silu(Wup @ x_norm)) / gating_weight=2.0,
// Wup: [rank=16, dim], Wdown: [dim, rank], weights BF16, compute F32.
cudaError_t RunZeroCenteredGatedNorm(const float *x, int rows, int dim,
                                     const uint8_t *w, const uint8_t *w_up,
                                     const uint8_t *w_down, float eps,
                                     float *out, cudaStream_t stream);
}
```
(Weights are raw BF16 bytes; the kernel decodes `__nv_bfloat16`.)

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_gigachat_moe_devkernels.cu`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "gigachat_moe_kernels.h"
#include "gigachat_moe_device_math.h"
#include <cuda_runtime.h>
#include <vector>

TEST_CASE("gigachat zero-centered gated norm parity", "[gigachat][inference]")
{
    const int rows = 3, dim = 128, rank = 16;
    std::vector<float> x(rows * dim), w(dim), wup(rank * dim), wdown(dim * rank);
    for (int i = 0; i < rows * dim; ++i) x[i] = 0.01f * (i % 97) - 0.5f;
    for (int i = 0; i < dim; ++i) w[i] = 0.001f * (i % 13);
    for (int i = 0; i < rank * dim; ++i) wup[i] = 0.002f * (i % 11) - 0.01f;
    for (int i = 0; i < dim * rank; ++i) wdown[i] = 0.003f * (i % 7) - 0.01f;
    std::vector<float> out(rows * dim), ref(rows * dim);
    cudaMalloc; copy x,w,wup,wdown; launch; copy back; sync; free.
    // host reference: compute x_norm per row, gate = Wdown@silu(Wup@x_norm)
    for (int r = 0; r < rows; ++r) {
        float ss = 0; for (int i = 0; i < dim; ++i) ss += x[r*dim+i]*x[r*dim+i];
        float inv = 1.0f / sqrtf(ss/dim + 1e-6f);
        std::vector<float> xn(dim);
        for (int i = 0; i < dim; ++i) xn[i] = x[r*dim+i]*inv;
        std::vector<float> h(rank, 0.0f);
        for (int k = 0; k < rank; ++k) { float a=0; for (int i=0;i<dim;++i) a += wup[k*dim+i]*xn[i]; h[k]=lsxcommon::gigachat::dev::Silu(a); }
        for (int i = 0; i < dim; ++i) {
            float g=0; for (int k=0;k<rank;++k) g += wdown[i*rank+k]*h[k];
            ref[r*dim+i] = lsxcommon::gigachat::dev::ZeroCenteredGatedNormElement(xn[i], w[i], g);
        }
    }
    for (int i = 0; i < rows*dim; ++i) CHECK(out[i] == Catch::Approx(ref[i]).margin(2e-4f));
}
```
(The test body must use a small RAII `DeviceBuffer` helper; see `tests/test_blackwell_tc_mma.cu`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build --preset linux-debug --target test_gigachat_moe_devkernels`
Expected: FAIL — `gigachat_moe_kernels.h` / target missing.

- [ ] **Step 3: Implement the kernel**

`gigachat_moe_norm_kernels.cu`: one block per row (dim ≤ 1024), thread 0 computes the row sum-of-squares into shared memory behind `__syncthreads()` (deterministic sequential order), then every thread computes `xn[i]`, the rank-16 gate projection (sequential F32), and writes `out`. Decode weights with `__nv_bfloat16` → float. Declare the entry point in `gigachat_moe_kernels.h`. Add the `.cu` to `lsxcommon_gigachat_cu`; register `test_gigachat_moe_devkernels` as an OBJECT-lib test (mirror `test_nemotron_moe_kernels_cu`) with label `inference`, and add it to the suite-guard kernels stage for `gigachat3.5-432b-a28b-reasoning`.

- [ ] **Step 4: Run to green**

Run the parity test. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/gigachat_moe_norm_kernels.cu libs/lsxcommon/src/gigachat_moe_kernels.h \
  tests/test_gigachat_moe_devkernels.cu libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: gigachat zero-centered gated norm kernel"
```

---

### Task 3: GDN output norm (zero-centered sigmoid gated RMSNorm)

**Files:**
- Modify: `libs/lsxcommon/src/gigachat_moe_norm_kernels.cu`, `gigachat_moe_kernels.h`
- Modify: `tests/test_gigachat_moe_devkernels.cu`

**Interfaces:**
- Produces:
```cpp
cudaError_t RunGdnOutputNorm(const float *o, const float *z, const uint8_t *w,
                             int heads, int head_dim, float eps, float scale,
                             float *out, cudaStream_t stream);
// out[h][d] = GdnOutputNormElement(rmsnorm(o[h])[d], w[d], z[h][d], eps, scale)
```

- [ ] **Step 1: Write the failing parity test**

Add a case computing `rmsnorm` per head with `w` BF16, `z` gate, `scale=2.0`, and comparing device vs host (`GdnOutputNormElement`).

- [ ] **Step 2: Run to verify it fails** — link error `RunGdnOutputNorm`.
- [ ] **Step 3: Implement** — per-head (block) deterministic RMSNorm + gate, mirroring Task 2's sandbox.
- [ ] **Step 4: Run to green.**
- [ ] **Step 5: Commit** `feat: gigachat gdn output norm kernel`.

---

### Task 4: Sigmoid router + `swiglu_limit` MLP + gated-attention sigmoid

**Files:**
- Create: `libs/lsxcommon/src/gigachat_moe_kernels.cu`
- Modify: `gigachat_moe_kernels.h`, `tests/test_gigachat_moe_devkernels.cu`, `libs/lsxcommon/CMakeLists.txt`

**Interfaces:**
- Produces:
```cpp
// Router: logits = x . W^T (F32); scores = sigmoid(logits);
// choice = scores + e_score_correction_bias; idx = top-8 (lowest index wins);
// w = scores[idx]; w /= sum + 1e-20; w *= routed_scaling_factor (2.5).
// W is BF16 [E, hidden]; bias F32 [E]; deterministic block-reduce top-k.
cudaError_t RunGigaChatRouterTopK(const float *x, const uint8_t *w,
                                  const float *bias, int num_experts, int top_k,
                                  float routed_scale, int *out_idx,
                                  float *out_weights, cudaStream_t stream);

// SwiGLU MLP with limit: mid = silu(clamp(g,max=L))*clamp(u,-L,L) (F32),
// in/out are the neutral NVFP4 weights; this entry only computes the
// elementwise activation+product for `n` elements.
cudaError_t RunSwigluLimit(const float *gate, const float *up, float limit,
                           int n, float *mid, cudaStream_t stream);

// Gated attention: y[i] *= sigmoid(gate[i]).
cudaError_t RunSigmoidGate(float *y, const float *gate, int n,
                           cudaStream_t stream);
```

- [ ] **Step 1: Write the failing tests**
  - Router: `E=8`, `top_k=2`; known logits where the correction bias flips the top-2; assert indices, normalized `scores` weights ×2.5 (host reference mirroring `GigaChat35TopkRouter.forward`).
  - SwiGLU: random gate/up, `L=10`, host `SwigluLimitElement` parity.
  - Sigmoid gate: random y/gate, host parity.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Router uses one block, thread-0-only sequential selection over `E` in shared memory (deterministic; `E=256` is cheap for one token), or a deterministic block-reduce with a fixed lowest-index tie rule; no atomics.
- [ ] **Step 4: Run to green.**
- [ ] **Step 5: Commit** `feat: gigachat router, swiglu-limit and sigmoid-gate kernels`.

---

## Plan 2a verification

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug-fast
ctest --preset linux-debug-inference -R gigachat
./build/linux-debug/tests/test_model_suite_guard
```

## Out of scope (Plan 2b)

MLA decode (q/kv latents, `alpha_q`/`alpha_kv`, interleaved yarn RoPE, `kAttnScale`), GatedDeltaNet decode (fused qkvz/ba split, depthwise conv, L2-norm QK, delta rule, recurrent+conv state), KV/recurrent state allocation, `gigachat_moe_expert_cache.{h,cpp}` host mmap store, the 40-layer decode pipeline with `pre_post` residuals, device weight loader, `Infer` wiring, and the real forward-parity test.
