# CUDA Unification + Phase-4 Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated CUDA kernels across all model object libraries (shared header-template library), then push the Qwen3.5-9B fast-math path toward roofline (280–320 tok/s @ RTX 5090; report GB/s extrapolation for RTX 5060 Ti 70–80 tok/s).

**Architecture:** Phase A introduces header-only kernel templates (`lsx_kernels_*.cuh`) included by existing object libraries — single textual source, per-flavor instantiation preserves bit-exact contracts. Phase B reworks the qwen fast-path dataflow: fused gate+up weights (one GEMM, one act-quant), swiglu folded into the decode-GEMM epilogue, TMA bulk-tensor staging replacing the cp.async ring.

**Tech Stack:** CUDA 13.3, inline PTX (`mma.kind::mxf4nvf4`, `cp.async.bulk.tensor`, mbarrier), Catch2 v3, CMake presets.

## Global Constraints

- Blackwell only: `120-real;121-real`; arch-`a` targets (`sm_120a/sm_121a`) ONLY where `kind::mxf4nvf4`/TMA PTX requires it.
- Bit-exact models (GLM/Kimi/DeepSeek): their numerics must remain byte-for-byte identical; default scalar path untouched.
- All logging via spdlog; no `std::cout`.
- GPU tests run SEQUENTIALLY (`ctest --test-dir build/linux-debug`, never `-j`). Single-run perf deltas are +-15% noise: always >=3 runs.
- Build/test commands: `cmake --build --preset linux-debug [--target X]`, `ctest --test-dir build/linux-debug -R <regex>`.
- Commits: English, prefixes `feat:`/`refactor:`/`fix:`/`test:`/`perf:`/`docs:`.
- NVCC-safe headers only in anything transitively included by `.cu` (no absl containers).
- Shared math kernels use explicit intrinsics (`__fmaf_rn`, `__fmul_rn`, `__fadd_rn`) so results do not depend on `-fmad`.

---

### Task 1: Kernel-test harness

**Files:**
- Create: `tests/test_lsx_kernels.cu`
- Modify: `tests/CMakeLists.txt`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add fmad-policy comment near object libs)

**Interfaces:**
- Produces: ctest target `test_lsx_kernels`; helper `CopyDeviceToHost<T>(ptr, n)` and `RequireDeviceEqual<T>(a_ref, b_ref, n)` used by Tasks 2–6.

- [ ] **Step 1: CMake wiring**

In `tests/CMakeLists.txt`, next to `test_blackwell_tc_cu`, add:

```cmake
add_library(test_lsx_kernels_cu OBJECT test_lsx_kernels.cu)
set_target_properties(test_lsx_kernels_cu PROPERTIES
    CUDA_ARCHITECTURES "120-real;121-real"
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(test_lsx_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(test_lsx_kernels_cu PRIVATE
    ${CMAKE_SOURCE_DIR}/libs/lsxcommon/src)

add_executable(test_lsx_kernels test_lsx_kernels_main.cpp)
target_link_libraries(test_lsx_kernels PRIVATE
    test_lsx_kernels_cu Catch2::Catch2WithMain lsxcommon)
add_test(NAME test_lsx_kernels COMMAND test_lsx_kernels)
```

(Create empty `tests/test_lsx_kernels_main.cpp` with `int main(){return 0;}` if the repo's other suites use standalone mains; otherwise follow the existing Catch2 default-main pattern used by `test_blackwell_tc_mma.cu`.)

- [ ] **Step 2: Harness skeleton**

```cpp
// tests/test_lsx_kernels.cu
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace {
template <typename T>
std::vector<T> ToHost(const T* dev, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <typename T>
void RequireDeviceEqual(const T* a, const T* b, size_t n) {
    auto ha = ToHost(a, n), hb = ToHost(b, n);
    REQUIRE(memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0);
}
} // namespace

TEST_CASE("harness smoke", "[lsxk]") {
    int* d = nullptr; cudaMalloc(&d, 4); cudaMemset(d, 0, 4); cudaFree(d);
    REQUIRE(true);
}
```

- [ ] **Step 3: Build & run green**

Run: `cmake --build --preset linux-debug --target test_lsx_kernels && ctest --test-dir build/linux-debug -R test_lsx_kernels`
Expected: PASS (1 test case).

- [ ] **Step 4: fmad policy comment**

In `libs/lsxcommon/CMakeLists.txt` above `add_library(lsxcommon_qwen_cu ...)`:

```cmake
# FMA policy: object-lib -fmad flags are historical (qwen_moe/glm/kimi off).
# Shared kernels in lsx_kernels_*.cuh use explicit *_rn intrinsics and are
# therefore flag-independent; DO NOT rely on contraction in any new kernel.
```

- [ ] **Step 5: Commit** `test: lsx_kernels equivalence harness`

---

### Task 2: Quant cluster — dequant/F8/E2M1 encode

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_quant.cuh`
- Modify: `shared_experts.cu`, `gemma_inference.cu:28`, `kimi_inference.cu:636`, `glm_inference_load.cu:50`, `benchmark_gpu.cu:34,45`, `safetensors_gpu.cu:63,82,96`, `deepseek_inference_attn.cu:30,49`
- Test: `tests/test_lsx_kernels.cu`

**Interfaces:**
- Produces (namespace `lsxcommon::kernels`):

```cpp
template <typename OutT>                       // __half / __nv_bfloat16 / float
__device__ OutT Nvfp4Decode(uint8_t nib, uint8_t scale_e4m3, float global_scale);

template <typename OutT>
__global__ void Nvfp4DequantKernel(const uint8_t* packed, const uint8_t* scales,
                                   float global_scale, OutT* out, int64_t n);

__device__ __forceinline__ float F8E4m3ToFloat(uint8_t v);   // canonical
__device__ __forceinline__ float E8m0ToFloat(uint8_t v);
__device__ __forceinline__ uint8_t QuantizeE2m1(float x);     // threshold chain
```

- [ ] **Step 1: failing equivalence tests** — port ONE old kernel verbatim into the test file under `namespace ref` (e.g. ragged-tail body of `dequant_shared_nvfp4_kernel`) and assert `Nvfp4DequantKernel<__half>(...)` output memcmp-equal on random 4096 elements (seeded LCG), plus F8/E8M0 tables vs `ref_f8e4m3_to_float` over all 256 byte values.
- [ ] **Step 2:** run `-R test_lsx_kernels` → FAIL (templates undefined).
- [ ] **Step 3: implement** `lsx_kernels_quant.cuh`:

```cpp
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "nvfp4_codec.h"

namespace lsxcommon { namespace kernels {

__device__ __forceinline__ float F8E4m3ToFloat(uint8_t v) {
    // sign(1) exp(4) mant(3): matches safetensors_gpu.cu dev_f8e4m3_to_float
    const uint32_t s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0) return m ? __uint_as_float((126u << 23) | (m << 21)) * (s ? -1.f : 1.f)
                         : 0.f;
    if (e == 0xF && m == 7) return NAN;
    return __uint_as_float(((e + 120u) << 23) | (m << 20)) * (s ? -1.f : 1.f);
}
__device__ __forceinline__ float E8m0ToFloat(uint8_t v) {
    return __uint_as_float(uint32_t(v) << 23);
}

template <typename OutT>
__device__ __forceinline__ OutT CastOut(float v);
template <> __device__ __forceinline__ __half CastOut<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 CastOut<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ float CastOut<float>(float v) { return v; }

template <typename OutT>
__device__ __forceinline__ OutT Nvfp4Decode(uint8_t nib, uint8_t s, float gs) {
    return CastOut<OutT>(gpu::nvfp4::E2M1Value(nib) *
                         gpu::nvfp4::E4M3ScaleToFloat(s) * gs);
}

template <typename OutT>
__global__ void Nvfp4DequantKernel(const uint8_t* __restrict__ packed,
                                   const uint8_t* __restrict__ scales,
                                   float global_scale, OutT* __restrict__ out,
                                   int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t nib = (i & 1) ? (packed[i >> 1] >> 4) : (packed[i >> 1] & 0x0F);
    out[i] = Nvfp4Decode<OutT>(nib, scales[i >> 4], global_scale);
}

__device__ __forceinline__ uint8_t QuantizeE2m1(float x) {
    const float a = fabsf(x);
    uint8_t mag = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3
               : a < 2.5f  ? 4 : a < 3.5f  ? 5 : a < 5.0f  ? 6 : 7;
    return mag | ((x < 0.f) ? 8 : 0);
}

}} // namespace lsxcommon::kernels
```

NOTE: verify each moved decoder against its source byte-for-byte BEFORE deleting the source (the F8 subnormal/exp tables above must match `dev_f8e4m3_to_float` exactly — copy from `safetensors_gpu.cu:63`, do not trust the sketch).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: migrate launch sites** one file per build cycle: replace local kernel bodies with includes + launches of `Nvfp4DequantKernel<OutT>` (grid `(n+255)/256`); delete `dev_f8e4m3_to_float`/`dev_f8e8m0_to_float` duplicates and alias shims (`KimiE4M3`); remove `dequant_nvfp4_to_half_kernel` from `benchmark_gpu.cu` together with dead externs in `deepseek_inference_pipeline.cu:25`/`moe.cu:18`. Vectorized fast paths (`unpack_nibbles8`) may be kept as an internal specialization `Nvfp4DequantVecKernel` in the same header — elementwise result identical.
- [ ] **Step 6:** full build + `ctest -R "deepseek|glm|kimi|gemma|qwen|blackwell"` green; commit `refactor: unified nvfp4/f8 quant kernels; remove 5 dequant copies`.

---

### Task 3: Math cluster — add/axpy/embed/argmax/convert/clamp/reductions

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_math.cuh`
- Modify: `qwen_inference.cu:585,206,920`, `qwen_moe_inference.cu:289,135`, `kimi_inference.cu:323,815,853`, `glm_inference.cu:63,163,195`, `glm_inference_load.cu:1108`, `gemma_inference.cu:597,506`, `deepseek_inference_pipeline.cu:35,45`, `deepseek_inference_attn.cu:571`, `sampling.cu` (argmax stays canonical, others point at it)
- Test: `tests/test_lsx_kernels.cu`

**Interfaces:**
- Produces:

```cpp
template <typename T> __global__ void AddKernel(const T* src, T* dst, int64_t n);
template <typename T> __global__ void AxpyKernel(const T* x, float a, T* y, int64_t n);
template <typename T> __global__ void EmbedGatherKernel(const int* ids, const T* table,
                                                        T* out, int hidden);
template <typename T> __global__ void F32toTKernel(const float* src, T* dst, int64_t n,
                                                   float clamp_abs = INFINITY);
template <typename T> __device__ T LsxBlockReduceSum(T v, T* smem);   // CUB-free
template <typename T> __device__ T LsxBlockReduceMax(T v, T* smem);

// Shared numeric-flavor enum; lives HERE (earliest header) so Tasks 4–6
// can use it without intra-phase ordering hazards.
enum class ExpFlavor { Fast, Precise, Glibc };   // __expf / expf / glibc-replica
template <ExpFlavor F> __device__ __forceinline__ float ExpF(float x);
```

- [ ] **Step 1: failing tests** — ref-copy `qwen_add_kernel` + glm embed gather; assert equality (half bitwise, embed row indices 0..3).
- [ ] **Step 2:** FAIL. **Step 3:** implement header (bodies are the trivially-audited common forms; converts use `__float2half_rn` only where the replaced kernel did — check per site). **Step 4:** PASS.
- [ ] **Step 5: migrate** all listed sites; delete superseded locals. Argmax: kimi/glm switch launches to `sampling.cu`'s two-phase kernels instead of a new copy.
- [ ] **Step 6:** suite green; commit `refactor: unified elementwise/reduction kernels; drop 12 local copies`.

---

### Task 4: Act cluster — SwiGLU flavors (+fmad independence)

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_act.cuh`
- Modify: `qwen_inference.cu:576`, `qwen_moe_inference.cu:115`, `kimi_inference.cu:652`, `deepseek_inference_moe.cu:143,159,189`, `glm_inference.cu:48`, `glm_expert_kernels.cuh:33` (include-through)
- Test: `tests/test_lsx_kernels.cu`

**Interfaces:**
- Produces:

```cpp
// ExpFlavor/ExpF come from lsx_kernels_math.cuh (Task 3).
template <typename T, ExpFlavor F>
__global__ void SwigluClampKernel(const T* g, const T* u, T* y, int64_t n,
                                  float g_limit, float u_limit);
```

- [ ] **Step 1: failing tests** — instantiate all three flavors on fixed inputs; `memcmp` against ref bodies of `qwen_swiglu_kernel` (Fast) and `qwen_moe_swiglu_kernel` (Precise); GLM clamp variant vs `GlmSwigluClampKernel` ref (Glibc + limits). Explicit intrinsics: silu computed as `g * (1/(1+ExpF(-g)))` with `__fmul_rn/__fdiv_rn` — verify output does not change between TU compiled with and without `-fmad=false` (two instantiations in the same test binary, one TU per flag).
- [ ] **Step 2:** FAIL. **Step 3:** implement; `Glibc` delegates to `glm_device_math.h` replica. **Step 4:** PASS (both flag TUs identical).
- [ ] **Step 5: migrate** sites preserving each caller's current flavor/dtype/limits (dense qwen: Fast; qwen_moe: Precise; kimi: BF16+Precise; deepseek: F32/F16 clamp; glm: Glibc). Delete locals.
- [ ] **Step 6:** `ctest -R "glm|deepseek|qwen|kimi"` green (bit-exact suites unchanged); commit `refactor: flavor-templated swiglu; fmad-independent shared activation`.

---

### Task 5: Norm cluster — RMSNorm family

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_norm.cuh`
- Modify: `qwen_inference.cu:58,89,337`, `gemma_inference.cu:70,100`, `kimi_inference.cu:449,826`, `glm_expert_kernels.cuh:46`, `deepseek_norm_kernels.cu:26` + `deepseek_inference_load.cuh:197` (already-shared pair may simply include-through)
- Test: `tests/test_lsx_kernels.cu`

**Interfaces:**
- Produces:

```cpp
template <typename T, ExpFlavor F, bool HasOffset>
__global__ void RmsNormKernel(const T* x, const T* w, T* y,
                              int rows, int hidden, float eps, int offset = 0);
```

- [ ] Steps 1–6 mirror Task 4: ref-copy `rmsnorm_offset_kernel` + gemma BF16 variant; equality tests; implement; migrate; delete; suite green; commit `refactor: unified rmsnorm templates`.

---

### Task 6: Router cluster — softmax→top-k→renorm

**Files:**
- Create: `libs/lsxcommon/src/lsx_kernels_router.cuh`
- Modify: `qwen_moe_inference.cu:34`, `gemma_inference.cu:395`, `kimi_inference.cu:104` (GLM bit-exact variants stay GLM-local — documented deviation tolerance: unify only if byte-equal achievable cheaply, else leave)
- Test: `tests/test_lsx_kernels.cu`

**Interfaces:**
- Produces:

```cpp
// Block-per-token. scores[] precomputed logits (expert_count <= 1024).
// thread-0 selection: max-scan -> exp -> insertion-sort top-k -> renorm.
template <bool SigmoidBias, typename ScaleT>
__global__ void RouterTopkKernel(const float* logits, const float* bias,
                                 ScaleT per_expert_scale, int expert_count,
                                 int top_k, float* weights, int* indices);
```

- [ ] Step 1: failing test — synthetic logits E=256,k=8: qwen_moe ref kernel vs template (SigmoidBias=false, scale=1) memcmp weights/indices; gemma case with per_expert_scale=2.0. Step 2: FAIL. Step 3: implement (port thread-0 selection verbatim from `qwen_moe_router_topk_kernel`). Step 4: PASS.
- [ ] Step 5: migrate qwen_moe + gemma + kimi launches; delete superseded kernels. Step 6: suite green; commit `refactor: unified moe router topk template`.

---

### Task 7: Phase A wrap — full-suite verification

- [ ] Sequential full run: `ctest --test-dir build/linux-debug` → 103/103 PASS, zero expectation changes.
- [ ] `git grep -c "__global__" libs/lsxcommon/src/*.cu` sanity: removed-kernel names absent (`rg "qwen_swiglu_kernel|dequant_shared_nvfp4_kernel|dev_f8e4m3_to_float"` → only header hits).
- [ ] Commit (if residue) `refactor: phase A sweep`; update findings doc with dedup summary (`docs:` commit).

---

### Task 8: B1 — gate+up concatenation at load (dense)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference_load.h` (fused weight struct fields), `src/qwen_inference_load.cpp` (upload), `src/qwen_inference.cu` (`RunQwenMLP`:1512 / `RunQwenFusedMLP`:1543 fast branch)
- Test: `tests/test_qwen_loader.cpp` (roundtrip) + micro e2e

**Interfaces:**
- Produces: `QwenNvfp4Weight` gains `w_gate_up_packed`/`w_gate_up_scales` (layout `[2I,K]`, rows `[0,I)=gate`, `[I,2I)=up`); scalar fallback keeps old fields until Task 10 removes dead branches.

- [ ] **Step 1: failing loader test** — fixture uploads tiny gate[K=64,I=32]/up; assert `w_gate_up_packed` bytes equal gate-half followed by up-half (memcmp of both slices vs individually uploaded refs).
- [ ] **Step 2:** FAIL. **Step 3:** in loader after per-weight uploads, allocate fused buffers and `cudaMemcpyAsync` device-to-device halves into `[2I*K/2]` packed / `[2I*K/16]` scales; wire struct fields. **Step 4:** PASS.
- [ ] **Step 5: fast-path consumer** — in the fast_math branch of dense MLP: single `RunQwenTcNvfp4Gemm(x, fused..., y_cat, seq, K, 2I)`, then `SwigluClampKernel<__half, ExpFlavor::Fast>(y_cat, y_cat + I, h, I, INF, INF)`, then down GEMM. Scalar branch untouched.
- [ ] **Step 6:** unit + `test_glm_selfcheck`-style qwen selfcheck green; micro-benchmark mlp_gate_up us/call (expect ~45–55% of prior pair); commit `feat: fused gate+up weights; single-gemm dense mlp fast path`.

---

### Task 9: B1-MoE — expert-store concat

**Files:**
- Modify: `src/qwen_moe_inference_load.cpp` (`LoadQwenMoeExpertStore`, slot alloc), `src/qwen_moe_inference.cu` (`RunQwenMoeRoutedFFN`:185 fast branch), `include/lsxcommon/qwen_moe_expert_cache.h` (slot byte-size constants)
- Test: extend `tests/test_qwen_moe_loader.cpp`

Same red/green recipe as Task 8 at expert granularity: store layout becomes `[E][2I,K]` (w1||w3), routed FFN issues one TC GEMM + one swiglu per selected expert. Cache slot sizes double-check against `AllocateQwenMoeExpertCache`. Commit `feat: concatenated expert gate_up in moe expert store`.

---

### Task 10: B2 — swiglu-in-epilogue (decode seq==1)

**Files:**
- Modify: `src/qwen_tc_gemm.cu` (`TcGemmDecodeKernel`:136 epilogue + host wrapper `RunQwenTcNvfp4Gemm`:500 — new mode flag `EpilogueMode::SwigluSplit{I}`)
- Test: `tests/test_blackwell_tc_mma.cu`

**Interfaces:**
- Produces: `tc::SetEpilogueSwiglu(int up_offset_columns)` — when >0, decode kernel writes `silu(D[0..I))*D[I..2I)` (FP32 math, `CastOut<__half>`) instead of raw D.

- [ ] **Step 1: failing tile-level test** — N=128 (I=64) single-tile case: run decode kernel in raw mode capturing D, compute expected swiglu on host from captured D, then run fused mode → memcmp halves. This pins accumulator-index mapping (col-group `wn=warp>>1` geometry) BEFORE integration.
- [ ] **Step 2:** FAIL (mode absent). **Step 3:** epilogue: each warp writes its 4 accumulators to dynamic smem staging (`[kDecCols][16]`), `__syncthreads()`, then threads `tid < I` compute `y[tid] = silu(smem[tid]) * smem[I + tid])`. **Step 4:** PASS incl. sanitizer memcheck clean.
- [ ] **Step 5: integrate** — dense MLP fast path (Task 8 consumer) switches to fused mode for seq==1; standalone swiglu launch skipped. Prefill (seq>1, tile kernel) unchanged.
- [ ] **Step 6:** determinism 3×memcmp; selfcheck "Paris"; micro + e2e (>=3 runs); commit `perf: swiglu fused into tc decode epilogue`.

---

### Task 11: B3 — TMA bulk-copy staging

**Files:**
- Modify: `src/qwen_tc_gemm.cu` (probe + `TcGemmDecodeKernel` staging), CMake arch-`a` block already permits required PTX
- Test: `tests/test_blackwell_tc_mma.cu`

**Interfaces:**
- Produces: `tc::ProbeTmaBulkCopy()` (host, returns bool) + internal `stage_chunk_tma` replacing `stage_chunk` cp.async lambda behind runtime capability check with automatic fallback.

- [ ] **Step 1: failing probe test** — create `CUtensorMap` over 512 KB synthetic weight buffer (3D: `[K/2 bytes, N-tile, buf]`), single-warp `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes` copy to smem, compare vs global. Descriptor creation via `cuTensorMapEncodeTiled` driver API (link `CUDA_DRIVER_LIBRARY`).
- [ ] **Step 2:** FAIL/not-supported path verified on refusal. **Step 3:** implement probe kernel + encode helper. **Step 4:** PASS + sanitizer clean.
- [ ] **Step 5: integrate** — pipeline depth 4–6, mbarrier arrive/expect-tx per chunk; `wait` parity swapped per buffer; keep cp.async ring as compile-time-retained fallback selected once at first call (`static const bool has_tma = ProbeTmaBulkCopy()`).
- [ ] **Step 6:** determinism + selfcheck + e2e (>=3 runs); record effective GB/s = `bytes_weights_read / kernel_time`; commit `perf: tma bulk-tensor staging in tc decode path`.

---

### Task 12: B4 — Release profiling + final numbers

**Files:**
- Modify: `docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md` (new round section)
- No prod code changes unless profiling reveals a hot fix (each such fix = own commit with tests).

- [ ] Configure/build `cmake --preset linux-release-znver4 && cmake --build --preset linux-release-znver4`.
- [ ] nsys profile of one generation: `nsys profile -o /tmp/opencode/qwen_tok ctest -R test_blackwell_perf_qwen` (or direct harness binary); ncu on `TcGemmDecodeKernel` for achieved DRAM throughput.
- [ ] Apply at most the top profile-indicated tuning (block shape / `__launch_bounds__` / depth) with A/B >=3 runs.
- [ ] Record: tok/s @5090 (target 280–320), ttft, effective GB/s, extrapolated tok/s @5060 Ti (bandwidth-scaled), %roofline. Commit `docs: phase 4 results`.

---

### Task 13: Final sweep

- [ ] Full sequential suite BOTH configs: `ctest --test-dir build/linux-debug` (103/103) and release preset.
- [ ] Selfcheck stress: qwen fast-math selfcheck 3 cycles deterministic ("Paris").
- [ ] finishing-a-development-branch skill.

## Risks

- Fragment/accumulator index mistakes in B2 -> absorbed by Task 10 Step 1 tile-level probe.
- TMA descriptor/mbarrier geometry -> absorbed by Task 11 Step 1 single-copy probe.
- Expert-store layout change (Task 9) invalidates persisted caches -> store is rebuilt at load; no on-disk format change.
- Dedup numeric regressions -> caught by per-cluster memcmp gates (Tasks 2–6) and untouched bit-exact suites.
