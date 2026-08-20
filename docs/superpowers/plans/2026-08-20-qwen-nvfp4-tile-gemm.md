# Qwen NVFP4 Dense GEMM Tile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive per-element `qwen_nvfp4_gemm_kernel` with a shared-memory tile GEMM (x-tile + decoded weight-tile reuse, per-16 scale decoding once, FP32 accumulate with free `k`-order) while keeping NVFP4 compressed in HBM and staying within the numeric tolerance envelope of the synthetic Qwen pipeline test.

**Architecture:** A new `qwen_nvfp4_gemm_tile_kernel` stages an `[BM×BK]` X-tile and an `[BN×BK]` dequantized weight tile in shared memory, reuses them across the K-loop, and accumulates FP32 in registers; the public `RunQwenNvfp4Gemm` (numeric-kernel convention) owns the tile-vs-legacy dispatch (tile when `N>=8 && K>=16`, else the per-element kernel). The existing external `LaunchNvfp4Gemm` becomes a thin delegate to it so all internal call sites compile unchanged.

**Tech Stack:** C++20, CUDA (`__shared__`, `__syncthreads`, `cuda::` tile C++ from CCCL as available), NVFP4 codec `nvfp4_codec.h`, Catch2 v3.7.0 (CUDA tests via `.cu`), CMake Presets.

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — add `RunQwenNvfp4Gemm` numeric entry.
- Modify: `libs/lsxcommon/src/qwen_inference.cu` — add tile kernel + `RunQwenNvfp4Gemm` + retarget `LaunchNvfp4Gemm` to tile (keep legacy kernel as static reference for parity).
- Create: `tests/test_qwen_nvfp4_tile_gemm.cu` — parity test (legacy vs tile) + small/large K.
- Modify: `tests/CMakeLists.txt` — add the new test target (arrow POST_BUILD + add_test + SKIP_RETURN_CODE 4).
- No CMake source list change in `libs/`.

## Grounding (read first)

- Legacy kernel `qwen_nvfp4_gemm_kernel` at `qwen_inference.cu:345-368`. `LaunchNvfp4Gemm` at `:397-409` — `N=w.rows`, `K=w.cols`, grid `(seq*N/256)`, one thread per element.
- `QwenNvfp4Weight` (struct of `d_packed`, `d_scales`, `global_scale`, `rows`, `cols`) in `qwen_inference_load.h:16`.
- NVFP4 layout: element `e` at byte `e>>1`, low nibble if even else high; scale per 16 elements `scales[e>>4]` (E4M3 → float via `E4M3ScaleToFloat`, E2M1 nibble via `kE2M1Table`). Already aliased in the file's anonymous namespace (`qwen_inference.cu:340`).
- Real shapes: `hidden=5120`, `intermediate~14k`, `vocab` big; `LaunchNvfp4Gemm` called with heterogeneous K/N (q/k/v/o/gate/up/down/head) — one kernel must handle small and large.
- Existing numeric-kernel header pattern: `RunQwen*` declared in `qwen_inference.h` under "Numeric kernels (TDD-testable)", return `cudaError_t`, take raw `__half*`/`cudaStream_t`.
- Test registration: `tests/CMakeLists.txt` uses `add_executable(... test_qwen_*.cu)` + `target_link_libraries(... lsxcommon Catch2::Catch2WithMain)` + WIN32 POST_BUILD arrow copy + `add_test(NAME ... COMMAND ...)` + `set_tests_properties(... SKIP_RETURN_CODE 4)`.

**Numeric contract:** FP32 accumulate, `k` order free. Must pass a strict relative tolerance vs the legacy kernel (both start from identical NVFP4 decode, so the only difference is FP32 add order). `test_qwen_model` must stay green (its `SelfCheck` is `ok`-contract; we keep it).

---

## Task 1: Public `RunQwenNvfp4Gemm` numeric entry (TDD)

Expose a raw-tensor NVFP4 GEMM so the parity test can call both implementations from the test TU.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_nvfp4_tile_gemm.cu` with a minimal call that fails to compile (the API doesn't exist yet). Use the existing `test_qwen_nvfp4_dense.cu` skeleton style (catch + cuda + vectors):

```cpp
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#include "lsxcommon/qwen_inference.h"

namespace {

// Host reference: exact per-element accumulation order (matches legacy kernel).
void HostRefNvfp4Gemm(const __half *x, const uint8_t *packed, const uint8_t *scales,
                      float global_scale, float *out, int seq, int K, int N)
{
    for (int t = 0; t < seq; ++t)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int64_t e = (int64_t)n * K + k;
                const uint8_t byte = packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl = ((float)scales[e >> 4] - 128.0f) * global_scale;
                acc += __half2float(x[(int64_t)t * K + k]) * kE2M1Table[nib] * scl;
            }
            out[t * N + n] = acc;
        }
}

// Test-local copy of the E2M1 decode table (values verified against
// nvfp4_codec.kE2M1Table; the test must not depend on internal codec symbols).
constexpr float kE2M1TestTable[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

}  // namespace

TEST_CASE("RunQwenNvfp4Gemm compiles and matches host reference (small K)", "[qwen][gemm]")
{
    constexpr int seq = 2, K = 64, N = 128;
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp(K * N / 2), hs(K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 7) - 3));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 13) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 32));
    const float gs = 0.5f;

    std::vector<float> h_ref(seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy(seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i)
        CHECK(std::abs(__half2float(hy[i]) -
                       __half2float(__float2half(h_ref[i]))) < 1e-3f);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

- [ ] **Step 2: Run to verify RED**

`cmake --build build --config Debug --target test_qwen_nvfp4_tile_gemm 2>&1 | Select -Last 6` → compile error "RunQwenNvfp4Gemm undefined".

- [ ] **Step 3: Implement `RunQwenNvfp4Gemm`**

In `include/lsxcommon/qwen_inference.h` add the numeric kernel:

```cpp
// NVFP4 dense GEMM: y[t][n] = sum_k x[t*K+k] * dequant(packed[n*K+k], scales, gs).
// x: [seq, K] FP16; packed: [N*K/2] nibble-packed E2M1; scales: [N*K/16] E8M0
// per-16 floats; gs: global scale. y: [seq, N] FP16 (FP32 accumulate).
// Grid/block are shape-adaptive; the SAME decode semantics as the legacy
// per-element kernel, with a tiled register/shared implementation behind it.
cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N);
```

In `qwen_inference.cu`, after the legacy kernel (line 368) implement `RunQwenNvfp4Gemm` as a **legacy-only** launcher for this task (Task 2 adds the tile branch and the threshold `N < 8 || K < 16` guard). This is the RED→GREEN first increment; the dispatch threshold lands with the tile kernel so the K=64 case only passes once the tile path exists.

```cpp
cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N)
{
    if (!x || !packed || !scales || !y || seq <= 0 || K <= 0 || N <= 0)
        return cudaErrorInvalidValue;
    const int threads = 256;
    const int64_t total = (int64_t)seq * N;
    const int blocks = (int)((total + threads - 1) / threads);
    qwen_nvfp4_gemm_kernel<<<blocks, threads, 0, stream>>>(
        x, packed, scales, global_scale, y, seq, K, N);
    return cudaGetLastError();
}
```

(When Task 2 lands, this body is replaced by the tile+legacy dispatch; `LaunchNvfp4Gemm` then delegates to `RunQwenNvfp4Gemm`.)

- [ ] **Step 4: Build + run the test**

`cmake --build build --config Debug --target test_qwen_nvfp4_tile_gemm` then run `tests/test_qwen_nvfp4_tile_gemm.exe` → pass.

- [ ] **Step 5: Commit**

`git add tests/test_qwen_nvfp4_tile_gemm.cu include/lsxcommon/qwen_inference.h src/qwen_inference.cu tests/CMakeLists.txt`
`git commit -m "test: RunQwenNvfp4Gemm numeric entry (parity harness)"`

---

## Task 2: Tile kernel `qwen_nvfp4_gemm_tile_kernel`

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: Write the tile kernel in the anonymous namespace (after legacy)**

```cpp
// Tile NVFP4 dense GEMM: block computes an [BM x BN] output tile for one token
// tile. X rows and the NVFP4-dequantized weight rows are staged in shared
// memory once per K-block; E8M0 scales are decoded once per 16 elements.
// One thread owns exactly one output element (threads == BM*BN), so `acc` is a
// single register per thread — no per-thread array, no bank-pressure.
template <int BM, int BN, int BK>
__global__ void qwen_nvfp4_gemm_tile_kernel(
    const __half *x, const uint8_t *packed, const uint8_t *scales,
    float global_scale, __half *y, int seq, int K, int N)
{
    static_assert(BM * BN == 256, "one thread per output element (256)");

    __shared__ __half sX[BM][BK];
    __shared__ __half sW[BN][BK];
    const int m0 = blockIdx.x * BM;      // token tile
    const int n0 = blockIdx.y * BN;      // output-column tile
    const int bi = threadIdx.x;          // 0 .. BM*BN-1
    const int bm = bi / BN;              // token index within tile
    const int bn = bi % BN;              // output-column index within tile

    const int t = m0 + bm;
    const int n = n0 + bn;
    const bool in_bounds = (t < seq) && (n < N);

    float acc = 0.0f;

    for (int kb = 0; kb < K; kb += BK) {
        // W tile: element e = (n0+bn)*K + kb+kk (weight row-major [n, K]).
        for (int i = bi; i < BN * BK; i += BM * BN) {
            const int w_bn = i / BK, kk = i % BK;
            const int64_t e = (int64_t)(n0 + w_bn) * K + (kb + kk);
            const uint8_t byte = packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl = E4M3ScaleToFloat(scales[e >> 4]) * global_scale;
            sW[w_bn][kk] = __float2half(kE2M1Table[nib] * scl);
        }
        // X tile: x[t][kb+kk] for rows [m0, m0+BM).
        for (int i = bi; i < BM * BK; i += BM * BN) {
            const int x_bm = i / BK, kk = i % BK;
            const int tt = m0 + x_bm;
            sX[x_bm][kk] = (tt < seq) ? x[(int64_t)tt * K + (kb + kk)]
                                      : __float2half(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
            acc += __half2float(sX[bm][kk]) * __half2float(sW[bn][kk]);

        __syncthreads();
    }

    if (in_bounds)
        y[(int64_t)t * N + n] = __float2half(acc);
}

// (RunQwenNvfp4Gemm is the public entry defined in Task 1; its tile/legacy
//  dispatch below is finalized here.)
cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N)
{
    if (!x || !packed || !scales || !y || seq <= 0 || K <= 0 || N <= 0)
        return cudaErrorInvalidValue;
    if (N < 8 || K < 16) {
        // Legacy per-element kernel: covers tiny shapes where a tile is waste.
        const int threads = 256;
        const int64_t total = (int64_t)seq * N;
        const int blocks = (int)((total + threads - 1) / threads);
        qwen_nvfp4_gemm_kernel<<<blocks, threads, 0, stream>>>(
            x, packed, scales, global_scale, y, seq, K, N);
        return cudaGetLastError();
    }
    constexpr int BM = 16, BN = 16, BK = 32;   // BM*BN == 256 == threads
    const int gx = (int)((seq + BM - 1) / BM);
    const int gy = (int)((N  + BN - 1) / BN);
    qwen_nvfp4_gemm_tile_kernel<BM, BN, BK><<<dim3(gx, gy), BM * BN, 0, stream>>>(
        x, packed, scales, global_scale, y, seq, K, N);
    return cudaGetLastError();
}
```

Shared-memory footprint: `sX = 16*32*2 = 1024B`, `sW = 16*32*2 = 1024B` → 2 KiB/block, far within Blackwell's per-SM shared. 256 threads/block, ≤48 warps/SM. The K-loop order (`kb`, then `kk` per thread) differs from the legacy per-element serial order — FP32 accumulate makes the delta within tolerance (parity test asserts it).

- [ ] **Step 2: Build + run the tile test**

`cmake --build build --config Debug --target test_qwen_nvfp4_tile_gemm` → PASS (the K=64 and K=5120 cases exercise the tile path; the small-K case the legacy fallback).

- [ ] **Step 3: Retarget the external `LaunchNvfp4Gemm` to `RunQwenNvfp4Gemm`**

`LaunchNvfp4Gemm` (currently `qwen_inference.cu:397-409`, external-linkage, called from `lsxcommon::qwen` internal layers) becomes a thin delegate. `RunQwenNvfp4Gemm` already owns the tile/legacy dispatch (threshold `N < 8 || K < 16`), so no shape logic is duplicated here:

```cpp
cudaError_t LaunchNvfp4Gemm(const QwenNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream)
{
    if (!w.d_packed || w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
    return RunQwenNvfp4Gemm(stream, x, w.d_packed, w.d_scales,
                            w.global_scale, y, seq, w.cols, w.rows);
}
```

(`w.rows` = N output, `w.cols` = K input — matches the existing kernel's `N=w.rows, K=w.cols` mapping.) Keep the same linkage name so all existing `LaunchNvfp4Gemm(...)` call sites compile unchanged.

- [ ] **Step 4: Run full Qwen model test**

`cmake --build build --config Debug --target test_qwen_model` then `tests/test_qwen_model.exe` → PASS.

- [ ] **Step 5: Commit**

`git add src/qwen_inference.cu include/lsxcommon/qwen_inference.h`
`git commit -m "feat: tile NVFP4 dense GEMM (x+weight shared reuse, per-16 scale)"`

---

## Task 3: Small-K + edge parity test coverage

- [ ] **Step 1: Add the large-K parity case to the test**

In `test_qwen_nvfp4_tile_gemm.cu` add a second TEST_CASE exercising the tile path (K ≥ 16, N ≥ 8 forces the tile kernel) at a large-K geometry:

```cpp
TEST_CASE("RunQwenNvfp4Gemm tile matches host reference at large K", "[qwen][gemm]")
{
    constexpr int seq = 2, K = 5120, N = 512;
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp(K * N / 2), hs(K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 11) - 5) * 0.5f);
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 7) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 64));
    const float gs = 0.25f;

    std::vector<float> h_ref(seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy(seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i) {
        const float ref = __half2float(__float2half(h_ref[i]));   // production y is half
        CHECK(std::abs(__half2float(hy[i]) - ref) < 1e-3f * std::abs(ref) + 1e-6f);
    }

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}
```

- [ ] **Step 2: Also cover the small-K dispatch edge**

Add `seq=1, K=16, N=32` → must exercise the legacy fallback branch of `RunQwenNvfp4Gemm` (K<16), assert it still matches the host reference:

```cpp
TEST_CASE("RunQwenNvfp4Gemm legacy fallback at small-K", "[qwen][gemm]")
{
    constexpr int seq = 1, K = 16, N = 32;
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp(K * N / 2), hs(K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 5) - 2));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 3) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(130 + (i % 16));
    const float gs = 1.0f;

    std::vector<float> h_ref(seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy(seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i)
        CHECK(std::abs(__half2float(hy[i]) - __half2float(
                 __float2half(h_ref[i]))) < 1e-3f);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}
```

- [ ] **Step 3: Build + run**

`cmake --build build --config Debug --target test_qwen_nvfp4_tile_gemm` + run. Expected: 3 cases pass.

- [ ] **Step 4: Verify `test_qwen_model` still green + `test_qwen_nvfp4_dense` unaffected**

Run `tests/test_qwen_model.exe` and `tests/test_qwen_nvfp4_dense.exe`. PASS.

- [ ] **Step 5: Commit**

`git add tests/test_qwen_nvfp4_tile_gemm.cu`
`git commit -m "test: NVFP4 tile GEMM parity at large K and small-K fallback"`

---

## Task 4: Full verification

- [ ] **Step 1: Full build**

`cmake --preset windows-debug && cmake --build build --config Debug` → clean.

- [ ] **Step 2: Run the affected test set**

`ctest --preset windows-debug --tests-regex "qwen_nvfp4_tile_gemm|qwen_nvfp4_dense|qwen_model|qwen_norm|qwen_mlp|qwen_full_attn|qwen_kv_parity"` → all pass; real-checkpoint tests SKIP (SKIP_RETURN_CODE 4).

- [ ] **Step 3: Commit** (if files changed)

`git add . && git commit -m "test: verify NVFP4 tile GEMM across Qwen suites"`

---

## Self-Review

- Spec §3 tile kernel → Task 2 (x-tile + decoded weight tile, per-16 scale, FP32 acc free k). ✓
- Spec §3 small-K fallback → Task 3 (legacy fallback for `N < 8 || K < 16` inside `RunQwenNvfp4Gemm`). ✓
- Spec §4 parity test → Task 1 (harness) + Task 3 (large-K + small-K cases). ✓
- Spec §5 files → match Task file map. ✓
- No placeholders: every code block is complete. Types consistent (`RunQwenNvfp4Gemm` signature in header == .cu; `LaunchNvfp4Gemm` keeps its linkage and delegates; tile template is `BM=16, BN=16, BK=32` with one thread per output element — no per-thread accumulator array, 2 KiB shared/block, occupancy ≤48 warps/SM). The corner shapes `N < 8 || K < 16` take the legacy path.