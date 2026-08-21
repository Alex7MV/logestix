# Qwen Full-Attention Parallel Softmax — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thread-0-only production `full_attn_softmax_kernel` with a block-wide online-softmax attention kernel (parallel scores, block-reduced m/sum, single per-d exp pass) — FP32, tolerance envelope, preserving GQA/kv_base/causal and the `d_attn` layout.

**Architecture:** A new public numeric entry `RunQwenFullAttnKernel` (external linkage, mirroring the step-1 `RunQwenNvfp4Gemm` pattern) drives a rewritten `full_attn_softmax_kernel`: scores computed chunk-per-thread into shared `slogits`, global m/sum via a shared-tree block reduction, then per-output-dim `d` pass with shared `exp(s-m)*inv`. TDD: a new harness `test_qwen_full_attn_parallel.cu` compares the production kernel against an exact FP64 host reference (the same `attn_ref_fp64` style used by the existing suite) for a small and a long-context case.

**Tech Stack:** C++20, CUDA (`__shared__`, `__syncthreads`, shared-memory tree reduction), Catch2 (CUDA tests via `test_qwen_cu` OBJECT lib), CMake Presets.

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare `RunQwenFullAttnKernel`.
- Modify: `libs/lsxcommon/src/qwen_inference.cu` — rewrite `full_attn_softmax_kernel`; add `RunQwenFullAttnKernel` (external linkage); keep `RunQwenFullAttention` launch unchanged (it already passes grid/threads/smem).
- Create: `tests/test_qwen_full_attn_parallel.cu` — harness (small + long-context).
- Modify: `tests/CMakeLists.txt` — add the `.cu` to `test_qwen_cu` OBJECT lib + the `foreach` executable list.
- No other CMake change.

## Grounding (read first)

- Current kernel `full_attn_softmax_kernel` at `qwen_inference.cu:229-268` (inside the anonymous namespace 335-654 at the earlier position; note line numbers shifted after the step-1 edits — re-find by `rg -n "full_attn_softmax_kernel" qwen_inference.cu`).
- Launch site inside `RunQwenFullAttention` (`qwen_inference.cu:747-759`): `grid = dim3(seq, NH)`, `threads = min(HD, 512)`, `smem = (kv_base+seq+1)*sizeof(float)`. This contract is preserved.
- `kvh = h / (heads/kv_heads)`, `ctx = kv_base + t`, `scale = 1/sqrt(HD)`.
- `RunQwenNvfp4Gemm` step-1 pattern: public `cudaError_t Run...(cudaStream_t, ...)` declared in header, defined OUTSIDE the anonymous namespace (in `lsxcommon::qwen`) — that is what gives the test external linkage. Note: `qwen_inference.cu` has TWO anonymous namespaces (17-271 and 335-654); the kernel lives in the second one. The public entry must be outside both, after line 654's `} // namespace`.
- Test registration pattern (from `test_qwen_nvfp4_tile_gemm.cu` step 1): add `.cu` to the `test_qwen_cu` OBJECT lib + the `foreach(qt ...)` executable list; the OBJECT lib already has Arrow include dirs (added in step 1).

**Numeric contract:** FP32 accumulate, tolerance `1e-3 * |ref| + 1e-6` vs exact FP64 host reference. `test_qwen_model` (synthetic) stays green. `test_qwen_full_attn.cu` tests ITS OWN kernels and is untouched.

---

## Task 1: Public `RunQwenFullAttnKernel` + test harness (TDD RED→GREEN)

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_full_attn_parallel.cu` (mirrors the step-1 harness structure):

```cpp
// Production full-attention (causal GQA) parallel-softmax parity harness.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"

namespace {

// Exact FP64 reference: out[t][h][d] = sum_{s<=t} exp((q·k_s)*scale) v_s[d]
//                / (sum_{s<=t} exp((q·k_s)*scale) + 1e-10). kv_offset shifts the
// k/v row base for the incremental-decode kv_base case.
void AttnRefFp64(const __half *q, const __half *k, const __half *v,
                 __half *out, int seq, int heads, int head_dim,
                 int kv_heads, float scale, int kv_base)
{
    const int kvh_ratio = heads / kv_heads;
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const int kvh = h / kvh_ratio;
            const int ctx = kv_base + t;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0, denom = 0.0;
                for (int s = 0; s <= ctx; ++s) {
                    double score = 0.0;
                    const __half *qr = q + ((int64_t)t * heads + h) * head_dim;
                    const __half *kr = k + ((int64_t)s * kv_heads + kvh) * head_dim;
                    for (int dd = 0; dd < head_dim; ++dd)
                        score += __half2float(qr[dd]) * __half2float(kr[dd]);
                    score *= scale;
                    const double ex = exp(score);
                    denom += ex;
                    acc += ex * __half2float(
                        v[((int64_t)s * kv_heads + kvh) * head_dim + d]);
                }
                out[((int64_t)t * heads + h) * head_dim + d] =
                    __float2half((float)(acc / (denom + 1e-10)));
            }
        }
}

}  // namespace

TEST_CASE("RunQwenFullAttnKernel small case matches FP64 ref", "[qwen][attn][full]")
{
    constexpr int seq = 4, heads = 2, kv_heads = 1, head_dim = 8, kv_base = 0;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int nq = seq * heads * head_dim;
    const int nkv = (seq + kv_base) * kv_heads * head_dim;   // k/v rows

    std::vector<__half> hq(nq), hk(nkv), href(nq);
    std::vector<__half> hv(nkv);   // v has the same row count as k
    // deterministic pseudo-random fill
    for (int i = 0; i < nq; ++i) hq[i] = __float2half((float)((i * 7 + 1) % 13) * 0.2f - 1.2f);
    for (int i = 0; i < nkv; ++i) hk[i] = __float2half((float)((i * 11 + 3) % 17) * 0.3f - 1.5f);
    for (int i = 0; i < nkv; ++i) hv[i] = __float2half((float)((i * 5 + 2) % 11) * 0.4f - 0.8f);
    AttnRefFp64(hq.data(), hk.data(), hv.data(), href.data(),
                seq, heads, head_dim, kv_heads, scale, kv_base);

    __half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, nq * 2); cudaMalloc(&dk, nkv * 2); cudaMalloc(&dv, nkv * 2);
    cudaMalloc(&dout, nq * 2);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nkv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nkv * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenFullAttnKernel(
                nullptr, dq, dk, dv, dout, seq, heads, head_dim,
                kv_heads, scale, kv_base) == cudaSuccess);

    std::vector<__half> hout(nq);
    cudaMemcpy(hout.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nq; ++i)
        CHECK(std::abs(__half2float(hout[i]) - __half2float(href[i])) < 1e-2f);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}
```

The harness is RED: `RunQwenFullAttnKernel` is not yet declared.

- [ ] **Step 2: Run to verify RED**

`cmake --build build --config Debug --target test_qwen_full_attn_parallel 2>&1 | Select -Last 6` → "RunQwenFullAttnKernel undefined".

- [ ] **Step 3: Add header declaration**

In `include/lsxcommon/qwen_inference.h`, after the `RunQwenLinearAttention` declaration, add:

```cpp
// Scaled dot-product causal attention (GQA), FP16 in/out, FP32 accumulate.
// q: [seq, heads, head_dim]; k/v: [seq, kv_heads, head_dim] base rows
// (kv_base offsets the base for a persistent cache); out: [seq, heads, head_dim].
cudaError_t RunQwenFullAttnKernel(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base);
```

- [ ] **Step 4: Implement `RunQwenFullAttnKernel` (initial: launch a copy of the legacy thread-0 kernel so the harness passes)**

The new public entry initially launches the **existing** kernel (which is the reference semantics). In `qwen_inference.cu`, after the anonymous-namespace close at the current `} // namespace` for the inference section, add:

```cpp
cudaError_t RunQwenFullAttnKernel(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale, int kv_base)
{
    if (!q || !k || !v || !out || seq <= 0 || heads <= 0 || head_dim <= 0 ||
        kv_heads <= 0 || kv_base < 0)
        return cudaErrorInvalidValue;
    const dim3 grid(seq, heads);
    const int threads = std::min(head_dim, 512);
    const size_t smem = (static_cast<size_t>(kv_base) + seq + 1) * sizeof(float);
    full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
        q, k, v, out, seq, heads, head_dim, kv_heads, scale, kv_base);
    return cudaGetLastError();
}
```

This compiles because `full_attn_softmax_kernel` is the existing (thread-0) kernel in scope (anonymous namespace, visible via the implicit using-directive). The harness passes against the legacy semantics. (In Task 2 the body is replaced by the rewritten kernel; other callers keep the same launch.)

- [ ] **Step 5: Build + run the test**

`cmake --build build --config Debug --target test_qwen_full_attn_parallel` → built. Run `tests/test_qwen_full_attn_parallel.exe` → PASS (vs FP64 ref with 1e-2 tolerance). The exact `1e-2` tolerance absorbs half-roundtrip on 8-dim heads.

- [ ] **Step 6: Commit**

`git add tests/test_qwen_full_attn_parallel.cu include/lsxcommon/qwen_inference.h src/qwen_inference.cu tests/CMakeLists.txt`
`git commit -m "test: RunQwenFullAttnKernel numeric entry (attention parity harness)"`

---

## Task 2: Block-wide parallel softmax kernel (replace the body)

**Files:**
- Modify: `src/qwen_inference.cu`

- [ ] **Step 1: Replace `full_attn_softmax_kernel` body**

Replace the kernel (keeping its name/signature/launch contract) with the chunked parallel version — the committed code is:

```cpp
__global__ void full_attn_softmax_kernel(
    const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base)
{
    static_assert(blockDim.x <= 512, "block reduction scratch is 512 floats");
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int ctx = kv_base + t;
    const int nscores = ctx + 1;
    const int threads = blockDim.x;

    const __half *qrow = q + ((int64_t)t * heads + h) * head_dim;
    const __half *kbase = k + (int64_t)kvh * head_dim;
    const __half *vbase = v + (int64_t)kvh * head_dim;

    extern __shared__ float slogits[];      // dynamic: [nscores]
    __shared__ float sreduce[512];          // separate block-reduction scratch

    // 1. Scores, chunk-per-thread (parallel dots).
    for (int j = threadIdx.x; j < nscores; j += threads) {
        const __half *krow = kbase + (int64_t)j * kv_heads * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qrow[d]) * __half2float(krow[d]);
        slogits[j] = dot * scale;
    }
    __syncthreads();

    // 2. Global max via block reduction (scratch, NOT slogits).
    float m = -1e30f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        m = fmaxf(m, slogits[j]);
    sreduce[threadIdx.x] = m;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] = fmaxf(sreduce[threadIdx.x], sreduce[threadIdx.x + stride]);
        __syncthreads();
    }
    m = sreduce[0];
    __syncthreads();   // ensures all threads see `m` before the sum pass

    // 3. Softmax denominator via block reduction (one exp per score).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        sum += __expf(slogits[j] - m);
    sreduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / (sreduce[0] + 1e-10f);
    __syncthreads();   // `inv` visible to all threads before the value pass

    // 4. Per-output-dim value pass: thread d sums all scores with the SHARED
    //    exp(s-m)*inv. Stride-d loop (covers head_dim up to threads; idles when
    //    threads > head_dim).
    for (int d = threadIdx.x; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int j = 0; j < nscores; ++j) {
            const __half *vrow = vbase + (int64_t)j * kv_heads * head_dim;
            acc += __expf(slogits[j] - m) * inv * __half2float(vrow[d]);
        }
        out[((int64_t)t * heads + h) * head_dim + d] = __float2half(acc);
    }
}
```

**Reduction-scratch contract:**
- `sreduce[512]` is a separate *static* shared array — it cannot alias `slogits` (dynamic `extern __shared__`). CUDA allows static + dynamic shared coexistence; the reduction writes only `sreduce`, never `slogits` (which the sum and value passes re-read).
- The caller's dynamic `smem` must host `slogits[nscores]` **plus the block reduction scratch** — handled by passing `(nscores + 512) * sizeof(float)` at both launch sites (Step 2).
- No extra sync hazards: `slogits` is written (score pass) and synced; then only read afterward. The `sreduce` tree syncs are complete per phase.
- Value pass uses the stride-d loop, so it is robust for any `threads` vs `head_dim` ordering (real 9B: `head_dim ≤ 512` per `threads = min(HD,512)`, so `threads == head_dim` for the common case; idle threads beyond `head_dim` are skipped).

- [ ] **Step 2: Update both launch sites' smem to include the reduction scratch**

Both `RunQwenFullAttention` (`qwen_inference.cu:~753`) and `RunQwenFullAttnKernel` (added in Task 1) must pass `smem = (kv_base+seq+1)*sizeof(float) + 512*sizeof(float)`.

In `RunQwenFullAttention`:

```cpp
        const size_t smem =
            ((static_cast<size_t>(kv_base) + seq + 1) + 512) * sizeof(float);
```

In `RunQwenFullAttnKernel`:

```cpp
    const size_t smem = ((static_cast<size_t>(kv_base) + seq + 1) + 512) * sizeof(float);
```

- [ ] **Step 3: Build + run the harness**

`cmake --build build --config Debug --target test_qwen_full_attn_parallel` → PASS (same tolerance vs FP64 ref). Then run `tests/test_qwen_model.exe` (full pipeline through `RunQwenFullAttention`) → PASS.

- [ ] **Step 4: Commit**

`git add src/qwen_inference.cu`
`git commit -m "feat: parallel chunked softmax in Qwen full attention (block-reduce m/sum)"`

---

## Task 3: Long-context parity case

- [ ] **Step 1: Add the long-context TEST_CASE**

Append to `tests/test_qwen_full_attn_parallel.cu`:

```cpp
TEST_CASE("RunQwenFullAttnKernel long-context matches FP64 ref", "[qwen][attn][full]")
{
    constexpr int seq = 64, heads = 2, kv_heads = 1, head_dim = 8, kv_base = 2000;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int nq = seq * heads * head_dim;
    const int nkv = (seq + kv_base) * kv_heads * head_dim;

    std::vector<__half> hq(nq), hk(nkv), hv(nkv), href(nq);
    for (int i = 0; i < nq; ++i) hq[i] = __float2half((float)((i * 7 + 1) % 13) * 0.2f - 1.2f);
    for (int i = 0; i < nkv; ++i) hk[i] = __float2half((float)((i * 11 + 3) % 17) * 0.3f - 1.5f);
    for (int i = 0; i < nkv; ++i) hv[i] = __float2half((float)((i * 5 + 2) % 11) * 0.4f - 0.8f);
    AttnRefFp64(hq.data(), hk.data(), hv.data(), href.data(),
                seq, heads, head_dim, kv_heads, scale, kv_base);

    __half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, nq * 2); cudaMalloc(&dk, nkv * 2); cudaMalloc(&dv, nkv * 2);
    cudaMalloc(&dout, nq * 2);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nkv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nkv * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenFullAttnKernel(
                nullptr, dq, dk, dv, dout, seq, heads, head_dim,
                kv_heads, scale, kv_base) == cudaSuccess);

    std::vector<__half> hout(nq);
    cudaMemcpy(hout.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nq; ++i)
        CHECK(std::abs(__half2float(hout[i]) - __half2float(href[i])) < 1e-2f);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}
```

- [ ] **Step 2: Build + run**

`cmake --build build --config Debug --target test_qwen_full_attn_parallel` → run → 2 cases pass.

- [ ] **Step 3: Verify `test_qwen_model` + `test_qwen_full_attn` still green**

Run both binaries. PASS.

- [ ] **Step 4: Commit**

`git add tests/test_qwen_full_attn_parallel.cu`
`git commit -m "test: long-context attention parity (kv_base) case"`

---

## Task 4: Full verification

- [ ] **Step 1: Full build**

`cmake --preset windows-debug && cmake --build build --config Debug` → clean.

- [ ] **Step 2: Run the affected + Qwen suite**

`ctest --preset windows-debug --tests-regex "qwen_full_attn_parallel|qwen_full_attn|qwen_model|qwen_nvfp4_dense|qwen_mlp|qwen_kv_parity|qwen_rotary|qwen_norm"` → all pass.

- [ ] **Step 3: Commit** (if files changed)

`git add . && git commit -m "test: verify parallel attention across Qwen suites"`

---

## Self-Review

- Spec §3 parallel scores → Task 2 score pass (chunk-per-thread). ✓
- Spec §3 block-reduce m/sum → Task 2 reductions via `sreduce[]`. ✓
- Spec §3 value pass per d → Task 2 final pass (shared exp(s-m)*inv, stride-d loop). ✓
- Spec §4 harness → Task 1 (+ Task 3 long-context). ✓
- Spec §5 files → matches. ✓
- No placeholders: every code block is complete. Types consistent: `RunQwenFullAttnKernel` signature in header == .cu; both data-carrying callers (`RunQwenFullAttention` + `RunQwenFullAttnKernel`) updated for the `+512` reduction scratch in `smem`. The value pass is a stride-d loop, robust to any `threads` vs `head_dim` ordering.
- Cross-check: `test_qwen_full_attn.cu` (pre-existing, tests its own `test_causal_softmax_attn_kernel`) is untouched and stays green; our new file has a distinct name (no Catch2 case collision).