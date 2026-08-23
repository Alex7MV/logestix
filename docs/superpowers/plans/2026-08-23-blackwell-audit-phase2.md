# Blackwell Audit Phase 2 — Occupancy & Launch Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate measured launch-config waste in the Qwen/DeepSeek hot paths without changing any numeric result (bit-exact invariant), and document occupancy evidence for F-05.

**Architecture:** All changes preserve per-output-element floating-point operation order exactly. Verification gates: existing real-model suites (qwen 22 tests, deepseek 8 tests run with weights on this machine) + new bit-exact kernel equivalence tests + perf harnesses from Phase 0. GLM targets are OUT OF SCOPE here (no weights on this stand -> unverifiable); F-06 (-fmad scope) deferred to the GLM-weights-available session.

**Tech Stack:** CUDA 13.3 (sm_120), Catch2, CMake windows-debug preset.

**Key constraint discovered during survey:** `qwen_nvfp4_gemm_kernel` (per-element, FP32 dequant path) and `qwen_nvfp4_gemm_tile_kernel` (smem, FP16-rounded dequant path) are NOT bit-equivalent — never route traffic between them. Same-math specializations of the tile kernel ARE safe.

---

### Task 1: seq==1 tile specialization (decode GEMM waste fix)

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference.cu:1141-1146` (dispatch in `RunQwenNvfp4Gemm`)
- Test: extend `tests/test_qwen_nvfp4_tile_gemm.cpp`

**Rationale:** at `seq=1` the `<16,16,32>` tile stages 16 token rows and computes 15 dead ones per 16-column tile. `<1,256,32>` keeps `BM*BN==256`, stages only row 0, and preserves exact accumulation order (BK=32 sequential chunks, half*half products).

- [ ] **Step 1: Write failing equivalence test** — in `test_qwen_nvfp4_tile_gemm.cpp` add a case that runs the CURRENT `<16,16,32>` path and a reference CPU/host mirror for random inputs with seq=1 (incl. K%32!=0 tail, N%16!=0 tail), records logits hash. Then after Step 2 the same inputs must produce byte-identical outputs through `RunQwenNvfp4Gemm`. Follow the file's existing fixture pattern for building packed/scales buffers.

- [ ] **Step 2: Add the specialization**

```cpp
    constexpr int BK = 32;
    if (seq == 1) {
        // Decode fast path (audit F-05): <1,256,32> stages one token row
        // instead of 16 dead ones. Bit-exact: same BK-chunked sequential
        // accumulation, same FP16-rounded smem dequant as <16,16,32>.
        constexpr int BM1 = 1, BN1 = 256;
        const int gy = (int)((N + BN1 - 1) / BN1);
        qwen_nvfp4_gemm_tile_kernel<BM1, BN1, BK><<<dim3(1, gy), BM1 * BN1, 0, stream>>>(
            x, packed, scales, global_scale, y, seq, K, N, acc);
        return cudaGetLastError();
    }
    constexpr int BM = 16, BN = 16;
    const int gx = (int)((seq + BM - 1) / BM);
    const int gy = (int)((N  + BN - 1) / BN);
    qwen_nvfp4_gemm_tile_kernel<BM, BN, BK><<<dim3(gx, gy), BM * BN, 0, stream>>>(
        x, packed, scales, global_scale, y, seq, K, N, acc);
```

- [ ] **Step 3: Verify** — build `test_qwen_nvfp4_tile_gemm`, run it (must stay green = bit-exact); `ctest -R qwen -C Debug` full green; run `test_blackwell_perf_qwen.exe`, record tok/s/ttft delta in registry F-05.

- [ ] **Step 4: Commit** `fix: seq==1 nvfp4 gemm tile specialization removing 15/16 dead tile rows`

---

### Task 2: Vectorized elementwise residuals (Qwen)

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (`qwen_add_kernel`, ~line 497)
- Test: existing suites (elementwise, order-free per element)

- [ ] **Step 1:** Add a `__half2`-vectorized body processing pairs when pointers are 4-byte aligned and `n` even (guard scalar tail):

```cpp
__global__ void qwen_add_kernel(const __half *__restrict src,
                                __half *__restrict dst, int64_t n)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pairs = n >> 1;
    if ((idx < pairs) && (((uintptr_t)src | (uintptr_t)dst) % 4 == 0)) {
        const __half2 s = reinterpret_cast<const __half2 *>(src)[idx];
        __half2 d = reinterpret_cast<__half2 *>(dst)[idx];
        d = __hadd2(d, s);
        reinterpret_cast<__half2 *>(dst)[idx] = d;
        return;
    }
    const int64_t i = idx + pairs;   // scalar tail element
    if (i < n) dst[i] = __hadd(dst[i], src[i]);
}
```
Keep the launch site grid math `(n+threads-1)/threads` — now sized for pair count; adjust to `((n>>1)+threads-1)/threads`.

NOTE: verify `__hadd2` matches the scalar `__hadd(dst,src)` semantics used today (read the current body first; if it converts through float, mirror that exactly — bit-exactness gate decides).

- [ ] **Step 2:** Build + `ctest -R qwen` green + perf run, record delta.
- [ ] **Step 3:** Commit `fix: vectorize qwen residual add with half2`

---

### Task 3: DeepSeek single-block kernel block-size audit

**Files:**
- Modify (only where measurably better): `libs/lsxcommon/src/deepseek_inference_attn.cu` lines 628, 653, 690 (`<<<1,256>>>` small-row kernels)
- Test: existing `ctest -R deepseek` suite

- [ ] **Step 1:** For each `<<<1,256>>>` site, check the row width it covers vs loop structure; raise block to 512/1024 ONLY where the kernel loops `hidden` internally (more threads per row = fewer iterations, same serial order per element — bit-exact). Rebuild + deepseek suite green.
- [ ] **Step 2:** Run `test_blackwell_perf_deepseek.exe`; keep changes only where ttft/total improve; revert regressions.
- [ ] **Step 3:** Commit `fix: tune deepseek single-block decode kernels for sm_120` (or registry-only note if no site improves).

---

### Task 4: Occupancy evidence report (F-05 documentation)

**Files:**
- Create: `tests/test_blackwell_occupancy_report.cpp` + CMake registration (mirror perf-harness blocks)

- [ ] **Step 1:** Reporting test querying `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the hot kernels (qwen tile GEMM both variants, add/swiglu, deepseek attn/moe main kernels) and logging blocks/SM + theoretical warps/SM vs the 48 cap via SPDLOG_INFO. Asserts nothing except device presence.
- [ ] **Step 2:** Transcribe numbers into findings registry under F-05.
- [ ] **Step 3:** Commit `test: add sm_120 occupancy report for hot kernels`

---

### Task 5: Wrap-up

- [ ] Full `ctest -R "glm|deepseek|qwen|blackwell"` green.
- [ ] Registry: F-05 status -> fixed/partially-fixed with before/after table; note F-06 deferral reason.
- [ ] Commit `docs: record phase-2 measurements`.

## Self-review notes

- Spec coverage: Phase-2 scope from design doc (occupancy/launch tuning, bit-exact) mapped to Tasks 1-4; F-06 explicitly deferred (no GLM weights -> cannot prove bit-exactness of fmad narrowing on this machine).
- Type consistency: dispatch edit stays inside `RunQwenNvfp4Gemm(stream, x, packed, scales, global_scale, y, seq, K, N, acc)` signature verified at qwen_inference.cu:1126-1147; tile kernel template constraints verified at :418-475 (`BM*BN==256`, BK chunk loop, FP16 smem rounding).
