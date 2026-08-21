# Qwen Fused Decoder+Prefill MLP Layer — Implementation Plan (accumulate down-GEMM)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the MLP layer from 6 launches to 5 (decoder seq=1 and prefill seq>1) by absorbing the residual-add into the down GEMM via a new `accumulate` mode on `RunQwenNvfp4Gemm`, removing the separate `qwen_add_kernel` launch and the `d_mlp_out` round-trip.

**Architecture:** `RunQwenNvfp4Gemm` (and both `qwen_nvfp4_gemm_kernel`/`qwen_nvfp4_gemm_tile_kernel`) gain a trailing `bool acc=false` param. When true, the per-thread accumulator seeds from `y[idx]` (so `y = y + Σ`) instead of `0`, keeping `y` as both source and destination. `RunQwenMLP` then runs the down GEMM with `acc=true` writing directly into `d_act` (the residual stream), and the standalone `qwen_add_kernel` residual is removed. `acc=false` (default) is bit-identical to today.

**Tech Stack:** C++20, CUDA, NVFP4 codec, Catch2. Numeric: FP32; `acc` path adds one FP32 operand at the start (same order as the separate residual-add, within tolerance).

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — `RunQwenNvfp4Gemm` gains `bool acc = false`.
- Modify: `libs/lsxcommon/src/qwen_inference.cu` — thread `acc` through `RunQwenNvfp4Gemm` → both kernels; `RunQwenMLP` down-GEMM uses `acc=true` into `d_act`; drop the residual-add in `RunQwenMLP` and in the generator's two MLP invocations.
- Create: `tests/test_qwen_fused_mlp.cu` — accumulate unit + full-MLP split-vs-fused parity (seq=1, seq=2).
- Modify: `tests/CMakeLists.txt` — add `.cu` to `test_qwen_cu` OBJECT + foreach list.
- No codec/other-CMake change.

## Grounding (read first)

- `qwen_nvfp4_gemm_kernel` (`qwen_inference.cu:393-416`): legacy per-element, `float acc=0; for k {...}; y[idx]=half(acc)`. Add `bool acc_in`: seed `float a = acc_in ? __half2float(y[idx]) : 0.0f;`.
- `qwen_nvfp4_gemm_tile_kernel` (`:423-472`): tile, `float acc=0.0f`. Each thread owns output `(t,n)`. Seed `float a = acc_in ? __half2float(y[(int64_t)t*N+n]) : 0.0f;` (only meaningful for in-bounds thread). Store back to `y`.
- `RunQwenNvfp4Gemm` (`:917-938`) — dispatches to one of the two kernels; add `bool acc = false` trailing param and pass to both at the launch sites.
- `LaunchNvfp4Gemm` (`:940-945`) — delegate; pass `false`.
- `RunQwenMLP` (`qwen_inference.cu:705-728`): `d_gate = d_scratch; d_up = d_scratch + seq*I;` gate/up GEMMs, swiglu `(d_gate,d_up,d_gate)`, down GEMM `(d_gate → d_out)`. Change down call to `RunQwenNvfp4Gemm(..., d_act?... no — RunQwenMLP signature is `(stream, lw, seq, d_in, d_out, d_scratch)` and `d_out` is the MLP output buffer, and the generator does `d_act += d_out` separately. Two options: (a) change `RunQwenMLP` to take the residual buffer and dst-add into it; (b) keep `RunQwenMLP` writing `d_out` and change the generator to pass `d_act` as `d_out` with the down-GEMM accumulate reading the pre-residual `d_out`? — No: `d_out` is written fresh by down-GEMM then the generator adds. To use `acc`, the down-GEMM must write into a buffer that ALREADY holds the residual (d_act), and NOT separately add.
- Generator MLP sites (two: transient ~:1538-1570 and persistent ~:1668-1685): 
  ```
  RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin, seq, H, eps);
  RunQwenMLP(stream, lw, seq, d_postin, d_mlp_out, d_gate);
  qwen_add_kernel(d_mlp_out, d_act, seq*H);
  ```
  To use acc: change `RunQwenMLP` so its down-GEMM writes `d_act` with `acc=true` (seeding from d_act's residual) and returns that as the MLP output; then drop the `qwen_add_kernel`. But `RunQwenMLP` is a public generic entry used to produce `d_out` (the MLP result for callers that then residual-add). The cleanest: **keep `RunQwenMLP` semantics unchanged** (writes `d_out` = pure MLP, no residual) for other callers, and in the **generator** use a new internal path: post-norm → gate/up/swiglu → down-GEMM `acc=true` into `d_act`. So the generator should NOT call `RunQwenMLP` for these layers but run the same 5 steps with the down accumulate into d_act. Simpler and lower-risk: add a dedicated `RunQwenMLPAccumulate(stream, lw, seq, d_in, d_resid, d_scratch)` that runs norm? — no, norm is outside. 

  Decision (final): keep `RunQwenMLP` public semantics (`d_out` = pure MLP). Add `bool acc` to `RunQwenNvfp4Gemm`. In the **generator**, replace the MLP tail (post-norm + RunQwenMLP + add) with: post-norm → gate/up GEMMs → swiglu → down-GEMM `acc=true` into `d_act` — i.e., inline the 4 GEMM/swiglu calls (reusing `RunQwenNvfp4Gemm`) instead of the opaque `RunQwenMLP`, so the residual-add is the dst-add. This avoids changing `RunQwenMLP`'s contract and touches only the two generator sites. A shared helper `RunQwenFusedMLP(stream, lw, top, seq, d_in, d_resid, eps, scratch)` (host, external) encapsulates it; the generator calls it.

- Test pattern: add `.cu` to `test_qwen_cu` OBJECT + foreach list; Arrow includes already present.

---

## Task 1: `acc` flag in the NVFP4 GEMM (TDD unit)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`
- Create: `tests/test_qwen_fused_mlp.cu`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Register the test**

Add `test_qwen_fused_mlp.cu` to the `test_qwen_cu` OBJECT lib and the `foreach(qt ...)` list.

- [ ] **Step 2: Write the failing accumulate-unit test**

Create `tests/test_qwen_fused_mlp.cu` with a `FillNvfp4` helper (reuse pattern) and this unit case:

```cpp
TEST_CASE("RunQwenNvfp4Gemm accumulate adds into the output buffer", "[qwen][gemm][acc]")
{
    constexpr int seq = 2, K = 64, N = 128;
    std::vector<__half> hx(seq*K), hy(seq*N), hsrc(seq*N);
    std::vector<uint8_t> hp((size_t)K*N/2), hs((size_t)K*N/16);
    for (int i = 0; i < seq*K; ++i) hx[i] = __float2half((float)((i%7)-3));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i*13)&0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128+(i%32));
    for (int i = 0; i < seq*N; ++i) hsrc[i] = __float2half(0.5f + 0.01f*(float)(i%9));
    const float gs = 0.5f;

    __half *dx, *dout, *dacc; uint8_t *dp, *ds;
    cudaMalloc(&dx, seq*K*2);
    cudaMalloc(&dout, seq*N*2); cudaMalloc(&dacc, seq*N*2);
    cudaMalloc(&dp, hp.size()); cudaMalloc(&ds, hs.size());
    cudaMemcpy(dx, hx.data(), seq*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
        nullptr, dx, dp, ds, gs, dout, seq, K, N) == cudaSuccess);
    cudaMemcpy(dacc, hsrc.data(), seq*N*2, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
        nullptr, dx, dp, ds, gs, dacc, seq, K, N, /*acc=*/true) == cudaSuccess);

    std::vector<__half> ho(seq*N), ha(seq*N);
    cudaMemcpy(ho.data(), dout, seq*N*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), dacc, seq*N*2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq*N; ++i) {
        const float sum = __half2float(hsrc[i]) + __half2float(ho[i]);
        CHECK(std::abs(__half2float(ha[i]) - __half2float(__float2half(sum))) < 1e-3f);
    }
    for (auto p : {dx, dp, ds, dout, dacc}) cudaFree(p);
}
```

The test is RED: `RunQwenNvfp4Gemm` has no 6-th (`acc`) param.

- [ ] **Step 3: Implement the `acc` flag**

Header: add `bool acc = false` to `RunQwenNvfp4Gemm`.

```cpp
cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc = false);
```

`RunQwenNvfp4Gemm` body: pass `acc` at both launches. Legacy kernel `qwen_nvfp4_gemm_kernel` — add `bool acc_in` param and seed:

```cpp
    float acc0 = acc_in ? __half2float(y[idx]) : 0.0f;
    float acc = acc0;   // (rename local)
    ... loop ...
    y[idx] = __float2half(acc);
```

Tile kernel `qwen_nvfp4_gemm_tile_kernel` — add `bool acc_in`; before the K-loop:

```cpp
    float acc = acc_in ? __half2float(y[in_bounds ? ((int64_t)t*N+n) : 0]) : 0.0f;
```
(For an out-of-bounds thread, `acc` is unused since store is `if (in_bounds)`; reading `y[0]` for OOB is safe but wasteful — guard with `in_bounds ? ... : 0.0f`.)

Both launch sites in `RunQwenNvfp4Gemm` pass `acc`.

`LaunchNvfp4Gemm` passes `false` (keep its semantics).

- [ ] **Step 4: Build + run the unit**

`cmake --build build --config Debug --target test_qwen_fused_mlp` then run `tests/test_qwen_fused_mlp.exe` → PASS (accumulate unit + the object lib's other Qwen cases).

- [ ] **Step 5: Commit**

`git add libs/lsxcommon/include/lsxcommon/qwen_inference.h libs/lsxcommon/src/qwen_inference.cu tests/test_qwen_fused_mlp.cu tests/CMakeLists.txt`
`git commit -m "feat: NVFP4 GEMM accumulate mode (acc-into-output)"`

---

## Task 2: Fused MLP via acc-down; wire generator

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`

- [ ] **Step 1: Add a host `RunQwenFusedMLP` helper** (external scope, declared in header):

```cpp
// Fused MLP layer (decoder seq=1 and prefill seq>1): post-RMSNorm -> gate/up
// NVFP4 GEMMs -> swiglu -> down GEMM accumulated INTO d_resid (absorbs the
// residual-add). d_resid is BOTH the residual stream and the MLP output.
// scratch: [2*seq*I] halves for gate/up (contiguous).
cudaError_t RunQwenFusedMLP(
    cudaStream_t stream, const QwenLayerWeights &lw,
    const QwenTopology &top, int seq,
    const __half *d_in, __half *d_resid, __half *d_scratch)
{
    if (!lw.loaded || seq <= 0 || !d_in || !d_resid || !d_scratch)
        return cudaErrorInvalidValue;
    const int H = lw.mlp_gate.cols;
    const int I = lw.mlp_gate.rows;
    if (H <= 0 || I <= 0) return cudaErrorInvalidValue;
    if (lw.mlp_gate.d_packed && lw.mlp_up.d_packed && lw.mlp_down.d_packed &&
        lw.mlp_gate.d_scales && lw.mlp_up.d_scales && lw.mlp_down.d_scales) {
        // no separate post-norm buffer: apply post-norm into d_scratch head? No —
        // d_in is ALREADY the post-normed stream from the caller (d_postin). The
        // generator calls input-norm separately; pass d_postin as d_in. So this
        // helper does gate/up/swiglu/down-acc only.
        __half *d_gate = d_scratch;
        __half *d_up   = d_scratch + (int64_t)seq * I;
        cudaError_t e;
        e = RunQwenNvfp4Gemm(stream, d_in, lw.mlp_gate.d_packed, lw.mlp_gate.d_scales,
                             lw.mlp_gate.global_scale, d_gate, seq, H, I); if (e) return e;
        e = RunQwenNvfp4Gemm(stream, d_in, lw.mlp_up.d_packed, lw.mlp_up.d_scales,
                             lw.mlp_up.global_scale, d_up, seq, H, I); if (e) return e;
        {
            const int64_t n = (int64_t)seq * I;
            qwen_swiglu_kernel<<<(int)((n+255)/256), 256, 0, stream>>>(d_gate, d_up, d_gate, n);
            e = cudaGetLastError(); if (e) return e;
        }
        // down GEMM accumulates INTO d_resid (which holds the pre-MLP residual).
        return RunQwenNvfp4Gemm(stream, d_gate, lw.mlp_down.d_packed, lw.mlp_down.d_scales,
                                lw.mlp_down.global_scale, d_resid, seq, I, H, /*acc=*/true);
    }
    return cudaErrorInvalidValue;
}
```

(`qwen_swiglu_kernel` is visible at this scope; `RunQwenNvfp4Gemm` public. Declare `RunQwenFusedMLP` in the header.)

- [ ] **Step 2: Wire the generator's two MLP sites**

Transient branch (~:1566) and persistent branch (~:1678). Replace:

```cpp
                e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(...);
                // MLP
                e = RunQwenMLP(stream, lw, seq, d_postin, d_mlp_out, d_gate);
                if (e != cudaSuccess) return fail(...);
                // residual: d_act = d_act + d_mlp_out
                { qwen_add_kernel<<<...>>>(d_mlp_out, d_act, n); ... }
```
with:
```cpp
                e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(...);
                // fused MLP: down GEMM accumulates into d_act (no residual-add)
                e = RunQwenFusedMLP(stream, lw, top, seq, d_postin, d_act, d_gate);
                if (e != cudaSuccess) return fail(...);
```
The `d_mlp_out` buffer becomes unused in that path (keep allocated, harmless). Both sites updated identically.

- [ ] **Step 3: Build + run**

`cmake --build build --config Debug --target test_qwen_model test_qwen_fused_mlp` → both build. Run `test_qwen_model` (13/1) and `test_qwen_fused_mlp` (unit + others) → green. Prefill path in `test_qwen_model` uses seq>1 initially, decoder uses seq==1 — both now go through `RunQwenFusedMLP`.

- [ ] **Step 4: Add full-MLP parity test**

Append to `test_qwen_fused_mlp.cu`: build a tiny MLP via `FillNvfp4` (H=16, I=32), run (a) split `post-norm + gate/up + swiglu + down + qwen_add` and (b) `RunQwenFusedMLP`; assert `1e-2` tolerance, for seq=1 and seq=2. Uses the qwen_nvfp4 tile + public RunQwenNvfp4Gemm for the split reference.

- [ ] **Step 5: Commit**

`git add libs/lsxcommon/include/lsxcommon/qwen_inference.h libs/lsxcommon/src/qwen_inference.cu tests/test_qwen_fused_mlp.cu`
`git commit -m "feat: fused MLP layer (acc-down absorbs residual) wired into generator"`

---

## Task 3: Full verification

- [ ] **Step 1: Full build** — `cmake --build build --config Debug` clean.
- [ ] **Step 2: Run affected suite** — `ctest --preset windows-debug --tests-regex "qwen_mlp|qwen_nvfp4_tile_gemm|qwen_nvfp4_dense|qwen_fused_mlp|qwen_model|qwen_fused_decode_attn|qwen_full_attn|qwen_linear_attn|qwen_kv_parity|qwen_norm|qwen_rotary"` → all pass (kv_cache SKIP).
- [ ] **Step 3: Commit** (only if files changed) — `git add . && git commit -m "test: verify fused MLP across Qwen suites"`.

---

## Self-Review

- Spec §3.1 acc mode → Task 1 (flag in both kernels + RunQwenNvfp4Gemm). ✓
- Spec §3.2 composition → Task 2 (RunQwenFusedMLP with acc-down, both generator sites). ✓
- Spec §4 tests → Task 1 (unit) + Task 2 (full parity seq1/2). ✓
- Spec §5 files → matches. ✓
- `acc=false` default preserves bit-identical tile-GEMM; `test_qwen_nvfp4_tile_gemm`/`dense` unaffected (Task 3 verifies). ✓
- Post-norm stays separate (spec §3.3, YAGNI) — `RunQwenFusedMLP` takes `d_in` as the already-normed stream. ✓
- Type consistency: `RunQwenNvfp4Gemm` header `.h` == `.cu` (7 params + bool); `RunQwenFusedMLP` declared+defined; kernels `qwen_nvfp4_gemm_kernel`/`qwen_nvfp4_gemm_tile_kernel` both get `bool acc_in`. `LaunchNvfp4Gemm` stays `false`.
- No placeholders: unit test cleaned (no `ds2`/`dy` leftover); kernels seed-and-store complete.

The plan is complete. Save to `docs/superpowers/plans/2026-08-20-qwen-fused-mlp.md`.