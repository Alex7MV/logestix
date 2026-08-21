# Qwen Fused Decoder+Prefill MLP Layer — Design Spec (Direction A step 4)

**Date:** 2026-08-20
**Status:** Approved
**Scope:** `libs/lsxcommon` — Qwen MLP layer, both **decoder (seq=1)** and **prefill (seq>1)** paths, in `libs/lsxcommon/src/qwen_inference.cu`. Direction **A** of the Blackwell optimization strategy.
**Numeric contract:** FP32 accumulate; swiglu/residual/dst-add are elemental ops so their ordering does not change the result beyond FP32 rounding; the down-GEMM `accumulate` path keeps the same GEMM numerics and only changes the destination semantics (`out = src + Σ ...` vs `out = Σ ...`). Tolerance vs the split path.

## 1. Current state (what it fuses)

`RunQwenMLP` (`qwen_inference.cu:705-728`) plus its callers produce, per transformer layer:

```
post-RMSNorm(1) → gate GEMM(2) → up GEMM(3) → swiglu(4) → down GEMM(5) → residual-add(6)
```
= **6 launches** on the stream (both decoder seq=1 and prefill seq>1), with HBM round-trips through `d_postin`, `d_gate`, `d_up`, `d_mlp_out`, and the residual-add kernel.

- `post-RMSNorm`: `RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin, seq, H, eps)`.
- gate/up: `RunQwenNvfp4Gemm(..., d_in, seq, H, I)` for `mlp_gate`/`mlp_up` (H→I each).
- swiglu: `qwen_swiglu_kernel(d_gate, d_up, d_gate, seq*I)` (writes d_gate in place).
- down: `RunQwenNvfp4Gemm(..., d_gate, seq, I, H)` for `mlp_down` → `d_mlp_out`.
- residual: `qwen_add_kernel(d_mlp_out, d_act, seq*H)`.

## 2. Goal / non-goal

**Goal:** reduce 6 launches → **5** (or fewer where graph-legal) by fusing the surrounding elemental ops around the already-fused tile-GEMM, for BOTH seq=1 and seq>1, while keeping the tile-GEMM as the compute engine:

- **`RunQwenNvfp4Gemm` gains an optional `accumulate`/`dst-add` mode** on `qwen_nvfp4_gemm_tile_kernel` (and the legacy per-element fallback): the kernel computes `y = src + Σ_k ...` instead of `y = Σ_k ...`. Used for the **down GEMM** so the residual-add is absorbed: `d_act = d_act + down(swiglu(...))` becomes a single `DownGemmAccumulate(d_act, ...)` that adds into `d_act` directly, eliminating the separate `qwen_add_kernel` residual launch and the `d_mlp_out` round-trip.
- The `post-RMSNorm` stays a separate fast kernel (elemental, cheap); a dedicated fused kernel is NOT justified because it only saves 1 launch at elemental cost.

**Net:** `postnorm(1) → gate(2) → up(3) → swiglu(4) → down_gemm+residual(5)` = **5 launches** (was 6), residual launch + `d_mlp_out` round-trip removed.

**Non-goal:** no change to the swiglu math, RMSNorm, or the gate/up GEMMs. No single-`__global__` that fuses swiglu+down (they are different computational classes — elemental vs tile-GEMM — and forcing them into one kernel would require rewriting the GEMM). No change to the tile-GEMM's default (non-accumulate) path — it must remain bit-identical to today.

## 3. Design

### 3.1 `accumulate` mode in the NVFP4 GEMM

`RunQwenNvfp4Gemm` signature gains a trailing `bool accumulate = false`. When `true`, the launched kernel initializes its output register to the **source** value (`out_row = src_row`) instead of `0`, then adds the GEMM products, and stores `out = src + Σ`. Applied at the host to:
- `RunQwenNvfp4Gemm(stream, d_attn/d_gate, w, ... , dst, seq, K, N, accumulate)`.

For the tile kernel: the accumulator is per-thread; a thread owning output element `(t,n)` reads `src[t*N+n]` once at the start (`out_row_value`) and adds products. For the legacy per-element fallback: same — read `src[idx]`, start acc from it. **Default `accumulate=false` preserves the exact current behavior** (bit-identical), so the validation of the tile-GEMM (step at `test_qwen_nvfp4_tile_gemm`) is unaffected.

### 3.2 MLP composition with dst-add down

Replacement of the current MLP tail in `RunQwenMLP` and its callers (both decoder branches):

```
// unchanged: post-RMSNorm -> gate -> up -> swiglu
e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin, seq, H, eps);
e = RunQwenNvfp4Gemm(stream, d_postin, lw.mlp_gate..., d_gate, seq, H, I);
e = RunQwenNvfp4Gemm(stream, d_postin, lw.mlp_up...,   d_up,   seq, H, I);
e = qwen_swiglu_kernel(d_gate, d_up, d_gate, seq*I);
// down GEMM accumulates INTO d_act (no separate residual launch)
e = RunQwenNvfp4Gemm(stream, d_gate, lw.mlp_down..., d_act, seq, I, H, /*accumulate=*/true);
```
`d_act` is the residual stream directly (it already holds the pre-MLP residual). The old `qwen_add_kernel` and `d_mlp_out` become unused.

### 3.3 Why not fuse post-RMSNorm into a kernel

Post-RMSNorm is a cheap elemental op; fusing it into the gate/up GEMM would require the GEMM to accept a raw `d_act` + norm weights and apply norm inside — a larger, riskier change for 1 launch. Deferred (YAGNI).

## 4. Testing / acceptance

- **New** `tests/test_qwen_fused_mlp.cu`:
  - Build a tiny MLP layer (`mlp_gate/up/down` NVFP4 weights + `d_post_norm`, `d_input_norm`), seq=1 and seq=2 (decoder + prefill).
  - **split reference**: `post-norm + gate + up + swiglu + down + residual` (the existing kernels).
  - **fused**: the new `accumulate`-down composition.
  - `CHECK(|fused - split| < 1e-3 * |ref| + 1e-6)` per element.
  - Also a dedicated `accumulate` GEMM unit check: `RunQwenNvfp4Gemm(accumulate=true, src)` vs `src + RunQwenNvfp4Gemm(accumulate=false)` within `1e-2`.
- **Existing**: `test_qwen_nvfp4_tile_gemm` (default accumulate=false path unchanged, must stay green), `test_qwen_model` (both seq paths, must stay green), `test_qwen_mlp`.
- Real `test_qwen_selfcheck` SKIP without checkpoint.

## 5. Files

- Modify `libs/lsxcommon/src/qwen_inference.cu` — add `accumulate` param to `RunQwenNvfp4Gemm` (and thread it to `qwen_nvfp4_gemm_tile_kernel` + `qwen_nvfp4_gemm_kernel`); update the MLP tail + both decoder callers to use the dst-add down; `RunQwenMLP` uses it internally.
- Modify `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare the `accumulate` default arg.
- Create `tests/test_qwen_fused_mlp.cu`; add to `test_qwen_cu` OBJECT + foreach list (`tests/CMakeLists.txt`).
- No codec/other-CMake change.

## 6. Open decisions (resolve in plan)

- Whether to expose a separate public `RunQwenMlpFused(...)` wrapper or modify `RunQwenMLP` in place. Decide in plan: keep `RunQwenMLP` as the public entry (all callers unchanged) and implement the fused tail inside it — no new public symbol needed; tests drive `RunQwenMLP` directly with the split-vs-fused comparison.
- The `accumulate` flag plumbing: a bool on `RunQwenNvfp4Gemm` threaded to both kernels; the tile kernel's per-thread acc init from `src`.

## Self-review

- §1 quantifies 6 launches; §2 goal (→5, dst-add residual removed); §3.1 accumulate mode (default=false bit-identical); §3.2 composition; §3.3 post-norm defer (YAGNI); §4 tests; §5 files. Consistent with the operator's "tile-GEMM + fused-окружение" and "decoder+prefill" choices.
- Numeric: swiglu/residual/dst-add elemental; `accumulate` preserves default behavior. §4 covers accumulate-unit + full-MLP parity (seq 1 & 2).
- No AI-confusion: the residual-add is absorbed by dst-add; no single-launch fusion of swiglu+down is claimed.
- Scope: one kernel-flag + MLP composition + tests — a single plan. No placeholders.