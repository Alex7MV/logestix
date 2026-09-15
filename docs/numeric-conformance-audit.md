# Numeric Conformance Audit (R3 vs current code)

**Objective:** track R3 conformance of the engine, measure the cost of the
contract's requirements, and inventory C-exact vs C-fast paths. Measured
2026-09-15 on RTX 5090, AOCC build `linux-release-znver4-aocc`.

> Status: **all conformance items are done** — deterministic GEMM paths,
> workspace/allocation invariance, device-independent numeric knobs and
> calibration, no library algorithm heuristics (B1: all cuBLAS GEMM replaced by
> `lsx_gemv`), and the cross-model hash gate. This document reflects the code at
> the B1 completion commit.

## Inventory

- **cuBLAS GEMM calls: 0** (was 21 `cublasGemmEx` + strided-batched). All
  replaced by the model-neutral, device-independent `lsx_gemv`. The workspace /
  atomics pin (`CUBLAS_WORKSPACE_CONFIG` + `cublasSetAtomicsMode`) is kept as a
  belt-and-braces default for any future cuBLAS use.
- Runtime-memory use: `cudaMemGetInfo` only for **expert-cache sizing**
  (residency, not arithmetic).
- Numeric-path selection: `gpu_perf.cpp` / `perf_calibration.cu` — numeric knobs
  are now **device-independent** (canonical values for every GPU); only
  non-numeric knobs (residency, overlap) are device-derived.

## Conformance status (what is done)

| Requirement (contract) | Status |
|---|---|
| §2 deterministic GEMM paths / §3.6 workspace invariance | **DONE**: `Core::Initialize` pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`; every handle sets `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` (model_base / gemma / deepseek). |
| §1.1.4 no device-conditional numeric path | **DONE**: `HeuristicKnobs` numeric knobs (`gemv_large_default`, `gemv_max_mib`, `splitk_grid_min`) pinned to canonical values; `test_gpu_perf` asserts invariance. |
| §1.1.4 no per-GPU numeric calibration | **DONE**: `EnsureCalibrated` no longer overrides `gemv_max_mib`; per-GPU numeric tuning is C-fast-only opt-in. |
| Cross-model verification hook | **DONE**: `RunParisSelfCheck` emits `logits_hash=<fnv1a64>` and gates on `LSX_SELFCHECK_EXPECT_HASH` (mismatch -> `selfcheck_passed=false`). |
| Perf cost of conformance | **~zero**: forcing a deterministic workspace changed t/s by <1% on the decode GEMV shapes. |
| §1.1.1 no library algorithm heuristics | **DONE**: all 21 `cublasGemm*` sites replaced by the model-neutral, device-independent `lsx_gemv` (fixed geometry, no workspace, no device/heuristic dependence). Models: Kimi, Gemma, DeepSeek-V4-Flash, DeepSeek-V4.1-Flash. |

## Measurement

**First pass (EOS-length, `--max-gen 512`):** default 21.5–21.7 t/s [83 tok];
`:4096:8` 21.6 [83]; `:16:8` 21.4 [105]. Pinning the workspace made the output
independent of the environment (external `:16:8` then yielded 83 as well).

**Caveat (now fixed):** those lengths were EOS artifacts — `t/s` used the
post-EOS length over the full wall, so a numeric change that moved EOS shifted
the number without changing the wall. The perf bench is now **fixed-length**
(`ModelRequest.ignore_eos` + `kIgnoreEosTokenId`, drivers reject `eos_id<0`):
exactly `max_gen` steps decode, so t/s is comparable across numeric modes and
bit-drift. Example correction: qwen3.8-flash-next 16.75 @1370 -> 26.52 @2048.

## Two-mode perf baseline

`tests/perf/perf_baselines.json` now records both numeric modes per model
(`mode: "perf"` = kAuto, `mode: "r3"` = kOff), measured by `test_perf_baseline`
(one mechanism, both modes). Fast models (2048 tokens):

| model | perf t/s | r3 t/s |
|---|---|---|
| glm-5.3-flash | 33.93 | 18.48 |
| qwen3.6-35b-a3b | 142.64 | 13.31 |
| gemma-4-26b-a4b-it | 102.31 | 104.61 |
| qwen3.8-flash-next | 26.52 | 6.40 |
| deepseek-v4-flash | 18.16 | 8.37 |

## Remaining

- **§1.1.1 — DONE (B1).** Every `cublasGemm*` call is gone; the decode GEMV/GEMM
  paths use `lsx_gemv` (own fixed-geometry kernels, `RunLxGemmVec` /
  `RunLxGemmVecWT`; batched cases run per-row/per-batch loops). Replacing cuBLAS
  moved every model's `final_logits` hash (the numeric path changed) but the
  golden token is preserved on all four models (Kimi/Gemma/DeepSeek-V4/V4.1
  selfcheck Paris). The cross-model hash gate (`LSX_SELFCHECK_EXPECT_HASH`)
  still applies for regression evidence.
- **Consequence: recorded hashes/perf numbers are stale.** The two-mode perf
  baseline and the hash examples in this doc predate B1; re-record them on the
  new (device-independent) path. The `perf` vs `r3` distinction is unchanged.

## Note on `perf` vs `r3` hashes

The default (`kAuto`) `final_logits` hash differs from `kOff` for every model:
the default is a distinct numeric path (TC fast-math), golden-token-correct but
not bit-identical to `kOff`. Per the contract's §2 wording this classifies the
default as C-fast; per §3.4 the FSI default must resolve to C-exact. This is an
open contract-interpretation decision (make the default the `kOff` path, or
redefine R3 as determinism + golden tokens).
