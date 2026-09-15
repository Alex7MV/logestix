# Numeric Conformance Audit (R3 vs current code)

**Objective:** track R3 conformance of the engine, measure the cost of the
contract's requirements, and inventory C-exact vs C-fast paths. Measured
2026-09-15 on RTX 5090, AOCC build `linux-release-znver4-aocc`.

> Status: **all conformance items are done** — deterministic GEMM paths,
> workspace/allocation invariance, device-independent numeric knobs and
> calibration, no library algorithm heuristics (B1: all cuBLAS GEMM replaced by
> `lsx_gemv`), and the cross-model hash gate. Reflects the code at the B1
> completion commit.

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

## Conformance status (done)

| Requirement (contract) | Status |
|---|---|
| §2 deterministic GEMM paths / §3.6 workspace invariance | **DONE**: `Core::Initialize` pins `CUBLAS_WORKSPACE_CONFIG=:4096:8`; every handle sets `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` (model_base / gemma / deepseek). |
| §1.1.4 no device-conditional numeric path | **DONE**: `HeuristicKnobs` numeric knobs (`gemv_large_default`, `gemv_max_mib`, `splitk_grid_min`) pinned to canonical values; `test_gpu_perf` asserts invariance. |
| §1.1.4 no per-GPU numeric calibration | **DONE**: `EnsureCalibrated` no longer overrides `gemv_max_mib`; per-GPU numeric tuning is C-fast-only opt-in. |
| §1.1.1 no library algorithm heuristics | **DONE**: all 21 `cublasGemm*` sites replaced by the model-neutral, device-independent `lsx_gemv` (`RunLxGemmVec` N-major, `RunLxGemmVecWT` K-major; batched cases run per-row/per-batch loops). Models: Kimi, Gemma, DeepSeek-V4-Flash, DeepSeek-V4.1-Flash. |
| Cross-model verification hook | **DONE**: `RunParisSelfCheck` emits `logits_hash=<fnv1a64>` and gates on `LSX_SELFCHECK_EXPECT_HASH` (mismatch -> `selfcheck_passed=false`). |
| Perf cost of conformance | **~zero**: forcing a deterministic workspace changed t/s by <1%; the own-GEMV B1 replacement left the decode t/s unchanged. |

## Measurement notes

**Workspace experiment (EOS-length, `--max-gen 512`):** default 21.5–21.7 t/s
[83 tok]; `:4096:8` 21.6 [83]; `:16:8` 21.4 [105]. Pinning the workspace made the
output independent of the environment (external `:16:8` then yielded 83).

**Fixed-length measurement.** The old `t/s` used the post-EOS length over the
full wall, so a numeric change that moved EOS shifted the number without
changing the wall. The perf bench is now **fixed-length**
(`ModelRequest.ignore_eos` + `kIgnoreEosTokenId`; drivers reject `eos_id<0`):
exactly `max_gen` steps decode. Example: qwen3.8-flash-next 16.75 @1370 ->
26.53 @2048.

## Recorded baseline (B1 path, AOCC build, 2048 tokens)

`tests/perf/perf_baselines.json` records both numeric modes per model
(`mode: "perf"` = kAuto, `mode: "r3"` = kOff); hashes from
`lsxbenchmark --selfcheck [--bit-exact]`:

| model | perf t/s | r3 t/s | perf hash | r3 hash |
|---|---|---|---|---|
| glm-5.3-flash | 33.93 | 18.48 | 358d6dcbc80aac34 | b4df9a1e73b05b42 |
| qwen3.6-35b-a3b | 142.73 | 13.32 | 5209e2295509aa7d | 4759fe1466ffcf89 |
| gemma-4-26b-a4b-it | 102.31 | 104.61 | c06c0f18ae3cabda | f343ce41dd246552 |
| qwen3.8-flash-next | 26.53 | 6.40 | 139814718cb71a0c | 7dd22315feb10d6f |
| deepseek-v4-flash | 18.16 | 8.37 | 36ed938bf67c6eba | 15498f4a82ef1b21 |
| deepseek-v4.1-flash | — | — | cef7be674130a967 | cc41f19daf3af712 |
| kimi-k2.7-code | 5.10 | — | dcf5782ad6a10f1f | 6bd162f60b49405c |

B1 moved the hashes only for the models whose selfcheck path used the replaced
cuBLAS calls (Kimi, DeepSeek-V4, DeepSeek-V4.1); GLM/Qwen/Gemma were unchanged
(their paths already used own kernels). The golden token is preserved on all
four cuBLAS-affected models (selfcheck Paris). Slower models (kimi r3,
deepseek-v4.1) are not in the "fast" measurement set.

## Open point (contract interpretation)

The default (`kAuto`) `final_logits` hash differs from `kOff` for every model:
the default is a distinct numeric path (TC fast-math), golden-token-correct but
not bit-identical to `kOff`. Per the contract's §2 wording this classifies the
default as C-fast; per §3.4 the FSI default must resolve to C-exact. Operator
decision: make the default the `kOff` path, or define R3 as determinism + golden
tokens.
