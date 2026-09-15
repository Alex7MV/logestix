# Numeric Conformance Audit (R3 vs current code)

**Objective:** measure whether the `docs/numeric-contract.md` R3 requirements
cost performance, and inventory C-exact vs C-fast paths. Measured 2026-09-15 on
RTX 5090, AOCC build `linux-release-znver4-aocc`, model DeepSeek-V4-Flash
(cuBLAS-heavy legacy path), `--max-gen 512`.

## Inventory

- **cuBLAS GEMM calls: 23**, all `CUBLAS_GEMM_DEFAULT` (heuristic, workspace-
  dependent algorithm selection), in `deepseek_moe_v41_inference.cu`,
  `deepseek_moe_inference_{attn,head,moe}.cu`, `gemma_moe_inference.cu`,
  `kimi_moe_inference.cu`. None pin the workspace (`CUBLAS_WORKSPACE_CONFIG` /
  `cublasSetWorkspace`), atomics mode, or a deterministic algorithm.
- Runtime-memory use: `cudaMemGetInfo` only for **expert-cache sizing**
  (residency, not arithmetic) in qwen_moe_exp/glm/deepseek/qwen/gemma.
- Numeric-path selection: `gpu_perf.cpp` (`ProbeGpuPerfProfile`, split-K
  threshold, `tc_fast_math_default`) and `perf_calibration.cu` are **device-keyed
  (GPU SKU)**, hence invariant across instances of the same SKU — provided the
  calibration itself does not depend on runtime free memory.

## Measurement (t/s; gen token count in brackets)

| Setting | Run 1 | Run 2 | Notes |
|---|---|---|---|
| default (no env) | 21.5 [83] | 21.7 [83], 21.7 [83] | baseline, matches recorded 21.48 [83] |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 21.6 [83] | — | deterministic workspace, same output |
| `CUBLAS_WORKSPACE_CONFIG=:16:8` | 21.4 [105] | 21.4 [105] | **diverged output** |

## Findings

1. **Performance cost of R3 conformance is ~zero** for this path: forcing a
   deterministic cuBLAS workspace changes t/s by <1% (21.6 vs 21.5–21.7), well
   inside run noise. The decode GEMMs are M×1 / N=1 (memory-bound), so the
   deterministic-algorithm constraint is not a throughput tax here.
2. **Real conformance gap:** changing the cuBLAS workspace size (`:16:8`)
   deterministically flips the greedy sequence (83 → 105 tokens). The decode
   output therefore depends on the cuBLAS workspace / memory environment, which
   **violates R3 cross-instance** as written. Root cause: at least one GEMM's
   algorithm (split-K eligibility) depends on the available workspace.
   Mitigation is cheap: pin `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or
   `cublasSetWorkspace`) + `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` for
   C-exact, which costs nothing measurable.
3. The 83↔105 sensitivity also demonstrates (again) that greedy output is
   numerically fragile — consistent with the Kimi TC finding.

## Verdict against the contract

- Current default paths are **not yet R3-conformant** (§2 "deterministic GEMM
  execution paths"; §3.6 workspace/allocation invariance) because workspace is
  not pinned.
- Bringing them into conformance is **cheap and perf-neutral** on the measured
  decode shapes: no throughput is lost, only runtime-memory-driven algorithm
  variability is removed.
- The contract does **not** break the major perf levers (spec-decode,
  fused-dequant, CUDA graph); only runtime-memory-driven numeric autotuning is
  removed.

## Fix-list (C-exact conformance)

1. **DONE (commit "pin cuBLAS workspace + disable reduce atomics").** Initialize
   cuBLAS with a pinned workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8` in
   `Core::Initialize`) + `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` on
   every handle (model_base/gemma/deepseek). Verified: external `:16:8` no
   longer flips the output (105 -> 83, matching default); DeepSeek 21.6 t/s (no
   perf loss); Kimi golden selfcheck Paris x3; fast suite 47/47.
2. **Mandatory (R3 spans 5090 + 5060 Ti, different SM counts).** cuBLAS
   heuristics are SM-count-dependent, so `CUBLAS_GEMM_DEFAULT` is not
   cross-model safe even with the workspace pinned: replace split-K-eligible
   GEMMs with an explicit deterministic algorithm (or our own fixed-geometry
   kernels) and verify bit-identity across every supported model.
3. Gate `perf_calibration`/`gpu_perf` numeric knobs to values that do not change
   the numeric path across the supported models (device-dependent *perf* knobs
   are fine; device-dependent *numerics* are not); document that residency sizing
   is exempt.
4. Add cross-model + workspace-variant selfchecks to CI (run on 5090 and
   5060 Ti, and under `:16:8`/`:4096:8`) for C-exact models.

## Cross-model conformance (R3 spans RTX 5090 + RTX 5060 Ti, both sm_120)

Operator requirement: bit-identity must hold across the supported GPU-model set,
not just one SKU. Audit of device-dependent numeric paths:

| Finding | Status |
|---|---|
| `HeuristicKnobs` (fallback for unknown GPUs, i.e. 5060 Ti) derived **numeric** knobs from device props: `gemv_large_default` (base-only), `gemv_max_mib` (bandwidth-scaled), `splitk_grid_min` (SM-count: 64 vs 96) — a device-conditional numeric path (§1.1.4). | **FIXED**: numeric knobs pinned to the canonical (5090-base) values; only non-numeric knobs (expert-cache residency, overlap `dual_stream_staging`) stay device-derived. `test_gpu_perf` updated to assert device-independence. |
| `CUBLAS_GEMM_DEFAULT` heuristic is SM-count-dependent: 5090 (170 SMs) and 5060 Ti (~36 SMs) may select different algorithms for the same shape -> different bits. Workspace/atomics pinning does NOT fix this. | **Open (mandatory, fix-list #2)**: replace split-K-eligible GEMMs with an explicit deterministic algorithm or own fixed-geometry kernels; verify bit-identity across both models. |
| `perf_calibration` writes `perf-cache/<gpu-key>.json`; if a calibration entry sets a numeric knob, it becomes device-dependent. | **FIXED**: `EnsureCalibrated` no longer overrides `gemv_max_mib` (a numeric knob); it applies the canonical knobs only. Per-GPU numeric tuning is a C-fast-only, explicit-opt-in facility. |
| `lsx_tc_gemm.cu` `cudaGetDeviceProperties` probes (TC availability) | Same CC 12 across the set -> consistent; must never select a different numeric path per model (perf-only gating). |
| Cross-model verification | Only RTX 5090 is available locally; the 5060 Ti run must be done on that box. CI hook (fix-list #4) should run the selfcheck on both and record a cross-model diff. |
