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

1. Initialize cuBLAS with a pinned workspace + atomics off (single place in the
   engine init / cublas handle creation).
2. Replace `CUBLAS_GEMM_DEFAULT` where split-K is eligible with an explicit
   deterministic algorithm (or keep `CUBLAS_GEMM_DEFAULT` but verify bit-identity
   across `:16:8`/`:4096:8` after pinning).
3. Gate `perf_calibration`/`gpu_perf` numeric knobs to device-keyed values that
   do not read free memory; document that residency sizing is exempt.
4. Add a cross-instance/workspace-variant selfcheck to CI for C-exact models.
