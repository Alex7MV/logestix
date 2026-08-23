# Blackwell CUDA Bottleneck Audit — Design

Date: 2026-08-23
Status: Approved design (pending implementation plan)

## Goal

Comprehensive low-level audit of the transformer inference stack and its CUDA
usage, targeting hidden performance bottlenecks with Blackwell (sm_120)
specifics in mind. Every finding ends up either fixed test-first (TDD) or
explicitly documented as accepted in the findings registry.

## Scope

- **Models:** GLM-5.2 (`glm_v5_model.cpp` + `glm_inference.cu`),
  DeepSeek-V4-Flash (`deepseek_inference_pipeline.cu` and related),
  Qwen3.5-9B (`qwen_inference.cu`).
- **Primary tuning target:** RTX 5060 Ti 16GB (34 SMs, 48 warps/SM max,
  228 KB smem/SM). RTX 5090 32GB is a secondary validation target; code stays
  generic across both — no per-card specialization.
- **Accuracy policy (hybrid):** the existing bit-exact scalar path remains the
  default and keeps bit-exact `final_logits` guarantees; a native tensor-core
  NVFP4 path is added behind a runtime flag.

## Known Findings (pre-audit evidence)

| ID | Dimension | Evidence | Summary |
|---|---|---|---|
| F-01 | A | `glm_inference.cu:84` | Scalar FP64 GEMM kernel on CUDA cores; no tensor-core usage anywhere in `.cu` sources |
| F-02 | C | `qwen_inference.cu:1903-1904, 1996` | Per-token blocking `cudaMemcpy` H2D scalars + D2H logits inside generation loop |
| F-03 | C | `glm_inference.cu:1651-1700` | Multiple `cudaStreamSynchronize` in GLM hot path |
| F-04 | C | `deepseek_inference_pipeline.cu:772` | `cudaDeviceSynchronize` inside DeepSeek pipeline |
| F-05 | A | launch sites throughout | Degenerate configs (`<<<1,...>>>` RoPE kernels, flat 256-thread blocks), no occupancy tuning vs 48 warps/SM |
| F-06 | D/E | `CMakeLists.txt` (glm_cu target) | `-fmad=false` applied to entire object library instead of only bit-exactness-critical kernels |

The full registry (F-01..F-nn with severity and remediation status) is built up
during Phase 1 execution and lives in this document's companion report.

## Audit Dimensions

Each finding gets an ID, severity, and `file:line` evidence.

| ID | Dimension | What's checked |
|---|---|---|
| A | Kernel-level | Launch configs vs 48-warp/SM occupancy; scalar-FMA GEMM vs native `mma.sync.aligned.kind::mxf4nvf4.block_scale`; shared-memory RMW sandbox invariant; stride invariant |
| B | Memory/transfers | Per-token H2D/D2H, pinned staging, device arena / cudaMalloc frequency, VRAM headroom @16GB |
| C | Synchronization | Single-stream invariant violations: per-token syncs, mid-loop stream/device syncs |
| D | Blackwell fit | GeForce limits: cluster 1x1x1 (no multicast), TN-only MMA layout, TMA/tmem applicability; `-fmad=false` blast radius |
| E | Build flags | arch flags, redundant syncs induced by flags, CompileIQ opportunity |

Blackwell SM120 constraints that bound the solution space (per NVIDIA docs):
GeForce has no cluster multicast (cluster shape fixed 1x1x1); narrow-precision
MMA supports TN layout only; block-scaled `mxf4nvf4` MMA reaches 2x Ada FP8
tensor-core throughput (4x with FP32 accumulator).

## Approach

Layered bottom-up, four phases. Each phase is independently verifiable; risky
numerics work is isolated last.

### Phase 0 — Characterization (TDD baseline)

New Catch2 executables `test_blackwell_perf_glm`, `test_blackwell_perf_qwen`,
`test_blackwell_perf_deepseek`: run N-token decode on real models (SKIP without
weights, matching existing suite behavior), record tok/s + per-phase timings
via CUDA events through spdlog into machine-readable output. Reporting tests
only — no hard perf gates (no flaky CI). Establishes "before" numbers reused
after each phase.

### Phase 1 — Sync & transfer fixes (bit-exact)

- Qwen loop: persistent device buffers + `cudaMemcpyAsync` from pinned host
  replace per-token scalar H2D; token-id D2H via pinned buffer + stream event,
  no device-wide sync.
- GLM/DeepSeek: remove mid-loop `cudaStreamSynchronize` /
  `cudaDeviceSynchronize`; AGENTS.md single-stream invariant preserved; final
  sync only at generation boundary.
- Tests: existing bit-exact selfcheck suites pass unchanged; golden 8-token
  generation captured in Phase 0 must match exactly.

### Phase 2 — Occupancy & launch tuning (bit-exact)

- Audit every `<<<...>>>` against SM120 limits; fix degenerate configs; add
  vectorized 16B loads where alignment allows.
- Narrow `-fmad=false` to bit-exactness-critical kernels only.
- Tests: same golden-token tests; occupancy assertions via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` where meaningful.

### Phase 3 — Tensor-core NVFP4 path (runtime flag, non-default)

- New GEMM kernel using inline-PTX
  `mma.sync.aligned.kind::mxf4nvf4.block_scale` (compiled by nvcc 13.3; no new
  CUTLASS dependency). NVFP4 operand layout matches the existing E2M1-nibble +
  per-block E4M3 + global_scale format in `schema.h`.
- Flag: `--fast-math` CLI on fabric/taskmosaic apps, plumbed through
  `IModel::Load` options. Default off = current bit-exact behavior.
- Tests: kernel unit tests vs FP64 reference (tolerance ~1e-2 relative,
  consistent with NVFP4 quantization noise); run-to-run determinism (identical
  logits bit-for-bit across 3 runs even on fast path); SelfCheck top-1 "Paris"
  passes on fast path; 3-cycle stress patterned after `test_glm_selfcheck`.

## Error Handling

- Strict errors on missing/corrupt model files (existing loader pattern).
- Real-model tests SKIP gracefully without weights.
- Fast path validates scale-factor alignment at load; falls back to scalar
  path with spdlog warning rather than failing.

## Testing Strategy Summary

TDD throughout: baseline/golden tests first, then fix until green.

1. Bit-exactness: default path must keep existing suites passing unchanged.
2. Fast-path correctness: FP64-reference tolerance tests + run-to-run
   determinism + SelfCheck "Paris".
3. Performance: Phase 0 harness re-run after each phase; results recorded in
   the findings registry (report-only).

## Constraints (from AGENTS.md)

- Single-stream generation loop invariant (no per-token host round-trips).
- Shared-memory RMW sandbox: writes only by thread 0 under `__syncthreads()`.
- Stride invariant for Q/K/V addressing.
- Blackwell-only architectures (120/121); Hopper/Ada/Ampere forbidden.
- All logging via spdlog; preserve Doxygen comments when refactoring.
- abseil containers never in headers reachable from `.cu` TUs.

## Risks

- Inline-PTX MMA correctness curve → mitigated by FP64-reference unit tests
  before any model integration.
- `-fmad=false` scope narrowing could silently change numerics of non-critical
  kernels → each unscoped kernel needs an explicit bit-exactness check before
  allowing fmad.

## Success Criteria

- Completeness over raw speed: every finding documented with evidence and
  either fixed via TDD or explicitly accepted.
- Default path remains bit-exact (all existing suites green, unchanged).
- Fast path deterministic run-to-run and passing SelfCheck.
