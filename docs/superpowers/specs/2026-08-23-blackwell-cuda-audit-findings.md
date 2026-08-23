# Blackwell CUDA Audit — Findings Registry

Companion to `2026-08-23-blackwell-cuda-audit-design.md`. Every finding gets a
severity (P0 blocker / P1 major / P2 minor), file:line evidence, remediation
status, and before/after measurements when available.

| ID | Sev | Dimension | Evidence | Summary | Status |
|---|---|---|---|---|---|
| F-01 | TBD | A | glm_inference.cu:84 | Scalar GEMM kernels on CUDA cores; no tensor-core MMA usage anywhere | open |
| F-02 | TBD | B/C | qwen_inference.cu:1996 | Final-logits publish uses blocking null-stream cudaMemcpy inside timed region | open |
| F-03 | TBD | C | glm_inference.cu:1649-1653 | Per-token H2D roundtrip of sampled id + full cudaStreamSynchronize | open |
| F-04 | TBD | C | deepseek_inference_pipeline.cu:772 | cudaDeviceSynchronize inside pipeline | open |
| F-05 | TBD | A | various launch sites | Degenerate launch configs (<<<1,...>>>, flat 256-thread blocks); no occupancy tuning vs 48 warps/SM | open |
| F-06 | TBD | E | libs/lsxcommon/CMakeLists.txt | -fmad=false applied to whole lsxcommon_glm_cu object library | open |

## Baseline measurements (Phase 0)

Filled during execution. Format per model:
`<model> @ <gpu>: <tok/s> decode, ttft <ms>, total <ms> (<gen tokens>)`

## New findings discovered during execution

Append below with next free F-nn.
