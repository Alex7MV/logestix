# CUDA Inference Kernel Unification — Design Spec

**Date:** 2026-08-20
**Status:** Approved (pending implementation plan)
**Scope:** `libs/lsxcommon` CUDA inference kernels (`.cu` / `.cuh` only)
**Prerequisite:** Initial plumbing unification (ModelBase Load/Inspect/ResidentWeightBytes) is already merged to `main`.

## 1. Problem Statement

A kernel-overlap audit of the three models (GLM-5.2, DeepSeek-V4-Flash, Qwen3.5-9B /
Qwen3.8-27B) found real hardware-level DUPLICATES that, if left untouched,
would cause the upcoming optimization work to land the same algorithm up to 3×
in separate files. Two of these are model-independent and block efficient
optimization; several GLM-internal twins are quick wins. Attention and
model-specific fused kernels are genuinely model-specific and must be
optimized separately (this boundary is deliberately NOT unified).

## 2. Goals (non-goals)

**Goals**
- Collapse the duplicated NVFP4/E4M3 decode tables + scale functions into ONE
  shared device codec header.
- Replace the two independent MoE mmap-expert LRU loaders with ONE shared
  expert-store abstraction.
- Collapse the GLM-internal twin kernels into single parametric kernels.
- Explicitly preserve extensibility: shared points stay architecture-agnostic
  so future models plug in without deep-divide work (see §5).

**Non-goals**
- NO behavior change: bit-exact GLM semantics and per-model numeric behavior
  are preserved exactly.
- Do NOT unify attention (full/CSA/MLA/IndexShare), RMSNorm forms, Qwen's fused
  NVFP4-GEMM, or cuBLAS-vs-custom-GEMM strategies — these are model-specific and
  optimized separately.

## 3. Sections

### 3.1 NVFP4/E4M3 decoder → shared `dequant` codec (HIGH value, LOW risk)

Create one device header `libs/lsxcommon/src/nvfp4_codec.h`, namespace
`lsxcommon::gpu::nvfp4`, reachable by all three CUDA object libraries
(`lsxcommon_cu`, `lsxcommon_glm_cu`, `lsxcommon_qwen_cu`). It holds:

- ONE `__constant__` E2M1 table (16 entries) replacing:
  - `kE2M1Table` (`shared_experts.cu:41-44`)
  - `kGlmE2M1Table` (`glm_inference_load.cu:35-38`)
  - `qwen_e2m1_tab` (`qwen_inference.cu:337-340`)
- ONE `__device__` `e4m3_scale_to_float` replacing:
  - `e4m3_scale_to_float` (`shared_experts.cu:47`)
  - `GlmE4M3P2ToFloat` (`glm_inference_load.cu:40`)
  - `qwen_e4m3_scale` (`qwen_inference.cu:342`)
  - `e4m3_scale_to_float` (`benchmark_gpu.cu:44`, gpu-ns copy)
- Remove the dead `nvfp4_to_float` (different 1-2-4 mantissa formula,
  `benchmark_gpu.cu:33`-inline + `extern` decls in pipeline.cu:24 / moe.cu:18,
  never called). Unify to the single formula; the GLM variant must remain
  bit-identical.

Compile boundary: `__constant__`/`__device__` symbols have no host linkage, so
the 3-lib NVCC split is safe (each TU/const-bank gets its own device copy, as it
already does today). Reachability via include-dir per object lib.

Bit-exactness constraint: `GlmE4M3P2ToFloat` uses bitwise `(e-7+127)<<23`;
the shared function must be bit-identical for GLM. Verified by `test_glm_*`
(bit-exact logits), `test_deepseek_inference_kernels.cu`, `test_qwen_model`.

### 3.2 MoE expert store → shared abstraction (HIGH value, HIGH risk)

Replace the two independent mmap-expert LRU loaders with ONE header-only
template keyed on an expert-layout descriptor:
- DeepSeek `ExpertDb` (`deepseek_inference_load.cuh`, load_expert_index /
  ensure_expert_resident) — E8M0-FP4, per-32 scale.
- GLM `GlmExpertDb` (`glm_inference_load.h`, load_glm_expert_index /
  ensure_glm_expert_resident) — E4M1, per-32 scale.

Shared logic (`lsxcommon::gpu::expertstore::`):
- mmap all expert arrows
- deferred-parse into LRU bounded by CRT fd cap (~7000, Windows)
- `EnsureResident(layer, eid, stream)` — touch/evict + stream-barrier
- `StageTensors(...)` — the 6× `cudaMemcpyAsync` (w1/w2/w3 + 3 scales) into
  per-model staging.

The two format schemas differ (record fields, hidden dim, scale layout);
parameterize via template/descriptor. This is the highest-value unification but
also the highest complexity (two formats, two namespaces `glm` vs `gpu`,
stream-barrier subtleties). Preserve exact runtime behavior; verify via
`test_deepseek_inference_*` + `test_glm_experts` / `test_glm_moe_compute`.

**Splitting decision:** §3.2 is standalone large enough to warrant its OWN
spec+plan. This spec covers §3.1 and §3.3; §3.2 is deferred to a follow-on
plan (both still gate optimization, run after this one).

### 3.3 GLM-internal twin kernels → single parametric (quick wins)

Within the single `lsxcommon_glm_cu` library, collapse:
- `GlmGemmFp64Kernel` defined twice — `glm_inference.cu:93` and
  `glm_inference_load.cu:223` (byte-identical). One definition, one call site.
- `GlmRmsNormKernel` (`glm_inference.cu:35`) + `GlmMoERmsNormKernel`
  (`glm_inference_load.cu:114`) — same arithmetic, differ in weight-addressing /
  flag. One parametric kernel.
- 3× `GlmSwiglu*` — `GlmSwigluClampKernel` (`glm_inference.cu:61`),
  `GlmRealSwigluKernel` (`glm_inference.cu:937`),
  `GlmSwigluKernel` (`glm_inference_load.cu:86`). One kernel with a clamp/limit
  flag. All use the bit-exact `GlmExpfGlibc`, so bit-exactness is preserved.

Result: single optimization target size=1 for GEMM / RMSNorm / SwiGLU inside GLM.

## 4. What is NOT unified (explicit boundary)

| Area | Why separate |
|---|---|
| Attention (full/CSA/MLA/IndexShare) | different algorithms, zero shared symbols |
| RMSNorm forms | bit-exact sequential (GLM) vs float4 warp (DeepSeek) vs offset/gated (Qwen) |
| Qwen fused `qwen_nvfp4_gemm` | decode fused into GEMM for bandwidth, model-specific |
| cuBLAS vs custom FP64 GEMM | GLM/Qwen eschew cuBLAS for bit-exactness/quantized-GEMM |

These are optimized per-model; do NOT attempt shared kernels.

## 5. Extensibility preservation (model-roadmap guarantee)

Expanding the model line-up must remain EASIER after unification, not harder.
Contract for optimization work on the shared points:

1. Shared codec (`nvfp4_codec.h`), expert store, and LLM GEMM/RMS/SwiGLU shims
   stay **architecture-agnostic and parametrized**. A new dense/MoE model plugs
   in to read them without deep-dividing into their internals.
2. Per-model HOT paths (attention, fused GEMM, dead-path kernels that need
   bit-exactness) live in the model's OWN kernels — never buried into a shared
   core. This keeps per-model optimization from locking in a binding for
   future models.
3. Bit-exactness (`-fmad=false`) is GLM-only and must remain a per-MODEL policy,
   not a property of the shared codec. The codec is bit-identical for GLM but
   does not force the mode on others.
4. New architecture → new NVCC object library + CMake wiring (the 3-lib split
   pattern). Flag this as anticipated friction in the roadmap, not a blocker.

## 6. Error handling

Refactor-only: no new error paths. Expert-store errors reproduce the current
messages exactly (existing tests lock them). Codec removal of `nvfp4_to_float`
eliminates dead `extern` decls — no runtime path.

## 7. Verification

- Preserve bit-exact GLM: `test_glm_model`, `test_glm_selfcheck` (SKIP without
  checkpoint), `test_glm_experts`, `test_glm_moe_compute`, `test_glm_compute`.
- DeepSeek/Qwen unchanged: `test_deepseek_inference_kernels.cu`
  (4470 assertions), `test_deepseek_loader`, `test_qwen_model`,
  `test_qwen_3_8_27b_model`, `test_deepseek_selfcheck` / `test_qwen_selfcheck`.
- No `nvfp4_to_float` externs remain (grep).
- Full model subset `ctest -R glm -R qwen -R deepseek -R model` (real-model
  SelfChecks SKIP when checkpoints absent).
- Post-refactor deterministic SelfCheck of Qwen/DeepSeek unchanged.

## 8. Out of scope / deferred

- §3.2 shared expert store → separate follow-on spec+plan.
- Attention/RMSNorm-fused-GEMM optimization (post-unification, separate
  optimization plans).
- No new CUDA-architecture tuning in this pass (Blackwell CC 120/121 preserved).