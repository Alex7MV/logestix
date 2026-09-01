# DeepSeek TC W4A4 fast-math — Design Spec

**Date:** 2026-09-01
**Status:** Approved (design)
**Scope:** `libs/lsxcommon` DeepSeek-V4-Flash GEMM paths (routed experts, shared expert, dense attention projections). Resolves audit finding **F-12** (`2026-08-23-blackwell-cuda-audit-findings.md`).
**Branch:** `feat/nohandshake-graphs-5090`

## 1. Problem Statement

DeepSeek-V4-Flash is the slowest vault model on the Linux/5090 stack:
**0.6 tok/s, TTFT 20.6 s** (measured 2026-09-01, AOCC release, 32-token run)
vs 63.5 tok/s for Qwen3.5-35B-A3B on the same machine. Root cause: every
GEMM site dequants NVFP4/F8 weights to (FP16|FP32) **on every call** and
runs cuBLAS W16A16/W32A32 — ~4x weight traffic vs W4A4, plus per-call
dequant kernels on the hot path.

Qwen models already have a verified TC W4A4 route: warp-level
`mma.sync...kind::mxf4nvf4.block_scale` (`qwen_tc_gemm.cu`), validated on
sm_120a. DeepSeek weight format (E2M1 nibbles + E8M0 scale per 32) is the
MXFP4 class: `kind::mxf4` hardware combo — but the repo's proven kernel is
the NVFP4 (UE4M3 per-16) one.

## 2. Decisions (user-approved)

| Decision | Choice |
|---|---|
| Toggle | Reuse existing `ModelInitConfig::fast_math` (one flag for qwen + deepseek). Default OFF = current paths bit-exact. |
| Weight route | **B: repack + ready kernel.** Load-time GPU repack E8M0-per-32 → UE4M3-per-16 (E2M1 nibbles unchanged); call the existing `RunQwenTcNvfp4Gemm`. |
| Scope | **All GEMM sites** (routed FFN, shared expert, dense attention F8 projections), phased: routed FFN first (dominant), then shared/dense. |

## 3. Architecture

### 3.1 Routed experts — W4A4 (numeric class change, fast_math only)

- **Load-time repack (GPU kernel):** per-scale byte transform `e4 = e8 - 120`
  (E8M0 value `2^(e8-127)` == repo UE4M3 decode `(e4+120)<<23` for all
  representable pow2 values). Nibble packing untouched: both formats are
  E2M1, element order identical; only the scale layout differs
  (per-32 flat → per-16 pairs). Two consecutive 16-element blocks share one
  E8M0 scale → both UE4M3 bytes equal.
- **Range guard:** scale byte mapping `E = e8 - 120` (value identity:
  E8M0 `2^(e8-127)` == UE4M3 `2^(E-7)`). UE4M3 exponent nibble is 4 bits →
  exact only for `e8 ∈ [121..135]` (E ∈ [1..15]; E=15,M=0 stays finite
  `2^8`). `e8 > 135` → deterministic load error naming layer/expert.
  `e8 ≤ 120` → `E = 0` → hardware decodes the scale as 0, while legacy
  dequant gives `2^(e8-127) ≤ 2^-7` (|w| ≤ 6·2^-7 ≈ 0.047 underflow to
  zero) — bounded, deterministic deviation accepted as part of the W4A4
  numeric-class change (production pow2 scales sit in the exact range;
  validated by the repack unit test distribution).
- **VRAM:** repacked scale array alongside existing store (+~6%: scales are
  1/32 of payload → per-16 doubles them). Pack bytes shared.
- **Hot path (`fast_math=true`):** `QuantizeActsNvfp4` (per-16 pow2
  activations) → `RunQwenTcNvfp4Gemm` for W1/W3 → FP16 SwiGLU (existing
  silu_mul path) → re-quantize → W2 → epilogue `weight` accumulation into
  FP32 output (same as current).
- **Numeric class change:** W4A4 changes accumulation order vs the current
  dequant+cuBLAS route. Gated strictly behind `fast_math`; default routing
  bit-unchanged.
- Handshake/LRU/staging logic (`run_routed_experts_ex`, expert cache,
  pack_stage) is untouched — only the compute call inside the per-expert
  body dispatches.

### 3.2 Shared expert + dense attention — bit-exact dequant caching (always-on)

- `run_shared_experts_ffn` and the attention F8 sites (`wq_b`, `wo_a`,
  `wo_b` via `d_attn_dequant`) dequant weights **once at load** into a
  persistent FP16 cache in VRAM (~2.2 GB total for shared × 43 layers +
  small attention projections; fits the 32 GB budget).
- The cached FP16 weights feed the **same cuBLAS FP16 GEMMs** as today:
  identical kernel, identical inputs → **bit-exact** vs current behavior.
  Benefits the default (fast_math=off) path too.
- W2 stays FP32 GEMM where the numeric contract requires it (existing
  rationale in code comments preserved); its dequant is also cached (FP32
  or FP16-with-exact-pow2-fold — decided at plan stage by measurement of
  the exactness constraint, defaulting to FP32 cache to preserve bitness).

### 3.3 LM head

Unchanged (FP16 cuBLAS, DRAM-bound — TC W4A4 gives nothing at M=1).

## 4. Components & files (naming convention `family_type_role[_subsystem]`)

| Unit | File | Responsibility |
|---|---|---|
| Repack kernel | `deepseek_moe_load.cu` (+ `.cuh` decl) | E8M0→UE4M3 per-16 scale repack, range validation |
| FP16 weight cache | `deepseek_moe_load.cu` | load-time dequant → persistent FP16 arena slices |
| Dispatch | `deepseek_moe_inference_moe.cu` | `run_routed_experts_ex` / `run_shared_experts_ffn` body: fast_math ? TC : legacy |
| Toggle | mirror of `SetQwenTcFastMath` in the deepseek inference TU | process-wide, set from `ModelInitConfig::fast_math` at model construction |
| Tests | `tests/test_deepseek_moe_inference_kernels.cu` + new repack test | TDD per plan |

No new model files; all edits inside existing `deepseek_moe_*` units
(naming guard green).

## 5. Acceptance gate (strategy spec §5, binding)

1. **Bit-exact default:** `test_deepseek_selfcheck` fast_math=OFF — 3-cycle
   byte-identical `final_logits` (unchanged baseline).
2. **fast_math ON:** "Paris" top-1, deterministic across 3 cycles.
3. **Repack unit test:** host-reference parity incl. boundary e8 values
   {0, 120, 127, 248, 249→error}; round-trip through the qwen TC kernel
   matches dequant-reference within the NVFP4 bound.
4. **Synthetic parity:** `test_deepseek_moe_inference_kernels.cu` (4470
   assertions) green; loader/store tests green.
5. **Perf guard:** `test_blackwell_perf_deepseek` — tok/s must not regress;
   target: measurable improvement over the 0.6 tok/s / 20.6 s ttft baseline.
6. Registry update in the same commit series (F-12 status).

## 6. Non-goals

- No algorithm change to routing/top-k/LRU.
- No `kind::mxf8f6f4` W8A8 kernel for F8 sites (parked; candidate after
  phase-1 measurement).
- No GLM/DeepSeek store plumbing changes (staging untouched).
- No change to default (fast_math=off) numeric behavior except the
  bit-exact dequant caching of 3.2 (which is bit-identical by construction).
