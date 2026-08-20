# Blackwell Inference Optimization Strategy — Design Spec

**Date:** 2026-08-20
**Status:** Approved (strategic roadmap)
**Scope:** `libs/lsxcommon` CUDA inference kernels + MoE expert staging, all four registered models (GLM-5.2, DeepSeek-V4-Flash, Qwen3.5-9B, Qwen3.8-27B).
**Prerequisite state:** shared plumbing already unified — `ModelBase` (load/Inspect/ResidentWeightBytes/num_layers), `nvfp4_codec.h` (E2M1/E4M3), `expert_store.h` (MoE LRU), sampling seam, unified `IModel::Benchmark`. This spec is the *optimization* stage on top of that.

## 1. Problem Statement / Goal

The inference hot paths are functionally correct but not yet tuned to extract the hardware density of Blackwell (CC 12.0/12.1 — 48 warps/SM max, 5th-gen Tensor Cores, TMA). Now that the shared plumbing is in place, we can optimize the computational kernels without re-absorbing duplicated loader/registries.

**Metric contract (user-selected):**
- Optimize to raise **Blackwell occupancy / hardware-level packing** (TMA, tile programming, warp/slot utilization), NOT to change numerics.
- **Bit-exactness is the invariant** where a model has a deterministic selfcheck contract. GLM selfcheck (`"Paris"`, 3-cycle bit-exact `final_logits`) and the shared `RunParisSelfCheck` for Qwen/DeepSeek are locked. Compute kernels MUST NOT reorder FP operations, fuse `FMA`, or change rounding for those paths. (`-fmad=false` on `lsxcommon_glm_cu` stays.)
- A per-kernel optimization is **accepted only if** the affected real-model selfcheck stays bit-identical and the surrounding synthetic kernel tests (`test_deepseek_inference_kernels.cu`, `test_sampling`, `test_expert_store`, GLM parity suites) pass.

## 2. Goals (non-goals)

**Goals**

- Formalize the boundary: **which hot-paths are optimized separately per model** (attention/RMSNorm/GEMM) vs **which are optimized once on the shared store** (MoE routing + mmap staging).
- Enumerate the concrete kernel symbols on each path (project can point to them).
- Define the per-optimization acceptance gate (bit-exactness + occupancy/throughput).

**Non-goals**

- No algorithm change. No numerical relaxation. No change to the storage schema or the unified plumbing.
- No new inference feature.
- This is a **strategy/roadmap spec**, not a single implementation plan — the execution happens via one plan per target kernel (separate documents, `writing-plans`).

## 2. The two directions — why and what they give

### Direction A: separate per-model hot paths (attention / RMSNorm / GEMM)

These kernels are **architecturally model-specific** — they do NOT share device symbols across models and must NOT (bit-flags differ per model; a shared path would leak `-fmad=false` or a numeric contract across models).

Concrete kernel hot paths (from `rg __global__`):

| Model | Attention | RMSNorm | GEMM / FFN |
|---|---|---|---|
| GLM | `GlmMlaAttnKernel`, `GlmIdxAttnKernel` (`glm_inference.cu:819,328`), `GlmMlaStoreKvKernel` | `GlmRmsNormKernel` (`glm_expert_kernels.cuh:46`) | `GlmGemmFp64Kernel` (`glm_expert_kernels.cuh:75`), `GlmExpertAccumKernel`, `GlmMoELogitsFp64Kernel` |
| DeepSeek | `mla_attn_kernel`, `csa_sparse_attn_kernel`, `hc_pre/hc_head/hc_post_kernel`, `per_head_rms_kernel` (`deepseek_inference_attn.cu`) | `rms_norm_kernel` (`deepseek_inference_norm.cu`), `rms_norm_float_kernel` (`deepseek_norm_kernels.cu`) | `dequant_nvfp4_to_half_kernel`, `act_quant_block_kernel`, `silu_mul_kernel`, `swiglu_*`, `dequant_f8_*` |
| Qwen | `full_attn_softmax_kernel`, `qwen_gated_delta_kernel`, `qwen_split_query_kernel` (`qwen_inference.cu:229,534`) | `rmsnorm_offset_kernel`, `rmsnorm_gated_kernel` | `qwen_nvfp4_gemm_kernel`, `qwen_swiglu_kernel` |

**What tuning the separate path gives:**
- Isolates each model's numeric contract (GLM's `-fmad=false` never leaks; DeepSeek/Qwen keep their default FMA-packed arithmetic).
- Lets each attention/GEMM become aggressive per-arch under Blackwell (tile programming, TMA, warp/SM occupancy ≤ 48), without a cross-model hard contract.
- No regression on a sibling model when one path is tuned.

### Direction B — MoE routing & staging on the shared store

These ARE shared now: `ExpertStore<TEntry>::EnsureResident` (`experpt_store.h`), the LRU touch/evict, the pre-eviction `cudaStreamSynchronize` barrier, and the wrapper names `load_expert_index` / `load_glm_expert_index`.

**What tuning it shares:**
- An optimization to the eviction barrier / async pre-staging (host→device of the likely-next expert while the current one runs) applies to **both GLM and DeepSeek simultaneously** — a single place.
- The same mmap staging policy (bounded LRU warm-up, deferred parse) benefits every MoE model added later; a new model just instantiates `ExpertStore<T>` and inherits the optimization.
- Decode loops (`glm_inference_load.cu:755`, `deepseek_inference_pipeline.cu:574`) call the same registration path.

**Boundary rule:** the store owns memory/load/staging; the **dequant** (each model's re-decode of packed weights — `GlmDequantNvfp4Kernel` vs `dequant_nvfp4_to_half_kernel` in MoE) stays per-model and is NOT merged into the store. Only the LRU/route/staging is shared.

## 4. Decision table — which direction to apply where

| Optimization | Direction | Why |
|---|---|---|
| Attention (full/CSA/MLA/IndexShare) | **A (separate)** | architecturally model-specific; bit-contracts per arch; no shared device symbols |
| RMSNorm | **A** | fusion per layer dim, no cross-model benefit; `-fmad=false` per GLM |
| Dense GEMM, SwiGLU (Qwen) | **A** | fused-tile vs cuBLAS; model the acc contract |
| MoE routing (load_expert_index / ensure_resident / staging) | **B (shared)** | now one store; both GLM & DeepSeek benefit at once; deferred parse / mmap LRU / evict barrier tuned once |
| MoE dequant (expert weight decode) | **B-boundary** | staging is shared but decode per model (bit-exact) |

## 5. Acceptance gate per optimization

For each candidate optimization (per direction), the plan SHALL verify:

1. **Bit-exact contract preserved:**
   - GLM: `test_glm_selfcheck` (3-cycle `final_logits` byte-identical, `"Paris"` first-gen).
   - DeepSeek: `test_deepseek_selfcheck` (Paris, 3-cycle bit-exact).
   - Qwen: `test_qwen_selfcheck`.
   - SKIP-without-checkpoint is acceptable; with real models present they must be bit-exact.
2. **Synthetic parity**: `test_glm_*`, `test_deepseek_inference_kernels.cu` (4470), `test_sampling`, `test_expert_store`, `test_glm_moe_compute` / `test_glm_experts` pass unchanged.
3. **Occupancy/throughput guard**: measure (or reason) that warps/SM ≤48, no occupancy regression; where a kernel is benchmarked, tokens/s must not regress versus pre-change.
4. **No plumbing regression**: GLM/DeepSeek loader + store tests still pass; `rg "struct ExpertDb|struct GlmExpertDb"` stays empty.

No kernel may be committed if it breaks 1, 3, or 4.

## 6. Output / artifacts

- This spec is the strategic boundary doc.
- Each following work item (one attention tuning, one store regular optimization, …) spawns its own spec + `writing-plans` per `docs/superpowers/specs/YYYY-MM-DD-<opt>-design.md` + the corresponding implementation plan.

## 7. Out of scope / open items

- Measuring/targeting actual Blackwell occupancy (occupancy counters already exist in the Phase B plan; tune numbers per kernel there).
- Deciding whether each GEMM path uses cuBLAS vs a custom fused kernel — per-path, not this doc.
- Any algorithm change or numerical relaxation — non-goal, see above.

## Self-review

**Coverage:** The doc covers the boundary (A/B), a concrete kernel map per model, a decision table, an acceptance gate, output/artifacts, and the non-goal of "numerical relaxation". All sections align; no placeholders; scope is appropriate to a strategy (one plan per target unit).