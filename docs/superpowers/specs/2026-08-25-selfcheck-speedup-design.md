# SelfCheck Speedup: Qwen3.5-35B-A3B & gemma-4-26B-A4B-it

- **Date**: 2026-08-25
- **Status**: Approved (approach A — production-path caching)
- **Goal**: Reduce `test_qwen_moe_selfcheck` (~177 s) and `test_gemma4_selfcheck` (~87 s) to seconds by eliminating repeated one-time work inside `Infer`. Target: pragmatic maximum from safe optimizations ("сколько получится"), no fixed deadline.
- **Hardware assumption**: RTX 5060 Ti 16 GB VRAM is sufficient once redundant transfers are removed.

## Invariants

1. Bit-exact determinism of `final_logits` across 3 stress cycles + "Paris" top-1 — existing tests keep their semantics and serve as the TDD harness.
2. No math changes: `-fmad=false` stays; caches hold **the same bytes**, computed/copied once instead of per call.
3. 3 stress cycles, SKIP semantics (`SKIP_RETURN_CODE 4`), env knobs — unchanged.

## Diagnosis (measured anchors)

Both tests load the model once and call `Infer` N times; all per-cycle cost is inside `SelfCheck → Infer`.

### Gemma (87 s)

| Problem | Anchor |
|---|---|
| Every `Infer` rebuilds the expert index: opens + Arrow-parses all 3840 expert files | `gemma_base_model.cpp:110-113`, `expert_store.h:50-81` |
| No device expert cache: full w1/s1/w2/s2/w3/s3 re-upload H2D per routed expert per token; ~20 GB H2D per cycle (prefill runs experts per prompt token) | `gemma_inference.cu:903,917-939` |
| `cublasCreate/Destroy` per Infer + ~30 workspace mallocs per call | `gemma_inference.cu:1040,1067-1103,1234` |
| Two+ blocking `cudaStreamSynchronize` per layer per forward (~60+/cycle) | `gemma_inference.cu:893,608,721,1163,1216,1226` |

One-time Load: 4.58 GB pinned upload.

### Qwen MoE (177 s)

| Problem | Anchor |
|---|---|
| Full NVFP4 dequant of `embed_tokens` + `lm_head` (~2 × 1.02 GB FP16 out) every Infer | `qwen_moe_inference.cu:456-465` |
| ~2 GB workspace arena `cudaMalloc/free` per call | `qwen_moe_inference.cu:412` |
| Per step × per layer router D2H readback + busy-poll event spin (40× per generated token) | `qwen_moe_inference.cu:611-618` |
| LRU cache only 64 slots of 10240 experts → ~500 MB H2D misses per generated token | `qwen_3_5_moe_model.cpp:99-106`, `qwen_moe_inference_load.cpp:666-711` |

One-time Load: dense 1245 MB + expert store 17.3 GB read twice (~35 GB I/O) into pinned slab.

## Changes

### Gemma (`gemma_base_model.cpp`, `gemma_inference.cu`)

- **G1** Parse the expert index once in `Load()`; mmap handles live in `GemmaBaseModel`; `Infer` reuses them.
- **G2** LRU device cache for routed experts, adapted from `QwenMoeExpertCache` (`libs/lsxcommon/include/lsxcommon/qwen_moe_expert_cache.h`; host-bookkeeping already unit-testable without CUDA). Default slots ~1024 (~3.2 GB at avg 3.2 MB/expert).
- **G3** Single persistent `cublasHandle_t` + persistent workspace arena owned by the model.
- **G4** Folded into G2: cache hits eliminate duplicate per-(token,expert) uploads; remaining syncs (per-step argmax handshake) are required for greedy sampling.

### Qwen MoE (`qwen_3_5_moe_model.cpp`, `qwen_moe_inference.cu`)

- **Q1** Cache dequantized FP16 `embed_tokens` + `lm_head` in the model after first `Infer`; subsequent calls skip dequant.
- **Q2** Persistent workspace arena (alloc once, reuse).
- **Q3** Default LRU slots 64 → 512 (~0.9 GB); `LSX_QWEN_MOE_EXPERT_CACHE_SLOTS` knob preserved.
- **Q4** ~~Batch the 40 per-layer D2H router readbacks~~ **Rejected during planning**: layer l+1's router input depends on layer l's MoE output (strict sequential dependency). Replacement: **drop the unused `d_head_h` FP16 table** (V×H ≈ 1 GB allocated + dequanted every call; the head GEMM reads packed NVFP4 directly via `RunQwenNvfp4Gemm`, `qwen_moe_inference.cu:638`).
- **Q5** Eliminate the double read of the expert store at Load: pass 1 obtains file sizes via filesystem metadata (`std::filesystem::file_size`) instead of fully reading each file → ~35 GB I/O drops to ~17.3 GB.

### VRAM budget (16 GB)

| Model | Dense | New caches | Workspace/KV | Total est. |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 1.25 GB | embed table 1.0 + slots 0.9 | ~1 GB | ~4 GB |
| gemma-4-26B-A4B-it | 4.58 GB | expert slots ~3.2 | ~1 GB | ~9 GB |

Slot defaults are conservative and configurable for longer contexts.

## Observability

chrono phase timers via spdlog: Load duration, first cycle, subsequent cycles, cumulative expert-cache H2D byte counter. No time-based asserts in tests.

## Testing (TDD)

1. Existing selfcheck tests are the green fixture: run before changes to record the baseline, after each step as regression (bit-exactness ×3 cycles + "Paris").
2. New unit tests:
   - Gemma: expert index opened exactly once across N `Infer` calls (file-open counter); Gemma slot-cache hit/miss on host bookkeeping (no CUDA required).
   - Qwen: embed/head cache persists between `Infer` calls (dequant skipped — verified via flag/counter).
3. Verification: `ctest -R "qwen|gemma"` (Windows restriction per AGENTS.md), wall-time before/after.

## Risks

- Moving `load_gemma_expert_index` into `Load()` changes Gemma's memory profile → control handle lifetime / `max_open`.
- Larger LRU may pressure KV on 16 GB with long contexts → configurable slots, conservative defaults.
- Readback batching must not alter expert selection order — selections are identical, transport only.
