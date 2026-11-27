# GLM MoE Async Pre-Staging — Design Spec (first optimization under strategy)

**Date:** 2026-08-20
**Status:** Approved
**Scope:** `libs/lsxcommon` — GLM-5.2 MoE routed-expert **staging only** (`glm_inference_load.cu`, `glm_inference.cu` — on-demand path on the shared `ExpertStore`). Direction **B** of `docs/superpowers/specs/2026-08-20-blackwell-inference-optimization-strategy-design.md`.
**Bit-exactness invariant:** this optimization changes the *timing/overlap* of async H2D staging and the eviction barrier — it does NOT reorder FP operations, fuse FMA, or change rounding. GLM selfcheck (`"Paris"`, 3-cycle bit-exact `final_logits`) must stay bit-identical.

## 1. Problem

In `RunGlmExpertsStackOnDemand` (`glm_inference_load.cu:752-818`) the per-slot decode loop is:

```cpp
for (int s = 0; s < slots; ++s) {
    ensure_glm_expert_resident(db, layer, rs.eid, stream);   // line 755 — may cudaStreamSynchronize on miss
    ... check host_w*/bytes ...
    e = cudaMemcpyAsync(p1..p3, s1..s3, ..., stream);        // 6 H2D copies     (770-792)
    RunGlmExpertSlot(cublas_h, stream, eb, x_t, out_t, d_scratch, rs.coeff);  // 815
}
```

Two issues:

1. **`ensure_glm_expert_resident` can block the whole slot at line 755.** The `ExpertStore::EnsureResident` (shared, `expert_store.h`) calls `cudaStreamSynchronize(stream)` *only when an eviction is about to happen* — but a miss on a cold LRU is exactly that common case. That synchronize **stalls the async pipeline on every recent-miss** (no overlap of the previous slot's device work with this slot's mmap parse / upcoming H2D).
2. **Staging is single-buffered and single-streamed.** `RunGlmExpertsStackOnDemand` takes one `d_pack_stage`/`d_scale_stage` set (the header requires `>= one expert`) and runs both H2D and GEMM on ONE stream. The next slot's H2D **cannot start** until slot `s`'s GEMM finished the same staging buffer (data hazard), so the H2D of slot `s+1` cannot overlap slot `s`'s compute at all.

The compute per slot (`RunGlmExpertsSlot` — three FP64 GEMMs) is far heavier than the H2D (a few MiB). Overlapping the next slot's mmap+H2D with the current slot's GEMM — via two streams — removes a serialization bubble at low risk. Blackwell favors keeping the DMA/copy engine fed ahead of compute.

## 2. Goal / non-goal

**Goal:** overlap slot `s+1`'s mmap-residency + H2D staging with slot `s`'s `RunGlmExpertsSlot` compute, using **two streams** (compute + copy), **double-buffered staging** and **event-based waits** instead of the full `cudaStreamSynchronize` — while keeping the output byte-for-byte identical.

**Non-goal:** no change to the dequant / GEMM kernels (`GlmDequantNvfp4Kernel`, `GlmGemmFp64Kernel`, `GlmExpertAccumKernel`), no change to numbers, no new algorithm, no change to the store's data structures.

## 3. Design

### 3.1 Double-buffered staging

`RunGlmExpertsStackOnDemand` currently receives `d_pack_stage`/`d_scale_stage` (single set, sized `>= 1 expert` — `glm_inference.h:206-208`) and a single `stream`. We add a second staging set, a separate copy stream, and events:

```cpp
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,      // compute stream (GEMM)
    cudaStream_t stream_copy,                          // copy stream (H2D)
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage_a, uint8_t *d_pack_stage_b,
    uint8_t *d_scale_stage_a, uint8_t *d_scale_stage_b,
    cudaEvent_t ev_h2d[2], cudaEvent_t ev_gemm[2],
    cudaEvent_t ev_evict,
    GlmRoutedSlot *h_routing_staging = nullptr);
```

- The **`GLM` caller `glm_inference.cu:299` ** allocates `2×` (the config/fabric arena path stays 1× unless it changes; caller passes both sets).
- The loop alternates `buf = (s & 1) ? b : a` so slot `s` reads from set `(s&1)` while set `(s+1)&1` is being H2D'd.

Buffer layout (bytes) as today: `[w1][w2][w3]` packed then `[w1][w2][w3]` scales. Only the pointer set alternates; no layout change, so the kernel's `p1..p3/s1..s3` pointer math in `RunOnDemand` is unchanged — it just points into the current slot's set.

### 3.2 Two-stream overlap inside the slot loop

**Correction (architecturally critical):** on a SINGLE CUDA stream every
enqueued op is serialized — `cudaMemcpyAsync(s+1)` issued after
`RunGlmExpertSlot(s)` starts only when the GEMM completes (no overlap); placing
it before would only delay the GEMM. Real H2D/compute overlap requires TWO
independent streams with cross-stream events:

### 3.3 The eviction barrier

The H2D for `s+1` runs on `stream_copy` while the GEMM of `s` runs on `stream`
— independent queues overlap in hardware. The event before the GEMM of a set
binds that the copy into it finished; the GEMM-completion event guards the copy
stream from overwriting a set the compute stream is still reading.

### 3.3 The eviction barrier sits on the copy stream

`expert_store.h::EnsureResident` syncs before a munmap of a host mmap a prior
async H2D may still read. Under two-stream it must NOT drain the compute stream:

- `EnsureResident(int, int, cudaStream_t ev_stream, cudaEvent_t evict_done = nullptr)`
  — when `evict_done != nullptr`, the eviction site `cudaEventRecord(evict_done, ev_stream)`
  then `cudaStreamWaitEvent(ev_stream, evict_done, 0)` instead of
  `cudaStreamSynchronize`. `nullptr` keeps the legacy full-sync (DeepSeek unchanged).
- The GLM two-stream caller passes `(stream, ev_evict)` where `stream` is the
  COMPUTE stream and `ev_evict` is recorded on the COPY stream after the last
  H2D into a set — so the munmap of that set's source mmap is released after the
  copy actually finished reading it, without draining the compute stream.

Numerics untouched: FP order, `RunGlmExpertSlot` GEMMs, `-fmad=false` identical;
only the stream delivering bytes and the wait mechanism change. Bit-exactness of
selfchecks is the acceptance metric (§4).

## 4. Acceptance gate (from strategy spec §5)

1. `test_glm_selfcheck` — 3-cycle `final_logits` byte-identical, `"Paris"`. (SKIP w/o checkpoint acceptable; must be bit-exact if model is present on `D:\Programs\lsx_model`.)
2. `test_glm_experts`, `test_glm_moe_compute`, `test_glm_model`, `test_expert_store` → unchanged PASS.
3. Occupancy/throughput: measure that warps/SM ≤48 and no throughput regression on the on-demand MoE path. If the machinery can't be benchmarked in CI without a model, reason from the overlap math (bytes moved / GEMM time) — document expected gain.
4. No plumbing regression: `rg "struct ExpertDb|struct GlmExpertDb"` stays empty; DeepSeek loader + store tests still pass.

## 5. Files

- `libs/lsxcommon/include/lsxcommon/glm_inference.h` — expand `RunGlmExpertsStackOnDemand` to accept the second stage set AND a copy stream + events.
- `libs/lsxcommon/src/glm_inference.cu` — the on-demand caller allocates two stage sets, a `stream_copy`, and `ev_h2d[2]`/`ev_gemm[2]`/`ev_evict`; passes them through `RunGlmGenerationImpl` → `GlmLoopCtx` → `RunLoopExpertStack`.
- `libs/lsxcommon/src/glm_inference_load.cu` — two-stream slot loop: H2D on `stream_copy`, GEMM on `stream`, cross-stream waits via events.
- `libs/lsxcommon/src/expert_store.h` — `EnsureResident(int, int, cudaStream_t, cudaEvent_t evict_done = nullptr)` (already landed in Task 1 of the plan); eviction releases via event. DeepSeek passes none → unchanged.

(No separate event-store file; events owned by the GLM staging caller.)

## 6. Out of scope

- DeepSeek MoE (already batched top-K in `pipeline.cu:572-574`; keep separate).
- Any change to GEMM/FP64 numerics or `-fmad=false`.
- Store data-structure or eviction-policy change (only the *release mechanism* of an already-scheduled eviction becomes event-based, via the optional `evict_done` parameter).

## Self-review

Sections §1-6 present. Design is consistent: §3.2 two-stream overlap (compute + copy streams, cross-stream events), §3.3 event-based eviction on the copy stream. §4 gate draws from the strategy's acceptance criteria. §5 file list matches the design. Contradictions resolved: the earlier "same-stream ordering suffices" claim was corrected in §3.2 — single-stream serializes, so two streams are required for overlap.