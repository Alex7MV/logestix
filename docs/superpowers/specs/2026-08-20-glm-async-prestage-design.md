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
2. **Staging is single-buffered.** `RunGlmExpertsStackOnDemand` takes one `d_pack_stage`/`d_scale_stage` set (the header requires `>= one expert`). The next slot's H2D **cannot start** until `RunGlmExperts` for slot `s` has finished reading the same staging buffer (data hazard), so the H2D of slot `s+1` cannot overlap slot `s`'s compute.

The compute per slot (`RunGlmExpertsSlot` — three FP64 GEMMs) is far heavier than the H2D (a few MiB). Overlapping the next slot's mmap+H2D with the current slot's GEMM removes a serialization bubble at low risk. Blackwell also favors keeping the DMA engine fed ahead of compute.

## 2. Goal / non-goal

**Goal:** overlap slot `s+1`'s mmap-residency + H2D staging with slot `s`'s `RunGlmExpertsSlot` compute, using a **double-buffered staging** and **event-based wait** instead of the full `cudaStreamSynchronize` — while keeping the output byte-for-byte identical.

**Non-goal:** no change to the dequant / GEMM kernels (`GlmDequantNvfp4Kernel`, `GlmGemmFp64Kernel`, `GlmExpertAccumKernel`), no change to numbers, no new algorithm, no change to the store's data structures.

## 3. Design

### 3.1 Double-buffered staging

`RunGlmExpertsStackOnDemand` currently receives `d_pack_stage`/`d_scale_stage` (single set, sized `>= 1 expert` — `glm_inference.h:206-208`). We add a flag to request a second set and switch pointers per slot group:

```cpp
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage_a, uint8_t *d_pack_stage_b,
    uint8_t *d_scale_stage_a, uint8_t *d_scale_stage_b,
    GlmRoutedSlot *h_routing_staging = nullptr);
```

- The **`GLM` caller `glm_inference.cu:299` ** allocates `2×` (the config/fabric arena path stays 1× unless it changes; caller passes both sets).
- The loop alternates `buf = (s & 1) ? b : a` so slot `s` reads from set `(s&1)` while set `(s+1)&1` is being H2D'd.

Buffer layout (bytes) as today: `[w1][w2][w3]` packed then `[w1][w2][w3]` scales. Only the pointer set alternates; no layout change, so the kernel's `p1..p3/s1..s3` pointer math in `RunOnDemand` is unchanged — it just points into the current slot's set.

### 3.2 Overlap inside the slot loop

```
for (slot s):
    if s+1 < slots:
        eidNext = h_routing[s+1].eid
        next_set = (s+1) & 1
        ensure_resident(db, layer, eidNext, stream)   // now WITHOUT sync (see 3.3)
        if resident && bytes valid:
            issue 6 cudaMemcpyAsync(next_set, ...)     // enqueue H2D for s+1 NOW
    // release current slot staging: wait until the previously-enqueued H2D
    // destined for THIS slot's set has completed (cudaEvent or stream-ordered ok
    // since same stream — the H2D for THIS set was enqueued last iteration)
    e = RunGlmExpertsSlot(cublas_h, stream, eb_current, ...)  // uses (s&1) set, already staged
```

Because every H2D and `RunGlmExpertsSlot` run on the **same stream**, ordering is guaranteed by the stream itself: the GPU executes the `cudaMemcpyAsync` enqueued for slot `s+1` *after* the `RunGlmExpertsSlot` of slot `s` enqueued before it. Distinct buffers (double-buffer) prevent the write-set from colliding with the read-set. No explicit event is required to *order* H2D-vs-GEMM; events appear only for the eviction barrier (§3.3).

### 3.3 The eviction barrier

`expert_store.h::EnsureResident` calls `cudaStreamSynchronize(stream)` only when an eviction (munmap of a buffer whose async H2D may still be in flight) is about to happen. That sync is **correct and required** — a munmap while a `cudaMemcpyAsync` reads that host mmap is a use-after-free. The optimization is NOT to remove it, but to **move the sync so it does not stall the compute-H2D overlap**:

- Double-buffering means a cold-miss eviction for slot `s+1` (a buffer being written) **falls in a different set** than the one slot `s`'s GEMM is reading. The eviction barrier for the *next-set* H2D can be ordered on the stream **after** `RunGlmExpertsSlot(s)` has enqueued its GEMM, so the sync only drains outstanding async work *before* the munmaped buffer is released — never blocking the current compute.
- Concretely: enqueue slot `s+1`'s H2D, then (if an eviction will target a buffer `RunGlmExpertsSlot(s)` may still be reading) record a `cudaEvent` after that GEMM launch and `cudaStreamWaitEvent` before the munmap. This releases the barrier *without* `cudaStreamSynchronize` draining the whole stream.

So events DO appear, but only as the *eviction release*, not to order the H2D-vs-GEMM (which the stream already orders). §3.2 above is then the double-buffer overlap; §3.3 is the safe eviction under overlap.

## 4. Acceptance gate (from strategy spec §5)

1. `test_glm_selfcheck` — 3-cycle `final_logits` byte-identical, `"Paris"`. (SKIP w/o checkpoint acceptable; must be bit-exact if model is present on `D:\Programs\huggingface.co`.)
2. `test_glm_experts`, `test_glm_moe_compute`, `test_glm_model`, `test_expert_store` → unchanged PASS.
3. Occupancy/throughput: measure that warps/SM ≤48 and no throughput regression on the on-demand MoE path. If the machinery can't be benchmarked in CI without a model, reason from the overlap math (bytes moved / GEMM time) — document expected gain.
4. No plumbing regression: `rg "struct ExpertDb|struct GlmExpertDb"` stays empty; DeepSeek loader + store tests still pass.

## 5. Files

- `libs/lsxcommon/include/lsxcommon/glm_inference.h` — expand the `RunGlmExpertsStackOnDemand` signature to accept two stage sets (`d_pack_stage_a/b`, `d_scale_stage_a/b`).
- `libs/lsxcommon/src/glm_inference_load.cu` — restructure the on-demand slot loop: pre-stage slot `s+1`'s `ensure_glm_expert_resident` + 6 H2D into the alternate set, alternate per slot, and place the eviction `cudaEvent` release where the munmap is scheduled.
- `libs/lsxcommon/src/glm_inference.cu` — update the on-demand caller to allocate and pass two stage sets.
- `libs/lsxcommon/src/expert_store.h` — `EnsureResident(int, int, cudaStream_t, cudaEvent_t evict_done = nullptr)` gains an **optional** release event. When non-null, the eviction path waits via `cudaEventRecord(evict_done, stream)` then `cudaStreamWaitEvent`, instead of `cudaStreamSynchronize`. DeepSeek call sites pass none → unchanged behavior. Source-compatible additive change; data layout untouched.
- No CMake change.

(No separate event-store file; `cudaEvent_t` is owned by the GLM staging caller and passed in.)

## 6. Out of scope

- DeepSeek MoE (already batched top-K in `pipeline.cu:572-574`; keep separate).
- Any change to GEMM/FP64 numerics or `-fmad=false`.
- Store data-structure or eviction-policy change (only the *release mechanism* of an already-scheduled eviction becomes event-based, via the optional `evict_done` parameter).

## Self-review

Sections §1-6 present; no placeholders. Design is consistent: §3.2 double-buffer overlap, §3.3 event-based eviction release (stream orders H2D-vs-GEMM; events only guard munmap). §4 gate draws from the strategy's acceptance criteria. §5 file list matches the design (no `EnsureResidentAsync`; the store's `EnsureResident` gains an optional `evict_done` event only — DeepSeek unaffected). Contradictions checked and resolved: the earlier "no explicit events" language was reconciled with §3.3.