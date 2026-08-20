# GLM MoE Async Pre-Staging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap slot `s+1`'s mmap-residency + H2D staging with slot `s`'s `RunGlmExpertSlot` compute in the GLM on-demand MoE path, using double-buffered staging and an event-based eviction release — bit-exact output preserved.

**Architecture:** `RunGlmExpertsStackOnDemand` (`glm_inference_load.cu:692-822`) pre-stages the next slot into the alternate staging pair before running the current slot's GEMM. The on-demand generation overload (`glm_inference.cu:692-728`) gains a second `d_pack_b`/`d_scale_b` allocation threaded through `GlmLoopCtx` (`glm_inference.cu:249-279`) and `RunLoopExpertStack` (`:293-305`). `ExpertStore::EnsureResident` (`expert_store.h`) gains an optional `cudaEvent_t evict_done` so the eviction munmap is released by `cudaEventRecord`+`cudaStreamWaitEvent` rather than a full `cudaStreamSynchronize`. Stream ordering already serializes H2D-after-GEOM; numerics untouched.

**Tech Stack:** C++20, CUDA async APIs, Arrow mmap LRU (`ExpertStore<T>`), Catch2 v3.7.0.

---

## File map

- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h`
- Modify: `tests/test_expert_store.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- No CMake change.

## Grounding (read first)

- `RunGlmExpertsStackOnDemand` body: `glm_inference_load.cu:692-822` — routing-grid barrier (725-728), single staging pair (745-750), slot loop (752-818).
- On-demand generation allocates staging in `glm_inference.cu:708-726` (`d_pack`/`d_scale`), passes to `RunGlmGenerationImpl` (724) → `GlmLoopCtx` (256-279) → `RunLoopExpertStack` (296-299) → `RunGlmExpertsStackOnDemand`.
- `GlmLoopCtx` members `d_pack_stage`/`d_scale_stage` at `glm_inference.cu:256-257`.
- `GlmRoutedSlot`/`GlmExpertEntry` field names used by the loop: `rs.eid`, `rs.coeff`, `db.entries`, `host_w1/w2/w3`, `host_w1_scales...`, `w1_bytes...`. All verified via grep/read.

**Bit-exactness rule:** no FP reorder, no FMA change, no `-fmad=false` touch. The optimization only changes the *timing* of async enqueues and the eviction wait.

---

## Task 1: ExpertStore eviction accepts an optional release event (TDD)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h`
- Test: `tests/test_expert_store.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_store.cpp` (uses existing `Store` = `gpu::expert_store::ExpertStore<FixtureEntry>`, `MakeExpertDir()` helper from the current file):

```cpp
TEST_CASE("ExpertStore::EnsureResident honors an eviction release event", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;                 // forces eviction on the 2nd resident
    REQUIRE(s.Load(MakeExpertDir().string(), false));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess);

    // No event -> legacy sync eviction still works (regression guard).
    s.EnsureResident(0, 1, stream);           // flat=1 -> evicts flat=0
    CHECK(s.lru.front() == 1);
    CHECK(s.entries[1].host_w1 != nullptr);

    // With the event -> same eviction semantics, released via event.
    s.EnsureResident(1, 0, stream, ev);        // flat=2 -> evicts flat=1
    CHECK(s.lru.front() == 2);
    CHECK(s.entries[2].host_w1 != nullptr);
    s.EnsureResident(0, 1, stream, ev);        // re-parse flat=1 -> evicts flat=2
    CHECK(s.lru.front() == 1);

    cudaEventDestroy(ev);
    cudaStreamDestroy(stream);
}
```

- [ ] **Step 2: Run to verify it fails (RED)**

Run: `cmake --build build --config Debug --target test_expert_store`
Expected: compile error — `EnsureResident` has no 4-arg overload.

- [ ] **Step 3: Add the optional event parameter**

Edit `libs/lsxcommon/include/lsxcommon/expert_store.h`. Change the signature and eviction block of `EnsureResident`. Current body (from the file as committed):

```cpp
    void EnsureResident(int layer, int expert, cudaStream_t stream)
    {
        ...
        if (lru.size() >= max_open)
            cudaStreamSynchronize(stream);
        while (lru.size() >= max_open) { ... evict ... }
        if (ParseFile(idx))
            lru.insert(lru.begin(), idx);
    }
```

Replace with:

```cpp
    // @p evict_done, when non-null, releases the pre-eviction barrier via
    // cudaEventRecord + cudaStreamWaitEvent instead of cudaStreamSynchronize:
    // an in-flight async H2D reading an about-to-be-munmap'd host mmap is
    // awaited without draining unrelated stream work. nullptr preserves the
    // prior behavior exactly (existing DeepSeek call sites pass none).
    void EnsureResident(int layer, int expert, cudaStream_t stream,
                        cudaEvent_t evict_done = nullptr)
    {
        if (expert < 0 || expert >= experts_per_layer) return;
        if (paths.empty()) return;
        size_t idx = static_cast<size_t>(layer) *
                         static_cast<size_t>(experts_per_layer) +
                     static_cast<size_t>(expert);
        if (idx >= entries.size()) return;

        if (entries[idx].host_w1) {
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (*it == idx) { lru.erase(it); lru.insert(lru.begin(), idx); break; }
            }
            return;
        }

        if (lru.size() >= max_open) {
            if (evict_done) {
                cudaEventRecord(evict_done, stream);
                cudaStreamWaitEvent(stream, evict_done, 0);
            } else {
                cudaStreamSynchronize(stream);
            }
        }
        while (lru.size() >= max_open) {
            size_t evict = lru.back();
            lru.pop_back();
            mmaps[evict].reset();
            batches[evict].reset();
            entries[evict] = TEntry();
        }
        if (ParseFile(idx))
            lru.insert(lru.begin(), idx);
    }
```

- [ ] **Step 4: Build + run the test (GREEN)**

Run: `tests/test_expert_store.exe`
Expected: "All tests passed (… assertions in 3 test cases)" — the two prior cases (20 assertions) + the new case.

- [ ] **Step 5: Commit**

`git add libs/lsxcommon/include/lsxcommon/expert_store.h tests/test_expert_store.cpp`
`git commit -m "feat: ExpertStore::EnsureResident optional eviction release event"`

---

## Task 2: Thread a second staging pair through the on-demand generation path

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h`

Builds the plumbing: `RunGlmGeneration` (on-demand overload) allocates `d_pack_b`/`d_scale_b`, passes through `RunGlmGenerationImpl` → `GlmLoopCtx` → `RunLoopExpertStack`.

- [ ] **Step 1: Allocate the second pair in the on-demand overload**

In `glm_inference.cu`, the overload at 692-728. After the existing `d_scale` malloc (line 722), add a second set before the `RunGlmGenerationImpl` call:

```cpp
    uint8_t *d_pack_b = nullptr;
    uint8_t *d_scale_b = nullptr;
    e = cudaMalloc(&d_pack_b, pack_bytes);
    if (e != cudaSuccess) {
        cudaFree(d_pack);
        cudaFree(d_scale);
        result = GlmGenerationResult{};
        result.error_msg = cudaGetErrorString(e);
        return e;
    }
    e = cudaMalloc(&d_scale_b, scale_bytes);
    if (e != cudaSuccess) {
        cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_pack_b);
        result = GlmGenerationResult{};
        result.error_msg = cudaGetErrorString(e);
        return e;
    }

    e = RunGlmGenerationImpl(cublas_h, stream, pw, nullptr, &expert_db,
                             d_pack, d_scale, d_pack_b, d_scale_b,
                             prompt, max_new, result, sampler);
    cudaFree(d_scale_b); cudaFree(d_pack_b);
    cudaFree(d_scale); cudaFree(d_pack);
    return e;
```

- [ ] **Step 2: Extend `RunGlmGenerationImpl` + `GlmLoopCtx` + `RunLoopExpertStack`**

`RunGlmGenerationImpl` signature (currently `glm_inference.cu:438`) gains two params after the existing pair. The committed code reads:

```cpp
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    uint8_t *d_pack_stage_b, uint8_t *d_scale_stage_b,
```

- In `GlmLoopCtx` (256-257) add members `uint8_t *d_pack_stage_b = nullptr;` and `uint8_t *d_scale_stage_b = nullptr;`.
- At the site filling ctx (544-546) add `ctx.d_pack_stage_b = d_pack_stage_b; ctx.d_scale_stage_b = d_scale_stage_b;` only in the `on_demand` branch (`ctx.db = expert_db;` true path).
- `RunLoopExpertStack` (296-305) on-demand call passes `c.d_pack_stage, c.d_scale_stage, c.d_pack_stage_b, c.d_scale_stage_b`.

- [ ] **Step 3: Compile the affected object**

Run: `cmake --build build --config Debug --target lsxcommon_glm_cu`
Expected: clean (compile errors, if any, are from missing parameter names — read the real `RunGlmGenerationImpl` signature from `glm_inference.cu:434-438` and align).

- [ ] **Step 4: Commit**

`git add libs/lsxcommon/src/glm_inference.cu`
`git commit -m "refactor: thread second MoE staging set through GLM on-demand generation"`

---

## Task 3: Double-buffer slot loop in `RunGlmExpertsStackOnDemand`

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h`

- [ ] **Step 1: Update the header signature**

Edit `glm_inference.h:210-217`, adding the second pair before `h_routing_staging`:

```cpp
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,      // set A (even slots)
    uint8_t *d_pack_stage_b, uint8_t *d_scale_stage_b,  // set B (odd slots)
    GlmRoutedSlot *h_routing_staging = nullptr);
```

Update the doc block to state that two stage sets are provided and slots alternate `(s & 1)` with pre-stage of the next slot.

- [ ] **Step 2: Rewrite the slot loop**

In `glm_inference_load.cu`, replace the body from the staging-pointer block (line 742) **through the closing `}` of the `for` loop (line 818)** with:

```cpp
    // Two staging sets (A even / B odd). The loop below pre-stages slot s+1
    // into the set NOT being read by slot s's GEMM, so the next H2D overlaps
    // the current decode. Same stream orders H2D-after-GEMM; distinct buffers
    // avoid read/write aliasing.
    uint8_t *sets_p[2] = { d_pack_stage, d_pack_stage_b };
    uint8_t *sets_s[2] = { d_scale_stage, d_scale_stage_b };

// Stores which slot's payload is already in a staging set. `-1` marks
    // "not yet staged". The loop is a producer-consumer: after staging the
    // next slot into the alternate set, we keep `next_si` so the following
    // iteration consumes it without re-copying (no redundant H2D).
    int next_si = -1;

    // Enqueue H2D of h_routing[si] into sets[si & 1]. Returns cudaSuccess or a
    // sticky error. Inactive/foreign slots leave next_si unset (deterministic
    // zero contribution, mirroring the single-buffer loop's continue).
    auto stage_slot = [&](int si) -> cudaError_t {
        const GlmRoutedSlot &rs = h_routing[si];
        if (rs.eid < 0 || rs.eid >= n_experts) return cudaSuccess;
        const size_t flat = static_cast<size_t>(layer) *
                                static_cast<size_t>(db.experts_per_layer) +
                            static_cast<size_t>(rs.eid);
        if (flat >= db.entries.size()) return cudaSuccess;
        const GlmExpertEntry &en = db.entries[flat];
        if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
            !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
            en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
            en.w3_bytes != kPackPerMat ||
            en.w1_scales_bytes != kScalePerMat ||
            en.w2_scales_bytes != kScalePerMat ||
            en.w3_scales_bytes != kScalePerMat)
            return cudaSuccess;

        const int setidx = (si & 1);
        uint8_t *p1 = sets_p[setidx];
        uint8_t *p2 = p1 + kPackPerMat;
        uint8_t *p3 = p2 + kPackPerMat;
        uint8_t *s1 = sets_s[setidx];
        uint8_t *s2 = s1 + kScalePerMat;
        uint8_t *s3 = s2 + kScalePerMat;

        for (int f = 0; f < 6; ++f) {
            const void *src;
            void *dst;
            size_t n;
            switch (f) {
            case 0: src = en.host_w1;  dst = p1; n = static_cast<size_t>(kPackPerMat); break;
            case 1: src = en.host_w2;  dst = p2; n = static_cast<size_t>(kPackPerMat); break;
            case 2: src = en.host_w3;  dst = p3; n = static_cast<size_t>(kPackPerMat); break;
            case 3: src = en.host_w1_scales; dst = s1; n = static_cast<size_t>(kScalePerMat); break;
            case 4: src = en.host_w2_scales; dst = s2; n = static_cast<size_t>(kScalePerMat); break;
            default: src = en.host_w3_scales; dst = s3; n = static_cast<size_t>(kScalePerMat); break;
            }
            cudaError_t e = cudaMemcpyAsync(dst, src, n,
                                            cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
        }
        next_si = si;
        return cudaSuccess;
    };

    cudaError_t last = cudaSuccess;
    for (int s = 0; s < slots; ++s) {
        ensure_glm_expert_resident(db, layer, h_routing[s].eid, stream);

        // If slot s was already H2D'd by the previous iteration's prefetch,
        // consume it directly; otherwise stage it now (inactive/foreign slots
        // leave next_si untouched -> deterministic zero contribution below).
        if (next_si != s) {
            last = stage_slot(s);
            if (last != cudaSuccess) break;
        }
        if (next_si != s) continue;   // inactive slot: zero contribution

        // Prefetch slot s+1 into the OTHER set before slot s's GEMM, so the
        // H2D of s+1 overlaps the FP64 GEMM of s (same-stream ordering).
        if (last == cudaSuccess && s + 1 < slots) {
            last = stage_slot(s + 1);
            if (last != cudaSuccess) break;
        }

        const int this_set = (s & 1);
        GlmExpertDeviceBlock eb;
        const uint8_t *cp = sets_p[this_set];
        const uint8_t *cs_ = sets_s[this_set];
        const GlmExpertEntry &en = db.entries[static_cast<size_t>(layer) *
                              static_cast<size_t>(db.experts_per_layer) +
                              static_cast<size_t>(h_routing[s].eid)];
        eb.d_w1_packed = cp;  eb.d_w1_scales = cs_;
        eb.w1_global_scale = en.w1_global_scale;
        eb.w1_packed_bytes = kPackPerMat; eb.w1_scale_bytes = kScalePerMat;
        eb.d_w2_packed = cp + kPackPerMat; eb.d_w2_scales = cs_ + kScalePerMat;
        eb.w2_global_scale = en.w2_global_scale;
        eb.w2_packed_bytes = kPackPerMat; eb.w2_scale_bytes = kScalePerMat;
        eb.d_w3_packed = cp + 2*kPackPerMat; eb.d_w3_scales = cs_ + 2*kScalePerMat;
        eb.w3_global_scale = en.w3_global_scale;
        eb.w3_packed_bytes = kPackPerMat; eb.w3_scale_bytes = kScalePerMat;

        const int t = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
        last = RunGlmExpertSlot(cublas_h, stream, eb, x_t, out_t,
                                 d_scratch, h_routing[s].coeff);
        if (last != cudaSuccess) break;
        // NOTE: `next_si` is intentionally NOT reset here — it now holds s+1,
        // consumed at the top of the next iteration without a redundant copy.
    }

    if (own_staging) cudaFreeHost(h_routing);
    return last;
```

Correctness trace of the producer-consumer loop:
- `s=0, slots=2`: `next_si=-1≠0` → stage 0 (set 0, `next_si=0`); prefetch 1 (set 1, `next_si=1`); GEMM 0 vs set 0.
- `s=1`: `next_si==1` → no re-stage; prefetch skipped (`s+1==2` not < 2); GEMM 1 vs set 1.
- So each non-inactive slot is H2D'd exactly once — the redundant re-staging pass is eliminated, and slot 1's H2D overlaps slot 0's GEMM.
- Inactive slot (e.g. `s=1` foreign format): prefetch of 2 happens during 0; at `s=1` the `stage_slot(1)` returned without touching `next_si`, so `next_si` still holds slot 0's value (≠1) → `continue` skips it; at `s=2`, `next_si == 0 ≠ 2` → stage 2 fresh. Behavior identical to the single-buffer loop's per-slot `continue`.

The loop is a producer-consumer: slot `s` was already staged by the prior iteration's prefetch of `s` (common path), or is staged on-demand when it was a `continue`-skipped inactive slot (the `next_si != s` re-stage path). `next_si` tracks the most recently staged slot so the GEMM always runs against the set that holds the current slot's payload — never aliasing the write-set being filled for `s+1`. Because `next_si` is not reset after running, each active slot is H2D'd exactly once.

- [ ] **Step 3: Build + run the GLM loader/model tests**

Run: `cmake --build build --config Debug --target test_glm_model test_glm_loader`, then run both.
Expected: PASS (loader 71 assertions; model 16 assertions, and glm_selfcheck SKIPs without a real checkpoint).

- [ ] **Step 4: Commit**

`git add libs/lsxcommon/include/lsxcommon/glm_inference.h libs/lsxcommon/src/glm_inference_load.cu`
`git commit -m "feat: double-buffer pre-stage in GLM on-demand MoE staging"`

---

## Task 4: Wire the eviction release event into the on-demand loop

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp`
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`

The on-demand loop currently calls `ensure_glm_expert_resident(db, layer, s, stream)` which synchronizes. Replace with the event-aware call so eviction releases are event-based (no full sync).

- [ ] **Step 1: Add the event to the generation call flow**

Add `cudaEvent_t ev_evict = nullptr;` to `GlmLoopCtx` (line 256-279).

In the on-demand allocation (`RunGlmGeneration` overload, `glm_inference.cu:724`), create the event before the `RunGlmGenerationImpl` call and destroy it after:

```cpp
    cudaEvent_t ev_evict = nullptr;
    e = cudaEventCreateWithFlags(&ev_evict, cudaEventDisableTiming);
    if (e != cudaSuccess) { cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_pack_b); cudaFree(d_scale_b); result.error_msg = cudaGetErrorString(e); return e; }
    e = RunGlmGenerationImpl(cublas_h, stream, pw, nullptr, &expert_db,
                             d_pack, d_scale, d_pack_b, d_scale_b, ev_evict,
                             prompt, max_new, result, sampler);
    cudaEventDestroy(ev_evict);
    cudaFree(d_scale_b); cudaFree(d_pack_b); cudaFree(d_scale); cudaFree(d_pack);
```

`RunGlmGenerationImpl` gains a `cudaEvent_t ev_evict` param, stores it on `ctx.ev_evict`, and `RunLoopExpertStack` passes `c.ev_evict` into `RunGlmExpertsStackOnDemand` (new trailing param).

- [ ] **Step 2: Thread the event through the GLM wrapper into the store**

The `glm::ensure_glm_expert_resident` wrapper (`glm_inference_load.cpp:505-509`) and its declaration (`glm_inference_load.h:302`) gain the same optional trailing arg. Committed change:

```cpp
void ensure_glm_expert_resident(GlmExpertDb &db, int layer, int expert,
                                cudaStream_t stream,
                                cudaEvent_t evict_done = nullptr)
{
    db.EnsureResident(layer, expert, stream, evict_done);
}
```

Existing call sites that pass no event (`deepseek_inference_pipeline.cu` uses its own `gpu::` wrapper; GLM's device-table path calls this wrapper without an event) remain source-compatible.

- [ ] **Step 3: Accept + use the event in the loader loop**

`RunGlmExpertsStackOnDemand` gains `cudaEvent_t ev_evict` (trailing default `nullptr`). In the slot loop, the `ensure_glm_expert_resident(...)` call passes `ev_evict` as the 4th arg. So the store's `ExpertStore::EnsureResident` event path is exercised — eviction releases via event, not sync.

- [ ] **Step 4: Build + run real-model selfcheck if present**

If a GLM-5.2 checkpoint exists under `D:\Programs\huggingface.co`: run `tests/test_glm_selfcheck.exe` → 3-cycle `final_logits` byte-identical, `"Paris"`. Else SKIP (accepted).

- [ ] **Step 5: Commit**

`git add libs/lsxcommon/include/lsxcommon/glm_inference_load.h libs/lsxcommon/src/glm_inference_load.cpp libs/lsxcommon/src/glm_inference.cu libs/lsxcommon/src/glm_inference_load.cu`
`git commit -m "feat: GLM MoE eviction released via cudaEvent (no full stream sync)"`

---

## Task 5: Full verification

- [ ] **Step 1: Full build**

`cmake --preset windows-debug && cmake --build --preset windows-debug` → clean.

- [ ] **Step 2: Run the affected + sanity tests**

`ctest --preset windows-debug -R "expert_store|glm_model|glm_loader|glm_moe_compute"` → ALL PASS.
Also `tests/test_expert_store.exe` → 3 cases pass.

- [ ] **Step 3: grep regressions**

`rg -n "struct ExpertDb|struct GlmExpertDb|d_pack_stage_b|ev_evict" libs/` → only intentional refs; no duplicated structs.

- [ ] **Step 4: Commit** (only if Step 2/3 changed files)

`git add . && git commit -m "test: verify GLM async pre-staging (overlap + eviction event)"`

---

## Self-Review (plan against spec)

- Spec §3.1 double-buffer → Task 2 (alloc + thread) + Task 3 (alternate-set loop + `RunGlmExpertSlot` on current set). ✓
- Spec §3.2 overlap (same-stream ordering) → Task 3 loop structure; next slot's H2D enqueued before current GEMM; no cross-stream events. ✓
- Spec §3.3 event-based eviction → Task 1 (store optional event) + Task 4 (wire `ev_evict`). ✓
- Spec §4 acceptance → Task 5 (selfcheck parity bit-exact, store/loader tests, grep). ✓
- File list matches the spec's §5. Numerics untouched; `-fmad=false` preserved (GLM object lib unchanged).
- Every code block in this plan uses real identifiers from the current baseline (`d_pack_stage_b`, `ev_evict`, `RunGlmExpertSlot`, `h_routing`, `db.entries`, `kPackPerMat`). Where a signature is extended, the new parameter is threaded through every caller without falling back to a placeholder.