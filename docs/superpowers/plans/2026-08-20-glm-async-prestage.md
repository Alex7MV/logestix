# GLM MoE Async Pre-Staging — Implementation Plan (two-stream)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap slot `s+1`'s H2D staging with slot `s`'s `RunGlmExpertSlot` compute in the GLM on-demand MoE path via TWO CUDA streams (compute + copy), double-buffered staging, and cross-stream events — bit-exact output preserved.

**Architecture:** On a single stream H2D and GEMM serialize, so a dedicated copy stream runs the six `cudaMemcpyAsync` per slot while the existing compute stream runs the GEMMs. Two disable-timing events per set (`ev_h2d[2]`, `ev_gemm[2]`) hand the staging buffers between streams. Staging is double-buffered (set A/B alternate by slot parity). Eviction safety: `ensure_glm_expert_resident` is called with the copy stream, so a pre-munmap barrier drains only copy-stream work, never compute. Legacy single-set callers pass `nullptr` and keep the exact single-stream loop.

**Tech Stack:** C++20, CUDA `cudaStreamCreateWithFlags/NonBlocking`, `cudaEventCreateWithFlags(DisableTiming)`, `cudaEventRecord`, `cudaStreamWaitEvent`, Arrow mmap LRU (`ExpertStore<T>`), Catch2 v3.7.0.

---

## File map

- (Done) `libs/lsxcommon/include/lsxcommon/expert_store.h`, (Done) `tests/test_expert_store.cpp` — Task 1 (committed `7faf999`).
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- No CMake change.

## Grounding

- `RunGlmExpertsStackOnDemand` body `glm_inference_load.cu:692-824`. Constants `kPackPerMat`,`kScalePerMat` (701-703); validation (706); `slots` (712); routing-grid D2H barrier + `cudaStreamSynchronize` (718-742); staging pointers (744-752); slot loop (756-820); `own_staging` free (822-823).
- On-demand gen alloc `d_pack/d_scale` at `glm_inference.cu:708-722`; sizes `pack_bytes = kMatEl/2*3`, `scale_bytes = kMatEl/kGlmNvfp4Block*3` (703-707); freed at 726-727.
- `RunGlmGenerationImpl` signature `glm_inference.cu:434-443`. `GlmLoopCtx` members `d_pack_stage/d_scale_stage` 256-257; ctx-fill 546-557. `RunLoopExpertStack` 294-305.
- Real-gen caller `glm_inference.cu:1193` passes `c.d_pack, c.d_scale, c.h_routing` (single pair).
- Entry fields: `rs.eid`, `rs.coeff`, `en.host_w1/w2/w3`, `en.host_w1_scales/w2_scales/w3_scales`, `en.w1_bytes/w2_bytes/w3_bytes`, `en.w1_scales_bytes/.../w3_scales_bytes`, `en.w1_global_scale/.../w3_global_scale`.

**Bit-exactness rule:** no FP reorder, no FMA change, no `-fmad=false` touch. Only the stream carrying bytes and wait timing change. Fallback path is byte-identical to existing behavior.

---

## Task 1: (DONE) ExpertStore optional eviction release event

Committed `7faf999`. `EnsureResident(int,int,cudaStream_t,cudaEvent_t evict_done=nullptr)`; event when non-null, else sync. `test_expert_store` 33/3 green. No further change in this plan.

---

## Task 2: Allocate copy stream, events, and stage set B (on-demand gen)

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu`

- [ ] **Step 1: Add `d_pack_b`/`d_scale_b` after the existing staging malloc**

In the on-demand overload, immediately after `d_scale`'s successful cudaMalloc:

```cpp
    uint8_t *d_pack_b = nullptr;
    uint8_t *d_scale_b = nullptr;
    e = cudaMalloc(&d_pack_b, pack_bytes);
    if (e != cudaSuccess) {
        cudaFree(d_pack); cudaFree(d_scale);
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
```

- [ ] **Step 2: Create the copy stream and events**

After the staging malls:

```cpp
    cudaStream_t stream_copy = nullptr;
    cudaEvent_t ev_h2d[2]  = {nullptr, nullptr};
    cudaEvent_t ev_gemm[2] = {nullptr, nullptr};
    e = cudaStreamCreateWithFlags(&stream_copy, cudaStreamNonBlocking);
    if (e != cudaSuccess) {
        cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_pack_b); cudaFree(d_scale_b);
        result = GlmGenerationResult{};
        result.error_msg = cudaGetErrorString(e);
        return e;
    }
    for (int i = 0; i < 2; ++i) {
        e = cudaEventCreateWithFlags(&ev_h2d[i], cudaEventDisableTiming);
        if (e != cudaSuccess) {
            for (int j = 0; j < i; ++j) { cudaEventDestroy(ev_h2d[j]); cudaEventDestroy(ev_gemm[j]); }
            cudaStreamDestroy(stream_copy);
            cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_pack_b); cudaFree(d_scale_b);
            result = GlmGenerationResult{};
            result.error_msg = cudaGetErrorString(e);
            return e;
        }
        e = cudaEventCreateWithFlags(&ev_gemm[i], cudaEventDisableTiming);
        if (e != cudaSuccess) {
            for (int j = 0; j <= i; ++j) { cudaEventDestroy(ev_h2d[j]); if (j < i) cudaEventDestroy(ev_gemm[j]); }
            cudaStreamDestroy(stream_copy);
            cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_pack_b); cudaFree(d_scale_b);
            result = GlmGenerationResult{};
            result.error_msg = cudaGetErrorString(e);
            return e;
        }
    }
```

- [ ] **Step 3: Pass into `RunGlmGenerationImpl` and destroy on exit**

```cpp
    e = RunGlmGenerationImpl(cublas_h, stream, pw, nullptr, &expert_db,
                             d_pack, d_scale, d_pack_b, d_scale_b,
                             stream_copy, ev_h2d, ev_gemm,
                             prompt, max_new, result, sampler);
    for (int i = 0; i < 2; ++i) { cudaEventDestroy(ev_h2d[i]); cudaEventDestroy(ev_gemm[i]); }
    cudaStreamDestroy(stream_copy);
    cudaFree(d_scale_b); cudaFree(d_pack_b); cudaFree(d_scale); cudaFree(d_pack);
    return e;
```

Note: `RunGlmGenerationImpl` has the extended signature from Task 3; implement Tasks 2/3 edit passes in a single coherent edit so the intermediate compiles.

- [ ] **Step 4: Commit**

`git add libs/lxcommon/src/glm_inference.cu`
`git commit -m "feat: GLM on-demand gen allocates copy stream, events, stage set B"`

---

## Task 3: Thread copy stream + events + stage B through impl/ctx/caller

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu`

- [ ] **Step 1: Extend `RunGlmGenerationImpl` signature** (434-443)

```cpp
static cudaError_t RunGlmGenerationImpl(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw,
    const GlmExpertTables *experts, GlmExpertDb *expert_db,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    uint8_t *d_pack_stage_b, uint8_t *d_scale_stage_b,
    cudaStream_t stream_copy, cudaEvent_t ev_h2d[2], cudaEvent_t ev_gemm[2],
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler())
```

- [ ] **Step 2: Extend `GlmLoopCtx` (256-279)**

After `d_scale_stage` add:

```cpp
    uint8_t *d_pack_stage_b = nullptr;
    uint8_t *d_scale_stage_b = nullptr;
    cudaStream_t stream_copy = nullptr;
    cudaEvent_t ev_h2d[2]  = {nullptr, nullptr};
    cudaEvent_t ev_gemm[2] = {nullptr, nullptr};
```

In the `on_demand` branch (546-549) add:

```cpp
        ctx.d_pack_stage_b = d_pack_stage_b;
        ctx.d_scale_stage_b = d_scale_stage_b;
        ctx.stream_copy = stream_copy;
        ctx.ev_h2d[0] = ev_h2d[0];   ctx.ev_h2d[1] = ev_h2d[1];
        ctx.ev_gemm[0] = ev_gemm[0]; ctx.ev_gemm[1] = ev_gemm[1];
```

- [ ] **Step 3: Update the device-table caller (425-431)**

```cpp
    return RunGlmGenerationImpl(cublas_h, stream, pw, &experts, nullptr,
                                nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr,
                                prompt, max_new, result, sampler);
```

- [ ] **Step 4: Update `RunLoopExpertStack` on-demand branch (296-305)**

```cpp
        return RunGlmExpertsStackOnDemand(
            c.ch, c.st, c.stream_copy, c.d_routing, 1, c.d_normed,
            *c.db, l, kGlmNumRouted, c.d_moe, c.d_scratch_e,
            c.d_pack_stage, c.d_scale_stage,
            c.d_pack_stage_b, c.d_scale_stage_b,
            c.ev_h2d, c.ev_gemm, c.h_routing);
```

- [ ] **Step 5: Compile**

`cmake --build build --config Debug --target lsxcommon_glm_cu` until clean (header + loader signature must agree — see Task 4 Step 1).

- [ ] **Step 6: Commit**

`git add libs/lsxcommon/src/glm_inference.cu`
`git commit -m "refactor: thread copy stream, events, stage B through GLM on-demand gen"`

---

## Task 4: Two-stream slot loop + single-stream fallback

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu`
- Modify: `libs/lsxcommon/src/glm_inference.cu`

- [ ] **Step 1: Header signature (210-218)**

```cpp
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,
    cudaStream_t stream_copy,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    uint8_t *d_pack_stage_b, uint8_t *d_scale_stage_b,
    cudaEvent_t ev_h2d[2], cudaEvent_t ev_gemm[2],
    GlmRoutedSlot *h_routing_staging = nullptr);
```

- [ ] **Step 2: Rewrite the loader tail (744-823) as dual-path**

```cpp
    const bool two = stream_copy != nullptr && d_pack_stage_b && d_scale_stage_b &&
                     ev_h2d && ev_gemm;
    uint8_t *sets_p[2] = { d_pack_stage, d_pack_stage_b };
    uint8_t *sets_s[2] = { d_scale_stage, d_scale_stage_b };

    auto stage_slot = [&](int si, cudaStream_t cst) -> cudaError_t {
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
        const int si_p = (si & 1);
        uint8_t *p1 = sets_p[si_p], *p2 = p1 + kPackPerMat, *p3 = p2 + kPackPerMat;
        uint8_t *s1 = sets_s[si_p], *s2 = s1 + kScalePerMat, *s3 = s2 + kScalePerMat;
        const void *src[6] = { en.host_w1, en.host_w2, en.host_w3,
                               en.host_w1_scales, en.host_w2_scales, en.host_w3_scales };
        void *dst[6] = { p1, p2, p3, s1, s2, s3 };
        const size_t nb[6] = { (size_t)kPackPerMat, (size_t)kPackPerMat, (size_t)kPackPerMat,
                               (size_t)kScalePerMat, (size_t)kScalePerMat, (size_t)kScalePerMat };
        for (int f = 0; f < 6; ++f) {
            cudaError_t ce = cudaMemcpyAsync(dst[f], src[f], nb[f],
                                             cudaMemcpyHostToDevice, cst);
            if (ce != cudaSuccess) return ce;
        }
        return cudaSuccess;
    };

    cudaError_t last = cudaSuccess;

    if (!two) {
        // Single-stream fallback: identical to the original loop.
        for (int s = 0; s < slots; ++s) {
            const GlmRoutedSlot &rs = h_routing[s];
            if (rs.eid < 0 || rs.eid >= n_experts) continue;
            ensure_glm_expert_resident(db, layer, rs.eid, stream);
            const size_t flat = static_cast<size_t>(layer) *
                                    static_cast<size_t>(db.experts_per_layer) +
                                static_cast<size_t>(rs.eid);
            if (flat >= db.entries.size()) continue;
            const GlmExpertEntry &en = db.entries[flat];
            if (!en.host_w1 || !en.host_w2 || !en.host_w3 ||
                !en.host_w1_scales || !en.host_w2_scales || !en.host_w3_scales ||
                en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
                en.w3_bytes != kPackPerMat ||
                en.w1_scales_bytes != kScalePerMat ||
                en.w2_scales_bytes != kScalePerMat ||
                en.w3_scales_bytes != kScalePerMat)
                continue;
            last = stage_slot(s, stream);
            if (last != cudaSuccess) break;
            GlmExpertDeviceBlock eb;
            eb.d_w1_packed = sets_p[0]; eb.d_w1_scales = sets_s[0];
            eb.w1_global_scale = en.w1_global_scale;
            eb.w1_packed_bytes = kPackPerMat; eb.w1_scale_bytes = kScalePerMat;
            eb.d_w2_packed = sets_p[0] + kPackPerMat; eb.d_w2_scales = sets_s[0] + kScalePerMat;
            eb.w2_global_scale = en.w2_global_scale;
            eb.w2_packed_bytes = kPackPerMat; eb.w2_scale_bytes = kScalePerMat;
            eb.d_w3_packed = sets_p[0] + 2 * kPackPerMat; eb.d_w3_scales = sets_s[0] + 2 * kScalePerMat;
            eb.w3_global_scale = en.w3_global_scale;
            eb.w3_packed_bytes = kPackPerMat; eb.w3_scale_bytes = kScalePerMat;
            const int t = s / kGlmMoETopK;
            const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
            __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
            last = RunGlmExpertSlot(cublas_h, stream, eb, x_t, out_t,
                                    d_scratch, rs.coeff);
            if (last != cudaSuccess) break;
        }
        if (own_staging) cudaFreeHost(h_routing);
        return last;
    }

    // Two-stream pipeline. Slot 0 staged on the compute stream so its GEMM
    // needs no cross-stream wait. For s>0 the copy stream staged slot s during
    // iteration s-1 and recorded ev_h2d[s&1]; the compute stream waits it before
    // the GEMM. The copy producer for slot s+1 waits ev_gemm[(s+1)&1] (the GEMM
    // that last read that set) before overwriting.
    cudaError_t ef = stage_slot(0, stream);
    if (ef != cudaSuccess) { if (own_staging) cudaFreeHost(h_routing); return ef; }

    bool set_seen[2] = {false, false};
    for (int s = 0; s < slots; ++s) {
        const GlmRoutedSlot &rs = h_routing[s];
        if (rs.eid < 0 || rs.eid >= n_experts) continue;
        ensure_glm_expert_resident(db, layer, rs.eid, stream_copy);
        const int this_set = (s & 1);

        if (s > 0)
            cudaStreamWaitEvent(stream, ev_h2d[this_set], 0);

        if (s + 1 < slots) {
            const int nxt = (s + 1) & 1;
            // First write of a set has no prior GEMM to wait on (the set was
            // never read); skip the wait until the set has been used once.
            if (set_seen[nxt])
                cudaStreamWaitEvent(stream_copy, ev_gemm[nxt], 0);
            cudaError_t pe = stage_slot(s + 1, stream_copy);
            if (pe != cudaSuccess) { last = pe; break; }
            cudaEventRecord(ev_h2d[nxt], stream_copy);
        }

        const size_t flat = static_cast<size_t>(layer) *
                                static_cast<size_t>(db.experts_per_layer) +
                            static_cast<size_t>(rs.eid);
        if (flat >= db.entries.size()) continue;
        const GlmExpertEntry &en = db.entries[flat];
        if (en.w1_bytes != kPackPerMat || en.w2_bytes != kPackPerMat ||
            en.w3_bytes != kPackPerMat || en.w1_scales_bytes != kScalePerMat ||
            en.w2_scales_bytes != kScalePerMat || en.w3_scales_bytes != kScalePerMat)
            continue;

        GlmExpertDeviceBlock eb;
        eb.d_w1_packed = sets_p[this_set];  eb.d_w1_scales = sets_s[this_set];
        eb.w1_global_scale = en.w1_global_scale;
        eb.w1_packed_bytes = kPackPerMat; eb.w1_scale_bytes = kScalePerMat;
        eb.d_w2_packed = sets_p[this_set] + kPackPerMat; eb.d_w2_scales = sets_s[this_set] + kScalePerMat;
        eb.w2_global_scale = en.w2_global_scale;
        eb.w2_packed_bytes = kPackPerMat; eb.w2_scale_bytes = kScalePerMat;
        eb.d_w3_packed = sets_p[this_set] + 2 * kPackPerMat; eb.d_w3_scales = sets_s[this_set] + 2 * kScalePerMat;
        eb.w3_global_scale = en.w3_global_scale;
        eb.w3_packed_bytes = kPackPerMat; eb.w3_scale_bytes = kScalePerMat;

        const int t = s / kGlmMoETopK;
        const __half *x_t = d_act + static_cast<size_t>(t) * kGlmHidden;
        __half *out_t = d_out + static_cast<size_t>(t) * kGlmHidden;
        last = RunGlmExpertSlot(cublas_h, stream, eb, x_t, out_t,
                                d_scratch, rs.coeff);
        if (last != cudaSuccess) break;
        cudaEventRecord(ev_gemm[this_set], stream);
        set_seen[this_set] = true;
    }

    if (own_staging) cudaFreeHost(h_routing);
    return last;
```

Notes for the engineer:
- `eb.d_w1_packed` etc. and `GlmExpertDeviceBlock` match the real type (`glm_inference.h:65-81`). `RunGlmExpertSlot` takes `const GlmExpertDeviceBlock&`.
- `kGlmMoETopK`, `kGlmHidden` already in use in the file.
- `set_seen[2]` guards the first producer write of each set (no prior GEMM to wait on). Already folded into the code above.

- [ ] **Step 3: Update the real-gen caller (1193) to pass nulls for the new args**

```cpp
    e = RunGlmExpertsStackOnDemand(c.ch, c.st, nullptr, c.d_routing, 1, c.d_mlpn,
                                   *c.db, l, kGlmNumRouted, c.d_routed,
                                   c.d_scratch_e, c.d_pack, c.d_scale,
                                   nullptr, nullptr, nullptr, nullptr,
                                   c.h_routing);
```

- [ ] **Step 4: Build + run GLM tests**

`cmake --build build --config Debug --target test_glm_model test_glm_loader` → run each. Expect loader 71/3, model 16/2; selfcheck SKIPs w/o checkpoint.

- [ ] **Step 5: Commit**

`git add libs/lsxcommon/include/lsxcommon/glm_inference.h libs/lsxcommon/src/glm_inference_load.cu libs/lsxcommon/src/glm_inference.cu`
`git commit -m "feat: two-stream async pre-staging in GLM on-demand MoE staging"`

---

## Task 5: Bit-exact verification

- [ ] If `D:\Programs\lsx_model` has a GLM-5.2 checkpoint: run `test_glm_selfcheck.exe` → 3-cycle `final_logits` byte-identical, `"Paris"`. Else SKIP accepted.
- [ ] Full build clean; `ctest --preset windows-debug -R "expert_store|glm_model|glm_loader|glm_moe_compute"` → pass.
- [ ] Commit if files changed.

## Self-review

- Spec §3.2 two-stream → Task 4 loop (stage_slot on copy stream, GEMM on compute stream, per-set events + `set_seen` guard). ✓
- Spec §3.3 eviction on copy stream → `ensure_glm_cg_expert_resident(..., stream_copy)`. ✓
- Spec §4 acceptance → Task 5. ✓
- Names consistent across tasks: `stream_copy`, `ev_h2d[2]`, `ev_gemm[2]`, `d_pack_stage_b`, `d_scale_stage_b`, `sets_p/sets_s`, `stage_slot`, `this_set`, `nxt`, `set_seen`. Real-gen caller passes nulls.
- The `two` branch is complete (`set_seen` guard folded in). All identifiers (`stream_copy`, `ev_h2d`, `ev_gemm`, `sets_p`, `sets_s`, `stage_slot`, `GlmExpertDeviceBlock`) match the current baseline; the real-gen caller passes nulls.
