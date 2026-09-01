# Kimi Split-Phase Router + Device-Side Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (subagent dispatch is FORBIDDEN by AGENTS.md — execute inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two full-drain GPU syncs in Kimi-K2.7-Code inference — per-layer router readback (F-11) and per-step greedy argmax readback — via a split-phase router handshake, device-side token feed, host-truncate EOS, and an `eos_id` sourced from the tokenizer DB instead of a hardcoded constant.

**Architecture:** `KimiRouteBegin/Finish` mirrors DeepSeek `route_token_begin/finish` (top-k kernel + async D2H into pinned staging + event record; finish = busy-poll). Decode loop feeds `d_best` into the next embed-gather on device; one bulk D2H after the loop fills `gen.tokens`; EOS is applied host-side post-hoc (first-EOS cut + `eos_terminated`).

**Tech Stack:** CUDA 13.3 (streams/events/busy-poll), Catch2 v3, CMake preset `linux-release-znver4-aocc`.

## Global Constraints

- Branch: `feat/deepseek-expert-prestage` (already checked out; all commits here).
- Build dir: `build/linux-release-znver4-aocc`. Build targets: `cmake --build build/linux-release-znver4-aocc --target <t> -j 160`.
- Subagents forbidden (AGENTS.md). All work inline.
- Bit-exactness binding: `test_kimi_moe_selfcheck` "Paris", 3 cycles byte-identical, must stay green (real weights at `/mnt/llm_models/models/Kimi-K2.7-Code`).
- Inference on real weights is resource-critical (AGENTS.md): one GPU workload at a time; check `nvidia-smi` (~33 MiB idle) before runs.
- No `std::cout`/`printf` in library code (spdlog / `lsxcommon::log`); existing `fprintf(stderr,...)` trace lines in the kimi TU are pre-existing — do not add new ones.
- Naming guard: no new files under `libs/lsxcommon/{src,include}` — all edits inside existing kimi units + tests.
- Router tie-break, top-k math, expert FFN accumulation order: untouched.
- EOS: no per-step device flag / per-step poll (spec decision); host-truncate only.

---

### Task 1: `KimiRouteBegin`/`KimiRouteFinish` split API (TDD)

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h` (declarations after `KimiRouterTopK`, ~line 29)
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu` (implementation next to `KimiRouterTopK`, ~line 173)
- Test: `tests/test_kimi_moe_router.cu`

**Interfaces:**
- Consumes: existing `KimiRouterTopK` (same TU), `cudaEvent_t` + streams.
- Produces (used by Task 2):
```cpp
// kimi_moe_kernels.h, namespace lsxcommon::kimi:
// Split-phase router handshake (F-11): Begin launches the top-k kernel,
// issues the two async D2H copies of ids/weights into the caller's PINNED
// h_ids/h_w staging, and records ev on the stream. Finish busy-polls ev
// (event-scoped — never waits for unrelated stream work) and returns once
// h_ids/h_w are valid. Bitwise-identical results to the fused
// KimiRouterTopK + cudaStreamSynchronize path.
cudaError_t KimiRouteBegin(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_top_ids, float *d_top_w,
                           int *h_ids, float *h_w,
                           cudaEvent_t ev, cudaStream_t stream);
cudaError_t KimiRouteFinish(cudaEvent_t ev);
```

- [ ] **Step 1: Write the failing test** — append to `tests/test_kimi_moe_router.cu` (reuse the existing `RouterReference` helper and includes already present in that file):

```cpp
TEST_CASE("Kimi split-phase route matches fused path bitwise",
          "[kimi][router][split]")
{
    const int n = 384, top_k = 8;
    std::vector<float> scores(n), bias(n);
    // Adversarial: duplicate score values exercise the tie-break rule; bias
    // perturbs ordering.
    for (int i = 0; i < n; ++i) {
        scores[i] = -3.0f + 0.05f * ((i * 7) % 40);   // many duplicates
        bias[i] = 0.001f * ((i * 37) % 11);
    }

    // Fused reference: device top-k + blocking sync.
    float *d_scores = nullptr, *d_bias = nullptr, *d_w_ref = nullptr;
    int *d_ids_ref = nullptr;
    cudaMalloc(&d_scores, n * sizeof(float));
    cudaMalloc(&d_bias, n * sizeof(float));
    cudaMalloc(&d_ids_ref, top_k * sizeof(int));
    cudaMalloc(&d_w_ref, top_k * sizeof(float));
    cudaMemcpy(d_scores, scores.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreate(&st);
    REQUIRE(KimiRouterTopK(d_scores, d_bias, n, top_k, d_ids_ref, d_w_ref,
                           st) == cudaSuccess);
    std::vector<int> ref_ids(top_k);
    std::vector<float> ref_w(top_k);
    cudaMemcpy(ref_ids.data(), d_ids_ref, top_k * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_w.data(), d_w_ref, top_k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(st);

    // Split path: pinned staging + event + busy-poll.
    int *h_ids = nullptr; float *h_w = nullptr;
    REQUIRE(cudaHostAlloc(&h_ids, top_k * sizeof(int),
                          cudaHostAllocDefault) == cudaSuccess);
    REQUIRE(cudaHostAlloc(&h_w, top_k * sizeof(float),
                          cudaHostAllocDefault) == cudaSuccess);
    int *d_ids2 = nullptr; float *d_w2 = nullptr;
    cudaMalloc(&d_ids2, top_k * sizeof(int));
    cudaMalloc(&d_w2, top_k * sizeof(float));
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) ==
            cudaSuccess);
    // Prime the event so a Begin->Finish pair re-records over a known state.
    cudaEventRecord(ev, st);

    REQUIRE(KimiRouteBegin(d_scores, d_bias, n, top_k, d_ids2, d_w2,
                           h_ids, h_w, ev, st) == cudaSuccess);
    // Host does unrelated "work" here in production (batch resolve/upload);
    // the event fence must not depend on anything else on the stream.
    REQUIRE(KimiRouteFinish(ev) == cudaSuccess);

    for (int k = 0; k < top_k; ++k) {
        INFO("k=" << k << " ref=" << ref_ids[k] << "/" << ref_w[k]
             << " split=" << h_ids[k] << "/" << h_w[k]);
        REQUIRE(h_ids[k] == ref_ids[k]);
        REQUIRE(h_w[k] == ref_w[k]);   // bitwise float compare
    }

    cudaEventDestroy(ev);
    cudaFreeHost(h_ids); cudaFreeHost(h_w);
    cudaFree(d_ids2); cudaFree(d_w2);
    cudaFree(d_scores); cudaFree(d_bias);
    cudaFree(d_ids_ref); cudaFree(d_w_ref);
    cudaStreamDestroy(st);
}
```

Note: the test file already declares `KimiRouterTopK` via
`lsxcommon/kimi_moe_kernels.h` include — new functions come with it.

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build/linux-release-znver4-aocc --target test_kimi_moe_router -j 160`
Expected: FAIL — `KimiRouteBegin`/`KimiRouteFinish` not declared.

- [ ] **Step 3: Implement.** In `kimi_moe_kernels.h` after `KimiRouterTopK`:

```cpp
// ── Split-phase router handshake (F-11) ────────────────────────────────────
// Begin: top-k kernel + async D2H into the caller's PINNED h_ids/h_w +
// event record on the stream. Finish: busy-poll that event (event-scoped
// fence; never drains unrelated stream work). Bitwise-identical to the
// fused KimiRouterTopK + cudaStreamSynchronize path.
cudaError_t KimiRouteBegin(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_top_ids, float *d_top_w,
                           int *h_ids, float *h_w,
                           cudaEvent_t ev, cudaStream_t stream);
cudaError_t KimiRouteFinish(cudaEvent_t ev);
```

In `kimi_moe_inference.cu` directly after the `KimiRouterTopK` definition:

```cpp
cudaError_t KimiRouteBegin(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_top_ids, float *d_top_w,
                           int *h_ids, float *h_w,
                           cudaEvent_t ev, cudaStream_t stream)
{
    if (!h_ids || !h_w || !ev || top_k <= 0)
        return cudaErrorInvalidValue;
    cudaError_t e = KimiRouterTopK(d_scores, d_bias, n_routed, top_k,
                                   d_top_ids, d_top_w, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(h_ids, d_top_ids,
                        static_cast<size_t>(top_k) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(h_w, d_top_w,
                        static_cast<size_t>(top_k) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    return cudaEventRecord(ev, stream);
}

cudaError_t KimiRouteFinish(cudaEvent_t ev)
{
    if (!ev) return cudaErrorInvalidValue;
    // Event-scoped busy-poll (round-20 pattern): returns the moment the
    // router D2H retires without draining the rest of the stream.
    cudaError_t q;
    while ((q = cudaEventQuery(ev)) == cudaErrorNotReady) {}
    return q;
}
```

- [ ] **Step 4: Run tests until green**

Run: `cmake --build build/linux-release-znver4-aocc --target test_kimi_moe_router -j 160 && build/linux-release-znver4-aocc/tests/test_kimi_moe_router "[kimi][router]"`
Expected: PASS (existing router cases + new split case).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_kernels.h libs/lsxcommon/src/kimi_moe_inference.cu tests/test_kimi_moe_router.cu
git commit -m "feat: kimi split-phase router handshake (F-11) with bitwise TDD"
```

---

### Task 2: Swap the `KimiLayerStack` call site to the split API

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu:995-1020` (`KimiLayerStack` MoE branch)

**Interfaces:**
- Consumes: `KimiRouteBegin`/`KimiRouteFinish` (Task 1).
- Produces: layer-stack router fence is event-scoped; full-drain sync at
  `:1015` gone.

- [ ] **Step 1: Add persistent event + pinned staging to `KimiLayerStack`.** Add locals at the top of the MoE branch scope (`if (lw.d_router_gate) {`), freed at every return of the branch:

```cpp
if (lw.d_router_gate) {
    // ── MoE path ──
    // Split-phase router fence state (F-11): created once per stack call;
    // freed before every return inside this branch.
    cudaEvent_t route_ev = nullptr;
    if (cudaEventCreateWithFlags(&route_ev, cudaEventDisableTiming) !=
        cudaSuccess)
        return cudaGetLastError();
    int *h_ids = nullptr;
    float *h_w = nullptr;
    if (cudaHostAlloc(&h_ids, kKimiTopK * sizeof(int),
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_w, kKimiTopK * sizeof(float),
                      cudaHostAllocDefault) != cudaSuccess) {
        cudaFreeHost(h_ids);
        cudaFreeHost(h_w);
        cudaEventDestroy(route_ev);
        return cudaErrorMemoryAllocation;
    }
```

- [ ] **Step 2: Replace the fused call + sync** (`:1004-1015`) with the split pair. The fence sits where the code FIRST reads `h_ids` (both cache and legacy branches read immediately; the overlap win comes from prior shared-expert/attention kernels still queued behind the event, not from host work between phases):

```cpp
                cudaError_t re = KimiRouteBegin(
                    d_scores + static_cast<size_t>(t) * kKimiNumRouted,
                    d_bias_all + (size_t)l * kKimiNumRouted,
                    kKimiNumRouted, kKimiTopK, d_top_ids, d_top_w,
                    h_ids, h_w, route_ev, stream);
                if (re != cudaSuccess) {
                    cudaFreeHost(h_ids); cudaFreeHost(h_w);
                    cudaEventDestroy(route_ev);
                    return re;
                }
                // Event-scoped fence (F-11): busy-polls ONLY the router
                // readback instead of draining the whole stream.
                if (KimiRouteFinish(route_ev) != cudaSuccess) {
                    cudaFreeHost(h_ids); cudaFreeHost(h_w);
                    cudaEventDestroy(route_ev);
                    return cudaGetLastError();
                }
                if (cache && cache->device && cache->plan.Active()) {
                    // Pass 1: batch residency decision (LRU bookkeeping).
                    // EXISTING CODE UNCHANGED — h_ids/h_w now come from the
                    // pinned staging instead of the stack arrays.
                    KimiMoeCacheBatch b = KimiMoeResolveBatch(
                        cache->plan, l, kKimiNumRouted, h_ids, kKimiTopK);
```

The rest of the branch (Pass 2 upload, `cudaStreamWaitEvent`, Pass 3 GEMV
loop, legacy else-branch) keeps its existing code; only `h_ids`/`h_w`
provenance changed (pinned heap arrays instead of stack arrays).

- [ ] **Step 3: Free resources at the remaining returns** of the MoE branch — the FFN-error returns in the cache path (`return fe`) and legacy path (`return fe`) must free `h_ids`/`h_w`/`route_ev` first (3-line free before each of the 2 existing returns).

- [ ] **Step 4: Free the resources at every remaining return of the MoE branch.** The GEMV loop `return fe` and the legacy-branch `return fe` must free `h_ids`/`h_w`/`route_ev` first. Wrap the tail of the branch in a small cleanup lambda if the returns multiply; minimal form: add the 3-line free before each existing `return` inside the MoE branch (2 sites: FFN error in cache path, FFN error in legacy path).

- [ ] **Step 5: Build + verify no sync remains**

Run: `cmake --build build/linux-release-znver4-aocc --target test_kimi_moe_inference -j 160 2>&1 | grep -E ': error'`
Expected: clean. Then `grep -n "cudaStreamSynchronize" libs/lsxcommon/src/kimi_moe_inference.cu`
Expected: only the greedy-decode site remains (`:1300`) — removed by Task 3.

- [ ] **Step 6: Run router + inference suites**

Run: `build/linux-release-znver4-aocc/tests/test_kimi_moe_router && ctest --test-dir build/linux-release-znver4-aocc -R "kimi" --output-on-failure 2>&1 | tail -3`
Expected: all PASS (selfcheck unchanged behavior — same fence semantics, same bytes).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_inference.cu
git commit -m "perf: kimi layer-stack router fence is event-scoped (no full drain)"
```

---

### Task 3: Device-side decode feed + host-truncate EOS

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu:1240-1335` (`RunKimiGeneration` loop)
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h:66-73` (`RunKimiGeneration` signature: `int eos_id` param)
- Modify: `libs/lsxcommon/src/kimi_moe_model.cpp:88-94, 138-145` (thread `Tokenizer().eos_id`)

**Interfaces:**
- Consumes: `d_best`/`d_bestv` (existing argmax outputs), `d_ids` (existing).
- Produces:
```cpp
// kimi_moe_kernels.h — signature change:
cudaError_t RunKimiGeneration(cublasHandle_t cublas_h, cudaStream_t stream,
                              const KimiPinnedWeights &pw, KimiExpertDb &db,
                              const std::vector<int> &prompt_ids,
                              int max_gen_tokens, KimiGenerationResult &gen,
                              KimiExpertCacheState *cache, int eos_id);
// eos_id < 0 disables EOS truncation.
```

- [ ] **Step 1: Write the failing EOS-truncate test** — append to `tests/test_kimi_moe_inference.cpp` (host-only TU; the helper is a pure function — see Step 3):

```cpp
TEST_CASE("kimi EOS host-truncate cuts at first eos and flags termination",
          "[kimi][decode][eos]")
{
    namespace k = lsxcommon::kimi;
    std::vector<int> toks = {11, 22, 163586, 33, 44};
    auto r = k::TruncateAtEos(toks, 163586);
    REQUIRE(r.tokens.size() == 2);
    REQUIRE(r.tokens[0] == 11);
    REQUIRE(r.tokens[1] == 22);
    REQUIRE(r.eos_terminated);

    std::vector<int> no_eos = {11, 22, 33};
    auto r2 = k::TruncateAtEos(no_eos, 163586);
    REQUIRE(r2.tokens.size() == 3);
    REQUIRE_FALSE(r2.eos_terminated);

    // eos_id < 0: disabled.
    auto r3 = k::TruncateAtEos(toks, -1);
    REQUIRE(r3.tokens.size() == 5);
    REQUIRE_FALSE(r3.eos_terminated);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build/linux-release-znver4-aocc --target test_kimi_moe_inference -j 160`
Expected: FAIL — `TruncateAtEos` not declared.

- [ ] **Step 3: Implement the helper + signature.** In `kimi_moe_kernels.h` next to `KimiGenerationResult`:

```cpp
// Host-side EOS application (device-side decode, F-11): cuts tokens at the
// first occurrence of eos_id and reports whether the cut happened. eos_id
// < 0 disables the cut. Pure host function, unit-testable without CUDA.
struct KimiEosCut {
    std::vector<int> tokens;
    bool eos_terminated = false;
};
KimiEosCut TruncateAtEos(const std::vector<int> &tokens, int eos_id);
```

Implementation in `kimi_moe_inference.cu` (host section, near
`RunKimiGeneration`):

```cpp
KimiEosCut TruncateAtEos(const std::vector<int> &tokens, int eos_id)
{
    KimiEosCut r;
    if (eos_id < 0) {
        r.tokens = tokens;
        return r;
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == eos_id) {
            r.tokens.assign(tokens.begin(),
                            tokens.begin() + static_cast<long>(i));
            r.eos_terminated = true;
            return r;
        }
    }
    r.tokens = tokens;
    return r;
}
```

- [ ] **Step 4: Update `RunKimiGeneration` signature + decode loop.**
  1. Signature gains trailing `int eos_id` (header + definition + both call sites in `kimi_moe_model.cpp:92` and `:141` — pass `Tokenizer().eos_id`).
  2. Replace the per-step readback block (`:1293-1306`):

```cpp
        // OLD (delete):
        //   int next_id = -1; float next_maxv = 0.0f;
        //   cudaMemcpyAsync(&next_id, d_best, ...);
        //   cudaMemcpyAsync(&next_maxv, d_bestv, ...);
        //   cudaStreamSynchronize(stream);   // greedy decode needs the id
        // NEW: device-side feed; the next embed-gather reads d_best
        // directly (kernel arg), so no id crosses to the host here.
```

  3. The per-step tail (`:1312-1330`) becomes bookkeeping-only:

```cpp
        // Every loop iteration produces one token on device (d_ids grows
        // via the device-side feed below).
        ++generated;
        // NOTE: gen.tokens/final EOS handling happen once after the loop.
```

  4. Device-side id feed: at the TOP of the loop body, before
     `EmbedGatherKernel`, replace the step>=1 host-fed branch:

```cpp
        if (step == 0) {
            ::lsxcommon::kernels::EmbedGatherKernel<__nv_bfloat16>
                <<<(int)((prefill_seq * kKimiHidden + 255) / 256), 256, 0,
                   stream>>>(pw.d_embed, d_ids, d_hidden, prefill_seq,
                             kKimiHidden);
        } else {
            // Feed the previous argmax on device: d_ids slot i <- *d_best.
            // d_ids keeps ALL ids (prompt + generated) contiguous so the
            // final bulk D2H returns the full sequence.
            kimi_feed_token_kernel<<<1, 1, 0, stream>>>(
                d_ids + prefill_seq + step - 1, d_best);
            ::lsxcommon::kernels::EmbedGatherKernel<__nv_bfloat16>
                <<<(int)((kKimiHidden + 255) / 256), 256, 0, stream>>>(
                    pw.d_embed, d_ids + prefill_seq + step - 1,
                    d_hidden + (size_t)(prefill_seq + step - 1) * kKimiHidden,
                    1, kKimiHidden);
        }
```

  with the tiny kernel next to `kimi_argmax_kernel`:

```cpp
__global__ void kimi_feed_token_kernel(int *dst, const int *src)
{
    *dst = *src;
}
```

  5. `ttft_ms`: record events instead of host wall-clock. Keep the host
     `t0` steady_clock for `total_ms` (still valid — the loop no longer
     syncs, so total_ms measures enqueue + final drain; document this in a
     comment). For ttft use two timing events around step 0's layer stack +
     head:

```cpp
    cudaEvent_t ev_start = nullptr, ev_ttft = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_ttft);
    cudaEventRecord(ev_start, stream);
```

     and after step 0's argmax kernel:

```cpp
        if (step == 0) cudaEventRecord(ev_ttft, stream);
```

     after the loop:

```cpp
    cudaEventSynchronize(ev_ttft);
    cudaEventElapsedTime(&ms_ttft, ev_start, ev_ttft);
    gen.ttft_ms = static_cast<double>(ms_ttft);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_ttft);
```

  6. `first_maxv`: add `KIMI_ALLOC`-style device float already present
     (`d_bestv`); copy it in the FINAL bulk D2H:

```cpp
    // ── Final bulk drain (the ONLY sync in the decode loop) ──────────────
    const int gen_count = generated;   // tokens actually produced on device
    std::vector<int> dev_ids(static_cast<size_t>(gen_count));
    cudaMemcpy(dev_ids.data(), d_ids + prefill_seq,
               static_cast<size_t>(gen_count) * sizeof(int),
               cudaMemcpyDeviceToHost);
    float maxv = 0.f;
    cudaMemcpy(&maxv, d_bestv, sizeof(float), cudaMemcpyDeviceToHost);
    gen.first_maxv = maxv;
    const KimiEosCut cut = TruncateAtEos(dev_ids, eos_id);
    gen.tokens = std::move(cut.tokens);
    gen.eos_terminated = cut.eos_terminated;
    gen.kv_slots_used = prefill_seq +
                        static_cast<int>(gen.tokens.size());
```

  7. Remove the old `constexpr int kKimiEosId = 163586;` block and the
     `if (next_id == kKimiEosId) {...break;}` — EOS is host-truncate now.

- [ ] **Step 5: Run tests until green**

Run: `cmake --build build/linux-release-znver4-aocc --target test_kimi_moe_inference test_kimi_moe_model -j 160 && build/linux-release-znver4-aocc/tests/test_kimi_moe_inference "[kimi][decode][eos]" && ctest --test-dir build/linux-release-znver4-aocc -R kimi --output-on-failure 2>&1 | tail -3`
Expected: EOS cases PASS; full kimi suite PASS (selfcheck "Paris" ×3 bit-exact — token stream unchanged by construction).

- [ ] **Step 6: Verify the only sync left is the final drain**

Run: `grep -n "cudaStreamSynchronize" libs/lsxcommon/src/kimi_moe_inference.cu`
Expected: no output (zero full-drain syncs in the kimi inference TU).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_kernels.h libs/lsxcommon/src/kimi_moe_inference.cu libs/lsxcommon/src/kimi_moe_model.cpp tests/test_kimi_moe_inference.cpp
git commit -m "feat: kimi device-side decode feed + host-truncate EOS (config-sourced eos_id)"
```

---

### Task 4: Perf measurement + registry Round 25

**Files:**
- Modify: `docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md` (F-11 row + Round 25 section)

- [ ] **Step 1: Before/after tok/s.** Baseline was recorded pre-change in
  Step 1 of Task 2 execution (measure BEFORE applying Task 2 if not yet
  done: `cd build/linux-release-znver4-aocc/tests && LSX_TEST_KIMI_PERF_GEN_TOKENS=16 ./test_kimi_moe_selfcheck 2>&1 | tail -2` — record
  `tok/s`, `ttft_ms`, `total_ms` from the selfcheck run log; the harness
  prints them). After Task 3, rerun the identical command and record both
  rows. GPU must be idle before each run.

- [ ] **Step 2: Write Round 25 into the registry**: shipped items (split API
  + device-side decode + EOS host-truncate + eos_id de-hardcode),
  before/after table, verdict on F-11 row (`fixed`), and the note that the
  greedy argmax drain site is gone (F-11 listed it as a secondary concern).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md
git commit -m "docs: Round 25 — kimi split-phase + device-side decode measurements"
```
