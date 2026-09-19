# Kimi RuntimeTuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (subagents are FORBIDDEN in this repository —
> AGENTS.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Kimi-K2.7-Code to full GLM parity of the adaptive expert cache:
hits/misses telemetry from the CUDA pipeline + `lifecycle::RuntimeTuner` between
requests + a governor lease on cache residency.

**Architecture:** Three layers. (1) Host helpers in `kimi_moe_expert_cache.h` —
pure inline functions `KimiCountCacheHits`/`KimiExpertCachePerExpertBytes`
(a unit test without CUDA). (2) The CUDA layer — telemetry fields in `KimiGenerationResult`,
accumulation at the `KimiMoeResolveBatch` site, an atomic slot override
`SetKimiExpertCacheSlotsOverride` + arena realloc on a mismatch (parity with
`GlmExpertCacheSlotsOverride`, glm_moe_inference.cu:29). (3) The model —
`RuntimeTuner` in `KimiK27CodeModel`, a lease via `ModelBase::
AcquireExpertCacheLease` (already exists), `Report()` between requests.

**Tech Stack:** C++20, CUDA 13.3 (Blackwell sm_120/121), Catch2 3.7.0,
ResourceGovernor (leases), lifecycle::RuntimeTuner.

**Invariants:**
- Bit-exactness: telemetry is counters only; an arena realloc changes the number of
  slots, but a slot reads the SAME packed bytes with the same kernels
  (the kimi_moe_expert_cache.h:13 contract).
- TDD: each step RED → run → GREEN → run → commit.
- Windows restriction (AGENTS.md): run ONLY host-only tests
  (`test_kimi_moe_expert_cache`, `test_model_runtime`, ...); Kimi GPU tests
  (`test_kimi_moe_*`) — only BUILD on Windows, run on Linux
  (Ubuntu 26.04, `ctest --preset linux-debug -R kimi`).

---

### Task 1: Telemetry host helpers (`KimiCountCacheHits`, `KimiExpertCachePerExpertBytes`)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h`
- Test: `tests/test_kimi_moe_expert_cache.cpp`

- [ ] **Step 1: Write the failing test**

In `tests/test_kimi_moe_expert_cache.cpp`, after the existing test cases, add:

```cpp
TEST_CASE("KimiCountCacheHits accounts residents vs misses", "[kimi][cache]")
{
    using lsxcommon::kimi::KimiCountCacheHits;
    using lsxcommon::kimi::KimiMoeResolveBatch;
    using lsxcommon::kimi::KimiMoeExpertCachePlan;
    constexpr int kEPL = 4;
    KimiMoeExpertCachePlan p;
    p.Init(2, 2 * kEPL);   // 2 slots, 8 experts

    // First batch: all miss (the cache is empty) -> 0 hits / 8 misses.
    const int ids0[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    KimiMoeCacheBatch b0 = KimiMoeResolveBatch(p, 0, kEPL, ids0, 8);
    auto [h0, m0] = KimiCountCacheHits(b0, 8);
    CHECK(h0 == 0);
    CHECK(m0 == 8);

    // Repeat: all resident -> 8 hits / 0 misses.
    KimiMoeCacheBatch b1 = KimiMoeResolveBatch(p, 0, kEPL, ids0, 8);
    auto [h1, m1] = KimiCountCacheHits(b1, 8);
    CHECK(h1 == 8);
    CHECK(m1 == 0);

    // Mixed: 4 resident + 4 eviction-miss.
    const int ids2[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    KimiMoeCacheBatch b2 = KimiMoeResolveBatch(p, 1, kEPL, ids2, 8);
    auto [h2, m2] = KimiCountCacheHits(b2, 8);
    CHECK(h2 == 4);
    CHECK(m2 == 4);

    // Inactive plan: k tokens, all miss.
    KimiMoeExpertCachePlan off;
    off.Init(0, 2 * kEPL);
    KimiMoeCacheBatch b3 = KimiMoeResolveBatch(off, 0, kEPL, ids0, 8);
    auto [h3, m3] = KimiCountCacheHits(b3, 8);
    CHECK(h3 == 0);
    CHECK(m3 == 8);
}

TEST_CASE("KimiExpertCachePerExpertBytes matches the arena stride math",
          "[kimi][cache]")
{
    // 3 pack matrices (fp16 pairs) + 3 scale blocks (fp8/16 bytes per element).
    const std::size_t expect =
        3ull * (7168ull * 2048ull / 2) +      // packed w1|w3|w2
        3ull * (7168ull * 2048ull / 16);      // scales s1|s3|s2
    CHECK(lsxcommon::kimi::KimiExpertCachePerExpertBytes() == expect);
}
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target test_kimi_moe_expert_cache
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R test_kimi_moe_expert_cache --output-on-failure
```

Expected: compile FAIL — `KimiCountCacheHits`/`KimiExpertCachePerExpertBytes`
are not members of `lsxcommon::kimi` (error C2039).

- [ ] **Step 3: Write minimal implementation**

In `libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h` after
`KimiMoeResolveBatch` (after line ~167), add:

```cpp
// Telemetry (RuntimeTuner): residents get resident_slot >= 0 at resolve
// time; everything else in the batch is a miss (upload or hard-miss).
// Returns {hits, misses} for one resolved top-k batch.
inline std::pair<std::int64_t, std::int64_t>
KimiCountCacheHits(const KimiMoeCacheBatch &b, int k)
{
    std::int64_t hits = 0;
    for (int i = 0; i < k; ++i)
        if (b.resident_slot[i] >= 0) ++hits;
    const std::int64_t misses =
        static_cast<std::int64_t>(b.missing_count);
    return {hits, misses};
}

// Worst-case per-expert device footprint of one cache slot: packed
// [w1|w3|w2] fp16-pairs + scales [s1|s3|s2] (16-element E8M0 blocks).
// Single authority shared by the model Load lease/tuner sizing and the
// lazy arena allocation in kimi_moe_inference.cu (was a hardcoded
// expression there; kKimiExpertMatEl = 7168*2048).
inline std::size_t KimiExpertCachePerExpertBytes()
{
    constexpr std::size_t kMatEl = 7168ull * 2048ull;
    return 3ull * (kMatEl / 2) + 3ull * (kMatEl / 16);
}
```

(`#include <cstdint>` is already there; add `#include <utility>` for
`std::pair` if absent.)

- [ ] **Step 4: Run test to verify it passes**

The command from Step 2. Expected: PASS, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/kimi_moe_expert_cache.h tests/test_kimi_moe_expert_cache.cpp
git commit -m "feat: kimi cache telemetry host helpers (hits/misses, per-expert bytes)"
```

---

### Task 2: Slot override + arena realloc in the CUDA layer (GLM parity)

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h`
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu` (lazy block ~lines 1170-1205,
  resolve site ~lines 1015-1055)

**Note:** on Windows compilation only; the run — Linux.

- [ ] **Step 1: Write the failing test (compile-level RED)**

The RED host part is already covered by Task 1; for the CUDA layer RED is the compilation of the new
symbol `SetKimiExpertCacheSlotsOverride`, referenced by Task 4.
Skip the run, record the expectation: before implementation, Task 4 linking
fails with LNK2019. Go straight to Step 2.

- [ ] **Step 2: Override atomics in `kimi_moe_kernels.h`**

Next to `RunKimiGeneration` (after line 75), add:

```cpp
// RuntimeTuner hook (GLM parity, glm_moe_inference.cu:29): slots >= 0
// forces the lazy arena to that slot count; -1 falls back to the env
// heuristic. Called between inference requests, never in the hot loop.
void SetKimiExpertCacheSlotsOverride(int slots);
int KimiExpertCacheSlotsOverride();
```

- [ ] **Step 3: Implementation in `kimi_moe_inference.cu`**

At the top of the file (next to the other file-scope statics), add:

```cpp
std::atomic<int> g_kimi_cache_slots_override{-1};

void SetKimiExpertCacheSlotsOverride(int slots)
{
    g_kimi_cache_slots_override.store(slots, std::memory_order_relaxed);
}

int KimiExpertCacheSlotsOverride()
{
    return g_kimi_cache_slots_override.load(std::memory_order_relaxed);
}
```

Replace the lazy allocation block (~line 1170) with: override → env →
heuristic, WITHOUT computing per_expert by hand (the Task 1 helper), and with a realloc
on a slot-count mismatch:

```cpp
    if (cache && !cache->device) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const int total_experts = kKimiNumLayers * kKimiNumRouted;
            const size_t per_expert = KimiExpertCachePerExpertBytes();
            int want = KimiExpertCacheSlotsOverride();
            if (want < 0)
                want = KimiMoeExpertCacheSlots(
                    free_b, per_expert, total_experts,
                    std::getenv("LSX_KIMI_EXPERT_CACHE_SLOTS"));
            if (want > total_experts) want = total_experts;
            if (want < 0) want = 0;
            cache->plan.Init(want, total_experts);
            std::string cerr_;
            int try_slots = want;
            while (try_slots >= 1 && !cache->device) {
                if (AllocateKimiMoeExpertCache(db, try_slots, cache->device,
                                               cerr_) == cudaSuccess)
                    break;
                try_slots /= 2;
            }
            if (cache->device)
                log::info("[kimi-infer] expert cache: free=" +
                          std::to_string(free_b >> 20) + "MiB per_expert=" +
                          std::to_string(per_expert >> 20) + "MiB total=" +
                          std::to_string(total_experts) + " slots=" +
                          std::to_string(cache->device->slots));
            else if (want > 0)
                log::warn("[kimi-infer] expert cache unavailable (" + cerr_ +
                          "); legacy staging path");
        }
    } else if (cache && cache->device) {
        // RuntimeTuner resized: rebuild the arena when the wanted slot
        // count no longer matches. Same packed bytes per slot -> bit-exact.
        const int want = KimiExpertCacheSlotsOverride();
        if (want >= 0 && want != cache->device->slots) {
            FreeKimiMoeExpertDeviceCache(cache->device);
            cache->device = nullptr;
            std::string cerr_;
            int try_slots = want;
            const int total_experts = cache->plan.total_experts;
            while (try_slots >= 1 && !cache->device) {
                if (AllocateKimiMoeExpertCache(db, try_slots, cache->device,
                                               cerr_) == cudaSuccess)
                    break;
                try_slots /= 2;
            }
            cache->plan.Init(cache->device ? cache->device->slots : 0,
                             total_experts);
            if (cache->device)
                log::info("[kimi-infer] expert cache resized to " +
                          std::to_string(cache->device->slots) + " slots");
            else
                log::warn("[kimi-infer] expert cache rebuild failed (" +
                          cerr_ + "); legacy staging path");
        }
    }
```

In the `KimiGenerationResult` accumulation (resolve site ~line 1018), add
after `KimiMoeCacheBatch b = KimiMoeResolveBatch(...)`:

```cpp
                    const auto [b_hits, b_misses] =
                        KimiCountCacheHits(b, kKimiTopK);
                    batch_hits += b_hits;
                    batch_misses += b_misses;
```

(the variables `batch_hits`/`batch_misses` — local `std::int64_t = 0;`
next to the step counter declarations; structured bindings require C++20 —
enabled). At the end of `RunKimiGeneration`, before returning success, publish:

```cpp
    gen.cache_hits = batch_hits;
    gen.cache_misses = batch_misses;
```

- [ ] **Step 4: Verify build**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxcommon
```

Expected: compilation without errors.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_kernels.h libs/lsxcommon/src/kimi_moe_inference.cu
git commit -m "feat: kimi expert-cache slots override + arena rebuild + hits/misses telemetry"
```

---

### Task 3: Telemetry in `KimiGenerationResult`

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_kernels.h:55-65` (struct
  `KimiGenerationResult`)

Included in Task 2 de facto (the fields are needed to compile the accumulation), but
recorded as a separate step:

- [ ] **Step 1: Add the fields**

```cpp
struct KimiGenerationResult {
    std::vector<int> tokens;      // generated tokens only (never the prompt)
    float first_maxv = 0.0f;
    bool ok = false;
    bool eos_terminated = false;
    int kv_slots_used = 0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::string error_msg;
    std::vector<float> final_logits;
    // Expert-cache telemetry since generation start (RuntimeTuner input;
    // GLM parity: glm_moe_inference.h:404). Zero when the cache is off.
    std::int64_t cache_hits = 0;
    std::int64_t cache_misses = 0;
};
```

- [ ] **Step 2: Build + commit**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug --target lsxcommon
git add libs/lsxcommon/src/kimi_moe_kernels.h
git commit -m "feat: kimi generation result carries expert-cache telemetry"
```

---

### Task 4: RuntimeTuner in `KimiK27CodeModel`

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_model.h`
- Modify: `libs/lsxcommon/src/kimi_moe_model.cpp` (Load ~lines 60-105,
  Infer ~lines 105-135, TeardownWeights)

- [ ] **Step 1: Model members (header)**

In the private section of `KimiK27CodeModel` (after `expert_cache_`), add
the include `"lsxcommon/runtime_tuner.h"` at the top of the file and the members:

```cpp
    // Adaptive expert-cache controller (GLM parity): owns the cache lease
    // (moved from ModelBase::expert_cache_lease_) and resizes between
    // requests from hits/misses telemetry.
    std::unique_ptr<lifecycle::RuntimeTuner> tuner_;
    std::int64_t tuner_last_hits_ = 0;
    std::int64_t tuner_last_misses_ = 0;
```

- [ ] **Step 2: Load — lease + tuner after `AcquirePermanentWeights`**

In `KimiK27CodeModel::Load`, after the line
`if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();`
insert:

```cpp
    // Adaptive expert-cache lease + RuntimeTuner (GLM parity). Bounds span
    // the whole slot range: the tuner grows/shrinks from telemetry.
    const int total_experts = kimi::kKimiNumLayers * kimi::kKimiNumRouted;
    const std::uint64_t per_expert = kimi::KimiExpertCachePerExpertBytes();
    lifecycle::TuningBounds bounds;
    bounds.expert_cache_slots_min = 4;
    bounds.expert_cache_slots_max = static_cast<std::size_t>(total_experts);
    int initial_slots = 0;
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
            initial_slots = kimi::KimiMoeExpertCacheSlots(
                free_b, per_expert, total_experts,
                std::getenv("LSX_KIMI_EXPERT_CACHE_SLOTS"));
    }
    if (initial_slots > 0) {
        AcquireExpertCacheLease(
            static_cast<std::uint64_t>(initial_slots) * per_expert,
            bounds.expert_cache_slots_min * per_expert,
            bounds.expert_cache_slots_max * per_expert);
        auto &gov = ResourceGovernor::Instance();
        tuner_ = std::make_unique<lifecycle::RuntimeTuner>(
            &gov, std::move(expert_cache_lease_), per_expert, bounds);
        kimi::SetKimiExpertCacheSlotsOverride(
            static_cast<int>(tuner_->Slots()));
        log::info("[kimi-model] expert-cache tuner: initial slots=" +
                  std::to_string(tuner_->Slots()));
    } else {
        log::info("[kimi-model] expert cache disabled (env/heuristic); "
                  "legacy staging path");
    }
    tuner_last_hits_ = tuner_last_misses_ = 0;
```

(remove the old line `expert_cache_.plan.Init(0, ...)` in Load — the plan
is initialized by the lazy .cu block; `#include <cstdlib>` for `std::getenv`
is already present in the TU via other headers — add it if absent.)

- [ ] **Step 3: Infer — Report between requests**

In `KimiK27CodeModel::Infer`, after a successful `RunUnderResidencyLock` and
before `runtime::FillInferenceResult(...)`:

```cpp
    if (tuner_) {  // between requests: adapt the expert cache (GLM parity)
        tuner_->Report(gen.cache_hits - tuner_last_hits_,
                       gen.cache_misses - tuner_last_misses_);
        tuner_last_hits_ = gen.cache_hits;
        tuner_last_misses_ = gen.cache_misses;
        kimi::SetKimiExpertCacheSlotsOverride(
            static_cast<int>(tuner_->Slots()));
        ResourceGovernor::Instance().ReconcileVram();
    }
```

- [ ] **Step 4: TeardownWeights — tuner first**

At the start of `KimiK27CodeModel::TeardownWeights`, add `tuner_.reset();`
(the lease is released by the tuner destructor; the moved-from
`ModelBase::expert_cache_lease_` is empty — a repeated release in the base
Unload is safe).

- [ ] **Step 5: Verify build + host-only suites**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R "test_kimi_moe_expert_cache|test_model_runtime|test_resource_governor|test_runtime_tuner" --output-on-failure
```

Expected: build without errors; all 4 targets PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/kimi_moe_model.h libs/lsxcommon/src/kimi_moe_model.cpp
git commit -m "feat: kimi model RuntimeTuner wiring (lease, telemetry report, slots override)"
```

---

### Task 5: Verification (Linux + Windows) and finalization

- [ ] **Step 1: Windows — full build + host-only regression**

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" --test-dir build -C Debug -R "test_kimi_moe_expert_cache|test_model_runtime|test_model_engine|test_model_factory|test_geometry_guard|test_resource_governor|test_runtime_tuner|test_lifecycle_plan" --output-on-failure
```

Expected: 0 failed. Kimi GPU targets are only BUILT (AGENTS.md restriction).

- [ ] **Step 2: Linux (Ubuntu 26.04) — run the Kimi GPU tests**

```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R kimi --output-on-failure
```

Expected: `test_kimi_moe_expert_cache`, `test_kimi_moe_convert`,
`test_kimi_moe_model`, `test_kimi_moe_inference`, `test_kimi_moe_selfcheck`
— PASS/SKIP (without a checkpoint SKIP_RETURN_CODE 4). The real model must
generate "Paris" as before (bit-exactness untouched).

- [ ] **Step 3: Final commit (if any edits remain)**

```bash
git add -A
git commit -m "test: kimi runtime tuner verification pass"
git push
```

---

## Self-Review

- **Goal coverage:** telemetry (Task 1+3), CUDA accumulation + override/realloc
  (Task 2), the model tuner + lease (Task 4), verification (Task 5) — the goal
  is fully closed.
- **Placeholders:** none; every step contains code/commands.
- **Types:** `KimiCountCacheHits` returns `std::pair<std::int64_t,
  std::int64_t>` — used with structured bindings in Task 2 and the test in Task 1;
  `RuntimeTuner(gov, lease, stride, bounds)` — the signature
  `runtime_tuner.h:31`; `AcquireExpertCacheLease`/`expert_cache_lease_` —
  `model_base.h` (lifecycle-unification II, commit c2f6f72).
