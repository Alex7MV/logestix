# Kimi MoE Expert Cache + Batch Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. (AGENTS.md forbids Task-subagents — inline execution only.)
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Kimi expert device slot cache under a unified VRAM policy + a batched upload of the
layer's top-8 experts on a copy stream instead of 8 synchronous H2D per token.

**Architecture:** Host bookkeeping (`kimi_moe_expert_cache.h` — pure, without CUDA, unit-tested)
+ the CUDA part (`kimi_moe_expert_cache.cu`: slot arena, copy stream, event fence).
Generation: Pass1 ResolveBatch → Pass2 batched upload of missing ∥ GEMV of resident → Pass3 GEMV
of the rest from the slots. Fallback: 0 slots / env=0 → legacy per-call staging. Bit-exactness:
GEMV reads the same bytes from the cache slot; the kernel is unchanged.

**Tech Stack:** C++20/CUDA 13.3, Catch2 v3.7.0, CMake 4.3 (VS 2026 bundled).

**Spec:** `docs/superpowers/specs/2026-08-27-kimi-expert-cache-design.md`

**Toolchain (AGENTS.md):**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
```

**Key codebase facts (verified during planning):**
- `KimiExpertEntry` (in `expert_store.h` via `ExpertStore<KimiExpertEntry>` = `KimiExpertDb`):
  `host_w1/w2/w3`, `w{1,2,3}_bytes`, `host_w{1,3,2}_scales`, `w{1,3,2}_scales_bytes`,
  `w{1,2,3}_global_scale` (float).
- Geometry: `kKimiHidden=7168`, `kKimiMoeIntermediate=2048`, `kKimiNumLayers=61`,
  `kKimiNumRouted=384`, `kKimiTopK=8`. Packed w13 = `I*Hd/2` = 7 340 032 B,
  scales = `I*Hd/16` = 1 835 008 B → per_expert ≈ 26 MB.
- Staging layout (the slot reference): `RunKimiExpertFfn` `kimi_moe_inference.cu:654-662` —
  `[w1|w3|w2]` pack (align_up 256 sections), `[s1|s3|s2]` scales.
- Staging offset math: `off_p_w3 = off_p_w1 + a(w1_bytes)` etc. — the cache slots
  reproduce this scheme exactly.
- Model: `KimiK27CodeModel` (`src/kimi_moe_model.h`, kType `kimi-k2.7-code`),
  members `pw_`, `experts_` (`KimiExpertDb`); `EnsureResident(layer, expert, stream)`
  — the host-mmap LRU `ExpertStore`.
- `RunKimiGeneration` (`kimi_moe_inference.cu:1025+`), `KimiLayerStack` (`:845`),
  the top-k loop `:941-959` (exact text in Task 3).

---

### Task 1: Pure bookkeeping — TDD

**Files:**
- Create: `include/lsxcommon/kimi_moe_expert_cache.h`
- Test: `tests/test_kimi_moe_expert_cache.cpp`
- Modify: `tests/CMakeLists.txt` (at the end)

- [ ] **Step 1.1: Failing test**

```cpp
// tests/test_kimi_moe_expert_cache.cpp
// Kimi MoE expert device-cache bookkeeping (host-only, zero CUDA calls).
// Spec: docs/superpowers/specs/2026-08-27-kimi-expert-cache-design.md
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <vector>

#include "lsxcommon/kimi_moe_expert_cache.h"

using lsxcommon::kimi::KimiMoeCacheBatch;
using lsxcommon::kimi::KimiMoeExpertCachePlan;
using lsxcommon::kimi::KimiMoeExpertCacheSlots;
using lsxcommon::kimi::KimiMoeResolveBatch;

namespace {
constexpr int kEPL = 384;   // Kimi experts per layer
} // namespace

TEST_CASE("Kimi cache slots: policy wiring and env override", "[kimi][cache]")
{
    const std::size_t per_expert = 26ull << 20;   // ~26 MiB pack+scale

    // Policy path (env absent): usable = free - 4 GiB reserve.
    const std::size_t free_bytes = (4ull << 30) + 100 * per_expert;
    REQUIRE(KimiMoeExpertCacheSlots(free_bytes, per_expert, 23424, nullptr) == 100);
    // Clamped to total experts.
    REQUIRE(KimiMoeExpertCacheSlots(64ull << 30, per_expert, 23424, nullptr) == 23424);
    // Free below reserve -> nothing fits.
    REQUIRE(KimiMoeExpertCacheSlots(1ull << 30, per_expert, 23424, nullptr) == 0);

    // Explicit env wins (clamped); "0" disables the cache.
    REQUIRE(KimiMoeExpertCacheSlots(0, per_expert, 23424, "77") == 77);
    REQUIRE(KimiMoeExpertCacheSlots(0, per_expert, 23424, "999999") == 23424);
    REQUIRE(KimiMoeExpertCacheSlots(free_bytes, per_expert, 23424, "0") == 0);
}

TEST_CASE("Kimi cache batch: first touch assigns slots, second is resident",
          "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(4, 2 * kEPL);

    int ids[8] = {3, 7, 3, 100, 7, 11, 200, 3};
    KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 1, kEPL, ids, 8);

    // Duplicate ids collapse: residents share slots, uniques get uploads.
    std::vector<int> assigned;
    for (int i = 0; i < b.missing_count; ++i)
        assigned.push_back(b.missing_slot[i]);
    REQUIRE(assigned.size() <= 4);   // cache has 4 slots
    std::sort(assigned.begin(), assigned.end());
    REQUIRE(std::adjacent_find(assigned.begin(), assigned.end()) == assigned.end());

    // Same ids again: everything resident, nothing to upload.
    KimiMoeCacheBatch b2 = KimiMoeResolveBatch(p, 1, kEPL, ids, 8);
    REQUIRE(b2.missing_count == 0);
    for (int i = 0; i < 8; ++i) REQUIRE(b2.resident_slot[i] >= 0);
}

TEST_CASE("Kimi cache LRU: cold slot evicted before hot one", "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(2, 10 * kEPL);

    int id0[1] = {0}, id1[1] = {1}, id2[1] = {2};
    auto single = [&](int *ids) {
        KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 0, kEPL, ids, 1);
        REQUIRE(b.missing_count == 1);
        REQUIRE(b.resident_slot[0] == -1);
        return b.missing_slot[0];
    };
    REQUIRE(single(id0) == 0);   // empty slot 0 first
    REQUIRE(single(id1) == 1);   // empty slot 1
    REQUIRE(single(id2) == 0);   // evicts LRU victim (expert 0, slot 0)

    // Expert 1 still resident; expert 0 evicted.
    KimiMoeCacheBatch r1 = KimiMoeResolveBatch(p, 0, kEPL, id1, 1);
    REQUIRE(r1.missing_count == 0);
    REQUIRE(r1.resident_slot[0] == 1);
    KimiMoeCacheBatch r0 = KimiMoeResolveBatch(p, 0, kEPL, id0, 1);
    REQUIRE(r0.missing_count == 1);
}

TEST_CASE("Kimi cache rejects out-of-range input deterministically",
          "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(4, 2 * kEPL);

    int ok[1] = {0};
    REQUIRE(KimiMoeResolveBatch(p, -1, kEPL, ok, 1).invalid);
    int over[1] = {kEPL};
    REQUIRE(KimiMoeResolveBatch(p, 0, kEPL, over, 1).invalid);
    int neg[1] = {-1};
    REQUIRE(KimiMoeResolveBatch(p, 0, kEPL, neg, 1).invalid);
}

TEST_CASE("Kimi cache disabled plan is inert", "[kimi][cache]")
{
    KimiMoeExpertCachePlan p;
    p.Init(0, 23424);
    REQUIRE_FALSE(p.Active());
    int ids[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    KimiMoeCacheBatch b = KimiMoeResolveBatch(p, 0, kEPL, ids, 8);
    REQUIRE(b.missing_count == 8);
    REQUIRE(b.missing_slot[0] == -1);   // no slots to assign
}
```

- [ ] **Step 1.2: Register the target in `tests/CMakeLists.txt` (at the end)**

```cmake
# Kimi MoE expert cache bookkeeping (host-only, pure logic, no CUDA).
add_executable(test_kimi_moe_expert_cache test_kimi_moe_expert_cache.cpp)
target_link_libraries(test_kimi_moe_expert_cache PRIVATE Catch2::Catch2WithMain)
add_test(NAME test_kimi_moe_expert_cache COMMAND test_kimi_moe_expert_cache)
```

- [ ] **Step 1.3: Confirm it is red**

```powershell
& $cmake --build build --config Debug --target test_kimi_moe_expert_cache
```

Expected: `fatal error C1083: Cannot open include file: "lsxcommon/kimi_moe_expert_cache.h"`.

- [ ] **Step 1.4: Create the header (canonical, complete)**

```cpp
// include/lsxcommon/kimi_moe_expert_cache.h
// Kimi MoE resident-expert device cache: pure host bookkeeping.
//
// Slot policy, LRU and per-batch residency decisions live here (no CUDA) so
// they are unit-testable without a device. The CUDA arena (alloc, batch
// upload, event fencing) is declared at the bottom and implemented in
// src/kimi_moe_expert_cache.cu.
//
// Residency policy: unified launch-strategy estimate
// lifecycle::ExpertCacheSlots (resource_governor.h) with an explicit
// LSX_KIMI_EXPERT_CACHE_SLOTS override; value 0 disables the cache and the
// generation loop falls back to the legacy per-call staging path.
//
// Bit-exactness: GEMV reads the SAME packed bytes from a cache slot that the
// legacy path copies into staging — identical weights, identical kernels.
// Spec: docs/superpowers/specs/2026-08-27-kimi-expert-cache-design.md
#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "lsxcommon/resource_governor.h"

struct KimiExpertEntry;

namespace lsxcommon {
namespace kimi {

inline constexpr int kKimiCacheBatch = 8;   // router top-k

struct KimiMoeCacheBatch {
    int resident_slot[kKimiCacheBatch];  // device slot per batch idx, -1 = miss
    int missing_idx[kKimiCacheBatch];    // batch indices that need upload
    int missing_slot[kKimiCacheBatch];   // device slot per missing idx, -1 = none
    int missing_count = 0;
    bool invalid = false;                // out-of-range layer/id input
};

struct KimiMoeExpertCachePlan {
    int slots = 0;                  // device slots (0 = cache off)
    int total_experts = 0;          // num_layers * experts_per_layer
    std::vector<int> slot_of;       // flat expert -> slot, -1 = not resident
    std::vector<int> owner;         // slot -> flat expert, -1 = empty
    std::vector<unsigned long long> last_use;  // slot -> LRU step marker
    unsigned long long step = 0;

    void Init(int slots_arg, int total_experts_arg)
    {
        slots = slots_arg > 0 ? slots_arg : 0;
        total_experts = total_experts_arg > 0 ? total_experts_arg : 0;
        slot_of.assign(static_cast<size_t>(total_experts), -1);
        owner.assign(static_cast<size_t>(slots), -1);
        last_use.assign(static_cast<size_t>(slots), 0);
        step = 0;
    }

    bool Active() const { return slots > 0; }
};

inline int KimiMoeExpertCacheSlots(std::size_t free_bytes,
                                   std::size_t per_expert_bytes,
                                   int total_experts, const char *env_value)
{
    if (total_experts <= 0) return 0;
    if (env_value != nullptr && env_value[0] != '\0') {
        const int v = std::atoi(env_value);
        if (v <= 0) return 0;   // explicit 0 = disable
        return v < total_experts ? v : total_experts;
    }
    return lifecycle::ExpertCacheSlots(free_bytes, per_expert_bytes,
                                       total_experts);
}

// One layer's top-k residency decision. Advances the LRU clock; assigns
// missing experts to slots (empty first, then LRU victims with
// last_use < current step; ties break on the lowest slot index). Same-batch
// residents are never evicted (their last_use == current step).
inline KimiMoeCacheBatch KimiMoeResolveBatch(KimiMoeExpertCachePlan &plan,
                                             int layer, int experts_per_layer,
                                             const int *top_ids, int k)
{
    KimiMoeCacheBatch b;
    b.missing_count = 0;
    for (int i = 0; i < kKimiCacheBatch; ++i) {
        b.resident_slot[i] = -1;
        b.missing_idx[i] = -1;
        b.missing_slot[i] = -1;
    }
    if (layer < 0 || k <= 0 || k > kKimiCacheBatch || top_ids == nullptr ||
        experts_per_layer <= 0) {
        b.invalid = true;
        return b;
    }
    for (int i = 0; i < k; ++i) {
        if (top_ids[i] < 0 || top_ids[i] >= experts_per_layer) {
            b.invalid = true;
            return b;
        }
    }
    if (!plan.Active()) {
        for (int i = 0; i < k; ++i) b.missing_idx[i] = i;
        b.missing_count = k;    // everything missing; no slots to assign
        return b;
    }

    const unsigned long long cur = ++plan.step;

    for (int i = 0; i < k; ++i) {
        const int flat = layer * experts_per_layer + top_ids[i];
        const size_t f = static_cast<size_t>(flat);
        int slot = plan.slot_of[f];
        if (slot >= 0) {
            plan.last_use[static_cast<size_t>(slot)] = cur;
            b.resident_slot[i] = slot;
            continue;
        }
        // Victim: empty slot (owner == -1, last_use == 0) first; else min
        // last_use strictly below the current step.
        int victim = -1;
        for (int s = 0; s < plan.slots; ++s) {
            const size_t si = static_cast<size_t>(s);
            if (plan.owner[si] == -1) { victim = s; break; }
            const unsigned long long lu = plan.last_use[si];
            if (lu >= cur) continue;   // touched this batch — untouchable
            if (victim == -1 || lu < plan.last_use[static_cast<size_t>(victim)])
                victim = s;
        }
        if (victim < 0) {   // no slot could be freed (should not happen)
            b.missing_idx[b.missing_count] = i;
            ++b.missing_count;
            continue;
        }
        const size_t v = static_cast<size_t>(victim);
        if (plan.owner[v] >= 0)
            plan.slot_of[static_cast<size_t>(plan.owner[v])] = -1;
        plan.owner[v] = flat;
        plan.slot_of[f] = victim;
        plan.last_use[v] = cur;
        b.missing_idx[b.missing_count] = i;
        b.missing_slot[b.missing_count] = victim;
        ++b.missing_count;
    }
    return b;
}

// ── CUDA-side surface (implemented in src/kimi_moe_expert_cache.cu) ──

// Device-side slot pointers. Layout mirrors RunKimiExpertFfn staging:
// pack [w1|w3|w2], scale [s1|s3|s2], 256-aligned sections.
struct KimiMoeSlotView {
    const unsigned char *p_w1;
    const unsigned char *p_w3;
    const unsigned char *p_w2;
    const unsigned char *p_s1;
    const unsigned char *p_s3;
    const unsigned char *p_s2;
};

// CUDA arena + copy stream + fence event. Opaque here, defined in the .cu.
struct KimiMoeExpertDeviceCache;

struct KimiExpertCacheState {
    KimiMoeExpertCachePlan plan;                 // host bookkeeping
    KimiMoeExpertDeviceCache *device = nullptr;  // CUDA arena (nullable)
};

cudaError_t AllocateKimiMoeExpertCache(const struct ExpertStore<KimiExpertEntry> &db,
                                       int slots, KimiMoeExpertDeviceCache *&out,
                                       std::string &err);
void FreeKimiMoeExpertDeviceCache(KimiMoeExpertDeviceCache *&cache);
cudaError_t UploadKimiMoeExpertBatch(const struct ExpertStore<KimiExpertEntry> &db,
                                     const KimiMoeExpertCachePlan &plan,
                                     const KimiMoeCacheBatch &batch, int layer,
                                     int experts_per_layer,
                                     KimiMoeExpertDeviceCache &cache);
KimiMoeSlotView KimiMoeSlotPointers(const KimiMoeExpertDeviceCache &cache,
                                    int slot, const KimiExpertEntry &entry);

} // namespace kimi
} // namespace lsxcommon
```

- [ ] **Step 1.5: Green**

```powershell
& $cmake --build build --config Debug --target test_kimi_moe_expert_cache
& build\bin\Debug\test_kimi_moe_expert_cache.exe
```

Expected: `All tests passed` (5 test cases). If the LRU test is red — fix
the victim semantics in `KimiMoeResolveBatch` until green (the test = the specification).

- [ ] **Step 1.6: Commit**

```powershell
git add include/lsxcommon/kimi_moe_expert_cache.h tests/test_kimi_moe_expert_cache.cpp tests/CMakeLists.txt
git commit -m "feat: kimi expert cache host bookkeeping with LRU and policy wiring"
```

### Task 2: The cache CUDA part

**Files:**
- Create: `src/kimi_moe_expert_cache.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (OBJECT `lsxcommon_kimi_cu`)

- [ ] **Step 2.1: Create `src/kimi_moe_expert_cache.cu`**

```cpp
// src/kimi_moe_expert_cache.cu
// CUDA side of the Kimi resident-expert device cache: arena allocation,
// batched async upload on a dedicated copy stream, event fencing.
// Host bookkeeping: include/lsxcommon/kimi_moe_expert_cache.h (pure).
#include "lsxcommon/kimi_moe_expert_cache.h"

#include "lsxcommon/expert_store.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>

namespace lsxcommon {
namespace kimi {
namespace {

size_t AlignUp256(size_t v) { return (v + 255) / 256 * 256; }

size_t KimiPackStride(const KimiExpertEntry &e)
{
    return AlignUp256(e.w1_bytes) + AlignUp256(e.w3_bytes) +
           AlignUp256(e.w2_bytes);
}

size_t KimiScaleStride(const KimiExpertEntry &e)
{
    return AlignUp256(e.w1_scales_bytes) + AlignUp256(e.w3_scales_bytes) +
           AlignUp256(e.w2_scales_bytes);
}

} // namespace

struct KimiMoeExpertDeviceCache {
    unsigned char *d_pack = nullptr;   // slots * pack_stride  [w1|w3|w2]
    unsigned char *d_scale = nullptr;  // slots * scale_stride [s1|s3|s2]
    size_t pack_stride = 0;
    size_t scale_stride = 0;
    int slots = 0;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ready = nullptr;       // recorded after each batch upload
};

KimiMoeSlotView KimiMoeSlotPointers(const KimiMoeExpertDeviceCache &cache,
                                    int slot, const KimiExpertEntry &e)
{
    unsigned char *pk = cache.d_pack +
                        static_cast<size_t>(slot) * cache.pack_stride;
    unsigned char *sc = cache.d_scale +
                        static_cast<size_t>(slot) * cache.scale_stride;
    KimiMoeSlotView v;
    v.p_w1 = pk;
    v.p_w3 = pk + AlignUp256(e.w1_bytes);
    v.p_w2 = v.p_w3 + AlignUp256(e.w3_bytes);
    v.p_s1 = sc;
    v.p_s3 = sc + AlignUp256(e.w1_scales_bytes);
    v.p_s2 = v.p_s3 + AlignUp256(e.w3_scales_bytes);
    return v;
}

cudaError_t AllocateKimiMoeExpertCache(const KimiExpertDb &db, int slots,
                                       KimiMoeExpertDeviceCache *&out,
                                       std::string &err)
{
    out = nullptr;
    if (slots <= 0 || db.entries.empty()) {
        err = "cache disabled";
        return cudaErrorInvalidValue;
    }
    // Uniform geometry check: every expert must match entry 0's strides
    // (fixed INT4/NVFP4 layout). A mismatch disables the cache — the
    // generation loop then keeps the legacy per-call staging path.
    const size_t pack = KimiPackStride(db.entries[0]);
    const size_t scale = KimiScaleStride(db.entries[0]);
    for (const auto &e : db.entries) {
        if (KimiPackStride(e) != pack || KimiScaleStride(e) != scale) {
            err = "non-uniform expert geometry; cache disabled";
            return cudaErrorInvalidValue;
        }
    }
    auto *c = new KimiMoeExpertDeviceCache();
    c->pack_stride = pack;
    c->scale_stride = scale;
    c->slots = slots;
    cudaError_t e = cudaMalloc(&c->d_pack,
                               static_cast<size_t>(slots) * pack);
    if (e == cudaSuccess)
        e = cudaMalloc(&c->d_scale, static_cast<size_t>(slots) * scale);
    if (e == cudaSuccess)
        e = cudaStreamCreate(&c->copy_stream);
    if (e == cudaSuccess)
        e = cudaEventCreateWithFlags(&c->ready, cudaEventDisableTiming);
    if (e != cudaSuccess) {
        err = "kimi cache alloc failed";
        FreeKimiMoeExpertDeviceCache(c);
        return e;
    }
    // Prime the fence: a wait before the FIRST record must be a no-op.
    cudaEventRecord(c->ready, c->copy_stream);
    out = c;
    return cudaSuccess;
}

void FreeKimiMoeExpertDeviceCache(KimiMoeExpertDeviceCache *&c)
{
    if (!c) return;
    if (c->d_pack) cudaFree(c->d_pack);
    if (c->d_scale) cudaFree(c->d_scale);
    if (c->copy_stream) cudaStreamDestroy(c->copy_stream);
    if (c->ready) cudaEventDestroy(c->ready);
    delete c;
    c = nullptr;
}

cudaError_t UploadKimiMoeExpertBatch(const KimiExpertDb &db,
                                     const KimiMoeCacheBatch &batch, int layer,
                                     int experts_per_layer,
                                     KimiMoeExpertDeviceCache &cache)
{
    if (cache.slots <= 0 || batch.missing_count <= 0) return cudaSuccess;
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < batch.missing_count; ++i) {
        const int bid = batch.missing_idx[i];
        const int slot = batch.missing_slot[i];
        if (bid < 0 || slot < 0) continue;
        // Host-mmap residency for the DMA source (ExpertStore fd LRU);
        // cheap no-op when the mmap is already open.
        db.EnsureResident(layer, bid, cache.copy_stream);
        const KimiExpertEntry &ent =
            db.entries[static_cast<size_t>(layer) * experts_per_layer +
                       static_cast<size_t>(bid)];
        const KimiMoeSlotView v = KimiMoeSlotPointers(cache, slot, ent);
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w1), ent.host_w1,
                            ent.w1_bytes, cudaMemcpyHostToDevice,
                            cache.copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w3), ent.host_w3,
                            ent.w3_bytes, cudaMemcpyHostToDevice,
                            cache.copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w2), ent.host_w2,
                            ent.w2_bytes, cudaMemcpyHostToDevice,
                            cache.copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s1),
                            ent.host_w1_scales, ent.w1_scales_bytes,
                            cudaMemcpyHostToDevice, cache.copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s3),
                            ent.host_w3_scales, ent.w3_scales_bytes,
                            cudaMemcpyHostToDevice, cache.copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s2),
                            ent.host_w2_scales, ent.w2_scales_bytes,
                            cudaMemcpyHostToDevice, cache.copy_stream);
        if (e != cudaSuccess) return e;
    }
    return cudaEventRecord(cache.ready, cache.copy_stream);
}

} // namespace kimi
} // namespace lsxcommon
```

- [ ] **Step 2.2: CMake** — in `libs/lsxcommon/CMakeLists.txt` replace:

```cmake
add_library(lsxcommon_kimi_cu OBJECT
    src/kimi_inference.cu
)
```

with (the file `kimi_inference.cu` has already been renamed to `kimi_moe_inference.cu`
as part of the naming refactor — use the current name from the current file):

```cmake
add_library(lsxcommon_kimi_cu OBJECT
    src/kimi_moe_inference.cu
    src/kimi_moe_expert_cache.cu
)
```

- [ ] **Step 2.3: Build**

```powershell
& $cmake --build build --config Debug --target lsxcommon_kimi_cu
```

Expected: exit=0, 0 errors.

- [ ] **Step 2.4: Commit**

```powershell
git add libs/lsxcommon/src/kimi_moe_expert_cache.cu libs/lsxcommon/CMakeLists.txt
git commit -m "feat: kimi expert cache device arena with batched copy-stream upload"
```

### Task 3: Integration into generation

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_inference.cu` (new `RunKimiExpertFfnCached`
  after `RunKimiExpertFfn`; the signature `KimiLayerStack:845` + the loop `:941-959`;
  `RunKimiGeneration:1025+`; cache allocation after `d_staging:1064-1067`)
- Modify: `libs/lsxcommon/src/kimi_moe_model.h:33-39`, `kimi_moe_model.cpp`
  (Load/Infer/Unload)

- [ ] **Step 3.1: `RunKimiExpertFfnCached`**

In `kimi_moe_inference.cu`, after the body of `RunKimiExpertFfn`, add a function.
Mechanics: scratch offsets from zero (dequant + activations + y), NO H2D copies —
the weights are read from `view`. The GEMV tail is copied 1:1 from `RunKimiExpertFfn`
(from `cublasSetStream(cublas_h, stream);` to the end of the function, including
gate/up GemmEx, down GemmEx and the accumulation with `expert_weight` — parameters and
order unchanged):

```cpp
cudaError_t RunKimiExpertFfnCached(cublasHandle_t cublas_h,
                                   cudaStream_t stream,
                                   const KimiExpertEntry &expert,
                                   float expert_weight,
                                   const __nv_bfloat16 *d_x,
                                   __nv_bfloat16 *d_out, void *d_scratch,
                                   size_t scratch_bytes,
                                   const KimiMoeSlotView &view)
{
    if (!d_scratch) return cudaErrorInvalidValue;
    constexpr int I = kKimiMoeIntermediate;
    constexpr int Hd = kKimiHidden;
    const int64_t w13_el = static_cast<int64_t>(I) * Hd;

    auto at = [d_scratch](size_t off) {
        return reinterpret_cast<__nv_bfloat16 *>(
            static_cast<uint8_t *>(d_scratch) + off);
    };
    size_t off = 0;
    __nv_bfloat16 *dq_gate = at(off); off += static_cast<size_t>(w13_el) * 2;
    __nv_bfloat16 *dq_up   = at(off); off += static_cast<size_t>(w13_el) * 2;
    __nv_bfloat16 *dq_down = at(off); off += static_cast<size_t>(w13_el) * 2;
    __nv_bfloat16 *act_g = at(off); off += I * sizeof(__nv_bfloat16);
    __nv_bfloat16 *act_u = at(off); off += I * sizeof(__nv_bfloat16);
    __nv_bfloat16 *act_h = at(off); off += I * sizeof(__nv_bfloat16);
    __nv_bfloat16 *y     = at(off); off += Hd * sizeof(__nv_bfloat16);
    if (scratch_bytes < off) return cudaErrorInvalidValue;

    auto dequant = [&](const uint8_t *packed, const uint8_t *scales,
                       float gscale, __nv_bfloat16 *dst, int64_t numel) {
        const int block = 256;
        const int64_t grid = (numel + block - 1) / block;
        ::lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
            <<<static_cast<unsigned>(grid), block, 0, stream>>>(
                packed, scales, gscale, dst, numel);
    };
    dequant(view.p_w1, view.p_s1, expert.w1_global_scale, dq_gate, w13_el);
    dequant(view.p_w3, view.p_s3, expert.w3_global_scale, dq_up, w13_el);
    dequant(view.p_w2, view.p_s2, expert.w2_global_scale, dq_down,
            static_cast<int64_t>(Hd) * I);
    return RunKimiExpertFfnTail(cublas_h, stream, expert, expert_weight, d_x,
                                d_out, dq_gate, dq_up, dq_down, act_g, act_u,
                                act_h, y);
}
```

To avoid duplicating the GEMV tail: in `RunKimiExpertFfn`, extract the block from
`cublasSetStream` to the end into a separate function

```cpp
static cudaError_t RunKimiExpertFfnTail(
    cublasHandle_t cublas_h, cudaStream_t stream, float expert_weight,
    const __nv_bfloat16 *d_x, __nv_bfloat16 *d_out, __nv_bfloat16 *dq_gate,
    __nv_bfloat16 *dq_up, __nv_bfloat16 *dq_down, __nv_bfloat16 *act_g,
    __nv_bfloat16 *act_u, __nv_bfloat16 *act_h, __nv_bfloat16 *y)
```

— body: copy the lines from `cublasSetStream` to `return` from
`RunKimiExpertFfn` unchanged; in `RunKimiExpertFfn` replace this tail with a
call to `RunKimiExpertFfnTail(...)` using its local pointers. This
guarantees a single GEMV code path (DRY) and bit-exactness of both paths.

- [ ] **Step 3.2: The top-k loop (replace `kimi_moe_inference.cu:941-959`)**

Current code (anchor):

```cpp
                int h_ids[kKimiTopK];
                float h_w[kKimiTopK];
                cudaMemcpyAsync(h_ids, d_top_ids, sizeof(h_ids),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_w, d_top_w, sizeof(h_w),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);   // ids needed on host
                for (int k = 0; k < kKimiTopK; ++k) {
                    ensure_kimi_expert_resident(db, l, h_ids[k], stream);
                    cudaError_t fe = RunKimiExpertFfn(
                        cublas_h, stream,
                        db.entries[(size_t)l * kKimiNumRouted + h_ids[k]],
                        h_w[k], row, acc, d_staging, staging_bytes);
                    if (fe != cudaSuccess) {
                        fprintf(stderr, "[trace] FFN l=%d e=%d err=%s\n",
                                l, h_ids[k], cudaGetErrorString(fe));
                        return fe;
                    }
                }
```

New code:

```cpp
                int h_ids[kKimiTopK];
                float h_w[kKimiTopK];
                cudaMemcpyAsync(h_ids, d_top_ids, sizeof(h_ids),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_w, d_top_w, sizeof(h_w),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);   // ids needed on host
                if (cache && cache->device && cache->plan.Active()) {
                    // Pass 1: batch residency decision (LRU bookkeeping).
                    KimiMoeCacheBatch b = KimiMoeResolveBatch(
                        cache->plan, l, kKimiNumRouted, h_ids, kKimiTopK);
                    // Pass 2: batch upload of misses on the copy stream.
                    if (b.missing_count > 0) {
                        cudaError_t ue = UploadKimiMoeExpertBatch(
                            db, *cache, b, l, kKimiNumRouted,
                            *cache->device);
                        if (ue != cudaSuccess) return ue;
                    }
                    cudaStreamWaitEvent(stream, cache->device->ready, 0);
                    // Pass 3: GEMV from cache slots; accumulation order is
                    // the fixed k index order (bit-exact contract).
                    for (int k = 0; k < kKimiTopK; ++k) {
                        const int slot = b.resident_slot[k];
                        if (slot < 0) return cudaErrorInvalidValue;
                        const KimiExpertEntry &ent =
                            db.entries[(size_t)l * kKimiNumRouted + h_ids[k]];
                        const KimiMoeSlotView v =
                            KimiMoeSlotPointers(*cache->device, slot, ent);
                        cudaError_t fe = RunKimiExpertFfnCached(
                            cublas_h, stream, ent, h_w[k], row, acc,
                            d_staging, staging_bytes, v);
                        if (fe != cudaSuccess) {
                            fprintf(stderr, "[trace] FFN l=%d e=%d err=%s\n",
                                    l, h_ids[k], cudaGetErrorString(fe));
                            return fe;
                        }
                    }
                } else {
                    // Legacy per-call staging path (cache disabled / 0 slots).
                    for (int k = 0; k < kKimiTopK; ++k) {
                        ensure_kimi_expert_resident(db, l, h_ids[k], stream);
                        cudaError_t fe = RunKimiExpertFfn(
                            cublas_h, stream,
                            db.entries[(size_t)l * kKimiNumRouted + h_ids[k]],
                            h_w[k], row, acc, d_staging, staging_bytes);
                        if (fe != cudaSuccess) {
                            fprintf(stderr, "[trace] FFN l=%d e=%d err=%s\n",
                                    l, h_ids[k], cudaGetErrorString(fe));
                            return fe;
                        }
                    }
                }
```

- [ ] **Step 3.3: Signatures and allocation**

1. `KimiLayerStack` (`:845`): add a last parameter
   `KimiExpertCacheState *cache` (nullable); the single call — `:1125` —
   add `cache`.
2. `RunKimiGeneration` (`:1025`): add a last parameter
   `KimiExpertCacheState *cache`; pass it into `KimiLayerStack`.
3. After the `d_staging` malloc (`:1064-1067`), insert the lazy cache allocation:

```cpp
    if (cache && !cache->device) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const int total_experts = kKimiNumLayers * kKimiNumRouted;
            const size_t per_expert =
                3ull * (7168ull * 2048ull / 2) +   // packed w1|w3|w2
                3ull * (7168ull * 2048ull / 16);   // scales s1|s3|s2
            const int slots = KimiMoeExpertCacheSlots(
                free_b, per_expert, cache->plan.total_experts,
                std::getenv("LSX_KIMI_EXPERT_CACHE_SLOTS"));
            cache->plan.Init(slots, cache->plan.total_experts);
            std::string cerr_;
            int try_slots = slots;
            while (try_slots >= 1 &&
                   AllocateKimiMoeExpertCache(db, try_slots, cache->device,
                                              cerr_) != cudaSuccess) {
                try_slots /= 2;
            }
            if (cache->device)
                log::info("[kimi-infer] expert cache: free=" +
                          std::to_string(free_b >> 20) + "MiB per_expert=" +
                          std::to_string(per_expert >> 20) + "MiB total=" +
                          std::to_string(total_experts) + " slots=" +
                          std::to_string(cache->device->slots));
            else if (slots > 0)
                log::warn("[kimi-infer] expert cache unavailable (" + cerr_ +
                          ") - legacy staging path");
        }
    }
```

4. `kimi_moe_model.h` — add in the private section (next to `experts_`) and include
   `#include "lsxcommon/kimi_moe_expert_cache.h"`:

```cpp
    kimi::KimiExpertCacheState expert_cache_{};
```

5. In `Load()`, after `experts_` initialization:

```cpp
    expert_cache_.plan.Init(0, 61 * 384);   // slots decided lazily on first Infer
```

6. In `Infer` — pass `&expert_cache_` into `RunKimiGeneration`; in
   `Unload()` / `TeardownWeights()`:

```cpp
    kimi::FreeKimiMoeExpertDeviceCache(expert_cache_.device);
```

### Task 4: Verification and commit

- [ ] **Step 4.1: Full build**

```powershell
& $cmake --build build --config Debug
```

Expected: exit=0, 0 errors.

- [ ] **Step 4.2: Host tests**

```powershell
& $ctest --test-dir build -C Debug -R 'kimi|code_naming|resource_governor|geometry_guard|runtime_tuner' --output-on-failure
```

Expected: 100% passed (the real kimi model tests SKIP — there is no model on the
machine; the naming guard is green without touching the allowlist).

- [ ] **Step 4.3: Commit**

```powershell
git add -A
git commit -m "feat: kimi resident-expert cache integration with batched pipeline"
git push
```

---

## Self-Review

1. **Spec coverage:** cache + policy + env override (Task 1/2), the 3-pass batch
   pipeline (Task 3.2), the legacy fallback branch and env=0 (Task 3.2/3.3),
   shrink-retry (Task 3.3), the parity log line (Task 3.3), bit-exactness —
   the single GEMV tail `RunKimiExpertFfnTail` (Task 3.1), tests (Task 1),
   CMake/naming guard (Task 2, 4).
2. **Placeholders:** all code blocks are canonical (drafts from the first revision
   removed); the only anchor instruction is copying the GEMV tail
   `RunKimiExpertFfn:723-end` into `RunKimiExpertFfnTail` (exact lines
   specified, code duplication eliminated).
3. **Type consistency:** `KimiMoeCacheBatch`, `KimiMoeExpertCachePlan`,
   `KimiExpertCacheState`, `KimiMoeSlotView`, `KimiMoeExpertDeviceCache` —
   the .cu signatures match the Task 1 header declarations; the `cache` parameter
   is threaded through `KimiGeneration → KimiLayerStack → top-k loop`.