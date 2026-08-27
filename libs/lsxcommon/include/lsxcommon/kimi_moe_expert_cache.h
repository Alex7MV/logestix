// Kimi MoE resident-expert device cache: pure host bookkeeping.
//
// Slot policy, LRU and per-batch residency decisions live here (no CUDA)
// so they are unit-testable without a device. The CUDA side — arena
// allocation, batched copy-stream upload, event fencing — is declared at
// the bottom and implemented in src/kimi_moe_expert_cache.cu.
//
// Residency policy: unified launch-strategy estimate
// lifecycle::ExpertCacheSlots (resource_governor.h) with an explicit
// LSX_KIMI_EXPERT_CACHE_SLOTS override; value 0 disables the cache and the
// generation loop falls back to the legacy per-call staging path.
//
// Bit-exactness: GEMV reads the SAME packed bytes from a cache slot that
// the legacy path copies into staging — identical weights, identical
// kernels. Spec: docs/superpowers/specs/2026-08-27-kimi-expert-cache-design.md
#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/resource_governor.h"

struct KimiExpertEntry;

namespace lsxcommon {
template <typename TEntry>
struct ExpertStore;
using KimiExpertDb = ExpertStore<KimiExpertEntry>;   // defined in kimi_moe_load.h

template <typename TEntry>
struct ExpertStore;

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

// Slot count per the unified residency policy. env_value (may be nullptr or
// empty): explicit value wins and is clamped to total_experts; "0" disables.
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
// missing experts to slots (empty slot first, else LRU victim with
// last_use strictly below the current step; ties break on the lowest slot
// index). Same-batch residents are never evicted (their last_use equals
// the current step). Fixed arrays: zero allocations per call.
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
        // Victim: empty slot first, else min last_use strictly below the
        // current step (this-batch entries are untouchable).
        int victim = -1;
        for (int s = 0; s < plan.slots; ++s) {
            const size_t si = static_cast<size_t>(s);
            if (plan.owner[si] == -1) { victim = s; break; }
            const unsigned long long lu = plan.last_use[si];
            if (lu >= cur) continue;
            if (victim < 0 || lu < plan.last_use[static_cast<size_t>(victim)])
                victim = s;
        }
        if (victim < 0) {
            // No slot could be freed: report as missing without a slot
            // (caller's upload skips it; the expert stays streamed per-call
            // only if the caller treats -1 slots as fallback — for Kimi this
            // is impossible while Active(), so it is a hard error upstream).
            b.missing_idx[b.missing_count] = i;
            ++b.missing_count;
            continue;
        }
        const size_t vi = static_cast<size_t>(victim);
        if (plan.owner[vi] >= 0)
            plan.slot_of[static_cast<size_t>(plan.owner[vi])] = -1;
        plan.owner[vi] = flat;
        plan.slot_of[f] = victim;
        plan.last_use[vi] = cur;
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

struct KimiMoeExpertDeviceCache;   // CUDA arena payload, defined in the .cu

// Host state owned by the model across turns.
struct KimiExpertCacheState {
    KimiMoeExpertCachePlan plan;                 // bookkeeping
    KimiMoeExpertDeviceCache *device = nullptr;  // CUDA arena (nullable)
};

cudaError_t AllocateKimiMoeExpertCache(const ExpertStore<KimiExpertEntry> &db,
                                       int slots,
                                       KimiMoeExpertDeviceCache *&out,
                                       std::string &err);
void FreeKimiMoeExpertDeviceCache(KimiMoeExpertDeviceCache *&cache);
cudaError_t UploadKimiMoeExpertBatch(const ExpertStore<KimiExpertEntry> &db,
                                     const KimiExpertCacheState &state,
                                     const KimiMoeCacheBatch &batch, int layer,
                                     int experts_per_layer);
KimiMoeSlotView KimiMoeSlotPointers(const KimiMoeExpertDeviceCache &cache,
                                    int slot, const KimiExpertEntry &entry);

} // namespace kimi
} // namespace lsxcommon