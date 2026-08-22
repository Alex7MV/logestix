#ifndef LSXCOMMON_GLM_EXPERT_CACHE_H
#define LSXCOMMON_GLM_EXPERT_CACHE_H

// GLM routed-expert device cache (port of DeepSeek's ExpertDeviceCache).
// One device block holds [slots] expert sets laid out as
//   d_pack  = [slots][pack_stride]    (w1|w2|w3 E2M1 nibbles, packed)
//   d_scale = [slots][scale_stride]   (per-block E4M3 descriptors)
// The device buffers are owned by the generation context; this header only
// defines the layout plus the host-side LRU bookkeeping as named inline
// helpers so the semantics stay unit-testable without CUDA.
//
// Safety contract (mirrors deepseek_inference_load.cuh): all device access is
// stream-ordered by the caller; the bookkeeping arrays are caller-owned plain
// int32 storage — no absl, safe to include from .cu TUs.

#include <cstddef>
#include <cstdint>

namespace lsxcommon {
namespace glm {

struct GlmExpertDeviceCache {
    uint8_t *d_base = nullptr;        // one device block: [pack][scale]
    uint8_t *d_pack = nullptr;        // [slots][pack_stride]
    uint8_t *d_scale = nullptr;       // [slots][scale_stride]
    int slots = 0;
    int64_t pack_stride = 0;
    int64_t scale_stride = 0;
    int total_experts = 0;
    // Host bookkeeping (caller-owned storage).
    int32_t *h_slot_of = nullptr;     // [total_experts] flat idx -> slot, -1 = miss
    int32_t *h_slot_owner = nullptr;  // [slots] flat idx or -1
    int32_t *h_lru = nullptr;         // [slots]; h_lru[0] = MRU slot
    // Hit/miss counters (diagnostics): hits in CacheLookup, misses in CacheInsert.
    int64_t hits = 0;
    int64_t misses = 0;

    std::size_t DeviceBytes() const
    {
        return slots > 0
            ? static_cast<std::size_t>(slots) *
                  static_cast<std::size_t>(pack_stride + scale_stride)
            : 0;
    }
};

// Moves a slot to the MRU position of h_lru (remove + insert-at-front).
inline void CacheTouch(GlmExpertDeviceCache &c, int slot)
{
    int j = 0;
    while (j < c.slots && c.h_lru[j] != slot) ++j;
    if (j >= c.slots) return;
    for (int i = j; i > 0; --i) c.h_lru[i] = c.h_lru[i - 1];
    c.h_lru[0] = slot;
}

// Slot index for a flat expert id, or -1 on miss. A hit refreshes recency and
// counts toward hits. A miss is a pure query: no stats, no mapping changes.
inline int CacheLookup(GlmExpertDeviceCache &c, int flat)
{
    if (!c.h_slot_of || flat < 0 || flat >= c.total_experts) return -1;
    const int slot = c.h_slot_of[flat];
    if (slot < 0) return -1;
    CacheTouch(c, slot);
    ++c.hits;
    return slot;
}

// Claims a slot for a flat expert id: lowest free slot first, LRU tail victim
// when full. Commits the mapping both ways, refreshes recency and counts one
// miss. Returns the slot index.
inline int CacheInsert(GlmExpertDeviceCache &c, int flat)
{
    int slot = -1;
    for (int i = 0; i < c.slots; ++i) {
        if (c.h_slot_owner[i] < 0) { slot = i; break; }
    }
    if (slot < 0) {
        slot = c.h_lru[c.slots - 1];
        const int victim = c.h_slot_owner[slot];
        if (victim >= 0 && victim < c.total_experts) c.h_slot_of[victim] = -1;
    }
    c.h_slot_of[flat] = slot;
    c.h_slot_owner[slot] = flat;
    CacheTouch(c, slot);
    ++c.misses;
    return slot;
}

} // namespace glm
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_EXPERT_CACHE_H
