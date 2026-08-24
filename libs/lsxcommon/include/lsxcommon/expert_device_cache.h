#ifndef LSXCOMMON_EXPERT_DEVICE_CACHE_H
#define LSXCOMMON_EXPERT_DEVICE_CACHE_H

// Unified routed-expert device-slot LRU cache (model-unification 2026-08).
// One canonical implementation replacing the port-of-port chain
// DeepSeek ExpertDeviceCache -> GlmExpertDeviceCache -> QwenMoeExpertDeviceCache.
//
// One device block holds [slots] expert sets laid out as
//   d_pack  = [slots][pack_stride]    (w1|w2|w3 E2M1 nibbles, packed)
//   d_scale = [slots][scale_stride]   (per-block E4M3 descriptors)
// The device buffers are owned by the generation context; this header only
// defines the layout plus the host-side LRU bookkeeping as methods so the
// semantics stay unit-testable without CUDA.
//
// Safety contract: all device access is stream-ordered by the caller; the
// bookkeeping arrays are caller-owned plain int32 storage — no absl, no CUDA
// API, safe to include from .cu TUs.

#include <cstddef>
#include <cstdint>

namespace lsxcommon {
namespace gpu {

struct ExpertDeviceCache {
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
    // Hit/miss counters (diagnostics): hits in Lookup, misses in Insert.
    int64_t hits = 0;
    int64_t misses = 0;

    std::size_t DeviceBytes() const
    {
        return slots > 0
            ? static_cast<std::size_t>(slots) *
                  static_cast<std::size_t>(pack_stride + scale_stride)
            : 0;
    }

    // Moves a slot to the MRU position of h_lru (remove + insert-at-front).
    void Touch(int slot)
    {
        int j = 0;
        while (j < slots && h_lru[j] != slot) ++j;
        if (j >= slots) return;
        for (int i = j; i > 0; --i) h_lru[i] = h_lru[i - 1];
        h_lru[0] = slot;
    }

    // Slot index for a flat expert id, or -1 on miss. A hit refreshes recency
    // and counts toward hits. A miss is a pure query: no stats, no mapping
    // changes.
    int Lookup(int flat)
    {
        if (!h_slot_of || flat < 0 || flat >= total_experts) return -1;
        const int slot = h_slot_of[flat];
        if (slot < 0) return -1;
        Touch(slot);
        ++hits;
        return slot;
    }

    // Claims a slot for a flat expert id: lowest free slot first, LRU tail
    // victim when full. Commits the mapping both ways, refreshes recency and
    // counts one miss. Returns the slot index.
    int Insert(int flat)
    {
        int slot = -1;
        for (int i = 0; i < slots; ++i) {
            if (h_slot_owner[i] < 0) { slot = i; break; }
        }
        if (slot < 0) {
            slot = h_lru[slots - 1];
            const int victim = h_slot_owner[slot];
            if (victim >= 0 && victim < total_experts) h_slot_of[victim] = -1;
        }
        h_slot_of[flat] = slot;
        h_slot_owner[slot] = flat;
        Touch(slot);
        ++misses;
        return slot;
    }
};

} // namespace gpu
} // namespace lsxcommon

#endif // LSXCOMMON_EXPERT_DEVICE_CACHE_H
