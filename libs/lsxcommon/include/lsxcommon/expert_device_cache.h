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
    // O(1) LRU bookkeeping (doubly-linked over slot indices): prev/next in
    // recency order, h_lru_front = MRU slot, h_lru_back = LRU slot.
    // Maintained alongside h_lru; free slots form the back segment.
    int32_t *h_prev = nullptr;        // [slots] or null (legacy mode)
    int32_t *h_next = nullptr;        // [slots] or null (legacy mode)
    int32_t h_lru_front = -1;
    int32_t h_lru_back = -1;
    // O(1) free-slot stack: h_free[i] = slot index, h_free_top = count.
    int32_t *h_free = nullptr;        // [slots] or null (legacy mode)
    int h_free_top = 0;
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

    // O(1) unlink from the doubly-linked recency chain.
    void DlUnlink(int slot)
    {
        const int p = h_prev[slot], n = h_next[slot];
        if (p >= 0) h_next[p] = n; else h_lru_front = n;
        if (n >= 0) h_prev[n] = p; else h_lru_back = p;
    }

    // O(1) link at the MRU (front) end of the recency chain.
    void DlPushFront(int slot)
    {
        h_prev[slot] = -1;
        h_next[slot] = h_lru_front;
        if (h_lru_front >= 0) h_prev[h_lru_front] = slot;
        else h_lru_back = slot;
        h_lru_front = slot;
    }

    // Moves a slot to the MRU position of h_lru (remove + insert-at-front).
    // The legacy array shift runs only when the O(1) structures are absent.
    void Touch(int slot)
    {
        if (h_prev) {
            if (h_lru_front != slot) {
                DlUnlink(slot);
                DlPushFront(slot);
            }
            return;
        }
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
    // counts one miss. Returns the slot index. When non-null, *evicted_slot
    // receives the recycled slot index on an eviction (else -1); the caller
    // must fence any in-flight reader of that slot before overwriting it.
    int Insert(int flat, int *evicted_slot = nullptr)
    {
        if (evicted_slot) *evicted_slot = -1;
        int slot = -1;
        if (h_free) {
            if (h_free_top > 0) slot = h_free[--h_free_top];
        } else {
            for (int i = 0; i < slots; ++i) {
                if (h_slot_owner[i] < 0) { slot = i; break; }
            }
        }
        if (slot < 0) {
            // Evict the LRU tail.
            slot = h_prev ? h_lru_back : h_lru[slots - 1];
            const int victim = h_slot_owner[slot];
            if (victim >= 0 && victim < total_experts) h_slot_of[victim] = -1;
            if (h_prev) DlUnlink(slot);
            if (evicted_slot) *evicted_slot = slot;
        }
        h_slot_of[flat] = slot;
        h_slot_owner[slot] = flat;
        if (h_prev) DlPushFront(slot);
        else Touch(slot);
        ++misses;
        return slot;
    }
};

} // namespace gpu
} // namespace lsxcommon

#endif // LSXCOMMON_EXPERT_DEVICE_CACHE_H
