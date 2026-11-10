#pragma once

// Nemotron-3-Ultra-550B routed-expert host store + device slot LRU.
//
// 48 MoE layers x 512 experts = 24 576 experts at ~11.8 MB each: ~290 GB of
// converted payload. The dense backbone (~22.5 GB) is resident, and the routed
// experts stream through this store: a device slot LRU sized from free VRAM,
// backed by mmap views of the per-expert Arrow files. On a 32 GB card that is
// ~590 slots (2.4 % of the pool), so a typical token's 1 056 expert reads are
// mostly NVMe reads - the same regime as mistral-large-4-1t.
//
// Two rules from the ingestion protocol are structural here, not comments:
//   R1  the host store is the NEUTRAL `ExpertStore<TEntry>`, never a
//       family-private owning container; entries are views into the mmap.
//   R2  every read goes through the store's view loader, so nothing retains an
//       owning copy of a 11.8 MB payload.
//
// Layout: one slot is pack [up|down] then scales [up|down]; every section is
// 256-byte aligned (the streaming GEMV and NVFP4 scale rows both assume aligned
// section starts). Experts are non-gated relu2 MLPs, so the shared schema carries
// up in w3 and down in w2 and w1 stays empty.
//
// Host-only LRU bookkeeping (no CUDA calls) keeps the policy unit-testable, and
// the struct is an aggregate safe to include from an NVCC TU (expert_store.h is
// Arrow/NVCC-safe).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon::nemotron_ultra {

// 15-member aggregate in ExpertStore::DecodeEntry() order. Up -> w3, down -> w2;
// w1 unused (non-gated).
struct UltraExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;         // down packed
    const uint8_t *host_w3 = nullptr;         // up packed
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;  // down scales
    const uint8_t *host_w3_scales = nullptr;  // up scales
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;             // down global scale
    float w3_global_scale = 1.0f;             // up global scale
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;                     // down packed bytes
    int64_t w3_bytes = 0;                     // up packed bytes
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;              // down scale bytes
    int64_t w3_scales_bytes = 0;              // up scale bytes
};

using UltraExpertStore = gpu::expert_store::ExpertStore<UltraExpertEntry>;

inline constexpr int64_t kUltraExpertSlotAlign = 256;

// One device slot's section offsets/strides. pack_stride/scale_stride are the
// per-slot strides consumed by gpu::ExpertDeviceCache.
struct UltraExpertSlotLayout {
    int64_t up_pack_off = 0;
    int64_t down_pack_off = 0;
    int64_t pack_stride = 0;
    int64_t up_scale_off = 0;
    int64_t down_scale_off = 0;
    int64_t scale_stride = 0;
    bool ok = false;
};

// Pure host arithmetic; @p ok is false when any section is empty.
inline UltraExpertSlotLayout PlanUltraExpertSlot(int64_t up_pack, int64_t up_scale,
                                                int64_t down_pack,
                                                int64_t down_scale)
{
    UltraExpertSlotLayout l;
    if (up_pack <= 0 || up_scale <= 0 || down_pack <= 0 || down_scale <= 0) return l;
    auto aln = [](int64_t v) {
        return (v + kUltraExpertSlotAlign - 1) / kUltraExpertSlotAlign *
               kUltraExpertSlotAlign;
    };
    l.up_pack_off = 0;
    l.down_pack_off = aln(up_pack);
    l.pack_stride = l.down_pack_off + aln(down_pack);
    l.up_scale_off = 0;
    l.down_scale_off = aln(up_scale);
    l.scale_stride = l.down_scale_off + aln(down_scale);
    l.ok = true;
    return l;
}

inline int64_t UltraExpertPerExpertBytes(const UltraExpertSlotLayout &l)
{
    return l.ok ? l.pack_stride + l.scale_stride : 0;
}

// Device pointers + global scales for one resident expert.
struct UltraExpertView {
    const uint8_t *up_packed = nullptr;
    const uint8_t *up_scales = nullptr;
    const uint8_t *down_packed = nullptr;
    const uint8_t *down_scales = nullptr;
    float up_global_scale = 1.0f;
    float down_global_scale = 1.0f;
};

// Slot section pointers (global scales left at their default 1.0).
inline UltraExpertView UltraExpertSlotPointers(const gpu::ExpertDeviceCache &cache,
                                               int slot,
                                               const UltraExpertSlotLayout &l)
{
    UltraExpertView v;
    if (!l.ok || !cache.d_pack || !cache.d_scale || slot < 0 ||
        slot >= cache.slots)
        return v;
    const int64_t pack_base = static_cast<int64_t>(slot) * cache.pack_stride;
    const int64_t scale_base = static_cast<int64_t>(slot) * cache.scale_stride;
    v.up_packed = cache.d_pack + pack_base + l.up_pack_off;
    v.down_packed = cache.d_pack + pack_base + l.down_pack_off;
    v.up_scales = cache.d_scale + scale_base + l.up_scale_off;
    v.down_scales = cache.d_scale + scale_base + l.down_scale_off;
    return v;
}

// Model-lifetime expert state: the mmap-backed host store plus the device slot
// LRU. Only MoE layers carry experts, so the device flat id is a compacted MoE
// index while the host store is keyed by the real transformer layer.
struct UltraExpertCache {
    UltraExpertStore store;
    gpu::ExpertDeviceCache device;
    UltraExpertSlotLayout layout;
    int64_t per_expert_bytes = 0;
    int total_experts = 0;
    int experts_per_layer = 0;
    int moe_layers = 0;
    std::vector<int> moe_index_of_layer;  // [num_layers], -1 for non-MoE
    std::uint64_t cache_bytes = 0;
};

// VRAM kept aside for the session (KV + recurrent state + logits) and the CUDA
// context. The dense graph is already resident when slots are resolved, and this
// model's session is ~0.5 GB (403 MB of Mamba-2 state + 50 MB KV at 2048), so
// the reserve is deliberately larger than the session.
inline constexpr std::size_t kUltraExpertCacheVramReserveBytes = 2ull << 30;

// Slot count from the free-VRAM budget. @p env_value is the
// LSX_NEMOTRON_ULTRA_EXPERT_CACHE_SLOTS override (positive wins, clamped to the
// pool). Floors at one layer's top-k working set (the batched GEMV needs that
// many distinct slots) and never exceeds total_experts. Unknown geometry
// (per_expert_bytes == 0) degenerates to "everything resident".
inline int UltraExpertCacheSlots(std::size_t free_bytes,
                                 std::size_t per_expert_bytes, int total_experts,
                                 int topk, const char *env_value)
{
    if (total_experts <= 0) return 0;
    if (env_value != nullptr && env_value[0] != '\0') {
        const int v = std::atoi(env_value);
        if (v > 0) return std::min(v, total_experts);
    }
    const int floor = std::min(std::max(topk, 1), total_experts);
    if (per_expert_bytes == 0) return total_experts;
    const int by_budget = lifecycle::ExpertCacheSlots(
        free_bytes, per_expert_bytes, total_experts,
        perf::CurrentKnobs().expert_slots_pct, kUltraExpertCacheVramReserveBytes);
    return std::max(floor, by_budget);
}

// ── Device surface (implemented in src/nemotron_moe_ultra_expert_cache.cpp) ──

// Opens the routed-expert Arrow files under <model_path>/logestix/experts and
// warms the mmap views. No device allocation.
void LoadUltraExpertStore(const std::string &model_path, int num_layers,
                          int experts_per_layer, UltraExpertStore &store);

// Slot layout from the loaded store (max section bytes across the pool; the
// frozen geometry makes every expert's payload identical).
UltraExpertSlotLayout PlanUltraExpertSlotFromStore(const UltraExpertStore &store);

// Allocates @p slots device slots sized by cache.layout. cache.store must be
// loaded and cache.layout/experts_per_layer filled in.
bool AllocateUltraExpertCache(UltraExpertCache &cache, int slots, std::string &error);

void FreeUltraExpertCache(UltraExpertCache &cache);

// Fills every device slot on the load path (untimed): stages up to
// cache.device.slots experts in MoE-layer order and synchronizes @p stream.
cudaError_t PrewarmUltraExpertCache(UltraExpertCache &cache, cudaStream_t stream,
                                    std::string &error);

// Ensures (layer, eid) is resident, staging it into a slot on @p stream on a
// miss (LRU eviction when full).
cudaError_t StageUltraExpert(UltraExpertCache &cache, int layer, int eid,
                             cudaStream_t stream, std::string &error,
                             int *slot_out = nullptr, bool *missed_out = nullptr,
                             int *evicted_slot_out = nullptr);

// Stage + return the slot view (packed/scales pointers + global scales).
UltraExpertView ResolveUltraExpertView(UltraExpertCache &cache, int layer, int eid,
                                       cudaStream_t stream, std::string &error);

// Stage every routed id of one layer's top-k and fill @p out (k entries).
cudaError_t ResolveUltraExperts(UltraExpertCache &cache, int layer,
                                const int *top_ids, int k, cudaStream_t stream,
                                UltraExpertView *out, std::string &error);

} // namespace lsxcommon::nemotron_ultra
