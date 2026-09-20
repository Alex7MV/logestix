#ifndef LSXCOMMON_GIGACHAT_MOE_EXPERT_CACHE_H
#define LSXCOMMON_GIGACHAT_MOE_EXPERT_CACHE_H

// GigaChat3.5 routed-expert host store + device slot LRU cache.
//
// 9472 routed experts (37 MoE layers x 256) are the VRAM bill of this model
// (~208 GiB NVFP4). The dense backbone stays resident and experts stream
// through a device slot LRU sized from free VRAM (Nemotron pattern).
//
// Layout: one slot is pack [gate|up|down] then scales [gate|up|down]; every
// section is 256-byte aligned. GigaChat experts are gated 3-projection MLPs,
// so the shared ExpertSchema carries gate=w1, up=w3, down=w2.
//
// Host-only LRU bookkeeping (no CUDA API) so the policy is unit-testable; the
// store/cache struct is an aggregate safe to include from NVCC .cu TUs.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon::gigachat {

// 15-member aggregate in ExpertStore::DecodeEntry() order. gate -> w1,
// up -> w3, down -> w2.
struct GcExpertEntry {
    const uint8_t *host_w1 = nullptr;           // gate packed
    const uint8_t *host_w2 = nullptr;           // down packed
    const uint8_t *host_w3 = nullptr;           // up packed
    const uint8_t *host_w1_scales = nullptr;    // gate scales
    const uint8_t *host_w2_scales = nullptr;    // down scales
    const uint8_t *host_w3_scales = nullptr;    // up scales
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
};

using GcExpertStore = gpu::expert_store::ExpertStore<GcExpertEntry>;

inline constexpr int64_t kGcExpertSlotAlign = 256;

struct GcExpertSlotLayout {
    int64_t gate_pack_off = 0;
    int64_t up_pack_off = 0;
    int64_t down_pack_off = 0;
    int64_t pack_stride = 0;
    int64_t gate_scale_off = 0;
    int64_t up_scale_off = 0;
    int64_t down_scale_off = 0;
    int64_t scale_stride = 0;
    bool ok = false;
};

inline GcExpertSlotLayout PlanGcExpertSlot(int64_t gate_pack,
                                           int64_t gate_scale,
                                           int64_t up_pack, int64_t up_scale,
                                           int64_t down_pack,
                                           int64_t down_scale)
{
    GcExpertSlotLayout l;
    if (gate_pack <= 0 || gate_scale <= 0 || up_pack <= 0 || up_scale <= 0 ||
        down_pack <= 0 || down_scale <= 0)
        return l;
    auto aln = [](int64_t v) {
        return (v + kGcExpertSlotAlign - 1) / kGcExpertSlotAlign *
               kGcExpertSlotAlign;
    };
    l.gate_pack_off = 0;
    l.up_pack_off = aln(gate_pack);
    l.down_pack_off = l.up_pack_off + aln(up_pack);
    l.pack_stride = l.down_pack_off + aln(down_pack);
    l.gate_scale_off = 0;
    l.up_scale_off = aln(gate_scale);
    l.down_scale_off = l.up_scale_off + aln(up_scale);
    l.scale_stride = l.down_scale_off + aln(down_scale);
    l.ok = true;
    return l;
}

inline int64_t GcExpertCachePerExpertBytes(const GcExpertSlotLayout &l)
{
    return l.ok ? l.pack_stride + l.scale_stride : 0;
}

struct GcExpertView {
    const uint8_t *gate_packed = nullptr;
    const uint8_t *gate_scales = nullptr;
    const uint8_t *up_packed = nullptr;
    const uint8_t *up_scales = nullptr;
    const uint8_t *down_packed = nullptr;
    const uint8_t *down_scales = nullptr;
    float gate_global_scale = 1.0f;
    float up_global_scale = 1.0f;
    float down_global_scale = 1.0f;
};

inline GcExpertView GcExpertSlotPointers(const gpu::ExpertDeviceCache &cache,
                                         int slot, const GcExpertSlotLayout &l)
{
    GcExpertView v;
    if (!l.ok || !cache.d_pack || !cache.d_scale || slot < 0 ||
        slot >= cache.slots)
        return v;
    const int64_t pack_base = static_cast<int64_t>(slot) * cache.pack_stride;
    const int64_t scale_base = static_cast<int64_t>(slot) * cache.scale_stride;
    v.gate_packed = cache.d_pack + pack_base + l.gate_pack_off;
    v.up_packed = cache.d_pack + pack_base + l.up_pack_off;
    v.down_packed = cache.d_pack + pack_base + l.down_pack_off;
    v.gate_scales = cache.d_scale + scale_base + l.gate_scale_off;
    v.up_scales = cache.d_scale + scale_base + l.up_scale_off;
    v.down_scales = cache.d_scale + scale_base + l.down_scale_off;
    return v;
}

struct GcExpertCache {
    GcExpertStore store;
    gpu::ExpertDeviceCache device;
    GcExpertSlotLayout layout;
    int64_t per_expert_bytes = 0;
    int total_experts = 0;
    int experts_per_layer = 0;
    int moe_layers = 0;
    std::vector<int> moe_index_of_layer;  // [num_layers], -1 for non-MoE
    std::uint64_t cache_bytes = 0;
};

// VRAM reserved for session state (KV + GDN recurrent/conv state + logits).
inline constexpr std::size_t kGcExpertCacheVramReserveBytes = 2ull << 30;

inline int GcExpertCacheSlots(std::size_t free_bytes,
                              std::size_t per_expert_bytes, int total_experts,
                              int topk, const char *env_value)
{
    if (total_experts <= 0) return 0;
    if (env_value != nullptr && env_value[0] != '\0')
    {
        const int v = std::atoi(env_value);
        if (v > 0) return std::min(v, total_experts);
    }
    const int floor = std::min(std::max(topk, 1), total_experts);
    if (per_expert_bytes == 0) return total_experts;
    const int by_budget = lifecycle::ExpertCacheSlots(
        free_bytes, per_expert_bytes, total_experts,
        perf::CurrentKnobs().expert_slots_pct, kGcExpertCacheVramReserveBytes);
    return std::max(floor, by_budget);
}

// ── Device surface (src/gigachat_moe_expert_cache.cpp) ─────────────────────
void LoadGcExpertStore(const std::string &model_path, int num_layers,
                       int experts_per_layer, GcExpertStore &store);

GcExpertSlotLayout PlanGcExpertSlotFromStore(const GcExpertStore &store);

bool AllocateGcExpertCache(GcExpertCache &cache, int slots, std::string &error);

void FreeGcExpertCache(GcExpertCache &cache);

cudaError_t PrewarmGcExpertCache(GcExpertCache &cache, cudaStream_t stream,
                                 std::string &error);

cudaError_t StageGcExpert(GcExpertCache &cache, int layer, int eid,
                          cudaStream_t stream, std::string &error,
                          int *slot_out = nullptr, bool *missed_out = nullptr,
                          int *evicted_slot_out = nullptr);

GcExpertView ResolveGcExpertView(GcExpertCache &cache, int layer, int eid,
                                 cudaStream_t stream, std::string &error);

cudaError_t ResolveGcExperts(GcExpertCache &cache, int layer, const int *top_ids,
                             int k, cudaStream_t stream, GcExpertView *out,
                             std::string &error);

} // namespace lsxcommon::gigachat

#endif // LSXCOMMON_GIGACHAT_MOE_EXPERT_CACHE_H
