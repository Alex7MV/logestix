#ifndef LSXCOMMON_NEMOTRON_MOE_EXPERT_CACHE_H
#define LSXCOMMON_NEMOTRON_MOE_EXPERT_CACHE_H

// Nemotron-3.5 routed-expert host store + device LRU cache.
//
// The 2944 routed experts (23 MoE layers x 128) are the whole VRAM bill of
// this model: ~16 GiB resident, which is why the resident loader only fits on
// >=24 GiB boards. This module keeps the dense backbone always resident and
// streams routed experts through a device slot LRU sized from free VRAM, so
// the model also runs on a 16 GiB part.
//
// Bit-exactness: a slot holds the SAME packed E2M1 bytes + E4M3 descriptors
// the resident loader used to upload via cudaMalloc; the GEMV chain reads them
// unchanged (same k-order, same kernels). A cache sized to the full pool is
// exactly the old resident path, only with staging instead of per-expert
// cudaMalloc.
//
// Layout: one slot is pack [up|down] then scales [up|down]; every section is
// 256-byte aligned (NVFP4 scale rows and the streaming GEMV assume aligned
// section starts). Nemotron experts are non-gated (relu^2 MLP), so the shared
// ExpertSchema carries up in w3 and down in w2; w1 stays empty.
//
// Host-only LRU bookkeeping (no CUDA API) so the policy is unit-testable; the
// store/cache struct is an aggregate safe to include from NVCC .cu TUs
// (expert_store.h is arrow/NVCC-safe).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon::nemotron_moe {

// 15-member aggregate in ExpertStore::DecodeEntry() order (see expert_store.h
// DecodeEntry brace-init). Up -> w3, down -> w2; w1 unused (non-gated).
struct NmExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;          // down packed
    const uint8_t *host_w3 = nullptr;          // up packed
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;   // down scales
    const uint8_t *host_w3_scales = nullptr;   // up scales
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;              // down global scale
    float w3_global_scale = 1.0f;              // up global scale
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;                      // down packed bytes
    int64_t w3_bytes = 0;                      // up packed bytes
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;               // down scale bytes
    int64_t w3_scales_bytes = 0;               // up scale bytes
};

using NmExpertStore = gpu::expert_store::ExpertStore<NmExpertEntry>;

inline constexpr int64_t kNmExpertSlotAlign = 256;

// One device slot's section offsets/strides. pack_stride/scale_stride are the
// per-slot strides consumed by gpu::ExpertDeviceCache.
struct NmExpertSlotLayout {
    int64_t up_pack_off = 0;
    int64_t down_pack_off = 0;
    int64_t pack_stride = 0;
    int64_t up_scale_off = 0;
    int64_t down_scale_off = 0;
    int64_t scale_stride = 0;
    bool ok = false;
};

// Pure host arithmetic; @p ok is false when any section is empty.
inline NmExpertSlotLayout PlanNmExpertSlot(int64_t up_pack, int64_t up_scale,
                                           int64_t down_pack, int64_t down_scale)
{
    NmExpertSlotLayout l;
    if (up_pack <= 0 || up_scale <= 0 || down_pack <= 0 || down_scale <= 0)
        return l;
    auto aln = [](int64_t v) {
        return (v + kNmExpertSlotAlign - 1) / kNmExpertSlotAlign *
               kNmExpertSlotAlign;
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

inline int64_t NmExpertCachePerExpertBytes(const NmExpertSlotLayout &l)
{
    return l.ok ? l.pack_stride + l.scale_stride : 0;
}

// Device pointers + global scales for one resident expert. The GEMVs consume
// packed/scales; the weighted sums consume the global scales.
struct NmExpertView {
    const uint8_t *up_packed = nullptr;
    const uint8_t *up_scales = nullptr;
    const uint8_t *down_packed = nullptr;
    const uint8_t *down_scales = nullptr;
    float up_global_scale = 1.0f;
    float down_global_scale = 1.0f;
};

// Slot section pointers (global scales left at their default 1.0).
inline NmExpertView NmExpertSlotPointers(const gpu::ExpertDeviceCache &cache,
                                         int slot, const NmExpertSlotLayout &l)
{
    NmExpertView v;
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

// Model-lifetime expert state: the mmap-backed host store plus the device
// slot LRU. One instance per loaded model. Only MoE layers carry experts, so
// the device flat id is a compacted MoE-layer index while the host store is
// keyed by the real transformer layer (the Arrow path uses layer_<l>).
struct NmExpertCache {
    NmExpertStore store;
    gpu::ExpertDeviceCache device;
    NmExpertSlotLayout layout;
    int64_t per_expert_bytes = 0;
    int total_experts = 0;          // moe_layers * experts_per_layer
    int experts_per_layer = 0;
    int moe_layers = 0;
    std::vector<int> moe_index_of_layer;  // [num_layers], -1 for non-MoE
    std::uint64_t cache_bytes = 0;
};

// VRAM kept aside for the session (KV + recurrent state + logits) and the CUDA
// context; the dense backbone is already resident when slots are resolved.
// 2 GiB mirrors the kimi model-specific reserve (kExpertCacheVramReserveBytes
// is calibrated for ~150k-vocab generation arenas, far above this model's
// ~0.1 GiB session).
inline constexpr std::size_t kNmExpertCacheVramReserveBytes = 2ull << 30;

// Slot count from the free-VRAM budget. @p env_value is the
// LSX_NEMOTRON_EXPERT_CACHE_SLOTS override (positive wins, clamped to the
// pool). Floors at one layer's top-k working set (the batched GEMV needs that
// many distinct slots) and never exceeds total_experts. Unknown geometry
// (per_expert_bytes == 0) degenerates to "everything resident".
inline int NmExpertCacheSlots(std::size_t free_bytes,
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
        perf::CurrentKnobs().expert_slots_pct, kNmExpertCacheVramReserveBytes);
    return std::max(floor, by_budget);
}

// ── Device surface (implemented in src/nemotron_moe_expert_cache.cpp) ───────

// Opens the routed-expert Arrow files under <model_path>/logestix/experts and
// warms the mmap views. No device allocation.
void LoadNmExpertStore(const std::string &model_path, int num_layers,
                       int experts_per_layer, NmExpertStore &store);

// Slot layout from the loaded store (max section bytes across the pool; the
// frozen geometry makes every expert's payload identical).
NmExpertSlotLayout PlanNmExpertSlotFromStore(const NmExpertStore &store);

// Allocates @p slots device slots sized by cache.layout. cache.store must be
// loaded and cache.layout/experts_per_layer filled in.
bool AllocateNmExpertCache(NmExpertCache &cache, int slots, std::string &error);

void FreeNmExpertCache(NmExpertCache &cache);

// Fills every device slot on the load path (untimed): stages up to
// cache.device.slots experts in MoE-layer order and synchronizes @p stream.
// Mirrors the old resident loader's untimed cudaMalloc+H2D so a full-size
// cache costs nothing on the first decode token.
cudaError_t PrewarmNmExpertCache(NmExpertCache &cache, cudaStream_t stream,
                                 std::string &error);

// Ensures (layer, eid) is resident, staging it into a slot on @p stream on a
// miss (LRU eviction when full). @p slot_out receives the slot, @p missed_out
// whether bytes were staged, @p evicted_slot_out the recycled slot (-1 when
// none). Returns cudaSuccess; fills @p error on failure.
cudaError_t StageNmExpert(NmExpertCache &cache, int layer, int eid,
                          cudaStream_t stream, std::string &error,
                          int *slot_out = nullptr, bool *missed_out = nullptr,
                          int *evicted_slot_out = nullptr);

// Stage + return the slot view (packed/scales pointers + global scales).
NmExpertView ResolveNmExpertView(NmExpertCache &cache, int layer, int eid,
                                 cudaStream_t stream, std::string &error);

// Stage every routed id of one layer's top-k and fill @p out (k entries).
cudaError_t ResolveNmExperts(NmExpertCache &cache, int layer, const int *top_ids,
                             int k, cudaStream_t stream, NmExpertView *out,
                             std::string &error);

} // namespace lsxcommon::nemotron_moe

#endif // LSXCOMMON_NEMOTRON_MOE_EXPERT_CACHE_H
