#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"

#include "lsxcommon/log.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_ultra {

void LoadUltraExpertStore(const std::string &model_path, int num_layers,
                       int experts_per_layer, UltraExpertStore &store)
{
    store.log_prefix = "[nemotron_ultra_load] ";
    store.mmap_fail_verb = "ultra expert mmap fail ";
    store.num_layers = num_layers;
    store.experts_per_layer = experts_per_layer;
    // Warm every expert: 2944 files fit under the 65536 mmap cap, so the host
    // store stays fully resident and a slot miss is a pure H2D copy (no file
    // open on the decode path).
    store.Load(model_path, /*fallback=*/true);
}

UltraExpertSlotLayout PlanUltraExpertSlotFromStore(const UltraExpertStore &store)
{
    int64_t up_pack = 0, up_scale = 0, down_pack = 0, down_scale = 0;
    for (const auto &e : store.entries) {
        if (!e.host_w3) continue;  // unparsed / missing
        up_pack = std::max(up_pack, e.w3_bytes);
        up_scale = std::max(up_scale, e.w3_scales_bytes);
        down_pack = std::max(down_pack, e.w2_bytes);
        down_scale = std::max(down_scale, e.w2_scales_bytes);
    }
    return PlanUltraExpertSlot(up_pack, up_scale, down_pack, down_scale);
}

bool AllocateUltraExpertCache(UltraExpertCache &cache, int slots, std::string &error)
{
    if (!cache.layout.ok || slots <= 0 || cache.experts_per_layer <= 0 ||
        cache.store.num_layers <= 0 || cache.store.experts_per_layer <= 0) {
        error = "ultra expert cache: store/layout not prepared";
        return false;
    }
    if (cache.total_experts <= 0)
        cache.total_experts =
            cache.store.num_layers * cache.store.experts_per_layer;
    cache.per_expert_bytes = UltraExpertPerExpertBytes(cache.layout);

    gpu::ExpertDeviceCache c{};
    c.slots = slots;
    c.total_experts = cache.total_experts;
    c.pack_stride = cache.layout.pack_stride;
    c.scale_stride = cache.layout.scale_stride;
    const std::size_t bytes = c.DeviceBytes();
    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        error = "ultra expert device block alloc failed (";
        error += std::to_string(bytes >> 20);
        error += " MiB)";
        return false;
    }
    c.d_base = static_cast<uint8_t *>(base);
    c.d_pack = c.d_base;
    c.d_scale = c.d_base + static_cast<int64_t>(slots) * c.pack_stride;

    c.h_slot_of = new int32_t[static_cast<std::size_t>(c.total_experts)];
    c.h_slot_owner = new int32_t[static_cast<std::size_t>(slots)];
    c.h_lru = new int32_t[static_cast<std::size_t>(slots)];
    c.h_prev = new int32_t[static_cast<std::size_t>(slots)];
    c.h_next = new int32_t[static_cast<std::size_t>(slots)];
    c.h_free = new int32_t[static_cast<std::size_t>(slots)];
    for (int i = 0; i < c.total_experts; ++i) c.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        c.h_slot_owner[i] = -1;
        c.h_lru[i] = i;
        c.h_free[i] = slots - 1 - i;  // pops 0 first (lowest free slot)
        c.h_prev[i] = -1;
        c.h_next[i] = -1;
    }
    c.h_free_top = slots;
    c.h_lru_front = -1;
    c.h_lru_back = -1;

    cache.device = c;
    cache.cache_bytes = bytes;
    return true;
}

void FreeUltraExpertCache(UltraExpertCache &cache)
{
    if (cache.device.d_base) {
        cudaFree(cache.device.d_base);
        cache.device.d_base = nullptr;
    }
    delete[] cache.device.h_slot_of;
    delete[] cache.device.h_slot_owner;
    delete[] cache.device.h_lru;
    delete[] cache.device.h_prev;
    delete[] cache.device.h_next;
    delete[] cache.device.h_free;
    cache.device = gpu::ExpertDeviceCache{};
    cache.cache_bytes = 0;
    cache.store.Free();
}

cudaError_t PrewarmUltraExpertCache(UltraExpertCache &cache, cudaStream_t stream,
                                 std::string &error)
{
    if (cache.device.slots <= 0 || cache.store.num_layers <= 0)
        return cudaSuccess;
    int staged = 0;
    for (int layer = 0;
         layer < cache.store.num_layers && staged < cache.device.slots;
         ++layer) {
        if (!cache.moe_index_of_layer.empty()) {
            if (layer >= static_cast<int>(cache.moe_index_of_layer.size()) ||
                cache.moe_index_of_layer[static_cast<std::size_t>(layer)] < 0)
                continue;
        }
        for (int eid = 0;
             eid < cache.experts_per_layer && staged < cache.device.slots;
             ++eid) {
            bool missed = false;
            const cudaError_t ce = StageUltraExpert(cache, layer, eid, stream,
                                                 error, nullptr, &missed);
            if (ce != cudaSuccess) return ce;
            if (missed) ++staged;
        }
    }
    return cudaStreamSynchronize(stream);
}

cudaError_t StageUltraExpert(UltraExpertCache &cache, int layer, int eid,
                          cudaStream_t stream, std::string &error,
                          int *slot_out, bool *missed_out,
                          int *evicted_slot_out)
{
    if (missed_out) *missed_out = false;
    if (evicted_slot_out) *evicted_slot_out = -1;
    if (!cache.layout.ok || cache.device.slots <= 0 || cache.device.d_pack ==
            nullptr) {
        error = "ultra expert cache not allocated";
        return cudaErrorInvalidValue;
    }
    if (layer < 0 || layer >= cache.store.num_layers || eid < 0 ||
        eid >= cache.experts_per_layer) {
        error = "ultra expert out of range: layer " + std::to_string(layer) +
                " eid " + std::to_string(eid);
        return cudaErrorInvalidValue;
    }
    // Host store is keyed by the real transformer layer (Arrow path
    // layer_<l>); the device flat id is the compacted MoE-layer index. An
    // empty map degenerates to identity (all layers MoE).
    const int store_flat = layer * cache.store.experts_per_layer + eid;
    int moe_idx = layer;
    if (!cache.moe_index_of_layer.empty()) {
        if (layer >= static_cast<int>(cache.moe_index_of_layer.size())) {
            error = "ultra expert layer map too short";
            return cudaErrorInvalidValue;
        }
        moe_idx = cache.moe_index_of_layer[static_cast<std::size_t>(layer)];
        if (moe_idx < 0) {
            error = "ultra expert requested on a non-MoE layer " +
                    std::to_string(layer);
            return cudaErrorInvalidValue;
        }
    }
    const int flat = moe_idx * cache.experts_per_layer + eid;
    int slot = cache.device.Lookup(flat);
    if (slot < 0) {
        int evicted = -1;
        slot = cache.device.Insert(flat, &evicted);
        if (missed_out) *missed_out = true;
        if (evicted_slot_out) *evicted_slot_out = evicted;

        const std::size_t idx = static_cast<std::size_t>(store_flat);
        if (idx >= cache.store.entries.size()) {
            error = "ultra expert store entry missing";
            return cudaErrorInvalidValue;
        }
        if (!cache.store.entries[idx].host_w3) {
            // Cold entry: parse the Arrow file on demand (only reachable when
            // the store cap forced eviction of a host view).
            if (!cache.store.ParseFile(idx)) {
                error = "ultra expert host parse failed for layer " +
                        std::to_string(layer) + " eid " + std::to_string(eid);
                return cudaErrorInvalidValue;
            }
        }
        const UltraExpertEntry &e = cache.store.entries[idx];
        if (!e.host_w3 || !e.host_w2 || !e.host_w3_scales ||
            !e.host_w2_scales) {
            error = "ultra expert payload missing for layer " +
                    std::to_string(layer) + " eid " + std::to_string(eid);
            return cudaErrorInvalidValue;
        }
        uint8_t *dp =
            cache.device.d_pack +
            static_cast<int64_t>(slot) * cache.device.pack_stride;
        uint8_t *ds =
            cache.device.d_scale +
            static_cast<int64_t>(slot) * cache.device.scale_stride;
        auto stage = [&](const uint8_t *src, int64_t n, uint8_t *dst) {
            if (n <= 0) return cudaSuccess;
            return cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n),
                                   cudaMemcpyHostToDevice, stream);
        };
        cudaError_t ce = stage(e.host_w3, e.w3_bytes, dp + cache.layout.up_pack_off);
        if (ce == cudaSuccess)
            ce = stage(e.host_w2, e.w2_bytes, dp + cache.layout.down_pack_off);
        if (ce == cudaSuccess)
            ce = stage(e.host_w3_scales, e.w3_scales_bytes,
                       ds + cache.layout.up_scale_off);
        if (ce == cudaSuccess)
            ce = stage(e.host_w2_scales, e.w2_scales_bytes,
                       ds + cache.layout.down_scale_off);
        if (ce != cudaSuccess) {
            error = "ultra expert H2D staging failed";
            return ce;
        }
    }
    if (slot_out) *slot_out = slot;
    return cudaSuccess;
}

UltraExpertView ResolveUltraExpertView(UltraExpertCache &cache, int layer, int eid,
                                 cudaStream_t stream, std::string &error)
{
    int slot = -1;
    if (StageUltraExpert(cache, layer, eid, stream, error, &slot) != cudaSuccess)
        return UltraExpertView{};
    UltraExpertView v = UltraExpertSlotPointers(cache.device, slot, cache.layout);
    const std::size_t idx = static_cast<std::size_t>(
        layer * cache.store.experts_per_layer + eid);
    if (idx < cache.store.entries.size()) {
        v.up_global_scale = cache.store.entries[idx].w3_global_scale;
        v.down_global_scale = cache.store.entries[idx].w2_global_scale;
    }
    return v;
}

cudaError_t ResolveUltraExperts(UltraExpertCache &cache, int layer, const int *top_ids,
                             int k, cudaStream_t stream, UltraExpertView *out,
                             std::string &error)
{
    if (k < 0 || (k > 0 && (top_ids == nullptr || out == nullptr))) {
        error = "ultra expert resolve: null top-k input";
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < k; ++i) {
        out[i] = ResolveUltraExpertView(cache, layer, top_ids[i], stream, error);
        if (out[i].up_packed == nullptr || out[i].down_packed == nullptr)
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

} // namespace lsxcommon::nemotron_ultra
