// nemotron_moe_expert_cache_impl.h — shared routed-expert cache engine for
// Nemotron-3.5-Lightning and Nemotron-3-Ultra (dedup phase 7). The two family
// expert_cache.cpp files were identical modulo names and the log_prefix
// string (proven by normalized diff, ratio 0.9995); the bodies move here
// verbatim, templated on the family's cache/store/layout/view types.
//
// The family .cpp files keep their public API as explicit instantiations +
// thin delegates; the log strings come from a FamilyStrings policy so the
// observable messages stay byte-identical.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

namespace lsxcommon {
namespace nemotron_shared {

// Family policy: the log/error strings and the per-expert-bytes function.
struct NmStrings {
    static constexpr const char *kLogPrefix = "[nemotron_load] ";
    static constexpr const char *kMmapFailVerb = "nemotron expert mmap fail ";
    static constexpr const char *kTag = "nemotron";
};
struct UltraStrings {
    static constexpr const char *kLogPrefix = "[nemotron_ultra_load] ";
    static constexpr const char *kMmapFailVerb = "ultra expert mmap fail ";
    static constexpr const char *kTag = "ultra";
};

template <class Store, class Strings>
void LoadExpertStore(const std::string &model_path, int num_layers,
                     int experts_per_layer, Store &store)
{
    store.log_prefix = Strings::kLogPrefix;
    store.mmap_fail_verb = Strings::kMmapFailVerb;
    store.num_layers = num_layers;
    store.experts_per_layer = experts_per_layer;
    // Warm every expert: the files fit under the 65536 mmap cap, so the host
    // store stays fully resident and a slot miss is a pure H2D copy (no file
    // open on the decode path).
    store.Load(model_path, /*fallback=*/true);
}

template <class Store, class Layout, class PlanFn>
Layout PlanExpertSlotFromStore(const Store &store)
{
    int64_t up_pack = 0, up_scale = 0, down_pack = 0, down_scale = 0;
    for (const auto &e : store.entries) {
        if (!e.host_w3) continue;  // unparsed / missing
        up_pack = std::max(up_pack, e.w3_bytes);
        up_scale = std::max(up_scale, e.w3_scales_bytes);
        down_pack = std::max(down_pack, e.w2_bytes);
        down_scale = std::max(down_scale, e.w2_scales_bytes);
    }
    return PlanFn{}(up_pack, up_scale, down_pack, down_scale);
}

template <class Cache, class Strings, class PerExpertBytesFn>
bool AllocateExpertCache(Cache &cache, int slots, std::string &error)
{
    if (!cache.layout.ok || slots <= 0 || cache.experts_per_layer <= 0 ||
        cache.store.num_layers <= 0 || cache.store.experts_per_layer <= 0) {
        error = std::string(Strings::kTag) + " expert cache: store/layout not prepared";
        return false;
    }
    if (cache.total_experts <= 0)
        cache.total_experts =
            cache.store.num_layers * cache.store.experts_per_layer;
    cache.per_expert_bytes = PerExpertBytesFn{}(cache.layout);

    gpu::ExpertDeviceCache c{};
    c.slots = slots;
    c.total_experts = cache.total_experts;
    c.pack_stride = cache.layout.pack_stride;
    c.scale_stride = cache.layout.scale_stride;
    const std::size_t bytes = c.DeviceBytes();
    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        error = std::string(Strings::kTag) + " expert device block alloc failed (";
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

template <class Cache, class Strings>
cudaError_t StageExpert(Cache &cache, int layer, int eid,
                        cudaStream_t stream, std::string &error,
                        int *slot_out = nullptr, bool *missed_out = nullptr,
                        int *evicted_slot_out = nullptr)
{
    if (missed_out) *missed_out = false;
    if (evicted_slot_out) *evicted_slot_out = -1;
    if (!cache.layout.ok || cache.device.slots <= 0 || cache.device.d_pack ==
            nullptr) {
        error = std::string(Strings::kTag) + " expert cache not allocated";
        return cudaErrorInvalidValue;
    }
    if (layer < 0 || layer >= cache.store.num_layers || eid < 0 ||
        eid >= cache.experts_per_layer) {
        error = std::string(Strings::kTag) + " expert out of range: layer " +
                std::to_string(layer) + " eid " + std::to_string(eid);
        return cudaErrorInvalidValue;
    }
    // Host store is keyed by the real transformer layer (Arrow path
    // layer_<l>); the device flat id is the compacted MoE-layer index. An
    // empty map degenerates to identity (all layers MoE).
    const int store_flat = layer * cache.store.experts_per_layer + eid;
    int moe_idx = layer;
    if (!cache.moe_index_of_layer.empty()) {
        if (layer >= static_cast<int>(cache.moe_index_of_layer.size())) {
            error = std::string(Strings::kTag) + " expert layer map too short";
            return cudaErrorInvalidValue;
        }
        moe_idx = cache.moe_index_of_layer[static_cast<std::size_t>(layer)];
        if (moe_idx < 0) {
            error = std::string(Strings::kTag) +
                    " expert requested on a non-MoE layer " +
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
            error = std::string(Strings::kTag) + " expert store entry missing";
            return cudaErrorInvalidValue;
        }
        if (!cache.store.entries[idx].host_w3) {
            // Cold entry: parse the Arrow file on demand (only reachable when
            // the store cap forced eviction of a host view).
            if (!cache.store.ParseFile(idx)) {
                error = std::string(Strings::kTag) +
                        " expert host parse failed for layer " +
                        std::to_string(layer) + " eid " + std::to_string(eid);
                return cudaErrorInvalidValue;
            }
        }
        const auto &e = cache.store.entries[idx];
        if (!e.host_w3 || !e.host_w2 || !e.host_w3_scales ||
            !e.host_w2_scales) {
            error = std::string(Strings::kTag) +
                    " expert payload missing for layer " +
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
            error = std::string(Strings::kTag) + " expert H2D staging failed";
            return ce;
        }
    }
    if (slot_out) *slot_out = slot;
    return cudaSuccess;
}

template <class Cache, class Strings>
cudaError_t PrewarmExpertCache(Cache &cache, cudaStream_t stream,
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
            const cudaError_t ce = StageExpert<Cache, Strings>(
                cache, layer, eid, stream, error, nullptr, &missed);
            if (ce != cudaSuccess) return ce;
            if (missed) ++staged;
        }
    }
    return cudaStreamSynchronize(stream);
}

template <class Cache, class Strings, class View, class SlotPointersFn>
View ResolveExpertView(Cache &cache, int layer, int eid,
                       cudaStream_t stream, std::string &error)
{
    int slot = -1;
    if (StageExpert<Cache, Strings>(cache, layer, eid, stream, error, &slot) !=
        cudaSuccess)
        return View{};
    View v = SlotPointersFn{}(cache.device, slot, cache.layout);
    const std::size_t idx = static_cast<std::size_t>(
        layer * cache.store.experts_per_layer + eid);
    if (idx < cache.store.entries.size()) {
        v.up_global_scale = cache.store.entries[idx].w3_global_scale;
        v.down_global_scale = cache.store.entries[idx].w2_global_scale;
    }
    return v;
}

template <class Cache, class Strings, class View, class SlotPointersFn>
cudaError_t ResolveExperts(Cache &cache, int layer, const int *top_ids,
                           int k, cudaStream_t stream, View *out,
                           std::string &error)
{
    if (k < 0 || (k > 0 && (top_ids == nullptr || out == nullptr))) {
        error = std::string(Strings::kTag) + " expert resolve: null top-k input";
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < k; ++i) {
        out[i] = ResolveExpertView<Cache, Strings, View, SlotPointersFn>(
            cache, layer, top_ids[i], stream, error);
        if (out[i].up_packed == nullptr || out[i].down_packed == nullptr)
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

} // namespace nemotron_shared
} // namespace lsxcommon