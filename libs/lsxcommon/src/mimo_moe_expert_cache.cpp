#include "lsxcommon/mimo_moe_expert_cache.h"

#include "lsxcommon/log.h"

#include <cuda_runtime.h>

#include <cstdlib>

namespace lsxcommon {
namespace mimo {

void LoadMimoExpertStore(const std::string &model_path, int num_layers,
                         int first_moe_layer, int experts_per_layer,
                         MimoExpertStore &store)
{
    store.log_prefix = "[mimo_load] ";
    store.mmap_fail_verb = "mimo expert mmap fail ";
    store.num_layers = num_layers;
    store.experts_per_layer = experts_per_layer;
    store.first_moe_layer = first_moe_layer;
    // 12032 files fit under the 65536 mmap cap: a slot miss is a pure H2D copy.
    store.Load(model_path, /*fallback=*/true);

    // Lock the routed-expert payloads into host memory (LSX_EXPERT_PIN=0 opts
    // out) so the first prefill does not read the ~170 GiB store from disk.
    const char *pin_env = std::getenv("LSX_EXPERT_PIN");
    if (pin_env == nullptr || pin_env[0] != '0')
        store.PinChunked<MimoExpertStore::PinLayout::kStream>();
}

MimoExpertSlotLayout PlanMimoExpertSlotFromStore(const MimoExpertStore &store)
{
    int64_t gp = 0, gs = 0, up = 0, us = 0, dp = 0, ds = 0;
    for (const auto &e : store.entries)
    {
        if (!e.host_w1) continue;
        gp = std::max(gp, e.w1_bytes);
        gs = std::max(gs, e.w1_scales_bytes);
        up = std::max(up, e.w3_bytes);
        us = std::max(us, e.w3_scales_bytes);
        dp = std::max(dp, e.w2_bytes);
        ds = std::max(ds, e.w2_scales_bytes);
    }
    return PlanMimoExpertSlot(gp, gs, up, us, dp, ds);
}

bool InitMiMoExpertCache(const std::string &model_path, int num_layers,
                         int first_moe_layer, int experts_per_layer, int topk,
                         int slots_override, cudaStream_t stream,
                         MimoExpertCache &cache, std::string &error)
{
    if (num_layers <= 0 || experts_per_layer <= 0)
    {
        error = "mimo expert cache: invalid layer/expert geometry";
        return false;
    }
    cache.experts_per_layer = experts_per_layer;
    cache.moe_layers =
        num_layers > first_moe_layer ? num_layers - first_moe_layer : 0;
    cache.total_experts = num_layers * experts_per_layer;
    cache.moe_index_of_layer.assign(static_cast<std::size_t>(num_layers), -1);
    for (int l = first_moe_layer; l < num_layers; ++l)
        cache.moe_index_of_layer[static_cast<std::size_t>(l)] =
            l - first_moe_layer;

    LoadMimoExpertStore(model_path, num_layers, first_moe_layer,
                        experts_per_layer, cache.store);
    if (cache.store.entries.empty())
    {
        error = "mimo expert store is empty";
        return false;
    }
    cache.layout = PlanMimoExpertSlotFromStore(cache.store);
    if (!cache.layout.ok)
    {
        error = "mimo expert slot layout planning failed";
        return false;
    }
    int slots = slots_override;
    if (slots <= 0)
    {
        const int64_t per = MimoExpertCachePerExpertBytes(cache.layout);
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        slots = MimoExpertCacheSlots(free_b, static_cast<std::size_t>(per),
                                     cache.total_experts, topk,
                                     std::getenv("LSX_MIMO_EXPERT_CACHE_SLOTS"));
    }
    if (!AllocateMimoExpertCache(cache, slots, error)) return false;
    const cudaError_t ce = PrewarmMimoExpertCache(cache, stream, error);
    if (ce != cudaSuccess)
    {
        if (error.empty()) error = "mimo expert cache prewarm failed";
        FreeMimoExpertCache(cache);
        return false;
    }
    return true;
}

bool AllocateMimoExpertCache(MimoExpertCache &cache, int slots, std::string &error)
{
    if (!cache.layout.ok || slots <= 0 || cache.experts_per_layer <= 0 ||
        cache.store.num_layers <= 0 || cache.store.experts_per_layer <= 0)
    {
        error = "mimo expert cache: store/layout not prepared";
        return false;
    }
    if (cache.total_experts <= 0)
        cache.total_experts =
            cache.store.num_layers * cache.store.experts_per_layer;
    cache.per_expert_bytes = MimoExpertCachePerExpertBytes(cache.layout);

    gpu::ExpertDeviceCache c{};
    c.slots = slots;
    c.total_experts = cache.total_experts;
    c.pack_stride = cache.layout.pack_stride;
    c.scale_stride = cache.layout.scale_stride;
    const std::size_t bytes = c.DeviceBytes();
    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess)
    {
        error = "mimo expert device block alloc failed (";
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
    for (int i = 0; i < slots; ++i)
    {
        c.h_slot_owner[i] = -1;
        c.h_lru[i] = i;
        c.h_free[i] = slots - 1 - i;
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

void FreeMimoExpertCache(MimoExpertCache &cache)
{
    if (cache.device.d_base)
    {
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

cudaError_t PrewarmMimoExpertCache(MimoExpertCache &cache, cudaStream_t stream,
                                   std::string &error)
{
    if (cache.device.slots <= 0 || cache.store.num_layers <= 0)
        return cudaSuccess;
    int staged = 0;
    for (int layer = 0;
         layer < cache.store.num_layers && staged < cache.device.slots; ++layer)
    {
        if (!cache.moe_index_of_layer.empty())
        {
            if (layer >= static_cast<int>(cache.moe_index_of_layer.size()) ||
                cache.moe_index_of_layer[static_cast<std::size_t>(layer)] < 0)
                continue;
        }
        for (int eid = 0;
             eid < cache.experts_per_layer && staged < cache.device.slots; ++eid)
        {
            bool missed = false;
            const cudaError_t ce =
                StageMimoExpert(cache, layer, eid, stream, error, nullptr, &missed);
            if (ce != cudaSuccess) return ce;
            if (missed) ++staged;
        }
    }
    return cudaStreamSynchronize(stream);
}

cudaError_t StageMimoExpert(MimoExpertCache &cache, int layer, int eid,
                            cudaStream_t stream, std::string &error,
                            int *slot_out, bool *missed_out,
                            int *evicted_slot_out)
{
    if (missed_out) *missed_out = false;
    if (evicted_slot_out) *evicted_slot_out = -1;
    if (!cache.layout.ok || cache.device.slots <= 0 ||
        cache.device.d_pack == nullptr)
    {
        error = "mimo expert cache not allocated";
        return cudaErrorInvalidValue;
    }
    if (layer < 0 || layer >= cache.store.num_layers || eid < 0 ||
        eid >= cache.experts_per_layer)
    {
        error = "mimo expert out of range: layer " + std::to_string(layer) +
                " eid " + std::to_string(eid);
        return cudaErrorInvalidValue;
    }
    const int store_flat = layer * cache.store.experts_per_layer + eid;
    int moe_idx = layer;
    if (!cache.moe_index_of_layer.empty())
    {
        if (layer >= static_cast<int>(cache.moe_index_of_layer.size()))
        {
            error = "mimo expert layer map too short";
            return cudaErrorInvalidValue;
        }
        moe_idx = cache.moe_index_of_layer[static_cast<std::size_t>(layer)];
        if (moe_idx < 0)
        {
            error = "mimo expert requested on a non-MoE layer " +
                    std::to_string(layer);
            return cudaErrorInvalidValue;
        }
    }
    const int flat = moe_idx * cache.experts_per_layer + eid;
    int slot = cache.device.Lookup(flat);
    if (slot < 0)
    {
        int evicted = -1;
        slot = cache.device.Insert(flat, &evicted);
        if (missed_out) *missed_out = true;
        if (evicted_slot_out) *evicted_slot_out = evicted;

        const std::size_t idx = static_cast<std::size_t>(store_flat);
        if (idx >= cache.store.entries.size())
        {
            error = "mimo expert store entry missing";
            return cudaErrorInvalidValue;
        }
        if (!cache.store.entries[idx].host_w1)
        {
            if (!cache.store.ParseFile(idx))
            {
                error = "mimo expert host parse failed for layer " +
                        std::to_string(layer) + " eid " + std::to_string(eid);
                return cudaErrorInvalidValue;
            }
        }
        const MimoExpertEntry &e = cache.store.entries[idx];
        if (!e.host_w1 || !e.host_w3 || !e.host_w2 || !e.host_w1_scales ||
            !e.host_w3_scales || !e.host_w2_scales)
        {
            error = "mimo expert payload missing for layer " +
                    std::to_string(layer) + " eid " + std::to_string(eid);
            return cudaErrorInvalidValue;
        }
        uint8_t *dp = cache.device.d_pack +
                      static_cast<int64_t>(slot) * cache.device.pack_stride;
        uint8_t *ds = cache.device.d_scale +
                      static_cast<int64_t>(slot) * cache.device.scale_stride;
        // Diagnostic: exact-repack experts (pow2 E4M3 scales) carry the raw
        // source nibble order; the fallback experts were quantized with our
        // own order. Swap only the former when LSX_MIMO_EXPERT_SWAP is set.
        const bool swap_env = std::getenv("LSX_MIMO_EXPERT_SWAP") != nullptr;
        auto pow2_scales = [](const uint8_t *s, int64_t n) {
            for (int64_t i = 0; i < n; ++i)
                if ((s[i] & 0x7) != 0) return false;
            return true;
        };
        const bool exact =
            swap_env && pow2_scales(e.host_w1_scales, e.w1_scales_bytes);
        auto stage = [&](const uint8_t *src, int64_t n, uint8_t *dst,
                         bool allow_swap) {
            if (n <= 0) return cudaSuccess;
            if (allow_swap && n >= 2 && (n & 1) == 0)
            {
                std::vector<uint8_t> tmp(static_cast<size_t>(n));
                for (int64_t i = 0; i < n; ++i)
                    tmp[static_cast<size_t>(i)] =
                        static_cast<uint8_t>((src[i] >> 4) | (src[i] << 4));
                return cudaMemcpy(dst, tmp.data(), static_cast<size_t>(n),
                                  cudaMemcpyHostToDevice);
            }
            return cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n),
                                   cudaMemcpyHostToDevice, stream);
        };
        cudaError_t ce = stage(e.host_w1, e.w1_bytes,
                               dp + cache.layout.gate_pack_off, exact);
        if (ce == cudaSuccess)
            ce = stage(e.host_w3, e.w3_bytes, dp + cache.layout.up_pack_off,
                       exact);
        if (ce == cudaSuccess)
            ce = stage(e.host_w2, e.w2_bytes, dp + cache.layout.down_pack_off,
                       exact);
        if (ce == cudaSuccess)
            ce = stage(e.host_w1_scales, e.w1_scales_bytes,
                       ds + cache.layout.gate_scale_off, false);
        if (ce == cudaSuccess)
            ce = stage(e.host_w3_scales, e.w3_scales_bytes,
                       ds + cache.layout.up_scale_off, false);
        if (ce == cudaSuccess)
            ce = stage(e.host_w2_scales, e.w2_scales_bytes,
                       ds + cache.layout.down_scale_off, false);
        if (ce != cudaSuccess)
        {
            error = "mimo expert H2D staging failed";
            return ce;
        }
    }
    if (slot_out) *slot_out = slot;
    return cudaSuccess;
}

MimoExpertView ResolveMimoExpertView(MimoExpertCache &cache, int layer, int eid,
                                     cudaStream_t stream, std::string &error)
{
    int slot = -1;
    if (StageMimoExpert(cache, layer, eid, stream, error, &slot) != cudaSuccess)
        return MimoExpertView{};
    MimoExpertView v = MimoExpertSlotPointers(cache.device, slot, cache.layout);
    const std::size_t idx =
        static_cast<std::size_t>(layer * cache.store.experts_per_layer + eid);
    if (idx < cache.store.entries.size())
    {
        v.gate_global_scale = cache.store.entries[idx].w1_global_scale;
        v.up_global_scale = cache.store.entries[idx].w3_global_scale;
        v.down_global_scale = cache.store.entries[idx].w2_global_scale;
    }
    return v;
}

cudaError_t ResolveMimoExperts(MimoExpertCache &cache, int layer,
                               const int *top_ids, int k, cudaStream_t stream,
                               MimoExpertView *out, std::string &error)
{
    if (k < 0 || (k > 0 && (top_ids == nullptr || out == nullptr)))
    {
        error = "mimo expert resolve: null top-k input";
        return cudaErrorInvalidValue;
    }
    for (int i = 0; i < k; ++i)
    {
        out[i] = ResolveMimoExpertView(cache, layer, top_ids[i], stream, error);
        if (out[i].gate_packed == nullptr || out[i].up_packed == nullptr ||
            out[i].down_packed == nullptr)
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

} // namespace mimo
} // namespace lsxcommon
