// CUDA side of the Kimi resident-expert device cache: arena allocation,
// batched async upload on a dedicated copy stream, event fencing.
// Host bookkeeping: include/lsxcommon/kimi_moe_expert_cache.h (pure).
#include "lsxcommon/kimi_moe_expert_cache.h"

#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>
#include <cstdlib>

namespace lsxcommon {
namespace kimi {
namespace {

size_t AlignUp256(size_t v) { return (v + 255) / 256 * 256; }

size_t KimiPackStride(const KimiExpertEntry &e)
{
    return AlignUp256(e.w1_bytes) + AlignUp256(e.w3_bytes) +
           AlignUp256(e.w2_bytes);
}

size_t KimiScaleStride(const KimiExpertEntry &e)
{
    return AlignUp256(e.w1_scales_bytes) + AlignUp256(e.w3_scales_bytes) +
           AlignUp256(e.w2_scales_bytes);
}

} // namespace

struct KimiMoeExpertDeviceCache;   // definition in kimi_moe_expert_cache.h

KimiMoeSlotView KimiMoeSlotPointers(const KimiMoeExpertDeviceCache &cache,
                                    int slot, const KimiExpertEntry &e)
{
    unsigned char *pk = cache.d_pack +
                        static_cast<size_t>(slot) * cache.pack_stride;
    unsigned char *sc = cache.d_scale +
                        static_cast<size_t>(slot) * cache.scale_stride;
    KimiMoeSlotView v;
    v.p_w1 = pk;
    v.p_w3 = pk + AlignUp256(e.w1_bytes);
    v.p_w2 = v.p_w3 + AlignUp256(e.w3_bytes);
    v.p_s1 = sc;
    v.p_s3 = sc + AlignUp256(e.w1_scales_bytes);
    v.p_s2 = v.p_s3 + AlignUp256(e.w3_scales_bytes);
    return v;
}

cudaError_t AllocateKimiMoeExpertCache(KimiExpertDb &db, int slots,
                                       KimiMoeExpertDeviceCache *&out,
                                       std::string &err)
{
    out = nullptr;
    if (slots <= 0 || db.entries.empty()) {
        err = "cache disabled";
        return cudaErrorInvalidValue;
    }
    // Uniform geometry check across the NON-zero entries (fixed INT4/NVFP4
    // layout). Kimi layer 0 is dense: its expert entry has no backing file
    // and stays all-zero — a zero entry is a dense hole, not a geometry
    // sample. Comparing against entry 0's zero strides used to disable the
    // whole cache ("non-uniform expert geometry"), silently downgrading
    // decode to the legacy per-call staging path. A mismatch between real
    // experts still disables the cache — the generation loop then keeps
    // the legacy path.
    size_t ref_pack = 0, ref_scale = 0;
    for (const auto &e : db.entries) {
        const size_t p = KimiPackStride(e);
        const size_t s = KimiScaleStride(e);
        if (p != 0 && s != 0) {
            ref_pack = p;
            ref_scale = s;
            break;
        }
    }
    if (ref_pack == 0 || ref_scale == 0) {
        err = "no populated expert entries; cache disabled";
        return cudaErrorInvalidValue;
    }
    for (const auto &e : db.entries) {
        const size_t p = KimiPackStride(e);
        const size_t s = KimiScaleStride(e);
        if (p == 0 && s == 0) continue;   // dense hole
        if (p != ref_pack || s != ref_scale) {
            err = "non-uniform expert geometry; cache disabled";
            return cudaErrorInvalidValue;
        }
    }
    auto *c = new KimiMoeExpertDeviceCache();
    c->pack_stride = ref_pack;
    c->scale_stride = ref_scale;
    c->slots = slots;
    cudaError_t e = cudaMalloc(&c->d_pack, static_cast<size_t>(slots) * ref_pack);
    if (e == cudaSuccess)
        e = cudaMalloc(&c->d_scale, static_cast<size_t>(slots) * ref_scale);
    if (e == cudaSuccess)
        e = cudaStreamCreate(&c->copy_stream);
    if (e == cudaSuccess)
        e = cudaEventCreateWithFlags(&c->ready, cudaEventDisableTiming);
    if (e != cudaSuccess) {
        err = "kimi cache alloc failed";
        FreeKimiMoeExpertDeviceCache(c);
        return e;
    }
    // Prime the fence: a wait before the FIRST record must be a no-op.
    cudaEventRecord(c->ready, c->copy_stream);
    out = c;
    return cudaSuccess;
}

void FreeKimiMoeExpertDeviceCache(KimiMoeExpertDeviceCache *&c)
{
    if (!c) return;
    if (c->d_pack) cudaFree(c->d_pack);
    if (c->d_scale) cudaFree(c->d_scale);
    if (c->copy_stream) cudaStreamDestroy(c->copy_stream);
    if (c->ready) cudaEventDestroy(c->ready);
    delete c;
    c = nullptr;
}

cudaError_t UploadKimiMoeExpertBatch(KimiExpertDb &db,
                                     const KimiExpertCacheState &state,
                                     const KimiMoeCacheBatch &batch, int layer,
                                     int experts_per_layer)
{
    KimiMoeExpertDeviceCache *cache = state.device;
    if (!cache || cache->slots <= 0 || batch.missing_count <= 0)
        return cudaSuccess;
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < batch.missing_count; ++i) {
        const int bid = batch.missing_idx[i];
        const int slot = batch.missing_slot[i];
        if (bid < 0 || slot < 0) continue;
        // Host-mmap residency for the DMA source (ExpertStore fd LRU);
        // cheap no-op when the mmap is already open.
        db.EnsureResident(layer, bid, cache->copy_stream);
        const KimiExpertEntry &ent =
            db.entries[static_cast<size_t>(layer) * experts_per_layer +
                       static_cast<size_t>(bid)];
        const KimiMoeSlotView v = KimiMoeSlotPointers(*cache, slot, ent);
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w1), ent.host_w1,
                            ent.w1_bytes, cudaMemcpyHostToDevice,
                            cache->copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w3), ent.host_w3,
                            ent.w3_bytes, cudaMemcpyHostToDevice,
                            cache->copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_w2), ent.host_w2,
                            ent.w2_bytes, cudaMemcpyHostToDevice,
                            cache->copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s1),
                            ent.host_w1_scales, ent.w1_scales_bytes,
                            cudaMemcpyHostToDevice, cache->copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s3),
                            ent.host_w3_scales, ent.w3_scales_bytes,
                            cudaMemcpyHostToDevice, cache->copy_stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpyAsync(const_cast<unsigned char *>(v.p_s2),
                            ent.host_w2_scales, ent.w2_scales_bytes,
                            cudaMemcpyHostToDevice, cache->copy_stream);
        if (e != cudaSuccess) return e;
    }
    return cudaEventRecord(cache->ready, cache->copy_stream);
}

} // namespace kimi
} // namespace lsxcommon