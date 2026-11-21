// Nemotron-3-Ultra routed-expert cache — the engine is the shared
// nemotron_shared template (nemotron_moe_expert_cache_impl.h, dedup phase 7:
// the lightning copy was identical modulo names and log strings). Public API
// stays; the bodies are explicit instantiations of the shared engine.
#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"

#include "nemotron_moe_expert_cache_impl.h"

#include "lsxcommon/log.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_ultra {

namespace {
struct PerExpertBytes {
    int64_t operator()(const UltraExpertSlotLayout &l) const
    {
        return UltraExpertPerExpertBytes(l);
    }
};
struct PlanSlot {
    UltraExpertSlotLayout operator()(int64_t up_pack, int64_t up_scale,
                                  int64_t down_pack, int64_t down_scale) const
    {
        return PlanUltraExpertSlot(up_pack, up_scale, down_pack, down_scale);
    }
};
struct SlotPointers {
    UltraExpertView operator()(const gpu::ExpertDeviceCache &c, int slot,
                            const UltraExpertSlotLayout &l) const
    {
        return UltraExpertSlotPointers(c, slot, l);
    }
};
using Strings = nemotron_shared::UltraStrings;
} // namespace

void LoadUltraExpertStore(const std::string &model_path, int num_layers,
                       int experts_per_layer, UltraExpertStore &store)
{
    nemotron_shared::LoadExpertStore<UltraExpertStore, Strings>(
        model_path, num_layers, experts_per_layer, store);
}

UltraExpertSlotLayout PlanUltraExpertSlotFromStore(const UltraExpertStore &store)
{
    return nemotron_shared::PlanExpertSlotFromStore<UltraExpertStore,
                                                    UltraExpertSlotLayout,
                                                    PlanSlot>(store);
}

bool AllocateUltraExpertCache(UltraExpertCache &cache, int slots, std::string &error)
{
    return nemotron_shared::AllocateExpertCache<UltraExpertCache, Strings,
                                                PerExpertBytes>(cache, slots,
                                                                error);
}

void FreeUltraExpertCache(UltraExpertCache &cache)
{
    gpu::FreeExpertCacheCommon(cache);
}

cudaError_t PrewarmUltraExpertCache(UltraExpertCache &cache, cudaStream_t stream,
                                 std::string &error)
{
    return nemotron_shared::PrewarmExpertCache<UltraExpertCache, Strings>(
        cache, stream, error);
}

cudaError_t StageUltraExpert(UltraExpertCache &cache, int layer, int eid,
                          cudaStream_t stream, std::string &error,
                          int *slot_out, bool *missed_out,
                          int *evicted_slot_out)
{
    return nemotron_shared::StageExpert<UltraExpertCache, Strings>(
        cache, layer, eid, stream, error, slot_out, missed_out,
        evicted_slot_out);
}

UltraExpertView ResolveUltraExpertView(UltraExpertCache &cache, int layer, int eid,
                                 cudaStream_t stream, std::string &error)
{
    return nemotron_shared::ResolveExpertView<UltraExpertCache, Strings,
                                              UltraExpertView, SlotPointers>(
        cache, layer, eid, stream, error);
}

cudaError_t ResolveUltraExperts(UltraExpertCache &cache, int layer, const int *top_ids,
                             int k, cudaStream_t stream, UltraExpertView *out,
                             std::string &error)
{
    return nemotron_shared::ResolveExperts<UltraExpertCache, Strings,
                                           UltraExpertView, SlotPointers>(
        cache, layer, top_ids, k, stream, out, error);
}

} // namespace lsxcommon::nemotron_ultra
