// Nemotron-3.5-Lightning routed-expert cache — the engine is the shared
// nemotron_shared template (nemotron_moe_expert_cache_impl.h, dedup phase 7:
// the ultra copy was identical modulo names and log strings). Public API
// stays; the bodies are explicit instantiations of the shared engine.
#include "lsxcommon/nemotron_moe_expert_cache.h"

#include "nemotron_moe_expert_cache_impl.h"

#include "lsxcommon/log.h"

#include <cuda_runtime.h>

namespace lsxcommon::nemotron_moe {

namespace {
struct PerExpertBytes {
    int64_t operator()(const NmExpertSlotLayout &l) const
    {
        return NmExpertCachePerExpertBytes(l);
    }
};
struct PlanSlot {
    NmExpertSlotLayout operator()(int64_t up_pack, int64_t up_scale,
                                  int64_t down_pack, int64_t down_scale) const
    {
        return PlanNmExpertSlot(up_pack, up_scale, down_pack, down_scale);
    }
};
struct SlotPointers {
    NmExpertView operator()(const gpu::ExpertDeviceCache &c, int slot,
                            const NmExpertSlotLayout &l) const
    {
        return NmExpertSlotPointers(c, slot, l);
    }
};
using Strings = nemotron_shared::NmStrings;
} // namespace

void LoadNmExpertStore(const std::string &model_path, int num_layers,
                       int experts_per_layer, NmExpertStore &store)
{
    nemotron_shared::LoadExpertStore<NmExpertStore, Strings>(
        model_path, num_layers, experts_per_layer, store);
}

NmExpertSlotLayout PlanNmExpertSlotFromStore(const NmExpertStore &store)
{
    return nemotron_shared::PlanExpertSlotFromStore<NmExpertStore,
                                                    NmExpertSlotLayout,
                                                    PlanSlot>(store);
}

bool AllocateNmExpertCache(NmExpertCache &cache, int slots, std::string &error)
{
    return nemotron_shared::AllocateExpertCache<NmExpertCache, Strings,
                                                PerExpertBytes>(cache, slots,
                                                                error);
}

void FreeNmExpertCache(NmExpertCache &cache)
{
    gpu::FreeExpertCacheCommon(cache);
}

cudaError_t PrewarmNmExpertCache(NmExpertCache &cache, cudaStream_t stream,
                                 std::string &error)
{
    return nemotron_shared::PrewarmExpertCache<NmExpertCache, Strings>(
        cache, stream, error);
}

cudaError_t StageNmExpert(NmExpertCache &cache, int layer, int eid,
                          cudaStream_t stream, std::string &error,
                          int *slot_out, bool *missed_out,
                          int *evicted_slot_out)
{
    return nemotron_shared::StageExpert<NmExpertCache, Strings>(
        cache, layer, eid, stream, error, slot_out, missed_out,
        evicted_slot_out);
}

NmExpertView ResolveNmExpertView(NmExpertCache &cache, int layer, int eid,
                                 cudaStream_t stream, std::string &error)
{
    return nemotron_shared::ResolveExpertView<NmExpertCache, Strings,
                                              NmExpertView, SlotPointers>(
        cache, layer, eid, stream, error);
}

cudaError_t ResolveNmExperts(NmExpertCache &cache, int layer, const int *top_ids,
                             int k, cudaStream_t stream, NmExpertView *out,
                             std::string &error)
{
    return nemotron_shared::ResolveExperts<NmExpertCache, Strings,
                                           NmExpertView, SlotPointers>(
        cache, layer, top_ids, k, stream, out, error);
}

} // namespace lsxcommon::nemotron_moe
