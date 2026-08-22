#include "lsxcommon/lifecycle_strategy.h"

#include <algorithm>

#include <absl/container/flat_hash_map.h>
#include <spdlog/spdlog.h>

namespace lsxcommon {
namespace lifecycle {

namespace {
std::mutex g_cache_mu;
absl::flat_hash_map<std::string, LifecyclePlan> g_plan_cache;
} // namespace

LifecyclePlan GeneralStrategy::BuildPlan(const LifecycleInput &in,
                                         const HardwareProfile &hw,
                                         std::uint64_t vram_keep_out_bytes) const
{
    LifecyclePlan p;
    const std::uint64_t grantable =
        hw.vram_total_bytes > vram_keep_out_bytes
            ? hw.vram_total_bytes - vram_keep_out_bytes
            : 0;
    // Permanent residency: the conversion-time optimal footprint, capped at
    // what the machine can grant. Everything above stays host-pinned/mmap.
    p.permanent_weight_bytes = std::min(in.vram.optimal_bytes, grantable);

    const std::uint64_t remaining =
        grantable > p.permanent_weight_bytes
            ? grantable - p.permanent_weight_bytes
            : 0;
    // Reserve headroom for the infer arenas: max - optimal is the
    // conversion-time estimate of the transient workspace.
    const std::uint64_t arena_reserve =
        in.vram.max_bytes > in.vram.optimal_bytes
            ? in.vram.max_bytes - in.vram.optimal_bytes
            : 0;
    const std::uint64_t cache_budget =
        remaining > arena_reserve ? remaining - arena_reserve : 0;
    p.expert_cache_max_bytes =
        in.archetype == Archetype::MoE ? cache_budget : 0;

    p.host_upload_threads = std::clamp(hw.cpu_cores / 8, 1, 4);
    return p;
}

const ILifecycleStrategy &ResolveStrategy(Archetype archetype,
                                          const std::string &model_type)
{
    static GeneralStrategy general;
    if (archetype == Archetype::MoE) return general;  // replaced in Task 6
    (void)model_type;
    return general;
}

const LifecyclePlan &GetOrBuildPlan(const LifecycleInput &in)
{
    const auto &gov = ResourceGovernor::Instance();
    const ILifecycleStrategy &s =
        ResolveStrategy(in.archetype, in.model_type);
    LifecyclePlan p = s.BuildPlan(in, gov.hardware(),
                                  gov.options().vram_keep_out_bytes);
    spdlog::info("[lifecycle] plan for '{}': permanent={} MiB cache slots={} "
                 "[{}..{}] streams={} staging={}",
                 in.model_type, p.permanent_weight_bytes >> 20,
                 p.expert_cache_slots, p.tuner_bounds.expert_cache_slots_min,
                 p.tuner_bounds.expert_cache_slots_max, p.copy_streams,
                 p.staging_ring_bytes >> 20);
    std::lock_guard<std::mutex> lk(g_cache_mu);
    auto [ins, inserted] = g_plan_cache.emplace(in.model_type, std::move(p));
    (void)inserted;
    return ins->second;
}

void ResetPlanCache()
{
    std::lock_guard<std::mutex> lk(g_cache_mu);
    g_plan_cache.clear();
}

} // namespace lifecycle
} // namespace lsxcommon
