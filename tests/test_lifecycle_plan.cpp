#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "lsxcommon/lifecycle_strategy.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

namespace {

HardwareProfile FakeHw(std::uint64_t vram_gib)
{
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 64ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 148;
    hw.vram_total_bytes = vram_gib << 30;
    hw.vram_free_bytes = hw.vram_total_bytes;
    return hw;
}

VramProfile FakeVram(std::uint64_t min_gib, std::uint64_t optimal_gib,
                     std::uint64_t max_gib)
{
    VramProfile v;
    v.min_bytes = min_gib << 30;
    v.optimal_bytes = optimal_gib << 30;
    v.max_bytes = max_gib << 30;
    v.context_tokens = 4096;
    return v;
}

LifecycleInput MakeInput(VramProfile v, Archetype arch,
                         std::uint64_t mean_expert = 0)
{
    LifecycleInput in;
    in.vram = v;
    in.archetype = arch;
    in.model_type = "test-model";
    in.expert_mean_bytes = mean_expert;
    return in;
}

constexpr std::uint64_t kKeepOut = 0;  // tests grant the full budget

} // namespace

TEST_CASE("general plan caps residency to grantable vram", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
    // permanent = min(optimal 4 GiB, total 10 GiB) = 4 GiB
    REQUIRE(p.permanent_weight_bytes == (4ull << 30));
    // remaining 6 GiB - arena_reserve (max-optimal = 2 GiB) = 4 GiB cache,
    // but General archetype carries no experts:
    REQUIRE(p.expert_cache_max_bytes == 0);
    REQUIRE(p.expert_cache_slots == 0);
    REQUIRE(p.pin_globals_in_vram);
    REQUIRE(p.pin_dense_layers);
    // 16 cores / 8 -> 2 upload threads
    REQUIRE(p.host_upload_threads == 2);
}

TEST_CASE("general plan shrinks when vram is smaller than optimal", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 5), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(2), kKeepOut);
    REQUIRE(p.permanent_weight_bytes == (2ull << 30));  // capped at grantable
    REQUIRE(p.expert_cache_max_bytes == 0);
}

TEST_CASE("general plan handles keep-out margin", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 4), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(10), 3ull << 30);  // grantable 7 GiB
    REQUIRE(p.permanent_weight_bytes == (4ull << 30));
}

TEST_CASE("moe plan sizes expert cache from mean expert bytes", "[lifecycle]")
{
    MoEStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                              /*mean_expert*/ 512ull << 20);
    const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
    // cache_budget = 10 GiB - 4 GiB permanent - 2 GiB arena reserve = 4 GiB
    REQUIRE(p.expert_cache_max_bytes == (4ull << 30));
    // fit = 4 GiB / 512 MiB = 8 slots
    REQUIRE(p.expert_cache_slots == 8);
    REQUIRE(p.tuner_bounds.expert_cache_slots_min == 4);
    REQUIRE(p.tuner_bounds.expert_cache_slots_max == 8);
    REQUIRE(p.expert_cache_min_bytes == 4ull * (512ull << 20));
    REQUIRE(p.expert_prefetch_batch == 4);
}

TEST_CASE("moe plan clamps slots into sane bounds", "[lifecycle]")
{
    MoEStrategy s;
    SECTION("tiny experts would explode slot count -> clamped at 4096")
    {
        const auto in = MakeInput(FakeVram(1, 1, 2), Archetype::MoE,
                                  /*mean_expert*/ 1ull << 20);
        const auto p = s.BuildPlan(in, FakeHw(64), kKeepOut);
        REQUIRE(p.expert_cache_slots <= p.tuner_bounds.expert_cache_slots_max);
        REQUIRE(p.tuner_bounds.expert_cache_slots_max == 4096);
    }
    SECTION("huge experts -> floor of 4 slots")
    {
        const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                                  /*mean_expert*/ 8ull << 30);
        const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
        REQUIRE(p.expert_cache_slots == 4);
    }
}

TEST_CASE("glm-5.2 resolves the Glm52 strategy via the registry", "[lifecycle]")
{
    auto &gov = ResourceGovernor::Instance();
    gov.Shutdown();
    GovernorOptions gopts;
    gopts.override_hw = true;
    gopts.hw_override = FakeHw(10);
    gopts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(gopts));

    ResetPlanCache();
    LifecycleInput in;
    in.model_type = "glm-5.2";
    in.archetype = Archetype::MoE;
    in.vram = FakeVram(4, 4, 6);
    in.expert_mean_bytes = 512ull << 20;
    const auto &a = GetOrBuildPlan(in);
    const auto &b = GetOrBuildPlan(in);
    REQUIRE(&a == &b);  // cached: identical object
    REQUIRE(a.expert_prefetch_batch == 4);
    REQUIRE(a.pin_dense_layers);
    ResetPlanCache();
    gov.Shutdown();
}

TEST_CASE("plan cache keeps returned references stable across inserts",
          "[lifecycle]")
{
    // GetOrBuildPlan returns a reference into the process-global plan cache.
    // Those references must survive later inserts of OTHER model types: a
    // flat_hash_map relocates elements on rehash (no pointer stability), which
    // would dangle a previously handed-out plan. The cache must use a
    // node-based map (stable addresses).
    auto &gov = ResourceGovernor::Instance();
    gov.Shutdown();
    GovernorOptions gopts;
    gopts.override_hw = true;
    gopts.hw_override = FakeHw(10);
    gopts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(gopts));

    ResetPlanCache();
    LifecycleInput in;
    in.model_type = "stable-model";
    in.archetype = Archetype::MoE;
    in.vram = FakeVram(4, 4, 6);
    in.expert_mean_bytes = 512ull << 20;
    const LifecyclePlan &first = GetOrBuildPlan(in);

    // Force enough inserts to rehash a flat map several times.
    for (int i = 0; i < 1000; ++i) {
        LifecycleInput filler = in;
        filler.model_type = "filler-" + std::to_string(i);
        (void)GetOrBuildPlan(filler);
    }

    const LifecyclePlan &again = GetOrBuildPlan(in);
    REQUIRE(&first == &again);  // same object: no relocation/dangling
    REQUIRE(again.expert_prefetch_batch == first.expert_prefetch_batch);
    REQUIRE(again.pin_dense_layers);

    ResetPlanCache();
    gov.Shutdown();
}

TEST_CASE("plan building is deterministic", "[lifecycle]")
{
    MoEStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                              512ull << 20);
    const auto p1 = s.BuildPlan(in, FakeHw(10), kKeepOut);
    const auto p2 = s.BuildPlan(in, FakeHw(10), kKeepOut);
    REQUIRE(p1.permanent_weight_bytes == p2.permanent_weight_bytes);
    REQUIRE(p1.expert_cache_slots == p2.expert_cache_slots);
    REQUIRE(p1.expert_cache_max_bytes == p2.expert_cache_max_bytes);
    REQUIRE(p1.tuner_bounds.expert_cache_slots_max ==
            p2.tuner_bounds.expert_cache_slots_max);
    REQUIRE(p1.host_upload_threads == p2.host_upload_threads);
}
