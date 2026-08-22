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
