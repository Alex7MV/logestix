// Unified memory (design §2.8): Jetson Orin shares its RAM with the GPU.
//
// On a discrete card the governor's arithmetic is unchanged — that is what keeps
// every 5090 baseline valid. On unified memory two things break:
//   * the 8 GiB expert-cache reserve is HALF of a 16 GB Orin, so the cache is
//     planned against a budget the machine cannot back;
//   * `vram_total_bytes` read through cudaMemGetInfo is the fixed boot carve-out,
//     not system RAM, so budgeting against it under-reports badly — and a
//     permanent-weight lease and a host-RAM lease are drawn from ONE pool, which
//     the two-budget bookkeeping counts twice.
//
// These cases are host-only: the memory model is a hardware fact, injected
// through GovernorOptions.hw_override exactly like every other probe.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/resource_governor.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

TEST_CASE("a discrete card keeps the 8 GiB reserve exactly",
          "[governor][unified]") {
    // The 5090 arithmetic must not move: ExpertCacheReserveBytes(32 GiB) is the
    // constant the whole vault was calibrated against.
    CHECK(ExpertCacheReserveBytes(32ull << 30) == (8ull << 30));
    CHECK(ExpertCacheReserveBytes(64ull << 30) == (8ull << 30));
    // Below the constant the quarter rule takes over, so a small card is not
    // asked to set aside more than a quarter of its budget.
    CHECK(ExpertCacheReserveBytes(16ull << 30) == (4ull << 30));
    CHECK(ExpertCacheReserveBytes(8ull << 30) == (2ull << 30));
    // Degenerate budgets answer 0 rather than a negative or a bogus reserve.
    CHECK(ExpertCacheReserveBytes(0) == 0);
}

TEST_CASE("a unified 16 GB budget reserves a quarter, not half",
          "[governor][unified]") {
    // 16 GiB / 4 = 4 GiB, where the discrete constant would have taken 8 GiB —
    // half the machine, on the device this branch exists for.
    CHECK(ExpertCacheReserveBytes(16ull << 30) == (4ull << 30));
    CHECK(ExpertCacheReserveBytes(16ull << 30) < kExpertCacheVramReserveBytes);
}

TEST_CASE("the reserve function never exceeds a quarter of the budget",
          "[governor][unified]") {
    for (std::uint64_t budget : {1ull << 30, 2ull << 30, 3ull << 30,
                                 6ull << 30, 12ull << 30, 16ull << 30,
                                 24ull << 30, 32ull << 30, 64ull << 30}) {
        INFO("budget=" << budget);
        const auto reserve = ExpertCacheReserveBytes(budget);
        CHECK(reserve <= budget / 4);
        CHECK(reserve <= kExpertCacheVramReserveBytes);
    }
}

TEST_CASE("the loader-facing reserve keeps the discrete constant untouched",
          "[governor][unified]") {
    // This is the overload a load path calls. On a discrete card it must return
    // the calibrated constant EXACTLY — not min(8 GiB, 25 % of the free-VRAM
    // reading), which on a 5090 would hand the expert cache an extra quarter of
    // a GiB and move every baseline measured with it.
    HardwareProfile discrete;
    discrete.memory_model = MemoryModel::kDiscrete;
    discrete.vram_total_bytes = 32ull << 30;
    discrete.vram_free_bytes = 30ull << 30;
    CHECK(ExpertCacheReserveBytes(discrete) == kExpertCacheVramReserveBytes);

    // Even a small discrete card keeps the constant: the trade there is the
    // operator's, and shrinking the reserve silently is what §2.8 forbids.
    discrete.vram_total_bytes = 24ull << 30;
    CHECK(ExpertCacheReserveBytes(discrete) == kExpertCacheVramReserveBytes);

    // Unified: a quarter of the shared pool.
    HardwareProfile unified;
    unified.memory_model = MemoryModel::kUnified;
    unified.unified_shared_bytes = 16ull << 30;
    CHECK(ExpertCacheReserveBytes(unified) == (4ull << 30));
    // A unified profile with no size probed must not silently take half of an
    // unknown machine: it answers 0, and a caller with no budget gets no cache
    // rather than an over-commit.
    unified.unified_shared_bytes = 0;
    CHECK(ExpertCacheReserveBytes(unified) == 0);
}

TEST_CASE("memory model is probed, never guessed from the name",
          "[governor][unified]") {
    GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override.memory_model = MemoryModel::kUnified;
    opts.hw_override.unified_shared_bytes = 16ull << 30;
    opts.hw_override.vram_total_bytes = 8ull << 30;   // the boot carve-out
    opts.hw_override.vram_free_bytes = 7ull << 30;
    opts.hw_override.ram_total_bytes = 16ull << 30;

    auto &gov = ResourceGovernor::Instance();
    REQUIRE(gov.Initialize(opts));
    CHECK(gov.hardware().memory_model == MemoryModel::kUnified);
    CHECK(gov.hardware().unified_shared_bytes == (16ull << 30));

    // A profile built from a device name alone must not claim unified memory:
    // the default is discrete, and only the probe flips it.
    HardwareProfile blank;
    CHECK(blank.memory_model == MemoryModel::kDiscrete);
    CHECK(blank.unified_shared_bytes == 0);
    gov.Shutdown();
}

TEST_CASE("on unified memory the grantable budget is system RAM, not the carve-out",
          "[governor][unified]") {
    GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override.memory_model = MemoryModel::kUnified;
    opts.hw_override.unified_shared_bytes = 16ull << 30;
    opts.hw_override.vram_total_bytes = 8ull << 30;  // carve-out only
    opts.hw_override.vram_free_bytes = 7ull << 30;
    opts.hw_override.ram_total_bytes = 16ull << 30;

    auto &gov = ResourceGovernor::Instance();
    REQUIRE(gov.Initialize(opts));

    // Budgeting against the 8 GiB carve-out would refuse a 10 GiB model on a
    // machine that has 16 GiB of shared memory: the carve-out is fixed at boot
    // and does not describe what the model may use.
    {
        auto weights = gov.Acquire(LeaseKind::kPermanentWeights, 10ull << 30,
                                   LeasePolicy{});
        INFO("grantable=" << gov.GrantableVramBytes());
        CHECK(weights.Valid());
        CHECK(gov.GrantableVramBytes() >= (10ull << 30));
    }   // release before Shutdown: Shutdown resets the counters

    gov.Shutdown();
}

TEST_CASE("a permanent-weight lease and a host lease share ONE pool",
          "[governor][unified]") {
    // 16 GiB unified: a 10 GiB weight lease then a 10 GiB pinned lease must not
    // both succeed, because both are drawn from the same physical memory. On a
    // discrete card the second one legitimately would (different pools).
    GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override.memory_model = MemoryModel::kUnified;
    opts.hw_override.unified_shared_bytes = 16ull << 30;
    opts.hw_override.vram_total_bytes = 16ull << 30;
    opts.hw_override.vram_free_bytes = 15ull << 30;
    opts.hw_override.ram_total_bytes = 15ull << 30;

    auto &gov = ResourceGovernor::Instance();
    REQUIRE(gov.Initialize(opts));

    {
        auto weights = gov.Acquire(LeaseKind::kPermanentWeights, 10ull << 30,
                                   LeasePolicy{});
        REQUIRE(weights.Valid());
        auto pinned = gov.Acquire(LeaseKind::kHostPinned, 10ull << 30,
                                  LeasePolicy{});
        CHECK_FALSE(pinned.Valid());
    }
    gov.Shutdown();
}

TEST_CASE("a discrete card keeps two budgets", "[governor][unified]") {
    // The complement of the previous case: on discrete memory a 10 GiB weight
    // lease does not consume the host pool, and this is the arithmetic every
    // existing baseline was measured under.
    GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override.memory_model = MemoryModel::kDiscrete;
    opts.hw_override.vram_total_bytes = 32ull << 30;
    opts.hw_override.vram_free_bytes = 31ull << 30;
    opts.hw_override.ram_total_bytes = 256ull << 30;
    opts.hw_override.ram_available_bytes = 200ull << 30;

    auto &gov = ResourceGovernor::Instance();
    REQUIRE(gov.Initialize(opts));

    {
        auto weights = gov.Acquire(LeaseKind::kPermanentWeights, 10ull << 30,
                                   LeasePolicy{});
        REQUIRE(weights.Valid());
        auto pinned = gov.Acquire(LeaseKind::kHostPinned, 10ull << 30,
                                  LeasePolicy{});
        CHECK(pinned.Valid());
    }
    gov.Shutdown();
}
