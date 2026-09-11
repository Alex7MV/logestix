#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/resource_governor.h"

namespace {

lsxcommon::HardwareProfile FakeHw()
{
    lsxcommon::HardwareProfile hw;
    hw.cpu_cores = 8;
    hw.ram_total_bytes = hw.ram_available_bytes = 16ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 8;
    hw.vram_total_bytes = hw.vram_free_bytes = 8ull << 30;
    return hw;
}

} // namespace

TEST_CASE("governor init/shutdown lifecycle", "[governor]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();  // deterministic start regardless of other tests

    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;  // tests grant the full budget
    REQUIRE(gov.Initialize(opts));
    REQUIRE(gov.IsInitialized());
    REQUIRE(gov.hardware().gpu_name == "FakeGPU");
    REQUIRE(gov.hardware().vram_total_bytes == (8ull << 30));
    REQUIRE(gov.CommittedVramBytes() == 0);
    REQUIRE(gov.CommittedRamBytes() == 0);

    SECTION("re-init is idempotent: first options win")
    {
        lsxcommon::GovernorOptions same;
        same.override_hw = true;
        same.hw_override = FakeHw();
        same.vram_keep_out_bytes = 12345;
        REQUIRE(gov.Initialize(same));
        REQUIRE(gov.options().vram_keep_out_bytes == 0);
    }

    gov.Shutdown();
    REQUIRE_FALSE(gov.IsInitialized());
}

TEST_CASE("required lease granted within budget", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();  // 8 GiB VRAM
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             4ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());
    REQUIRE(lease.Bytes() == (4ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (4ull << 30));
    REQUIRE(gov.CommittedRamBytes() == 0);  // VRAM lease did not touch RAM

    {   // RAM lease lands in the other domain
        auto ram = gov.Acquire(lsxcommon::LeaseKind::kHostPinned,
                               1ull << 30, lsxcommon::LeasePolicy{});
        REQUIRE(ram.Valid());
        REQUIRE(gov.CommittedRamBytes() == (1ull << 30));
    }
    REQUIRE(gov.CommittedRamBytes() == 0);  // RAII release
    gov.Shutdown();
}

TEST_CASE("required lease refused on shortage (strict)", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    auto first = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(first.Valid());
    auto second = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                              4ull << 30, lsxcommon::LeasePolicy{});  // 10 > 8
    REQUIRE_FALSE(second.Valid());
    REQUIRE(gov.CommittedVramBytes() == (6ull << 30));  // unchanged
    gov.Shutdown();
}

TEST_CASE("keep-out margin shrinks the grantable budget", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 2ull << 30;  // grantable = 6 GiB
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());
    auto over = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                            1ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE_FALSE(over.Valid());
    gov.Shutdown();
}

TEST_CASE("relaxed mode grants required leases despite shortage", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    opts.relax_memory_refusal = true;  // TEST-ONLY semantics
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             10ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());  // granted anyway, accounting may exceed budget
    gov.Shutdown();
}

TEST_CASE("acquire before initialize fails", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             1ull << 20, lsxcommon::LeasePolicy{});
    REQUIRE_FALSE(lease.Valid());
}

TEST_CASE("adaptive lease clamps into policy bounds", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    lsxcommon::LeasePolicy adaptive{
        lsxcommon::LeasePolicy::Mode::kAdaptive,
        /*min*/ 2ull << 20, /*max*/ 8ull << 20};
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                             100ull << 30, adaptive);  // clamped to max
    REQUIRE(lease.Valid());
    REQUIRE(lease.Bytes() == (8ull << 20));
    gov.Shutdown();
}

TEST_CASE("resize grows within budget and refuses beyond it", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    lsxcommon::LeasePolicy adaptive{
        lsxcommon::LeasePolicy::Mode::kAdaptive,
        /*min*/ 1ull << 20, /*max*/ 8ull << 30};
    auto blocker = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                               6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(blocker.Valid());
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                             1ull << 30, adaptive);
    REQUIRE(lease.Valid());

    // Growth that fits: committed 6 GiB + 1 GiB -> +1.5 GiB <= 8 GiB.
    REQUIRE(lease.Resize(2ull << 30));
    REQUIRE(lease.Bytes() == (2ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (8ull << 30));

    // Growth beyond budget: refused, size kept (soft semantics).
    REQUIRE_FALSE(lease.Resize(4ull << 30));
    REQUIRE(lease.Bytes() == (2ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (8ull << 30));

    // Shrink always allowed: returns budget.
    REQUIRE(lease.Resize(1ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (7ull << 30));

    // Resize below the adaptive floor is clamped up to min_bytes.
    REQUIRE(lease.Resize(0));
    REQUIRE(lease.Bytes() == (1ull << 20));

    gov.Shutdown();
}

TEST_CASE("resize on invalid lease fails", "[governor][lease]")
{
    lsxcommon::Lease empty;
    REQUIRE_FALSE(empty.Resize(1ull << 20));
}

TEST_CASE("vram reconciliation detects over-accounting drift", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();  // total 8 GiB
    opts.vram_keep_out_bytes = 0;
    opts.relax_memory_refusal = true;
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());

    // Consistent world: 2 GiB free + 6 GiB committed == 8 GiB total.
    REQUIRE(gov.ReconcileVram(2ull << 30));
    // Drift: books claim more than the device can have outstanding.
    REQUIRE_FALSE(gov.ReconcileVram(3ull << 30));  // 3+6 > 8+slack(256MiB)
    gov.Shutdown();
}

// ── Expert-cache VRAM residency policy (launch strategy, all models) ──

TEST_CASE("ExpertCacheSlots: reserve subtracted, clamped, degenerate-safe",
          "[governor][residency]")
{
    using lsxcommon::lifecycle::ExpertCacheSlots;
    using lsxcommon::lifecycle::kExpertCacheVramReserveBytes;

    // Reserve is a real constant and stays bounded (never zero: decode
    // needs KV/activation/TC-workspace headroom).
    REQUIRE(kExpertCacheVramReserveBytes > 0);
    REQUIRE(kExpertCacheVramReserveBytes <= (8ull << 30));

    const std::size_t per_expert = 2ull << 20;  // 2 MiB

    SECTION("free below reserve -> nothing fits")
    {
        REQUIRE(ExpertCacheSlots((2ull << 30) - 1, per_expert, 10240) == 0);
        REQUIRE(ExpertCacheSlots(0, per_expert, 10240) == 0);
    }
    SECTION("affordable slots from usable bytes")
    {
        // usable = free - reserve
        const std::size_t free_bytes = kExpertCacheVramReserveBytes +
                                       (5 * per_expert);
        REQUIRE(ExpertCacheSlots(free_bytes, per_expert, 10240) == 5);
    }
    SECTION("clamped to total experts")
    {
        const std::size_t free_bytes = kExpertCacheVramReserveBytes +
                                       (9999 * per_expert);
        REQUIRE(ExpertCacheSlots(free_bytes, per_expert, 256) == 256);
    }
    SECTION("zero per-expert degenerates to total")
    {
        REQUIRE(ExpertCacheSlots((1ull << 30), 0, 512) == 512);
    }
    SECTION("zero total experts -> 0")
    {
        REQUIRE(ExpertCacheSlots((1ull << 30), per_expert, 0) == 0);
    }
}
