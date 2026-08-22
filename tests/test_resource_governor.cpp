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
