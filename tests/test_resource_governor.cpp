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
