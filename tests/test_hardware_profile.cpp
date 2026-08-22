#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

#include "lsxcommon/resource_governor.h"

namespace {

int FakeCores() { return 16; }
void FakeRam(std::uint64_t *total, std::uint64_t *avail)
{
    *total = 32ull << 30;
    *avail = 12ull << 30;
}
void FakeGpu(std::string *name, int *sm)
{
    *name = "TestGPU";
    *sm = 148;
}
void FakeVram(std::uint64_t *total, std::uint64_t *free_b)
{
    *total = 32ull << 30;
    *free_b = 31ull << 30;
}

} // namespace

TEST_CASE("hardware profile aggregates raw queries", "[governor][hw]")
{
    lsxcommon::HardwareQueries q{FakeCores, FakeRam, FakeGpu, FakeVram};
    const auto hw = lsxcommon::BuildHardwareProfile(q);
    REQUIRE(hw.cpu_cores == 16);
    REQUIRE(hw.ram_total_bytes == (32ull << 30));
    REQUIRE(hw.ram_available_bytes == (12ull << 30));
    REQUIRE(hw.gpu_name == "TestGPU");
    REQUIRE(hw.gpu_sm_count == 148);
    REQUIRE(hw.vram_total_bytes == (32ull << 30));
    REQUIRE(hw.vram_free_bytes == (31ull << 30));
}

TEST_CASE("null query entries leave fields defaulted", "[governor][hw]")
{
    const auto hw = lsxcommon::BuildHardwareProfile(lsxcommon::HardwareQueries{});
    REQUIRE(hw.cpu_cores == 0);
    REQUIRE(hw.ram_total_bytes == 0);
    REQUIRE(hw.gpu_name.empty());
    REQUIRE(hw.gpu_sm_count == 0);
    REQUIRE(hw.vram_total_bytes == 0);
}

TEST_CASE("real probe yields sane values", "[governor][hw][smoke]")
{
    const auto hw = lsxcommon::ProbeHardware();
    REQUIRE(hw.cpu_cores > 0);
    REQUIRE(hw.ram_total_bytes > 0);
    REQUIRE(hw.ram_available_bytes <= hw.ram_total_bytes);
    // GPU fields are machine-dependent (CI may lack a device): only check
    // consistency when a device was found.
    if (!hw.gpu_name.empty()) {
        REQUIRE(hw.gpu_sm_count > 0);
        REQUIRE(hw.vram_total_bytes > 0);
    }
}
