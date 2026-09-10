#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/gpu_perf.h"

using lsxcommon::perf::GpuPerfProfile;
using lsxcommon::perf::GpuProfileKey;
using lsxcommon::perf::PeakBandwidthGbs;

TEST_CASE("PeakBandwidthGbs: bus_width * mem_clock * 2 DDR factor", "[gpu_perf]")
{
    GpuPerfProfile g;
    g.bus_width_bits = 512;
    g.memory_clock_khz = 1750000;  // 1.75 GHz * 2 (DDR) * 512 bit / 8 = 224 GB/s
    REQUIRE(PeakBandwidthGbs(g) == 224.0);

    GpuPerfProfile ddr;
    ddr.bus_width_bits = 256;
    ddr.memory_clock_khz = 1000000;  // 1 GHz * 2 * 256/8 = 64 GB/s
    REQUIRE(PeakBandwidthGbs(ddr) == 64.0);
}

TEST_CASE("GpuProfileKey: lowercase, vendor prefix stripped", "[gpu_perf]")
{
    GpuPerfProfile g;
    g.name = "NVIDIA GeForce RTX 5090";
    REQUIRE(GpuProfileKey(g) == "rtx 5090");
    g.name = "NVIDIA RTX 6000 Ada";
    REQUIRE(GpuProfileKey(g) == "rtx 6000 ada");
    g.name = "";
    REQUIRE(GpuProfileKey(g).empty());
}
