// GPU power sampler: pure aggregation (fast, no GPU) + live NVML probe
// (skipped when NVML/driver is unavailable). Host-only.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "lsxcommon/gpu_power.h"

using lsxcommon::perf::ComputePowerStats;
using lsxcommon::perf::GpuPowerAvailable;
using lsxcommon::perf::GpuPowerSampler;
using lsxcommon::perf::PowerStats;
using lsxcommon::perf::ReadGpuPowerWatts;

TEST_CASE("ComputePowerStats: empty is invalid", "[gpu_power]")
{
    const PowerStats s = ComputePowerStats({});
    REQUIRE_FALSE(s.valid);
    REQUIRE(s.samples == 0);
    REQUIRE(s.avg_w == 0.0);
    REQUIRE(s.peak_w == 0.0);
}

TEST_CASE("ComputePowerStats: mean and peak", "[gpu_power]")
{
    const PowerStats s = ComputePowerStats({100.0, 200.0, 300.0});
    REQUIRE(s.valid);
    REQUIRE(s.samples == 3);
    REQUIRE(s.avg_w == 200.0);
    REQUIRE(s.peak_w == 300.0);
}

TEST_CASE("ComputePowerStats: a zero reading is a valid sample", "[gpu_power]")
{
    const PowerStats s = ComputePowerStats({0.0});
    REQUIRE(s.valid);
    REQUIRE(s.samples == 1);
    REQUIRE(s.avg_w == 0.0);
    REQUIRE(s.peak_w == 0.0);
}

TEST_CASE("GpuPowerSampler: injected reader aggregates samples",
          "[gpu_power]")
{
    std::atomic<int> n{0};
    auto reader = [&n](double &w) {
        const int i = n.fetch_add(1);
        w = 100.0 + static_cast<double>(i);
        return true;
    };
    GpuPowerSampler sampler(reader, 1);
    REQUIRE_FALSE(sampler.running());
    sampler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const PowerStats s = sampler.Stop();
    REQUIRE(s.valid);
    REQUIRE(s.samples > 0);
    REQUIRE(s.peak_w >= 100.0);
    // FP epsilon: the sum/n average can round marginally above the exact max.
    REQUIRE(s.peak_w + 1e-9 >= s.avg_w);
    REQUIRE_FALSE(sampler.running());
}

TEST_CASE("GpuPowerSampler: reader failure yields invalid stats", "[gpu_power]")
{
    GpuPowerSampler sampler([](double &) { return false; }, 1);
    sampler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const PowerStats s = sampler.Stop();
    REQUIRE_FALSE(s.valid);
    REQUIRE(s.samples == 0);
}

TEST_CASE("GpuPowerSampler: never started returns invalid", "[gpu_power]")
{
    GpuPowerSampler sampler([](double &) { return true; }, 1);
    const PowerStats s = sampler.Stop();
    REQUIRE_FALSE(s.valid);
}

TEST_CASE("NVML: live board power read is plausible", "[gpu_power]")
{
    if (!GpuPowerAvailable()) SKIP("NVML/driver unavailable");
    double w = -1.0;
    REQUIRE(ReadGpuPowerWatts(w));
    REQUIRE(w >= 0.0);
    REQUIRE(w < 10000.0);
}

TEST_CASE("NVML: live sampler observes power", "[gpu_power]")
{
    if (!GpuPowerAvailable()) SKIP("NVML/driver unavailable");
    GpuPowerSampler sampler({}, 10);
    sampler.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const PowerStats s = sampler.Stop();
    REQUIRE(s.valid);
    REQUIRE(s.samples > 0);
    REQUIRE(s.peak_w + 1e-9 >= s.avg_w);
}
