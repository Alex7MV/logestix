// GPU power sampler: pure aggregation (fast, no GPU) + live NVML probe
// (skipped when NVML/driver is unavailable). Host-only.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "lsxcommon/gpu_power.h"

using lsxcommon::perf::ComputePowerStats;
using lsxcommon::perf::CooldownState;
using lsxcommon::perf::CooldownStep;
using lsxcommon::perf::CoolDownGpu;
using lsxcommon::perf::GpuPowerAvailable;
using lsxcommon::perf::GpuPowerSampler;
using lsxcommon::perf::PowerStats;
using lsxcommon::perf::ReadGpuPowerWatts;
using lsxcommon::perf::ReadGpuTemperatureC;

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

TEST_CASE("CooldownStep: unreadable temperature ends the wait", "[gpu_power]")
{
    CooldownState s;
    REQUIRE(CooldownStep(s, false, 0.0, 3, 0.5));
}

TEST_CASE("CooldownStep: ends on a plateau at the running minimum",
          "[gpu_power]")
{
    CooldownState s;
    // A falling reading restarts the plateau counter ...
    REQUIRE_FALSE(CooldownStep(s, true, 70.0, 3, 0.5));
    REQUIRE_FALSE(CooldownStep(s, true, 65.0, 3, 0.5));
    REQUIRE_FALSE(CooldownStep(s, true, 60.0, 3, 0.5));
    // ... equal readings build it ...
    REQUIRE_FALSE(CooldownStep(s, true, 60.0, 3, 0.5));
    REQUIRE_FALSE(CooldownStep(s, true, 60.0, 3, 0.5));
    // ... and the third stable poll ends the wait.
    REQUIRE(CooldownStep(s, true, 60.0, 3, 0.5));
}

TEST_CASE("CooldownStep: a reading above the minimum keeps counting",
          "[gpu_power]")
{
    CooldownState s;
    REQUIRE_FALSE(CooldownStep(s, true, 55.0, 2, 0.5));
    REQUIRE_FALSE(CooldownStep(s, true, 56.0, 2, 0.5));
    REQUIRE(CooldownStep(s, true, 56.0, 2, 0.5));
}

TEST_CASE("CoolDownGpu: disabled or unreadable returns immediately",
          "[gpu_power]")
{
    REQUIRE(CoolDownGpu(0, 1));
    REQUIRE(CoolDownGpu(5, 1, 3, 0.5, [](double &) { return false; }));
}

TEST_CASE("CoolDownGpu: waits until the reader plateaus", "[gpu_power]")
{
    double t = 70.0;
    auto reader = [&t](double &out) {
        out = t;
        if (t > 60.0) t -= 5.0;  // cool, then hold at 60
        return true;
    };
    REQUIRE(CoolDownGpu(5, 1, 3, 0.5, reader));
}

TEST_CASE("CoolDownGpu: a never-settling reader times out", "[gpu_power]")
{
    double t = 200.0;
    auto reader = [&t](double &out) {
        out = t;
        t -= 1.0;  // always falling -> never plateaus
        return true;
    };
    REQUIRE_FALSE(CoolDownGpu(1, 1, 3, 0.5, reader));
}

TEST_CASE("NVML: live temperature read is plausible", "[gpu_power]")
{
    double c = -1.0;
    if (!ReadGpuTemperatureC(c)) SKIP("NVML temperature unavailable");
    REQUIRE(c >= -50.0);
    REQUIRE(c < 200.0);
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
