#pragma once

// ── GPU board-power sampling (model-neutral, host-only) ─────────────────────
// NVML is resolved at runtime (dlopen on Linux / LoadLibrary on Windows), so
// the build has no link dependency on the driver and a headless or driverless
// host degrades to "unavailable" instead of failing to start. NEVER include
// from .cu/.cuh TUs (mirrors gpu_perf.h).

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace lsxcommon {
namespace perf {

struct PowerStats {
    double avg_w = 0.0;
    double peak_w = 0.0;
    int samples = 0;
    bool valid = false;  // false => no reading (NVML unavailable / all failed)
};

// Pure aggregation: empty -> invalid; otherwise mean and max over the samples.
PowerStats ComputePowerStats(const std::vector<double> &samples_w);

// One instantaneous board-power read (NVML device 0), in watts. Returns false
// when NVML or the device handle is unavailable.
bool ReadGpuPowerWatts(double &out_w);

// True when the NVML library loaded and a device handle was resolved.
bool GpuPowerAvailable();

// Enforced power limit in watts (0 when unknown). Context for a measurement.
double GpuPowerLimitW();

// Samples board power on a background thread between Start() and Stop(). The
// reader is injectable for tests; the default is ReadGpuPowerWatts. A reader
// returning false stops sampling (the samples seen so far are kept).
class GpuPowerSampler {
public:
    explicit GpuPowerSampler(std::function<bool(double &)> reader = {},
                             int interval_ms = 20);
    ~GpuPowerSampler();

    GpuPowerSampler(const GpuPowerSampler &) = delete;
    GpuPowerSampler &operator=(const GpuPowerSampler &) = delete;

    void Start();
    PowerStats Stop();
    bool running() const { return running_.load(); }

private:
    std::function<bool(double &)> reader_;
    int interval_ms_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::vector<double> samples_;
};

} // namespace perf
} // namespace lsxcommon
