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

// One instantaneous GPU-core temperature read (NVML device 0), degrees Celsius.
// Returns false when NVML or the temperature query is unavailable.
bool ReadGpuTemperatureC(double &out_c);

// True when the NVML library loaded and a device handle was resolved.
bool GpuPowerAvailable();

// State threaded through the cooldown wait (see CooldownStep).
struct CooldownState {
    double min_c = 0.0;
    int flat = 0;
    bool have_min = false;
};

// Pure plateau predicate for the between-target cooldown. The wait ends when
// the GPU has stopped shedding heat: `stable_polls` consecutive readings lie
// within `eps_c` of the running minimum (as cool as the environment allows).
// An unreadable temperature also ends the wait, so a driverless or CI host
// never blocks. An absolute temperature target is deliberately NOT used: a
// fixed threshold that the ambient cannot reach would spin to the timeout.
bool CooldownStep(CooldownState &state, bool have_temp, double temp_c,
                  int stable_polls, double eps_c);

// Block until CooldownStep ends the wait, polling every poll_ms, giving up
// after timeout_s (<= 0 disables the wait). Returns true when cooled (or not
// measurable / disabled), false on timeout. `reader` is injectable for tests;
// the default reads NVML. The perf harness calls this before each model: a
// card reused while still hot settles ~11% below its max SM clock here
// (measured 2812/3150 MHz), costing ~4% of t/s and reading as a false
// DEGRADED against a cool-room reference.
bool CoolDownGpu(int timeout_s, int poll_ms = 3000, int stable_polls = 3,
                 double eps_c = 0.5, std::function<bool(double &)> reader = {});

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
