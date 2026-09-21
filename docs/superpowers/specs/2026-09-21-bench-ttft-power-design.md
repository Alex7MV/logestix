# Benchmark TTFT + GPU Power Measurement Design

Date: 2026-09-21
Status: approved (design dialogue 2026-09-21)

## Problem

`lsxbenchmark --bench` and the perf-baseline test already measure TTFT (the
per-family generation result carries `ttft_ms`, and it is printed by
`RunCliBenchmark`), but the committed reference `tests/perf/perf_baselines.json`
records only `tokens_per_sec`. TTFT is therefore lost between runs: a regression
in prefill/first-token latency is invisible to the baseline guard.

Inference energy is not measured at all. The driver exposes instantaneous board
power through NVML (`nvmlDeviceGetPowerUsage`, plus `nvmlDeviceGetEnforcedPowerLimit`
for context), so average and peak watts over a run can be captured without
changing the numeric path.

## Decisions (design dialogue 2026-09-21)

1. Persist TTFT and power in `perf_baselines.json`; the pass/degrade verdict
   stays driven by `tokens_per_sec` only (TTFT/power are recorded evidence, not
   gates yet).
2. Measure power with NVML loaded dynamically (`dlopen` / `LoadLibrary`) so the
   build has no hard link dependency and headless / driverless hosts degrade
   gracefully to "unavailable".
3. Record both average and peak watts, plus the number of samples, so a coarse
   sampling run is distinguishable from an absent one.
4. Sampling runs on a background host thread for the duration of `Infer`.

## Part A — baseline schema

`PerfBaselineEntry` (host-only, `lsxcommon/perf_baseline.h`) gains:

```cpp
double ttft_ms = 0.0;        // stream start -> first generated token
double avg_power_w = 0.0;    // mean board power over the run (0 = unknown)
double peak_power_w = 0.0;   // peak board power over the run (0 = unknown)
int    power_samples = 0;    // 0 => NVML unavailable / not sampled
```

`kPerfBaselineFormatVersion` becomes 3. The parser accepts 1, 2 and 3; the new
fields are optional so committed v1/v2 files keep loading. The serializer always
writes them (stable order, after `gen_tokens`). A 0 in the power fields is the
"unknown" sentinel — no extra presence flag.

`tests/test_perf_baseline.cpp` captures `r.ttft_ms` and the sampler stats in
`MeasureModel` / `MeasureResult`, and writes them on the upsert path. The
`JudgePerfLen` verdict is unchanged.

`RunCliBenchmark` prints a `power` line (avg/peak W) in addition to the existing
`ttft` line.

## Part B — `gpu_power` module

New model-neutral host-only files (naming-guard exempt, mirroring `gpu_perf`):
`libs/lsxcommon/include/lsxcommon/gpu_power.h` + `src/gpu_power.cpp`. NEVER
included from `.cu`/`.cuh`.

```cpp
namespace lsxcommon::perf {

struct PowerStats {
    double avg_w = 0.0;
    double peak_w = 0.0;
    int    samples = 0;
    bool   valid = false;    // false => NVML unavailable, all values 0
};

// Pure aggregation (unit-tested without a GPU): empty -> invalid; otherwise
// mean and max over the samples.
PowerStats ComputePowerStats(const std::vector<double> &samples_w);

// One instantaneous board-power read in watts (NVML device 0). false when
// NVML/driver is unavailable.
bool ReadGpuPowerWatts(double &out_w);

// True when the NVML library loaded and a device handle was resolved.
bool GpuPowerAvailable();

// Samples board power on a background thread between Start() and Stop().
// The reader is injectable for tests; the default is ReadGpuPowerWatts.
class GpuPowerSampler {
public:
    explicit GpuPowerSampler(
        std::function<bool(double &)> reader = {}, int interval_ms = 20);
    ~GpuPowerSampler();          // stops if still running
    void Start();
    PowerStats Stop();           // joins the thread, returns the aggregate
    bool running() const;
private:
    ...
};

} // namespace lsxcommon::perf
```

NVML is resolved lazily on first use: `libnvidia-ml.so.1` on Linux, `nvml.dll`
on Windows. Function pointers for `nvmlInit_v2`, `nvmlDeviceGetHandleByIndex_v2`,
`nvmlDeviceGetPowerUsage`, `nvmlDeviceGetEnforcedPowerLimit`, `nvmlShutdown`.
Initialization is once-guarded; every failure path leaves the module
unavailable and never throws. `nvmlDeviceGetPowerUsage` returns milliwatts.

The include is the CUDA-toolkit `nvml.h` (CUDA 13.4). No `nvml*` symbol is
referenced directly, so nothing links against the driver library or its stubs.

## Testing (TDD)

Fast label, no GPU/weights:
- JSON round-trip of the four new fields; v1/v2 files without them still parse;
  serializer field order; foreign version rejected (extend
  `test_perf_baseline_logic.cpp`).
- `ComputePowerStats`: empty -> invalid; single sample; multiple samples
  (mean, max); all-zero run.

GPU/driver-gated (`SKIP` + `SKIP_RETURN_CODE 4` when NVML unavailable):
- `ReadGpuPowerWatts` returns true, `0 < w <= enforced_limit`; a
  `GpuPowerSampler` around a short busy loop yields `samples > 0` and
  `valid == true`.

Integration: `test_perf_baseline.cpp` records the real TTFT/power; the
`--bench` smoke path is covered by the existing lsxbenchmark CLI test.

## Non-goals

- No power-based pass/degrade verdict (recording only).
- No per-kernel or per-phase energy attribution.
- No sampling during `Load` (weights streaming is I/O, not the measured path).
