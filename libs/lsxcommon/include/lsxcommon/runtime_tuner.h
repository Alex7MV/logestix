#pragma once

#include <cstddef>
#include <cstdint>

#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {
namespace lifecycle {

// Sliding-window controller for the expert LRU cache size (the only tunable
// in this iteration, spec: RuntimeTuner). Reports arrive between inference
// requests - never inside the hot loop. Growth goes through the governor
// (soft-refused when the budget is occupied); shrink returns budget.
class RuntimeTuner {
public:
    struct Config {
        std::size_t window_steps = 16;      // reports per decision window
        double grow_miss_ratio = 0.10;      // sustained misses -> grow x2
        double shrink_miss_ratio = 0.02;    // sustained hits -> shrink /2
        std::size_t min_interval_steps = 32;  // hysteresis between resizes
    };

    // Takes ownership of the cache lease. slots_ starts at
    // lease.Bytes() / slot_stride_bytes clamped into bounds.
    // NOTE: no default `const Config &cfg = {}` argument here — a default
    // argument referencing a nested class with NSDMIs is ill-formed C++
    // (the type must be complete before the end of the enclosing class;
    // GCC 15+ rejects what GCC 13 tolerated). Use the 4-argument overload.
    RuntimeTuner(ResourceGovernor *gov, Lease lease,
                 std::uint64_t slot_stride_bytes, const TuningBounds &bounds);
    RuntimeTuner(ResourceGovernor *gov, Lease lease,
                 std::uint64_t slot_stride_bytes, const TuningBounds &bounds,
                 const Config &cfg);

    // Feed one report (hits/misses deltas since the previous call).
    void Report(std::int64_t hits_delta, std::int64_t misses_delta);

    std::size_t Slots() const { return slots_; }
    int Resizes() const { return resizes_; }  // diagnostics/tests
    const Lease &CacheLease() const { return lease_; }

private:
    ResourceGovernor *gov_ = nullptr;
    Lease lease_;
    std::uint64_t stride_ = 1;
    TuningBounds bounds_{};
    Config cfg_{};
    std::size_t slots_ = 0;
    std::size_t window_pos_ = 0;
    std::size_t since_resize_ = 0;
    std::int64_t w_hits_ = 0;
    std::int64_t w_misses_ = 0;
    int resizes_ = 0;
};

} // namespace lifecycle
} // namespace lsxcommon
