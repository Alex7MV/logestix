#include "lsxcommon/runtime_tuner.h"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace lsxcommon {
namespace lifecycle {

RuntimeTuner::RuntimeTuner(ResourceGovernor *gov, Lease lease,
                           std::uint64_t slot_stride_bytes,
                           const TuningBounds &bounds, const Config &cfg)
    : gov_(gov), lease_(std::move(lease)),
      stride_(slot_stride_bytes > 0 ? slot_stride_bytes : 1),
      bounds_(bounds), cfg_(cfg)
{
    slots_ = lease_.Valid()
                 ? std::clamp<std::size_t>(
                       static_cast<std::size_t>(lease_.Bytes() / stride_),
                       bounds_.expert_cache_slots_min,
                       bounds_.expert_cache_slots_max)
                 : bounds_.expert_cache_slots_min;
    // Large-but-safe value: the first full window passes the hysteresis gate
    // (a ~SIZE_MAX initializer would wrap to 0 on the first ++).
    since_resize_ = std::size_t{1} << 30;
}

void RuntimeTuner::Report(std::int64_t hits_delta, std::int64_t misses_delta)
{
    w_hits_ += hits_delta;
    w_misses_ += misses_delta;
    ++window_pos_;
    ++since_resize_;
    if (window_pos_ < cfg_.window_steps) return;
    window_pos_ = 0;

    const std::int64_t total = w_hits_ + w_misses_;
    const double miss_ratio =
        total > 0 ? static_cast<double>(w_misses_) / static_cast<double>(total)
                  : 0.0;
    w_hits_ = 0;
    w_misses_ = 0;
    if (total <= 0 || since_resize_ < cfg_.min_interval_steps) return;

    if (miss_ratio >= cfg_.grow_miss_ratio &&
        slots_ < bounds_.expert_cache_slots_max) {
        const std::size_t want =
            std::min(slots_ * 2, bounds_.expert_cache_slots_max);
        if (lease_.Resize(want * stride_)) {
            spdlog::info("[tuner] expert cache {} -> {} slots", slots_, want);
            slots_ = want;
            ++resizes_;
            since_resize_ = 0;
        }
    } else if (miss_ratio <= cfg_.shrink_miss_ratio &&
               slots_ > bounds_.expert_cache_slots_min) {
        const std::size_t want =
            std::max(slots_ / 2, bounds_.expert_cache_slots_min);
        if (lease_.Resize(want * stride_)) {
            spdlog::info("[tuner] expert cache {} -> {} slots", slots_, want);
            slots_ = want;
            ++resizes_;
            since_resize_ = 0;
        }
    }
}

} // namespace lifecycle
} // namespace lsxcommon
