#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/runtime_tuner.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

namespace {

HardwareProfile FakeHw()
{
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 16ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 8;
    hw.vram_total_bytes = hw.vram_free_bytes = 8ull << 30;
    return hw;
}

constexpr std::uint64_t kStride = 100;  // bytes per cache slot

struct Harness {
    ResourceGovernor &gov = ResourceGovernor::Instance();
    // window=4 reports; hysteresis gap=8 steps (> one window, so exactly one
    // full window is skipped between resizes).
    RuntimeTuner::Config cfg{/*.window_steps*/ 4, /*grow*/ 0.10,
                             /*shrink*/ 0.02, /*min_interval*/ 8};
    Harness()
    {
        gov.Shutdown();
        GovernorOptions o;
        o.override_hw = true;
        o.hw_override = FakeHw();
        o.vram_keep_out_bytes = 0;
        (void)gov.Initialize(o);
    }
    ~Harness() { gov.Shutdown(); }

    RuntimeTuner Make(std::size_t start_slots,
                      LeasePolicy::Mode mode = LeasePolicy::Mode::kAdaptive)
    {
        LeasePolicy p{mode, 2 * kStride, 16 * kStride};
        Lease lease = gov.Acquire(LeaseKind::kExpertCache,
                                  start_slots * kStride, p);
        return RuntimeTuner(&gov, std::move(lease), kStride,
                            TuningBounds{2, 16}, cfg);
    }
};

} // namespace

TEST_CASE("tuner starts at the lease-derived slot count", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    REQUIRE(t.Slots() == 4);
    REQUIRE(t.Resizes() == 0);
}

TEST_CASE("sustained misses grow the cache x2 within bounds", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    // Window 1: heavy misses -> grow 4 -> 8 (first window always acts).
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 8);
    REQUIRE(t.Resizes() == 1);
    // Window 2 ends with since_resize_ = 4 < min_interval (8) -> skipped.
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 8);
    REQUIRE(t.Resizes() == 1);
    // Window 3 ends with since_resize_ = 8 -> gate open -> grow 8 -> 16 (max).
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 16);
    REQUIRE(t.Resizes() == 2);
    // Already at max: further misses do nothing.
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 16);
    REQUIRE(t.Resizes() == 2);
}

TEST_CASE("sustained hits shrink the cache", "[tuner]")
{
    Harness h;
    auto t = h.Make(8);
    for (int i = 0; i < 4; ++i) t.Report(10, 0);  // ratio 0 <= shrink 0.02
    REQUIRE(t.Slots() == 4);
}

TEST_CASE("growth is soft-refused when the budget is occupied", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    REQUIRE(t.CacheLease().Valid());
    // Fill the remaining budget exactly: growth (+400 B) would exceed it.
    const std::uint64_t rest =
        (8ull << 30) - t.CacheLease().Bytes();
    auto blocker = h.gov.Acquire(LeaseKind::kPermanentWeights, rest,
                                 LeasePolicy{});
    REQUIRE(blocker.Valid());
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 4);   // refused: size kept
    REQUIRE(t.Resizes() == 0); // no crash, soft failure
}
