// Bandwidth floor for the pinned expert staging ring.
//
// WHY A BUDGET AND NOT A PERF CLAIM: the ring exists because the pageable copy
// path measured 5.2 GB/s on real expert files while the same bytes through
// pinned staging measured 10.5 GB/s (39.8 GB/s array ceiling). This test pins
// the second number so a future change cannot quietly hand the ring back to the
// slow path — a single cudaMemcpyAsync from a pageable source instead of the
// pinned slot, a pread that falls back to single-threaded, an event recorded
// before the copies.
//
// WHAT IT DOES NOT MEASURE, so nobody reads it as more than it is: it stages
// from a SYNTHETIC store whose files are already in the page cache, so it is the
// pinned-DMA + pread-from-cache leg only. The cold-NVMe dimension belongs to the
// end-to-end perf row (`perf baseline: kimi-k3.1-pro`), where the tree is 1.4 TB
// against a 56 GB page cache and every read is a real fault.
//
// Nor is it a floor for the model's throughput: staging is one term of
// wall/token, and the ring's engagement counters (stage_stats) are what say
// whether the term applies at all.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/schema.h"
#include "test_kernel_budget.h"

namespace {

struct StageEntry {
    const uint8_t *host_w1 = nullptr, *host_w2 = nullptr, *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr, *host_w2_scales = nullptr,
                  *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f, w2_global_scale = 1.0f, w3_global_scale = 1.0f;
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    int64_t w1_scales_bytes = 0, w2_scales_bytes = 0, w3_scales_bytes = 0;
};

using Store = lsxcommon::gpu::expert_store::ExpertStore<StageEntry>;

// Real Kimi-K3.1-Pro plane ratio for the latent MoE: 3584x3072 packed is
// 5 505 024 bytes with 688 128 of scales, i.e. 8:1. Sized so the measured work
// is dominated by the copy rather than by per-call latency: 16 experts (one
// decode layer's top-k) per repetition.
constexpr int kExperts = 16;
constexpr size_t kPacked = 1u << 20;        // 1 MiB per packed plane
constexpr size_t kScales = kPacked / 8;     // 8:1, as on the real tree

std::filesystem::path MakeStore(int experts)
{
    const auto container =
        std::filesystem::temp_directory_path() /
        ("lsx_stage_bw_" + std::to_string(std::rand()));
    const auto exps = container / "logestix" / "experts";
    std::filesystem::create_directories(exps / "layer_0");
    for (int i = 0; i < experts; ++i) {
        lsxcommon::PackedExpert pe;
        pe.w1_packed.assign(kPacked, static_cast<uint8_t>(0x10 + i));
        pe.w2_packed.assign(kPacked, static_cast<uint8_t>(0x20 + i));
        pe.w3_packed.assign(kPacked, static_cast<uint8_t>(0x30 + i));
        pe.w1_scales.assign(kScales, static_cast<uint8_t>(0x80 + i));
        pe.w2_scales.assign(kScales, static_cast<uint8_t>(0x88 + i));
        pe.w3_scales.assign(kScales, static_cast<uint8_t>(0x90 + i));
        pe.w1_global_scale = 1.0f;
        pe.w2_global_scale = 1.0f;
        pe.w3_global_scale = 1.0f;
        auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
        REQUIRE(batch.ok());
        REQUIRE(lsxcommon::WriteRecordBatchToFile(
                    *batch, exps / "layer_0" / ("exp_" + std::to_string(i) +
                                                ".arrow"))
                    .ok());
    }
    return container;
}

}  // namespace

TEST_CASE("ExpertStore staging ring sustains its bandwidth floor",
          "[expert_store][stage][budget]")
{
    // A Debug build measures an unoptimized binary; there is no useful number
    // from it (same rule as test_perf_baseline).
#ifndef NDEBUG
    SKIP("bandwidth budget requires a Release build (NDEBUG)");
#else
    const auto dir = MakeStore(kExperts);
    Store s;
    s.num_layers = 1;
    s.experts_per_layer = kExperts;
    s.max_open = kExperts;
    s.stage_ring.slots = kExperts;      // one slot per expert: no recycle stalls
    s.stage_ring.pread_threads = 4;
    REQUIRE(s.Load(dir.string(), false));
    REQUIRE(s.InitStageRing());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const size_t one = 3 * (kPacked + kScales);
    const int64_t off[6] = {0,
                            static_cast<int64_t>(2 * (kPacked + kScales)),
                            static_cast<int64_t>(kPacked + kScales),
                            static_cast<int64_t>(kPacked),
                            static_cast<int64_t>(2 * (kPacked + kScales) +
                                                 kPacked),
                            static_cast<int64_t>(kPacked + kScales + kPacked)};
    uint8_t *dev = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev), one) == cudaSuccess);

    // Warm-up pass: first touch of each file pays the page fault and the first
    // cudaHostAlloc has already happened, so measuring it would report the
    // filesystem rather than the ring.
    for (size_t idx = 0; idx < static_cast<size_t>(kExperts); ++idx) {
        REQUIRE(s.EnqueueStage(idx, dev, off, stream));
        s.AwaitStage(idx, stream);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Best of five, per the house rule for budget measurements.
    double best_gbps = 0.0;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t idx = 0; idx < static_cast<size_t>(kExperts); ++idx) {
            REQUIRE(s.EnqueueStage(idx, dev, off, stream));
            s.AwaitStage(idx, stream);
        }
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        const double gbps =
            static_cast<double>(one) * kExperts / secs / 1e9;
        if (gbps > best_gbps) best_gbps = gbps;
    }

    std::printf("[stage_bw] %.2f GB/s over %d experts x %zu B\n", best_gbps,
                kExperts, one);
    // The budget is machine-keyed: a record for this gpu_key asserts, no record
    // SKIPs (never judge), LSX_KERNEL_BUDGET_WRITE=1 records. A bare constant
    // here could not tell a regression from a different card.
    kernel_budget::Require(
        "expert_store.stage_ring_gbps", best_gbps, kernel_budget::Kind::kMin,
        "pinned pread + DMA staging of one expert layer (16 x 3.56 MiB), best "
        "of 5, files page-cache warm. Reference on rtx 5090: pageable copies "
        "5.2 GB/s vs this path 10.5 GB/s on the real tree; the cold-NVMe "
        "dimension is the E2E perf row, not this test.");

    CHECK(s.stage_stats.staged_experts >= 6 * kExperts);
    CHECK(s.stage_stats.short_reads == 0);

    cudaFree(dev);
    cudaStreamDestroy(stream);
    std::filesystem::remove_all(dir);
#endif
}
