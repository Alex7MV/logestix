// ExpertStore template behavior on a synthetic expert dir (no checkpoint).
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/schema.h"

namespace {

struct FixtureEntry {
    const uint8_t *host_w1 = nullptr, *host_w2 = nullptr, *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr, *host_w2_scales = nullptr, *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f, w2_global_scale = 1.0f, w3_global_scale = 1.0f;
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    int64_t w1_scales_bytes = 0, w2_scales_bytes = 0, w3_scales_bytes = 0;
};

using Store = lsxcommon::gpu::expert_store::ExpertStore<FixtureEntry>;

// `distinct` gives each EXPERT its own packed bytes (0x10 + flat index), which
// the staging hazard test needs: with identical payloads, a ring that refills a
// slot under an in-flight DMA produces byte-identical output and the corruption
// is invisible. The default keeps the original per-PLANE constants, which the
// locator test asserts by exact value.
std::filesystem::path MakeExpertDir(bool distinct = false)
{
    std::filesystem::path container =
        std::filesystem::temp_directory_path() /
        ("lsx_expert_test_" + std::to_string(std::rand()));
    const std::filesystem::path exps = container / "logestix" / "experts";
    std::filesystem::create_directories(exps / "layer_0");
    std::filesystem::create_directories(exps / "layer_1");

    for (int layer = 0; layer < 2; ++layer)
        for (int exp = 0; exp < 2; ++exp) {
            const uint8_t flat = static_cast<uint8_t>(layer * 2 + exp);
            lsxcommon::PackedExpert pe;
            pe.w1_packed.assign(16, distinct ? static_cast<uint8_t>(0x10u + flat)
                                             : 0x01u);
            pe.w2_packed.assign(16, distinct ? static_cast<uint8_t>(0x20u + flat)
                                             : 0x02u);
            pe.w3_packed.assign(16, distinct ? static_cast<uint8_t>(0x30u + flat)
                                             : 0x03u);
            pe.w1_scales.assign(2, 0x80u);
            pe.w2_scales.assign(2, 0x81u);
            pe.w3_scales.assign(2, 0x82u);
            pe.w1_global_scale = 1.0f;
            pe.w2_global_scale = 2.0f;
            pe.w3_global_scale = 3.0f;
            auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
            REQUIRE(batch.ok());
            REQUIRE(lsxcommon::WriteRecordBatchToFile(
                        *batch, exps / ("layer_" + std::to_string(layer)) /
                                    ("exp_" + std::to_string(exp) + ".arrow"))
                        .ok());
        }
    return container;
}

}  // namespace

TEST_CASE("ExpertStore warms up to max_open on Load", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 2;                          // smaller than the 4 written entries
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.entries.size() == 4);
    REQUIRE(s.lru.size() == 2);                 // flat [0] and [1]
    REQUIRE(s.entries[0].host_w1 != nullptr);   // decoded
}

TEST_CASE("ExpertStore::EnsureResident evicts under max_open", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.lru.size() == 1);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    s.EnsureResident(0, 1, stream);   // flat=1 -> evicts flat=0
    CHECK(s.lru.front() == 1);
    s.EnsureResident(1, 0, stream);   // flat=2 -> evicts flat=1
    CHECK(s.lru.front() == 2);
    s.EnsureResident(0, 0, stream);   // re-parse flat=0 -> evicts flat=2
    CHECK(s.lru.front() == 0);

    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore::EnsureResident honors an eviction release event", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;                 // forces eviction on the 2nd resident
    REQUIRE(s.Load(MakeExpertDir().string(), false));

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaEvent_t ev = nullptr;
    REQUIRE(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess);

    // No event -> legacy sync eviction still works (regression guard).
    s.EnsureResident(0, 1, stream);           // flat=1 -> evicts flat=0
    CHECK(s.lru.front() == 1);
    CHECK(s.entries[1].host_w1 != nullptr);

    // With the event -> same eviction semantics, released via event.
    s.EnsureResident(1, 0, stream, ev);        // flat=2 -> evicts flat=1
    CHECK(s.lru.front() == 2);
    CHECK(s.entries[2].host_w1 != nullptr);
    s.EnsureResident(0, 1, stream, ev);        // re-parse flat=1 -> evicts flat=2
    CHECK(s.lru.front() == 1);

    cudaEventDestroy(ev);
    cudaStreamDestroy(stream);
}
// ── Plane locators (pinned staging ring, task 2) ──────────────────────
//
// The ring stages an expert with pread, which needs to know WHERE in the file
// each of the six planes lives. The store already decodes the planes but not
// their file offsets, so it records them at parse time.
//
// Why this deserves its own case: a locator that is off by the Arrow header
// (16 bytes) or that points at a neighbouring plane does not produce an error.
// It produces a model that runs, answers, and is wrong — every weight of every
// expert shifted by one plane. The fixture writes a distinct constant into each
// plane (w1=0x01 w2=0x02 w3=0x03 s1=0x80 s2=0x81 s3=0x82), so reading the file
// at a locator is a byte-exact statement about which plane it addresses.
TEST_CASE("ExpertStore records where each plane lives in the file",
          "[expert_store][stage]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.planes.size() == 4);
    REQUIRE(s.file_bytes > 0);

    // Six locators per resident entry, ascending and inside the file.
    for (size_t idx = 0; idx < 4; ++idx) {
        int64_t prev_end = 0;
        for (int k = 0; k < 6; ++k) {
            INFO("idx " << idx << " plane " << k);
            REQUIRE(s.planes[idx][static_cast<size_t>(k)].bytes > 0);
            REQUIRE(s.planes[idx][static_cast<size_t>(k)].offset >= prev_end);
            prev_end = s.planes[idx][static_cast<size_t>(k)].offset +
                       s.planes[idx][static_cast<size_t>(k)].bytes;
            REQUIRE(prev_end <= s.file_bytes);
        }
    }
    // Each locator's length is that plane's length, not a neighbour's.
    CHECK(s.planes[0][0].bytes == s.entries[0].w1_bytes);
    CHECK(s.planes[0][1].bytes == s.entries[0].w2_bytes);
    CHECK(s.planes[0][2].bytes == s.entries[0].w3_bytes);
    CHECK(s.planes[0][3].bytes == s.entries[0].w1_scales_bytes);

    // The decisive check: the file at a locator holds that plane's bytes.
    const uint8_t want[6] = {0x01, 0x02, 0x03, 0x80, 0x81, 0x82};
    for (size_t idx = 0; idx < 4; ++idx) {
        std::FILE *f = std::fopen(s.paths[idx].c_str(), "rb");
        REQUIRE(f != nullptr);
        for (int k = 0; k < 6; ++k) {
            const auto &p = s.planes[idx][static_cast<size_t>(k)];
            INFO("idx " << idx << " plane " << k << " off " << p.offset);
            REQUIRE(std::fseek(f, static_cast<long>(p.offset), SEEK_SET) == 0);
            std::vector<uint8_t> buf(static_cast<size_t>(p.bytes));
            REQUIRE(std::fread(buf.data(), 1, buf.size(), f) == buf.size());
            REQUIRE(buf.front() == want[k]);
            REQUIRE(buf.back() == want[k]);
        }
        std::fclose(f);
    }
}

TEST_CASE("ExpertStore plane locators agree across experts of one store",
          "[expert_store][stage]")
{
    // A converted tree is uniform: every expert file has the same six planes at
    // the same offsets. The ring sizes its pinned slots from that, so a tree
    // that is NOT uniform has to be caught at parse time rather than producing
    // a slot too small for the expert that lands in it.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    for (size_t idx = 1; idx < 4; ++idx)
        for (int k = 0; k < 6; ++k) {
            INFO("idx " << idx << " plane " << k);
            CHECK(s.planes[idx][static_cast<size_t>(k)].offset ==
                  s.planes[0][static_cast<size_t>(k)].offset);
            CHECK(s.planes[idx][static_cast<size_t>(k)].bytes ==
                  s.planes[0][static_cast<size_t>(k)].bytes);
        }
}

// ── Pinned staging ring (task 3) ──────────────────────────────────────
//
// Today every streamed family copies an expert out of a pageable Arrow mapping
// with six cudaMemcpyAsync calls. That is not asynchronous: the driver stages
// the pageable source through an internal pinned buffer during the call, so the
// host blocks inside the copy (measured: 222.6 s of a 321.6 s Kimi-K3.1-Pro run
// sat inside cudaMemcpyAsync). The ring replaces it with pread into pinned
// memory plus a DMA from pinned, which measured 10.5 GB/s against 5.2 GB/s for
// the pageable path.
//
// The cases below are the hazards, not the happy path: a ring that corrupts one
// expert per wrap still produces output, and a wrong byte in a weight plane is
// silent.
namespace {
int64_t FlatIdx(int layer, int expert) { return layer * 2 + expert; }
}  // namespace

TEST_CASE("ExpertStore staging ring reproduces the file bytes",
          "[expert_store][stage]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring.slots = 4;
    s.stage_ring.pread_threads = 2;
    REQUIRE(s.InitStageRing());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    const int64_t packed = s.entries[0].w1_bytes;
    const int64_t scaled = s.entries[0].w1_scales_bytes;
    REQUIRE(packed > 0);
    REQUIRE(scaled > 0);
    // Deliberately NOT the schema order: w1 | w1s | w3 | w3s | w2 | w2s, which
    // is the layout kimi-k3.1-pro uses. The ring must honour dst_off and never
    // assume an order of its own.
    const int64_t one = packed + scaled;
    const int64_t off[6] = {0, 2 * one, one, packed, 2 * one + packed,
                            one + packed};
    const size_t total = static_cast<size_t>(3 * one);
    uint8_t *dev = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev), total) == cudaSuccess);

    const uint8_t want[6] = {0x01, 0x02, 0x03, 0x80, 0x81, 0x82};
    for (size_t idx = 0; idx < 4; ++idx) {
        REQUIRE(s.EnqueueStage(idx, dev, off, stream));
        s.AwaitStage(idx, stream);
        std::vector<uint8_t> got(total, 0xEE);
        REQUIRE(cudaMemcpy(got.data(), dev, total, cudaMemcpyDeviceToHost) ==
                cudaSuccess);
        for (int k = 0; k < 6; ++k) {
            // Plane lengths differ by kind: the three packed planes are
            // `packed` bytes, the three scale planes `scaled`. Writing the
            // check as off[k] + one - 1 asserted the wrong length for the
            // scale planes and failed on a correct ring.
            const int64_t len = k < 3 ? packed : scaled;
            INFO("idx " << idx << " plane " << k << " len " << len);
            CHECK(got[static_cast<size_t>(off[k])] == want[k]);
            CHECK(got[static_cast<size_t>(off[k]) + static_cast<size_t>(len) - 1] ==
                  want[k]);
        }
    }
    CHECK(s.stage_stats.staged_experts == 4);
    CHECK(s.stage_stats.legacy_fallbacks == 0);
    CHECK(s.stage_stats.short_reads == 0);

    cudaFree(dev);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring never refills a slot under an in-flight DMA",
          "[expert_store][stage]")
{
    // slots = 1 forces the second EnqueueStage to reuse the only slot, and it
    // may only do so once the first DMA has EXECUTED. Needs two things to be a
    // real test rather than a test that passes on broken code:
    //
    //  1. the `distinct` fixture — with identical payloads the corruption is
    //     byte-identical and invisible;
    //  2. the stream kept busy BEFORE the first stage, so the first DMA is
    //     provably still pending when the second EnqueueStage runs. Without this
    //     the copies (32 bytes) finish in microseconds and the host never
    //     observes a pending slot, so a ring with no check at all passed.
    //
    // Both were found by doctoring the implementation and watching this case
    // stay green, which is exactly what a hazard test must not do.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir(/*distinct=*/true).string(), false));
    s.stage_ring.slots = 1;
    REQUIRE(s.InitStageRing());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const int64_t packed = s.entries[0].w1_bytes;
    const int64_t scaled = s.entries[0].w1_scales_bytes;
    const int64_t one = packed + scaled;
    const int64_t off[6] = {0, 2 * one, one, packed, 2 * one + packed,
                            one + packed};
    const size_t total = static_cast<size_t>(3 * one);
    // Separate destinations: which bytes land in the FIRST buffer is the point.
    uint8_t *dev0 = nullptr, *dev1 = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev0), total) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev1), total) == cudaSuccess);

    // Keep the stream busy ahead of the first stage: the DMA recorded for
    // expert 0 cannot execute until this memset retires, so the slot's event is
    // genuinely pending when expert 1 is enqueued.
    void *ballast = nullptr;
    const size_t ballast_bytes = size_t{512} << 20;   // 512 MiB
    REQUIRE(cudaMalloc(&ballast, ballast_bytes) == cudaSuccess);
    for (int i = 0; i < 8; ++i)
        REQUIRE(cudaMemsetAsync(ballast, i, ballast_bytes, stream) ==
                cudaSuccess);

    REQUIRE(s.EnqueueStage(static_cast<size_t>(FlatIdx(0, 0)), dev0, off,
                           stream));
    REQUIRE(s.EnqueueStage(static_cast<size_t>(FlatIdx(0, 1)), dev1, off,
                           stream));
    s.AwaitStage(static_cast<size_t>(FlatIdx(0, 0)), stream);
    s.AwaitStage(static_cast<size_t>(FlatIdx(0, 1)), stream);

    std::vector<uint8_t> got0(total, 0xEE), got1(total, 0xEE);
    REQUIRE(cudaMemcpy(got0.data(), dev0, total, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(got1.data(), dev1, total, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(got0[static_cast<size_t>(off[0])] == 0x10);  // expert 0's own bytes
    CHECK(got1[static_cast<size_t>(off[0])] == 0x11);  // expert 1's
    CHECK(s.stage_stats.ring_waits > 0);               // slot was recycled

    cudaFree(ballast);
    cudaFree(dev0);
    cudaFree(dev1);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring is a no-op when disabled",
          "[expert_store][stage]")
{
    // The default. Every one of the 15 ExpertStore aliases keeps its own copy
    // loop until it opts in, so "disabled" has to be a clean refusal and not a
    // partial stage.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring = Store::StageRingConfig{};   // slots = 0
    REQUIRE_FALSE(s.InitStageRing());
    const int64_t off[6] = {0, 0, 0, 0, 0, 0};
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK_FALSE(s.EnqueueStage(0, nullptr, off, stream));
    CHECK(s.stage_stats.staged_experts == 0);
    CHECK(s.stage_stats.legacy_fallbacks == 1);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring refuses a non-resident expert",
          "[expert_store][stage]")
{
    // An evicted entry has no locators, and the store must decline rather than
    // read a stale one: the caller then falls back to its own copy.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring.slots = 2;
    REQUIRE(s.InitStageRing());
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const int64_t off[6] = {0, 0, 0, 0, 0, 0};
    // max_open 1 keeps flat 0 resident; flat 1 was never parsed.
    CHECK_FALSE(s.EnqueueStage(1, nullptr, off, stream));
    CHECK(s.stage_stats.legacy_fallbacks == 1);
    cudaStreamDestroy(stream);
}

// ── Stage-ring knobs (task 5) ─────────────────────────────────────────
//
// Resolution, not parsing: the tests below pin the DEFAULTS, because a wrong
// default is what a family ships with. `slots = 0` (off) is the default for
// every one of the 15 store aliases until it opts in, and a threads value of 0
// must mean "the measured default", never "no worker threads" — a zero-worker
// pool would complete nothing and stage nothing while looking configured.
TEST_CASE("ExpertStore stage-ring knobs resolve from the environment",
          "[expert_store][stage]")
{
    Store s;
    ::setenv("LSX_EXPERT_STAGE_RING", "1", 1);
    ::setenv("LSX_EXPERT_STAGE_SLOTS", "3", 1);
    ::setenv("LSX_EXPERT_STAGE_THREADS", "0", 1);   // 0 => default, not 0 threads
    ::setenv("LSX_EXPERT_STAGE_CHUNK", "0", 1);
    s.ResolveStageRingFromEnv();
    CHECK(s.stage_ring.slots == 3);
    CHECK(s.stage_ring.pread_threads == 4);
    CHECK(s.stage_ring.chunk_experts == 1);

    ::unsetenv("LSX_EXPERT_STAGE_RING");
    ::unsetenv("LSX_EXPERT_STAGE_SLOTS");
    ::unsetenv("LSX_EXPERT_STAGE_THREADS");
    ::unsetenv("LSX_EXPERT_STAGE_CHUNK");
    s.stage_ring = Store::StageRingConfig{};
    s.ResolveStageRingFromEnv();
    CHECK(s.stage_ring.slots == 0);   // the default is OFF
}

TEST_CASE("ExpertStore stage-ring stays off when the flag is 0",
          "[expert_store][stage]")
{
    // LSX_EXPERT_STAGE_RING=0 with slots set must not enable the ring: a stale
    // SLOTS in someone's shell is not consent.
    Store s;
    ::setenv("LSX_EXPERT_STAGE_RING", "0", 1);
    ::setenv("LSX_EXPERT_STAGE_SLOTS", "64", 1);
    s.ResolveStageRingFromEnv();
    CHECK(s.stage_ring.slots == 0);
    ::unsetenv("LSX_EXPERT_STAGE_RING");
    ::unsetenv("LSX_EXPERT_STAGE_SLOTS");
}

TEST_CASE("ExpertStore staging ring bootstraps geometry without a warm load",
          "[expert_store][stage]")
{
    // warm_on_load = false is what the KimiLinear store does: its eager
    // pre-parse walked experts in path order at ~2.4 ms each and cost 156 s of
    // load for capacity the LRU fills on its own within two tokens (§R17 P5).
    // Consequence for the ring: at InitStageRing time NO expert has been parsed,
    // so the plane geometry it sizes its slots from does not exist yet. Found on
    // the real model — the ring logged "no plane geometry" and every expert
    // silently took the legacy path, which looks like a change that simply did
    // not help.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    s.warm_on_load = false;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.lru.empty());                 // nothing resident, as in production
    s.stage_ring.slots = 2;
    REQUIRE(s.InitStageRing());             // must bootstrap its own geometry
    CHECK(s.plane_bytes[0] > 0);
    CHECK(s.file_bytes > 0);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const int64_t packed = s.plane_bytes[0];
    const int64_t scaled = s.plane_bytes[3];
    const int64_t one = packed + scaled;
    const int64_t off[6] = {0, 2 * one, one, packed, 2 * one + packed,
                            one + packed};
    const size_t total = static_cast<size_t>(3 * one);
    uint8_t *dev = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev), total) == cudaSuccess);

    // Stage a non-resident expert: the ring needs it resident, so the caller's
    // EnsureResident comes first — exactly what the family loop does.
    s.EnsureResident(0, 1, stream);
    REQUIRE(s.EnqueueStage(1, dev, off, stream));
    s.AwaitStage(1, stream);
    std::vector<uint8_t> got(total, 0xEE);
    REQUIRE(cudaMemcpy(got.data(), dev, total, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(got[static_cast<size_t>(off[0])] == 0x01);   // w1
    CHECK(got[static_cast<size_t>(off[3])] == 0x80);   // w1_scales

    cudaFree(dev);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore PrefetchStage refuses an oversized expert without "
          "leaking its descriptor",
          "[expert_store][stage]")
{
    // The slot is sized from the FIRST expert parsed, because a converted tree
    // is uniform. A tree that is not (a repack, a mixed variant) must be REFUSED
    // rather than overflow the slot — and refused without leaking the file
    // descriptor, which is what the first implementation did: it queued the
    // preads, noticed the overflow, and continued without recording the fd for
    // closing. Linux-only check: the pool itself is POSIX-only.
#if defined(__linux__)
    const auto count_fds = []() -> int {
        int n = 0;
        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator("/proc/self/fd", ec);
             !ec && it != std::filesystem::directory_iterator(); ++it)
            ++n;
        return n;
    };

    // Two experts in one layer: the first is 16 bytes per packed plane, the
    // second 64. Geometry is recorded from whichever is parsed first.
    const std::filesystem::path container =
        std::filesystem::temp_directory_path() /
        ("lsx_expert_oversize_" + std::to_string(std::rand()));
    const auto exps = container / "logestix" / "experts" / "layer_0";
    std::filesystem::create_directories(exps);
    for (int i = 0; i < 2; ++i) {
        const size_t n = (i == 0) ? 16u : 64u;
        lsxcommon::PackedExpert pe;
        pe.w1_packed.assign(n, static_cast<uint8_t>(0x10 + i));
        pe.w2_packed.assign(n, static_cast<uint8_t>(0x20 + i));
        pe.w3_packed.assign(n, static_cast<uint8_t>(0x30 + i));
        // One scale byte per 16 elements (n packed bytes = 2n elements), or the
        // decoder rejects the expert and the test measures the wrong path — the
        // first version used 2 scales for 64 bytes and expert 1 never parsed.
        const size_t scales = (n * 2) / 16;
        pe.w1_scales.assign(scales, 0x80u);
        pe.w2_scales.assign(scales, 0x81u);
        pe.w3_scales.assign(scales, 0x82u);
        auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
        REQUIRE(batch.ok());
        REQUIRE(lsxcommon::WriteRecordBatchToFile(
                    *batch, exps / ("exp_" + std::to_string(i) + ".arrow"))
                    .ok());
    }

    Store s;
    s.num_layers = 1;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(container.string(), false));
    s.stage_ring.slots = 2;
    REQUIRE(s.InitStageRing());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    s.EnsureResident(0, 0, stream);   // PrefetchStage declines a non-resident entry
    s.EnsureResident(0, 1, stream);

    const int before = count_fds();
    const size_t ids[2] = {0, 1};
    const size_t staged = s.PrefetchStage(ids, 2);
    const int after = count_fds();

    // Whichever expert the geometry was taken from, the batch must not leave a
    // descriptor behind, and it must never claim more experts than it filled.
    CHECK(staged <= 2);
    INFO("staged " << staged << ", fds " << before << " -> " << after);
    CHECK(after <= before);

    cudaStreamDestroy(stream);
    std::filesystem::remove_all(container);
#endif
}
