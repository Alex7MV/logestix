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

std::filesystem::path MakeExpertDir()
{
    std::filesystem::path container =
        std::filesystem::temp_directory_path() /
        ("lsx_expert_test_" + std::to_string(std::rand()));
    const std::filesystem::path exps = container / "logestix" / "experts";
    std::filesystem::create_directories(exps / "layer_0");
    std::filesystem::create_directories(exps / "layer_1");

    lsxcommon::PackedExpert pe;
    pe.w1_packed.assign(16, 0x01u);
    pe.w2_packed.assign(16, 0x02u);
    pe.w3_packed.assign(16, 0x03u);
    pe.w1_scales.assign(2, 0x80u);
    pe.w2_scales.assign(2, 0x81u);
    pe.w3_scales.assign(2, 0x82u);
    pe.w1_global_scale = 1.0f;
    pe.w2_global_scale = 2.0f;
    pe.w3_global_scale = 3.0f;
    auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
    REQUIRE(batch.ok());
    for (int layer = 0; layer < 2; ++layer)
        for (int exp = 0; exp < 2; ++exp)
            REQUIRE(lsxcommon::WriteRecordBatchToFile(
                        *batch, exps / ("layer_" + std::to_string(layer)) /
                                    ("exp_" + std::to_string(exp) + ".arrow"))
                        .ok());
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
