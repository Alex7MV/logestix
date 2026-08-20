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