// TDD — DeepSeek-V4-Flash model lifecycle stage (host-only).
//
// The suite guard (test_model_suite_guard.cpp) requires an explicit model
// stage per suite. DeepSeek-V4-Flash has no converter in the repo: it ingests
// a pre-converted logestix layout, so the model stage covers classification
// and the topology contract of that layout on the synthetic fixture:
//   - detect_model_type / LsxModelFactory::DetectType (index normalization)
//   - load_topology (layer/expert counts + VRAM profile + error paths)
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "deepseek_moe_load.cuh"
#include "lsxcommon/model_factory.h"

#include "test_deepseek_moe_fixture.h"

namespace {

std::filesystem::path TempDir(const std::string &name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

void WriteIndex(const std::filesystem::path &dir, const std::string &json)
{
    std::filesystem::create_directories(dir / "logestix");
    std::ofstream f(dir / "logestix" / "model.index.json");
    f << json;
}

} // namespace

TEST_CASE("detect_model_type normalizes the index model field",
          "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_detect");
    REQUIRE(dsfixtest::BuildSyntheticModel(dir).ok());
    CHECK(lsxcommon::gpu::detect_model_type(dir.string()) ==
          "deepseek-v4-flash");
    std::filesystem::remove_all(dir);
}

TEST_CASE("detect_model_type returns empty for broken index",
          "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_detect_broken");
    WriteIndex(dir, "{not json");
    CHECK(lsxcommon::gpu::detect_model_type(dir.string()).empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("load_topology parses the synthetic converted layout",
          "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_topo");
    REQUIRE(dsfixtest::BuildSyntheticModel(dir).ok());
    const auto topo = lsxcommon::gpu::load_topology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.num_layers == dsfixtest::kNumLayers);
    CHECK(topo.experts_per_layer == 2);
    CHECK(topo.total_experts ==
          static_cast<int64_t>(dsfixtest::kNumLayers) * 2);
    CHECK(topo.vram.min_bytes == 1024);
    CHECK(topo.vram.optimal_bytes == 2048);
    CHECK(topo.vram.max_bytes == 4096);
    std::filesystem::remove_all(dir);
}

TEST_CASE("load_topology rejects broken index JSON", "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_topo_broken");
    WriteIndex(dir, "]]] not an object");
    const auto topo = lsxcommon::gpu::load_topology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("load_topology reports the index format", "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_topo_format");
    REQUIRE(dsfixtest::BuildSyntheticModel(dir).ok());
    const auto topo = lsxcommon::gpu::load_topology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.format == "logestix-v1-nvfp4");
    std::filesystem::remove_all(dir);
}

TEST_CASE("model load rejects the outdated logestix-v1 format",
          "[deepseek-model]")
{
    namespace fs = std::filesystem;
    const auto dir = TempDir("lsx_ds_model_v1_reject");
    REQUIRE(dsfixtest::BuildSyntheticModel(dir).ok());
    // Rewind the fixture index to the pre-NVFP4 tag.
    {
        std::ofstream f(dir / "logestix" / "model.index.json");
        f << "{\"format\": \"logestix-v1\", \"model\": \"DeepSeek-V4-Flash\","
             " \"num_layers\": 3, \"experts_per_layer\": 2,"
             " \"total_experts\": 6, \"vram\": {\"min_bytes\": 1,"
             " \"optimal_bytes\": 2, \"max_bytes\": 4,"
             " \"context_tokens\": 8}}\n";
    }
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = dir.string();
    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model);
    CHECK_FALSE(model->Load());
    std::filesystem::remove_all(dir);
}

TEST_CASE("load_topology rejects missing vram block", "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_topo_novram");
    WriteIndex(dir,
               "{\"model\":\"DeepSeek-V4-Flash\",\"num_layers\":3,"
               "\"experts_per_layer\":2,\"total_experts\":6}");
    const auto topo = lsxcommon::gpu::load_topology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("factory classifies the synthetic deepseek dir", "[deepseek-model]")
{
    const auto dir = TempDir("lsx_ds_model_factory");
    REQUIRE(dsfixtest::BuildSyntheticModel(dir).ok());
    CHECK(lsxcommon::LsxModelFactory::DetectType(dir.string()) ==
          "deepseek-v4-flash");
    std::filesystem::remove_all(dir);
}
