// DeepSeek-V4-Pro-0813 model lifecycle stage (host-only).
//
// Classification + topology + frozen-passport contract of the converted
// layout, on the synthetic Pro fixture:
//   - detect_model_type / LsxModelFactory::DetectType (index normalization)
//   - load_topology (layer/expert counts + VRAM profile + error paths)
//   - Model::Load rejects a stale format tag
//   - the Pro passport values themselves (stage 0 triage is machine-pinned)
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "deepseek_moe_pro_constants.h"
#include "deepseek_moe_pro_load.cuh"
#include "lsxcommon/model_factory.h"

#include "test_deepseek_moe_pro_fixture.h"

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

TEST_CASE("deepseek-v4-pro detect_model_type normalizes the index",
          "[deepseek][pro][model]")
{
    const auto dir = TempDir("lsx_ds_pro_model_detect");
    REQUIRE(dsprofixtest::BuildSyntheticModel(dir).ok());
    CHECK(lsxcommon::deepseek_pro::gpu::detect_model_type(dir.string()) ==
          "deepseek-v4-pro-0813");
    CHECK(lsxcommon::LsxModelFactory::DetectType(dir.string()) ==
          "deepseek-v4-pro-0813");
    std::filesystem::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro detect_model_type returns empty for a broken index",
          "[deepseek][pro][model]")
{
    const auto dir = TempDir("lsx_ds_pro_model_detect_broken");
    WriteIndex(dir, "{not json");
    CHECK(
        lsxcommon::deepseek_pro::gpu::detect_model_type(dir.string()).empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro load_topology parses the converted layout",
          "[deepseek][pro][model]")
{
    const auto dir = TempDir("lsx_ds_pro_model_topo");
    REQUIRE(dsprofixtest::BuildSyntheticModel(dir).ok());
    const auto topo =
        lsxcommon::deepseek_pro::gpu::load_topology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.num_layers == dsprofixtest::kNumLayers);
    CHECK(topo.experts_per_layer == 2);
    CHECK(topo.total_experts ==
          static_cast<int64_t>(dsprofixtest::kNumLayers) * 2);
    CHECK(topo.vram.min_bytes == 1024);
    CHECK(topo.vram.optimal_bytes == 2048);
    CHECK(topo.vram.max_bytes == 4096);
    CHECK(topo.format == "logestix-v1-nvfp4");
    std::filesystem::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro load_topology rejects a missing vram block",
          "[deepseek][pro][model]")
{
    const auto dir = TempDir("lsx_ds_pro_model_novram");
    WriteIndex(dir,
               "{\"model\":\"DeepSeek-V4-Pro-0813\",\"num_layers\":61,"
               "\"experts_per_layer\":384,\"total_experts\":23424}");
    const auto topo =
        lsxcommon::deepseek_pro::gpu::load_topology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro model Load rejects the outdated format tag",
          "[deepseek][pro][model]")
{
    const auto dir = TempDir("lsx_ds_pro_model_v1_reject");
    REQUIRE(dsprofixtest::BuildSyntheticModel(dir).ok());
    {
        std::ofstream f(dir / "logestix" / "model.index.json");
        f << "{\"format\": \"logestix-v1\", \"model\": "
             "\"DeepSeek-V4-Pro-0813\", \"num_layers\": 3,"
             " \"experts_per_layer\": 2, \"total_experts\": 6,"
             " \"vram\": {\"min_bytes\": 1, \"optimal_bytes\": 2,"
             " \"max_bytes\": 4, \"context_tokens\": 8}}\n";
    }
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = dir.string();
    lsxcommon::EnsureModelRegistry();
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model);
    CHECK_FALSE(model->Load());
    std::filesystem::remove_all(dir);
}

TEST_CASE("deepseek-v4-pro passport matches the source checkpoint",
          "[deepseek][pro][model][passport]")
{
    using namespace lsxcommon::deepseek_pro;
    // config.json / inference/config.json of DeepSeek-V4-Pro-0813.
    CHECK(kLayers == 61);
    CHECK(kHidden == 7168);
    CHECK(kHeads == 128);
    CHECK(kHeadDim == 512);
    CHECK(kQLoraRank == 1536);
    CHECK(kOLoraRank == 1024);
    CHECK(kOGroups == 16);
    CHECK(kMoeIntermediate == 3072);
    CHECK(kNumRouted == 384);
    CHECK(kTopK == 6);
    CHECK(kVocab == 129280);
    CHECK(kIndexNHeads == 64);
    CHECK(kIndexHeadDim == 128);
    CHECK(kIndexTopk == 1024);
    CHECK(kHashLayers == 3);
    CHECK(kMtpLayers == 3);
    // Compression grid: 128,128, then 4/128 alternating.
    CHECK(CompressRatio(0) == 128);
    CHECK(CompressRatio(1) == 128);
    CHECK(CompressRatio(2) == 4);
    CHECK(CompressRatio(3) == 128);
    CHECK(CompressRatio(59) == 128);
    CHECK(CompressRatio(60) == 4);
    for (int l = 0; l < kLayers; ++l)
        CHECK(HasIndexer(l) == (CompressRatio(l) == 4));
}
