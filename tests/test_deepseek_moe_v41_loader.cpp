// tests/test_deepseek_moe_v41_loader.cpp — DeepSeek-V4.1-Flash load-stage
// (host-only): the logestix-v41-nvfp4 index contract. Synthetic fixtures only
// (suite guard); real weights live in the selfcheck stage.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"

#include <filesystem>

#include "deepseek_moe_v41_load.h"

#include "test_deepseek_moe_v41_fixture.h"

namespace fs = std::filesystem;

TEST_CASE("deepseek-v41 load_topology parses the synthetic index",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndex(dir).ok());

    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK(topo.error_msg.empty());
    CHECK(topo.layout == lsxcommon::kIndexLayout);
    CHECK(topo.model == "DeepSeek-V4.1-Flash");
    CHECK(topo.num_layers == lsxcommon::deepseek_v41::kLayers);
    CHECK(topo.experts_per_layer == lsxcommon::deepseek_v41::kNumRouted);
    CHECK(topo.total_experts ==
          static_cast<int64_t>(lsxcommon::deepseek_v41::kLayers) *
              lsxcommon::deepseek_v41::kNumRouted);
    CHECK(topo.vram.min_bytes == 1024);
    CHECK(topo.vram.optimal_bytes == 2048);
    CHECK(topo.vram.max_bytes == 4096);

    std::string err;
    CHECK(lsxcommon::deepseek_v41::ValidateGeometry(topo, &err));
    CHECK(err.empty());
    CHECK(lsxcommon::deepseek_v41::IsSupportedFormat(topo));
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 load_topology rejects broken JSON",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo_broken";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndexRaw(dir, "]]] not an object")
                .ok());
    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());
    CHECK(topo.num_layers == 0);
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 topology rejects foreign geometry and format",
          "[deepseek-v41][load]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_topo_foreign";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndex(dir, /*num_layers=*/43,
                                            /*experts=*/256,
                                            /*layout=*/1)
                .ok());
    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());  // format gate at parse time
    std::string err;
    CHECK_FALSE(lsxcommon::deepseek_v41::ValidateGeometry(topo, &err));
    CHECK_FALSE(err.empty());
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 card name resolves each sibling tree to its own model",
          "[deepseek-v41][load][sibling]")
{
    // Two checkpoints of this contour share one architecture, so the card name
    // in the index is the ONLY thing that tells the trees apart. It must fold
    // (exactly as gpu::detect_model_type folds it) to that checkpoint's kType:
    // a 1017 tree resolving to the parent's kType loads the parent's class and
    // silently compares two models' outputs.
    using namespace lsxcommon::deepseek_v41;
    CHECK(NormalizeCardName("DeepSeek-V4.1-Flash") == "deepseek-v4.1-flash");
    CHECK(NormalizeCardName("DeepSeek-V4.1-Flash-1017") ==
          "deepseek-v4.1-flash-1017");

    const auto dir = fs::temp_directory_path() / "lsx_v41_topo_card";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticIndex(dir).ok());
    const auto topo = LoadTopology(dir.string());
    CHECK(MatchesModelType(topo, "deepseek-v4.1-flash"));
    CHECK_FALSE(MatchesModelType(topo, "deepseek-v4.1-flash-1017"));

    // An index that names no model cannot be disproved here; the format and
    // frozen-geometry gates still run.
    ModelTopology anonymous = topo;
    anonymous.model.clear();
    CHECK(MatchesModelType(anonymous, "deepseek-v4.1-flash-1017"));
    fs::remove_all(dir);
}
