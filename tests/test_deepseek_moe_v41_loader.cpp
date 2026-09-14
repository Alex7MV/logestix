// tests/test_deepseek_moe_v41_loader.cpp — DeepSeek-V4.1-Flash load-stage
// (host-only): the logestix-v41-nvfp4 index contract. Synthetic fixtures only
// (suite guard); real weights live in the selfcheck stage.
#include <catch2/catch_test_macros.hpp>

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
    CHECK(topo.format == lsxcommon::deepseek_v41::kIndexFormat);
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
                                            /*format=*/"logestix-v1-nvfp4")
                .ok());
    const auto topo = lsxcommon::deepseek_v41::LoadTopology(dir.string());
    CHECK_FALSE(topo.error_msg.empty());  // format gate at parse time
    std::string err;
    CHECK_FALSE(lsxcommon::deepseek_v41::ValidateGeometry(topo, &err));
    CHECK_FALSE(err.empty());
    fs::remove_all(dir);
}
