// TDD — MiMo layout loader (Model Ingestion Protocol step 3). Synthetic
// converted fixture only (no vault). Label inference.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <cstring>
#include <fstream>

#include <lsxcommon/mimo_moe_convert.h>
#include <lsxcommon/mimo_moe_load.h>

#include "test_mimo_moe_fixture.h"
#include "mimo_moe_constants.h"

namespace fs = std::filesystem;

namespace {

fs::path BuildConverted()
{
    const fs::path root = fs::temp_directory_path() / "lsx_mimo_loader";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path in = root / "src";
    const fs::path out = root / "dst";
    mimotest::BuildSyntheticMiMoCheckpoint(in);
    REQUIRE(lsxcommon::mimo::ConvertMiMoModel(in.string(), out.string()).ok());
    return out;
}

} // namespace

TEST_CASE("mimo loader resolves the logestix-mimo-nvfp4 layout",
          "[mimo][loader]")
{
    const fs::path out = BuildConverted();
    auto layout = lsxcommon::mimo::LoadMiMoLayout(out.string());
    INFO(layout.status().ToString());
    REQUIRE(layout.ok());
    CHECK(layout->num_layers == 2);
    CHECK(layout->layer_raw.size() == 2);
    CHECK(layout->layer_nvfp4.size() == 2);
    CHECK(fs::exists(layout->global_file));
    CHECK(fs::exists(layout->experts_root));
    fs::remove_all(out.parent_path());
}

TEST_CASE("mimo loader rejects a foreign format tag", "[mimo][loader]")
{
    const fs::path out = BuildConverted();
    const fs::path idx = out / "logestix" / "model.index.json";
    auto text = mimotest::ReadFileText(idx);
    const auto pos = text.find("logestix-mimo-nvfp4");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, std::strlen("logestix-mimo-nvfp4"), "logestix-other");
    std::ofstream(idx) << text;
    auto layout = lsxcommon::mimo::LoadMiMoLayout(out.string());
    CHECK_FALSE(layout.ok());
    CHECK(layout.status().ToString().find("format") != std::string::npos);
    fs::remove_all(out.parent_path());
}

TEST_CASE("mimo frozen geometry rejects a mismatch", "[mimo][loader]")
{
    lsxcommon::mimo::MiMoTopology t;
    using namespace lsxcommon::mimo::frozen;
    t.hidden = kHidden; t.layers = kLayers; t.vocab = kVocab;
    t.heads = kHeads; t.kv_heads_ga = kKvHeadsGa; t.kv_heads_swa = kKvHeadsSwa;
    t.head_dim = kQkHeadDim; t.v_head_dim = kVHeadDim;
    t.num_experts = kNumRouted; t.top_k = kTopK;
    t.moe_intermediate = kMoeIntermediate; t.dense_intermediate = kDenseIntermediate;
    REQUIRE(lsxcommon::mimo::ValidateMiMoGeometry(t).ok());
    t.num_experts = 2;
    auto st = lsxcommon::mimo::ValidateMiMoGeometry(t);
    CHECK_FALSE(st.ok());
    CHECK(st.ToString().find("num_experts") != std::string::npos);
}
