// TDD — Nemotron-3.5 converter (Model Ingestion Protocol step 1). Synthetic
// checkpoint only: the real model is never converted in a test. The ctest
// TIMEOUT is the mechanical guard against a real full conversion.
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_fixture.h"

#include "lsxcommon/nemotron_moe_convert.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_moe;

TEST_CASE("nemotron converter emits the logestix-nemotron-nvfp4 layout",
          "[nemotron][convert]")
{
    const auto src = nemotrontest::MakeTempDir("nm_src");
    const auto dst = nemotrontest::MakeTempDir("nm_dst");
    nemotrontest::WriteNemotronCheckpoint(src);

    auto st = ConvertNemotronModel(src.string(), dst.string());
    REQUIRE(st.ok());

    std::ifstream in(dst / "model.index.json");
    REQUIRE(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();

    CHECK(json.find("\"layout\": 2") != std::string::npos);
    CHECK(json.find("\"attn_layer_idx\": [2]") != std::string::npos);
    CHECK(json.find("\"experts_no_gate\": true") != std::string::npos);
    CHECK(json.find("\"deferred\": {\"mtp\": \"cut\"") != std::string::npos);
    CHECK(json.find("\"mtp_tensors\": 1") != std::string::npos);

    // Per-expert files for the single MoE layer (layer 1), w1 unused.
    for (int e = 0; e < nemotrontest::kTExperts; ++e) {
        CHECK(fs::exists(dst / "experts" / "layer_1" /
                         ("exp_" + std::to_string(e) + ".arrow")));
    }
    CHECK(fs::exists(dst / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_0.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_1_nvfp4.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_2_nvfp4.arrow"));
}

TEST_CASE("nemotron classifier tolerates malformed layer/expert indices",
          "[nemotron][convert]")
{
    CHECK_NOTHROW(classify_nemotron_tensor("backbone.layers..mixer.norm.weight"));
    CHECK_NOTHROW(classify_nemotron_tensor(
        "backbone.layers.0.mixer.experts..up_proj.weight"));
    CHECK(classify_nemotron_tensor("backbone.layers..mixer.norm.weight").kind ==
          NemotronTensorKind::Unknown);
    CHECK(classify_nemotron_tensor(
              "backbone.layers.0.mixer.experts..up_proj.weight").kind ==
          NemotronTensorKind::Unknown);
}

TEST_CASE("nemotron converter rejects an unrecognized tensor",
          "[nemotron][convert]")
{
    const auto src = nemotrontest::MakeTempDir("nm_bad_src");
    const auto dst = nemotrontest::MakeTempDir("nm_bad_dst");
    nemotrontest::WriteNemotronCheckpoint(src);

    // Append a bogus tensor to the shard inventory by re-writing the file.
    std::vector<synthtest::FakeTensor> extra;
    extra.push_back({"backbone.layers.0.mixer.mystery.weight", "BF16", {2, 2},
                     synthtest::MakeBf16(4)});
    synthtest::WriteSafetensors(src, "bogus.safetensors", extra);

    auto st = ConvertNemotronModel(src.string(), dst.string());
    CHECK_FALSE(st.ok());
}
