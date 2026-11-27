// Hy-SimulMT-30B-A3B converter stage (Model Ingestion Protocol stage 1/2):
// the hy_v3 text backbone with its layout deltas (flat rope_theta, no
// `mlp_layer_types`, real `moe_intermediate_size`) and the speech encoder
// stripped + counted. Synthetic checkpoint, label `convert`, bounded TIMEOUT.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hy_moe_convert.h"
#include "lsxcommon/hy_moe_simulmt_convert.h"
#include "test_hy_moe_simulmt_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon;
namespace fs = std::filesystem;

namespace {

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct SyntheticEnv {
    std::string prev;
    bool had = false;
    SyntheticEnv()
    {
        const char *p = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
        had = p != nullptr;
        if (had) prev = p;
        setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    }
    ~SyntheticEnv()
    {
        if (had) setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", prev.c_str(), 1);
        else unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    }
};

} // namespace

TEST_CASE("Hy-SimulMT config: flat rope_theta and first_k_dense_replace dispatch",
          "[hy-moe-simulmt-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_hy_simulmt_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    hy_simulmt_test::WriteConfig(dir);

    auto top = hy::ParseHyConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = hy_simulmt_test::TestCfg();
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    CHECK(top.rope_theta == Catch::Approx(11158840.0));
    CHECK(top.moe_intermediate_size == c.moe_inter);
    CHECK(top.shared_intermediate_size == c.shared_inter);
    CHECK(top.num_experts == c.experts);
    REQUIRE(top.mlp_types.size() == static_cast<size_t>(c.layers));
    CHECK(top.mlp_types[0] == "dense");
    CHECK(top.mlp_types[1] == "sparse");
    CHECK(top.bos_token_id == 3);
    CHECK(top.eos_token_id == 4);
    // The generation_config's array adds a second stop id (63 in the fixture).
    REQUIRE(top.eos_ids.size() == 2);
    CHECK(top.eos_ids[0] == 4);
    CHECK(top.eos_ids[1] == 63);
    std::error_code ec;
    fs::remove_all(dir, ec);
}

TEST_CASE("Hy-SimulMT converter strips and counts the speech encoder",
          "[hy-moe-simulmt-convert]")
{
    SyntheticEnv env;
    fs::path in = fs::temp_directory_path() / "lsx_hy_simulmt_in";
    fs::path out = fs::temp_directory_path() / "lsx_hy_simulmt_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    REQUIRE(hy_simulmt_test::BuildSyntheticCheckpoint(in) > 0);

    auto st = hy_simulmt::ConvertHySimulMTModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const auto &c = hy_simulmt_test::TestCfg();

    // 1) The index carries the SimulMT identity and the ballast count.
    {
        REQUIRE(fs::exists(base / "model.index.json"));
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"logestix-hy-simulmt-nvfp4\"") != std::string::npos);
        CHECK(j.find("\"model_type\": \"hy-simulmt-30b-a3b\"") !=
              std::string::npos);
        CHECK(j.find("\"model\": \"Hy-SimulMT-30B-A3B\"") != std::string::npos);
        CHECK(j.find("\"skipped_visual_tensors\": " +
                     std::to_string(hy_simulmt_test::kAudioTensorCount)) !=
              std::string::npos);
        CHECK(j.find("\"hidden_size\": " + std::to_string(c.hidden)) !=
              std::string::npos);
        CHECK(j.find("\"first_moe_layer\": 1") != std::string::npos);
        CHECK(j.find("\"eos_token_ids\": [4,63]") != std::string::npos);
        CHECK(j.find("\"total_experts\": " +
                     std::to_string(c.experts * (c.layers - c.first_moe_layer))) !=
              std::string::npos);
    }

    // 2) No audio tensor reaches ANY converted arrow file (it is ballast).
    {
        for (const auto &entry : fs::recursive_directory_iterator(base)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".arrow") continue;
            for (const auto &n : synthtest::TensorNamesIn(entry.path())) {
                INFO("ballast leaked into " << entry.path().string() << ": " << n);
                CHECK(n.find("audio_tower.") == std::string::npos);
            }
        }
    }

    // 3) The text graph is exactly the parent's layout.
    CHECK(fs::exists(base / "pinned" / "global.arrow"));
    CHECK(fs::exists(base / "pinned" / "layer_0_nvfp4.arrow"));
    CHECK(fs::exists(base / "shared" / "layer_1_nvfp4.arrow"));
    for (int l = 0; l < c.layers; ++l)
        for (int e = 0; e < c.experts; ++e) {
            const bool want = l >= c.first_moe_layer;
            CHECK(fs::exists(base / "experts" / ("layer_" + std::to_string(l)) /
                             ("exp_" + std::to_string(e) + ".arrow")) == want);
        }

    // 4) The parent's spec (no ballast policy) rejects the same tree: the
    // audio tensors are only skipped because SimulMT asks for it.
    {
        fs::path other = fs::temp_directory_path() / "lsx_hy_simulmt_out_hy";
        fs::remove_all(other);
        auto r = hy::ConvertHyMoeModel(in.string(), other.string());
        CHECK_FALSE(r.ok());
        CHECK(r.ToString().find("audio_tower.") != std::string::npos);
        std::error_code ec;
        fs::remove_all(other, ec);
    }

    // 5) Determinism: a second conversion writes a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_hy_simulmt_out2";
        fs::remove_all(out2);
        REQUIRE(hy_simulmt::ConvertHySimulMTModel(in.string(), out2.string()).ok());
        CHECK(ReadAll(base / "model.index.json") ==
              ReadAll(out2 / "model.index.json"));
        std::error_code ec;
        fs::remove_all(out2, ec);
    }

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
}
