#include <catch2/catch_test_macros.hpp>

#include "test_ornith_moe_fixture.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/ornith_moe_convert.h"
#include "ornith_moe_model.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace lsxcommon;

namespace {

struct ConvertedModel {
    std::filesystem::path in, out;
    ConvertedModel()
    {
        in = std::filesystem::temp_directory_path() / "lsx_ornith_model_in";
        out = std::filesystem::temp_directory_path() / "lsx_ornith_model_out";
        std::filesystem::remove_all(in);
        std::filesystem::remove_all(out);
        ornithtest::BuildSyntheticOrnithCheckpoint(in);
        ornithtest::OrnithSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
        auto st = ornith::ConvertOrnithMoeModel(in.string(), out.string());
        if (!st.ok()) FAIL("fixture conversion failed: " + st.ToString());
    }
    ~ConvertedModel()
    {
        std::error_code ec;
        std::filesystem::remove_all(in, ec);
        std::filesystem::remove_all(out, ec);
    }
};

} // namespace

TEST_CASE("factory resolves ornith-1.5-35b-a3b", "[ornith-moe-model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = OrnithMoeModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == OrnithMoeModel::kType);
}

TEST_CASE("fast_math toggle propagates from Ornith construction",
          "[ornith-moe-model]")
{
    EnsureModelRegistry();
    ModelInitConfig on;
    on.model_type = OrnithMoeModel::kType;
    on.fast_math = true;
    auto m_on = LsxModelFactory::Create(on);
    REQUIRE(m_on != nullptr);
    CHECK(lsx_nvfp4::LsxTcFastMath());

    ModelInitConfig off;
    off.model_type = OrnithMoeModel::kType;
    off.fast_math = false;
    off.fast_math_mode = FastMathMode::kOff;
    auto m_off = LsxModelFactory::Create(off);
    REQUIRE(m_off != nullptr);
    CHECK_FALSE(lsx_nvfp4::LsxTcFastMath());
}

TEST_CASE("ornith geometry mismatch rejected at load", "[ornith-moe-model]")
{
    EnsureModelRegistry();
    ConvertedModel fix;
    auto idx = fix.out / "model.index.json";
    {
        std::ifstream f(idx);
        std::stringstream ss; ss << f.rdbuf();
        std::string j = ss.str();
        auto pos = j.find("\"hidden_size\": 64");
        REQUIRE(pos != std::string::npos);
        j.replace(pos, strlen("\"hidden_size\": 64"), "\"hidden_size\": 63");
        std::ofstream g(idx, std::ios::trunc);
        g << j;
    }
    ornithtest::OrnithUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    ModelInitConfig cfg;
    cfg.model_type = OrnithMoeModel::kType;
    cfg.model_path = fix.out.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
    ornithtest::OrnithSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
}

TEST_CASE("ornith eos truncation helper", "[ornith-moe-decode]")
{
    std::vector<int32_t> t{5, 6, 7, 8};
    CHECK(moe::MoETruncateTokensAtEos(t, 7) == true);
    CHECK((t == std::vector<int32_t>{5, 6, 7}));
    std::vector<int32_t> w{1};
    CHECK(moe::MoETruncateTokensAtEos(w, -1) == false);
}
