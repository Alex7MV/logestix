#include <catch2/catch_test_macros.hpp>

#include "qwen_moe_test_fixture.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "qwen_3_5_moe_model.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace lsxcommon;

namespace {

// Builds the synthetic checkpoint, converts it and returns the output dir.
struct ConvertedModel {
    std::filesystem::path in;
    std::filesystem::path out;

    ConvertedModel()
    {
        in = std::filesystem::temp_directory_path() / "lsx_qwen_moe_model_in";
        out = std::filesystem::temp_directory_path() / "lsx_qwen_moe_model_out";
        std::filesystem::remove_all(in);
        std::filesystem::remove_all(out);
        qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
        setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
        auto st = qwen_moe::ConvertQwen35MoeModel(in.string(), out.string());
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

TEST_CASE("factory resolves qwen3.5-35b-a3b", "[qwen-moe-model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = Qwen35MoeModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    // No weights behind an empty path: creation must still resolve the type.
    REQUIRE(model != nullptr);
    CHECK(model->Type() == Qwen35MoeModel::kType);
}

TEST_CASE("geometry mismatch rejected at load", "[qwen-moe-model]")
{
    EnsureModelRegistry();
    ConvertedModel fix;

    // Corrupt hidden_size in the index: ValidateGeometry must refuse.
    auto idx = fix.out / "logestix-qwen35moe" / "model.index.json";
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

    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "", 1);   // hard gate ON
    unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    ModelInitConfig cfg;
    cfg.model_type = Qwen35MoeModel::kType;
    cfg.model_path = fix.out.string();
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());

    // Synthetic escape hatch re-enabled for other tests in this binary.
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
}
