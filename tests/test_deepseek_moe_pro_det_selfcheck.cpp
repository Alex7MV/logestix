// DETSelfCheck — deepseek-v4-pro-0813: verifies the converted model reproduces the
// golden answers captured from the source safetensors model. Separate
// functionality from SelfCheck; SKIPs without converted weights or golden.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include <lsxcommon/det_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#ifndef LSX_DET_GOLDEN_DIR
#define LSX_DET_GOLDEN_DIR "tests/fixtures/det"
#endif

namespace {

constexpr const char *kModelType = "deepseek-v4-pro-0813";

std::filesystem::path DeepseekModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_DEEPSEEK_PRO_MODEL_PATH"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "DeepSeek-V4-Pro-0813";
}

std::filesystem::path DeepseekGoldenPath()
{
    if (const char *env = std::getenv("LSX_DET_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_DET_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("deepseek-v4-pro-0813 DET SelfCheck: golden answers match the converted "
          "model", "[det][real-model]")
{
    const std::filesystem::path root = DeepseekModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("deepseek-v4-pro-0813 converted model directory not present");
    const std::filesystem::path golden_path = DeepseekGoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("DET golden not captured for deepseek-v4-pro-0813");

    lsxcommon::det::DetGolden golden;
    std::string err;
    REQUIRE(lsxcommon::det::LoadDetGoldenFile(golden_path.string(), golden,
                                              &err));
    REQUIRE(golden.model_type == kModelType);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == kModelType);
    REQUIRE(model->Load());

    const lsxcommon::det::DetResult result =
        lsxcommon::det::RunDetSelfCheck(*model, golden);
    for (const lsxcommon::det::DetCaseResult &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected
                     << "' extracted='" << c.extracted << "' raw='" << c.raw
                     << "' error='" << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
