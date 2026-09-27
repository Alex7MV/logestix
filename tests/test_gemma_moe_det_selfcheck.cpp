// DETSelfCheck — gemma-4-26b-a4b-it: verifies the converted model reproduces the
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

constexpr const char *kModelType = "gemma-4-26b-a4b-it";

std::filesystem::path GemmaModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_GEMMA_MODEL_PATH"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "gemma-4-26B-A4B-it";
}

std::filesystem::path GemmaGoldenPath()
{
    if (const char *env = std::getenv("LSX_DET_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_DET_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("gemma-4-26b-a4b-it DET SelfCheck: golden answers match the converted "
          "model", "[det][real-model]")
{
    const std::filesystem::path root = GemmaModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("gemma-4-26b-a4b-it converted model directory not present");
    const std::filesystem::path golden_path = GemmaGoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("DET golden not captured for gemma-4-26b-a4b-it");

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
