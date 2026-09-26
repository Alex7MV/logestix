// FSISelfCheck — glm-5.3-flash: verifies the converted model reproduces the
// golden answers captured from the source safetensors model. Separate
// functionality from SelfCheck; SKIPs without converted weights or golden.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include <lsxcommon/fsi_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#ifndef LSX_FSI_GOLDEN_DIR
#define LSX_FSI_GOLDEN_DIR "tests/fixtures/fsi"
#endif

namespace {

constexpr const char *kModelType = "glm-5.3-flash";

std::filesystem::path GlmFlashModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_GLM_FLASH_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "GLM-5.3-Flash";
}

std::filesystem::path GlmFlashGoldenPath()
{
    if (const char *env = std::getenv("LSX_FSI_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_FSI_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("glm-5.3-flash FSI SelfCheck: golden answers match the converted "
          "model", "[fsi][real-model]")
{
    const std::filesystem::path root = GlmFlashModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("glm-5.3-flash converted model directory not present");
    const std::filesystem::path golden_path = GlmFlashGoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("FSI golden not captured for glm-5.3-flash");

    lsxcommon::fsi::FsiGolden golden;
    std::string err;
    REQUIRE(lsxcommon::fsi::LoadFsiGoldenFile(golden_path.string(), golden,
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

    const lsxcommon::fsi::FsiResult result =
        lsxcommon::fsi::RunFsiSelfCheck(*model, golden);
    for (const lsxcommon::fsi::FsiCaseResult &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected
                     << "' extracted='" << c.extracted << "' raw='" << c.raw
                     << "' error='" << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
