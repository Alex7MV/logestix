// CODESelfCheck — nemotron-3.5-lightning-30b-a3b-nvfp4: verifies the converted
// model reproduces the golden compiler-syntax answers captured from the source
// safetensors model. Separate functionality from SelfCheck and FSISelfCheck;
// SKIPs without converted weights or golden.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

#include <lsxcommon/code_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#ifndef LSX_CODE_GOLDEN_DIR
#define LSX_CODE_GOLDEN_DIR "tests/fixtures/code"
#endif

namespace {

constexpr const char *kModelType = "nemotron-3.5-lightning-30b-a3b-nvfp4";

std::filesystem::path NemotronNvfp4ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_NEMOTRON_NVFP4_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
}

std::filesystem::path NemotronNvfp4GoldenPath()
{
    if (const char *env = std::getenv("LSX_CODE_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_CODE_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("nemotron-3.5-lightning-30b-a3b-nvfp4 CODESelfCheck: golden answers "
          "match the converted model", "[code][real-model]")
{
    const std::filesystem::path root = NemotronNvfp4ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("nemotron-3.5-lightning-30b-a3b-nvfp4 converted model directory "
             "not present");
    const std::filesystem::path golden_path = NemotronNvfp4GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("CODE golden not captured for "
             "nemotron-3.5-lightning-30b-a3b-nvfp4");

    lsxcommon::code::CodeGolden golden;
    std::string err;
    REQUIRE(lsxcommon::code::LoadCodeGoldenFile(golden_path.string(), golden,
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

    const lsxcommon::code::CodeResult result =
        lsxcommon::code::RunCodeSelfCheck(*model, golden);
    for (const lsxcommon::code::CodeCaseResult &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected
                     << "' extracted='" << c.extracted << "' raw='" << c.raw
                     << "' error='" << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
