// FSISelfCheck — Qwen3.6-35B-A3B: verifies the converted model reproduces the
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

constexpr const char *kModelType = "qwen3.6-35b-a3b";

std::filesystem::path Qwen36ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN36_MODEL_PATH"))
        return std::filesystem::path(env);
    const char *src = std::getenv("LSX_TEST_QWEN36_SOURCE_PATH");
    return src ? std::filesystem::path(src)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Qwen3.6-35B-A3B";
}

std::filesystem::path Qwen36GoldenPath()
{
    if (const char *env = std::getenv("LSX_FSI_GOLDEN_DIR"))
        return std::filesystem::path(env) /
               (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_FSI_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("Qwen3.6-35B-A3B FSI SelfCheck: golden answers match the converted "
          "model", "[qwen36][model][fsi][real-model]")
{
    const std::filesystem::path root = Qwen36ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "pinned"))
        SKIP("Qwen3.6-35B-A3B converted model directory not present");
    const std::filesystem::path golden_path = Qwen36GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("FSI golden not captured for qwen3.6-35b-a3b");

    lsxcommon::fsi::FsiGolden golden;
    std::string err;
    REQUIRE(lsxcommon::fsi::LoadFsiGoldenFile(golden_path.string(), golden,
                                              &err));
    REQUIRE(golden.model_type == kModelType);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == kModelType);
    REQUIRE(model->Load());

    const lsxcommon::fsi::FsiResult result =
        lsxcommon::fsi::RunFsiSelfCheck(*model, golden);
    for (const lsxcommon::fsi::FsiCaseResult &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected
                     << "' extracted='" << c.extracted << "' error='"
                     << c.error << "'");
    REQUIRE(result.all_passed);

    model->Unload();
}
