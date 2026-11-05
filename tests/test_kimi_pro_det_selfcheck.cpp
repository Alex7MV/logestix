// Golden parity gate for Kimi-K3.1-Pro (Model Ingestion
// Protocol stage 8): the converted NVFP4 model must reproduce the answer the
// SOURCE safetensors model gave for the same captured input, at FastMathMode
// kOff. SKIPs (4) without the converted weights or the golden - a missing
// golden is a gap to report, never a pass.
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

constexpr const char *kModelType = "kimi-k3.1-pro";

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_KIMI_PRO_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Kimi-K3.1-Pro";
}

std::filesystem::path GoldenPath()
{
    if (const char *env = std::getenv("LSX_DET_GOLDEN_DIR"))
        return std::filesystem::path(env) / (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_DET_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

} // namespace

TEST_CASE("kimi-k3.1-pro det selfcheck: golden answers match", "[det][real-model]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("kimi-k3.1-pro converted model not present");
    const std::filesystem::path golden_path = GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("DET golden not captured for kimi-k3.1-pro");

    lsxcommon::det::DetGolden golden;
    std::string err;
    REQUIRE(lsxcommon::det::LoadDetGoldenFile(golden_path.string(), golden, &err));
    REQUIRE(golden.model_type == kModelType);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    const auto result = lsxcommon::det::RunDetSelfCheck(*model, golden);
    for (const auto &c : result.cases)
        INFO("case " << c.id << " expected='" << c.expected << "' extracted='"
                     << c.extracted << "' error='" << c.error << "'");
    REQUIRE(result.all_passed);
    model->Unload();
}
