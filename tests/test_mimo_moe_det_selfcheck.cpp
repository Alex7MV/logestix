// DETSelfCheck — mimo-v2.6-flash-rl: verifies the converted model reproduces
// the golden deterministic token captured from the source safetensors model.
// SKIPs without converted weights or golden.
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

constexpr const char *kModelType = "mimo-v2.6-flash-rl";

std::filesystem::path MiMoModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_MIMO_MODEL_PATH"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "MiMo-V2.6-Flash-RL";
}

} // namespace

TEST_CASE("mimo-v2.6-flash-rl DET SelfCheck: golden token matches the converted "
          "model", "[det][real-model]")
{
    const std::filesystem::path root = MiMoModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("mimo-v2.6-flash-rl converted model directory not present");
    const char *env = std::getenv("LSX_DET_GOLDEN_DIR");
    const std::filesystem::path golden_path =
        (env ? std::filesystem::path(env)
             : std::filesystem::path(LSX_DET_GOLDEN_DIR)) /
        (std::string(kModelType) + ".json");
    if (!std::filesystem::exists(golden_path))
        SKIP("DET golden not captured for mimo-v2.6-flash-rl");

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
    REQUIRE(model->Type() == kModelType);
    REQUIRE(model->Load());

    const lsxcommon::det::DetResult result =
        lsxcommon::det::RunDetSelfCheck(*model, golden);
    REQUIRE(result.all_passed);
    model->Unload();
}
