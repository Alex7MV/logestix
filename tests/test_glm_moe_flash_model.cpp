// Glm5NextFlashModel factory/registry + load lifecycle (real model SKIP).
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "lsxcommon/model_factory.h"

#include "glm_moe_flash_model.h"

using namespace lsxcommon;

namespace fs = std::filesystem;

TEST_CASE("registry: glm-5.3-flash type resolves factory")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "glm-5.3-flash");
}

TEST_CASE("Glm5NextFlashModel: Load on real checkpoint")
{
    const std::string root = "/tmp/glm53flash_conv_test";
    if (!fs::exists(root + "/logestix/model.index.json")) {
        SKIP("converted GLM-5.3-Flash not present");
    }
    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    REQUIRE(model->ResidentWeightBytes() > 0);
    model->Unload();
}
