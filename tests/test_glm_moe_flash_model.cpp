// Glm5NextFlashModel factory/registry + load lifecycle on the SYNTHETIC
// converted fixture (unified model-stage discipline: the model loads its own
// tiny checkpoint, never the model vault — real-weight integration lives in
// the selfcheck/perf suites).
//
// The load-side synthetic-geometry gate is enabled via the ctest ENVIRONMENT
// (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1); production loads stay strict.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "lsxcommon/model_factory.h"

#include "glm_moe_flash_model.h"

#include "test_glm_moe_flash_fixture.h"

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

TEST_CASE("Glm5NextFlashModel: Load lifecycle on synthetic converted fixture")
{
    glmflash::ConvertedFlashFixture fix;
    fix.in = fs::temp_directory_path() / "lsx_glmflash_model_in";
    fix.out = fs::temp_directory_path() / "lsx_glmflash_model_out";
    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
    INFO("convert status: " << fix.Convert().ToString());
    REQUIRE(fix.Convert().ok());

    ModelInitConfig cfg;
    cfg.model_path = fix.out.string();
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    REQUIRE(model->ResidentWeightBytes() > 0);
    // Conversion-time VRAM profile is written into the index and surfaced
    // through IModel::VramRequirements (lsxcli --info).
    const VramProfile &vp = model->VramRequirements();
    REQUIRE(vp.min_bytes > 0);
    REQUIRE(vp.optimal_bytes >= vp.min_bytes);
    REQUIRE(vp.max_bytes >= vp.optimal_bytes);
    REQUIRE(vp.context_tokens > 0);
    model->Unload();
    // Idempotent re-load support: Load() tears down and rebuilds.
    REQUIRE(model->Load());
    REQUIRE(model->ResidentWeightBytes() > 0);
    model->Unload();

    std::error_code ec;
    fs::remove_all(fix.in, ec);
    fs::remove_all(fix.out, ec);
}
