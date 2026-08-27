#include <catch2/catch_test_macros.hpp>

#include "qwen_moe_test_fixture.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "qwen_moe_model.h"
#include "lsxcommon/lifecycle_strategy.h"

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
        qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
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

TEST_CASE("fast_math toggle propagates from MoE model construction",
          "[qwen-moe-model]")
{
    EnsureModelRegistry();
    ModelInitConfig on;
    on.model_type = Qwen35MoeModel::kType;
    on.fast_math = true;
    auto m_on = LsxModelFactory::Create(on);
    REQUIRE(m_on != nullptr);
    CHECK(lsxcommon::qwen::QwenTcFastMath());

    ModelInitConfig off;
    off.model_type = Qwen35MoeModel::kType;
    off.fast_math = false;
    auto m_off = LsxModelFactory::Create(off);
    REQUIRE(m_off != nullptr);
    CHECK_FALSE(lsxcommon::qwen::QwenTcFastMath());
}

TEST_CASE("geometry mismatch rejected at load", "[qwen-moe-model]")
{
    EnsureModelRegistry();
    ConvertedModel fix;

    // Corrupt hidden_size in the index: ValidateGeometry must refuse.
    auto idx = fix.out / "logestix" / "model.index.json";
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

    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");   // hard gate ON
    ModelInitConfig cfg;
    cfg.model_type = Qwen35MoeModel::kType;
    cfg.model_path = fix.out.string();
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());

    // Synthetic escape hatch re-enabled for other tests in this binary.
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
}

TEST_CASE("lifecycle MoE plan bounds for qwen3.5-35b-a3b", "[qwen-moe-model]")
{
    using namespace lsxcommon::lifecycle;
    auto &gov = ResourceGovernor::Instance();
    gov.Shutdown();
    GovernorOptions gopts;
    gopts.override_hw = true;
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 64ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.vram_total_bytes = hw.vram_free_bytes = 32ull << 30;
    gopts.hw_override = hw;
    gopts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(gopts));

    LifecycleInput in;
    in.archetype = Archetype::MoE;
    in.model_type = "qwen3.5-35b-a3b";  // archetype fallback (no named strategy)
    in.expert_mean_bytes = 2ull << 20;  // ~2 MiB NVFP4 expert payload
    ResetPlanCache();
    const auto &plan = GetOrBuildPlan(in);
    CHECK(plan.expert_cache_slots >= plan.tuner_bounds.expert_cache_slots_min);
    CHECK(plan.expert_cache_slots <= plan.tuner_bounds.expert_cache_slots_max);
    CHECK(plan.expert_cache_min_bytes > 0);
    CHECK(plan.tuner_bounds.expert_cache_slots_max >= 4);
    ResetPlanCache();
    gov.Shutdown();
}
