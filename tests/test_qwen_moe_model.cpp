#include <catch2/catch_test_macros.hpp>

#include "test_qwen_moe_fixture.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "qwen_moe_model.h"
#include "lsxcommon/lifecycle_strategy.h"

#include <cstdlib>
#include <cstring>
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

    // Tri-state (gpu_perf): explicit kOff still forces the bit-exact class
    // at construction; kAuto would leave the engine-init policy untouched.
    ModelInitConfig off;
    off.model_type = Qwen35MoeModel::kType;
    off.fast_math = false;
    off.fast_math_mode = lsxcommon::FastMathMode::kOff;
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

// ── Device-side token feed (80-tok/s decode round) ──────────────────────────

TEST_CASE("eos truncation helper", "[qwen-moe-decode]")
{
    std::vector<int32_t> t{5, 6, 7, 8};
    CHECK(qwen_moe::TruncateTokensAtEos(t, 7) == true);
    CHECK((t == std::vector<int32_t>{5, 6, 7}));
    std::vector<int32_t> u{5, 6, 8};
    CHECK(qwen_moe::TruncateTokensAtEos(u, 7) == false);
    CHECK((u == std::vector<int32_t>{5, 6, 8}));
    std::vector<int32_t> v{7};
    CHECK(qwen_moe::TruncateTokensAtEos(v, 7) == true);
    CHECK(v.empty() == false);   // legacy loop KEEPS the EOS token
    CHECK((v == std::vector<int32_t>{7}));
    std::vector<int32_t> w{1};
    CHECK(qwen_moe::TruncateTokensAtEos(w, -1) == false);  // disabled
}

// Kimi-race gate: two Infer calls on the same session must produce
// bit-identical final_logits with the device-fed decode loop.
TEST_CASE("device-fed decode run-to-run determinism", "[qwen-moe-decode]")
{
    ConvertedModel fx;
    // The model path needs tokenizer arrows: copy the shared synthetic BPE
    // tokenizer into the fixture output and re-run the conversion stage.
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        fx.in / "tokenizer.json",
        std::filesystem::copy_options::overwrite_existing);
    REQUIRE(qwen_moe::ConvertQwen35MoeModel(fx.in.string(), fx.out.string()).ok());
    ModelInitConfig cfg;
    cfg.model_type = Qwen35MoeModel::kType;
    cfg.model_path = fx.out.string();
    cfg.max_gen_tokens = 12;
    cfg.fast_math = true;
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());
    ModelRequest req;
    req.input_ids = {1, 2, 3, 4, 5};
    auto r1 = m->Infer(req);
    REQUIRE(r1.ok);
    auto r2 = m->Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);
    CHECK(r1.gen_tokens == r2.gen_tokens);
    m->Unload();
    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}
