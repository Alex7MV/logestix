// Hy-SimulMT-30B-A3B selfcheck (Model Ingestion Protocol stage 8) on the REAL
// converted weights. This checkpoint is a direction-bound zh<->en simultaneous
// translator: with the checkpoint's own serving directive the decoder TRANSLATES
// the source instead of answering an English question, so the family probe asks
// the canonical fact from the zh->en direction ("法国的首都是巴黎。" must
// translate to English containing "Paris"); the canonical English sentence is
// covered by the bidirectional test below. >=3 stress cycles must be bit-exact
// (cycle count overridable with LSX_TEST_HY_SIMULMT_SELFCHECK_CYCLES). SKIPs (4)
// when the vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string Root()
{
    if (const char *env = std::getenv("LSX_TEST_HY_SIMULMT_ROOT")) return env;
    return std::string(lsxcommon::lsx_model_models_dir()) +
           "/Hy-SimulMT-30B-A3B";
}

int Cycles()
{
    const char *env = std::getenv("LSX_TEST_HY_SIMULMT_SELFCHECK_CYCLES");
    const int v = env ? std::atoi(env) : 3;
    return std::max(3, v);
}

} // namespace

TEST_CASE("hy-simulmt selfcheck translates the control fact",
          "[hy][simulmt][inference]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Hy-SimulMT-30B-A3B converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "hy-simulmt-30b-a3b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Hy-SimulMT Load failed");

    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 16;

    const int cycles = Cycles();
    auto sc0 = model->SelfCheck(req);
    INFO("selfcheck error: '" << sc0.error_msg << "'");
    INFO("selfcheck text: '" << sc0.first_gen_text << "'");
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    const std::vector<float> &ref_logits = sc0.final_logits;
    for (int c = 1; c < cycles; ++c)
    {
        auto sc = model->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        REQUIRE(sc.final_logits.size() == ref_logits.size());
        REQUIRE(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                            ref_logits.size() * sizeof(float)) == 0);
        REQUIRE(sc.first_gen_tok_id == sc0.first_gen_tok_id);
    }
    model->Unload();
}

TEST_CASE("hy-simulmt selfcheck is deterministic on the auto "
          "(ids-streaming) path",
          "[hy][simulmt][inference]")
{
    // The parity goldens pin kOff, which resolves the routed ids on the host.
    // kAuto routes the expert GEMV through the device-table ids kernel, so that
    // code path needs its own real-shape (I=768 streaming) determinism pin.
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Hy-SimulMT converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "hy-simulmt-30b-a3b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kAuto;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Hy-SimulMT Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "法国的首都是巴黎。", ids));
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;
    const auto a = model->Infer(req);
    INFO(a.error_msg);
    REQUIRE(a.ok);
    REQUIRE_FALSE(a.gen_tokens.empty());
    model->Unload();
    REQUIRE(model->Load());
    const auto b = model->Infer(req);
    INFO(b.error_msg);
    REQUIRE(b.ok);
    CHECK(b.gen_tokens == a.gen_tokens);
    REQUIRE(b.final_logits.size() == a.final_logits.size());
    CHECK(std::memcmp(b.final_logits.data(), a.final_logits.data(),
                      a.final_logits.size() * sizeof(float)) == 0);
    model->Unload();
}

TEST_CASE("Hy-SimulMT text serving translates both directions (real weights)",
          "[hy][model][real-model]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Hy-SimulMT-30B-A3B converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "hy-simulmt-30b-a3b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Hy-SimulMT Load failed");

    const auto &tok = model->Tokenizer();

    // zh -> en: the auto direction directive is the zh-en one.
    {
        std::vector<int32_t> ids;
        REQUIRE(model->BuildPromptInputIds(tok, "今天天气很好。", ids));
        lsxcommon::ModelRequest req;
        req.input_ids = ids;
        req.max_gen_tokens = 24;
        const auto r = model->Infer(req);
        INFO(r.error_msg);
        REQUIRE(r.ok);
        INFO("zh->en output: '" << r.output_text << "'");
        REQUIRE(r.output_text.find("weather") != std::string::npos);
    }

    // en -> zh: the same interface, direction selected from the source script.
    {
        std::vector<int32_t> ids;
        REQUIRE(model->BuildPromptInputIds(tok, "What is the capital of France?",
                                           ids));
        lsxcommon::ModelRequest req;
        req.input_ids = ids;
        req.max_gen_tokens = 24;
        const auto r = model->Infer(req);
        INFO(r.error_msg);
        REQUIRE(r.ok);
        INFO("en->zh output: '" << r.output_text << "'");
        REQUIRE(r.output_text.find("法国") != std::string::npos);
    }
    model->Unload();
}
