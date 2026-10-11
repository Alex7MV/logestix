// Mistral-Small-4-119B selfcheck (Model Ingestion Protocol stage 8): "The
// capital of France is" must answer "Paris" on the REAL converted weights, plus
// >=3 stress cycles with bit-exact final_logits determinism (cycle count
// overridable with LSX_TEST_MISTRAL_SELFCHECK_CYCLES). SKIPs (4) when the vault
// entry is absent.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#include "test_tool_roundtrip.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string Root()
{
    if (const char *env = std::getenv("LSX_TEST_MISTRAL_ROOT")) return env;
    return std::string(lsxcommon::lsx_model_models_dir()) +
           "/Mistral-Small-4-119B-2603-NVFP4";
}

int Cycles()
{
    const char *env = std::getenv("LSX_TEST_MISTRAL_SELFCHECK_CYCLES");
    const int v = env ? std::atoi(env) : 3;
    return std::max(3, v);
}

} // namespace

TEST_CASE("mistral selfcheck answers Paris", "[mistral][inference]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Mistral-Small-4-119B converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "mistral-small-4-119b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Mistral Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 16;
    auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    std::string gen_ids;
    for (int32_t t : r.gen_tokens) gen_ids += std::to_string(t) + " ";
    INFO("gen_tokens: [" << gen_ids << "] output: '" << r.output_text << "'");
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    const int cycles = Cycles();
    auto sc0 = model->SelfCheck(req);
    INFO("selfcheck error: '" << sc0.error_msg << "'");
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

TEST_CASE("Mistral end-to-end tool round-trip (real weights)",
          "[mistral][model][tool][real-model]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Mistral-Small-4-119B converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "mistral-small-4-119b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Mistral Load failed");

    std::vector<int32_t> gen_tokens;
    const std::vector<lsxcommon::ToolCall> calls =
        RunToolRoundTrip(*model, 512, &gen_tokens);
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    model->Unload();
}
