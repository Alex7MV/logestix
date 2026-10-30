// flint selfcheck (Model Ingestion Protocol stage 8): "The
// capital of France is" must answer "Paris" on the REAL converted weights, plus
// >=3 stress cycles with bit-exact final_logits determinism (cycle count
// overridable with LSX_TEST_FLINT_SELFCHECK_CYCLES). SKIPs (4) when the vault
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
    if (const char *env = std::getenv("LSX_TEST_FLINT_ROOT")) return env;
    return std::string(lsxcommon::lsx_model_models_dir()) +
           "/flint";
}

int Cycles()
{
    const char *env = std::getenv("LSX_TEST_FLINT_SELFCHECK_CYCLES");
    const int v = env ? std::atoi(env) : 3;
    return std::max(3, v);
}

} // namespace

TEST_CASE("flint selfcheck answers Paris", "[flint][inference]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("flint converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "flint";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("flint Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 256;
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

TEST_CASE("flint selfcheck is deterministic on the auto (ids-streaming) path",
          "[flint][inference]")
{
    // The parity goldens pin kOff, which resolves the routed ids on the host.
    // kAuto routes the expert GEMV through the device-table ids kernel, so that
    // code path needs its own real-shape (I=768 streaming) determinism pin.
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("flint converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "flint";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kAuto;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("flint Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
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

TEST_CASE("flint end-to-end tool round-trip (real weights)",
          "[flint][model][tool][real-model]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("flint converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "flint";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("flint Load failed");

    std::vector<int32_t> gen_tokens;
    const std::vector<lsxcommon::ToolCall> calls =
        RunToolRoundTrip(*model, 512, &gen_tokens);
    // The codec itself is verified byte-for-byte against chat_template.jinja by
    // test_flint_moe_tool_framing (render, parse, marker injection). On the
    // CONVERTED weights the model emits the native `<tool_call>`/`<arg_key>`
    // markers (so the parse path is exercised) but its free-form call name
    // drifts: the 4-bit routed-expert plane is the dominant residual error and
    // an 8-bit expert set does not fit this card (docs/OPEN_TASKS.md §R).
    REQUIRE(!calls.empty());
    CHECK_FALSE(calls[0].name.empty());
    model->Unload();
}
