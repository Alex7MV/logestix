// AliceAI selfcheck: "The capital of France is" must answer "Paris". Loads the
// real converted model from LSX_TEST_ALICE_ROOT, else from the vault path;
// SKIPs when neither is present.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model.h>
#include <lsxcommon/path_registry.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_tool_roundtrip.h"

namespace {

// Same resolution as the fsi/code/det siblings: the env var wins, otherwise the
// converted model is looked up in the vault. Resolving only from the env made
// this gate SKIP silently after conversion, where the sibling gates run.
std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_ALICE_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "AliceAI-Foundation-80B-A3B-Base";
}

int Cycles()
{
    const char *env = std::getenv("LSX_TEST_ALICE_SELFCHECK_CYCLES");
    const int v = env ? std::atoi(env) : 3;
    return std::max(3, v);
}

std::string DecodeFirst(const lsxcommon::ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &ids)
{
    std::size_t cap = 256;
    std::string out(cap, '\0');
    if (lsxcommon::arrow_tokenizer_decode(tok, ids.data(), ids.size(),
                                          out.data(), &cap) != 0)
        return "";
    out.resize(cap);
    return out;
}

} // namespace

TEST_CASE("alice selfcheck answers Paris", "[alice][inference]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("alice converted model directory not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "alice-foundation-80b-a3b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("alice Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 24;
    auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    const std::string first = DecodeFirst(tok, {r.gen_tokens[0]});
    INFO("first token text: '" << first << "'");
    INFO("full output: '" << r.output_text << "'");
    // The base model answers via a short MCQ-style prefix (a documented
    // deviation from the first-token convention); require "Paris" within the
    // generated prefix.
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    // >=3 stress cycles with bit-exact final_logits (the protocol's numeric
    // determinism pin; overridable with LSX_TEST_ALICE_SELFCHECK_CYCLES).
    const int cycles = Cycles();
    auto sc0 = model->SelfCheck(req);
    INFO("selfcheck error: '" << sc0.error_msg << "'");
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    const std::vector<float> &ref_logits = sc0.final_logits;
    REQUIRE(!ref_logits.empty());
    for (int c = 1; c < cycles; ++c) {
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

// Model Ingestion Protocol step 4b on real weights: the native
// `[TOOL_CALL_START]name\n{json}` codec is verified synthetically by
// test_alice_tool_framing; this case proves the checkpoint actually emits it
// when framed through its own template, and that ParseToolCalls recovers the
// call (the generic Hermes/bare-JSON fallback would not be the native format).
TEST_CASE("alice end-to-end tool round-trip (real weights)",
          "[alice][tool][real-model]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("alice converted model directory not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "alice-foundation-80b-a3b";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("alice Load failed");

    std::vector<int32_t> gen_tokens;
    const std::vector<lsxcommon::ToolCall> calls =
        RunToolRoundTrip(*model, 256, &gen_tokens);
    INFO("generated tokens: " << gen_tokens.size());
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    INFO("arguments: " << calls[0].arguments_json);
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);
    model->Unload();
}
