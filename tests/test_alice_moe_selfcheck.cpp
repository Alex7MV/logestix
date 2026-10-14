// AliceAI selfcheck: "The capital of France is" must answer "Paris". Loads the
// real converted model from LSX_TEST_ALICE_ROOT, else from the vault path;
// SKIPs when neither is present.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model.h>
#include <lsxcommon/path_registry.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

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
}
