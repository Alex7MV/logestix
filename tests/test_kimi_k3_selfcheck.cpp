// Kimi-K3 selfcheck: "The capital of France is" must answer "Paris". Loads the
// real converted model from LSX_TEST_KIMI_K3_ROOT, else from the vault path;
// SKIPs when neither is present.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_KIMI_K3_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Kimi-K3";
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

}  // namespace

TEST_CASE("kimi-k3 selfcheck answers Paris", "[kimi-k3][inference]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("kimi-k3 converted model directory not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "kimi-k3";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("kimi-k3 Load failed");

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
    CHECK(first.find("Paris") != std::string::npos);

    // Three stress cycles must be bit-exact.
    for (int cycle = 0; cycle < 3; ++cycle) {
        const auto again = model->Infer(req);
        REQUIRE(again.ok);
        REQUIRE(again.gen_tokens == r.gen_tokens);
        REQUIRE(again.final_logits.size() == r.final_logits.size());
        for (size_t i = 0; i < r.final_logits.size(); ++i)
            REQUIRE(again.final_logits[i] == r.final_logits[i]);
    }
    model->Unload();
}
