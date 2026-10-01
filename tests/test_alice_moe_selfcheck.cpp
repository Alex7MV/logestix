// AliceAI selfcheck: "The capital of France is" must answer "Paris". Loads the
// real converted model from LSX_TEST_ALICE_ROOT; SKIPs when absent.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

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
    const char *root = std::getenv("LSX_TEST_ALICE_ROOT");
    if (!root) SKIP("LSX_TEST_ALICE_ROOT not set");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "alice-foundation-80b-a3b";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("alice Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;
    auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    const std::string first = DecodeFirst(tok, {r.gen_tokens[0]});
    INFO("first token text: '" << first << "'");
    INFO("full output: '" << r.output_text << "'");
    REQUIRE(first.find("Paris") != std::string::npos);
}
