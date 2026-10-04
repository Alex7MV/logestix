// Xing4.0-29B-A4B selfcheck (Model Ingestion Protocol stage 8): "The capital of
// France is" must answer "Paris" on the REAL converted weights, plus >=3 stress
// cycles with bit-exact final_logits determinism (cycle count overridable with
// LSX_TEST_XING_SELFCHECK_CYCLES). SKIPs (4) when the vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/model.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/model_factory.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string Root()
{
    if (const char *env = std::getenv("LSX_TEST_XING_ROOT")) return env;
    return std::string(lsxcommon::lsx_model_models_dir()) + "/Xing4.0-29B-A4B";
}

int Cycles()
{
    const char *env = std::getenv("LSX_TEST_XING_SELFCHECK_CYCLES");
    const int v = env ? std::atoi(env) : 3;
    return std::max(3, v);
}

} // namespace

TEST_CASE("xing selfcheck answers Paris", "[xing][inference]")
{
    const std::string root = Root();
    if (!std::filesystem::exists(root + "/logestix/model.index.json"))
        SKIP("Xing4.0-29B-A4B converted model not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "xing4.0-29b-a4b";
    // Parity mode: the control prompt must reproduce the SOURCE model's answer,
    // so the numeric class is bit-exact.
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Xing Load failed");

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

    std::string first;
    {
        std::size_t cap = 256;
        std::string out(cap, '\0');
        REQUIRE(lsxcommon::arrow_tokenizer_decode(tok, &r.gen_tokens[0], 1,
                                                  out.data(), &cap) == 0);
        out.resize(cap);
        first = out;
    }
    INFO("first token text: '" << first << "'");
    INFO("output: '" << r.output_text << "'");
    // SentencePiece encodes "Paris" without a leading space after "is", so the
    // first generated token is "Paris" (the upstream greedy continuation of
    // this prompt is "Paris. and the capital of Germany is").
    REQUIRE(first == "Paris");
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    // Stress cycles: same prompt, fresh request, bit-identical logits.
    // Stress cycles: same prompt, bit-identical logits. SelfCheck() is the
    // greedy path; Infer() samples, so its token ids are not a determinism
    // oracle (the kimi/qwen pattern compares final_logits across cycles).
    const int cycles = Cycles();
    auto sc0 = model->SelfCheck(req);
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
