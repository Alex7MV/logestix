// TDD — Kimi-K2.7-Code SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted Kimi checkpoint is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_KIMI_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path KimiModelPath()
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Kimi-K2.7-Code";
}

std::vector<int32_t> EncodeSelfCheckPrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("Kimi-K2.7-Code SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[kimi][model][selfcheck][real-model]")
{
    const std::filesystem::path model = KimiModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("converted Kimi model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "kimi-k2.7-code");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    // DEBUG: decode round-trip of the prompt.
    {
        std::vector<char> buf(1024, 0);
        size_t cap2 = buf.size();
        lsxcommon::arrow_tokenizer_decode(
            m->Tokenizer(), ids.data(), ids.size(), buf.data(), &cap2);
        std::string decoded(buf.data(), cap2);
        CHECK(decoded == "The capital of France is");   // encode round-trip
    }

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL("Infer not ok: " + sc0.error_msg);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    CHECK(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
