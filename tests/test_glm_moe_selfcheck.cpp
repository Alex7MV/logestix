// TDD — GLM-5.2 SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no GLM-5.2 checkpoint is present (CTest
// SKIP_RETURN_CODE 4, same as the other GLM real-weight tests).
//
// Contract under test (GlmV5Model::SelfCheck):
//   - the control prompt "The capital of France is" (GLM chat-template framed)
//     must correctly produce the reference answer "Paris" (top-1,
//     decoded via the incremental 1->2-token prefix),
//   - r.ok == true && r.selfcheck_passed == true,
//   - at least 3 consecutive selfcheck cycles are bit-exact deterministic:
//     the raw [vocab] logits row published in r.final_logits is byte-identical
//     across every run.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#include "test_streaming_assert.h"

namespace {

constexpr int kStressCycles = 3;

// Stress-cycle count, defaulting to the spec minimum (3). LSX_TEST_GLM_SELFCHECK_CYCLES
// lets CI/dev loops tune the real-model workload (each cycle is a full 79-layer
// forward) without changing the committed default.
int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GLM_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path GlmModelPath()
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "GLM-5.2";
}

// Encodes the shared "capital of France" self-check prompt with the GLM-5.2
// chat-template framing for the non-thinking generation mode (chat_template.jinja:
// [gMASK]<sop><|user|>{prompt}<|assistant|><think></think>). Empty on any
// tokenizer miss.
std::vector<int32_t> EncodeSelfCheckPrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);

    const int32_t gmask =
        lsxcommon::arrow_tokenizer_lookup(tok, "[gMASK]", 7);
    const int32_t sop = lsxcommon::arrow_tokenizer_lookup(tok, "<sop>", 5);
    const int32_t usr = lsxcommon::arrow_tokenizer_lookup(tok, "<|user|>", 8);
    const int32_t asst =
        lsxcommon::arrow_tokenizer_lookup(tok, "<|assistant|>", 13);
    const int32_t think =
        lsxcommon::arrow_tokenizer_lookup(tok, "<think>", 7);
    const int32_t endthink =
        lsxcommon::arrow_tokenizer_lookup(tok, "</think>", 8);
    if (gmask < 0 || sop < 0 || usr < 0 || asst < 0 || think < 0 ||
        endthink < 0)
        return {};

    size_t cap = ids.size() - 6;
    int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data() + 3, &cap);
    if (n <= 0) return {};
    ids[0] = gmask;
    ids[1] = sop;
    ids[2] = usr;
    ids[3 + n] = asst;
    ids[4 + n] = think;
    ids[5 + n] = endthink;
    n += 6;
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE(
    "GLM-5.2 SelfCheck: 'The capital of France is' -> 'Paris', "
    "3 stress cycles with bit-exact deterministic logits",
    "[glm][model][selfcheck][real-model]")
{
    const std::filesystem::path model = GlmModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "glm-5.2");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 6);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text == "Paris");
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    // Cooperative cancel (OPEN_TASKS §E1a): RunGlmRealGeneration polls the flag
    // per step; a cancelled request must stop short of the budget. SKIPs with
    // the rest of this case when GLM-5.2 is absent.
    ExpectCooperativeCancelStops(*m, req);

    m->Unload();
}