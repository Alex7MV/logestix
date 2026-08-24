// TDD — Gemma-4-26B-A4B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no Gemma checkpoint is present (CTest
// SKIP_RETURN_CODE 4). Prompt framing (Gemma chat template) happens inside
// GemmaBaseModel::SelfCheck via the prompt_framing registry.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GEMMA_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path GemmaModelPath()
{
    const char *env = std::getenv("LSX_TEST_GEMMA_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "gemma-4-26B-A4B-it";
}

} // namespace

TEST_CASE("Gemma-4-26B-A4B SelfCheck: framed 'The capital of France is' -> "
          "'Paris', stress cycles with bit-exact deterministic logits",
          "[gemma][model][selfcheck][real-model]")
{
    const std::filesystem::path model = GemmaModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gemma-4-26b-a4b-it");
    REQUIRE(m->Load());

    // input_ids are unused: GemmaBaseModel::SelfCheck frames the canonical
    // prompt through the chat-template registry itself.
    lsxcommon::ModelRequest req;
    req.input_ids = {2}; // <bos> placeholder
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL("SelfCheck failed: " << sc0.error_msg);

    // Diagnostics (LSX_TEST_GEMMA_DEBUG=1): framed ids, decode, top-5.
    if (std::getenv("LSX_TEST_GEMMA_DEBUG")) {
        auto &tok = m->Tokenizer();
        std::string decoded;
        if (!sc0.gen_tokens.empty())
            decoded = lsxcommon::runtime::DecodeGeneratedText(tok,
                                                              sc0.gen_tokens);
        // Decode the framed prompt back for sanity.
        std::vector<int32_t> framed;
        m->BuildPromptInputIds(tok, "The capital of France is", framed);
        std::string prompt_text = lsxcommon::runtime::DecodeGeneratedText(
            tok, framed);
        std::printf("[dbg] framed_ids(%zu):", framed.size());
        for (auto id : framed) std::printf(" %d", id);
        std::printf("\n[dbg] framed_decode: ");
        for (auto id : framed) {
            std::string piece = lsxcommon::runtime::DecodeGeneratedText(
                tok, std::vector<int32_t>{id});
            std::printf("[%d='%s'] ", id, piece.c_str());
        }
        std::printf("\n");
        std::fflush(stdout);
        float mn = 1e30f, mx = -1e30f;
        size_t nans = 0;
        for (float v : sc0.final_logits) {
            if (std::isnan(v)) ++nans;
            else { mn = std::min(mn, v); mx = std::max(mx, v); }
        }
        // Top-5 tokens by logit.
        std::vector<std::pair<float, int>> top;
        for (size_t i = 0; i < sc0.final_logits.size(); ++i)
            top.push_back({sc0.final_logits[i], static_cast<int>(i)});
        std::sort(top.begin(), top.end(), [](auto &a, auto &b) {
            return a.first > b.first;
        });
        std::printf("[dbg] framed_ids=%zu prompt_decode='%s'\n", framed.size(),
                    prompt_text.c_str());
        std::printf("[dbg] logits min=%g max=%g nan=%zu gen=%s\n", mn, mx,
                    nans, decoded.c_str());
        for (int i = 0; i < 5 && i < static_cast<int>(top.size()); ++i)
            std::printf("[dbg] top%d: id=%d logit=%g\n", i, top[i].second,
                        top[i].first);
        std::fflush(stdout);
    }

    REQUIRE(sc0.selfcheck_passed);
    // This checkpoint greedily starts its answer with
    // "The capital of France is **Paris**" — the canonical answer string is
    // expected within the first generated tokens (verified against HF
    // transformers greedy decode).
    REQUIRE(sc0.first_gen_text.find("Paris") != std::string::npos);
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text.find("Paris") != std::string::npos);
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
