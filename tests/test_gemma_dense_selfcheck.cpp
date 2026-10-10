// TDD — Gemma-4-31B-it (dense) SelfCheck harness (Model Ingestion Protocol
// step 8). Real-model test: SKIPs when no converted checkpoint is present
// (CTest SKIP_RETURN_CODE 4). Framing happens inside the model's own
// SelfCheck via the shared Gemma-4 chat-template registry.
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

#include <lsxcommon/gemma_dense_constants.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>
#include "test_tool_roundtrip.h"

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GEMMA_DENSE_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path GemmaDenseModelPath()
{
    const char *env = std::getenv("LSX_TEST_GEMMA_DENSE_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "gemma-4-31B-it";
}

} // namespace

TEST_CASE("Gemma-4-31B-it SelfCheck: framed 'The capital of France is' -> "
          "'Paris', stress cycles with bit-exact deterministic logits",
          "[gemma_dense][model][selfcheck][real-model]")
{
    const std::filesystem::path model = GemmaDenseModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma-4-31B-it real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "gemma-4-31b-it";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gemma-4-31b-it");
    REQUIRE(m->Load());

    // input_ids are unused: the model frames the canonical prompt itself.
    lsxcommon::ModelRequest req;
    req.input_ids = {2};  // <bos> placeholder
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL("SelfCheck failed: " << sc0.error_msg);

    REQUIRE(sc0.selfcheck_passed);
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

TEST_CASE("Gemma-4-31B-it final logits are the real softcapped row whose "
          "argmax is the emitted token",
          "[gemma_dense][model][logits][real-model]") {
    const std::filesystem::path model = GemmaDenseModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma-4-31B-it real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "gemma-4-31b-it";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {2};  // <bos> placeholder
    req.max_gen_tokens = 2;

    auto r = m->SelfCheck(req);
    if (!r.ok) FAIL("SelfCheck failed: " << r.error_msg);
    REQUIRE(r.ok);

    // Determinism alone cannot catch a broken final-logits COPY: a buffer that
    // is never written is still bit-identical across cycles. These assertions
    // pin the values, so a wrong-direction memcpy (host buffer left zeroed)
    // fails here instead of passing silently.
    const std::vector<float> &l = r.final_logits;
    REQUIRE(!l.empty());
    CHECK(l.size() == static_cast<size_t>(
                         lsxcommon::gemma_dense::frozen::kVocabSize));

    float lo = l[0], hi = l[0];
    for (const float v : l) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    INFO("logit range [" << lo << ", " << hi << "]");
    // A real distribution is spread around zero; an all-zero (unwritten) row
    // fails both bounds at once.
    CHECK(hi > 0.f);
    CHECK(lo < 0.f);
    CHECK(std::fabs(hi - lo) > 1.f);
    // softcap = tanh(l / 30) * 30 is bounded by the cap by construction.
    for (size_t i = 0; i < l.size(); ++i) {
        REQUIRE(std::fabs(l[i]) <=
                lsxcommon::gemma_dense::frozen::kFinalLogitSoftcap + 1e-3f);
    }

    // final_logits is the row sampled at the LAST decode step, so its argmax is
    // the LAST emitted token (first_gen_tok_id is the prefill token, a
    // different position).
    REQUIRE(!r.gen_tokens.empty());
    const int32_t tok = r.gen_tokens.back();
    REQUIRE(tok >= 0);
    REQUIRE(static_cast<size_t>(tok) < l.size());
    float row_max = l[static_cast<size_t>(tok)];
    for (size_t i = 0; i < l.size(); ++i) {
        if (i == static_cast<size_t>(tok)) continue;
        CAPTURE(i);
        REQUIRE(l[i] <= row_max);
        if (i < static_cast<size_t>(tok)) REQUIRE(l[i] < row_max);
    }

    m->Unload();
}

TEST_CASE("Gemma-4-31B-it end-to-end tool round-trip (real weights)",
          "[gemma_dense][model][tool][real-model]")
{
    const std::filesystem::path model = GemmaDenseModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma-4-31B-it real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "gemma-4-31b-it";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<lsxcommon::ToolCall> calls = RunToolRoundTrip(*m);
    REQUIRE(!calls.empty());
    // The model may namespace the function (Gemma emits e.g. "google:weather");
    // the codec contract is that a native call is rendered AND parsed back with
    // the right function token and argument, not the exact spelling.
    CHECK(calls[0].name.find("weather") != std::string::npos);
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
