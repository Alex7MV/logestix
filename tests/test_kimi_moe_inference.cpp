// TDD — Kimi-K2.7-Code full Inference parity (Model Ingestion Protocol).
// Real-model test: SKIPs when no converted Kimi checkpoint is present.
#include <catch2/catch_test_macros.hpp>

#include "test_streaming_assert.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "kimi_moe_kernels.h"

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace {

std::filesystem::path KimiModelPath()
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Kimi-K2.7-Code";
}

std::vector<int32_t> EncodePrompt(lsxcommon::ArrowTokenizerDb &tok,
                                  const char *prompt)
{
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("Kimi-K2.7-Code full inference is deterministic across calls",
          "[kimi][model][inference][real-model]")
{
    const std::filesystem::path model = KimiModelPath();
    if (!std::filesystem::exists(model / "pinned"))
        SKIP("converted Kimi model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<int32_t> ids =
        EncodePrompt(m->Tokenizer(), "The capital of France is");
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 4;

    auto r1 = m->Infer(req);
    if (!r1.ok) FAIL("r1 failed: " + r1.error_msg);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 4);

    // Streaming contract: one chunk per decoded token, in order. This is what
    // lsxfabric --serve publishes, so a family that never invokes
    // req.token_callback serves an empty stream.
    ExpectStreamMatchesTokens(*m, req);

    // Cooperative cancellation must stop the decode before the budget, both
    // when the flag is pre-set and when it is raised mid-generation from the
    // token callback (the real --serve path). OPEN_TASKS §E1a.
    ExpectCooperativeCancelStops(*m, req);
    REQUIRE(!r1.final_logits.empty());

    // A failed call after the cancel runs would be a lifecycle leak: the abort
    // path must leave the instance usable.
    auto r2 = m->Infer(req);
    if (!r2.ok) FAIL("infer after cancel failed: " + r2.error_msg);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r1.gen_tokens);
    if (std::memcmp(r2.final_logits.data(), r1.final_logits.data(),
                    r1.final_logits.size() * sizeof(float)) != 0) {
        // First differing float + magnitudes (diagnostics).
        const size_t nf = r1.final_logits.size();
        for (size_t i = 0; i < nf; ++i) {
            if (r1.final_logits[i] != r2.final_logits[i]) {
                WARN("first diff at " << i << ": r1=" << r1.final_logits[i]
                     << " r2=" << r2.final_logits[i]);
                break;
            }
        }
        float mx = 0.f;
        for (size_t i = 0; i < nf; ++i)
            mx = std::fmax(mx, std::fabs(r1.final_logits[i] -
                                         r2.final_logits[i]));
        WARN("max abs diff=" << mx);
    }
    CHECK(std::memcmp(r2.final_logits.data(), r1.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);

    // Inspect reports the safetensors inventory of the SOURCE checkpoint.
    (void)m->Inspect();

    m->Unload();
}

// ── EOS host-truncate (device-side decode, F-11 Round 25) ──────────────────
TEST_CASE("Kimi-K2.7-Code repeated inference keeps the instance usable",
          "[kimi][model][inference][real-model]")
{
    const std::filesystem::path model = KimiModelPath();
    if (!std::filesystem::exists(model / "pinned"))
        SKIP("converted Kimi model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = EncodePrompt(m->Tokenizer(), "The capital of France is");
    REQUIRE(!req.input_ids.empty());
    req.max_gen_tokens = 4;

    // Every call must succeed and reproduce the first call's tokens. The
    // decode path allocates per-generation workspaces and re-plans the expert
    // cache, so a leak shows up as a failure or a divergence on a later call.
    std::vector<int32_t> first;
    for (int call = 0; call < 6; ++call) {
        const auto r = m->Infer(req);
        INFO("call " << call << " error: " << r.error_msg);
        REQUIRE(r.ok);
        REQUIRE(r.gen_tokens.size() == 4);
        if (call == 0) first = r.gen_tokens;
        else CHECK(r.gen_tokens == first);
    }
    m->Unload();
}

TEST_CASE("kimi EOS host-truncate cuts at first eos and flags termination",
          "[kimi][decode][eos]")
{
    namespace k = lsxcommon::kimi;
    std::vector<int> toks = {11, 22, 163586, 33, 44};
    auto r = k::TruncateAtEos(toks, 163586);
    REQUIRE(r.tokens.size() == 2);
    REQUIRE(r.tokens[0] == 11);
    REQUIRE(r.tokens[1] == 22);
    REQUIRE(r.eos_terminated);

    std::vector<int> no_eos = {11, 22, 33};
    auto r2 = k::TruncateAtEos(no_eos, 163586);
    REQUIRE(r2.tokens.size() == 3);
    REQUIRE_FALSE(r2.eos_terminated);

    // eos_id < 0: disabled.
    auto r3 = k::TruncateAtEos(toks, -1);
    REQUIRE(r3.tokens.size() == 5);
    REQUIRE_FALSE(r3.eos_terminated);

    // EOS as the very first token: empty output, still terminated.
    std::vector<int> immediate = {163586, 11};
    auto r4 = k::TruncateAtEos(immediate, 163586);
    REQUIRE(r4.tokens.empty());
    REQUIRE(r4.eos_terminated);
}
