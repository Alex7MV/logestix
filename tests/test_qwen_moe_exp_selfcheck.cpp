// TDD — Qwen3.8-Flash-Next SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted Qwen3.8-Flash-Next is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#include "qwen_moe_exp_model.h"
#include "test_streaming_assert.h"
#include "test_tool_roundtrip.h"

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_QWEN38_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

// Converted root: unified vault convention <model>/logestix.
std::filesystem::path Qwen38ConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN38_MODEL_DIR"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Qwen3.8-Flash-Next";
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

TEST_CASE("Qwen3.8-Flash-Next SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[qwen38][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = Qwen38ConvertedPath();
    const bool have_pinned =
        std::filesystem::exists(model_dir / "logestix" / "pinned");
    if (!have_pinned)
        SKIP("Qwen3.8-Flash-Next converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    cfg.model_type = lsxcommon::Qwen38FlashNextModel::kType;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "qwen3.8-flash-next");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL(sc0.error_msg);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());
    // TTFT must be measured over the first forward, not a placeholder.
    REQUIRE(sc0.ttft_ms > 0.0);
    REQUIRE(sc0.total_time_ms >= sc0.ttft_ms);

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

    // Plan 2 Task 12: the streamed callback must reproduce gen_tokens exactly.
    ExpectStreamMatchesTokens(*m, req);

    // Cooperative cancel (OPEN_TASKS §E1a / §E1c). This is the family the
    // lifecycle audit proved was unguarded, and the callback half of this gate
    // is specifically the E1c gate: the flag is raised from inside the token
    // callback, which can only stop the decode if tokens are published per step
    // instead of in one bulk drain after the whole budget has been generated.
    ExpectCooperativeCancelStops(*m, req);

    m->Unload();
}

// OPEN_TASKS §E1c: the driver used to generate the whole max_gen_tokens budget
// and truncate at the first EOS afterwards, so the tokens after EOS were decoded
// and thrown away, and a streaming consumer saw nothing until the whole budget
// had run. This prompt reaches EOS on the 5th generated token, so a 64-token
// budget separates the two behaviours cleanly.
TEST_CASE("Qwen3.8-Flash-Next end-to-end tool round-trip (real weights)",
          "[qwen38][model][tool][real-model]")
{
    const std::filesystem::path model_dir = Qwen38ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Qwen3.8-Flash-Next converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    cfg.model_type = lsxcommon::Qwen38FlashNextModel::kType;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<lsxcommon::ToolCall> calls = RunToolRoundTrip(*m);
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
