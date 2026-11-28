// TDD — Blackwell perf baseline for DeepSeek-V4-Flash (audit Phase 0).
// Reporting test: measures decode throughput and logs a machine-readable
// summary. SKIPs without the real model (SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/tokenizer.h>

namespace {

std::filesystem::path DeepSeekModelPath()
{
    const char *env = std::getenv("LSX_TEST_DEEPSEEK_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : manifest_query::ConvertedDirFor("deepseek-v4-flash");
}

int PerfGenTokens()
{
    if (const char *p = std::getenv("LSX_TEST_PERF_GEN_TOKENS")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return 16;
}

// DeepSeek consumes the raw prompt (no chat-template framing).
std::vector<int32_t> EncodePerfPrompt(lsxcommon::ArrowTokenizerDb &tok)
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

TEST_CASE("DeepSeek-V4-Flash perf baseline: decode throughput",
          "[deepseek][perf][real-model]")
{
    const std::filesystem::path model = DeepSeekModelPath();
    if (!std::filesystem::exists(model / "pinned"))
        SKIP(std::string("DeepSeek real model directory not present: ") + model.string());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "deepseek-v4-flash");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodePerfPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    // Always decode and print the generated text (audit round-8 lesson:
    // throughput without output validation measured garbage generation).
    const std::vector<int32_t> gen_ids(r.gen_tokens.begin(),
                                       r.gen_tokens.end());
    std::string text(64 * gen_ids.size() + 1, '\0');
    size_t text_cap = text.size() - 1;
    lsxcommon::arrow_tokenizer_decode(m->Tokenizer(), gen_ids.data(),
                                      static_cast<int>(gen_ids.size()),
                                      text.data(), &text_cap);
    text.resize(text_cap);
    SPDLOG_INFO("[PERF][deepseek-v4-flash] tok/s={:.1f} ttft_ms={:.2f} "
                "total_ms={:.2f} gen_tokens={} resident_mb={:.0f} "
                "text='{}'",
                r.tokens_per_sec, r.ttft_ms, r.total_time_ms,
                r.gen_tokens.size(),
                static_cast<double>(r.weight_resident_bytes) /
                    (1024.0 * 1024.0),
                text);

    m->Unload();
}
