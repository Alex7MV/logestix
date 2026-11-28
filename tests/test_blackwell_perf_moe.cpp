// MoE decode throughput baselines (Qwen3.5-35B-A3B, gemma-4-26B-A4B-it) on
// real converted weights. Prints tok/s / ttft and the decoded text every run
// so fast-path regressions are visible in the log.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <vector>

namespace {

// Converted root: unified vault convention <model>/logestix.
// The artifact of a model, by the identity the registry knows. The old helper
// took the STORE's directory name and joined it onto the source root, which has
// no org — so it pointed at `<root>/Qwen3.5-35B-A3B`, a path that never exists —
// and it read the Qwen env var for the gemma case as well.
std::filesystem::path ModelPath(const char *model_type, const char *env_name)
{
    if (const char *env = std::getenv(env_name))
        return std::filesystem::path(env);
    return manifest_query::ConvertedDirFor(model_type);
}

std::filesystem::path QwenMoePerfModelPath()
{
    return ModelPath("qwen3.5-35b-a3b", "LSX_TEST_QWEN_MOE_MODEL_PATH");
}

int PerfGenTokens()
{
    if (const char *p = std::getenv("LSX_TEST_PERF_GEN_TOKENS")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return 16;
}

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

TEST_CASE("Qwen3.5-35B-A3B perf baseline: decode throughput",
          "[qwen35moe][perf][real-model]")
{
    const std::filesystem::path model = QwenMoePerfModelPath();
    if (!std::filesystem::exists(model / "pinned"))
        SKIP(std::string("Qwen3.5-35B-A3B converted model directory not present: ") + model.string());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    // LSX_TEST_FAST_MATH=1 -> tensor-core W4A4 path.
    cfg.fast_math = std::getenv("LSX_TEST_FAST_MATH") != nullptr;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "qwen3.5-35b-a3b");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodePerfPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    const std::vector<int32_t> gen_ids(r.gen_tokens.begin(),
                                       r.gen_tokens.end());
    std::string text(64 * gen_ids.size() + 1, '\0');
    size_t text_cap = text.size() - 1;
    lsxcommon::arrow_tokenizer_decode(m->Tokenizer(), gen_ids.data(),
                                      static_cast<int>(gen_ids.size()),
                                      text.data(), &text_cap);
    text.resize(text_cap);
    std::string id_list;
    for (size_t i = 0; i < std::min<size_t>(gen_ids.size(), 6); ++i)
        id_list += std::to_string(gen_ids[i]) + " ";
    SPDLOG_INFO("[PERF][qwen3.5-35b-a3b] tok/s={:.1f} ttft_ms={:.2f} "
                "total_ms={:.2f} gen_tokens={} resident_mb={:.0f} "
                "ids[0..5]=[{}] text='{}'",
                r.tokens_per_sec, r.ttft_ms, r.total_time_ms,
                r.gen_tokens.size(),
                static_cast<double>(r.weight_resident_bytes) /
                    (1024.0 * 1024.0),
                id_list, text);
}

TEST_CASE("gemma-4-26B-A4B-it perf baseline: decode throughput",
          "[gemma26moe][perf][real-model]")
{
    const std::filesystem::path model =
        ModelPath("gemma-4-26b-a4b-it", "LSX_TEST_GEMMA_MOE_MODEL_PATH");
    if (!std::filesystem::exists(model / "pinned"))
        SKIP(std::string("gemma-4-26B-A4B-it converted model directory not present: ") + model.string());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.fast_math = std::getenv("LSX_TEST_FAST_MATH") != nullptr;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gemma-4-26b-a4b-it");
    REQUIRE(m->Load());

    // Gemma frames the canonical prompt through the chat-template registry
    // (BOS + template); raw tokenizer encode produces garbage continuations.
    std::vector<int32_t> ids;
    REQUIRE(m->BuildPromptInputIds(m->Tokenizer(),
                                   "The capital of France is", ids));
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());

    const std::vector<int32_t> gen_ids(r.gen_tokens.begin(),
                                       r.gen_tokens.end());
    std::string text(64 * gen_ids.size() + 1, '\0');
    size_t text_cap = text.size() - 1;
    lsxcommon::arrow_tokenizer_decode(m->Tokenizer(), gen_ids.data(),
                                      static_cast<int>(gen_ids.size()),
                                      text.data(), &text_cap);
    text.resize(text_cap);
    std::string id_list;
    for (size_t i = 0; i < std::min<size_t>(gen_ids.size(), 6); ++i)
        id_list += std::to_string(gen_ids[i]) + " ";
    SPDLOG_INFO("[PERF][gemma-4-26b-a4b-it] tok/s={:.1f} ttft_ms={:.2f} "
                "total_ms={:.2f} gen_tokens={} resident_mb={:.0f} "
                "ids[0..5]=[{}] text='{}'",
                r.tokens_per_sec, r.ttft_ms, r.total_time_ms,
                r.gen_tokens.size(),
                static_cast<double>(r.weight_resident_bytes) /
                    (1024.0 * 1024.0),
                id_list, text);
}
