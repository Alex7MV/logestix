// Kimi-K2.7-Code decode throughput baseline (real weights; SKIP without
// model). Reporting test: prints tok/s every run so lever impact is visible.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/model_factory.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <vector>

namespace {

std::filesystem::path KimiPerfModelPath()
{
    if (const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Kimi-K2.7-Code";
}

int PerfGenTokens()
{
    if (const char *p = std::getenv("LSX_TEST_PERF_GEN_TOKENS")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return 16;
}

} // namespace

TEST_CASE("Kimi-K2.7-Code perf baseline: decode throughput",
          "[kimi][perf][real-model]")
{
    const std::filesystem::path model = KimiPerfModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("converted Kimi model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.fast_math = std::getenv("LSX_TEST_FAST_MATH") != nullptr;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "kimi-k2.7-code");
    REQUIRE(m->Load());

    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        m->Tokenizer(), prompt, std::strlen(prompt), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = PerfGenTokens();

    auto r = m->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    SPDLOG_INFO("[PERF][kimi-k2.7-code] tok/s={:.1f} ttft_ms={:.2f} "
                "total_ms={:.2f} gen_tokens={} resident_mb={:.0f}",
                r.tokens_per_sec, r.ttft_ms, r.total_time_ms,
                r.gen_tokens.size(),
                static_cast<double>(r.weight_resident_bytes) /
                    (1024.0 * 1024.0));
}
