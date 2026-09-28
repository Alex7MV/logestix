// TDD — GigaChat3.5-432B-A28B-Reasoning SelfCheck harness (Model Ingestion
// Protocol step 8). Real-model test: SKIPs when no converted GigaChat model is
// present (CTest SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#endif

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

#include "lsxcommon/gigachat_moe_model.h"
#include "test_tool_roundtrip.h"

namespace {

constexpr int kStressCycles = 3;

// Drops the routed-expert payloads from the OS page cache so the next Load
// sees a cold store (the state of a freshly booted host). This is the
// regression trigger: an unpinned mmap store page-faults from disk inside
// the first prefill and inflates TTFT (~115 s cold vs ~15 s warm on the
// 218 GiB GigaChat store).
void EvictExpertPageCache(const std::filesystem::path &root)
{
#if !defined(_WIN32)
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = root / "logestix" / "experts";
    if (!fs::exists(dir, ec)) return;
    for (fs::recursive_directory_iterator it(
             dir, fs::directory_options::skip_permission_denied, ec),
         end;
         it != end; it.increment(ec)) {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        const int fd = ::open(it->path().c_str(), O_RDONLY);
        if (fd < 0) continue;
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
#else
    (void)root;
#endif
}

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GIGACHAT_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path ConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_GIGACHAT_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "GigaChat3.5-432B-A28B-Reasoning";
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

TEST_CASE("GigaChat3.5 SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[gigachat][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "model.index.json"))
        SKIP("GigaChat3.5 converted model directory not present");

    // Cold-store regression trigger: force the expert payloads out of page
    // cache before Load so an unpinned store pays disk page-in inside TTFT.
    EvictExpertPageCache(model_dir);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gigachat3.5-432b-a28b-reasoning");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    INFO("selfcheck error: " << sc0.error_msg);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());
    // The cold store must not page-fault inside the measured first forward:
    // TTFT reflects the model's warm first-token latency, not disk I/O.
    REQUIRE(sc0.ttft_ms > 0.0);
    REQUIRE(sc0.ttft_ms < 15000.0);
    REQUIRE(sc0.total_time_ms >= sc0.ttft_ms);

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

TEST_CASE("GigaChat3.5-432B-A28B end-to-end tool round-trip (real weights)",
          "[gigachat][model][tool][real-model]")
{
    const std::filesystem::path model_dir = ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "model.index.json"))
        SKIP("GigaChat3.5 converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<lsxcommon::ToolCall> calls =
        RunToolRoundTrip(*m, 512);
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
