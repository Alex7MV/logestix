// TDD — Qwen3.6-35B-A3B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted Qwen3.6-35B-A3B is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

#include <spdlog/spdlog.h>

#include "qwen_moe_36_model.h"

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_QWEN36_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

// Converted root: unified vault convention <model>/logestix.
std::filesystem::path Qwen36ConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN36_MODEL_PATH"))
        return std::filesystem::path(env);
    const char *src = std::getenv("LSX_TEST_QWEN36_SOURCE_PATH");
    return src ? std::filesystem::path(src)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Qwen3.6-35B-A3B";
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

TEST_CASE("Qwen3.6-35B-A3B SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[qwen36][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = Qwen36ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Qwen3.6-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    // Auto-detection path: index "model" -> "qwen3.6-35b-a3b".
    REQUIRE(m->Type() == "qwen3.6-35b-a3b");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

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

    m->Unload();
}

TEST_CASE("Qwen3.6-35B-A3B generation scratch persists across SelfCheck cycles",
          "[qwen36][model]")
{
    const std::filesystem::path model_dir = Qwen36ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Qwen3.6-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);

    auto *qm = dynamic_cast<lsxcommon::Qwen36MoeModel *>(m.get());
    REQUIRE(qm != nullptr);
    // Load() runs a 1-token warmup generation, so the arena is first sized
    // for a tiny sequence and the first SelfCheck may grow it exactly once.
    // The invariant under test is that subsequent cycles never reallocate.
    const int allocs0 = qm->GenScratchForTest().arena_allocs;
    CHECK(qm->GenScratchForTest().dequant_runs == 1);
    CHECK(allocs0 <= 2);

    auto sc1 = m->SelfCheck(req);
    REQUIRE(sc1.ok);
    CHECK(qm->GenScratchForTest().dequant_runs == 1);
    CHECK(qm->GenScratchForTest().arena_allocs == allocs0);

    REQUIRE(sc1.final_logits.size() == sc0.final_logits.size());
    CHECK(std::memcmp(sc1.final_logits.data(), sc0.final_logits.data(),
                      sc0.final_logits.size() * sizeof(float)) == 0);

    m->Unload();
}

TEST_CASE("Qwen3.6-35B-A3B end-to-end tool round-trip (real weights)",
          "[qwen36][model][tool][real-model]")
{
    const std::filesystem::path model_dir = Qwen36ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Qwen3.6-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    // Native ChatML/XML conversation with one tool.
    lsxcommon::ChatConversation conv;
    conv.messages.push_back(
        {"system",
         "You are a helpful assistant. Use the provided tools when needed.",
         "", {}});
    conv.messages.push_back(
        {"user",
         "What is the current weather in Paris? Use the get_weather tool.", "",
         {}});
    lsxcommon::ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]})";
    conv.tools.push_back(t);

    std::vector<int32_t> ids;
    REQUIRE(m->BuildConversationInputIds(m->Tokenizer(), conv, ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 256;
    auto r = m->Infer(req);
    spdlog::info("[qwen36-tool] output='{}' gen_tokens={}", r.output_text,
                 r.gen_tokens.size());
    REQUIRE(r.ok);

    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(m->ParseToolCalls(m->Tokenizer(), r.gen_tokens, calls));
    for (const auto &c : calls)
        spdlog::info("[qwen36-tool] call name='{}' args={}", c.name,
                     c.arguments_json);

    // The model must have produced the native ChatML XML call (not just a
    // bare JSON object), and the bridge parsed it back.
    const int32_t tc_open = lsxcommon::arrow_tokenizer_lookup(
        m->Tokenizer(), "<tool_call>", std::strlen("<tool_call>"));
    CHECK(std::find(r.gen_tokens.begin(), r.gen_tokens.end(), tc_open) !=
          r.gen_tokens.end());

    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
