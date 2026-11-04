// TDD — Qwen3.5-9B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no Qwen checkpoint is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_QWEN_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path QwenModelPath()
{
    const char *env = std::getenv("LSX_TEST_QWEN_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Qwen3.5-9B";
}

// Raw BPE encode of the shared "capital of France" prompt. Qwen has no
// registered chat-template framing (prompt_framing.cpp only frames GLM), so the
// raw encode is the canonical path — matching BuildPromptInputIds("qwen3.5-9b").
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

TEST_CASE("Qwen3.5-9B SelfCheck: 'The capital of France is' -> 'Paris', "
          "3 stress cycles with bit-exact deterministic logits",
          "[qwen][model][selfcheck][real-model]")
{
    const std::filesystem::path model = QwenModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Qwen real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "qwen3.5-9b");
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
// Stage 4b on real weights. Every qwen member frames with the same ChatML/XML
// codec, and the 2026-11-04 correction moved the shared wire to what all their
// templates actually render (`render_content|trim` on message content and
// `tool|tojson` for tool definitions). qwen3.6/qwen3.8-27b/ornith/3.8-flash-next
// each already had this round-trip; qwen3.5-9b and qwen3.5-35b-a3b did not, so
// their framing could change with no real-weight evidence at all -- the exact
// instrument gap that let the compact-JSON deviation ship. This closes it for
// this member.
TEST_CASE("Qwen3.5-9B end-to-end tool round-trip (real weights)",
          "[qwen][model][tool][real-model]")
{
    const std::filesystem::path model = QwenModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Qwen3.5-9B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

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
    spdlog::info("[qwen35-9b-tool] output='{}' gen_tokens={}", r.output_text,
                 r.gen_tokens.size());
    REQUIRE(r.ok);

    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(m->ParseToolCalls(m->Tokenizer(), r.gen_tokens, calls));
    for (const auto &c : calls)
        spdlog::info("[qwen35-9b-tool] call name='{}' args={}", c.name,
                     c.arguments_json);

    // The native ChatML XML call, not a bare JSON object.
    const int32_t tc_open = lsxcommon::arrow_tokenizer_lookup(
        m->Tokenizer(), "<tool_call>", std::strlen("<tool_call>"));
    CHECK(std::find(r.gen_tokens.begin(), r.gen_tokens.end(), tc_open) !=
          r.gen_tokens.end());

    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
