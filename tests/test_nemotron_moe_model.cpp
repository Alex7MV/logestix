// TDD — Nemotron-3.5 model/factory stage (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_fixture.h"
#include "test_streaming_assert.h"

#include "nemotron_moe_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer_arrows.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>

using namespace lsxcommon;

namespace {

void SetEnv(const char *k, const char *v)
{
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

} // namespace

TEST_CASE("factory resolves nemotron-3.5-lightning-30b-a3b", "[nemotron][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == NemotronHModel::kType);
}

TEST_CASE("nemotron Load accepts a synthetic converted model", "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_model_src");
    const auto dst = nemotrontest::MakeTempDir("nm_model_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    auto st = nemotron_moe::ConvertNemotronModel(src.string(), dst.string());
    REQUIRE(st.ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    CHECK(model->Type() == NemotronHModel::kType);
    model->Unload();
}

TEST_CASE("nemotron Infer runs the hybrid pipeline end-to-end on a tiny model",
          "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_infer_src");
    const auto dst = nemotrontest::MakeTempDir("nm_infer_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.max_gen_tokens = 2;
    req.ignore_eos = true;
    auto r = model->Infer(req);
    REQUIRE(r.error_msg.empty());
    REQUIRE(r.ok);
    CHECK(r.num_tokens == 2);
    CHECK(r.gen_tokens.size() == 2);
    CHECK(r.final_logits.size() == 32);
    ExpectStreamMatchesTokens(*model, req);
    model->Unload();
}

TEST_CASE("nemotron Load rejects a frozen-geometry mismatch", "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
    const auto root = nemotrontest::MakeTempDir("nm_model_geo");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/4096);

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
}

TEST_CASE("nemotron expert LRU stays bit-identical under slot eviction",
          "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_lru_src");
    const auto dst = nemotrontest::MakeTempDir("nm_lru_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    struct Run {
        std::vector<int32_t> tokens;
        std::vector<float> logits;
        int slots = 0;
        std::uint64_t cache_bytes = 0;
    };
    auto run = [&](const char *slots_env) -> Run {
        SetEnv("LSX_NEMOTRON_EXPERT_CACHE_SLOTS", slots_env);
        ModelInitConfig cfg;
        cfg.model_path = dst.string();
        cfg.model_type = NemotronHModel::kType;
        auto model = LsxModelFactory::Create(cfg);
        REQUIRE(model != nullptr);
        REQUIRE(model->Load());

        ModelRequest req;
        req.input_ids = {1, 2, 3};
        req.max_gen_tokens = 4;
        req.ignore_eos = true;
        auto r = model->Infer(req);
        REQUIRE(r.error_msg.empty());
        REQUIRE(r.ok);

        Run out;
        out.tokens = r.gen_tokens;
        out.logits = r.final_logits;
        const auto &cache =
            static_cast<NemotronHModel *>(model.get())->ExpertCacheForTest();
        out.slots = cache.device.slots;
        out.cache_bytes = cache.cache_bytes;
        model->Unload();
        return out;
    };

    // 4 routed experts, top-k 2: the full cache is the old resident path; the
    // 2-slot cache reserves half the device bytes. Output must not shift.
    const Run full = run("4");
    const Run tight = run("2");
    SetEnv("LSX_NEMOTRON_EXPERT_CACHE_SLOTS", "");

    CHECK(full.slots == 4);
    CHECK(tight.slots == 2);
    CHECK(tight.cache_bytes < full.cache_bytes);

    CHECK(tight.tokens == full.tokens);
    REQUIRE(tight.logits.size() == full.logits.size());
    CHECK(std::memcmp(tight.logits.data(), full.logits.data(),
                      full.logits.size() * sizeof(float)) == 0);
}

TEST_CASE("nemotron single-prompt framing uses the native chat template",
          "[nemotron][fast][prompt_framing]")
{
    // A chat model must never see an unframed prompt: the single-prompt path
    // (lsxbenchmark --run, lsxfabric --testprompt) must emit the same frame as
    // the conversation renderer — bare role word + "\n", generation prompt
    // opening <think> (verified against the checkpoint's chat_template.jinja).
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_frame_src");
    const auto dst = nemotrontest::MakeTempDir("nm_frame_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    nemotrontest::WriteNemotronTokenizer(
        src, TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json");
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    ArrowTokenizerDb db;
    REQUIRE(LoadTokenizerArrows(dst.string(), db));
    REQUIRE(db.valid());

    static constexpr char kPrompt[] = "The capital of France is";
    std::vector<int32_t> ids;
    NemotronHModel framing_probe(ModelInitConfig{});
    REQUIRE(framing_probe.BuildPromptInputIds(db, kPrompt, ids));
    REQUIRE(!ids.empty());

    // Round-trip the ids: the frame must be exactly the native single turn.
    const std::string expected = std::string("user\n") + kPrompt +
                                 "\nassistant\n<think>\n";
    const std::string decoded = runtime::DecodeGeneratedText(db, ids);
    INFO("decoded frame: [" << decoded << "]");
    REQUIRE(decoded == expected);

    // And the raw encoder must NOT be what the prompt path uses.
    std::vector<int32_t> bare;
    REQUIRE(prompt::EncodeRawPrompt(db, kPrompt, bare));
    REQUIRE(runtime::DecodeGeneratedText(db, bare) == kPrompt);
}

TEST_CASE("nemotron tool framing renders the XML declaration block",
          "[nemotron][tool_framing]")
{
    ChatConversation conv;
    conv.messages.push_back({"system", "sys", "", {}});
    conv.messages.push_back({"user", "What's the weather in Paris?", "", {}});
    ToolSpec t;
    t.name = "get_weather";
    t.description = "Get the current weather for a city";
    t.parameters_json =
        R"({"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]})";
    conv.tools.push_back(t);
    std::string out;
    REQUIRE(NemotronHModel::RenderNemotronConversation(conv, out));
    spdlog::info("[nemotron-render]\n{}", out);
    CHECK(out.find("<name>get_weather</name>") != std::string::npos);
    CHECK(out.find("</tools>") != std::string::npos);
}
