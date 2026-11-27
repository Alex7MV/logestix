// TDD — Nemotron-3.5 model/factory stage (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_fixture.h"
#include "test_streaming_assert.h"

#include "nemotron_moe_model.h"
#include "nemotron_moe_constants.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer_arrows.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

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

// The body of one prompt from tools/fsi_prompts.json - the same file the
// capture tool reads, so the frame test and the capture cannot diverge on the
// prompt text (only on the framing around it, which is what it pins).
std::string FsiPromptBody(const std::string &id)
{
    const std::filesystem::path path =
        std::filesystem::path(LSX_SOURCE_DIR) / "tools" / "fsi_prompts.json";
    REQUIRE(std::filesystem::exists(path));
    std::ifstream in(path, std::ios::binary);
    const std::string json((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    simdjson::dom::array prompts;
    REQUIRE(doc["prompts"].get_array().get(prompts) == simdjson::SUCCESS);
    for (auto p : prompts) {
        std::string_view pid;
        if (p["id"].get_string().get(pid) == simdjson::SUCCESS && pid == id) {
            std::string_view body;
            REQUIRE(p["body"].get_string().get(body) == simdjson::SUCCESS);
            return std::string(body);
        }
    }
    FAIL("prompt " << id << " not found in fsi_prompts.json");
    return {};
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
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    CHECK(model->Type() == NemotronHModel::kType);
    model->Unload();
}

TEST_CASE("nemotron Load rejects a tree built for the other distribution",
          "[nemotron][fast][model]")
{
    // Both distributions emit the SAME `logestix-nemotron-nvfp4` format tag, so
    // only the index's `model` field separates their trees - and each carries
    // its own FSI/CODE/DET goldens. NemotronNvfp4Model has refused a foreign
    // tree since ingestion; the BF16 leaf did not, so the NVFP4 tree loaded as
    // `nemotron-3.5-lightning-30b-a3b` and would have verified against the
    // wrong golden set. (Found 2026-11-13 while giving this model its own.)
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_dist_src");
    const auto dst = nemotrontest::MakeTempDir("nm_dist_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    auto st = nemotron_moe::ConvertNemotronModel(src.string(), dst.string());
    REQUIRE(st.ok());

    const auto idx = dst / "logestix" / "model.index.json";
    {
        std::ifstream in(idx);
        const std::string json((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        REQUIRE(json.find("\"model\": \"" +
                          std::string(nemotron_moe::frozen::kModelName) + "\"") !=
                std::string::npos);
        const std::string other =
            "\"model\": \"Nemotron-3.5-Lightning-30B-A3B-NVFP4\"";
        std::string rewritten = json;
        static constexpr std::string_view kKey = "\"model\": \"";
        const auto at = rewritten.find(kKey);
        REQUIRE(at != std::string::npos);
        // Skip the OPENING quote of the value, then find the closing one.
        const auto end = rewritten.find('"', at + kKey.size());
        REQUIRE(end != std::string::npos);
        rewritten.replace(at, end + 1 - at, other);
        std::ofstream out(idx, std::ios::trunc);
        out << rewritten;
    }

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronHModel::kType;
    NemotronHModel model(cfg);
    CHECK_FALSE(model.Load());
    // A failed Load must not become a success on retry (partial state).
    CHECK_FALSE(model.Load());
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
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
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

// Cooperative cancel (OPEN_TASKS §E1a): nemotron steps the session host-side,
// so it must poll the abort flag there and drop a cancelled decode.
TEST_CASE("nemotron honours cooperative cancel",
          "[nemotron][model][lifecycle]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_cancel_src");
    const auto dst = nemotrontest::MakeTempDir("nm_cancel_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
}

TEST_CASE("nemotron Load rejects a frozen-geometry mismatch", "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
    const auto root = nemotrontest::MakeTempDir("nm_model_geo");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/4096);

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
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
    // Fixture: this tree was built by the family converter into a temp
    // directory, so it predates artifact.fb. The storage resolve and the
    // codec gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
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
    // (lsxbenchmark --run, lsxfabric --testprompt) must emit exactly what the
    // checkpoint's chat_template.jinja renders.
    //
    // The template assigns `system_message = ""` when messages[0] is not a
    // system turn and then tests `system_message is defined` — which is
    // therefore ALWAYS true — so the rendered frame opens with an EMPTY system
    // turn even when the caller supplied none. Rendering without it produces a
    // frame the checkpoint never sees; that divergence is the one that made
    // Ornith-1.5 collapse into repetition (see the 2026-10-07 framing fix).
    // Verified by rendering the vault's chat_template.jinja directly.
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

    // Round-trip the ids: the frame must be byte-identical to the template's
    // render of a single user message (empty leading system turn + the
    // generation prompt opening <think>).
    const std::string expected =
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" +
        std::string(kPrompt) +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const std::string decoded = runtime::DecodeGeneratedText(db, ids);
    INFO("decoded frame: [" << decoded << "]");
    REQUIRE(decoded == expected);

    // And the raw encoder must NOT be what the prompt path uses.
    std::vector<int32_t> bare;
    REQUIRE(prompt::EncodeRawPrompt(db, kPrompt, bare));
    REQUIRE(runtime::DecodeGeneratedText(db, bare) == kPrompt);
}

TEST_CASE("nemotron FSI-capture frame equals the served frame",
          "[nemotron][fast][prompt_framing]")
{
    // The parity gates feed the SOURCE a prompt built by tools/*_capture.py
    // through the checkpoint's chat_template.jinja, and then feed the ENGINE
    // the same captured ids. If those two frames ever differ, every parity
    // golden describes a prompt the served model never sees - and the gate
    // still goes green, because it compares the engine against a capture taken
    // from the same wrong frame.
    //
    // This checkpoint's generation prompt opens `<think>` (thinking ON by
    // default). The capture tools default to enable_thinking=False, which
    // renders `<think></think>` - a DIFFERENT frame. The fixture records what
    // the capture actually sent; this case pins the renderer to it.
    const std::filesystem::path frame_path =
        std::filesystem::path(LSX_SOURCE_DIR) /
        "tests/fixtures/nemotron/fsi01_frame.txt";
    REQUIRE(std::filesystem::exists(frame_path));
    std::ifstream frame_in(frame_path, std::ios::binary);
    const std::string expected((std::istreambuf_iterator<char>(frame_in)),
                               std::istreambuf_iterator<char>());
    REQUIRE_FALSE(expected.empty());

    // One user turn, no tools: exactly what tools/fsi_capture.py frames, with
    // the prompt body read from the same prompt set the capture reads.
    ChatConversation conv;
    conv.messages.push_back({"user", FsiPromptBody("FSI-TEST-01"), "", {}});
    std::string rendered;
    REQUIRE(NemotronHModel::RenderNemotronConversation(conv, rendered));
    INFO("rendered: [" << rendered << "]");
    INFO("captured: [" << expected << "]");
    REQUIRE(rendered == expected);
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
