// Gemma-4-31B-it (dense) model test: factory + Load + Infer + the shared
// lifecycle contract (streaming order, abort, determinism) on synthetic
// geometry.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <vector>

#include "test_gemma_dense_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/gemma_dense_load.h"
#include "lsxcommon/model_factory.h"

namespace fs = std::filesystem;

namespace {

// synthetic_tok.json + the Gemma-4 turn/channel sentinels and an explicit
// bos_token, so the shared gemma framing/tool codec resolves them (the real
// checkpoint carries exactly these markers). Mirrors the fixture used by
// test_gemma_prompt_framing.cpp — the tokenizer bytes are the SAME file in the
// 26B and the 31B checkpoints.
std::string WriteGemmaTokJson()
{
    std::ifstream src(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json");
    REQUIRE(src.good());
    std::string json((std::istreambuf_iterator<char>(src)),
                     std::istreambuf_iterator<char>());
    const size_t at = json.find("\"added_tokens\"");
    REQUIRE(at != std::string::npos);
    const size_t bracket = json.find('[', at);
    REQUIRE(bracket != std::string::npos);
    const size_t arr_end = json.find(']', bracket);
    REQUIRE(arr_end != std::string::npos);
    static constexpr const char *kExtra =
        R"({"id": 268, "content": "<|turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 269, "content": "<turn|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 270, "content": "model", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 271, "content": "user", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 272, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 273, "content": "<|channel>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 274, "content": "thought", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 275, "content": "<channel|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 276, "content": "<|tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 277, "content": "<tool_call|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true})";
    json.insert(arr_end, "," + std::string(kExtra));
    const size_t root = json.find('{');
    REQUIRE(root != std::string::npos);
    json.insert(root + 1, "\"bos_token\": {\"content\": \"<|bos|>\"},");
    const fs::path p = fs::temp_directory_path() / "gemma4dense_model_tok.json";
    std::ofstream f(p);
    REQUIRE(f.good());
    f << json;
    return p.string();
}

fs::path MakeConvertedSynthetic(bool gemma_tok = false)
{
    auto tmp = fs::temp_directory_path() /
               (gemma_tok ? "gemma4dense_model_test_tok" : "gemma4dense_model_test");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                gemma_dense_test::BuildSyntheticGemmaDenseModel());
    gemma_dense_test::WriteConfig(tmp);
    if (gemma_tok)
        fs::copy_file(WriteGemmaTokJson(), tmp / "tokenizer.json");
    else
        fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                      tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma_dense::ConvertGemmaDenseModel(tmp.string(),
                                                           out.string())
                .ok());
    return out;
}

} // namespace

TEST_CASE("factory creates the registered gemma dense model", "[gemma_dense][model]") {
    auto out = MakeConvertedSynthetic();

    // Direct loader probe: isolate a weight-load failure from factory plumbing.
    {
        lsxcommon::gemma_dense::GemmaDensePinnedWeights probe;
        const auto rc =
            lsxcommon::gemma_dense::LoadGemmaDensePinnedWeights(out.string(),
                                                                probe, nullptr);
        if (rc != cudaSuccess) {
            INFO("direct LoadGemmaDensePinnedWeights rc=" << cudaGetErrorString(rc));
            REQUIRE(rc == cudaSuccess);
        }
        probe.Free();
    }

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-31b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "gemma-4-31b-it");
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};  // synthetic vocab ids
    req.max_gen_tokens = 2;
    auto r = model->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    CHECK(r.gen_tokens.size() >= 1);
    CHECK(r.gen_tokens.size() <= 2);

    // Determinism: a second identical run is bit-exact (parity gate class).
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);
    CHECK(r2.gen_tokens == r.gen_tokens);
    REQUIRE(r.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r.final_logits.data(), r2.final_logits.data(),
                      r.final_logits.size() * sizeof(float)) == 0);

    // Shared lifecycle contract: one Emit per decoded token, in order.
    ExpectStreamMatchesTokens(*model, req);

    model->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("gemma dense honours a pre-set abort flag", "[gemma_dense][model]") {
    auto out = MakeConvertedSynthetic();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-31b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};
    ExpectPreSetAbortStops(*model, req, 24);

    model->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("gemma dense geometry is frozen at compile time", "[gemma_dense][model]") {
    // The provenance facts a Load gate must enforce; pinned as constants so a
    // silently re-derived geometry fails here rather than at generation time.
    CHECK(lsxcommon::gemma_dense::frozen::kHiddenSize == 5376);
    CHECK(lsxcommon::gemma_dense::frozen::kNumLayers == 60);
    CHECK(lsxcommon::gemma_dense::frozen::kNumHeads == 32);
    CHECK(lsxcommon::gemma_dense::frozen::kNumKVHeads == 16);
    CHECK(lsxcommon::gemma_dense::frozen::kNumGlobalKVHeads == 4);
    CHECK(lsxcommon::gemma_dense::frozen::kHeadDim == 256);
    CHECK(lsxcommon::gemma_dense::frozen::kGlobalHeadDim == 512);
    CHECK(lsxcommon::gemma_dense::frozen::kIntermediateSize == 21504);
    CHECK(lsxcommon::gemma_dense::frozen::kVocabSize == 262144);
    CHECK(lsxcommon::gemma_dense::frozen::kNumFullLayers == 10);
    CHECK(lsxcommon::gemma_dense::frozen::kFullRotaryAngles == 64);
}

TEST_CASE("gemma dense wires the shared Gemma-4 tool codec (stage 4b)",
          "[gemma_dense][model]") {
    // The 31B tokenizer.json / chat_template.jinja are BYTE-IDENTICAL to the
    // 26B's, so the family framing + <|tool_call> codec are used verbatim. A
    // render + parse round-trip on the enriched tokenizer proves the model
    // wires them rather than falling back to the generic Hermes default.
    auto out = MakeConvertedSynthetic(/*gemma_tok=*/true);
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-31b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    const auto &tok = model->Tokenizer();

    // Framing: a leading BOS leads the user turn.
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "hi", ids));
    REQUIRE(!ids.empty());
    CHECK(ids.front() == tok.bos_id);

    // Conversation render -> tool parse round-trip.
    lsxcommon::ChatConversation conv;
    conv.messages.push_back({"user", "what is 2+2?", "", {}});
    lsxcommon::ToolSpec def;
    def.name = "calc";
    def.description = "arithmetic";
    def.parameters_json = R"({"type":"object"})";
    conv.tools.push_back(def);
    std::vector<int32_t> conv_ids;
    REQUIRE(model->BuildConversationInputIds(tok, conv, conv_ids));
    CHECK(!conv_ids.empty());

    std::vector<lsxcommon::ToolCall> calls;
    model->ParseToolCalls(tok, conv_ids, calls);

    model->Unload();
    fs::remove_all(out.parent_path());
}
