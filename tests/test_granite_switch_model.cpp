#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_convert.h>
#include <lsxcommon/granite_dense_switch_inference.h>
#include <lsxcommon/granite_dense_switch_load.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#include "test_granite_switch_fixture.h"
#include "test_streaming_assert.h"

#include <simdjson.h>

#ifndef LSX_GRANITE_SWITCH_FIXTURE_DIR
#define LSX_GRANITE_SWITCH_FIXTURE_DIR "tests/fixtures/granite_switch"
#endif

// The model under test. ONE source, two targets: the family shares converter,
// kernels and lifecycle body, so the stage-8 cases are the same cases — only the
// checkpoint and its frozen passport differ (`granite-switch-4.1-30b` default,
// `granite-switch-4.1-8b` when the 8B target recompiles this file).
#ifndef LSX_GSW_MODEL
#define LSX_GSW_MODEL "granite-switch-4.1-30b"
#endif
#ifndef LSX_GSW_ROOT_ENV
#define LSX_GSW_ROOT_ENV "LSX_TEST_GRANITE_SWITCH_MODEL_PATH"
#endif
namespace {
constexpr const char *kModelType = LSX_GSW_MODEL;
std::string JoinInts(const std::vector<int> &v)
{
    std::string s;
    for (int x : v) s += std::to_string(x) + " ";
    return s;
}

} // namespace

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    // The converter takes the model's identity from the checkpoint directory
    // name, so the fixture must live in a directory spelled like the vault entry
    // (`granite-switch-4.1-30b`). A generic temp dir name would make the emitted
    // index claim a model that does not exist — the same failure a mis-named
    // vault copy would produce.
    auto root = temp_directory_path() / "granite_switch_model_test";
    auto tmp = root / kModelType;
    remove_all(root);
    create_directories(tmp);
    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(lsxcommon::granite_switch::ConvertGraniteSwitchModel(
                tmp.string(), tmp.string())
                .ok());
    return tmp;
}

} // namespace

TEST_CASE("granite-switch model autodetects from the converted index",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();
    CHECK(LsxModelFactory::DetectType(tmp.string()) ==
          kModelType);
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch generation tracks the active adapter",
          "[granite_switch][model][inference]")
{
    auto tmp = BuildConvertedFixture();
    granite_switch::GraniteSwitchPinnedWeights pw;
    REQUIRE(granite_switch::LoadGraniteSwitchPinnedWeights(tmp.string(), pw,
                                                           0));
    granite_switch::GraniteSwitchKvCache kv;
    granite_switch::GraniteSwitchGenerationResult r;
    auto &sampler = sample::DefaultSampler();
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {5, 64, 6, 7}, 3, -1, r, sampler, &kv) == cudaSuccess);
    REQUIRE(r.ok);
    CHECK(r.last_adapter == 1);
    CHECK(kv.last_adapter == 1);
    CHECK(r.tokens.size() == 3);

    INFO("kv after prefill rows=" << kv.kv_rows << " adapter=" << kv.last_adapter << " last=" << kv.last_tok);
    granite_switch::GraniteSwitchGenerationResult c;
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {kv.last_tok}, 2, -1, c, sampler, &kv) == cudaSuccess);
    REQUIRE(c.ok);
    INFO("kv after cont rows=" << kv.kv_rows << " adapter=" << kv.last_adapter << " c.last=" << c.last_adapter);
    CHECK(c.last_adapter == 1);

    // A decision control token switches the route mid-session.
    granite_switch::GraniteSwitchGenerationResult d;
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {65, 8}, 2, -1, d, sampler, nullptr) == cudaSuccess);
    REQUIRE(d.ok);
    // A wrong adapter here means the prefix scan or the adapter metadata is off,
    // and the two look identical from `last_adapter` alone. Assert the metadata
    // the scan reads first, so a failure names the layer that is wrong.
    CHECK(pw.top.adapter_token_ids == std::vector<int>({64, 65}));
    CHECK(pw.top.adapter_substitute_token_ids == std::vector<int>({6, 7}));

    // The routing invariant, not a fixed number. This prompt routes to adapter 2,
    // but the decode loop inherits the adapter of whatever control token the
    // MODEL samples — and on a 2-layer synthetic checkpoint the argmax is
    // arbitrary, so a generated 64/65 legitimately re-routes. Asserting `== 2`
    // made the case depend on that argmax: it passed until a weight change moved
    // the sampled token onto a control id (it did, 2026-11-07). The scan itself
    // is covered exactly by test_granite_switch_prepare; what this case owns is
    // that generation follows the rule.
    int expected_adapter = 2;   // from the prompt {65, 8}
    for (int32_t tok : d.tokens) {
        for (int a = 0; a < pw.top.num_adapters; ++a) {
            if (tok == pw.top.adapter_token_ids[static_cast<size_t>(a)]) {
                expected_adapter = a + 1;
                break;
            }
        }
    }
    CHECK(d.last_adapter == expected_adapter);

    pw.Free();
    kv.Free();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch model loads, infers and is deterministic",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = kModelType;

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == kModelType);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    req.max_gen_tokens = 3;

    auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    CHECK(!r1.gen_tokens.empty());
    CHECK(r1.num_layers == gswtest::Cfg().num_layers);

    auto model2 = LsxModelFactory::Create(cfg);
    REQUIRE(model2 != nullptr);
    REQUIRE(model2->Load());
    auto r2 = model2->Infer(req);
    REQUIRE(r2.ok);
    CHECK(r1.gen_tokens == r2.gen_tokens);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    for (size_t i = 0; i < r1.final_logits.size(); ++i)
        REQUIRE(r1.final_logits[i] == r2.final_logits[i]);

    ExpectStreamMatchesTokens(*model, req);

    model2->Unload();
    model->Unload();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch honours cooperative cancel",
          "[granite_switch][model][lifecycle]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = kModelType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch continuation matches a fresh prefill",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = kModelType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 64, 11, 12, 13};
    req.max_gen_tokens = 3;
    auto first = model->Infer(req);
    REQUIRE(first.ok);

    ModelRequest cont;
    cont.input_ids = first.gen_tokens;
    cont.max_gen_tokens = 3;
    auto a = model->Infer(cont);
    REQUIRE(a.ok);

    // Fresh prefill of the concatenated prompt must reproduce the same tokens.
    ModelRequest fresh;
    fresh.input_ids = req.input_ids;
    fresh.input_ids.insert(fresh.input_ids.end(), first.gen_tokens.begin(),
                           first.gen_tokens.end());
    fresh.max_gen_tokens = 3;
    auto model2 = LsxModelFactory::Create(cfg);
    REQUIRE(model2 != nullptr);
    REQUIRE(model2->Load());
    auto b = model2->Infer(fresh);
    REQUIRE(b.ok);
    CHECK(a.gen_tokens == b.gen_tokens);

    model2->Unload();
    model->Unload();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch native tool round-trip (real weights)",
          "[granite_switch][model][tool][real-model]")
{
    const char *env = std::getenv(LSX_GSW_ROOT_ENV);
    const std::filesystem::path root =
        env ? std::filesystem::path(env)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                  kModelType;
    if (!std::filesystem::exists(root / "model.index.json"))
        SKIP(std::string(kModelType) + " converted model not present");
    const std::filesystem::path golden_path =
        std::filesystem::path(LSX_GRANITE_SWITCH_FIXTURE_DIR) /
        "tool_roundtrip.json";
    if (!std::filesystem::exists(golden_path))
        SKIP("granite-switch tool golden not captured");

    std::ifstream gf(golden_path);
    std::string gj((std::istreambuf_iterator<char>(gf)),
                   std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(simdjson::padded_string(gj)).get(doc) ==
            simdjson::SUCCESS);
    std::vector<int32_t> golden_ids;
    for (auto x : doc["input_ids"].get_array())
        golden_ids.push_back(static_cast<int32_t>(int64_t(x)));

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ChatConversation conv;
    conv.messages.push_back(
        {"user", "What is the current weather in Paris?", "", {}});
    lsxcommon::ToolSpec tool;
    tool.name = "get_weather";
    tool.description = "Get the current weather for a city";
    tool.parameters_json =
        R"({"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]})";
    conv.tools.push_back(tool);

    std::vector<int32_t> ids;
    REQUIRE(model->BuildConversationInputIds(model->Tokenizer(), conv, ids));
    if (ids != golden_ids) {
        std::string a, b;
        for (auto x : ids) a += std::to_string(x) + " ";
        for (auto x : golden_ids) b += std::to_string(x) + " ";
        INFO("got:  " << a << "\nwant: " << b);
    }
    REQUIRE(ids == golden_ids);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 128;
    auto r = model->Infer(req);
    REQUIRE(r.ok);
    INFO("output: " << r.output_text);
    std::vector<lsxcommon::ToolCall> calls;
    REQUIRE(model->ParseToolCalls(model->Tokenizer(), r.gen_tokens, calls));
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    model->Unload();
}

