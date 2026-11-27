// TDD — NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 model/factory stage
// (Model Ingestion Protocol step 3). Synthetic PRE-QUANTIZED checkpoint only;
// the real vault is never converted here.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"

#include "test_nemotron_moe_nvfp4_fixture.h"
#include "test_streaming_assert.h"

#include "nemotron_moe_nvfp4_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/nemotron_moe_nvfp4_convert.h"

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

TEST_CASE("factory resolves nemotron-3.5-lightning-30b-a3b-nvfp4",
          "[nemotron][nvfp4][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = NemotronNvfp4Model::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == NemotronNvfp4Model::kType);
}

TEST_CASE("nemotron-nvfp4 Load accepts a synthetic converted model",
          "[nemotron][nvfp4][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotronv4test::MakeTempDir("nmv4_model_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_model_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);
    REQUIRE(nemotron_nvfp4::ConvertNemotronNvfp4Model(src.string(), dst.string())
                .ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronNvfp4Model::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    CHECK(model->Type() == NemotronNvfp4Model::kType);
    model->Unload();
}

TEST_CASE("nemotron-nvfp4 Infer runs the hybrid pipeline end-to-end on a tiny model",
          "[nemotron][nvfp4][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotronv4test::MakeTempDir("nmv4_infer_src");
    const auto dst = nemotronv4test::MakeTempDir("nmv4_infer_dst");
    nemotronv4test::WriteNemotronNvfp4Checkpoint(src);
    REQUIRE(nemotron_nvfp4::ConvertNemotronNvfp4Model(src.string(), dst.string())
                .ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronNvfp4Model::kType;
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

    // Cooperative cancel (shared lifecycle contract, OPEN_TASKS §E1a): a raised
    // abort flag must stop the decode both before the first step and from the
    // token callback mid-generation.
    ExpectCooperativeCancelStops(*model, req, /*budget=*/24);
    model->Unload();
}

TEST_CASE("nemotron-nvfp4 Load rejects a frozen-geometry mismatch",
          "[nemotron][nvfp4][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
    const auto root = nemotronv4test::MakeTempDir("nmv4_model_geo");
    nemotronv4test::WriteNemotronNvfp4Index(root, /*hidden_override=*/4096);

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronNvfp4Model::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
}

TEST_CASE("nemotron-nvfp4 Load rejects the BF16 sibling's tree",
          "[nemotron][nvfp4][model]")
{
    // Both distributions emit the SAME format tag, so only the index `model`
    // field separates them; a mixed-up tree must fail Load rather than verify
    // against the wrong goldens.
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
    const auto root = nemotronv4test::MakeTempDir("nmv4_model_sibling");
    nemotronv4test::WriteNemotronNvfp4Index(
        root, /*hidden_override=*/2688, /*experts_no_gate=*/true,
        /*layout=*/lsxcommon::kIndexLayout,
        /*model_name=*/"Nemotron-3.5-Lightning-30B-A3B-BF16");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = NemotronNvfp4Model::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
}

TEST_CASE("nemotron-nvfp4 tool framing renders the XML declaration block",
          "[nemotron][nvfp4][tool_framing]")
{
    // Stage 4b: the variant reuses the parent's Qwen-style XML render/parse.
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
    REQUIRE(NemotronNvfp4Model::RenderNemotronConversation(conv, out));
    spdlog::info("[nemotron-nvfp4-render]\n{}", out);
    CHECK(out.find("<name>get_weather</name>") != std::string::npos);
    CHECK(out.find("</tools>") != std::string::npos);
}
