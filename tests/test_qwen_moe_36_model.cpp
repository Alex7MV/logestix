#include <catch2/catch_test_macros.hpp>

#include "test_qwen_moe_fixture.h"
#include "test_streaming_assert.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_36_convert.h"
#include "qwen_moe_36_model.h"
#include "lsxcommon/lifecycle_strategy.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace lsxcommon;

namespace {

// Builds the synthetic checkpoint, converts it with the 3.6 pipeline and
// returns the output dir.
struct ConvertedModel36 {
    std::filesystem::path in;
    std::filesystem::path out;

    ConvertedModel36()
    {
        in = std::filesystem::temp_directory_path() / "lsx_qwen36_model_in";
        out = std::filesystem::temp_directory_path() / "lsx_qwen36_model_out";
        std::filesystem::remove_all(in);
        std::filesystem::remove_all(out);
        qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
        qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
        auto st = qwen_moe_36::ConvertQwen36MoeModel(in.string(), out.string());
        if (!st.ok()) FAIL("fixture conversion failed: " + st.ToString());
    }
    ~ConvertedModel36()
    {
        std::error_code ec;
        std::filesystem::remove_all(in, ec);
        std::filesystem::remove_all(out, ec);
    }
};

} // namespace

TEST_CASE("factory resolves qwen3.6-35b-a3b", "[qwen36-model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = Qwen36MoeModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    // No weights behind an empty path: creation must still resolve the type.
    REQUIRE(model != nullptr);
    CHECK(model->Type() == Qwen36MoeModel::kType);
}

TEST_CASE("fast_math toggle propagates from Qwen3.6 MoE model construction",
          "[qwen36-model]")
{
    EnsureModelRegistry();
    ModelInitConfig on;
    on.model_type = Qwen36MoeModel::kType;
    on.fast_math = true;
    auto m_on = LsxModelFactory::Create(on);
    REQUIRE(m_on != nullptr);
    CHECK(lsxcommon::qwen::QwenTcFastMath());

    ModelInitConfig off;
    off.model_type = Qwen36MoeModel::kType;
    off.fast_math = false;
    off.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto m_off = LsxModelFactory::Create(off);
    REQUIRE(m_off != nullptr);
    CHECK_FALSE(lsxcommon::qwen::QwenTcFastMath());
}

TEST_CASE("geometry mismatch rejected at load (qwen3.6)", "[qwen36-model]")
{
    EnsureModelRegistry();
    ConvertedModel36 fix;

    // Corrupt hidden_size in the index: ValidateGeometry must refuse.
    auto idx = fix.out / "model.index.json";
    {
        std::ifstream f(idx);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string j = ss.str();
        auto pos = j.find("\"hidden_size\": 64");
        REQUIRE(pos != std::string::npos);
        j.replace(pos, strlen("\"hidden_size\": 64"), "\"hidden_size\": 63");
        std::ofstream g(idx, std::ios::trunc);
        g << j;
    }

    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");  // hard gate
    ModelInitConfig cfg;
    cfg.model_type = Qwen36MoeModel::kType;
    cfg.model_path = fix.out.string();
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());

    // Synthetic escape hatch re-enabled for other tests in this binary.
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
}

// Kimi-race gate parity: two Infer calls on the same session must produce
// bit-identical final_logits with the device-fed decode loop.
TEST_CASE("device-fed decode run-to-run determinism (qwen3.6)",
          "[qwen36-model][qwen36-decode]")
{
    ConvertedModel36 fx;
    // The model path needs tokenizer arrows: copy the shared synthetic BPE
    // tokenizer into the fixture input and re-run the conversion stage.
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        fx.in / "tokenizer.json",
        std::filesystem::copy_options::overwrite_existing);
    REQUIRE(
        qwen_moe_36::ConvertQwen36MoeModel(fx.in.string(), fx.out.string()).ok());
    ModelInitConfig cfg;
    cfg.model_type = Qwen36MoeModel::kType;
    cfg.model_path = fx.out.string();
    cfg.max_gen_tokens = 12;
    cfg.fast_math = true;
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());
    ModelRequest req;
    req.input_ids = {1, 2, 3, 4, 5};
    auto r1 = m->Infer(req);
    REQUIRE(r1.ok);
    auto r2 = m->Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);
    CHECK(r1.gen_tokens == r2.gen_tokens);
    ExpectStreamMatchesTokens(*m, req);
    m->Unload();
    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}

// Cooperative cancel (OPEN_TASKS §E1a): a raised abort_flag must stop the
// decode loop. qwen3.5 asserts this on the shared MoE pipeline; 3.6 must pin it
// on its own geometry so a change that breaks only this variant is caught.
TEST_CASE("qwen3.6-35b-a3b honours cooperative cancel",
          "[qwen36-model][qwen36-decode]")
{
    ConvertedModel36 fx;
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        fx.in / "tokenizer.json",
        std::filesystem::copy_options::overwrite_existing);
    REQUIRE(
        qwen_moe_36::ConvertQwen36MoeModel(fx.in.string(), fx.out.string()).ok());

    ModelInitConfig cfg;
    cfg.model_type = Qwen36MoeModel::kType;
    cfg.model_path = fx.out.string();
    cfg.fast_math = true;
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    ModelRequest req;
    req.input_ids = {1, 2, 3, 4, 5};
    ExpectCooperativeCancelStops(*m, req);

    m->Unload();
    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}
