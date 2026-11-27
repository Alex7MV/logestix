#include <catch2/catch_all.hpp>

#include <lsxcommon/dory_dense_convert.h>
#include <lsxcommon/lsx_inference_nvfp4.h>
#include <lsxcommon/model_factory.h>

#include "test_dory_dense_fixture.h"
#include "test_streaming_assert.h"

#include <cstring>
#include <filesystem>
#include <memory>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "dory_model_test";
    remove_all(tmp);
    create_directories(tmp);
    dorytest::WriteSafetensors(tmp, "model.safetensors",
                               dorytest::BuildSyntheticDoryModel());
    dorytest::WriteConfig(tmp);
    // The Dory fixture tokenizer carries the ChatML/think/tool sentinels the
    // model's token bridge injects as their own ids; the plain
    // synthetic_tok.json cannot exercise BuildPromptInputIds.
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_dory_tok.json",
              tmp / "tokenizer.json");
    auto st = dory::ConvertDoryModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return tmp;
}

} // namespace

TEST_CASE("dory model type autodetects from the converted index",
          "[dory][model]")
{
    auto tmp = BuildConvertedFixture();
    CHECK(LsxModelFactory::DetectType(tmp.string()) == "dory-1.0-4b");
    std::filesystem::remove_all(tmp);
}

TEST_CASE("dory model loads, infers and is deterministic", "[dory][model]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "dory-1.0-4b";
    cfg.fast_math_mode = FastMathMode::kOff;

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "dory-1.0-4b");
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14, 15};
    req.max_gen_tokens = 4;

    auto r1 = model->Infer(req);
    INFO(r1.error_msg);
    REQUIRE(r1.ok);
    CHECK(!r1.gen_tokens.empty());
    CHECK(r1.num_layers == dorytest::TestCfg().num_layers);

    // A fresh instance must reproduce the token stream and the final logits
    // bit-exactly (the decode is a fixed reduction tree, no atomics).
    auto model2 = LsxModelFactory::Create(cfg);
    REQUIRE(model2 != nullptr);
    REQUIRE(model2->Load());
    auto r2 = model2->Infer(req);
    INFO(r2.error_msg);
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

TEST_CASE("dory honours cooperative cancel", "[dory][model][lifecycle]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "dory-1.0-4b";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14, 15};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    std::filesystem::remove_all(tmp);
}

// The KV cache must APPEND on a continuation (at kv_rows), never restart at 0:
// a continuation whose prompt is the previous generated tail has to produce the
// same tokens as a fresh prefill of prompt+tail on a clean instance.
TEST_CASE("dory continuation matches a fresh prefill", "[dory][model]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "dory-1.0-4b";

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13};
    req.max_gen_tokens = 3;
    auto first = model->Infer(req);
    REQUIRE(first.ok);
    REQUIRE(first.gen_tokens.size() >= 2);

    ModelRequest cont;
    cont.input_ids = first.gen_tokens;   // the incremental-session contract
    cont.max_gen_tokens = 3;
    auto inc = model->Infer(cont);
    INFO(inc.error_msg);
    REQUIRE(inc.ok);

    // Fresh: prompt + tail, so the second call sees one more token.
    auto fresh_model = LsxModelFactory::Create(cfg);
    REQUIRE(fresh_model != nullptr);
    REQUIRE(fresh_model->Load());
    ModelRequest fresh;
    fresh.input_ids = req.input_ids;
    for (int32_t t : first.gen_tokens) fresh.input_ids.push_back(t);
    fresh.max_gen_tokens = 3;
    auto ref = fresh_model->Infer(fresh);
    REQUIRE(ref.ok);

    CHECK(inc.gen_tokens == ref.gen_tokens);

    fresh_model->Unload();
    model->Unload();
    std::filesystem::remove_all(tmp);
}

// The Paris diagnostic must be entry-point invariant. The shared gate driver
// sends the RAW probe ids (`selfcheck_frame_prompt: false`), while
// `lsxbenchmark --selfcheck` frames the prompt through BuildPromptInputIds.
// For this model the framed form is a chat turn whose answer restates the
// question ("The user …" with reasoning on, "The capital of France is Paris"
// with it off), so a framed probe can never satisfy the strict 1-2 token
// "Paris" contract. SelfCheck therefore encodes its own canonical probe, and
// this case proves the two entry points agree bit-for-bit.
TEST_CASE("dory selfcheck is invariant to how the caller framed the prompt",
          "[dory][model]")
{
    auto tmp = BuildConvertedFixture();
    lsxcommon::EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "dory-1.0-4b";
    cfg.fast_math_mode = FastMathMode::kOff;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest raw;
    {
        std::vector<int32_t> ids(64, 0);
        size_t cap = ids.size();
        const char *kProbe = "The capital of France is";
        const int32_t n = arrow_tokenizer_encode(
            model->Tokenizer(), kProbe, std::strlen(kProbe), ids.data(), &cap);
        REQUIRE(n > 0);
        ids.resize(static_cast<size_t>(n));
        raw.input_ids = ids;
    }
    raw.max_gen_tokens = 2;
    // NOTE: `selfcheck_passed` cannot be asserted here — the synthetic fixture
    // carries random weights, so it never answers "Paris". The invariance is
    // what this case pins; the answer itself is the real-weight gate.
    const auto a = model->SelfCheck(raw);
    INFO(a.error_msg);
    REQUIRE(a.ok);

    ModelRequest framed = raw;
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is",
                                       framed.input_ids));
    CHECK(framed.input_ids != raw.input_ids);   // the framed form really differs
    const auto b = model->SelfCheck(framed);
    REQUIRE(b.ok);
    CHECK(b.first_gen_tok_id == a.first_gen_tok_id);
    CHECK(b.gen_tokens == a.gen_tokens);
    REQUIRE(b.final_logits.size() == a.final_logits.size());
    for (size_t i = 0; i < a.final_logits.size(); ++i)
        REQUIRE(b.final_logits[i] == a.final_logits[i]);

    model->Unload();
    std::filesystem::remove_all(tmp);
}

// The fast-math PIN (measured 2026-11-10). The Blackwell W4A4 tensor-core path
// is tolerance-class per GEMM (relL2 0.06 at seq 1, 0.19 at seq 32 against the
// scalar reference), and Dory amplifies that beyond any other family: every
// mixer output is L2-normalized before it is fed forward, so a few-percent
// relative error in a projection becomes a DIRECTION error of the residual
// stream at the next layer, compounded over 2x40 recurrent passes.
//
// Symptom before the pin: "The capital of France is" framed as a chat answered
// "# Prompt\n\nWrite a short story about a man who is a professional
// photographer…" under the auto policy (TC on) versus "The user is asking a
// simple factual question … Paris" under the strict path. The model MUST pin
// kOff, so this case drives the process-global routing to the TC path before
// Infer and requires the same tokens a strict run produces.
TEST_CASE("dory pins the strict routing (TC fast-math cannot change the output)",
          "[dory][model][regression]")
{
    auto tmp = BuildConvertedFixture();
    lsxcommon::EnsureModelRegistry();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "dory-1.0-4b";
    cfg.fast_math_mode = FastMathMode::kOff;

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14, 15};
    req.max_gen_tokens = 6;

    // The strict reference.
    auto strict_model = LsxModelFactory::Create(cfg);
    REQUIRE(strict_model != nullptr);
    REQUIRE(strict_model->Load());
    const auto strict = strict_model->Infer(req);
    REQUIRE(strict.ok);
    strict_model->Unload();

    // Same request with the process globals pre-set to the TC fast-math path
    // and the model configured for the auto policy: the pin must win.
    lsx_nvfp4::SetLsxTcFastMath(true);
    lsx_nvfp4::SetLsxSeq1Gemv(true);
    cfg.fast_math_mode = FastMathMode::kAuto;
    auto auto_model = LsxModelFactory::Create(cfg);
    REQUIRE(auto_model != nullptr);
    REQUIRE(auto_model->Load());
    const auto auto_res = auto_model->Infer(req);
    REQUIRE(auto_res.ok);
    CHECK(auto_res.gen_tokens == strict.gen_tokens);
    REQUIRE(auto_res.final_logits.size() == strict.final_logits.size());
    for (size_t i = 0; i < strict.final_logits.size(); ++i)
        REQUIRE(auto_res.final_logits[i] == strict.final_logits[i]);
    auto_model->Unload();
    lsx_nvfp4::SetLsxTcFastMath(false);
    lsx_nvfp4::SetLsxSeq1Gemv(false);

    std::filesystem::remove_all(tmp);
}

// Frozen-geometry gate: the loaded passport must match the compiled constants
// (the synthetic path is exempt through LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
TEST_CASE("dory frozen passport matches the checkpoint", "[dory][model]")
{
    CHECK(dory::frozen::kHiddenSize == 2560);
    CHECK(dory::frozen::kNumLayers == 72);
    CHECK(dory::frozen::kNumHeads == 16);
    CHECK(dory::frozen::kNumKVHeads == 4);
    CHECK(dory::frozen::kHeadDim == 256);
    CHECK(dory::frozen::kQSize == 4096);
    CHECK(dory::frozen::kKVSize == 1024);
    CHECK(dory::frozen::kIntermediateSize == 7168);
    CHECK(dory::frozen::kVocabSize == 131072);
    CHECK(dory::frozen::kInputLayers + dory::frozen::kRecurrentLayers +
              dory::frozen::kOutputLayers == dory::frozen::kNumLayers);
    CHECK(dory::frozen::kRecurrentLoops == 2);
    CHECK(dory::frozen::kNumAttentionLayers == 36);
    CHECK(dory::frozen::kKvSlots == 8 + 20 * 2 + 8);
    // The per-layer pattern is what config.layer_types pins.
    CHECK(dory::frozen::LayerIsAttention(0));
    CHECK_FALSE(dory::frozen::LayerIsAttention(1));
    // full_attention iff layer % 8 == 6: 6, 14, 22, ... 70 (nine of the
    // thirty-six attention layers).
    CHECK(dory::frozen::LayerIsFullAttention(6));
    CHECK_FALSE(dory::frozen::LayerIsFullAttention(12));
    CHECK(dory::frozen::LayerIsFullAttention(14));
    CHECK(dory::frozen::LayerIsFullAttention(54));
    CHECK(dory::frozen::LayerIsFullAttention(70));
    CHECK(dory::frozen::LayerIsRecurrent(16));
    CHECK_FALSE(dory::frozen::LayerIsRecurrent(15));
    CHECK_FALSE(dory::frozen::LayerIsRecurrent(56));
}
