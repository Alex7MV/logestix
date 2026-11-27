// MiniMax-M3 model stage (Model Ingestion Protocol stage 3/8): registry
// round-trip, the shared lifecycle contract (residency lock, streaming, abort,
// session state, result fill) and run-to-run determinism, on a synthetic
// converted tree.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cuda_runtime.h>

#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/minimax_moe_convert.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "test_minimax_moe_fixture.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

fs::path BuildModel()
{
    fs::path in = fs::temp_directory_path() / "lsx_minimax_model_in";
    fs::path out = fs::temp_directory_path() / "lsx_minimax_model_out";
    fs::remove_all(in);
    fs::remove_all(out);
    const fs::path tok = fs::path(LSX_SOURCE_DIR) / "tests" / "data" /
                         "synthetic_minimax_tok.json";
    REQUIRE(minimax_test::BuildSyntheticCheckpoint(in, true, tok) > 0);
    auto st = lsxcommon::minimax::ConvertMinimaxMoeModel(in.string(),
                                                         out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    return out;
}

std::vector<int32_t> PromptIds()
{
    // 3 in-vocabulary ids (the fixture vocab is 64 wide).
    return {5, 9, 13};
}

} // namespace

TEST_CASE("MiniMax model is registered under its kType", "[minimax][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "minimax-m3";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "minimax-m3");
    CHECK(model->KvLayout() == lsxcommon::KvCacheLayout::kPosMajor);
    // Unloaded instances must unload cleanly (idempotent teardown).
    model->Unload();
    model->Unload();
}

TEST_CASE("MiniMax model runs the full lifecycle on a synthetic tree",
          "[minimax][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "minimax-m3";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("load failed");

    lsxcommon::ModelRequest req;
    req.input_ids = PromptIds();
    req.max_gen_tokens = 4;

    // Streaming: one Emit per decoded token, in order.
    std::vector<int32_t> streamed;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        streamed.push_back(static_cast<int32_t>(c.token_id));
    };
    const auto r1 = model->Infer(req);
    INFO(r1.error_msg);
    REQUIRE(r1.ok);
    CHECK(r1.gen_tokens.size() == 4);
    CHECK(streamed.size() == r1.gen_tokens.size());
    CHECK(streamed == r1.gen_tokens);
    CHECK(r1.first_gen_tok_id == r1.gen_tokens.front());
    CHECK(r1.weight_resident_bytes > 0);
    CHECK_FALSE(r1.output_text.empty());
    CHECK(r1.kv_slots_used >= static_cast<int>(req.input_ids.size()));
    REQUIRE(r1.final_logits.size() ==
            static_cast<size_t>(minimax_test::TestCfg().vocab));

    // Determinism (greedy, kOff): the same prompt from a FRESH session must
    // reproduce the same tokens and bit-identical final logits.
    model->Unload();
    REQUIRE(model->Load());
    lsxcommon::ModelRequest req2 = req;
    req2.token_callback = nullptr;
    const auto r2 = model->Infer(req2);
    INFO(r2.error_msg);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r1.gen_tokens);
    REQUIRE(r2.final_logits.size() == r1.final_logits.size());
    bool bit_equal = true;
    for (size_t i = 0; i < r1.final_logits.size(); ++i)
        if (r1.final_logits[i] != r2.final_logits[i]) bit_equal = false;
    CHECK(bit_equal);

    // Abort before the first step: no tokens, no crash, published result.
    std::atomic<bool> abort{true};
    lsxcommon::ModelRequest ra = req;
    ra.abort_flag = &abort;
    ra.token_callback = nullptr;
    const auto r3 = model->Infer(ra);
    CHECK(r3.ok);
    CHECK(r3.gen_tokens.empty());

    // Session continuation: a request whose ids equal the previous generated
    // tail resumes the KV cache instead of re-prefilling.
    lsxcommon::ModelRequest rc = req;
    rc.token_callback = nullptr;
    rc.input_ids = r1.gen_tokens;
    const auto r4 = model->Infer(rc);
    INFO(r4.error_msg);
    REQUIRE(r4.ok);
    CHECK(r4.kv_slots_used > r1.kv_slots_used);

    model->Unload();
}

TEST_CASE("MiniMax SelfCheck runs the framed Paris probe", "[minimax][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "minimax-m3";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 16;
    const auto r = model->SelfCheck(req);
    // Random synthetic weights cannot answer "Paris"; what this pins is that
    // the probe is framed, runs and publishes a verdict.
    INFO(r.error_msg);
    CHECK(r.error_msg.empty());
    CHECK(r.num_tokens > 0);
    CHECK_FALSE(r.first_gen_text.empty());
    model->Unload();
}
