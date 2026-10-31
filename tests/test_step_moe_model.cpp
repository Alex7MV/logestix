// Step-5-Preview model stage (Model Ingestion Protocol stage 3/8): registry
// round-trip, the shared lifecycle contract (residency lock, streaming, abort,
// session state, result fill) and run-to-run determinism on a synthetic
// converted tree.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/step_moe_convert.h"
#include "test_step_moe_fixture.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

fs::path BuildModel()
{
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_step_model_in";
    fs::path out = fs::temp_directory_path() / "lsx_step_model_out";
    fs::remove_all(in);
    fs::remove_all(out);
    const fs::path tok =
        fs::path(LSX_SOURCE_DIR) / "tests" / "data" / "synthetic_step_tok.json";
    REQUIRE(step_test::BuildSyntheticCheckpoint(in, true, tok) > 0);
    auto st = lsxcommon::step::ConvertStepMoeModel(in.string(), out.string());
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

TEST_CASE("Step-5 model is registered under its kType", "[step][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "step-5-preview";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "step-5-preview");
    CHECK(model->KvLayout() == lsxcommon::KvCacheLayout::kPosMajor);
    model->Unload();
    model->Unload();
}

TEST_CASE("Step-5 model runs the full lifecycle on a synthetic tree", "[step][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "step-5-preview";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("load failed");

    lsxcommon::ModelRequest req;
    req.input_ids = PromptIds();
    req.max_gen_tokens = 4;
    // The synthetic weights are arbitrary, so the first sampled id can be an
    // EOS id; the lifecycle test pins the token count, not the content.
    req.ignore_eos = true;

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
    CHECK(r1.kv_slots_used >= static_cast<int>(req.input_ids.size()));
    REQUIRE(r1.final_logits.size() ==
            static_cast<size_t>(step_test::TestCfg().vocab));

    // Same session, same input: bit-identical logits (in-session determinism).
    {
        lsxcommon::ModelRequest req1b = req;
        req1b.token_callback = nullptr;
        const auto r1b = model->Infer(req1b);
        REQUIRE(r1b.ok);
        REQUIRE(r1b.final_logits.size() == r1.final_logits.size());
        bool same = true;
        double maxdiff = 0.0;
        size_t ndiff = 0;
        for (size_t i = 0; i < r1.final_logits.size(); ++i) {
            const double d = std::fabs(static_cast<double>(r1.final_logits[i]) -
                                       static_cast<double>(r1b.final_logits[i]));
            if (d > 0.0) ++ndiff;
            if (d > maxdiff) maxdiff = d;
            if (r1.final_logits[i] != r1b.final_logits[i]) same = false;
        }
        INFO("in-session logits differ: ndiff=" << ndiff << " max=" << maxdiff);
        CHECK(same);

        // Third call (both warm): is the residue cold/warm or true
        // run-to-run nondeterminism?
        const auto r1c = model->Infer(req1b);
        REQUIRE(r1c.ok);
        size_t ndiff_bc = 0;
        double max_bc = 0.0;
        for (size_t i = 0; i < r1b.final_logits.size(); ++i) {
            const double d = std::fabs(static_cast<double>(r1b.final_logits[i]) -
                                       static_cast<double>(r1c.final_logits[i]));
            if (d > 0.0) ++ndiff_bc;
            if (d > max_bc) max_bc = d;
        }
        INFO("warm-vs-warm: ndiff=" << ndiff_bc << " max=" << max_bc);
        CHECK(ndiff_bc == 0);
    }

    // Determinism (greedy, kOff): same prompt from a FRESH session reproduces
    // the same tokens and bit-identical final logits.
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
    model->Unload();
}
