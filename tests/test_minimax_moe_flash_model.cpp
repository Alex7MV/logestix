// MiniMax-M3.1-Flash model stage (Model Ingestion Protocol stage 3/8):
// registry round-trip, the shared lifecycle contract (residency lock,
// streaming, abort, session state, result fill) and run-to-run determinism on
// a synthetic converted tree. The sibling reuses the parent's generation
// pipeline; what differs is its passport (theta 1e7, MSA on every layer) and
// its chat template policy.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cuda_runtime.h>

#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/minimax_moe_flash_convert.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "test_minimax_moe_flash_fixture.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

fs::path BuildModel()
{
    fs::path in = fs::temp_directory_path() / "lsx_minimax_flash_model_in";
    fs::path out = fs::temp_directory_path() / "lsx_minimax_flash_model_out";
    fs::remove_all(in);
    std::error_code ec;
    fs::remove_all(out, ec);
    const fs::path tok = fs::path(LSX_SOURCE_DIR) / "tests" / "data" /
                         "synthetic_minimax_tok.json";
    REQUIRE(minimax_flash_test::BuildSyntheticCheckpoint(in, true, tok) > 0);
    auto st = lsxcommon::minimax::ConvertMinimaxFlashModel(in.string(),
                                                           out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());
    return out;
}

std::vector<int32_t> PromptIds()
{
    return {5, 9, 13};
}

} // namespace

TEST_CASE("M3.1 model is registered under its kType", "[minimax-flash][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "minimax-m3.1-flash";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "minimax-m3.1-flash");
    CHECK(model->KvLayout() == lsxcommon::KvCacheLayout::kPosMajor);
    model->Unload();
    model->Unload();
}

TEST_CASE("M3.1 model runs the full lifecycle on a synthetic tree",
          "[minimax-flash][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "minimax-m3.1-flash";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("load failed");

    lsxcommon::ModelRequest req;
    req.input_ids = PromptIds();
    req.max_gen_tokens = 4;

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
            static_cast<size_t>(minimax_flash_test::TestCfg().vocab));

    // Determinism from a FRESH session.
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

    // Pre-set abort: zero tokens, published result.
    std::atomic<bool> abort{true};
    lsxcommon::ModelRequest ra = req;
    ra.abort_flag = &abort;
    ra.token_callback = nullptr;
    const auto r3 = model->Infer(ra);
    CHECK(r3.ok);
    CHECK(r3.gen_tokens.empty());

    // Session continuation: feeding the generated tail resumes the KV cache.
    lsxcommon::ModelRequest rc = req;
    rc.token_callback = nullptr;
    rc.input_ids = r1.gen_tokens;
    const auto r4 = model->Infer(rc);
    INFO(r4.error_msg);
    REQUIRE(r4.ok);
    CHECK(r4.kv_slots_used > r1.kv_slots_used);

    model->Unload();
}

TEST_CASE("M3.1 SelfCheck runs the framed adaptive probe", "[minimax-flash][model]")
{
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "minimax-m3.1-flash";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 192;
    const auto r = model->SelfCheck(req);
    INFO(r.error_msg);
    CHECK(r.error_msg.empty());
    CHECK(r.num_tokens > 0);
    CHECK_FALSE(r.first_gen_text.empty());
    model->Unload();
}
