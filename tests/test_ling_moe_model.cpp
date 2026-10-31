// Ling-3.1-flash model stage: registry round-trip, the shared lifecycle
// contract (residency lock, streaming, abort, session state, result fill) and
// run-to-run determinism on a synthetic converted tree.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/ling_moe_convert.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "test_ling_moe_fixture.h"

namespace fs = std::filesystem;

namespace {

fs::path BuildModel()
{
    fs::path in = fs::temp_directory_path() / "lsx_ling_model_in";
    fs::path out = fs::temp_directory_path() / "lsx_ling_model_out";
    fs::remove_all(in);
    fs::remove_all(out);
    ling_test::WriteCheckpoint(in);
    auto st = lsxcommon::ling::ConvertLingModel(in.string(), out.string());
    INFO(st.ToString());
    REQUIRE(st.ok());
    return out;
}

std::vector<int32_t> PromptIds() { return {5, 9, 13}; }

bool HasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

} // namespace

TEST_CASE("ling model is registered under its kType", "[ling][model]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "ling";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "ling");
    CHECK(model->KvLayout() == lsxcommon::KvCacheLayout::kPosMajor);
    model->Unload();
    model->Unload();
}

TEST_CASE("ling model runs the lifecycle on a synthetic tree", "[ling][model]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path root = BuildModel();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "ling";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("load failed");

    std::vector<int32_t> streamed;
    lsxcommon::ModelRequest req;
    req.input_ids = PromptIds();
    req.max_gen_tokens = 3;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        streamed.push_back(static_cast<int32_t>(c.token_id));
    };
    auto r = model->Infer(req);
    INFO(r.error_msg);
    REQUIRE(r.ok);
    REQUIRE_FALSE(r.gen_tokens.empty());
    CHECK(r.gen_tokens.size() <= 3);
    CHECK(streamed == r.gen_tokens);
    for (int32_t t : r.gen_tokens) {
        CHECK(t >= 0);
        CHECK(t < ling_test::TestCfg().vocab);
    }
    CHECK(r.kv_slots_used > 0);
    CHECK(r.weight_resident_bytes > 0);
    CHECK_FALSE(r.final_logits.empty());
    const std::vector<int32_t> first_tokens = r.gen_tokens;
    const std::vector<float> first_logits = r.final_logits;

    // Fresh prompt on the same instance resets the session; greedy decoding
    // must reproduce the same token stream and the same final logits bits.
    streamed.clear();
    auto r2 = model->Infer(req);
    INFO(r2.error_msg);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == first_tokens);
    REQUIRE(r2.final_logits.size() == first_logits.size());
    CHECK(std::memcmp(r2.final_logits.data(), first_logits.data(),
                      first_logits.size() * sizeof(float)) == 0);

    // Continuation: feeding the generated tail continues the session instead
    // of re-prefilling.
    lsxcommon::ModelRequest cont = req;
    cont.input_ids = first_tokens;
    cont.max_gen_tokens = 1;
    cont.token_callback = nullptr;
    auto r3 = model->Infer(cont);
    INFO(r3.error_msg);
    REQUIRE(r3.ok);
    CHECK(r3.kv_slots_used > r2.kv_slots_used);

    // Pre-set abort: no tokens are decoded and no Emit fires.
    std::atomic<bool> stop{true};
    streamed.clear();
    lsxcommon::ModelRequest ab = req;
    ab.max_gen_tokens = 100;
    ab.abort_flag = &stop;
    auto ra = model->Infer(ab);
    INFO(ra.error_msg);
    REQUIRE(ra.ok);
    CHECK(ra.gen_tokens.empty());
    CHECK(streamed.empty());

    model->Unload();
}
