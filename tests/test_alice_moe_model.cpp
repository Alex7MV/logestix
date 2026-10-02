// AliceAI-Foundation-80B-A3B model-stage tests. The passport case is host-only
// and runs first; the loader/pipeline cases land in later stages.
#include <catch2/catch_test_macros.hpp>

#include "alice_moe_constants.h"
#include "lsxcommon/alice_moe_convert.h"
#include "lsxcommon/alice_moe_load.h"
#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#include "test_alice_moe_fixture.h"
#include "test_streaming_assert.h"

using namespace lsxcommon::alice::frozen;

TEST_CASE("alice passport matches the audited checkpoint", "[alice][fast]")
{
    STATIC_REQUIRE(kHidden == 2048);
    STATIC_REQUIRE(kLayers == 48);
    STATIC_REQUIRE(kFullAttnInterval == 4);
    STATIC_REQUIRE(kBlockAttnRes == 4);
    STATIC_REQUIRE(kHeads == 16);
    STATIC_REQUIRE(kKVHeads == 2);
    STATIC_REQUIRE(kHeadDim == 256);
    STATIC_REQUIRE(kRotaryDim == 64);
    STATIC_REQUIRE(kLinearKeyHeads == 32);
    STATIC_REQUIRE(kLinearValueHeads == 32);
    STATIC_REQUIRE(kLinearKeyHeadDim == 128);
    STATIC_REQUIRE(kLinearValueHeadDim == 128);
    STATIC_REQUIRE(kLinearConvKernel == 4);
    STATIC_REQUIRE(kLinearKvRepeat == 1);
    STATIC_REQUIRE(kNumExperts == 512);
    STATIC_REQUIRE(kTopK == 10);
    STATIC_REQUIRE(kMoeIntermediate == 512);
    STATIC_REQUIRE(kSharedIntermediate == 512);
    STATIC_REQUIRE(kVocab == 129024);
    STATIC_REQUIRE(kBosId == 1);
    STATIC_REQUIRE(kEosId == 2);
    STATIC_REQUIRE(kMtpLayers == 1);
}

TEST_CASE("alice kType resolves in the factory", "[alice][fast]")
{
    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "alice-foundation-80b-a3b";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "alice-foundation-80b-a3b");
}

TEST_CASE("alice tiny hf parity", "[alice][model]")
{
    const char *hf = std::getenv("LSX_TEST_ALICE_TINY_HF");
    const char *out = std::getenv("LSX_TEST_ALICE_TINY_OUT");
    if (!hf || !out) SKIP("LSX_TEST_ALICE_TINY_HF/OUT not set");
    namespace fs = std::filesystem;
    fs::remove_all(out);
    auto st = lsxcommon::alice::ConvertAliceModel(hf, out);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out;
    cfg.model_type = "alice-foundation-80b-a3b";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("tiny load failed");
    lsxcommon::ModelRequest req;
    if (const char *ids_s = std::getenv("LSX_TEST_ALICE_TINY_IDS")) {
        std::string s(ids_s), cur;
        for (char c : s) {
            if (c == ',') { if (!cur.empty()) req.input_ids.push_back(std::stoi(cur)); cur.clear(); }
            else cur += c;
        }
        if (!cur.empty()) req.input_ids.push_back(std::stoi(cur));
    } else {
        req.input_ids = {1, 5, 7};
    }
    req.max_gen_tokens = 1;
    auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    const char *exp = std::getenv("LSX_TEST_ALICE_TINY_EXPECT");
    const int expect = exp ? std::atoi(exp) : 23;
    INFO("our argmax=" << r.first_gen_tok_id << " expect=" << expect);
    REQUIRE(r.first_gen_tok_id == expect);
}

TEST_CASE("alice synthetic generation is deterministic", "[alice][model]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    namespace fs = std::filesystem;
    const fs::path in = fs::temp_directory_path() / "alice_model_in";
    const fs::path out = fs::temp_directory_path() / "alice_model_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    alicetest::WriteAliceConfigJson(in);
    alicetest::WriteAliceShard(in);
    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());

    {
        lsxcommon::alice::AliceExpertHost host(out.string());
        const auto *pe = host.Get(0, 0);
        REQUIRE(pe != nullptr);
        REQUIRE(!pe->w1_packed.empty());
    }

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "alice-foundation-80b-a3b";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {1, 5, 7};
    req.max_gen_tokens = 2;
    auto r1 = model->Infer(req);
    if (!r1.ok) FAIL(r1.error_msg);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 2);

    auto r2 = model->Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r2.gen_tokens == r1.gen_tokens);
    REQUIRE(r2.final_logits == r1.final_logits);

    // The generation driver's timing contract. RunAliceGeneration reports
    // prefill-only ttft (Alice recomputes the whole sequence per step, so
    // "first token" is one full prefill) and a total that must cover it. A
    // zeroed or inverted pair is not a crash — it is worse, because every
    // perf reading derived from it is silently wrong.
    REQUIRE(r1.ttft_ms > 0.0);
    REQUIRE(r1.total_time_ms >= r1.ttft_ms);
    INFO("alice timings: ttft=" << r1.ttft_ms
                                 << " ms total=" << r1.total_time_ms
                                 << " ms tokens_per_sec=" << r1.tokens_per_sec);
    REQUIRE(r1.tokens_per_sec > 0.0);
}

namespace {

// Converts the synthetic checkpoint once per test case into a stable temp dir.
std::filesystem::path ConvertedSyntheticAlice(const char *tag)
{
    namespace fs = std::filesystem;
    const fs::path in = fs::temp_directory_path() /
                        (std::string("alice_model_in_") + tag);
    const fs::path out = fs::temp_directory_path() /
                         (std::string("alice_model_out_") + tag);
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    alicetest::WriteAliceConfigJson(in);
    alicetest::WriteAliceShard(in);
    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    return out;
}

std::unique_ptr<lsxcommon::IModel> LoadSyntheticAlice(const char *tag)
{
    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = ConvertedSyntheticAlice(tag).string();
    cfg.model_type = "alice-foundation-80b-a3b";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("alice synthetic load failed");
    return model;
}

} // namespace

TEST_CASE("alice Infer streams one chunk per generated token",
          "[alice][model][lifecycle]")
{
    // lsxfabric --serve / TaskMosaic stream exclusively through
    // req.token_callback: a family that never invokes it delivers an empty
    // stream (only the final DONE frame) to every client.
    auto model = LoadSyntheticAlice("stream");
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 5, 7};
    req.max_gen_tokens = 3;
    ExpectStreamMatchesTokens(*model, req);
    model->Unload();
}

TEST_CASE("alice Infer honours a pre-set abort flag",
          "[alice][model][lifecycle]")
{
    auto model = LoadSyntheticAlice("abort");
    std::atomic<bool> abort{true};
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 5, 7};
    req.max_gen_tokens = 4;
    req.abort_flag = &abort;
    const auto r = model->Infer(req);
    if (!r.ok) FAIL(r.error_msg);
    CHECK(r.ok);
    // Cancellation must stop the decode well before the requested budget.
    CHECK(r.gen_tokens.size() < static_cast<std::size_t>(req.max_gen_tokens));
    model->Unload();
}

TEST_CASE("alice publishes its resident weight bytes", "[alice][model][lifecycle]")
{
    // `lsxbenchmark --bench` prints `resident:` from
    // InferenceResult::weight_resident_bytes; a family that never fills it
    // reports 0 MiB no matter what it holds.
    auto model = LoadSyntheticAlice("resident");
    const auto r = model->Infer([&] {
        lsxcommon::ModelRequest req;
        req.input_ids = {1, 5, 7};
        req.max_gen_tokens = 1;
        return req;
    }());
    if (!r.ok) FAIL(r.error_msg);
    INFO("weight_resident_bytes=" << r.weight_resident_bytes);
    REQUIRE(r.weight_resident_bytes > 0);
    REQUIRE(r.weight_resident_bytes == model->ResidentWeightBytes());
    model->Unload();
}

TEST_CASE("alice repeated generation does not grow device memory",
          "[alice][model][lifecycle]")
{
    // RunAliceGeneration allocates ~40 device buffers per call. If any exit
    // path skipped the free block, repeated generations would ratchet VRAM up
    // until OOM — check the steady state over several calls.
    auto model = LoadSyntheticAlice("leak");
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 5, 7};
    req.max_gen_tokens = 2;

    std::size_t free_after_warmup = 0, total = 0;
    for (int i = 0; i < 4; ++i) {
        const auto r = model->Infer(req);
        if (!r.ok) FAIL(r.error_msg);
        std::size_t f = 0;
        cudaMemGetInfo(&f, &total);
        if (i == 1) free_after_warmup = f;
        if (i == 3) {
            const std::int64_t delta =
                static_cast<std::int64_t>(free_after_warmup) -
                static_cast<std::int64_t>(f);
            INFO("device free delta over 2 calls: " << delta << " bytes");
            // One cached CUDA context growth is tolerated; a leaked
            // workspace (tens of MiB here) is not.
            REQUIRE(delta < 8ll * 1024 * 1024);
        }
    }
    model->Unload();
}
