// Gemma-4 model test: factory + Load + Infer on synthetic geometry.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstring>

#include "test_gemma_moe_fixture.h"

#include <filesystem>

#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/gemma_moe_load.h"
#include "lsxcommon/model_factory.h"

#include "gemma_base_model.h"
#include "test_streaming_assert.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeConvertedSynthetic()
{
    auto tmp = fs::temp_directory_path() / "gemma4_model_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_gemma_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());
    return out;
}

} // namespace

TEST_CASE("factory creates registered gemma model on synthetic fixture",
          "[gemma][model]") {
    auto out = MakeConvertedSynthetic();

    // Direct loader probe: isolate weight-load failure from factory plumbing.
    {
        lsxcommon::gemma::GemmaPinnedWeights probe;
        auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), probe,
                                                           nullptr);
        if (rc != cudaSuccess) {
            INFO("direct LoadGemmaPinnedWeights rc="
                 << cudaGetErrorString(rc));
            REQUIRE(rc == cudaSuccess);
        }
        probe.Free();
    }

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "gemma-4-26b-a4b-it");
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7}; // synthetic vocab ids
    req.max_gen_tokens = 2;
    auto r = model->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    CHECK(r.gen_tokens.size() >= 1);
    CHECK(r.gen_tokens.size() <= 2);

    // Determinism: second identical run must be bit-exact.
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);
    CHECK(r2.gen_tokens == r.gen_tokens);
    REQUIRE(r.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r.final_logits.data(), r2.final_logits.data(),
                      r.final_logits.size() * sizeof(float)) == 0);

    ExpectStreamMatchesTokens(*model, req);

    model->Unload();
    fs::remove_all(out.parent_path());
}


// Cooperative cancel (OPEN_TASKS §E1a): gemma-4-26b-a4b-it must drop a raised
// abort_flag mid-decode, not run the full budget and publish it. The gate's
// control run is what makes the two cancel assertions meaningful, and it only
// became satisfiable once §E1e's swallowed decode error was fixed.
TEST_CASE("gemma MoE honours cooperative cancel", "[gemma][model][lifecycle]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    fs::remove_all(out.parent_path());
}



TEST_CASE("gemma expert index parsed once across Infers", "[gemma][model]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};
    req.max_gen_tokens = 2;

    auto r1 = model->Infer(req);
    if (!r1.ok) FAIL("first Infer failed: " << r1.error_msg);
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);

    auto *gm = dynamic_cast<lsxcommon::GemmaBaseModel *>(model.get());
    REQUIRE(gm != nullptr);
    const auto &idx = gm->ExpertIndexForTest();
    CHECK(idx.parse_calls == idx.num_layers * idx.experts_per_layer);
    CHECK(idx.parse_ok == idx.num_layers * idx.experts_per_layer);

    model->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("gemma failed generation leaves no workspace behind (A3.6)",
          "[gemma][model][leak]") {
    // EnsureWorkspace's alloc chain short-circuits on the first failure and
    // returns WITHOUT ws.Free(): every buffer allocated before the failing
    // one stays live in the context. Drive a real mid-chain failure: fill
    // VRAM with a ballast so d_logits (alloc #11, V*4 B) fails after ~10
    // smaller buffers succeeded, then destroy the context exactly like
    // DestroyGemmaGenContext does. Free VRAM must return to the baseline.
    auto out = MakeConvertedSynthetic();
    lsxcommon::gemma::GemmaPinnedWeights pw;
    REQUIRE(lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), pw,
                                                     nullptr) == cudaSuccess);
    lsxcommon::gemma::GemmaExpertDb db;
    db.num_layers = pw.top.num_layers;
    db.experts_per_layer = pw.top.num_experts;
    db.entries.resize(static_cast<size_t>(db.num_layers) *
                      static_cast<size_t>(db.experts_per_layer));
    lsxcommon::gemma::GemmaKvCache kv;
    REQUIRE(kv.Allocate(pw, 16) == cudaSuccess);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Mid-chain failure at the LAST device alloc (d_gemv_o, #47): raise
    // q_rows/kv_rows so the three tail GEMV buffers are ~64 MB each, and
    // ballast VRAM down to ~96 MiB free. Every earlier buffer (tens of KiB
    // on the synthetic geometry) plus the first two tail buffers fit; the
    // last one fails. This leaves the MAXIMAL partial set live when
    // EnsureWorkspace returns without ws.Free().
    pw.vocab_size = 1000000;
    pw.top.vocab_size = 1000000;
    for (auto &lw : pw.layers) {
        lw.q_rows = 16000000;   // d_gemv_q/o: 32 MB each
        lw.kv_rows = 16000000;  // d_gemv_kv: 32 MB
    }
    size_t fb = 0, tb = 0;
    REQUIRE(cudaMemGetInfo(&fb, &tb) == cudaSuccess);
    void *ballast = nullptr;
    REQUIRE(fb > (64u << 20));
    size_t ballast_bytes = fb - (96u << 20);
    cudaError_t be = cudaMalloc(&ballast, ballast_bytes);
    while (be != cudaSuccess && ballast_bytes > (1u << 30)) {
        (void)cudaGetLastError();
        ballast_bytes -= 64u << 20;
        be = cudaMalloc(&ballast, ballast_bytes);
    }
    REQUIRE(be == cudaSuccess);

    // Warm-up: one failed generation + context destroy settles bookkeeping.
    {
        auto *warm = lsxcommon::gemma::CreateGemmaGenContext();
        lsxcommon::gemma::GemmaGenerationResult gen;
        (void)lsxcommon::gemma::RunGemmaGeneration(
            stream, pw, db, {1, 2, 3}, 2, {-1}, 0.f, gen, kv, warm);
        lsxcommon::gemma::DestroyGemmaGenContext(warm);
        (void)cudaGetLastError();
    }

    size_t free_before = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &tb) == cudaSuccess);
    for (int i = 0; i < 3; ++i) {
        auto *ctx = lsxcommon::gemma::CreateGemmaGenContext();
        lsxcommon::gemma::GemmaGenerationResult gen;
        const auto e = lsxcommon::gemma::RunGemmaGeneration(
            stream, pw, db, {1, 2, 3}, 2, {-1}, 0.f, gen, kv, ctx);
        INFO("gen rc=" << static_cast<int>(e) << " err=" << gen.error_msg);
        REQUIRE(e == cudaErrorMemoryAllocation);
        (void)cudaGetLastError();
        // Contract: a failed EnsureWorkspace leaves NO live workspace
        // pointer behind (the caller cannot know which allocs succeeded).
        REQUIRE(ctx->ws.d_x == nullptr);
        REQUIRE(ctx->ws.d_normed == nullptr);
        REQUIRE(ctx->ws.d_logits == nullptr);
        REQUIRE(ctx->ws.d_gemv_q == nullptr);
        REQUIRE(ctx->ws.d_gemv_kv == nullptr);
        REQUIRE(ctx->ws.h_sel == nullptr);
        lsxcommon::gemma::DestroyGemmaGenContext(ctx);
    }
    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &tb) == cudaSuccess);
    const std::int64_t delta = static_cast<std::int64_t>(free_before) -
                               static_cast<std::int64_t>(free_after);
    INFO("free VRAM delta over 3 failed gemma generations: "
         << delta << " bytes");
    // The pre-logits set is a few hundred KiB per call; three leaked sets
    // exceed 1 MiB. 512 KiB slack for allocator granularity.
    REQUIRE(delta < 512 * 1024);

    cudaFree(ballast);
    kv.Free();
    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(out.parent_path());
}
