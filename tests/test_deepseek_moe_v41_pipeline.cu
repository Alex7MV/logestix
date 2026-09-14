// tests/test_deepseek_moe_v41_pipeline.cu — DeepSeek-V4.1-Flash production
// pipeline smoke on a synthetic FULL-GEOMETRY converted model (one pure-SWA
// layer, two routed experts). Validates the execution graph (embed -> HC ->
// SWA attention -> router -> routed/shared MoE -> head -> sample) and
// run-to-run determinism. Model numerics live in the selfcheck suite.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "deepseek_moe_v41_load_internal.h"
#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include "test_deepseek_moe_v41_fixture.h"

namespace lsxcommon {
namespace gpu {
InferenceResult RunDeepSeekV41Inference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, ArrowTokenizerDb *tok_db,
    const deepseek_v41::gpu::PinnedWeights &pw,
    deepseek_v41::gpu::ExpertDb &expert_db, sample::ISampler &sampler,
    cudaStream_t stream, cublasHandle_t cublas);
} // namespace gpu
} // namespace lsxcommon

namespace fs = std::filesystem;

TEST_CASE("deepseek-v41 pipeline runs a synthetic full-geometry model",
          "[deepseek-v41][pipeline]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_pipeline";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticFullGeometry(dir, /*vocab=*/64,
                                                   /*experts=*/2)
                .ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);
    cublasHandle_t handle = nullptr;
    REQUIRE(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);

    lsxcommon::deepseek_v41::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::deepseek_v41::gpu::load_pinned_weights(
                dir.string(), pw, stream) == cudaSuccess);
    auto db =
        lsxcommon::deepseek_v41::gpu::load_expert_index(dir.string(), 1, 2);

    const std::vector<int32_t> ids = {0, 1, 2, 3};
    auto r = lsxcommon::gpu::RunDeepSeekV41Inference(
        dir.string(), ids, /*max_gen_tokens=*/2, /*tok_db=*/nullptr, pw, db,
        lsxcommon::sample::DefaultSampler(), stream, handle);
    INFO("pipeline error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.num_tokens >= 1);
    REQUIRE(r.final_logits.size() == 64);

    // Determinism: a second run produces the same tokens.
    auto r2 = lsxcommon::gpu::RunDeepSeekV41Inference(
        dir.string(), ids, /*max_gen_tokens=*/2, nullptr, pw, db,
        lsxcommon::sample::DefaultSampler(), stream, handle);
    REQUIRE(r2.ok);
    REQUIRE(r.gen_tokens == r2.gen_tokens);

    lsxcommon::deepseek_v41::gpu::free_expert_index(db);
    pw.Free();
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    fs::remove_all(dir);
}

TEST_CASE("deepseek-v41 pipeline runs a synthetic CED (ratio-1) layer",
          "[deepseek-v41][pipeline]")
{
    const auto dir = fs::temp_directory_path() / "lsx_v41_pipeline_ced";
    fs::remove_all(dir);
    REQUIRE(v41fixtest::BuildSyntheticCedGeometry(dir, /*vocab=*/64,
                                                  /*experts=*/2)
                .ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);
    cublasHandle_t handle = nullptr;
    REQUIRE(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);

    lsxcommon::deepseek_v41::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::deepseek_v41::gpu::load_pinned_weights(
                dir.string(), pw, stream) == cudaSuccess);
    auto db =
        lsxcommon::deepseek_v41::gpu::load_expert_index(dir.string(), 2, 2);

    const std::vector<int32_t> ids = {0, 1, 2, 3, 4};
    auto r = lsxcommon::gpu::RunDeepSeekV41Inference(
        dir.string(), ids, /*max_gen_tokens=*/3, /*tok_db=*/nullptr, pw, db,
        lsxcommon::sample::DefaultSampler(), stream, handle);
    INFO("pipeline error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.num_tokens >= 1);

    auto r2 = lsxcommon::gpu::RunDeepSeekV41Inference(
        dir.string(), ids, /*max_gen_tokens=*/3, nullptr, pw, db,
        lsxcommon::sample::DefaultSampler(), stream, handle);
    REQUIRE(r2.ok);
    REQUIRE(r.gen_tokens == r2.gen_tokens);

    lsxcommon::deepseek_v41::gpu::free_expert_index(db);
    pw.Free();
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    fs::remove_all(dir);
}
