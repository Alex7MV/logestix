#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <vector>

#include <cuda_runtime.h>

#include <lsxcommon/qwen_dense_inference_pipeline.h>
#include <lsxcommon/lsx_attn_inference.h>
#include <lsxcommon/qwen_inference_load.h>

#include <spdlog/spdlog.h>

#include "test_qwen_dense_fixture.h"

using namespace std::filesystem;

// Local bit-exactness gate for the incremental path (spec item 15): with the
// same synthetic weights, RunQwenGeneration(kv = nullptr) is the untouched
// full-recompute path, and RunQwenGeneration(kv = &cache) runs the new
// one-token-per-step incremental path. Both must produce identical tokens and
// bit-identical final logits.
TEST_CASE("Qwen incremental KV cache is bit-identical to full recompute")
{
    auto tmp = temp_directory_path() / "qwen_kv_parity";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");
    REQUIRE(lsxcommon::qwen::ConvertQwenModel(tmp.string(), tmp.string()).ok());

    lsxcommon::qwen::QwenPinnedWeights pw;
    cudaStream_t st = nullptr;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::LoadQwenPinnedWeights(tmp.string(), pw, st));
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    const int eos_id = 2;
    const std::vector<int32_t> prompt = {0, 1, 2, 3};
    const int max_new = 6;

    // Full-recompute baseline (kv == nullptr).
    lsxcommon::qwen::QwenGenerationResult full;
    REQUIRE(lsxcommon::qwen::RunQwenGeneration(
                st, pw, prompt, max_new, eos_id, full) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    REQUIRE(full.ok);
    REQUIRE(full.tokens.size() == max_new);

    // Incremental session (kv != nullptr): same single call.
    lsxcommon::qwen::QwenKvCache cache;
    lsxcommon::qwen::QwenGenerationResult incr;
    REQUIRE(lsxcommon::qwen::RunQwenGeneration(
                st, pw, prompt, max_new, eos_id, incr,
                lsxcommon::sample::DefaultSampler(), &cache) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    REQUIRE(incr.ok);
    REQUIRE(incr.tokens.size() == max_new);

    REQUIRE(incr.tokens == full.tokens);
    REQUIRE(incr.final_logits.size() == full.final_logits.size());
    if (incr.final_logits.size() == full.final_logits.size()) {
        for (size_t i = 0; i < full.final_logits.size(); ++i) {
            // Bit-exact: both paths run the same FP32 ops in the same order.
            REQUIRE(incr.final_logits[i] == full.final_logits[i]);
        }
    }
    REQUIRE(cache.kv_rows == static_cast<int>(prompt.size()) + max_new - 1);

    // A continuation split (3 + 3) must reproduce the same 6-token prefix.
    lsxcommon::qwen::QwenKvCache cache2;
    lsxcommon::qwen::QwenGenerationResult a, b;
    REQUIRE(lsxcommon::qwen::RunQwenGeneration(
                st, pw, prompt, 3, eos_id, a,
                lsxcommon::sample::DefaultSampler(), &cache2) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    REQUIRE(a.tokens.size() == 3);

    const std::vector<int32_t> cont = {cache2.last_tok};
    REQUIRE(lsxcommon::qwen::RunQwenGeneration(
                st, pw, cont, 3, eos_id, b,
                lsxcommon::sample::DefaultSampler(), &cache2) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    std::vector<int32_t> joined = a.tokens;
    joined.insert(joined.end(), b.tokens.begin(), b.tokens.end());
    REQUIRE(joined == full.tokens);

    cache.Free();
    cache2.Free();
    pw.Free();
    cudaStreamDestroy(st);
    remove_all(tmp);
}
