// Muse-Glimmer-30B generation pipeline test: the production
// RunMuseDenseGeneration on a synthetic converted model — determinism across
// repeats, streaming order, cooperative cancel. Synthetic geometry only.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "test_muse_dense_fixture.h"

#include "lsxcommon/muse_dense_convert.h"
#include "lsxcommon/muse_dense_inference.h"
#include "lsxcommon/muse_dense_load.h"
#include "lsxcommon/token_stream.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeConvertedSynthetic()
{
    ::setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    auto tmp = fs::temp_directory_path() / "muse_dense_pipeline_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    synthtest::WriteSafetensors(tmp, "model.safetensors",
                                muse_dense_test::BuildSyntheticMuseDenseModel());
    muse_dense_test::WriteConfig(tmp);
    muse_dense_test::WriteTokenizerJson(tmp);
    auto out = tmp / "out";
    const auto st = lsxcommon::muse_dense::ConvertMuseDenseModel(
        tmp.string(), out.string());
    if (!st.ok()) FAIL("ConvertMuseDenseModel: " << st.ToString());
    return out;
}

struct Loaded {
    lsxcommon::muse_dense::MuseDensePinnedWeights pw;
    lsxcommon::muse_dense::MuseDenseKvCache kv;
    cudaStream_t stream = nullptr;
    explicit Loaded(const fs::path &out)
    {
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
        REQUIRE(lsxcommon::muse_dense::LoadMuseDensePinnedWeights(
                    out.string(), pw, stream) == cudaSuccess);
        REQUIRE(kv.Allocate(pw, 64) == cudaSuccess);
    }
    ~Loaded()
    {
        kv.Free();
        pw.Free();
        if (stream) cudaStreamDestroy(stream);
    }
};

} // namespace

TEST_CASE("muse pipeline generates deterministically", "[muse_dense][pipeline]") {
    auto out = MakeConvertedSynthetic();
    Loaded L(out);
    const std::vector<int32_t> prompt = {259, 5, 10, 20};
    const std::vector<int> eos = {260, 261};

    lsxcommon::muse_dense::MuseDenseGenerationResult a, b;
    REQUIRE(lsxcommon::muse_dense::RunMuseDenseGeneration(
                L.stream, L.pw, prompt, 4, eos, a, L.kv, nullptr) == cudaSuccess);
    REQUIRE(a.ok);
    REQUIRE_FALSE(a.tokens.empty());
    CHECK(a.tokens.size() <= 4);
    CHECK(a.final_logits.size() == static_cast<size_t>(L.pw.vocab_size));

    // A second identical run is bit-exact (the determinism contract).
    REQUIRE(lsxcommon::muse_dense::RunMuseDenseGeneration(
                L.stream, L.pw, prompt, 4, eos, b, L.kv, nullptr) == cudaSuccess);
    REQUIRE(b.ok);
    CHECK(b.tokens == a.tokens);
    REQUIRE(b.final_logits.size() == a.final_logits.size());
    CHECK(std::memcmp(b.final_logits.data(), a.final_logits.data(),
                      a.final_logits.size() * sizeof(float)) == 0);
    fs::remove_all(out.parent_path());
}

TEST_CASE("muse pipeline streams one Emit per token in order",
          "[muse_dense][pipeline]") {
    auto out = MakeConvertedSynthetic();
    Loaded L(out);
    const std::vector<int32_t> prompt = {259, 5, 10};
    const std::vector<int> eos = {260, 261};
    std::vector<uint32_t> emitted;
    lsxcommon::TokenCallback cb = [&](const lsxcommon::TokenChunk &c) {
        emitted.push_back(c.token_id);
    };
    lsxcommon::GenerationStream gs;
    gs.on_token = &cb;
    lsxcommon::muse_dense::MuseDenseGenerationResult r;
    REQUIRE(lsxcommon::muse_dense::RunMuseDenseGeneration(
                L.stream, L.pw, prompt, 3, eos, r, L.kv, nullptr,
                lsxcommon::sample::DefaultSampler(), &gs) == cudaSuccess);
    REQUIRE(r.ok);
    REQUIRE(emitted.size() == r.tokens.size());
    for (size_t i = 0; i < emitted.size(); ++i)
        CHECK(emitted[i] == static_cast<uint32_t>(r.tokens[i]));
    fs::remove_all(out.parent_path());
}

TEST_CASE("muse pipeline honours a pre-set abort", "[muse_dense][pipeline]") {
    auto out = MakeConvertedSynthetic();
    Loaded L(out);
    const std::vector<int32_t> prompt = {259, 5, 10};
    const std::vector<int> eos = {260, 261};
    std::atomic<bool> abort{true};
    lsxcommon::GenerationStream gs;
    gs.abort_flag = &abort;
    lsxcommon::muse_dense::MuseDenseGenerationResult r;
    const auto rc = lsxcommon::muse_dense::RunMuseDenseGeneration(
        L.stream, L.pw, prompt, 8, eos, r, L.kv, nullptr,
        lsxcommon::sample::DefaultSampler(), &gs);
    // Prefill emits the first token; the decode loop must stop immediately.
    CHECK(rc == cudaSuccess);
    CHECK(r.tokens.size() <= 1);
    fs::remove_all(out.parent_path());
}