#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <atomic>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <thread>
#include <vector>

#include "lsxcommon/core.h"
#include "lsxcommon/sampling.h"

using lsxcommon::sample::GreedySampler;
using lsxcommon::sample::LogitType;

namespace {

int32_t HostArgmax(const std::vector<float>& row) {
    int32_t best_i = 0;
    float best = row[0];
    for (int i = 1; i < static_cast<int>(row.size()); ++i) {
        if (row[i] > best) { best = row[i]; best_i = i; }
    }
    return best_i;
}

}  // namespace

TEST_CASE("GreedySampler selects first-maximum on ties", "[sampling]") {
    const std::vector<float> row = {1.0f, -5.0f, 3.0f, 3.0f, 2.0f, -1.0f};
    const int vocab = static_cast<int>(row.size());

    float* d_row = nullptr;
    REQUIRE(cudaMalloc(&d_row, vocab * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_row, row.data(), vocab * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    int32_t* d_tok = nullptr;
    float* d_maxv = nullptr;
    REQUIRE(cudaMalloc(&d_tok, sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_maxv, sizeof(float)) == cudaSuccess);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GreedySampler g;
    REQUIRE(g.Sample(stream, d_row, vocab, LogitType::kF32, d_tok, d_maxv)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    int32_t tok = -1;
    float maxv = 0.0f;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost)
            == cudaSuccess);

    REQUIRE(tok == HostArgmax(row));
    REQUIRE(tok == 2);       // FIRST 3.0, index 2
    REQUIRE(maxv == Catch::Approx(3.0f));

    cudaStreamDestroy(stream);
    cudaFree(d_maxv);
    cudaFree(d_tok);
    cudaFree(d_row);
}

TEST_CASE("GreedySampler handles FP16 rows", "[sampling]") {
    const std::vector<float> row = {0.5f, 100.0f, -2.0f, 42.0f};
    const int vocab = static_cast<int>(row.size());

    std::vector<__half> half_row(row.size());
    for (int i = 0; i < vocab; ++i) half_row[i] = __float2half(row[i]);

    float* d_row = nullptr;
    REQUIRE(cudaMalloc(&d_row, vocab * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_row, half_row.data(), vocab * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    int32_t* d_tok = nullptr;
    float* d_maxv = nullptr;
    REQUIRE(cudaMalloc(&d_tok, sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_maxv, sizeof(float)) == cudaSuccess);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GreedySampler g;
    REQUIRE(g.Sample(stream, d_row, vocab, LogitType::kF16, d_tok, d_maxv)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    int32_t tok = -1;
    float maxv = 0.0f;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost)
            == cudaSuccess);

    REQUIRE(tok == 1);
    REQUIRE(maxv == Catch::Approx(100.0f));

    cudaStreamDestroy(stream);
    cudaFree(d_maxv);
    cudaFree(d_tok);
    cudaFree(d_row);
}

TEST_CASE("GreedySampler rejects invalid arguments", "[sampling]") {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GreedySampler g;
    float d_stub = 0.0f;
    int32_t d_tok = 0;
    REQUIRE(g.Sample(stream, nullptr, 10, LogitType::kF32, &d_tok, &d_stub)
            == cudaErrorInvalidValue);
    REQUIRE(g.Sample(stream, &d_stub, 0, LogitType::kF32, &d_tok, &d_stub)
            == cudaErrorInvalidValue);
    REQUIRE(g.Sample(stream, &d_stub, 10, static_cast<LogitType>(7),
                     &d_tok, &d_stub) == cudaErrorInvalidValue);
    cudaStreamDestroy(stream);
}

// Two concurrent Sample() calls on independent streams must not corrupt each
// other through shared sampler scratch. Large vocab => many partial blocks =>
// a real overlap window; each stream's argmax is distinct, so any cross-talk
// yields the wrong token.
TEST_CASE("GreedySampler is safe across concurrent streams", "[sampling]") {
    const int vocab = 1 << 20;
    std::vector<float> row_a(vocab, -10.0f);
    std::vector<float> row_b(vocab, -10.0f);
    row_a[7] = 5.0f;
    row_b[vocab - 3] = 6.0f;

    float* d_a = nullptr;
    float* d_b = nullptr;
    REQUIRE(cudaMalloc(&d_a, vocab * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_b, vocab * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_a, row_a.data(), vocab * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_b, row_b.data(), vocab * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    std::atomic<int> wrong_a{0};
    std::atomic<int> wrong_b{0};
    auto worker = [&](float* d_row, int32_t expect, std::atomic<int>& wrong) {
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) { wrong += 1000; return; }
        int32_t* d_tok = nullptr;
        float* d_maxv = nullptr;
        cudaMalloc(&d_tok, sizeof(int32_t));
        cudaMalloc(&d_maxv, sizeof(float));
        GreedySampler g;
        for (int it = 0; it < 300; ++it) {
            if (g.Sample(s, d_row, vocab, LogitType::kF32, d_tok, d_maxv) !=
                cudaSuccess) {
                ++wrong;
                break;
            }
            cudaStreamSynchronize(s);
            int32_t tok = -1;
            cudaMemcpy(&tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (tok != expect) ++wrong;
        }
        cudaFree(d_maxv);
        cudaFree(d_tok);
        cudaStreamDestroy(s);
    };

    std::thread ta(worker, d_a, /*expect=*/7, std::ref(wrong_a));
    std::thread tb(worker, d_b, /*expect=*/vocab - 3, std::ref(wrong_b));
    ta.join();
    tb.join();

    REQUIRE(wrong_a.load() == 0);
    REQUIRE(wrong_b.load() == 0);

    cudaFree(d_b);
    cudaFree(d_a);
}

TEST_CASE("cudaFreeIgnoreShutdown tolerates null and frees real memory",
          "[sampling][core]")
{
    lsxcommon::cudaFreeIgnoreShutdown(nullptr);  // no-op
    void *p = nullptr;
    REQUIRE(cudaMalloc(&p, 64) == cudaSuccess);
    lsxcommon::cudaFreeIgnoreShutdown(p);
    CHECK(cudaGetLastError() == cudaSuccess);
}

TEST_CASE("DefaultSampler is a stable singleton", "[sampling]") {
    auto& a = lsxcommon::sample::DefaultSampler();
    auto& b = lsxcommon::sample::DefaultSampler();
    REQUIRE(&a == &b);
}