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
using lsxcommon::sample::SamplingParams;
using lsxcommon::sample::SamplingSampler;

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


namespace {

struct RowDevice {
    float *d = nullptr;
    int n = 0;
    explicit RowDevice(const std::vector<float> &row) : n((int)row.size()) {
        REQUIRE(cudaMalloc(&d, row.size() * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, row.data(), row.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~RowDevice() { cudaFree(d); }
};

int32_t Draw(SamplingSampler &s, cudaStream_t st, float *d_row, int vocab,
             int32_t *d_tok, float *d_maxv) {
    REQUIRE(s.Sample(st, d_row, vocab, LogitType::kF32, d_tok, d_maxv) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    int32_t tok = -1;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(tok), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return tok;
}

}  // namespace

TEST_CASE("SamplerFor is greedy by default and reconfigures per request",
          "[sampling]") {
    SamplingParams greedy;
    REQUIRE(&lsxcommon::sample::SamplerFor(greedy) ==
            &lsxcommon::sample::DefaultSampler());

    SamplingParams p;
    p.temperature = 0.8f;
    p.seed = 42;
    auto &a = lsxcommon::sample::SamplerFor(p);
    REQUIRE(&a != &lsxcommon::sample::DefaultSampler());

    RowDevice row({2.0f, 1.0f, 0.0f, -1.0f});
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));
    const int32_t first = Draw(*static_cast<SamplingSampler *>(&a), st, row.d,
                               4, d_tok, d_maxv);
    auto &b = lsxcommon::sample::SamplerFor(p);  // same seed, fresh sequence
    const int32_t again = Draw(*static_cast<SamplingSampler *>(&b), st, row.d,
                               4, d_tok, d_maxv);
    REQUIRE(first == again);
    cudaFree(d_maxv);
    cudaFree(d_tok);
    cudaStreamDestroy(st);
}

TEST_CASE("temperature sampling follows the softmax distribution",
          "[sampling]") {
    const std::vector<float> row = {0.0f, 0.5f, 1.0f, 1.5f};
    RowDevice dev(row);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    SamplingParams p;
    p.temperature = 1.0f;
    p.seed = 7;
    SamplingSampler s(p);
    std::vector<int> counts(row.size(), 0);
    constexpr int kDraws = 8000;
    for (int i = 0; i < kDraws; ++i)
        ++counts[Draw(s, st, dev.d, (int)row.size(), d_tok, d_maxv)];

    double z = 0.0;
    for (float v : row) z += std::exp(v);
    for (size_t i = 0; i < row.size(); ++i) {
        const double p_expect = std::exp(row[i]) / z;
        const double p_got = double(counts[i]) / kDraws;
        INFO("token " << i << " expected " << p_expect << " got " << p_got);
        CHECK(std::fabs(p_got - p_expect) < 0.03);
    }
    cudaFree(d_maxv); cudaFree(d_tok); cudaStreamDestroy(st);
}

TEST_CASE("top_k keeps exactly the k most likely tokens", "[sampling]") {
    // Distinct logits; the three largest are indices 6, 2 and 9.
    std::vector<float> row(24, -8.0f);
    row[6] = 8.0f; row[2] = 7.0f; row[9] = 6.5f;
    RowDevice dev(row);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 3;
    p.seed = 11;
    SamplingSampler s(p);
    int counts[24] = {0};
    for (int i = 0; i < 3000; ++i)
        ++counts[Draw(s, st, dev.d, 24, d_tok, d_maxv)];
    CHECK(counts[6] > 0);
    CHECK(counts[2] > 0);
    CHECK(counts[9] > 0);
    for (int i = 0; i < 24; ++i)
        if (i != 6 && i != 2 && i != 9) CHECK(counts[i] == 0);
    cudaFree(d_maxv); cudaFree(d_tok); cudaStreamDestroy(st);
}

TEST_CASE("top_p nucleus keeps the dominant token", "[sampling]") {
    std::vector<float> row(16, -20.0f);
    row[4] = 10.0f;
    RowDevice dev(row);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    SamplingParams p;
    p.temperature = 1.0f;
    p.top_p = 0.9f;   // the dominant token carries ~1 - 16*e^-30 of the mass
    p.seed = 3;
    SamplingSampler s(p);
    int other = 0;
    for (int i = 0; i < 500; ++i)
        if (Draw(s, st, dev.d, 16, d_tok, d_maxv) != 4) ++other;
    CHECK(other == 0);
    cudaFree(d_maxv); cudaFree(d_tok); cudaStreamDestroy(st);
}

TEST_CASE("sampling is reproducible for a seed and differs across seeds",
          "[sampling]") {
    const std::vector<float> row = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    RowDevice dev(row);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    SamplingParams p;
    p.temperature = 1.0f;
    p.seed = 12345;
    SamplingSampler a(p), b(p);
    p.seed = 999;
    SamplingSampler c(p);
    std::vector<int32_t> sa, sb, sc;
    for (int i = 0; i < 32; ++i) {
        sa.push_back(Draw(a, st, dev.d, 6, d_tok, d_maxv));
        sb.push_back(Draw(b, st, dev.d, 6, d_tok, d_maxv));
        sc.push_back(Draw(c, st, dev.d, 6, d_tok, d_maxv));
    }
    CHECK(sa == sb);
    CHECK(sa != sc);
    cudaFree(d_maxv); cudaFree(d_tok); cudaStreamDestroy(st);
}

TEST_CASE("SamplingSampler handles FP16 rows and rejects greedy calls",
          "[sampling]") {
    std::vector<__half> hrow(8);
    for (int i = 0; i < 8; ++i) hrow[i] = __float2half(float(i));
    __half *d_row = nullptr;
    REQUIRE(cudaMalloc(&d_row, hrow.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_row, hrow.data(), hrow.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    SamplingParams p;
    REQUIRE(SamplingSampler(p).Sample(st, d_row, 8, LogitType::kF16, d_tok,
                                      d_maxv) == cudaErrorInvalidValue);
    p.temperature = 0.7f;
    SamplingSampler s(p);
    REQUIRE(s.Sample(st, d_row, 8, LogitType::kF16, d_tok, d_maxv) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    int32_t tok = -1;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(tok), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(tok >= 0);
    CHECK(tok < 8);
    cudaFree(d_maxv); cudaFree(d_tok); cudaFree(d_row); cudaStreamDestroy(st);
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