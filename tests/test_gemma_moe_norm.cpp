// Gemma-4 RMSNorm test (w-form: output = rms(x) * w) against a CPU FP64 reference.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <random>
#include <vector>

#include "lsxcommon/gemma_moe_inference.h"

namespace {
inline bool BfTol(float got, double want)
{
    // BF16 storage: 8-bit mantissa => relative ulp ~2^-8.
    return std::abs(got - want) <=
           std::max(0.03, 0.01 * std::abs(want));
}
} // namespace

TEST_CASE("RunGemmaRMSNorm matches CPU FP64 (w-form)", "[gemma][kernels]") {
    constexpr int rows = 3, hidden = 64;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(rows * hidden), w(hidden);
    for (auto &v : x) v = nd(rng);
    for (auto &v : w) v = 0.5f * nd(rng);
    const float eps = 1e-6f;

    // CPU FP64 reference.
    std::vector<double> inv(rows);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int c = 0; c < hidden; ++c)
            ss += double(x[r * hidden + c]) * double(x[r * hidden + c]);
        inv[r] = 1.0 / std::sqrt(ss / hidden + eps);
    }

    std::vector<__nv_bfloat16> hx(rows * hidden), hw(hidden);
    for (int i = 0; i < rows * hidden; ++i) hx[i] = __float2bfloat16(x[i]);
    for (int i = 0; i < hidden; ++i) hw[i] = __float2bfloat16(w[i]);

    __nv_bfloat16 *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaRMSNorm(dx, dw, dout, rows, hidden, eps,
                                              nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> hout(rows * hidden);
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < hidden; ++c) {
            // Gemma-4 authentic HF semantics: plain weight multiply (no +1).
            double want = double(x[r * hidden + c]) * inv[r] *
                          double(w[c]);
            double got = double(__bfloat162float(hout[r * hidden + c]));
            INFO("r=" << r << " c=" << c << " got=" << got << " want=" << want);
            CHECK(BfTol(got, want));
        }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

TEST_CASE("RunGemmaRMSNorm handles multi-block-hidden rows", "[gemma][kernels]") {
    constexpr int rows = 2, hidden = 1024; // > blockDim (256): grid-stride path
    std::vector<__nv_bfloat16> hx(rows * hidden), hw(hidden), hout(rows * hidden);
    for (int i = 0; i < rows * hidden; ++i)
        hx[i] = __float2bfloat16(((i % 17) - 8) * 0.25f);
    for (int i = 0; i < hidden; ++i)
        hw[i] = __float2bfloat16(0.01f * (i % 7));

    __nv_bfloat16 *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaRMSNorm(dx, dw, dout, rows, hidden,
                                              1e-6f, nullptr) == cudaSuccess);
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int c = 0; c < hidden; ++c)
            ss += double(__bfloat162float(hx[r * hidden + c])) *
                  double(__bfloat162float(hx[r * hidden + c]));
        double inv = 1.0 / std::sqrt(ss / hidden + 1e-6);
        for (int c = 0; c < hidden; ++c) {
            double want = double(__bfloat162float(hx[r * hidden + c])) * inv *
                          double(__bfloat162float(hw[c]));
            CHECK(BfTol(__bfloat162float(hout[r * hidden + c]), want));
        }
    }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

// Decode is one row per norm and gemma-4 runs 211 weighted RMSNorms per token,
// so the per-launch latency of the single-row kernel is a first-order decode
// cost (second only to the dense GEMV family). The 256-thread launch measures
// ~4.2 us at hidden=2816; the latency is the per-thread grid-stride chain, not
// bandwidth, so a wider block shortens it. Pin the single-row latency.
TEST_CASE("gemma RMSNorm decode latency budget", "[gemma][kernels][budget]")
{
#if defined(NDEBUG)
    constexpr int H = 2816;
    constexpr int kIters = 2000;
    std::vector<__nv_bfloat16> hx(H, __float2bfloat16(0.5f));
    std::vector<__nv_bfloat16> hw(H, __float2bfloat16(1.0f));
    __nv_bfloat16 *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, H * 2) == cudaSuccess);
    cudaMemcpy(dx, hx.data(), H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw.data(), H * 2, cudaMemcpyHostToDevice);
    for (int i = 0; i < 50; ++i)
        REQUIRE(lsxcommon::gemma::RunGemmaRMSNorm(dx, dw, dout, 1, H, 1e-6f,
                                                  0) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    REQUIRE(cudaEventRecord(e0) == cudaSuccess);
    for (int i = 0; i < kIters; ++i)
        REQUIRE(lsxcommon::gemma::RunGemmaRMSNorm(dx, dw, dout, 1, H, 1e-6f,
                                                  0) == cudaSuccess);
    REQUIRE(cudaEventRecord(e1) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
    const double us = static_cast<double>(ms) * 1000.0 / kIters;
    INFO("single-row gemma RMSNorm latency = " << us << " us");
    // 256-thread launch ~4.2 us; a 512-thread launch ~2.3 us. Budget between.
    CHECK(us < 3.4);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
#endif
}

