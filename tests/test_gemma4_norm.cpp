// Тест Gemma-4 RMSNorm (offset-weight форма) против CPU FP64-референса.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <random>
#include <vector>

#include "lsxcommon/gemma4_inference.h"

TEST_CASE("RunGemmaRMSNorm matches CPU FP64 (1+w offset)", "[gemma][kernels]") {
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

    std::vector<__half> hx(rows * hidden), hw(hidden);
    for (int i = 0; i < rows * hidden; ++i) hx[i] = __float2half(x[i]);
    for (int i = 0; i < hidden; ++i) hw[i] = __float2half(w[i]);

    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaRMSNorm(dx, dw, dout, rows, hidden, eps,
                                              nullptr) == cudaSuccess);

    std::vector<__half> hout(rows * hidden);
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < hidden; ++c) {
            double want = double(x[r * hidden + c]) * inv[r] *
                          (1.0 + double(w[c]));
            double got = double(__half2float(hout[r * hidden + c]));
            INFO("r=" << r << " c=" << c << " got=" << got << " want=" << want);
            CHECK(std::abs(got - want) < 4e-3);
        }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

TEST_CASE("RunGemmaRMSNorm handles multi-block-hidden rows", "[gemma][kernels]") {
    constexpr int rows = 2, hidden = 1024; // > blockDim (256): grid-stride path
    std::vector<__half> hx(rows * hidden), hw(hidden), hout(rows * hidden);
    for (int i = 0; i < rows * hidden; ++i)
        hx[i] = __float2half(((i % 17) - 8) * 0.25f);
    for (int i = 0; i < hidden; ++i)
        hw[i] = __float2half(0.01f * (i % 7));

    __half *dx, *dw, *dout;
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
            ss += double(__half2float(hx[r * hidden + c])) *
                  double(__half2float(hx[r * hidden + c]));
        double inv = 1.0 / std::sqrt(ss / hidden + 1e-6);
        for (int c = 0; c < hidden; ++c) {
            double want = double(__half2float(hx[r * hidden + c])) * inv *
                          (1.0 + double(__half2float(hw[c])));
            CHECK(std::abs(double(__half2float(hout[r * hidden + c])) - want) <
                  1e-3);
        }
    }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}
