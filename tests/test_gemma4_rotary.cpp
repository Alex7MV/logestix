// Тест Gemma-4 partial RoPE против CPU FP64-референса (HF rotate_half).
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <random>
#include <vector>

#include "lsxcommon/gemma4_inference.h"

namespace {
inline bool BfTol(float got, double want)
{
    return std::abs(got - want) <=
           std::max(0.03, 0.01 * std::abs(want));
}
} // namespace

TEST_CASE("RunGemmaPartialRope matches CPU FP64", "[gemma][kernels]") {
    constexpr int seq = 4, heads = 2, head_dim = 16, rotary_dim = 8;
    constexpr double theta = 1000000.0;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(seq * heads * head_dim);
    for (auto &v : x) v = nd(rng);

    // cos/sin tables [seq][rotary_dim/2].
    const int half = rotary_dim / 2;
    std::vector<double> cosv(seq * half), sinv(seq * half);
    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < half; ++d) {
            double inv =
                1.0 / std::pow(theta, static_cast<double>(2 * d) / rotary_dim);
            double ang = static_cast<double>(t + 1) * inv; // pos = t+1
            cosv[t * half + d] = std::cos(ang);
            sinv[t * half + d] = std::sin(ang);
        }

    // HF rotate_half reference: pair (j, j+half), pass-through the tail.
    std::vector<float> ref(x.size());
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const int64_t row = (static_cast<int64_t>(t) * heads + h) * head_dim;
            for (int j = 0; j < half; ++j) {
                float c = static_cast<float>(cosv[t * half + j]);
                float s = static_cast<float>(sinv[t * half + j]);
                float x1 = x[row + j];
                float x2 = x[row + j + half];
                ref[row + j] = x1 * c - x2 * s;
                ref[row + j + half] = x2 * c + x1 * s;
            }
            for (int p = rotary_dim; p < head_dim; ++p)
                ref[row + p] = x[row + p];
        }

    std::vector<__nv_bfloat16> hx(x.size());
    for (size_t i = 0; i < x.size(); ++i) hx[i] = __float2bfloat16(x[i]);
    std::vector<__nv_bfloat16> hcos(seq * half), hsin(seq * half);
    for (int i = 0; i < seq * half; ++i) {
        hcos[i] = __float2bfloat16(static_cast<float>(cosv[i]));
        hsin[i] = __float2bfloat16(static_cast<float>(sinv[i]));
    }
    std::vector<int32_t> pos(seq);
    for (int t = 0; t < seq; ++t) pos[t] = t + 1;

    __nv_bfloat16 *dx, *dcos, *dsin, *dout;
    int32_t *dpos;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dcos, hcos.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dsin, hsin.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dpos, pos.size() * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dcos, hcos.data(), hcos.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dsin, hsin.data(), hsin.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dpos, pos.data(), pos.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaRopeCosSin(dpos, dcos, dsin, seq,
                                                 rotary_dim, theta,
                                                 nullptr) == cudaSuccess);
    REQUIRE(lsxcommon::gemma::RunGemmaPartialRope(dx, dcos, dsin, dout, seq,
                                                  heads, head_dim, rotary_dim,
                                                  nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> hout(hx.size());
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < ref.size(); ++i) {
        INFO("i=" << i << " got=" << __bfloat162float(hout[i]) << " want="
                  << ref[i]);
        CHECK(BfTol(__bfloat162float(hout[i]), ref[i]));
    }

    cudaFree(dx);
    cudaFree(dout);
    cudaFree(dcos);
    cudaFree(dsin);
    cudaFree(dpos);
}


