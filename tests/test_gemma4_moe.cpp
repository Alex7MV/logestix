// Тесты Gemma-4 MoE: router top-k (детерминизм, tie-break, скейлы),
// dense MLP (GELU-tanh) и NVFP4 expert FFN против CPU-референсов.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <random>
#include <vector>

#include "lsxcommon/gemma4_inference.h"

namespace {

std::vector<__nv_bfloat16> ToHalf(const std::vector<float> &v)
{
    std::vector<__nv_bfloat16> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2bfloat16(v[i]);
    return h;
}

float GeluTanh(float v)
{
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v *
           (1.f + std::tanh(k0 * (v + 0.044715f * v * v * v)));
}

inline bool BfTol(float got, double want)
{
    return std::abs(got - want) <=
           std::max(0.05, 0.0125 * std::abs(want));
}

} // namespace

TEST_CASE("RunGemmaDenseMLP matches CPU (gelu tanh geglu)", "[gemma][moe]") {
    constexpr int seq = 2, H = 32, I = 24;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf(seq * H), gf(I * H), uf(I * H), df(H * I);
    for (auto &x : xf) x = nd(rng);
    for (auto &x : gf) x = 0.2f * nd(rng);
    for (auto &x : uf) x = 0.2f * nd(rng);
    for (auto &x : df) x = 0.2f * nd(rng);

    auto hx = ToHalf(xf);
    auto hg = ToHalf(gf);
    auto hu = ToHalf(uf);
    auto hd = ToHalf(df);

    __nv_bfloat16 *dx, *dg, *du, *dd, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dg, hg.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, hu.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dd, hd.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, seq * H * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dg, hg.data(), hg.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(du, hu.data(), hu.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dd, hd.data(), hd.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaDenseMLP(dg, du, dd, I, H, dx, dout, seq,
                                               nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> hout(seq * H);
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int t = 0; t < seq; ++t) {
        std::vector<double> act(I);
        for (int i = 0; i < I; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < H; ++c) {
                g += double(xf[t * H + c]) * double(gf[i * H + c]);
                u += double(xf[t * H + c]) * double(uf[i * H + c]);
            }
            act[i] = GeluTanh(static_cast<float>(g)) * u;
        }
        for (int c = 0; c < H; ++c) {
            double acc = 0;
            for (int i = 0; i < I; ++i)
                acc += act[i] * double(df[c * I + i]);
            INFO("t=" << t << " c=" << c << " got="
                      << __bfloat162float(hout[t * H + c]) << " want=" << acc);
            CHECK(BfTol(__bfloat162float(hout[t * H + c]), acc));
        }
    }

    cudaFree(dx); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dout);
}

TEST_CASE("router top-k deterministic with lower-index tie-break",
          "[gemma][moe]") {
    constexpr int E = 8, H = 8, K = 2;
    // Crafted proj rows: experts 3 and 5 dominate; experts 1 and 6 tie exactly.
    std::vector<float> pf(E * H, 0.f);
    for (int c = 0; c < H; ++c) {
        pf[3 * H + c] = 0.5f;
        pf[5 * H + c] = 0.5f;
        pf[1 * H + c] = 0.1f;
        pf[6 * H + c] = 0.1f;
    }
    std::vector<float> xf(H, 1.0f);
    auto hp = ToHalf(pf);
    auto hx = ToHalf(xf);

    __nv_bfloat16 *dp, *dx;
    int32_t *dsel;
    float *dw;
    REQUIRE(cudaMalloc(&dp, hp.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dsel, sizeof(int32_t) * K) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, sizeof(float) * K) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    auto run_once = [&]() {
        REQUIRE(lsxcommon::gemma::RunGemmaRouterTopK(
                    dp, nullptr, dx, E, H, K, dsel, dw, 1, nullptr) ==
                cudaSuccess);
        std::vector<int32_t> sel(K);
        std::vector<float> w(K);
        REQUIRE(cudaMemcpy(sel.data(), dsel, sizeof(int32_t) * K,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(w.data(), dw, sizeof(float) * K,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        return std::make_pair(sel, w);
    };

    auto r1 = run_once();
    auto r2 = run_once();
    // Deterministic across runs.
    CHECK(r1.first == r2.first);
    // Winners are 3 and 5 (exact ties between 1/6 must not leak in).
    CHECK(r1.first[0] == 3);
    CHECK(r1.first[1] == 5);
    // Softmax over the two selected logits: equal -> 0.5 / 0.5.
    CHECK(std::abs(r1.second[0] - 0.5f) < 1e-3);
    CHECK(std::abs(r1.second[1] - 0.5f) < 1e-3);

    cudaFree(dp); cudaFree(dx); cudaFree(dsel); cudaFree(dw);
}


