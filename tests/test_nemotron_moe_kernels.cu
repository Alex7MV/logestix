// TDD — Nemotron-3.5 Mamba-2 kernel tests (Model Ingestion Protocol step 5).
// Device kernels are validated against the host reference in
// nemotron_moe_device_math.h.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nemotron_moe_device_math.h"
#include "nemotron_moe_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

using namespace lsxcommon::nemotron_moe;

namespace {

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> FromDevice(const T *d, size_t n)
{
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

} // namespace

TEST_CASE("mamba conv step matches the causal depthwise conv", "[nemotron][kernels]")
{
    const int C = 4, K = 4;
    const std::vector<float> xBC = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w(C * K), bias(C, 0.0f);
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k) w[c * K + k] = 0.1f * (c + 1) * (k + 1);
    std::vector<float> state0(C * (K - 1), 0.0f);

    float *d_x = ToDevice(xBC), *d_w = ToDevice(w), *d_b = ToDevice(bias);
    float *d_s0 = ToDevice(state0);
    float *d_s1 = nullptr, *d_o = nullptr;
    cudaMalloc(&d_s1, state0.size() * sizeof(float));
    cudaMalloc(&d_o, C * sizeof(float));

    RunNemotronConvStep(d_x, d_w, d_b, d_s0, d_s1, d_o, C, K, 0);
    cudaDeviceSynchronize();

    auto out = FromDevice(d_o, C);
    auto st1 = FromDevice(d_s1, state0.size());

    for (int c = 0; c < C; ++c) {
        float acc = w[c * K + (K - 1)] * xBC[c];  // history zeros
        const float expect = acc / (1.0f + std::exp(-acc));
        CHECK(out[c] == Catch::Approx(expect).margin(1e-6f));
    }
    // State shifts in the newest input.
    CHECK(st1[(K - 1) - 1] == Catch::Approx(xBC[0]));
    CHECK(st1[2 * (K - 1) + (K - 2)] == Catch::Approx(xBC[2]));

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_s0); cudaFree(d_s1); cudaFree(d_o);
}

TEST_CASE("selective state update matches the host reference", "[nemotron][kernels]")
{
    const int heads = 2, head_dim = 4, groups = 1, state_size = 3;
    std::vector<float> state0(heads * head_dim * state_size, 0.0f);
    for (size_t i = 0; i < state0.size(); ++i) state0[i] = 0.01f * (i % 7);
    std::vector<float> xs(heads * head_dim);
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = 0.1f * (i + 1);
    std::vector<float> B(groups * state_size), Cc(groups * state_size);
    for (int i = 0; i < state_size; ++i) { B[i] = 0.2f * (i + 1); Cc[i] = 0.3f * (i + 1); }
    std::vector<float> D = {1.0f, 0.5f};
    std::vector<float> dt = {0.1f, -0.2f};
    std::vector<float> dt_bias = {0.05f, 0.0f};
    std::vector<float> A_log = {0.0f, std::log(2.0f)};

    std::vector<float> ref_state(state0.size()), ref_out(heads * head_dim);
    ref::SelectiveStateUpdate(state0.data(), xs.data(), B.data(), Cc.data(),
                              D.data(), dt.data(), dt_bias.data(),
                              A_log.data(), heads, head_dim, groups, state_size,
                              ref_state.data(), ref_out.data());

    float *d_s0 = ToDevice(state0), *d_xs = ToDevice(xs), *d_B = ToDevice(B);
    float *d_C = ToDevice(Cc), *d_D = ToDevice(D), *d_dt = ToDevice(dt);
    float *d_db = ToDevice(dt_bias), *d_Al = ToDevice(A_log);
    float *d_s1 = nullptr, *d_o = nullptr;
    cudaMalloc(&d_s1, state0.size() * sizeof(float));
    cudaMalloc(&d_o, ref_out.size() * sizeof(float));

    RunNemotronSelectiveStateStep(d_s0, d_xs, d_B, d_C, d_D, d_dt, d_db, d_Al,
                                  heads, head_dim, groups, state_size, d_s1, d_o,
                                  0);
    cudaDeviceSynchronize();

    auto got_state = FromDevice(d_s1, ref_state.size());
    auto got_out = FromDevice(d_o, ref_out.size());
    for (size_t i = 0; i < ref_state.size(); ++i)
        CHECK(got_state[i] == Catch::Approx(ref_state[i]).margin(1e-5f));
    for (size_t i = 0; i < ref_out.size(); ++i)
        CHECK(got_out[i] == Catch::Approx(ref_out[i]).margin(1e-5f));

    cudaFree(d_s0); cudaFree(d_xs); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_dt); cudaFree(d_db); cudaFree(d_Al); cudaFree(d_s1); cudaFree(d_o);
}

TEST_CASE("gated rmsnorm matches the host reference", "[nemotron][kernels]")
{
    const int group_size = 8, n = 16;
    std::vector<float> x(n), g(n), w(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 0.1f * (i + 1);
        g[i] = (i % 2 == 0) ? 0.5f : -0.25f;
        w[i] = 0.5f + 0.01f * i;
    }
    const float eps = 1e-5f;
    std::vector<float> ref_out(n);
    ref::GatedRmsNorm(x.data(), g.data(), w.data(), n, group_size, eps,
                      ref_out.data());

    float *d_x = ToDevice(x), *d_g = ToDevice(g), *d_w = ToDevice(w);
    float *d_o = nullptr;
    cudaMalloc(&d_o, n * sizeof(float));
    RunNemotronGatedRmsNorm(d_x, d_g, d_w, n, group_size, eps, d_o, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_o, n);
    for (int i = 0; i < n; ++i)
        CHECK(got[i] == Catch::Approx(ref_out[i]).margin(1e-5f));

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_w); cudaFree(d_o);
}
