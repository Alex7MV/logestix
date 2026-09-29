// Model-neutral, device-independent GEMV (lsxcommon::gemv) contract tests.
//
// Both entries are used by several families' decode paths; neither had a
// direct parity test. Pins the documented alpha/beta semantics
// (y = alpha*A*x + beta*y) for the [N,K] and the transposed [K,N] layouts and
// the f16 / bf16 / f32 output kinds.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/lsx_gemv.h"

namespace gemv = lsxcommon::gemv;

namespace {

uint32_t g_state = 12345u;
float NextRand()
{
    g_state = g_state * 1664525u + 1013904223u;
    return (float)((g_state >> 8) % 2001) / 1000.0f - 1.0f;  // [-1, 1]
}

struct F16Buf {
    __half *d = nullptr;
    std::vector<__half> h;
    void Alloc(size_t n)
    {
        h.resize(n);
        for (auto &v : h) v = __float2half(NextRand());
        REQUIRE(cudaMalloc(&d, n * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, h.data(), n * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }
    void Free() { cudaFree(d); }
};

// Reference for RunLxGemmVec: W is [N,K], x is [seq,K], y is [seq,N].
void RefNK(const std::vector<__half> &w, const std::vector<__half> &x,
           const std::vector<float> &y0, float alpha, float beta, int seq,
           int N, int K, std::vector<float> &y)
{
    y.assign((size_t)seq * N, 0.0f);
    for (int s = 0; s < seq; ++s)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(w[(size_t)n * K + k]) *
                       __half2float(x[(size_t)s * K + k]);
            y[(size_t)s * N + n] = alpha * acc + beta * y0[(size_t)s * N + n];
        }
}

// Reference for RunLxGemmVecWT: W is [K,N].
void RefKN(const std::vector<__half> &w, const std::vector<__half> &x,
           const std::vector<float> &y0, float alpha, float beta, int seq,
           int N, int K, std::vector<float> &y)
{
    y.assign((size_t)seq * N, 0.0f);
    for (int s = 0; s < seq; ++s)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(x[(size_t)s * K + k]) *
                       __half2float(w[(size_t)k * N + n]);
            y[(size_t)s * N + n] = alpha * acc + beta * y0[(size_t)s * N + n];
        }
}

}  // namespace

TEST_CASE("RunLxGemmVec honors alpha/beta (f16, bf16 and f32 out)",
          "[lsxgemv]")
{
    constexpr int seq = 3, N = 130, K = 96;
    F16Buf w, x;
    w.Alloc((size_t)N * K);
    x.Alloc((size_t)seq * K);
    std::vector<float> y0((size_t)seq * N);
    for (auto &v : y0) v = NextRand();

    std::vector<float> ref;
    RefNK(w.h, x.h, y0, 1.5f, 0.25f, seq, N, K, ref);

    SECTION("f16 output")
    {
        F16Buf y;
        y.Alloc((size_t)seq * N);
        std::vector<__half> seed((size_t)seq * N);
        for (int i = 0; i < seq * N; ++i) seed[i] = __float2half(y0[i]);
        REQUIRE(cudaMemcpy(y.d, seed.data(), seed.size() * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(gemv::RunLxGemmVec(nullptr, w.d, x.d, seq, N, K,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.5f, y.d, gemv::LxDtype::kF16, 0.25f) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__half> got((size_t)seq * N);
        REQUIRE(cudaMemcpy(got.data(), y.d, got.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < seq * N; ++i)
            CHECK(std::abs(__half2float(got[i]) - ref[i]) <
                  3e-3f * std::abs(ref[i]) + 2e-3f);
        y.Free();
    }
    SECTION("bf16 output")
    {
        std::vector<__nv_bfloat16> seed((size_t)seq * N);
        for (int i = 0; i < seq * N; ++i)
            seed[i] = __float2bfloat16_rn(y0[i]);
        void *dy = nullptr;
        REQUIRE(cudaMalloc(&dy, (size_t)seq * N * sizeof(__nv_bfloat16)) ==
                cudaSuccess);
        REQUIRE(cudaMemcpy(dy, seed.data(),
                           seed.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(gemv::RunLxGemmVec(nullptr, w.d, x.d, seq, N, K,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.5f, dy, gemv::LxDtype::kBF16, 0.25f) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__nv_bfloat16> got((size_t)seq * N);
        REQUIRE(cudaMemcpy(got.data(), dy,
                           got.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < seq * N; ++i)
            CHECK(std::abs(__bfloat162float(got[i]) - ref[i]) <
                  1.5e-2f * std::abs(ref[i]) + 5e-3f);
        cudaFree(dy);
    }
    SECTION("f32 output")
    {
        float *dy = nullptr;
        REQUIRE(cudaMalloc(&dy, (size_t)seq * N * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(dy, y0.data(), y0.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(gemv::RunLxGemmVec(nullptr, w.d, x.d, seq, N, K,
                                   gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                   1.5f, dy, gemv::LxDtype::kF32, 0.25f) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> got((size_t)seq * N);
        REQUIRE(cudaMemcpy(got.data(), dy, got.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < seq * N; ++i)
            CHECK(std::abs(got[i] - ref[i]) <
                  1e-4f * std::abs(ref[i]) + 1e-5f);
        cudaFree(dy);
    }
    w.Free(); x.Free();
}

TEST_CASE("RunLxGemmVecWT honors alpha/beta (K-major layout)", "[lsxgemv]")
{
    constexpr int seq = 2, N = 65, K = 100;
    F16Buf w, x, y;
    w.Alloc((size_t)K * N);
    x.Alloc((size_t)seq * K);
    y.Alloc((size_t)seq * N);
    std::vector<float> y0((size_t)seq * N);
    for (auto &v : y0) v = NextRand();

    std::vector<__half> seed((size_t)seq * N);
    for (int i = 0; i < seq * N; ++i) seed[i] = __float2half(y0[i]);
    REQUIRE(cudaMemcpy(y.d, seed.data(), seed.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(gemv::RunLxGemmVecWT(nullptr, w.d, x.d, seq, N, K,
                                 gemv::LxDtype::kF16, gemv::LxDtype::kF16, 2.0f,
                                 y.d, gemv::LxDtype::kF16, 0.5f) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> ref;
    RefKN(w.h, x.h, y0, 2.0f, 0.5f, seq, N, K, ref);
    std::vector<__half> got((size_t)seq * N);
    REQUIRE(cudaMemcpy(got.data(), y.d, got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < seq * N; ++i)
        CHECK(std::abs(__half2float(got[i]) - ref[i]) <
                  3e-3f * std::abs(ref[i]) + 2e-3f);
    w.Free(); x.Free(); y.Free();
}
