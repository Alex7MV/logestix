// NVFP4 dense GEMM parity harness: RunQwenNvfp4Gemm vs exact host reference.
//
// Direction A (Blackwell inference optimization): the Qwen dense NVFP4 GEMM is
// reworked from a per-element kernel to a shared-memory tile kernel. The
// numeric contract is FP32 accumulate with free k-order, so these tests verify
// that the production entry matches an exact per-element host reference within
// a half-roundtrip tolerance envelope, for both the tile path (K>=16, N>=8)
// and the legacy fallback path (K<16).
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"

namespace {

// Test-local copy of the E2M1 decode table (host-visible; values match
// nvfp4_codec's kE2M1Table so the reference is exact).
constexpr float kE2M1TestTable[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Bit-exact host replica of nvfp4_codec.h's E4M3ScaleToFloat:
// e4 == 0 -> 0.0f, else (e4+120)<<23 as a float.
float E4M3ScaleToFloatHost(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Host reference: exact per-element accumulation order (matches legacy kernel).
// out is FP32 [seq, N]; packed is nibble-packed E2M1 [N, K] row-major, scales
// per-16 E4M3.
void HostRefNvfp4Gemm(const __half *x, const uint8_t *packed, const uint8_t *scales,
                      float global_scale, float *out, int seq, int K, int N)
{
    for (int t = 0; t < seq; ++t)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int64_t e = (int64_t)n * K + k;
                const uint8_t byte = packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl = E4M3ScaleToFloatHost(scales[e >> 4]) * global_scale;
                acc += __half2float(x[(int64_t)t * K + k]) * kE2M1TestTable[nib] * scl;
            }
            out[(int64_t)t * N + n] = acc;
        }
}

}  // namespace

TEST_CASE("RunQwenNvfp4Gemm matches host reference (small-K, tile path)", "[qwen][gemm][nvfp4]")
{
    constexpr int seq = 2, K = 64, N = 128;   // N>=8, K>=16 -> tile kernel
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp((size_t)K * N / 2), hs((size_t)K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 7) - 3));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 13) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 32));
    const float gs = 0.5f;

    std::vector<float> h_ref((size_t)seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy((size_t)seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i)
        CHECK(std::abs(__half2float(hy[i]) -
                       __half2float(__float2half(h_ref[(size_t)i]))) < 1e-3f);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

TEST_CASE("RunQwenNvfp4Gemm legacy fallback at truly-small K", "[qwen][gemm][nvfp4]")
{
    constexpr int seq = 1, K = 8, N = 16;    // K<16 -> legacy per-element path
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp((size_t)K * N / 2), hs((size_t)K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 5) - 2));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 3) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(130 + (i % 16));
    const float gs = 1.0f;

    std::vector<float> h_ref((size_t)seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy((size_t)seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i)
        CHECK(std::abs(__half2float(hy[i]) -
                       __half2float(__float2half(h_ref[(size_t)i]))) < 1e-3f);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

TEST_CASE("RunQwenNvfp4Gemm tile matches host reference at large K", "[qwen][gemm][nvfp4]")
{
    constexpr int seq = 2, K = 5120, N = 512;
    std::vector<__half> hx(seq * K);
    std::vector<uint8_t> hp((size_t)K * N / 2), hs((size_t)K * N / 16);
    for (int i = 0; i < seq * K; ++i) hx[i] = __float2half(float((i % 11) - 5) * 0.5f);
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 7) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 64));
    const float gs = 0.25f;

    std::vector<float> h_ref((size_t)seq * N);
    HostRefNvfp4Gemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

    __half *dx, *dy;  uint8_t *dp, *ds;
    cudaMalloc(&dx, (size_t)seq * K * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, (size_t)seq * N * 2);
    cudaMemcpy(dx, hx.data(), (size_t)seq * K * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, gs, dy, seq, K, N) == cudaSuccess);

    std::vector<__half> hy((size_t)seq * N);
    cudaMemcpy(hy.data(), dy, (size_t)seq * N * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq * N; ++i) {
        const float ref = __half2float(__float2half(h_ref[(size_t)i]));  // y stored as half
        CHECK(std::abs(__half2float(hy[i]) - ref) < 1e-3f * std::abs(ref) + 1e-6f);
    }

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}