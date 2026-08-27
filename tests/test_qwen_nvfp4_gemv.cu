// seq==1 decode-path NVFP4 GEMV: weight-streaming CUDA-core dot product.
//
// Round-18 findings lever: at seq==1 the routed/shared GEMMs read ~1.73 MB
// per expert but sustain only ~30 GB/s through the 16x16x32 tile kernel
// (M=1 wastes the tile M-dimension entirely). The GEMV kernel streams
// packed E2M1 weights straight from global memory with coalesced uint4
// loads and a shfl tree reduction -- no shared-memory staging, no MMA.
//
// Numeric contract: FP32 accumulate, k-order free (tolerance class), exactly
// like the TC fast-math paths. Determinism contract: fixed partitioning ->
// bit-stable across repeated launches (run-to-run), pinned here.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"

namespace {

// Test-local E2M1 decode table (values match nvfp4_codec kE2M1Table).
constexpr float kE2M1TestTable[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Bit-exact host replica of nvfp4_codec.h's E4M3ScaleToFloat (e4-field zero).
float E4M3ScaleToFloatHost(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void HostRefGemv(const __half *x, const uint8_t *packed,
                 const uint8_t *scales, float global_scale, float *out,
                 int K, int N)
{
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int64_t e = static_cast<int64_t>(n) * K + k;
            const uint8_t byte = packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl =
                E4M3ScaleToFloatHost(scales[e >> 4]) * global_scale;
            acc += __half2float(x[k]) * kE2M1TestTable[nib] * scl;
        }
        out[n] = acc;
    }
}

struct GemvCase {
    int K, N;
};

void RunParity(const GemvCase &c, bool acc_in, float gs)
{
    std::vector<__half> hx(c.K);
    std::vector<uint8_t> hp(static_cast<size_t>(c.K) * c.N / 2),
        hs(static_cast<size_t>(c.K) * c.N / 16);
    for (int i = 0; i < c.K; ++i)
        hx[i] = __float2half(float((i % 13) - 6) * 0.25f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 48));

    std::vector<float> h_ref(c.N);
    HostRefGemv(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), c.K, c.N);

    __half *dx, *dy;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(c.K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, static_cast<size_t>(c.N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(c.K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
    const float seed_val = acc_in ? 0.5f : 0.0f;
    std::vector<__half> hseed(c.N, __float2half(seed_val));
    cudaMemcpy(dy, hseed.data(), static_cast<size_t>(c.N) * 2,
               cudaMemcpyHostToDevice);

    cudaError_t e = lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
        nullptr, dx, dp, ds, gs, dy, c.K, c.N, acc_in);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(c.N);
    cudaMemcpy(hy.data(), dy, static_cast<size_t>(c.N) * 2, cudaMemcpyDeviceToHost);
    for (int n = 0; n < c.N; ++n) {
        const float ref = __half2float(__float2half(h_ref[n])) + seed_val;
        INFO("n=" << n << " ref=" << ref << " got=" << __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - ref) <
              2e-3f * std::abs(ref) + 2e-4f);
    }

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

}  // namespace

TEST_CASE("NVFP4 GEMV parity vs host reference", "[qwen][gemm][nvfp4][gemv]")
{
    // Real decode shapes: w13-like [K=hidden, N=moe_i] and w2-like inverse.
    const GemvCase cases[] = {{2048, 512}, {512, 2048}, {256, 96}};
    for (const auto &c : cases)
        RunParity(c, /*acc=*/false, /*gs=*/0.125f);
}

// Pins the within-chunk scale-block split: elements 0..15 of every
// 32-element uint4 chunk use scale byte sb, elements 16..31 use sb+1.
// Adversarial contrast (sb decodes to ~0, sb+1 to the max pow2) makes ANY
// lo/hi-nibble <-> scale-half mixup explode far beyond tolerance.
TEST_CASE("NVFP4 GEMV chunk scale-block split", "[qwen][gemm][nvfp4][gemv]")
{
    constexpr int K = 512, N = 64;
    const float gs = 1.0f;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i) hx[i] = __float2half(i % 2 ? 1.0f : -1.0f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = 0x11;   // ±1 nibbles
    for (size_t i = 0; i < hs.size(); i += 2) {
        hs[i] = 0x08;                                      // 2^-6
        if (i + 1 < hs.size()) hs[i + 1] = 0xf0;           // 2^7
    }

    std::vector<float> h_ref(N);
    HostRefGemv(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), K, N);

    __half *dx, *dy; uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, gs, dy, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(N);
    cudaMemcpy(hy.data(), dy, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    for (int n = 0; n < N; ++n) {
        INFO("n=" << n << " ref=" << h_ref[n] << " got=" <<
             __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - h_ref[n]) <
              2e-3f * std::abs(h_ref[n]) + 2e-4f);
    }
    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

TEST_CASE("NVFP4 GEMV deterministic across launches",
          "[qwen][gemm][nvfp4][gemv]")
{
    constexpr int K = 2048, N = 512;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i) hx[i] = __float2half(float((i % 9) - 4));
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 37) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 24));

    __half *dx, *d1, *d2;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&d1, static_cast<size_t>(N) * 2);
    cudaMalloc(&d2, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.25f, d1, K, N, false) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.25f, d2, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> r1(N), r2(N);
    cudaMemcpy(r1.data(), d1, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2.data(), d2, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(r1.data(), r2.data(), r1.size() * sizeof(__half)) == 0);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(d1); cudaFree(d2);
}

TEST_CASE("RunQwenNvfp4Gemm fast-math seq==1 dispatch reaches GEMV "
          "(bitwise equal)",
          "[qwen][gemm][nvfp4][gemv]")
{
    lsxcommon::qwen::SetQwenTcFastMath(true);
    lsxcommon::qwen::SetQwenSeq1Gemv(true);
    constexpr int K = 2048, N = 512;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i)
        hx[i] = __float2half(float((i % 11) - 5) * 0.5f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 17) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 32));

    __half *dx, *d_tile, *d_gemv;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&d_tile, static_cast<size_t>(N) * 2);
    cudaMalloc(&d_gemv, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, 0.125f, d_tile, 1, K, N) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.125f, d_gemv, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> t(N), g(N);
    cudaMemcpy(t.data(), d_tile, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(), d_gemv, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    // Same kernel -> identical bits. Any mismatch means the dispatcher did
    // NOT reach the GEMV kernel for the supported seq==1 shape.
    CHECK(std::memcmp(t.data(), g.data(), N * sizeof(__half)) == 0);

    lsxcommon::qwen::SetQwenSeq1Gemv(false);
    lsxcommon::qwen::SetQwenTcFastMath(false);
    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(d_tile); cudaFree(d_gemv);
}
