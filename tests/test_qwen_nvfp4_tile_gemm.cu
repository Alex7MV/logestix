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
#include "nvfp4_codec.h"
#include "nvfp4_scale.h"

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

// Regression: pow2-E4M3 scale decode must test the e4 FIELD, not the whole
// byte. Bytes with sign/mantissa bits set but a zero exponent field
// (0x80..0x87) decode to exactly 0 — the contract shared by nvfp4_codec.h
// E4M3ScaleToFloat and nvfp4_scale.h E4m3P2ToFloat. A whole-byte zero check
// instead decodes 0x82 as 2^-7, silently corrupting every GEMM whose stored
// scale pattern contains such bytes.
TEST_CASE("pow2-E4M3 scale decode is field-zero based", "[qwen][gemm][nvfp4][regression]")
{
    using lsxcommon::gpu::nvfp4::E4M3ScaleToFloat;
    using lsxcommon::nvfp4::E4m3P2ToFloat;

    // Zero exponent field -> exactly zero, regardless of sign/mantissa bits.
    for (uint8_t b : {0x00, 0x01, 0x03, 0x07, 0x80, 0x81, 0x82, 0x84, 0x87}) {
        INFO("byte=0x" << std::hex << int(b) << std::dec);
        CHECK(E4M3ScaleToFloat(b) == 0.0f);
        CHECK(E4m3P2ToFloat(b) == 0.0f);
    }
    // Canonical powers of two remain exact (and both decoders agree).
    const struct { uint8_t b; float v; } cases[] = {
        {0x08, 0.015625f}, {0x38, 1.0f}, {0x40, 2.0f},
        {0x78, 256.0f}, {0x88, 0.015625f}, {0xf0, 128.0f}, {0xff, 256.0f},
    };
    for (const auto& c : cases) {
        INFO("byte=0x" << std::hex << int(c.b) << std::dec);
        CHECK(E4M3ScaleToFloat(c.b) == c.v);
        CHECK(E4m3P2ToFloat(c.b) == c.v);
    }
}

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
    for (int i = 0; i < seq * N; ++i) {
        CAPTURE(i, __half2float(hy[i]),
                __half2float(__float2half(h_ref[(size_t)i])));
        CHECK(std::abs(__half2float(hy[i]) -
                       __half2float(__float2half(h_ref[(size_t)i]))) < 1e-3f);
    }

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
    for (int i = 0; i < seq * N; ++i) {
        CAPTURE(i, __half2float(hy[i]),
                __half2float(__float2half(h_ref[(size_t)i])));
        CHECK(std::abs(__half2float(hy[i]) -
                       __half2float(__float2half(h_ref[(size_t)i]))) < 1e-3f);
    }

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

// Host replica of the TILE kernel's exact math (not the legacy per-element
// path): weight dequant is rounded to FP16 before the multiply, accumulation
// runs in BK=32 chunks with zero-padded tails. Bit-exact mirror of
// qwen_nvfp4_gemm_tile_kernel, used to pin the seq==1 decode specialization.
void HostRefTileGemm(const __half *x, const uint8_t *packed, const uint8_t *scales,
                     float global_scale, __half *out, int seq, int K, int N)
{
    constexpr int BK = 32;
    for (int t = 0; t < seq; ++t)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int kb = 0; kb < K; kb += BK)
                for (int kk = 0; kk < BK; ++kk) {
                    const int k = kb + kk;
                    const float xv =
                        (k < K) ? __half2float(x[(int64_t)t * K + k]) : 0.0f;
                    float wv = 0.0f;
                    if (k < K) {
                        const int64_t e = (int64_t)n * K + k;
                        const uint8_t byte = packed[e >> 1];
                        const uint8_t nib =
                            (e & 1) ? (byte >> 4) : (byte & 0x0F);
                        wv = kE2M1TestTable[nib] *
                             E4M3ScaleToFloatHost(scales[e >> 4]) * global_scale;
                    }
                    // smem round-trip: dequantized weight stored as FP16.
                    acc += xv * __half2float(__float2half(wv));
                }
            out[(int64_t)t * N + n] = __float2half(acc);
        }
}

TEST_CASE("RunQwenNvfp4Gemm seq==1 decode path is bit-exact vs tile-kernel "
          "host replica (incl. K/N tails)",
          "[qwen][gemm][nvfp4][decode]")
{
    constexpr int seq = 1, K = 5136, N = 520;   // K%32!=0, N%256!=0 tails
    std::vector<__half> hx((size_t)seq * K);
    std::vector<uint8_t> hp((size_t)((int64_t)K * N / 2)), hs((size_t)((int64_t)K * N / 16));
    for (int i = 0; i < seq * K; ++i)
        hx[i] = __float2half(float((i % 13) - 6) * 0.25f);
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 29) & 0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 48));
    const float gs = 0.125f;

    std::vector<__half> h_ref((size_t)seq * N);
    HostRefTileGemm(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), seq, K, N);

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
    // Bit-exact contract: not a tolerance check.
    CHECK(std::memcmp(hy.data(), h_ref.data(),
                      hy.size() * sizeof(__half)) == 0);

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