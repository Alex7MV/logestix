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
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/schema.h"
#include "nvfp4_scale.h"
#include "lsx_kernels_quant.cuh"

namespace {

// Test-local copy of the E2M1 decode table (host-visible; values match
// nvfp4_codec's kE2M1Table so the reference is exact).
constexpr float kE2M1TestTable[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// The canonical block-scale decoder itself, not a replica of it.
float E4M3ScaleToFloatHost(uint8_t e4m3)
{
    return lsxcommon::e4m3_to_float(e4m3);
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

// Regression: the scale decode follows the COMPLETE E4M3FN value table over
// the valid domain 0x00..0x7F (bit 7 is the reserved sign bit of the unsigned
// hardware scale type and is masked). In particular e4 == 0 is the SUBNORMAL
// range (m*2^-9), not zero — the source ModelOpt W4A16_NVFP4 block scales use
// those bytes. Both nvfp4_codec.h E4M3ScaleToFloat and nvfp4_scale.h
// E4m3ToFloat implement this single table.
TEST_CASE("full-E4M3 scale decode follows the E4M3FN table",
          "[qwen][gemm][nvfp4][regression]")
{
    using lsxcommon::gpu::nvfp4::E4M3ScaleToFloat;
    using lsxcommon::nvfp4::E4m3ToFloat;

    // 0x00 (and its bit-7 image 0x80) decode to exactly 0.
    for (uint8_t b : {0x00, 0x80}) {
        INFO("byte=0x" << std::hex << int(b) << std::dec);
        CHECK(E4M3ScaleToFloat(b) == 0.0f);
        CHECK(E4m3ToFloat(b) == 0.0f);
    }
    // e4 == 0 with a non-zero mantissa is a subnormal m*2^-9; bit 7 is masked,
    // so 0x80|b decodes exactly like b.
    const struct { uint8_t b; float v; } subnormals[] = {
        {0x01, 0.001953125f}, {0x03, 0.005859375f}, {0x07, 0.013671875f},
        {0x81, 0.001953125f}, {0x82, 0.00390625f}, {0x84, 0.0078125f},
        {0x87, 0.013671875f},
    };
    for (const auto &c : subnormals) {
        INFO("byte=0x" << std::hex << int(c.b) << std::dec);
        CHECK(E4M3ScaleToFloat(c.b) == c.v);
        CHECK(E4m3ToFloat(c.b) == c.v);
    }
    // Canonical powers of two remain exact (and both decoders agree).
    const struct { uint8_t b; float v; } cases[] = {
        {0x08, 0.015625f}, {0x38, 1.0f}, {0x40, 2.0f},
        {0x78, 256.0f}, {0x88, 0.015625f}, {0xf0, 128.0f},
    };
    for (const auto& c : cases) {
        INFO("byte=0x" << std::hex << int(c.b) << std::dec);
        CHECK(E4M3ScaleToFloat(c.b) == c.v);
        CHECK(E4m3ToFloat(c.b) == c.v);
    }
    // The reserved NaN slot decodes to NaN on both (never emitted by the
    // encoder; under the retired pow2 codec 0xFF read as 256).
    CHECK(std::isnan(E4M3ScaleToFloat(0xFF)));
    CHECK(std::isnan(E4m3ToFloat(0x7F)));
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
namespace {
inline float silu_ref(float v) { return v / (1.0f + std::exp(-v)); }
} // namespace

// ── Producer-fused swiglu+quant: byte parity vs standalone act-quant ──────
TEST_CASE("RunQwenSwigluQuant matches swiglu+ActQuant byte-for-byte",
          "[qwen][gemm][nvfp4][swigluquant]")
{
    // Sizes exercise full groups, one tail group, and multi-block grids.
    const int64_t sizes[] = {16, 272, 4096, 12288};
    for (int64_t n : sizes) {
        std::vector<__half> a(n), b(n);
        uint32_t st = 77;
        for (int64_t i = 0; i < n; ++i) {
            // Mixed magnitudes: small values, normal range, and a few
            // extremes that push absmax/6 past the canonical e4=14 clamp
            // (fp16 activations above ~768).
            const float mag = (i % 97 == 0) ? 1500.0f
                            : (i % 31 == 0) ? 300.0f
                            : (i % 7 == 0)  ? -8.5f
                                            : ((st = st * 1664525u + 1013904223u) %
                                               2000) /
                                                  1000.0f - 1.0f;
            a[i] = __float2half(mag);
            b[i] = __float2half(((st * 7919u) % 3000) / 1000.0f - 1.5f);
        }

        __half *da, *db, *dy1, *dy2;
        uint8_t *dp1, *ds1, *dp2, *ds2;
        cudaMalloc(&da, n * 2); cudaMalloc(&db, n * 2);
        cudaMalloc(&dy1, n * 2); cudaMalloc(&dy2, n * 2);
        cudaMalloc(&dp1, n / 2); cudaMalloc(&ds1, (n + 15) / 16);
        cudaMalloc(&dp2, n / 2); cudaMalloc(&ds2, (n + 15) / 16);
        cudaMemcpy(da, a.data(), n * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), n * 2, cudaMemcpyHostToDevice);

        // Fused path.
        REQUIRE(lsxcommon::qwen::RunQwenSwigluQuant(
                    nullptr, da, db, dy1, dp1, ds1, n) == cudaSuccess);
        // Reference: plain swiglu then the standalone act-quant kernel —
        // reached through the public GEMM entry's QuantizeActsNvfp4 is not
        // exported, so quantize dy2 via RunLsxTcGemmPreQuant-free path:
        // run ActQuant by calling the fused launcher's sibling: use
        // QuantizeActsNvfp4 through RunQwenNvfp4Gemm? Not exposed either.
        // Instead compare against the scalar host model below.

        cudaDeviceSynchronize();
        std::vector<__half> y1(n);
        std::vector<uint8_t> p1(n / 2), s1((n + 15) / 16);
        cudaMemcpy(y1.data(), dy1, n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(p1.data(), dp1, n / 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(s1.data(), ds1, (n + 15) / 16, cudaMemcpyDeviceToHost);

        // Host reference: fp16 round-trip then canonical encode.
        int mism_y = 0, mism_s = 0, mism_p = 0;
        for (int64_t g = 0; g < (n + 15) / 16; ++g) {
            const int64_t base = g * 16;
            const int len =
                static_cast<int>(std::min<int64_t>(16, n - base));
            float vals[16];
            float absmax = 0.0f;
            for (int i = 0; i < len; ++i) {
                const __half hv = __float2half(
                    silu_ref(__half2float(a[base + i])) *
                    __half2float(b[base + i]));
                if (__half2float(hv) != __half2float(y1[base + i]))
                    ++mism_y;
                vals[i] = __half2float(hv);
                absmax = std::fmax(absmax, std::fabs(vals[i]));
            }
            const uint8_t sb = lsxcommon::nvfp4::FloatToE4m3(absmax *
                                                               (1.0f / 6.0f));
            if (sb != s1[g]) ++mism_s;
            const float inv_s = E4M3ScaleToFloatHost(sb);
            const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;
            for (int i = 0; i < len; i += 2) {
                const uint8_t lo =
                    lsxcommon::kernels::QuantizeE2m1TieUp(vals[i] * inv);
                const uint8_t hi =
                    (i + 1 < len)
                        ? lsxcommon::kernels::QuantizeE2m1TieUp(vals[i + 1] *
                                                                inv)
                        : 0;
                if (static_cast<uint8_t>((hi << 4) | lo) !=
                    p1[(base + i) / 2])
                    ++mism_p;
            }
        }
        INFO("n=" << n << " mism_y=" << mism_y << " mism_s=" << mism_s
                  << " mism_p=" << mism_p);
        REQUIRE(mism_y == 0);
        REQUIRE(mism_s == 0);
        REQUIRE(mism_p == 0);

        cudaFree(da); cudaFree(db); cudaFree(dy1); cudaFree(dy2);
        cudaFree(dp1); cudaFree(ds1); cudaFree(dp2); cudaFree(ds2);
    }
}
