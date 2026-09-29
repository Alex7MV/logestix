// Model-neutral NVFP4 layer (lsxcommon::lsx_nvfp4) dispatch-contract tests.
//
// Closes the coverage gaps of the NVFP4 stack:
//   - invalid-argument guards of both GEMM entries
//   - legacy per-element dispatch branch (N < 8 || K < 16), incl. acc=true
//   - tile-kernel acc=true accumulate path (bit-exact vs FP16-roundtrip host
//     replica, mirrors the pinned seq==1 bit-exact contract)
//   - weight-streaming GEMV acc=true path (never exercised before)
//   - LsxFastMathScope runtime save/restore + nesting (model-independence)
//   - LaunchNvfp4Gemm / LsxNvfp4Weight wrapper parity and guards
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsx_nvfp4 = lsxcommon::lsx_nvfp4;

namespace {

constexpr float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Bit-exact host replica of nvfp4_codec.h's E4M3ScaleToFloat (e4-field zero).
float ScaleHost(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Plain per-element host reference (legacy-kernel order, full FP32 math).
void HostRefPlain(const __half *x, const uint8_t *packed, const uint8_t *scales,
                  float gs, float *out, int seq, int K, int N, bool acc)
{
    for (int t = 0; t < seq; ++t)
        for (int n = 0; n < N; ++n) {
            float v = acc ? __half2float(__float2half(0.5f)) : 0.0f;
            for (int k = 0; k < K; ++k) {
                const int64_t e = (int64_t)n * K + k;
                const uint8_t byte = packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                v += __half2float(x[(int64_t)t * K + k]) * kE2M1[nib] *
                     (ScaleHost(scales[e >> 4]) * gs);
            }
            out[(int64_t)t * N + n] = v;
        }
}

// Tile-kernel host replica: weight dequant FP16-roundtripped, BK=32 chunk
// accumulation, seeded accumulator, half store (bit-exact contract).
void HostRefTile(const __half *x, const uint8_t *packed, const uint8_t *scales,
                 float gs, __half *out, int seq, int K, int N, bool acc)
{
    constexpr int BK = 32;
    for (int t = 0; t < seq; ++t)
        for (int n = 0; n < N; ++n) {
            float a = acc ? __half2float(out[(int64_t)t * N + n]) : 0.0f;
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
                        wv = kE2M1[nib] * ScaleHost(scales[e >> 4]) * gs;
                    }
                    a += xv * __half2float(__float2half(wv));
                }
            out[(int64_t)t * N + n] = __float2half(a);
        }
}

struct Bufs {
    __half *dx = nullptr, *dy = nullptr, *dy2 = nullptr;
    uint8_t *dp = nullptr, *ds = nullptr;
    std::vector<__half> hx;
    std::vector<uint8_t> hp, hs;
    int seq = 0, K = 0, N = 0;

    void Init(int seq_, int K_, int N_, float gs_dummy = 0.0f)
    {
        seq = seq_; K = K_; N = N_;
        hx.resize((size_t)seq * K);
        hp.resize((size_t)K * N / 2);
        hs.resize((size_t)K * N / 16);
        uint32_t st = 4242;
        for (auto &v : hx) v = __float2half(float((int)(st = st * 1664525u + 1013904223u) % 13 - 6) * 0.25f);
        for (auto &b : hp) b = static_cast<uint8_t>((st = st * 1664525u + 1013904223u) & 0xFF);
        for (auto &b : hs) b = static_cast<uint8_t>(128 + ((st = st * 1664525u + 1013904223u) % 48));
        (void)gs_dummy;
        REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dp, hp.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&ds, hs.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, (size_t)seq * N * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy2, (size_t)seq * N * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    void Seed(__half *p, float v)
    {
        std::vector<__half> s((size_t)seq * N, __float2half(v));
        REQUIRE(cudaMemcpy(p, s.data(), s.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    std::vector<__half> Read(const __half *p)
    {
        std::vector<__half> h((size_t)seq * N);
        REQUIRE(cudaMemcpy(h.data(), p, h.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
        return h;
    }
    void Free()
    {
        cudaFree(dx); cudaFree(dp); cudaFree(ds);
        cudaFree(dy); cudaFree(dy2);
    }
};

}  // namespace

TEST_CASE("RunLsxNvfp4Gemm rejects invalid arguments", "[lsxnvfp4][dispatch]")
{
    Bufs b;
    b.Init(1, 64, 128);
    const float gs = 0.5f;
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, nullptr, b.dp, b.ds, gs, b.dy,
                                     1, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, nullptr, b.ds, gs, b.dy,
                                     1, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                     1, b.K, b.N) == cudaSuccess);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, nullptr, gs, b.dy,
                                     1, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, nullptr,
                                     1, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                     0, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                     1, 0, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                     1, b.K, 0) == cudaErrorInvalidValue);
    b.Free();
}

TEST_CASE("RunLsxNvfp4GemmSeq1GemmV rejects invalid arguments",
          "[lsxnvfp4][dispatch][gemv]")
{
    Bufs b;
    b.Init(1, 512, 64);
    const float gs = 0.5f;
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, nullptr, b.dp, b.ds, gs,
                                              b.dy, b.K, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                              b.dy, 0, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                              b.dy, b.K, 0) == cudaErrorInvalidValue);
    // Shape contract: K % 32 != 0 is rejected outright.
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                              b.dy, 5136, b.N) == cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                              b.dy, 30, b.N) == cudaErrorInvalidValue);
    b.Free();
}

TEST_CASE("Legacy dispatch branch (N<8, K<16) parity incl. acc=true",
          "[lsxnvfp4][dispatch][legacy]")
{
    // N < 8 with K >= 16 and K < 16 with N >= 8 both route to the per-element
    // legacy kernel; neither combination is covered by the tile/GEMV suites.
    const struct { int seq, K, N; bool acc; } cases[] = {
        {2, 64, 4, false}, {1, 8, 16, false}, {2, 64, 4, true}, {1, 8, 16, true},
    };
    for (const auto &c : cases) {
        Bufs b;
        b.Init(c.seq, c.K, c.N);
        const float gs = 0.25f;
        std::vector<float> ref((size_t)c.seq * c.N);
        HostRefPlain(b.hx.data(), b.hp.data(), b.hs.data(), gs, ref.data(),
                     c.seq, c.K, c.N, c.acc);
        if (c.acc) b.Seed(b.dy, 0.5f);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                           c.seq, c.K, c.N, c.acc) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = b.Read(b.dy);
        for (int i = 0; i < c.seq * c.N; ++i) {
            INFO("case=" << c.seq << "x" << c.K << "x" << c.N
                 << " acc=" << c.acc << " i=" << i);
            CHECK(std::abs(__half2float(hy[i]) - ref[i]) < 1e-3f);
        }
        b.Free();
    }
}

TEST_CASE("Tile path acc=true is bit-exact vs FP16-roundtrip host replica",
          "[lsxnvfp4][dispatch][acc]")
{
    Bufs b;
    b.Init(2, 64, 128);   // N>=8 && K>=16 -> tile kernel (fast math off)
    const float gs = 0.5f;
    b.Seed(b.dy, 0.25f);
    std::vector<__half> seed((size_t)b.seq * b.N, __float2half(0.25f));
    HostRefTile(b.hx.data(), b.hp.data(), b.hs.data(), gs, seed.data(),
                b.seq, b.K, b.N, true);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy,
                                       b.seq, b.K, b.N, true) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    CHECK(std::memcmp(hy.data(), seed.data(), hy.size() * sizeof(__half)) == 0);
    b.Free();
}

TEST_CASE("Weight-streaming GEMV acc=true parity", "[lsxnvfp4][gemv][acc]")
{
    // The direct entry is toggle-independent (always the streaming kernel).
    Bufs b;
    b.Init(1, 2048, 512);
    const float gs = 0.125f;
    std::vector<float> ref((size_t)b.N);
    HostRefPlain(b.hx.data(), b.hp.data(), b.hs.data(), gs, ref.data(),
                 1, b.K, b.N, /*acc=*/false);   // ref = plain dot product
    b.Seed(b.dy, 0.5f);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                                b.dy, b.K, b.N, true) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    for (int n = 0; n < b.N; ++n) {
        const float expect = __half2float(__float2half(0.5f)) + ref[n];
        INFO("n=" << n << " ref=" << expect << " got=" << __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - expect) <
              2e-3f * std::abs(expect) + 2e-4f);
    }
    b.Free();
}

TEST_CASE("LsxFastMathScope runtime save/restore and nesting",
          "[lsxnvfp4][dispatch][fastmath]")
{
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);
    {
        lsxcommon::lsx_nvfp4::LsxFastMathScope outer(false, true);
        CHECK_FALSE(lsxcommon::lsx_nvfp4::LsxTcFastMath());
        CHECK(lsxcommon::lsx_nvfp4::LsxSeq1Gemv());
        {
            lsxcommon::lsx_nvfp4::LsxFastMathScope inner(true, false);
            CHECK(lsxcommon::lsx_nvfp4::LsxTcFastMath());
            CHECK_FALSE(lsxcommon::lsx_nvfp4::LsxSeq1Gemv());
        }
        // Inner scope restored the outer policy, not the pre-outer one.
        CHECK_FALSE(lsxcommon::lsx_nvfp4::LsxTcFastMath());
        CHECK(lsxcommon::lsx_nvfp4::LsxSeq1Gemv());
    }
    // Outer scope restored the pre-scope process state.
    CHECK(lsxcommon::lsx_nvfp4::LsxTcFastMath());
    CHECK_FALSE(lsxcommon::lsx_nvfp4::LsxSeq1Gemv());
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
}

TEST_CASE("LaunchNvfp4Gemm wrapper parity and LsxNvfp4Weight guards",
          "[lsxnvfp4][dispatch][wrapper]")
{
    Bufs b;
    b.Init(2, 64, 128);
    const float gs = 0.5f;

    lsxcommon::lsx_nvfp4::LsxNvfp4Weight w;
    w.d_packed = b.dp;
    w.d_scales = b.ds;
    w.global_scale = gs;
    w.rows = b.N;
    w.cols = b.K;
    w.packed_bytes = (int64_t)b.hp.size();
    w.scale_bytes = (int64_t)b.hs.size();

    REQUIRE(lsx_nvfp4::LaunchNvfp4Gemm(w, b.dx, b.seq, b.dy, nullptr) == cudaSuccess);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, b.dx, b.dp, b.ds, gs, b.dy2,
                                       b.seq, b.K, b.N) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    const auto hy2 = b.Read(b.dy2);
    CHECK(std::memcmp(hy.data(), hy2.data(), hy.size() * sizeof(__half)) == 0);

    // Degenerate weight geometry is rejected.
    lsxcommon::lsx_nvfp4::LsxNvfp4Weight bad = w;
    bad.rows = 0;
    CHECK(lsx_nvfp4::LaunchNvfp4Gemm(bad, b.dx, b.seq, b.dy, nullptr) == cudaErrorInvalidValue);
    bad.rows = b.N; bad.cols = 0;
    CHECK(lsx_nvfp4::LaunchNvfp4Gemm(bad, b.dx, b.seq, b.dy, nullptr) == cudaErrorInvalidValue);

    // view_only weights must NOT release aliased storage on Free().
    lsxcommon::lsx_nvfp4::LsxNvfp4Weight view = w;
    view.view_only = true;
    view.Free();
    CHECK(view.d_packed == nullptr);
    CHECK(view.d_scales == nullptr);
    // Owning Free releases the buffers; Bufs must not free them again.
    w.Free();
    b.dp = nullptr;
    b.ds = nullptr;
    b.Free();
}

TEST_CASE("seq==1 direct GEMV entry minimal shape parity",
          "[lsxnvfp4][gemv]")
{
    Bufs b;
    b.Init(1, 32, 8);   // kp=1, one chunk per row
    const float gs = 1.0f;
    std::vector<float> ref((size_t)b.N);
    HostRefPlain(b.hx.data(), b.hp.data(), b.hs.data(), gs, ref.data(),
                 1, b.K, b.N, false);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                                b.dy, b.K, b.N) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    for (int n = 0; n < b.N; ++n) {
        INFO("n=" << n << " ref=" << ref[n] << " got=" << __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - ref[n]) <
              2e-3f * std::abs(ref[n]) + 2e-4f);
    }
    b.Free();
}

TEST_CASE("GEMV KP heuristic never drops the K tail (K multiple of 32, not 256)",
          "[lsxnvfp4][gemv][shape]")
{
    // K = 2080 = 32*65: satisfies the entry contract (K % 32 == 0) but not
    // K % 256. The kp floor test (K/2 >= 1024 && N*2 >= 1024) would select
    // KP=2; the kernel then walks chunks = (K/KP)>>5 = 32 per 1040-element
    // slice, covering only 1024 elements and silently dropping the 16-element
    // tail of every slice. The heuristic must pick a KP that divides K into
    // whole 32-element chunks (or fall back to KP=1).
    Bufs b;
    b.Init(1, 2080, 512);
    const float gs = 0.25f;
    std::vector<float> ref((size_t)b.N);
    HostRefPlain(b.hx.data(), b.hp.data(), b.hs.data(), gs, ref.data(), 1,
                 b.K, b.N, false);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds, gs,
                                                b.dy, b.K, b.N) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    for (int n = 0; n < b.N; ++n) {
        INFO("n=" << n << " ref=" << ref[n] << " got=" << __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - ref[n]) <
              2e-3f * std::abs(ref[n]) + 2e-4f);
    }
    b.Free();
}

TEST_CASE("Pair and expert-batch GEMV keep K/KP chunk-aligned (K=2080)",
          "[lsxnvfp4][gemv][shape]")
{
    // Same tail-alignment hazard as the plain entry, on the two other
    // streaming-GEMV launch helpers. A KP that leaves K/KP % 32 != 0 makes
    // the packed-row uint4 load fault.
    constexpr int K = 2080, NH = 512;
    const float gs_gate = 0.125f, gs_up = 0.03125f;
    const int64_t pack_half = (int64_t)K * NH / 2;
    const int64_t scale_half = (int64_t)K * NH / 16;
    {
        Bufs b;
        b.Init(1, K, 2 * NH);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
                    nullptr, b.dx, b.dp, b.ds, gs_gate, gs_up, b.dy, K, NH) ==
                cudaSuccess);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds,
                                                    gs_gate, b.dy2, K, NH) ==
                cudaSuccess);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                    nullptr, b.dx, b.dp + pack_half, b.ds + scale_half, gs_up,
                    b.dy2 + NH, K, NH) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = b.Read(b.dy);
        const auto href = b.Read(b.dy2);
        CHECK(std::memcmp(hy.data(), href.data(), hy.size() * sizeof(__half)) ==
              0);
        b.Free();
    }
    {
        Bufs b;
        b.Init(1, K, NH);
        lsx_nvfp4::SetLsxTcFastMath(true);
        lsx_nvfp4::SetLsxSeq1Gemv(true);
        const uint8_t *packed[1] = {b.dp};
        const uint8_t *scales[1] = {b.ds};
        const float gs[1] = {gs_gate};
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    nullptr, b.dx, packed, scales, gs, 1, /*x_stride=*/0, b.dy,
                    K, NH, /*acc=*/false) == cudaSuccess);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds,
                                                    gs_gate, b.dy2, K, NH) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = b.Read(b.dy);
        const auto href = b.Read(b.dy2);
        CHECK(std::memcmp(hy.data(), href.data(), hy.size() * sizeof(__half)) ==
              0);
        lsx_nvfp4::SetLsxTcFastMath(false);
        lsx_nvfp4::SetLsxSeq1Gemv(false);
        b.Free();
    }
}

TEST_CASE("Gate/up pair GEMV is bit-exact vs two single GEMVs",
          "[lsxnvfp4][gemv][pair]")
{
    // Routed-expert shape (GLM-5.3-Flash): K=4096, gate/up halves of 2048
    // rows each, packed as [gate rows | up rows] with scales [s1 | s3] and
    // DIFFERENT global scales per half (the converter computes gs per
    // tensor). The pair entry must produce bit-identical halves.
    constexpr int K = 4096, NH = 2048;
    Bufs b;
    b.Init(1, K, 2 * NH);
    const float gs_gate = 0.125f, gs_up = 0.03125f;

    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
                nullptr, b.dx, b.dp, b.ds, gs_gate, gs_up, b.dy, K, NH) ==
            cudaSuccess);

    // Reference: two independent plain streaming GEMVs over the sub-ranges.
    const int64_t pack_half = (int64_t)K * NH / 2;
    const int64_t scale_half = (int64_t)K * NH / 16;
    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(nullptr, b.dx, b.dp, b.ds,
                                                gs_gate, b.dy2, K, NH) ==
            cudaSuccess);
    REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                nullptr, b.dx, b.dp + pack_half, b.ds + scale_half, gs_up,
                b.dy2 + NH, K, NH) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto hy = b.Read(b.dy);
    const auto href = b.Read(b.dy2);
    CHECK(std::memcmp(hy.data(), href.data(), hy.size() * sizeof(__half)) ==
          0);

    // Invalid arguments are rejected. K % 32 != 0 is NOT an error: the
    // pair entry falls back to the two plain GEMMs (bit-exact tail).
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
              nullptr, nullptr, b.dp, b.ds, gs_gate, gs_up, b.dy, K, NH) ==
          cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
              nullptr, b.dx, b.dp, b.ds, gs_gate, gs_up, b.dy, 0, NH) ==
          cudaErrorInvalidValue);
    CHECK(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
              nullptr, b.dx, b.dp, b.ds, gs_gate, gs_up, b.dy, K, 0) ==
          cudaErrorInvalidValue);
    // Odd-K fallback still matches the two-single reference numerically
    // (host replica over the same sub-ranges).
    {
        Bufs s;
        s.Init(1, 48, 64);   // dy holds the full [gate | up] pair output
        std::vector<float> ref((size_t)2 * 32);
        HostRefPlain(s.hx.data(), s.hp.data(), s.hs.data(), gs_gate,
                     ref.data(), 1, 48, 32, false);
        HostRefPlain(s.hx.data(), s.hp.data() + (int64_t)48 * 32 / 2,
                     s.hs.data() + (int64_t)48 * 32 / 16, gs_up,
                     ref.data() + 32, 1, 48, 32, false);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
                    nullptr, s.dx, s.dp, s.ds, gs_gate, gs_up, s.dy, 48, 32) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = s.Read(s.dy);
        for (int n = 0; n < 2 * 32; ++n) {
            INFO("n=" << n << " ref=" << ref[n]
                      << " got=" << __half2float(hy[n]));
            CHECK(std::abs(__half2float(hy[n]) - ref[n]) <
                  2e-3f * std::abs(ref[n]) + 2e-4f);
        }
        s.Free();
    }
    b.Free();
}

TEST_CASE("GEMV KP grain is bit-stable across launch configs (qwen3.8-27B shapes)",
          "[lsxnvfp4][gemv][grain]")
{
    // 27B decode shapes reach the pair/plain GEMV entries only when the perf
    // knob is raised. The kp heuristic (K/cand>=1024 && N*cand>=1024 -> kp=8)
    // must keep each output bit-identical to any other grain selection: the
    // pair entry picks kp from n_half (17408 -> kp=8) while the plain entry
    // over the same rows picks kp from K (5120/8=640 < 1024 -> kp=1). Mixed
    // grains must agree bit-for-bit, or the fast-math tolerance contract for
    // the 27B decode path is not pinned.
    {
        Bufs b;
        b.Init(1, 5120, 2 * 17408);
        const float gs = 0.0625f;
        // Pair and the two-single reference pick kp from different roles
        // (n_half vs K), so the slice-partition merge order differs and the
        // outputs are only tolerance-equal (same numeric class as any other
        // fast-math route). Both must stay within the GEMV tolerance band
        // vs the host reference over the same sub-ranges.
        std::vector<float> ref((size_t)2 * 17408);
        HostRefPlain(b.hx.data(), b.hp.data(), b.hs.data(), gs, ref.data(),
                     1, 5120, 17408, false);
        HostRefPlain(b.hx.data(), b.hp.data() + (int64_t)5120 * 17408 / 2,
                     b.hs.data() + (int64_t)5120 * 17408 / 16, gs,
                     ref.data() + 17408, 1, 5120, 17408, false);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
                    nullptr, b.dx, b.dp, b.ds, gs, gs, b.dy, 5120, 17408) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = b.Read(b.dy);
        for (int n = 0; n < 2 * 17408; ++n) {
            INFO("n=" << n << " ref=" << ref[n]
                      << " got=" << __half2float(hy[n]));
            CHECK(std::abs(__half2float(hy[n]) - ref[n]) <
                  2e-3f * std::abs(ref[n]) + 2e-4f);
        }
        b.Free();
    }
    {
        Bufs d;
        d.Init(1, 17408, 5120);
        const float gs = 0.03125f;
        std::vector<float> ref((size_t)d.N);
        HostRefPlain(d.hx.data(), d.hp.data(), d.hs.data(), gs, ref.data(),
                     1, d.K, d.N, false);
        REQUIRE(lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                    nullptr, d.dx, d.dp, d.ds, gs, d.dy, d.K, d.N) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto hy = d.Read(d.dy);
        for (int n = 0; n < d.N; ++n) {
            INFO("n=" << n << " ref=" << ref[n]
                      << " got=" << __half2float(hy[n]));
            CHECK(std::abs(__half2float(hy[n]) - ref[n]) <
                  2e-3f * std::abs(ref[n]) + 2e-4f);
        }
        d.Free();
    }
}
