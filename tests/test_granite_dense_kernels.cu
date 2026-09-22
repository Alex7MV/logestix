// Granite dense kernel-level parity tests (Model Ingestion Protocol step 5).
// Host-only CPU references in FP64/FP32 against the exposed launchers:
//   RunGraniteRmsNormKernel / RunGraniteRopeKernel / RunGraniteAttentionKernel.
#include <catch2/catch_all.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <lsxcommon/granite_dense_inference.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::granite;
namespace lsxg = lsxcommon::lsx_nvfp4;

namespace {

__half H(float v) { return __float2half(v); }

float lcg(int64_t &s)
{
    s = (s * 6364136223846793005LL + 1442695040888963407LL);
    return static_cast<float>((s >> 40) & 0xffff) / 65536.0f - 0.5f;
}

std::vector<__half> ToHalf(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

} // namespace

TEST_CASE("granite rmsnorm kernel matches a CPU reference",
          "[granite][kernels]")
{
    const int rows = 3, hidden = 8;
    const float eps = 1e-5f;
    int64_t seed = 12345;
    std::vector<float> x(rows * hidden), w(hidden);
    for (auto &v : x) v = lcg(seed);
    for (auto &v : w) v = lcg(seed) + 1.0f;

    auto hx = ToHalf(x), hw = ToHalf(w);
    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunGraniteRmsNormKernel(nullptr, dout, dx, dw, rows, hidden,
                                    eps) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> out(rows * hidden);
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r) {
        float ss = 0.f;
        for (int j = 0; j < hidden; ++j)
            ss += x[r * hidden + j] * x[r * hidden + j];
        const float inv = 1.0f / std::sqrt(ss / hidden + eps);
        for (int j = 0; j < hidden; ++j) {
            const float ref = x[r * hidden + j] * inv * w[j];
            INFO("r=" << r << " j=" << j);
            CHECK(std::fabs(__half2float(out[r * hidden + j]) - ref) < 2e-2f);
        }
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

TEST_CASE("granite rope kernel matches a CPU reference", "[granite][kernels]")
{
    const int seq = 2, nh = 2, nkv = 1, hd = 8;
    const double theta = 10000.0;
    std::vector<int32_t> pos = {0, 3};
    int64_t seed = 999;
    std::vector<float> q(seq * nh * hd), k(seq * nkv * hd);
    for (auto &v : q) v = lcg(seed);
    for (auto &v : k) v = lcg(seed);

    auto hq = ToHalf(q), hk = ToHalf(k);
    __half *dq, *dk;
    int32_t *dp;
    REQUIRE(cudaMalloc(&dq, hq.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, hk.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dp, pos.size() * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, pos.data(), pos.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(RunGraniteRopeKernel(nullptr, dq, dk, dp, seq, nh, nkv, hd,
                                 theta) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> oq(hq.size()), ok(hk.size());
    REQUIRE(cudaMemcpy(oq.data(), dq, oq.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(ok.data(), dk, ok.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    const int half = hd / 2;
    for (int t = 0; t < seq; ++t)
        for (int head = 0; head < nh; ++head)
            for (int i = 0; i < half; ++i) {
                const float freq = std::pow(
                    static_cast<float>(theta),
                    -2.0f * i / static_cast<float>(hd));
                const float ang = static_cast<float>(pos[t]) * freq;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x1 = q[(t * nh + head) * hd + i];
                const float x2 = q[(t * nh + head) * hd + i + half];
                const float r1 = x1 * c - x2 * s;
                const float r2 = x1 * s + x2 * c;
                INFO("q t=" << t << " h=" << head << " i=" << i);
                CHECK(std::fabs(__half2float(oq[(t * nh + head) * hd + i]) -
                                r1) < 3e-2f);
                CHECK(std::fabs(
                          __half2float(oq[(t * nh + head) * hd + i + half]) -
                          r2) < 3e-2f);
            }
    for (int t = 0; t < seq; ++t)
        for (int head = 0; head < nkv; ++head)
            for (int i = 0; i < half; ++i) {
                const float freq = std::pow(
                    static_cast<float>(theta),
                    -2.0f * i / static_cast<float>(hd));
                const float ang = static_cast<float>(pos[t]) * freq;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x1 = k[(t * nkv + head) * hd + i];
                const float x2 = k[(t * nkv + head) * hd + i + half];
                const float r1 = x1 * c - x2 * s;
                const float r2 = x1 * s + x2 * c;
                CHECK(std::fabs(__half2float(ok[(t * nkv + head) * hd + i]) -
                                r1) < 3e-2f);
                CHECK(std::fabs(
                          __half2float(ok[(t * nkv + head) * hd + i + half]) -
                          r2) < 3e-2f);
            }

    cudaFree(dq); cudaFree(dk); cudaFree(dp);
}

TEST_CASE("granite causal GQA attention kernel matches a CPU reference",
          "[granite][kernels]")
{
    const int seq = 3, nh = 2, nkv = 1, hd = 4;
    const float scale = 0.5f;
    int64_t seed = 4242;
    std::vector<float> q(seq * nh * hd), k(seq * nkv * hd), v(seq * nkv * hd);
    for (auto &x : q) x = lcg(seed);
    for (auto &x : k) x = lcg(seed);
    for (auto &x : v) x = lcg(seed);

    auto hq = ToHalf(q), hk = ToHalf(k), hv = ToHalf(v);
    __half *dq, *dk, *dv, *dout;
    REQUIRE(cudaMalloc(&dq, hq.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, hk.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dv, hv.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hq.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, hq.data(), hq.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dv, hv.data(), hv.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(RunGraniteAttentionKernel(nullptr, dq, dk, dv, dout, seq, nh, nkv,
                                      hd, scale, 0) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> out(hq.size());
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    const int group = nh / nkv;
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / group;
            std::vector<float> sc(t + 1);
            float m = -1e30f;
            for (int s = 0; s <= t; ++s) {
                float dot = 0.f;
                for (int d = 0; d < hd; ++d)
                    dot += q[(t * nh + h) * hd + d] *
                           k[(s * nkv + kvh) * hd + d];
                sc[s] = dot * scale;
                m = std::max(m, sc[s]);
            }
            float sum = 0.f;
            for (int s = 0; s <= t; ++s) {
                sc[s] = std::exp(sc[s] - m);
                sum += sc[s];
            }
            for (int d = 0; d < hd; ++d) {
                float acc = 0.f;
                for (int s = 0; s <= t; ++s)
                    acc += sc[s] * v[(s * nkv + kvh) * hd + d];
                const float ref = acc / sum;
                INFO("t=" << t << " h=" << h << " d=" << d);
                CHECK(std::fabs(__half2float(out[(t * nh + h) * hd + d]) -
                                ref) < 3e-2f);
            }
        }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}

// ── Decode-shape NVFP4 GEMV bandwidth budget (perf lever #1) ────────────────
// Granite decode is 86% seq==1 NVFP4 GEMV (nsys, 2026-09-22): the aggregate
// weight-streaming rate is the model's serial bottleneck. This budget pins the
// effective GB/s across the REAL Granite decode shapes so a dispatch/kernel
// regression is caught here instead of only end-to-end. L2 is flushed between
// launches (the real decode walks a ~16 GB weight set, never L2-resident).
TEST_CASE("granite seq1 GEMV decode-shape aggregate bandwidth budget",
          "[granite][kernels][nvfp4][budget]")
{
#if defined(NDEBUG)
    
        lsxg::SetLsxTcFastMath(true);
        lsxg::SetLsxSeq1Gemv(true);

    struct Shape { int K, N; const char *name; };
    // q/o, fused gate_up, down, head — the warmup-free warprow set.
    const Shape shapes[] = {
        {4096, 4096, "q_o"},      {4096, 65536, "gate_up"},
        {32768, 4096, "down"},    {4096, 100352, "head"},
    };
    constexpr int kIters = 30;
    constexpr size_t kFlushBytes = 128u << 20;  // > 5090 L2 (96 MiB)
    uint8_t *flush = nullptr;
    REQUIRE(cudaMalloc(&flush, kFlushBytes) == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    double total_bytes = 0.0, total_us = 0.0;
    std::string report;
    for (const auto &c : shapes) {
        const size_t pbytes = static_cast<size_t>(c.K) * c.N / 2;
        const size_t sbytes = static_cast<size_t>(c.K) * c.N / 16;
        std::vector<__half> hx(static_cast<size_t>(c.K));
        std::vector<uint8_t> hp(pbytes), hs(sbytes);
        for (int i = 0; i < c.K; ++i)
            hx[static_cast<size_t>(i)] = __float2half(float((i % 13) - 6) * 0.25f);
        for (size_t i = 0; i < pbytes; ++i)
            hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
        for (size_t i = 0; i < sbytes; ++i)
            hs[i] = static_cast<uint8_t>(128 + (i % 48));
        __half *dx = nullptr, *dy = nullptr;
        uint8_t *dp = nullptr, *ds = nullptr;
        REQUIRE(cudaMalloc(&dx, static_cast<size_t>(c.K) * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dp, pbytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&ds, sbytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, static_cast<size_t>(c.N) * 2) == cudaSuccess);
        cudaMemcpy(dx, hx.data(), static_cast<size_t>(c.K) * 2,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dp, hp.data(), pbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), sbytes, cudaMemcpyHostToDevice);
        REQUIRE(lsxg::RunLsxNvfp4GemmSeq1GemmV(nullptr, dx, dp, ds, 0.125f, dy, c.K,
                                         c.N, false) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        double shape_us = 0.0;
        for (int i = 0; i < kIters; ++i) {
            REQUIRE(cudaMemsetAsync(flush, 0, kFlushBytes) == cudaSuccess);
            REQUIRE(cudaEventRecord(e0) == cudaSuccess);
            REQUIRE(lsxg::RunLsxNvfp4GemmSeq1GemmV(nullptr, dx, dp, ds, 0.125f, dy,
                                             c.K, c.N, false) == cudaSuccess);
            REQUIRE(cudaEventRecord(e1) == cudaSuccess);
            REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
            float ms = 0.0f;
            REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
            shape_us += static_cast<double>(ms) * 1000.0;
        }
        const double gbps =
            static_cast<double>(pbytes + sbytes) * kIters /
            (shape_us * 1e-6) / 1e9;
        report += std::string(c.name) + " (" + std::to_string(c.K) + "x" +
                  std::to_string(c.N) + ") = " + std::to_string(gbps) +
                  " GB/s\n";
        total_us += shape_us;
        total_bytes += static_cast<double>(pbytes + sbytes) * kIters;
        cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
    }
    const double agg = total_bytes / (total_us * 1e-6) / 1e9;
    INFO(report << "aggregate = " << agg << " GB/s");
    // Measured 2026-09-22 (RTX 5090): 812 GB/s (one-warp-per-row everywhere)
    // -> 873 (occupancy floor) -> 918 (128-bit activation loads + half22float2
    // in the slice/warp-row kernels). Raw streaming ceiling on this part =
    // ~1618 GB/s. Budget sits just below the current figure so a silent
    // regression to the low-occupancy / scalar-x launch fails here.
    CHECK(agg > 900.0);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(flush);
        lsxg::SetLsxTcFastMath(false);
        lsxg::SetLsxSeq1Gemv(false);
#endif
}
