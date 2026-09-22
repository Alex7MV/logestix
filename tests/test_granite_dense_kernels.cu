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
