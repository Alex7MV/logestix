// Gemma-4-31B-it (dense) kernel tests: every device kernel against a CPU FP64
// reference derived from transformers/models/gemma4 (Model Ingestion Protocol
// step 5/6).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "lsxcommon/gemma_dense_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace {

using lsxcommon::gemma_dense::RunGemmaDenseAttention;
using lsxcommon::gemma_dense::RunGemmaDenseAttentionSplit;
using lsxcommon::gemma_dense::RunGemmaDenseEmbedGatherNvfp4;
using lsxcommon::gemma_dense::RunGemmaDenseGeGLU;
using lsxcommon::gemma_dense::RunGemmaDenseHeadNorm;
using lsxcommon::gemma_dense::RunGemmaDenseLogitsArgmax;
using lsxcommon::gemma_dense::RunGemmaDenseRMSNorm;
using lsxcommon::gemma_dense::RunGemmaDenseRope;

std::mt19937 g_rng(20261009);
std::normal_distribution<float> g_nd(0.f, 1.f);

std::vector<__half> RandomHalves(size_t n)
{
    std::vector<__half> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = __float2half(g_nd(g_rng));
    return v;
}

bool Close(float got, double want, double tol = 3e-2)
{
    return std::abs(static_cast<double>(got) - want) <=
           std::max(tol, tol * std::abs(want));
}

std::vector<float> ToFloat(const std::vector<__half> &h)
{
    std::vector<float> v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = __half2float(h[i]);
    return v;
}

// Gemma4RMSNorm: x * (mean(x^2) + eps)^(-1/2), then * w (no +1 offset —
// Gemma-4 dropped the Gemma-2/3 `1 + w` form).
std::vector<double> RefRmsNorm(const std::vector<float> &x, int rows,
                               int hidden, float eps,
                               const std::vector<float> *w)
{
    std::vector<double> out(static_cast<size_t>(rows) * hidden);
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int d = 0; d < hidden; ++d) {
            const double v = x[static_cast<size_t>(r) * hidden + d];
            ss += v * v;
        }
        const double inv = std::pow(ss / hidden + eps, -0.5);
        for (int d = 0; d < hidden; ++d) {
            double v = x[static_cast<size_t>(r) * hidden + d] * inv;
            if (w) v *= (*w)[static_cast<size_t>(d)];
            out[static_cast<size_t>(r) * hidden + d] = v;
        }
    }
    return out;
}

// Gemma-4 "proportional"/default RoPE: cos/sin span the WHOLE head dim and
// rotate_half pairs (i, i + head_dim/2); only the first `angles` frequency
// pairs carry a frequency theta^(-2i/head_dim), the rest are zero (cos=1,
// sin=0). A sliding layer is the `angles == head_dim/2` case of the same form.
std::vector<double> RefRope(const std::vector<float> &x, const int *pos,
                            int seq, int heads, int head_dim, int angles,
                            double theta)
{
    std::vector<double> out(x.size());
    const int half = head_dim / 2;
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const size_t row =
                (static_cast<size_t>(t) * heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                const double c = i < angles
                                     ? std::cos(pos[t] * std::pow(
                                             theta, -2.0 * i / head_dim))
                                     : 1.0;
                const double s = i < angles
                                     ? std::sin(pos[t] * std::pow(
                                             theta, -2.0 * i / head_dim))
                                     : 0.0;
                const double a = x[row + i];
                const double b = x[row + i + half];
                out[row + i] = a * c - b * s;
                out[row + i + half] = b * c + a * s;
            }
        }
    return out;
}

// Causal (+ optional sliding band) GQA attention with attention scale applied
// as given (Gemma-4 uses 1.0). Cache is pos-major [pos][head][dim].
std::vector<double> RefAttention(const std::vector<float> &q,
                                 const std::vector<float> &k,
                                 const std::vector<float> &v, int seq, int nh,
                                 int nkv, int hd, int window, float scale,
                                 int pos_base, int slots)
{
    std::vector<double> out(static_cast<size_t>(seq) * nh * hd, 0.0);
    const int group = nh / nkv;
    for (int t = 0; t < seq; ++t) {
        const int qpos = pos_base + t;
        for (int h = 0; h < nh; ++h) {
            const int kvh = h / group;
            const int lo = window > 0 ? std::max(0, qpos - window + 1) : 0;
            std::vector<double> sc;
            std::vector<int> idx;
            for (int s = lo; s <= qpos && s < slots; ++s) {
                double dot = 0.0;
                for (int d = 0; d < hd; ++d)
                    dot += static_cast<double>(
                               q[(static_cast<size_t>(t) * nh + h) * hd + d]) *
                           k[(static_cast<size_t>(s) * nkv + kvh) * hd + d];
                sc.push_back(dot * scale);
                idx.push_back(s);
            }
            double m = -1e30;
            for (double x : sc) m = std::max(m, x);
            double sum = 0.0;
            std::vector<double> ex(sc.size());
            for (size_t i = 0; i < sc.size(); ++i) {
                ex[i] = std::exp(sc[i] - m);
                sum += ex[i];
            }
            for (int d = 0; d < hd; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < ex.size(); ++i)
                    acc += ex[i] * v[(static_cast<size_t>(idx[i]) * nkv +
                                      kvh) * hd + d];
                out[(static_cast<size_t>(t) * nh + h) * hd + d] = acc / sum;
            }
        }
    }
    return out;
}

// ACT2FN["gelu_pytorch_tanh"]: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))).
double RefGeluTanh(double x)
{
    constexpr double kSqrt2OverPi = 0.7978845608028654;
    return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

// NVFP4 E2M1 value table (canonical codec, mirrors nvfp4_codec.h).
float E2M1(int nib)
{
    static const float kVal[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                   4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                   -2.0f, -3.0f, -4.0f, -6.0f};
    return kVal[nib & 0xF];
}

float E4M3(uint8_t b)
{
    if ((b & 0x7F) == 0) return 0.f;
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0x0F;
    const int m = b & 0x07;
    if (e == 0) return s ? -m * 0.001953125f : m * 0.001953125f;
    float v = std::ldexp(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

} // namespace

TEST_CASE("gemma dense RMSNorm matches the Gemma4RMSNorm reference",
          "[gemma_dense][kernels]") {
    constexpr int rows = 3, hidden = 512;
    const auto hx = RandomHalves(rows * hidden);
    const auto hw = RandomHalves(hidden);
    const auto x = ToFloat(hx), w = ToFloat(hw);
    const float eps = 1e-6f;

    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    SECTION("with weight (the four per-layer norms)")
    {
        REQUIRE(RunGemmaDenseRMSNorm(nullptr, dx, dw, dout, rows, hidden, eps) ==
                cudaSuccess);
        const auto ref = RefRmsNorm(x, rows, hidden, eps, &w);
        std::vector<__half> h(hx.size());
        REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < h.size(); ++i) {
            INFO("i=" << i << " got=" << __half2float(h[i])
                      << " want=" << ref[i]);
            CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
        }
    }
    SECTION("without weight (v_norm)")
    {
        REQUIRE(RunGemmaDenseRMSNorm(nullptr, dx, nullptr, dout, rows, hidden,
                                     eps) == cudaSuccess);
        const auto ref = RefRmsNorm(x, rows, hidden, eps, nullptr);
        std::vector<__half> h(hx.size());
        REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < h.size(); ++i)
            CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
    }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

TEST_CASE("gemma dense per-head norm covers q_norm / k_norm / v_norm",
          "[gemma_dense][kernels]") {
    constexpr int seq = 4, heads = 3, hd = 32;
    const auto hx = RandomHalves(seq * heads * hd);
    const auto hw = RandomHalves(hd);
    const auto x = ToFloat(hx), w = ToFloat(hw);

    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Each (row, head) is its own RMSNorm over head_dim - NOT over the row.
    const float eps = 1e-6f;
    const auto ref_w = RefRmsNorm(x, seq * heads, hd, eps, &w);
    REQUIRE(RunGemmaDenseHeadNorm(nullptr, dx, dw, dout, seq, heads, hd, eps) ==
            cudaSuccess);
    std::vector<__half> h(hx.size());
    REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < h.size(); ++i) {
        INFO("i=" << i << " got=" << __half2float(h[i])
                  << " want=" << ref_w[i]);
        CHECK(Close(__half2float(h[i]), ref_w[i], 6e-3));
    }

    // Plain form (v_norm: RMSNorm with_scale=False).
    const auto ref_p = RefRmsNorm(x, seq * heads, hd, eps, nullptr);
    REQUIRE(RunGemmaDenseHeadNorm(nullptr, dx, nullptr, dout, seq, heads, hd,
                                  eps) == cudaSuccess);
    REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < h.size(); ++i)
        CHECK(Close(__half2float(h[i]), ref_p[i], 6e-3));

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
}

TEST_CASE("gemma dense RoPE is HF-exact for BOTH layer kinds",
          "[gemma_dense][kernels]") {
    // Sliding layers: default rope, angles == head_dim/2 (full rotation).
    // Global layers: "proportional" 0.25 -> angles == head_dim/8, the middle
    // of the head untouched and the frequencies keyed to the FULL head dim.
    const int cases[][5] = {
        // seq, heads, head_dim, angles, layer
        {5, 4, 64, 32, 0},    // sliding
        {3, 2, 64, 8, 1},     // global (0.25 * 64 / 2)
    };
    for (const auto &c : cases) {
        const int seq = c[0], heads = c[1], hd = c[2], angles = c[3];
        const double theta = c[4] == 0 ? 10000.0 : 1000000.0;
        CAPTURE(seq, heads, hd, angles, theta);

        const auto hq = RandomHalves(seq * heads * hd);
        const auto hk = RandomHalves(seq * heads * hd);
        const auto q = ToFloat(hq), k = ToFloat(hk);
        std::vector<int> pos(seq);
        for (int i = 0; i < seq; ++i) pos[static_cast<size_t>(i)] = 3 + i;

        int32_t *dpos;
        __half *dq, *dk;
        REQUIRE(cudaMalloc(&dpos, seq * 4) == cudaSuccess);
        REQUIRE(cudaMalloc(&dq, hq.size() * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dk, hk.size() * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(dpos, pos.data(), seq * 4,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(dq, hq.data(), hq.size() * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(dk, hk.data(), hk.size() * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);

        REQUIRE(RunGemmaDenseRope(nullptr, dpos, dq, dk, seq, heads, heads, hd,
                                   angles, theta) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const auto ref_q = RefRope(q, pos.data(), seq, heads, hd, angles, theta);
        const auto ref_k = RefRope(k, pos.data(), seq, heads, hd, angles, theta);
        std::vector<__half> hq2(hq.size()), hk2(hk.size());
        REQUIRE(cudaMemcpy(hq2.data(), dq, hq2.size() * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(hk2.data(), dk, hk2.size() * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        for (size_t i = 0; i < hq.size(); ++i) {
            INFO("q i=" << i);
            CHECK(Close(__half2float(hq2[i]), ref_q[i], 6e-3));
        }
        for (size_t i = 0; i < hk.size(); ++i) {
            INFO("k i=" << i);
            CHECK(Close(__half2float(hk2[i]), ref_k[i], 6e-3));
        }
        cudaFree(dpos);
        cudaFree(dq);
        cudaFree(dk);
    }
}

TEST_CASE("gemma dense attention: causal, GQA and the sliding band",
          "[gemma_dense][kernels]") {
    constexpr int nh = 4, nkv = 2, hd = 16, slots = 12;
    for (const int window : {0, 4}) {
        for (const int seq : {1, 3}) {
            CAPTURE(window, seq);
            const int pos_base = 5;
            const auto hq = RandomHalves(seq * nh * hd);
            const auto hk = RandomHalves(slots * nkv * hd);
            const auto hv = RandomHalves(slots * nkv * hd);
            const auto q = ToFloat(hq), k = ToFloat(hk), v = ToFloat(hv);

            __half *dq, *dk, *dv, *dout;
            REQUIRE(cudaMalloc(&dq, hq.size() * 2) == cudaSuccess);
            REQUIRE(cudaMalloc(&dk, hk.size() * 2) == cudaSuccess);
            REQUIRE(cudaMalloc(&dv, hv.size() * 2) == cudaSuccess);
            REQUIRE(cudaMalloc(&dout, hq.size() * 2) == cudaSuccess);
            REQUIRE(cudaMemcpy(dq, hq.data(), hq.size() * 2,
                               cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaMemcpy(dk, hk.data(), hk.size() * 2,
                               cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaMemcpy(dv, hv.data(), hv.size() * 2,
                               cudaMemcpyHostToDevice) == cudaSuccess);

            const auto geo = lsxcommon::KvCacheGeometry::Make(
                lsxcommon::KvCacheLayout::kPosMajor, slots, nkv, hd, 1);
            const auto vgeo = geo;

            REQUIRE(RunGemmaDenseAttention(nullptr, dq, dk, dv, dout, seq, nh,
                                           nkv, hd, window, 1.0f, pos_base,
                                           geo, vgeo) == cudaSuccess);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
            const auto ref = RefAttention(q, k, v, seq, nh, nkv, hd, window,
                                          1.0f, pos_base, slots);
            std::vector<__half> h(hq.size());
            REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            for (size_t i = 0; i < h.size(); ++i) {
                INFO("i=" << i);
                CHECK(Close(__half2float(h[i]), ref[i], 2e-2));
            }
            cudaFree(dq);
            cudaFree(dk);
            cudaFree(dv);
            cudaFree(dout);
        }
    }
}

TEST_CASE("split-KV decode attention agrees with the unsplit kernel",
          "[gemma_dense][kernels]") {
    constexpr int nh = 8, nkv = 4, hd = 32, seq = 1, slots = 300;
    const int pos_base = 299;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(slots * nkv * hd);
    const auto hv = RandomHalves(slots * nkv * hd);
    const auto geo = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, slots, nkv, hd, 1);

    __half *dq, *dk, *dv, *d1, *d2;
    float *partials = nullptr;
    REQUIRE(cudaMalloc(&dq, hq.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, hk.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dv, hv.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d1, hq.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d2, hq.size() * 2) == cudaSuccess);
    const size_t pbytes =
        static_cast<size_t>(32) * seq * nh * (hd + 2) * sizeof(float);
    REQUIRE(cudaMalloc(&partials, pbytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, hq.data(), hq.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dk, hk.data(), hk.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dv, hv.data(), hv.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // A sliding window over a long context is the case that must not read
    // out-of-band rows.
    const int window = 128;
    REQUIRE(RunGemmaDenseAttention(nullptr, dq, dk, dv, d1, seq, nh, nkv, hd,
                                   window, 1.0f, pos_base, geo, geo) ==
            cudaSuccess);
    REQUIRE(RunGemmaDenseAttentionSplit(nullptr, dq, dk, dv, d2, partials, seq,
                                        nh, nkv, hd, window, 1.0f, pos_base, 16,
                                        geo, geo) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto ref = RefAttention(ToFloat(hq), ToFloat(hk), ToFloat(hv), seq,
                                  nh, nkv, hd, window, 1.0f, pos_base, slots);
    std::vector<__half> h1(hq.size()), h2(hq.size());
    REQUIRE(cudaMemcpy(h1.data(), d1, h1.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(h2.data(), d2, h2.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < h1.size(); ++i) {
        INFO("i=" << i);
        CHECK(Close(__half2float(h1[i]), ref[i], 3e-2));
        CHECK(Close(__half2float(h2[i]), ref[i], 3e-2));
    }
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(d1);
    cudaFree(d2);
    cudaFree(partials);
}

TEST_CASE("gemma dense GeGLU matches gelu_pytorch_tanh", "[gemma_dense][kernels]") {
    constexpr int I = 64, seq = 3;
    const auto hg = RandomHalves(seq * I);
    const auto hu = RandomHalves(seq * I);
    const auto g = ToFloat(hg), u = ToFloat(hu);

    __half *dg, *du, *dout;
    REQUIRE(cudaMalloc(&dg, hg.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, hu.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hg.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dg, hg.data(), hg.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(du, hu.data(), hu.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunGemmaDenseGeGLU(nullptr, dg, du, dout,
                               static_cast<int64_t>(seq) * I) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> h(static_cast<size_t>(seq) * I);
    REQUIRE(cudaMemcpy(h.data(), dout, h.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int t = 0; t < seq; ++t)
        for (int i = 0; i < I; ++i) {
            const size_t idx = static_cast<size_t>(t) * I + i;
            INFO("idx=" << idx << " got=" << __half2float(h[idx])
                      << " g=" << g[idx] << " u=" << u[idx]);
            const double want = RefGeluTanh(g[idx]) * u[idx];
            CHECK(Close(__half2float(h[idx]), want, 6e-3));
        }
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
}

TEST_CASE("gemma dense NVFP4 embed gather applies sqrt(hidden) once",
          "[gemma_dense][kernels]") {
    constexpr int vocab = 64, hidden = 128, seq = 5;
    // A tiny deterministic NVFP4 table: nibble pattern + E4M3 scale bytes.
    const size_t numel = static_cast<size_t>(vocab) * hidden;
    std::vector<uint8_t> packed(numel / 2);
    std::vector<uint8_t> scales(numel / 16);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(i * 7 + 1);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = 0x38;  // 1.0
    const float gs = 2.0f;
    const float embed_scale = 73.5f;

    std::vector<int32_t> ids(seq);
    for (int i = 0; i < seq; ++i) ids[static_cast<size_t>(i)] = (i * 11) % vocab;

    uint8_t *dp, *ds;
    int32_t *dids;
    __half *dout;
    REQUIRE(cudaMalloc(&dp, packed.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, scales.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dids, seq * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, numel * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, packed.data(), packed.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds, scales.data(), scales.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dids, ids.data(), seq * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(RunGemmaDenseEmbedGatherNvfp4(nullptr, dp, ds, gs, dids, dout, seq,
                                          vocab, hidden, embed_scale) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> h(numel);
    REQUIRE(cudaMemcpy(h.data(), dout, numel * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < hidden; ++d) {
            const size_t flat =
                static_cast<size_t>(ids[static_cast<size_t>(t)]) * hidden + d;
            const uint8_t byte = packed[flat >> 1];
            const int nib = (flat & 1) ? (byte >> 4) : (byte & 0xF);
            const double want =
                E2M1(nib) * E4M3(scales[flat >> 4]) * gs * embed_scale;
            INFO("t=" << t << " d=" << d);
            CHECK(Close(__half2float(h[static_cast<size_t>(t) * hidden + d]),
                        want, 1e-2));
        }
    cudaFree(dp);
    cudaFree(ds);
    cudaFree(dids);
    cudaFree(dout);
}

TEST_CASE("gemma dense logits are softcapped and argmax is deterministic",
          "[gemma_dense][kernels]") {
    constexpr int vocab = 256, hidden = 64;
    const size_t numel = static_cast<size_t>(vocab) * hidden;
    std::vector<uint8_t> packed(numel / 2), scales(numel / 16, 0x38);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<uint8_t>((i * 13) % 16);
    const float gs = 1.0f;
    const auto hx = RandomHalves(hidden);
    const auto x = ToFloat(hx);

    uint8_t *dp, *ds;
    __half *dx;
    int32_t *darg;
    float *dmax, *dlogits;
    __half *dlogits_h;
    REQUIRE(cudaMalloc(&dp, packed.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, scales.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, hidden * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&darg, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dmax, 4) == cudaSuccess);
    // vocab floats PLUS ceil(vocab/128)*2 trailing words for the argmax
    // block winners (see RunGemmaDenseLogitsArgmax).
    const size_t arg_tail = (vocab + 127) / 128 * 2;
    REQUIRE(cudaMalloc(&dlogits, (vocab + arg_tail) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&dlogits_h, vocab * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, packed.data(), packed.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds, scales.data(), scales.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hidden * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const float cap = 30.0f;
    REQUIRE(RunGemmaDenseLogitsArgmax(nullptr, dp, ds, gs, dx, vocab, hidden,
                                      cap, darg, dmax, dlogits,
                                      dlogits_h) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> logits(vocab);
    int32_t arg = -1;
    float maxv = 0.f;
    REQUIRE(cudaMemcpy(logits.data(), dlogits, vocab * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(&arg, darg, 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, dmax, 4, cudaMemcpyDeviceToHost) == cudaSuccess);

    int ref_arg = -1;
    // The running maximum stays in DOUBLE: rounding it to float per step
    // makes an all-equal row advance the winner on every comparison (the
    // rounded value can sit below the exact one), and the "lowest index wins a
    // tie" contract silently becomes "last index wins".
    double ref_max = -1e30;
    std::vector<double> ref_logits(vocab);
    for (int v = 0; v < vocab; ++v) {
        double acc = 0.0;
        for (int d = 0; d < hidden; ++d) {
            const size_t flat = static_cast<size_t>(v) * hidden + d;
            const uint8_t byte = packed[flat >> 1];
            const int nib = (flat & 1) ? (byte >> 4) : (byte & 0xF);
            acc += static_cast<double>(x[static_cast<size_t>(d)]) *
                   E2M1(nib) * E4M3(scales[flat >> 4]) * gs;
        }
        const double capped = std::tanh(acc / cap) * cap;
        ref_logits[static_cast<size_t>(v)] = capped;
        if (capped > ref_max) {
            ref_max = capped;
            ref_arg = v;
        }
    }
    CHECK(arg == ref_arg);
    CHECK(Close(maxv, ref_max, 5e-2));
    // Every logit must be the softcapped value, and inside the cap.
    for (int v = 0; v < vocab; ++v) {
        INFO("v=" << v);
        CHECK(logits[static_cast<size_t>(v)] <= cap);
        CHECK(logits[static_cast<size_t>(v)] >= -cap);
        CHECK(Close(logits[static_cast<size_t>(v)],
                    ref_logits[static_cast<size_t>(v)], 5e-2));
    }

    cudaFree(dp);
    cudaFree(ds);
    cudaFree(dx);
    cudaFree(darg);
    cudaFree(dmax);
    cudaFree(dlogits);
    cudaFree(dlogits_h);
}

TEST_CASE("gemma dense head norm: real-shape, zero rows, in place",
          "[gemma_dense][kernels]") {
    // The model calls the head norm IN PLACE (x == out) at exactly this shape
    // (sliding geometry: seq rows x 32 heads x 256). A zero row must yield a
    // zero row (not NaN), which is the property the real checkpoint exercised.
    const int seq = 18, heads = 32, hd = 256;
    const size_t n = static_cast<size_t>(seq) * heads * hd;
    std::vector<__half> x(n), w(hd);
    for (size_t i = 0; i < n; ++i)
        x[i] = __float2half(static_cast<float>(static_cast<int>(i % 977) - 488) /
                            50.f);
    // Rows 16..17 are all-zero, the exact input that previously came back NaN.
    for (size_t i = 16 * static_cast<size_t>(heads) * hd; i < n; ++i)
        x[i] = __float2half(0.0f);
    for (int i = 0; i < hd; ++i)
        w[i] = __float2half(1.0f + 0.001f * static_cast<float>(i));

    __half *dx = nullptr, *dw = nullptr;
    REQUIRE(cudaMalloc(&dx, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hd * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, x.data(), n * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dw, w.data(), hd * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(RunGemmaDenseHeadNorm(nullptr, dx, dw, dx, seq, heads, hd, 1e-6f) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> h(n);
    REQUIRE(cudaMemcpy(h.data(), dx, n * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    const auto xf = ToFloat(x), wf = ToFloat(w);
    const auto ref = RefRmsNorm(xf, seq * heads, hd, 1e-6f, &wf);
    for (size_t i = 0; i < n; ++i) {
        const float f = __half2float(h[i]);
        REQUIRE_FALSE(std::isnan(f));
        REQUIRE_FALSE(std::isinf(f));
    }
    for (size_t i = 0; i < n; i += 97)
        CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
    cudaFree(dx);
    cudaFree(dw);
}

// The per-tensor `global_scale` of the tied table must reach the logits dot.
//
// A fixture with gs == 1 cannot see a dropped scale: multiplied by one, every
// reference agrees. That is exactly how gemma-4-26b shipped 7.01x-hot logits
// once. The 12B subtree's `global_scale` is not 1 (the quantizer scales each
// tensor by its own amax), so a dropped factor would move every logit by a
// constant ratio and flip the fork the FSI gate sits on.
TEST_CASE("gemma dense logits apply the table's global_scale (gs != 1)",
          "[gemma_dense][kernels]") {
    constexpr int vocab = 64, hidden = 64;
    const size_t numel = static_cast<size_t>(vocab) * hidden;
    std::vector<uint8_t> packed(numel / 2), scales(numel / 16, 0x38);
    for (size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<uint8_t>((i * 5 + 3) % 16);
    const auto hx = RandomHalves(hidden);
    const auto x = ToFloat(hx);
    const float cap = 30.0f;
    const size_t arg_tail = (vocab + 127) / 128 * 2;

    uint8_t *dp, *ds;
    __half *dx, *dlogits_h;
    int32_t *darg;
    float *dmax, *dlogits;
    REQUIRE(cudaMalloc(&dp, packed.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, scales.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, hidden * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&darg, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dmax, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dlogits, (vocab + arg_tail) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&dlogits_h, vocab * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, packed.data(), packed.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds, scales.data(), scales.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hidden * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const auto run = [&](float gs, std::vector<float> *out) {
        REQUIRE(RunGemmaDenseLogitsArgmax(nullptr, dp, ds, gs, dx, vocab, hidden,
                                          cap, darg, dmax, dlogits,
                                          dlogits_h) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        out->assign(vocab, 0.f);
        REQUIRE(cudaMemcpy(out->data(), dlogits, vocab * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    };

    std::vector<float> one, three;
    run(1.0f, &one);
    run(3.5f, &three);

    // Reference with gs applied, per vocab row.
    for (int v = 0; v < vocab; ++v) {
        double acc = 0.0;
        for (int d = 0; d < hidden; ++d) {
            const size_t flat = static_cast<size_t>(v) * hidden + d;
            const uint8_t byte = packed[flat >> 1];
            const int nib = (flat & 1) ? (byte >> 4) : (byte & 0xF);
            acc += static_cast<double>(x[static_cast<size_t>(d)]) * E2M1(nib) *
                   E4M3(scales[flat >> 4]);
        }
        const double want = std::tanh(acc * 3.5 / cap) * cap;
        INFO("v=" << v);
        CHECK(Close(three[static_cast<size_t>(v)], want, 3e-2));
    }
    // And the two runs must actually DIFFER: a dropped scale makes them equal.
    double max_delta = 0.0;
    for (int v = 0; v < vocab; ++v)
        max_delta = std::max(max_delta,
                             std::abs(static_cast<double>(
                                          three[static_cast<size_t>(v)] -
                                          one[static_cast<size_t>(v)])));
    INFO("max |logits(gs=3.5) - logits(gs=1)| = " << max_delta);
    CHECK(max_delta > 1e-2);

    cudaFree(dp);
    cudaFree(ds);
    cudaFree(dx);
    cudaFree(darg);
    cudaFree(dmax);
    cudaFree(dlogits);
    cudaFree(dlogits_h);
}

// ── The residual plane's arithmetic ────────────────────────────────────────
//
// The decode runs the base GEMV, then ONE more GEMV of the residual plane with
// acc=true into the same output, so the arithmetic is "base dot rounds to F16,
// then the residual dot adds and rounds once more". This pins that composition
// against a host reference that rounds between the two dots — the engine must
// not fuse them (a fused dot would be a different numeric class and would move
// the FSI fork the plane was added to fix).
TEST_CASE("gemma dense residual plane accumulates after the base dot",
          "[gemma_dense][kernels]") {
    constexpr int K = 128, N = 32;
    // Deterministic nibble patterns for base and residual, unit block scales.
    auto make_plane = [&](int seed, std::vector<uint8_t> *packed,
                          std::vector<uint8_t> *scales) {
        packed->assign(static_cast<size_t>(N) * K / 2, 0);
        scales->assign(static_cast<size_t>(N) * K / 16, 0x38);  // 1.0
        for (size_t i = 0; i < packed->size(); ++i)
            (*packed)[i] = static_cast<uint8_t>((i * 7 + seed) % 16 | (((i * 3 + seed) % 16) << 4));
    };
    std::vector<uint8_t> bp, bs, rp, rs;
    make_plane(1, &bp, &bs);
    make_plane(5, &rp, &rs);
    const float bgs = 0.75f, rgs = 0.25f;
    const auto hx = RandomHalves(K);
    const auto x = ToFloat(hx);

    uint8_t *dbp, *dbs, *drp, *drs;
    __half *dx, *dy;
    REQUIRE(cudaMalloc(&dbp, bp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dbs, bs.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&drp, rp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&drs, rs.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, K * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dy, static_cast<size_t>(N) * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dbp, bp.data(), bp.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dbs, bs.data(), bs.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(drp, rp.data(), rp.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(drs, rs.data(), rs.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), K * 2, cudaMemcpyHostToDevice) == cudaSuccess);

    // base (acc=false), then residual (acc=true) — the decode's composition.
    REQUIRE(lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, dx, dbp, dbs, bgs,
                                                  dy, 1, K, N, false) ==
            cudaSuccess);
    REQUIRE(lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(nullptr, dx, drp, drs, rgs,
                                                  dy, 1, K, N, true) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> got(N);
    REQUIRE(cudaMemcpy(got.data(), dy, static_cast<size_t>(N) * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int n = 0; n < N; ++n) {
        const auto dot = [&](const std::vector<uint8_t> &pk,
                             const std::vector<uint8_t> &sc, float gs) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const size_t flat = static_cast<size_t>(n) * K + k;
                const uint8_t byte = pk[flat >> 1];
                const int nib = (flat & 1) ? (byte >> 4) : (byte & 0xF);
                acc += static_cast<double>(x[size_t(k)]) * E2M1(nib) *
                       E4M3(sc[flat >> 4]) * gs;
            }
            return acc;
        };
        // Round to F16 between the two dots: that is the composition under test.
        double want = static_cast<double>(__half2float(__float2half(dot(bp, bs, bgs))));
        want = static_cast<double>(__half2float(
            __float2half(static_cast<float>(want + dot(rp, rs, rgs)))));
        INFO("n=" << n);
        CHECK(Close(__half2float(got[static_cast<size_t>(n)]), want, 3e-2));
    }
    cudaFree(dbp);
    cudaFree(dbs);
    cudaFree(drp);
    cudaFree(drs);
    cudaFree(dx);
    cudaFree(dy);
}
