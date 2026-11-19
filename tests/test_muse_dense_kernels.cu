// Muse-Glimmer-30B kernel tests: every device kernel against a CPU FP64
// reference derived from transformers/models/muse_glimmer (Model Ingestion
// Protocol step 5/6). The four facts that are easy to get wrong each have a
// case: centered (1+w) vs plain norms, the sigmoid attention gate applied
// BEFORE o_proj, the sliding band / NoPE split, and the softcap with the
// output multiplier on the FP16 head.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "lsxcommon/muse_dense_inference.h"
#include "lsxcommon/lsx_kv_cache.h"

namespace {

using namespace lsxcommon::muse_dense;

std::mt19937 g_rng(20261118);
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

// Gemma2RMSNorm (centered): x * (mean(x^2)+eps)^(-1/2) * (1 + w).
std::vector<double> RefCenteredNorm(const std::vector<float> &x, int rows,
                                    int hidden, float eps,
                                    const std::vector<float> &w)
{
    std::vector<double> out(static_cast<size_t>(rows) * hidden);
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int d = 0; d < hidden; ++d) {
            const double v = x[static_cast<size_t>(r) * hidden + d];
            ss += v * v;
        }
        const double inv = std::pow(ss / hidden + eps, -0.5);
        for (int d = 0; d < hidden; ++d)
            out[static_cast<size_t>(r) * hidden + d] =
                x[static_cast<size_t>(r) * hidden + d] * inv *
                (1.0 + w[static_cast<size_t>(d)]);
    }
    return out;
}

// Gemma4RMSNorm (plain): x * (mean(x^2)+eps)^(-1/2) * w.
std::vector<double> RefPlainNorm(const std::vector<float> &x, int rows,
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

// Weightless per-head QK norm: q *= qk_scale_factor after the norm, k plain.
std::vector<double> RefQkNorm(const std::vector<float> &x, int rows, int heads,
                              int hd, float eps, float scale)
{
    std::vector<double> out(x.size());
    for (int r = 0; r < rows * heads; ++r) {
        double ss = 0.0;
        for (int d = 0; d < hd; ++d) {
            const double v = x[static_cast<size_t>(r) * hd + d];
            ss += v * v;
        }
        const double inv = std::pow(ss / hd + eps, -0.5);
        for (int d = 0; d < hd; ++d)
            out[static_cast<size_t>(r) * hd + d] =
                x[static_cast<size_t>(r) * hd + d] * inv * scale;
    }
    return out;
}

// HF rotate_half over the WHOLE head dim, frequency theta^(-2i/hd).
std::vector<double> RefRope(const std::vector<float> &x, const int *pos,
                            int seq, int heads, int hd, double theta)
{
    std::vector<double> out(x.size());
    const int half = hd / 2;
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const size_t row = (static_cast<size_t>(t) * heads + h) * hd;
            for (int i = 0; i < half; ++i) {
                const double ang = pos[t] * std::pow(theta, -2.0 * i / hd);
                const double c = std::cos(ang), s = std::sin(ang);
                const double a = x[row + i], b = x[row + i + half];
                out[row + i] = a * c - b * s;
                out[row + i + half] = b * c + a * s;
            }
        }
    return out;
}

// Causal (+ optional sliding band) GQA attention, pos-major cache.
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

double RefSilu(double x) { return x / (1.0 + std::exp(-x)); }
double RefSigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct DevBuf {
    void *p = nullptr;
    explicit DevBuf(size_t bytes) { REQUIRE(cudaMalloc(&p, bytes) == cudaSuccess); }
    ~DevBuf() { if (p) cudaFree(p); }
    template <typename T> T *as() { return static_cast<T *>(p); }
};

void Upload(void *d, const void *h, size_t bytes)
{
    REQUIRE(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
}

template <typename T>
std::vector<T> Download(const void *d, size_t n)
{
    std::vector<T> h(n);
    REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return h;
}

} // namespace

TEST_CASE("muse centered norm matches Gemma2RMSNorm (1+w)",
          "[muse_dense][kernels]") {
    constexpr int rows = 3, hidden = 512;
    const auto hx = RandomHalves(rows * hidden);
    // Non-zero weights: a plain-w kernel would differ from (1+w) here.
    const auto hw = RandomHalves(hidden);
    const auto x = ToFloat(hx), w = ToFloat(hw);
    const float eps = 1e-5f;

    DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dout(hx.size() * 2);
    Upload(dx.p, hx.data(), hx.size() * 2);
    Upload(dw.p, hw.data(), hw.size() * 2);
    REQUIRE(RunMuseDenseCenteredNorm(nullptr, dx.as<__half>(), dw.as<__half>(),
                                     dout.as<__half>(), rows, hidden, eps) ==
            cudaSuccess);
    const auto ref = RefCenteredNorm(x, rows, hidden, eps, w);
    const auto h = Download<__half>(dout.p, hx.size());
    for (size_t i = 0; i < h.size(); ++i) {
        INFO("i=" << i << " got=" << __half2float(h[i]) << " want=" << ref[i]);
        CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
    }
}

TEST_CASE("muse plain norm matches Gemma4RMSNorm (w and weightless)",
          "[muse_dense][kernels]") {
    constexpr int rows = 2, hidden = 256;
    const auto hx = RandomHalves(rows * hidden);
    const auto hw = RandomHalves(hidden);
    const auto x = ToFloat(hx), w = ToFloat(hw);
    const float eps = 1e-5f;
    DevBuf dx(hx.size() * 2), dw(hw.size() * 2), dout(hx.size() * 2);
    Upload(dx.p, hx.data(), hx.size() * 2);
    Upload(dw.p, hw.data(), hw.size() * 2);

    SECTION("final norm (plain x*w)") {
        REQUIRE(RunMuseDensePlainNorm(nullptr, dx.as<__half>(), dw.as<__half>(),
                                      dout.as<__half>(), rows, hidden, eps) ==
                cudaSuccess);
        const auto ref = RefPlainNorm(x, rows, hidden, eps, &w);
        const auto h = Download<__half>(dout.p, hx.size());
        for (size_t i = 0; i < h.size(); ++i)
            CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
    }
    SECTION("embed norm (weightless)") {
        REQUIRE(RunMuseDensePlainNorm(nullptr, dx.as<__half>(), nullptr,
                                      dout.as<__half>(), rows, hidden, eps) ==
                cudaSuccess);
        const auto ref = RefPlainNorm(x, rows, hidden, eps, nullptr);
        const auto h = Download<__half>(dout.p, hx.size());
        for (size_t i = 0; i < h.size(); ++i)
            CHECK(Close(__half2float(h[i]), ref[i], 6e-3));
    }
}

TEST_CASE("muse QK norm is weightless per-head with the Q scale",
          "[muse_dense][kernels]") {
    constexpr int seq = 4, nh = 4, nkv = 2, hd = 32;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(seq * nkv * hd);
    const auto q = ToFloat(hq), k = ToFloat(hk);
    const float eps = 1e-5f, qscale = 3.87f;
    DevBuf dq(hq.size() * 2), dk(hk.size() * 2);
    Upload(dq.p, hq.data(), hq.size() * 2);
    Upload(dk.p, hk.data(), hk.size() * 2);
    REQUIRE(RunMuseDenseQkNormScale(nullptr, dq.as<__half>(), dk.as<__half>(),
                                    seq, nh, nkv, hd, eps, qscale) ==
            cudaSuccess);
    const auto refq = RefQkNorm(q, seq, nh, hd, eps, qscale);
    const auto refk = RefQkNorm(k, seq, nkv, hd, eps, 1.0f);
    const auto oq = Download<__half>(dq.p, hq.size());
    const auto ok = Download<__half>(dk.p, hk.size());
    for (size_t i = 0; i < oq.size(); ++i)
        CHECK(Close(__half2float(oq[i]), refq[i], 6e-3));
    for (size_t i = 0; i < ok.size(); ++i)
        CHECK(Close(__half2float(ok[i]), refk[i], 6e-3));
}

TEST_CASE("muse RoPE rotates the whole head (sliding layers)",
          "[muse_dense][kernels]") {
    constexpr int seq = 5, nh = 4, nkv = 2, hd = 32;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(seq * nkv * hd);
    const auto q = ToFloat(hq), k = ToFloat(hk);
    const int pos[seq] = {0, 1, 7, 100, 2047};
    const double theta = 500000.0;
    DevBuf dq(hq.size() * 2), dk(hk.size() * 2), dpos(seq * 4);
    Upload(dq.p, hq.data(), hq.size() * 2);
    Upload(dk.p, hk.data(), hk.size() * 2);
    Upload(dpos.p, pos, seq * 4);
    REQUIRE(RunMuseDenseRope(nullptr, dpos.as<int32_t>(), dq.as<__half>(),
                             dk.as<__half>(), seq, nh, nkv, hd, theta) ==
            cudaSuccess);
    const auto refq = RefRope(q, pos, seq, nh, hd, theta);
    const auto refk = RefRope(k, pos, seq, nkv, hd, theta);
    const auto oq = Download<__half>(dq.p, hq.size());
    const auto ok = Download<__half>(dk.p, hk.size());
    for (size_t i = 0; i < oq.size(); ++i)
        CHECK(Close(__half2float(oq[i]), refq[i], 2e-2));
    for (size_t i = 0; i < ok.size(); ++i)
        CHECK(Close(__half2float(ok[i]), refk[i], 2e-2));
}

TEST_CASE("muse attention honours the sliding band and GQA",
          "[muse_dense][kernels]") {
    constexpr int seq = 1, nh = 8, nkv = 2, hd = 32;
    constexpr int slots = 40, window = 8, pos_base = 30;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(slots * nkv * hd);
    const auto hv = RandomHalves(slots * nkv * hd);
    const auto q = ToFloat(hq), k = ToFloat(hk), v = ToFloat(hv);
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)-class
    DevBuf dq(hq.size() * 2), dk(hk.size() * 2), dv(hv.size() * 2),
        dout(hq.size() * 2);
    Upload(dq.p, hq.data(), hq.size() * 2);
    Upload(dk.p, hk.data(), hk.size() * 2);
    Upload(dv.p, hv.data(), hv.size() * 2);
    const auto geo = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, slots, nkv, hd, 1);
    REQUIRE(RunMuseDenseAttention(nullptr, dq.as<__half>(), dk.as<__half>(),
                                  dv.as<__half>(), dout.as<__half>(), seq, nh,
                                  nkv, hd, window, scale, pos_base, geo,
                                  geo) == cudaSuccess);
    const auto ref = RefAttention(q, k, v, seq, nh, nkv, hd, window, scale,
                                  pos_base, slots);
    const auto o = Download<__half>(dout.p, hq.size());
    for (size_t i = 0; i < o.size(); ++i) {
        INFO("i=" << i << " got=" << __half2float(o[i]) << " want=" << ref[i]);
        CHECK(Close(__half2float(o[i]), ref[i], 3e-2));
    }
}

TEST_CASE("muse attention full prefix (NoPE layer, window 0)",
          "[muse_dense][kernels]") {
    constexpr int seq = 1, nh = 8, nkv = 2, hd = 32;
    constexpr int slots = 40, pos_base = 30;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(slots * nkv * hd);
    const auto hv = RandomHalves(slots * nkv * hd);
    const auto q = ToFloat(hq), k = ToFloat(hk), v = ToFloat(hv);
    const float scale = 0.08838834764831845f;
    DevBuf dq(hq.size() * 2), dk(hk.size() * 2), dv(hv.size() * 2),
        dout(hq.size() * 2);
    Upload(dq.p, hq.data(), hq.size() * 2);
    Upload(dk.p, hk.data(), hk.size() * 2);
    Upload(dv.p, hv.data(), hv.size() * 2);
    const auto geo = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, slots, nkv, hd, 1);
    REQUIRE(RunMuseDenseAttention(nullptr, dq.as<__half>(), dk.as<__half>(),
                                  dv.as<__half>(), dout.as<__half>(), seq, nh,
                                  nkv, hd, 0, scale, pos_base, geo, geo) ==
            cudaSuccess);
    const auto ref = RefAttention(q, k, v, seq, nh, nkv, hd, 0, scale,
                                  pos_base, slots);
    const auto o = Download<__half>(dout.p, hq.size());
    for (size_t i = 0; i < o.size(); ++i)
        CHECK(Close(__half2float(o[i]), ref[i], 3e-2));
}

TEST_CASE("muse attention gate is elementwise sigmoid BEFORE o_proj",
          "[muse_dense][kernels]") {
    constexpr int64_t n = 4096;
    const auto ha = RandomHalves(n);
    const auto hg = RandomHalves(n);
    const auto a = ToFloat(ha), g = ToFloat(hg);
    DevBuf da(n * 2), dg(n * 2), dout(n * 2);
    Upload(da.p, ha.data(), n * 2);
    Upload(dg.p, hg.data(), n * 2);
    REQUIRE(RunMuseDenseAttnGate(nullptr, da.as<__half>(), dg.as<__half>(),
                                 dout.as<__half>(), n) == cudaSuccess);
    const auto o = Download<__half>(dout.p, n);
    for (int64_t i = 0; i < n; ++i) {
        const double want = a[i] * RefSigmoid(g[i]);
        CHECK(Close(__half2float(o[i]), want, 6e-3));
    }
}

TEST_CASE("muse SwiGLU is silu(gate)*up", "[muse_dense][kernels]") {
    constexpr int64_t n = 2048;
    const auto hg = RandomHalves(n);
    const auto hu = RandomHalves(n);
    const auto g = ToFloat(hg), u = ToFloat(hu);
    DevBuf dg(n * 2), du(n * 2), dout(n * 2);
    Upload(dg.p, hg.data(), n * 2);
    Upload(du.p, hu.data(), n * 2);
    REQUIRE(RunMuseDenseSwiGLU(nullptr, dg.as<__half>(), du.as<__half>(),
                               dout.as<__half>(), n) == cudaSuccess);
    const auto o = Download<__half>(dout.p, n);
    for (int64_t i = 0; i < n; ++i)
        CHECK(Close(__half2float(o[i]), RefSilu(g[i]) * u[i], 6e-3));
}

TEST_CASE("muse embed gather copies FP16 rows", "[muse_dense][kernels]") {
    constexpr int vocab = 64, hidden = 128, seq = 3;
    const auto hemb = RandomHalves(vocab * hidden);
    const int32_t ids[seq] = {5, 0, 63};
    DevBuf demb(hemb.size() * 2), dids(seq * 4), dout(seq * hidden * 2);
    Upload(demb.p, hemb.data(), hemb.size() * 2);
    Upload(dids.p, ids, seq * 4);
    REQUIRE(RunMuseDenseEmbedGatherFp16(nullptr, demb.as<__half>(),
                                        dids.as<int32_t>(), dout.as<__half>(),
                                        seq, hidden) == cudaSuccess);
    const auto o = Download<__half>(dout.p, seq * hidden);
    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < hidden; ++d)
            CHECK(__half2float(o[t * hidden + d]) ==
                  __half2float(hemb[ids[t] * hidden + d]));
}

TEST_CASE("muse logits apply the output multiplier AND the softcap",
          "[muse_dense][kernels]") {
    // FP16 head: logits = cap * tanh(raw * multiplier / cap). A kernel that
    // drops the multiplier ships ~5.1x hot logits; a non-unit fixture pins it.
    constexpr int vocab = 96, hidden = 64;
    const auto hw = RandomHalves(vocab * hidden);
    const auto hx = RandomHalves(hidden);
    const auto w = ToFloat(hw), x = ToFloat(hx);
    const float multiplier = 0.19611613513818404f, cap = 20.0f;
    DevBuf dw(hw.size() * 2), dx(hx.size() * 2);
    DevBuf dargmax(4), dmaxv(4);
    const int blocks = (vocab + 127) / 128;
    DevBuf dlogits((vocab + blocks * 2) * 4);
    Upload(dw.p, hw.data(), hw.size() * 2);
    Upload(dx.p, hx.data(), hx.size() * 2);
    REQUIRE(RunMuseDenseLogitsArgmax(nullptr, dw.as<__half>(), dx.as<__half>(),
                                     vocab, hidden, multiplier, cap,
                                     dargmax.as<int32_t>(), dmaxv.as<float>(),
                                     dlogits.as<float>()) == cudaSuccess);
    // Host FP64 reference: raw = W @ x, then the softcap.
    std::vector<double> ref(vocab);
    int best = 0;
    for (int v = 0; v < vocab; ++v) {
        double dot = 0.0;
        for (int k = 0; k < hidden; ++k)
            dot += static_cast<double>(w[v * hidden + k]) * x[k];
        ref[v] = cap * std::tanh(dot * multiplier / cap);
        if (ref[v] > ref[best]) best = v;
    }
    const auto logits = Download<float>(dlogits.p, vocab);
    for (int v = 0; v < vocab; ++v) {
        INFO("v=" << v << " got=" << logits[v] << " want=" << ref[v]);
        CHECK(Close(logits[v], ref[v], 2e-2));
    }
    const auto am = Download<int32_t>(dargmax.p, 1);
    CHECK(am[0] == best);
}

TEST_CASE("muse kernels are run-to-run deterministic", "[muse_dense][kernels]") {
    constexpr int rows = 2, hidden = 256;
    const auto hx = RandomHalves(rows * hidden);
    const auto hw = RandomHalves(hidden);
    DevBuf dx(hx.size() * 2), dw(hw.size() * 2), d1(hx.size() * 2),
        d2(hx.size() * 2);
    Upload(dx.p, hx.data(), hx.size() * 2);
    Upload(dw.p, hw.data(), hw.size() * 2);
    REQUIRE(RunMuseDenseCenteredNorm(nullptr, dx.as<__half>(), dw.as<__half>(),
                                     d1.as<__half>(), rows, hidden, 1e-5f) ==
            cudaSuccess);
    REQUIRE(RunMuseDenseCenteredNorm(nullptr, dx.as<__half>(), dw.as<__half>(),
                                     d2.as<__half>(), rows, hidden, 1e-5f) ==
            cudaSuccess);
    const auto a = Download<__half>(d1.p, hx.size());
    const auto b = Download<__half>(d2.p, hx.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}
TEST_CASE("muse split-KV attention matches the unsplit kernel",
          "[muse_dense][kernels]") {
    constexpr int seq = 1, nh = 8, nkv = 2, hd = 32;
    constexpr int slots = 600, pos_base = 599;
    const auto hq = RandomHalves(seq * nh * hd);
    const auto hk = RandomHalves(slots * nkv * hd);
    const auto hv = RandomHalves(slots * nkv * hd);
    const auto q = ToFloat(hq), k = ToFloat(hk), v = ToFloat(hv);
    const float scale = 0.08838834764831845f;
    DevBuf dq(hq.size() * 2), dk(hk.size() * 2), dv(hv.size() * 2),
        d1(hq.size() * 2), d2(hq.size() * 2);
    Upload(dq.p, hq.data(), hq.size() * 2);
    Upload(dk.p, hk.data(), hk.size() * 2);
    Upload(dv.p, hv.data(), hv.size() * 2);
    const auto geo = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, slots, nkv, hd, 1);
    // Full prefix (NoPE): window 0.
    REQUIRE(RunMuseDenseAttention(nullptr, dq.as<__half>(), dk.as<__half>(),
                                  dv.as<__half>(), d1.as<__half>(), seq, nh,
                                  nkv, hd, 0, scale, pos_base, geo, geo) ==
            cudaSuccess);
    const int splits = 8;
    const int64_t slots_n = static_cast<int64_t>(splits) * seq * nh;
    DevBuf dpart(slots_n * (hd + 2) * 4);
    REQUIRE(RunMuseDenseAttentionSplit(nullptr, dq.as<__half>(), dk.as<__half>(),
                                       dv.as<__half>(), d2.as<__half>(),
                                       dpart.as<float>(), seq, nh, nkv, hd, 0,
                                       scale, pos_base, splits, geo, geo) ==
            cudaSuccess);
    const auto ref = RefAttention(q, k, v, seq, nh, nkv, hd, 0, scale,
                                  pos_base, slots);
    const auto a = Download<__half>(d1.p, hq.size());
    const auto b = Download<__half>(d2.p, hq.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(Close(__half2float(a[i]), ref[i], 3e-2));
        CHECK(Close(__half2float(b[i]), ref[i], 3e-2));
    }
}
