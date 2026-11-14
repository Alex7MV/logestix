// Inkling-2-Small numeric kernels (Model Ingestion Protocol stages 5/6).
//
// Every case here compares the device kernel against an INDEPENDENT host
// reference written from the same prose, not against the kernel's own output:
// the five deltas of this family (AttnRes depth mixing, learned per-head L2
// clipping, the learned relative-position bias, gated attention with learned
// sinks, and the two DIFFERENT short-convolution tap orders) are exactly the
// places where a plausible-looking implementation decodes the wrong text.
//
// Label `kernels`; SKIPs (4) when no CUDA device is present.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/lsx_kv_cache.h"

using namespace lsxcommon::inkling;
using lsxcommon::KvCacheGeometry;
using lsxcommon::KvCacheLayout;
using Catch::Approx;

namespace {

bool HaveCuda()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

cudaStream_t Stream0() { return nullptr; }

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

std::vector<float> FPattern(int64_t n, float scale, int64_t seed)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] =
            static_cast<float>(static_cast<int64_t>((i * 7 + seed * 13) % 23) - 11) *
            scale;
    return v;
}

std::vector<__half> HPattern(int64_t n, float scale, int64_t seed)
{
    const auto f = FPattern(n, scale, seed);
    std::vector<__half> v(static_cast<size_t>(n));
    for (size_t i = 0; i < f.size(); ++i) v[i] = H(f[i]);
    return v;
}

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, sizeof(T) * v.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), sizeof(T) * v.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

template <typename T>
void FromDevice(const T *d, std::vector<T> &v)
{
    REQUIRE(cudaMemcpy(v.data(), d, sizeof(T) * v.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

void Sync()
{
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

// ── host references ───────────────────────────────────────────────────────
// Unweighted RMSNorm, exactly as Inkling2UnweightedRMSNorm: FP32 square mean,
// rsqrt(mean + eps), then the scale.
std::vector<float> RefDepthMix(const std::vector<float> &sources, int n_src,
                               const std::vector<float> &queries, int nq,
                               int hidden, float eps)
{
    std::vector<float> norm(static_cast<size_t>(n_src) * hidden);
    for (int j = 0; j < n_src; ++j) {
        double ss = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double v = sources[static_cast<size_t>(j) * hidden + i];
            ss += v * v;
        }
        const float inv = static_cast<float>(
            1.0 / std::sqrt(ss / hidden + static_cast<double>(eps)));
        for (int i = 0; i < hidden; ++i)
            norm[static_cast<size_t>(j) * hidden + i] =
                sources[static_cast<size_t>(j) * hidden + i] * inv;
    }
    std::vector<float> out(static_cast<size_t>(nq) * hidden, 0.0f);
    const double qscale = 1.0 / std::sqrt(static_cast<double>(hidden));
    for (int q = 0; q < nq; ++q) {
        std::vector<double> logit(static_cast<size_t>(n_src));
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_src; ++j) {
            double dot = 0.0;
            for (int i = 0; i < hidden; ++i)
                dot += static_cast<double>(queries[static_cast<size_t>(q) * hidden + i]) *
                       norm[static_cast<size_t>(j) * hidden + i];
            logit[static_cast<size_t>(j)] = dot * qscale;
            mx = std::max(mx, logit[static_cast<size_t>(j)]);
        }
        double sum = 0.0;
        for (int j = 0; j < n_src; ++j) {
            logit[static_cast<size_t>(j)] = std::exp(logit[static_cast<size_t>(j)] - mx);
            sum += logit[static_cast<size_t>(j)];
        }
        for (int j = 0; j < n_src; ++j) {
            const double w = logit[static_cast<size_t>(j)] / sum;
            for (int i = 0; i < hidden; ++i)
                out[static_cast<size_t>(q) * hidden + i] +=
                    static_cast<float>(w * norm[static_cast<size_t>(j) * hidden + i]);
        }
    }
    return out;
}

std::vector<float> RefL2Clip(const std::vector<float> &x,
                             const std::vector<float> *scale, int rows, int dim,
                             float threshold)
{
    std::vector<float> out(x.size());
    for (int r = 0; r < rows; ++r) {
        std::vector<float> v(static_cast<size_t>(dim));
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) {
            const size_t idx = static_cast<size_t>(r) * dim + i;
            v[static_cast<size_t>(i)] = scale ? x[idx] * (*scale)[idx] : x[idx];
            ss += static_cast<double>(v[static_cast<size_t>(i)]) * v[static_cast<size_t>(i)];
        }
        const double norm = std::sqrt(ss);
        const double factor =
            std::min(static_cast<double>(threshold) / (norm + 1e-6), 1.0);
        for (int i = 0; i < dim; ++i)
            out[static_cast<size_t>(r) * dim + i] =
                static_cast<float>(v[static_cast<size_t>(i)] * factor);
    }
    return out;
}

std::vector<float> RefRelBias(const std::vector<float> &rel_states,
                              const std::vector<float> &proj, int heads,
                              int d_rel, int rel_extent, int q_pos, int kv_len)
{
    std::vector<float> out(static_cast<size_t>(heads) * kv_len, 0.0f);
    for (int h = 0; h < heads; ++h) {
        std::vector<double> profile(static_cast<size_t>(rel_extent));
        for (int d = 0; d < rel_extent; ++d) {
            double acc = 0.0;
            for (int c = 0; c < d_rel; ++c)
                acc += static_cast<double>(rel_states[static_cast<size_t>(h) * d_rel + c]) *
                       proj[static_cast<size_t>(c) * rel_extent + d];
            profile[static_cast<size_t>(d)] = acc;
        }
        for (int j = 0; j < kv_len; ++j) {
            const int d = q_pos - j;
            if (d < 0 || d >= rel_extent) continue;
            out[static_cast<size_t>(h) * kv_len + j] =
                static_cast<float>(profile[static_cast<size_t>(d)]);
        }
    }
    return out;
}

double Softcap(double v, double cap) { return std::tanh(v / cap) * cap; }

}  // namespace

TEST_CASE("inkling depth mix matches the host reference", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kHidden = 32;
    constexpr float kEps = 1e-6f;
    for (const int n_src : {1, 3, 9}) {
        for (const int nq : {1, 3}) {
            const auto src = FPattern(n_src * kHidden, 0.5f, n_src);
            const auto qry = FPattern(nq * kHidden, 0.25f, nq + 5);
            auto ref = RefDepthMix(src, n_src, qry, nq, kHidden, kEps);

            float *d_src = ToDevice(src);
            float *d_qry = ToDevice(qry);
            float *d_out = nullptr;
            REQUIRE(cudaMalloc(&d_out, sizeof(float) * ref.size()) == cudaSuccess);
            REQUIRE(RunInklingDepthMix(Stream0(), d_src, n_src, d_qry, nq,
                                       d_out, kHidden, kEps) == cudaSuccess);
            Sync();
            std::vector<float> got(ref.size());
            FromDevice(d_out, got);
            for (size_t i = 0; i < ref.size(); ++i) {
                INFO("n_src=" << n_src << " nq=" << nq << " i=" << i);
                REQUIRE(got[i] == Approx(ref[i]).margin(1e-5));
            }
            cudaFree(d_src);
            cudaFree(d_qry);
            cudaFree(d_out);
        }
    }
}

TEST_CASE("inkling L2 clip matches the host reference", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kRows = 4;
    constexpr int kDim = 16;
    for (const bool with_scale : {false, true}) {
        // A large pattern so some heads clip and some do not.
        auto x = FPattern(kRows * kDim, 3.0f, with_scale ? 3 : 7);
        std::vector<float> scale;
        const std::vector<float> *sp = nullptr;
        if (with_scale) {
            scale = FPattern(kRows * kDim, 0.1f, 11);
            for (auto &v : scale) v = 1.0f + v;
            sp = &scale;
        }
        auto ref = RefL2Clip(x, sp, kRows, kDim, 22.627416997969522f);

        float *d_x = ToDevice(x);
        float *d_s = with_scale ? ToDevice(scale) : nullptr;
        REQUIRE(RunInklingL2Clip(Stream0(), d_x, d_s, kRows, kDim,
                                 22.627416997969522f) == cudaSuccess);
        Sync();
        std::vector<float> got(x.size());
        FromDevice(d_x, got);
        for (size_t i = 0; i < ref.size(); ++i) {
            INFO("with_scale=" << with_scale << " i=" << i);
            REQUIRE(got[i] == Approx(ref[i]).margin(1e-4));
        }
        cudaFree(d_x);
        if (d_s) cudaFree(d_s);
    }

    // An unclipped row must be untouched by the weight-free form.
    std::vector<float> small(8, 0.5f);
    float *d_small = ToDevice(small);
    REQUIRE(RunInklingL2Clip(Stream0(), d_small, nullptr, 1, 8,
                             22.627416997969522f) == cudaSuccess);
    Sync();
    std::vector<float> got(8);
    FromDevice(d_small, got);
    for (float v : got) REQUIRE(v == Approx(0.5f).margin(1e-6));
    cudaFree(d_small);
}

TEST_CASE("inkling unweighted norm matches the host reference",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kRows = 3;
    constexpr int kDim = 16;
    auto x = FPattern(kRows * kDim, 0.7f, 2);
    std::vector<float> ref(x.size());
    for (int r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (int i = 0; i < kDim; ++i) {
            const double v = x[static_cast<size_t>(r) * kDim + i];
            ss += v * v;
        }
        const float inv =
            static_cast<float>(1.0 / std::sqrt(ss / kDim + 1e-6));
        for (int i = 0; i < kDim; ++i)
            ref[static_cast<size_t>(r) * kDim + i] =
                x[static_cast<size_t>(r) * kDim + i] * inv;
    }
    float *d_x = ToDevice(x);
    REQUIRE(RunInklingUnweightedNorm(Stream0(), d_x, kRows, kDim, 1e-6f) ==
            cudaSuccess);
    Sync();
    std::vector<float> got(x.size());
    FromDevice(d_x, got);
    for (size_t i = 0; i < ref.size(); ++i)
        REQUIRE(got[i] == Approx(ref[i]).margin(1e-5));
    cudaFree(d_x);
}

TEST_CASE("inkling relative bias gathers by backward distance",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kHeads = 3;
    constexpr int kDRel = 4;
    constexpr int kExtent = 8;
    for (const int kv_len : {1, 5, 8, 13}) {
        const int q_pos = kv_len - 1;
        const auto rel = FPattern(kHeads * kDRel, 0.4f, kv_len);
        const auto proj = FPattern(kDRel * kExtent, 0.3f, kv_len + 1);
        auto ref = RefRelBias(rel, proj, kHeads, kDRel, kExtent, q_pos, kv_len);

        float *d_rel = ToDevice(rel);
        float *d_proj = ToDevice(proj);
        float *d_bias = nullptr;
        REQUIRE(cudaMalloc(&d_bias, sizeof(float) * ref.size()) == cudaSuccess);
        REQUIRE(RunInklingRelBias(Stream0(), d_rel, d_proj, d_bias, kHeads,
                                  kDRel, kExtent, q_pos, kv_len) == cudaSuccess);
        Sync();
        std::vector<float> got(ref.size());
        FromDevice(d_bias, got);
        for (size_t i = 0; i < ref.size(); ++i) {
            INFO("kv_len=" << kv_len << " i=" << i);
            REQUIRE(got[i] == Approx(ref[i]).margin(1e-5));
        }
        cudaFree(d_rel);
        cudaFree(d_proj);
        cudaFree(d_bias);
    }
}

TEST_CASE("inkling attention: sinks absorb mass, window bounds, bias shifts",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kHeads = 2;
    constexpr int kKvHeads = 1;  // GQA repeat 2
    constexpr int kDim = 8;
    constexpr int kCap = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
    const int q_pos = 11;  // 12 cached positions

    for (const int window : {0, 4}) {
        for (const bool use_sink : {false, true}) {
            for (const bool use_bias : {false, true}) {
                const auto q = HPattern(kHeads * kDim, 0.6f, 1);
                auto k_hist = HPattern((q_pos + 1) * kKvHeads * kDim, 0.5f, 2);
                auto v_hist = HPattern((q_pos + 1) * kKvHeads * kDim, 0.5f, 3);
                std::vector<float> sink(kHeads), bias(kHeads * (q_pos + 1));
                const float *d_sink = nullptr;
                const float *d_bias = nullptr;
                if (use_sink) {
                    sink = FPattern(kHeads, 0.5f, 4);
                    d_sink = ToDevice(sink);
                }
                if (use_bias) {
                    bias = FPattern(kHeads * (q_pos + 1), 0.5f, 5);
                    d_bias = ToDevice(bias);
                }

                // ── host reference (per query head, GQA repeat) ──
                std::vector<float> ref(kHeads * kDim, 0.0f);
                for (int h = 0; h < kHeads; ++h) {
                    std::vector<double> logit(q_pos + 1);
                    double mx = -std::numeric_limits<double>::infinity();
                    for (int j = 0; j <= q_pos; ++j) {
                        const int kvh = h / (kHeads / kKvHeads);
                        double dot = 0.0;
                        for (int i = 0; i < kDim; ++i)
                            dot += static_cast<double>(F(q[static_cast<size_t>(h) * kDim + i])) *
                                   F(k_hist[(static_cast<size_t>(j) * kKvHeads + kvh) * kDim + i]);
                        double l = dot * scale;
                        if (use_bias) l += bias[static_cast<size_t>(h) * (q_pos + 1) + j];
                        const bool visible = (window == 0) || (j > q_pos - window);
                        logit[static_cast<size_t>(j)] = visible ? l : -1e30;
                        mx = std::max(mx, logit[static_cast<size_t>(j)]);
                    }
                    if (use_sink) mx = std::max(mx, static_cast<double>(sink[static_cast<size_t>(h)]));
                    double sum = 0.0;
                    for (int j = 0; j <= q_pos; ++j) {
                        logit[static_cast<size_t>(j)] =
                            std::exp(logit[static_cast<size_t>(j)] - mx);
                        sum += logit[static_cast<size_t>(j)];
                    }
                    if (use_sink) sum += std::exp(static_cast<double>(sink[static_cast<size_t>(h)]) - mx);
                    for (int j = 0; j <= q_pos; ++j) {
                        const double p = logit[static_cast<size_t>(j)] / sum;
                        const int kvh = h / (kHeads / kKvHeads);
                        for (int i = 0; i < kDim; ++i)
                            ref[static_cast<size_t>(h) * kDim + i] +=
                                static_cast<float>(p * F(v_hist[(static_cast<size_t>(j) * kKvHeads + kvh) * kDim + i]));
                    }
                }

                const KvCacheGeometry geom = KvCacheGeometry::Make(
                    KvCacheLayout::kPosMajor, kCap, kKvHeads, kDim, 1);
                __half *d_k = nullptr;
                __half *d_v = nullptr;
                REQUIRE(cudaMalloc(&d_k, sizeof(__half) * geom.Elements()) == cudaSuccess);
                REQUIRE(cudaMalloc(&d_v, sizeof(__half) * geom.Elements()) == cudaSuccess);
                // Fill the cache with the history: pos-major [pos][head][dim].
                REQUIRE(cudaMemcpy2D(d_k, geom.pos_stride * sizeof(__half), k_hist.data(),
                                     kKvHeads * kDim * sizeof(__half),
                                     kKvHeads * kDim * sizeof(__half), q_pos + 1,
                                     cudaMemcpyHostToDevice) == cudaSuccess);
                REQUIRE(cudaMemcpy2D(d_v, geom.pos_stride * sizeof(__half), v_hist.data(),
                                     kKvHeads * kDim * sizeof(__half),
                                     kKvHeads * kDim * sizeof(__half), q_pos + 1,
                                     cudaMemcpyHostToDevice) == cudaSuccess);
                __half *d_q = ToDevice(q);
                __half *d_out = nullptr;
                REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kHeads * kDim) == cudaSuccess);
                REQUIRE(RunInklingAttention(Stream0(), d_q, d_k, d_v, geom,
                                            kHeads, q_pos, d_sink, d_bias,
                                            window, scale, d_out) == cudaSuccess);
                Sync();
                std::vector<__half> got(static_cast<size_t>(kHeads) * kDim);
                FromDevice(d_out, got);
                for (size_t i = 0; i < got.size(); ++i) {
                    INFO("window=" << window << " sink=" << use_sink
                                   << " bias=" << use_bias << " i=" << i);
                    REQUIRE(F(got[i]) == Approx(ref[i]).margin(2e-3));
                }
                cudaFree(d_q);
                cudaFree(d_k);
                cudaFree(d_v);
                cudaFree(d_out);
                if (use_sink) cudaFree(const_cast<float *>(d_sink));
                if (use_bias) cudaFree(const_cast<float *>(d_bias));
            }
        }
    }
}

TEST_CASE("inkling kv append writes the current position", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kKvHeads = 2;
    constexpr int kDim = 4;
    constexpr int kCap = 8;
    const KvCacheGeometry geom = KvCacheGeometry::Make(KvCacheLayout::kPosMajor,
                                                       kCap, kKvHeads, kDim, 1);
    const auto k_new = HPattern(kKvHeads * kDim, 0.5f, 1);
    const auto v_new = HPattern(kKvHeads * kDim, 0.5f, 2);
    __half *d_k = nullptr;
    __half *d_v = nullptr;
    REQUIRE(cudaMalloc(&d_k, sizeof(__half) * geom.Elements()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_v, sizeof(__half) * geom.Elements()) == cudaSuccess);
    REQUIRE(cudaMemset(d_k, 0, sizeof(__half) * geom.Elements()) == cudaSuccess);
    REQUIRE(cudaMemset(d_v, 0, sizeof(__half) * geom.Elements()) == cudaSuccess);
    const int pos = 5;
    __half *d_kn = ToDevice(k_new);
    __half *d_vn = ToDevice(v_new);
    REQUIRE(RunInklingKvAppend(Stream0(), d_kn, d_vn, d_k, d_v, geom, pos) ==
            cudaSuccess);
    Sync();
    std::vector<__half> got(geom.Elements());
    FromDevice(d_k, got);
    for (int h = 0; h < kKvHeads; ++h)
        for (int i = 0; i < kDim; ++i)
            REQUIRE(F(got[static_cast<size_t>(pos) * geom.pos_stride +
                          static_cast<size_t>(h) * geom.head_stride + i]) ==
                    Approx(F(k_new[static_cast<size_t>(h) * kDim + i])));
    // Nothing before the written position.
    for (size_t i = 0; i < static_cast<size_t>(pos) * geom.pos_stride; ++i)
        REQUIRE(F(got[i]) == 0.0f);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_kn);
    cudaFree(d_vn);
}

TEST_CASE("inkling short conv: the two tap orders are not interchangeable",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kCh = 6;
    constexpr int kK = 4;
    const float inv_static = 1.0f / std::sqrt(static_cast<float>(kK + 1));
    const float inv_dyn = 1.0f / std::sqrt(static_cast<float>(2 * kK + 1));

    // The ring holds the last `k` tokens BEFORE the push (oldest first); the
    // kernel pushes the current token itself, so the post-push ring is
    // [old[1], ..., old[k-1], x_t].
    std::vector<float> ring(static_cast<size_t>(kK) * kCh);
    for (int j = 0; j < kK; ++j)
        for (int c = 0; c < kCh; ++c)
            ring[static_cast<size_t>(j) * kCh + c] =
                0.1f * static_cast<float>(j * kCh + c + 1);
    const auto w = FPattern(kCh * kK, 0.05f, 3);   // conv1d taps: w[c][j]
    const auto dyn = FPattern(kCh * kK, 0.05f, 4); // lag taps:  dyn[c][lag]
    const auto x_t = HPattern(kCh, 0.5f, 5);

    // Post-push ring: the current token becomes the NEWEST entry.
    std::vector<float> post(ring.size());
    for (int j = 0; j + 1 < kK; ++j)
        for (int c = 0; c < kCh; ++c)
            post[static_cast<size_t>(j) * kCh + c] =
                ring[static_cast<size_t>(j + 1) * kCh + c];
    for (int c = 0; c < kCh; ++c)
        post[static_cast<size_t>(kK - 1) * kCh + c] =
            F(x_t[static_cast<size_t>(c)]);

    SECTION("static (conv1d) taps: tap k-1 weights the current token")
    {
        std::vector<float> ref(static_cast<size_t>(kCh));
        for (int c = 0; c < kCh; ++c) {
            double acc = 0.0;
            for (int j = 0; j < kK; ++j)
                acc += static_cast<double>(w[static_cast<size_t>(c) * kK + j]) *
                       post[static_cast<size_t>(j) * kCh + c];
            ref[static_cast<size_t>(c)] =
                static_cast<float>((acc + static_cast<double>(F(x_t[static_cast<size_t>(c)]))) *
                                   inv_static);
        }
        std::vector<float> dev_state = ring;
        float *d_state = ToDevice(dev_state);
        float *d_w = ToDevice(w);
        __half *d_x = ToDevice(x_t);
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kCh) == cudaSuccess);
        REQUIRE(RunInklingShortConv(Stream0(), d_x, d_state, d_w, nullptr,
                                    d_out, kCh, kK, inv_static) == cudaSuccess);
        Sync();
        std::vector<__half> got(static_cast<size_t>(kCh));
        FromDevice(d_out, got);
        for (int c = 0; c < kCh; ++c) {
            INFO("c=" << c);
            REQUIRE(F(got[static_cast<size_t>(c)]) ==
                    Approx(ref[static_cast<size_t>(c)]).epsilon(0.002));
        }
        std::vector<float> after(dev_state.size());
        FromDevice(d_state, after);
        for (size_t i = 0; i < post.size(); ++i)
            REQUIRE(after[i] == Approx(post[i]).margin(1e-6));
        cudaFree(d_state);
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_out);
    }

    SECTION("dynamic (lag) taps: lag 0 weights the current token")
    {
        std::vector<float> ref(static_cast<size_t>(kCh));
        for (int c = 0; c < kCh; ++c) {
            double acc = 0.0;
            for (int lag = 0; lag < kK; ++lag) {
                const int sj = kK - 1 - lag;  // lag 0 == newest
                const double tap =
                    static_cast<double>(dyn[static_cast<size_t>(c) * kK + lag]);
                acc += tap * post[static_cast<size_t>(sj) * kCh + c];
            }
            ref[static_cast<size_t>(c)] =
                static_cast<float>((acc + static_cast<double>(F(x_t[static_cast<size_t>(c)]))) *
                                   inv_dyn);
        }
        std::vector<float> dev_state = ring;
        float *d_state = ToDevice(dev_state);
        float *d_w = ToDevice(w);
        float *d_dyn = ToDevice(dyn);
        __half *d_x = ToDevice(x_t);
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kCh) == cudaSuccess);
        REQUIRE(RunInklingShortConv(Stream0(), d_x, d_state, d_w, d_dyn, d_out, kCh,
                                    kK, inv_dyn) == cudaSuccess);
        Sync();
        std::vector<__half> got(static_cast<size_t>(kCh));
        FromDevice(d_out, got);
        for (int c = 0; c < kCh; ++c) {
            INFO("c=" << c);
            REQUIRE(F(got[static_cast<size_t>(c)]) ==
                    Approx(ref[static_cast<size_t>(c)]).epsilon(0.002));
        }
        // The two tap orders must NOT agree on this fixture: if they did, the
        // test would pass with either convention and pin nothing.
        std::vector<float> static_ref(static_cast<size_t>(kCh));
        for (int c = 0; c < kCh; ++c) {
            double acc = 0.0;
            for (int j = 0; j < kK; ++j)
                acc += static_cast<double>(w[static_cast<size_t>(c) * kK + j]) *
                       post[static_cast<size_t>(j) * kCh + c];
            static_ref[static_cast<size_t>(c)] =
                static_cast<float>((acc + static_cast<double>(F(x_t[static_cast<size_t>(c)]))) *
                                   inv_dyn);
        }
        int differing = 0;
        for (int c = 0; c < kCh; ++c)
            if (std::fabs(static_ref[static_cast<size_t>(c)] -
                          ref[static_cast<size_t>(c)]) > 1e-3f)
                ++differing;
        REQUIRE(differing > 0);
        cudaFree(d_state);
        cudaFree(d_w);
        cudaFree(d_dyn);
        cudaFree(d_x);
        cudaFree(d_out);
    }
}

TEST_CASE("inkling log scaling and dynamic conv taps", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");

    SECTION("log scaling multiplies the CLIPPED q and rel")
    {
        constexpr int64_t kNq = 20;
        constexpr int64_t kNrel = 8;
        auto q = FPattern(kNq, 0.5f, 1);
        auto rel = FPattern(kNrel, 0.25f, 2);
        // tau at position 200000 with floor 128000 and alpha 0.1.
        const double pos = 200000.0;
        const float tau = static_cast<float>(
            1.0 + 0.1 * std::log(std::max((pos + 1.0) / 128000.0, 1.0)));
        REQUIRE(tau > 1.0f);  // the scaling only ever grows with position
        const auto q0 = q;
        const auto r0 = rel;
        float *d_q = ToDevice(q);
        float *d_r = ToDevice(rel);
        REQUIRE(RunInklingLogScale(Stream0(), d_q, d_r, kNq, kNrel, tau) ==
                cudaSuccess);
        Sync();
        std::vector<float> gq(static_cast<size_t>(kNq)), gr(static_cast<size_t>(kNrel));
        FromDevice(d_q, gq);
        FromDevice(d_r, gr);
        for (int64_t i = 0; i < kNq; ++i)
            REQUIRE(gq[static_cast<size_t>(i)] ==
                    Approx(q0[static_cast<size_t>(i)] * tau).margin(1e-6));
        for (int64_t i = 0; i < kNrel; ++i)
            REQUIRE(gr[static_cast<size_t>(i)] ==
                    Approx(r0[static_cast<size_t>(i)] * tau).margin(1e-6));
        cudaFree(d_q);
        cudaFree(d_r);
    }

    SECTION("dynamic taps: static[lag][ch] plus the head's own projection")
    {
        constexpr int kChannels = 6;   // 3 heads x head_dim 2
        constexpr int kHeadDim = 2;
        constexpr int kK = 4;
        constexpr int kHeads = kChannels / kHeadDim;
        const auto static_w = FPattern(kK * kChannels, 0.1f, 3);   // [k][channels]
        const auto proj = FPattern(kHeads * kK, 0.2f, 4);          // [heads*k]
        std::vector<float> ref(static_cast<size_t>(kChannels) * kK);
        for (int ch = 0; ch < kChannels; ++ch) {
            const int head = ch / kHeadDim;
            for (int lag = 0; lag < kK; ++lag)
                ref[static_cast<size_t>(ch) * kK + lag] =
                    static_w[static_cast<size_t>(lag) * kChannels + ch] +
                    proj[static_cast<size_t>(head) * kK + lag];
        }
        float *d_s = ToDevice(static_w);
        float *d_p = ToDevice(proj);
        float *d_t = nullptr;
        REQUIRE(cudaMalloc(&d_t, sizeof(float) * ref.size()) == cudaSuccess);
        REQUIRE(RunInklingDynamicTaps(Stream0(), d_s, d_p, d_t, kChannels, kHeadDim,
                                      kK) == cudaSuccess);
        Sync();
        std::vector<float> got(ref.size());
        FromDevice(d_t, got);
        for (size_t i = 0; i < ref.size(); ++i)
            REQUIRE(got[i] == Approx(ref[i]).margin(1e-6));
        // Two channels inside one head share the PROJECTION term but not the
        // static one; that is the property a channel-indexed projection would
        // break.
        REQUIRE(got[0 * kK + 1] - static_w[1 * kChannels + 0] ==
                Approx(got[1 * kK + 1] - static_w[1 * kChannels + 1]).margin(1e-6));
        cudaFree(d_s);
        cudaFree(d_p);
        cudaFree(d_t);
    }
}

TEST_CASE("inkling softcapped SwiGLU matches the host reference",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int64_t kN = 40;
    const auto g = HPattern(kN, 3.0f, 1);  // wide enough to exercise both caps
    const auto u = HPattern(kN, 9.0f, 2);
    std::vector<float> ref(static_cast<size_t>(kN));
    for (int64_t i = 0; i < kN; ++i) {
        const double gv = F(g[static_cast<size_t>(i)]);
        const double uv = F(u[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = static_cast<float>(
            Softcap(gv, 4.0) * (1.0 / (1.0 + std::exp(-gv))) * Softcap(uv, 25.0));
    }
    __half *d_g = ToDevice(g);
    __half *d_u = ToDevice(u);
    __half *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kN) == cudaSuccess);
    REQUIRE(RunInklingSoftcapSwiGLU(Stream0(), d_g, d_u, d_out, kN, 4.0f,
                                    25.0f) == cudaSuccess);
    Sync();
    std::vector<__half> got(static_cast<size_t>(kN));
    FromDevice(d_out, got);
    for (int64_t i = 0; i < kN; ++i) {
        INFO("i=" << i);
        // FP16 store: an output of magnitude ~90 has a ulp of 0.0625, so the
        // tolerance must be relative, not absolute.
        REQUIRE(F(got[static_cast<size_t>(i)]) ==
                Approx(ref[static_cast<size_t>(i)])
                    .margin(std::fabs(ref[static_cast<size_t>(i)]) * 2e-3 + 2e-3));
    }
    cudaFree(d_g);
    cudaFree(d_u);
    cudaFree(d_out);
}

TEST_CASE("inkling router: bias steers the SELECTION, not the weights",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kE = 8;
    constexpr int kK = 2;
    constexpr int kH = 16;
    const float route_scale = 8.0f;
    // 0.02 (not 0.2): the larger scale saturates sigmoid to exactly 1.0 for
    // three experts, which makes the fixture an exact tie and pins nothing.
    const auto w = FPattern(kE * kH, 0.02f, 1);
    const auto x = HPattern(kH, 0.5f, 2);

    // ── reference ──
    std::vector<double> score(kE);
    for (int e = 0; e < kE; ++e) {
        double dot = 0.0;
        for (int i = 0; i < kH; ++i)
            dot += static_cast<double>(w[static_cast<size_t>(e) * kH + i]) *
                   F(x[static_cast<size_t>(i)]);
        score[static_cast<size_t>(e)] = 1.0 / (1.0 + std::exp(-dot));
    }
    auto select = [&](const std::vector<float> &bias, std::vector<int> &ids) {
        std::vector<bool> taken(kE, false);
        ids.clear();
        for (int k = 0; k < kK; ++k) {
            int best = -1;
            double best_v = -1e300;
            for (int e = 0; e < kE; ++e) {
                if (taken[static_cast<size_t>(e)]) continue;
                const double v = score[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)];
                if (v > best_v) { best_v = v; best = e; }
            }
            taken[static_cast<size_t>(best)] = true;
            ids.push_back(best);
        }
        std::sort(ids.begin(), ids.end());
    };

    for (const bool use_bias : {false, true}) {
        // A bias that flips the choice away from the two highest scores.
        std::vector<float> bias(kE, 0.0f);
        if (use_bias) {
            for (int e = 0; e < kE; ++e) bias[static_cast<size_t>(e)] = (e >= 6) ? 10.0f : -10.0f;
        }
        std::vector<int> want;
        select(bias, want);
        double sum = 0.0;
        for (int id : want) sum += score[static_cast<size_t>(id)];
        std::vector<float> coeff;
        for (int id : want)
            coeff.push_back(static_cast<float>(score[static_cast<size_t>(id)] / sum * route_scale));

        float *d_w = ToDevice(w);
        __half *d_x = ToDevice(x);
        float *d_bias = use_bias ? ToDevice(bias) : nullptr;
        InklingRoutedSlot *d_slots = nullptr;
        int32_t *d_ids = nullptr;
        REQUIRE(cudaMalloc(&d_slots, sizeof(InklingRoutedSlot) * kK) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_ids, sizeof(int32_t) * kK) == cudaSuccess);
        REQUIRE(RunInklingRouter(Stream0(), d_w, d_bias, d_x, kE, kK, kH,
                                 route_scale, d_slots, d_ids) == cudaSuccess);
        Sync();
        std::vector<InklingRoutedSlot> slots(kK);
        std::vector<int32_t> ids(kK);
        FromDevice(d_slots, slots);
        FromDevice(d_ids, ids);
        std::vector<int> got_ids;
        for (auto s : slots) got_ids.push_back(s.eid);
        std::sort(got_ids.begin(), got_ids.end());
        INFO("use_bias=" << use_bias);
        REQUIRE(got_ids == want);
        for (int k = 0; k < kK; ++k) {
            REQUIRE(ids[static_cast<size_t>(k)] == slots[static_cast<size_t>(k)].eid);
            const auto it = std::find(want.begin(), want.end(),
                                      slots[static_cast<size_t>(k)].eid);
            const size_t idx = static_cast<size_t>(it - want.begin());
            REQUIRE(slots[static_cast<size_t>(k)].coeff ==
                    Approx(coeff[idx]).margin(1e-5));
        }
        // The weights renormalise to route_scale exactly.
        double total = 0.0;
        for (auto s : slots) total += s.coeff;
        REQUIRE(total == Approx(route_scale).margin(1e-4));
        cudaFree(d_w);
        cudaFree(d_x);
        if (d_bias) cudaFree(d_bias);
        cudaFree(d_slots);
        cudaFree(d_ids);
    }
}

TEST_CASE("inkling router tie-breaks to the lower expert id",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kE = 4;
    constexpr int kK = 1;
    constexpr int kH = 8;
    // Experts 0 and 2 carry the IDENTICAL row; 1 and 3 are far below.
    std::vector<float> w(static_cast<size_t>(kE) * kH, 0.0f);
    for (int i = 0; i < kH; ++i) {
        w[static_cast<size_t>(0) * kH + i] = 0.5f;
        w[static_cast<size_t>(2) * kH + i] = 0.5f;
        w[static_cast<size_t>(1) * kH + i] = -0.5f;
        w[static_cast<size_t>(3) * kH + i] = -0.5f;
    }
    std::vector<__half> x(static_cast<size_t>(kH));
    for (int i = 0; i < kH; ++i) x[static_cast<size_t>(i)] = H(0.25f);

    float *d_w = ToDevice(w);
    __half *d_x = ToDevice(x);
    InklingRoutedSlot *d_slots = nullptr;
    REQUIRE(cudaMalloc(&d_slots, sizeof(InklingRoutedSlot) * kK) == cudaSuccess);
    REQUIRE(RunInklingRouter(Stream0(), d_w, nullptr, d_x, kE, kK, kH, 8.0f,
                             d_slots, nullptr) == cudaSuccess);
    Sync();
    std::vector<InklingRoutedSlot> slot(kK);
    FromDevice(d_slots, slot);
    INFO("eid=" << slot[0].eid);
    REQUIRE(slot[0].eid == 0);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_slots);
}

TEST_CASE("inkling plumbing: weighted norm, fp16 matvec, embed row",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kRows = 2;
    constexpr int kHidden = 32;

    SECTION("weighted RMSNorm, FP32 in and FP16 out")
    {
        const auto x = FPattern(kRows * kHidden, 0.6f, 1);
        const auto w = HPattern(kHidden, 0.3f, 2);
        std::vector<float> ref(static_cast<size_t>(kRows) * kHidden);
        for (int r = 0; r < kRows; ++r) {
            double ss = 0.0;
            for (int i = 0; i < kHidden; ++i) {
                const double v = x[static_cast<size_t>(r) * kHidden + i];
                ss += v * v;
            }
            const float inv = static_cast<float>(1.0 / std::sqrt(ss / kHidden + 1e-6));
            for (int i = 0; i < kHidden; ++i)
                ref[static_cast<size_t>(r) * kHidden + i] =
                    x[static_cast<size_t>(r) * kHidden + i] * inv *
                    F(w[static_cast<size_t>(i)]);
        }
        float *d_x = ToDevice(x);
        __half *d_w = ToDevice(w);
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(__half) * ref.size()) == cudaSuccess);
        REQUIRE(RunInklingNormF32In(Stream0(), d_x, d_w, d_out, kRows, kHidden,
                                    1e-6f) == cudaSuccess);
        Sync();
        std::vector<__half> got(ref.size());
        FromDevice(d_out, got);
        for (size_t i = 0; i < ref.size(); ++i)
            REQUIRE(F(got[i]) == Approx(ref[i]).epsilon(0.002));
        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_out);
    }

    SECTION("fp16 matvec with an FP32 accumulator")
    {
        constexpr int kOut = 3;
        const auto w = HPattern(kOut * kHidden, 0.25f, 3);
        const auto x = HPattern(kHidden, 0.4f, 4);
        std::vector<float> ref(static_cast<size_t>(kOut));
        for (int r = 0; r < kOut; ++r) {
            double acc = 0.0;
            for (int c = 0; c < kHidden; ++c)
                acc += static_cast<double>(F(w[static_cast<size_t>(r) * kHidden + c])) *
                       F(x[static_cast<size_t>(c)]);
            ref[static_cast<size_t>(r)] = static_cast<float>(acc);
        }
        __half *d_w = ToDevice(w);
        __half *d_x = ToDevice(x);
        float *d_y = nullptr;
        REQUIRE(cudaMalloc(&d_y, sizeof(float) * kOut) == cudaSuccess);
        REQUIRE(RunInklingMatVecF16(Stream0(), d_w, d_x, d_y, kOut, kHidden) ==
                cudaSuccess);
        Sync();
        std::vector<float> got(static_cast<size_t>(kOut));
        FromDevice(d_y, got);
        for (int r = 0; r < kOut; ++r)
            REQUIRE(got[static_cast<size_t>(r)] ==
                    Approx(ref[static_cast<size_t>(r)]).margin(1e-4));
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    SECTION("NVFP4 embed row dequantizes from a DEVICE token id")
    {
        constexpr int kVocab = 4;
        const auto h = HPattern(kVocab * kHidden, 0.5f, 5);
        // Pack to NVFP4 on the host with the neutral codec's own rules so the
        // decode is checked against the values that were packed.
        const float gs = 0.5f;
        std::vector<uint8_t> packed(static_cast<size_t>(kVocab) * kHidden / 2, 0);
        std::vector<uint8_t> scales(static_cast<size_t>(kVocab) * kHidden / 16,
                                    0x38);  // 2^0
        std::vector<float> expect(static_cast<size_t>(kVocab) * kHidden);
        const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        for (int row = 0; row < kVocab; ++row) {
            for (int i = 0; i < kHidden; ++i) {
                const float v = F(h[static_cast<size_t>(row) * kHidden + i]) / gs;
                int best = 0;
                float bestd = 1e30f;
                for (int q = 0; q < 8; ++q) {
                    const float d = std::fabs(levels[q] - std::fabs(v));
                    if (d < bestd) { bestd = d; best = q; }
                }
                const int nib = (v < 0.0f ? 8 : 0) | best;
                const size_t byte = static_cast<size_t>(row) * kHidden / 2 + i / 2;
                packed[byte] = (i & 1)
                                   ? static_cast<uint8_t>(packed[byte] | (nib << 4))
                                   : static_cast<uint8_t>(packed[byte] | nib);
                expect[static_cast<size_t>(row) * kHidden + i] =
                    (nib & 8 ? -levels[best] : levels[best]) * gs;
            }
        }
        uint8_t *d_p = ToDevice(packed);
        uint8_t *d_s = ToDevice(scales);
        const int32_t want_row = 2;
        int32_t *d_id = ToDevice(std::vector<int32_t>{want_row});
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kHidden) == cudaSuccess);
        REQUIRE(RunInklingEmbedRow(Stream0(), d_p, d_s, gs, d_id, d_out, kHidden) ==
                cudaSuccess);
        Sync();
        std::vector<__half> got(static_cast<size_t>(kHidden));
        FromDevice(d_out, got);
        for (int i = 0; i < kHidden; ++i)
            REQUIRE(F(got[static_cast<size_t>(i)]) ==
                    Approx(expect[static_cast<size_t>(want_row) * kHidden + i])
                        .margin(1e-6));
        cudaFree(d_p);
        cudaFree(d_s);
        cudaFree(d_id);
        cudaFree(d_out);
    }
}

TEST_CASE("inkling routed accumulate and the logit softcap",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int kCount = 3;
    constexpr int kRows = 12;
    const float route_scale = 8.0f;
    std::vector<InklingRoutedSlot> slots(kCount);
    slots[0] = {1, 0.5f * route_scale / 3.0f};
    slots[1] = {4, route_scale / 3.0f};
    slots[2] = {7, 1.5f * route_scale / 3.0f};
    auto y = HPattern(kCount * kRows, 0.5f, 6);
    std::vector<float> ref(static_cast<size_t>(kRows), 0.0f);
    for (int k = 0; k < kCount; ++k)
        for (int i = 0; i < kRows; ++i)
            ref[static_cast<size_t>(i)] +=
                slots[static_cast<size_t>(k)].coeff *
                F(y[static_cast<size_t>(k) * kRows + i]);

    InklingRoutedSlot *d_slots = ToDevice(slots);
    __half *d_y = ToDevice(y);
    float *d_acc = nullptr;
    REQUIRE(cudaMalloc(&d_acc, sizeof(float) * kRows) == cudaSuccess);
    REQUIRE(RunInklingExpertAccumulate(Stream0(), d_acc, d_y, d_slots,
                                       kCount, kRows) == cudaSuccess);
    Sync();
    std::vector<float> got(static_cast<size_t>(kRows));
    FromDevice(d_acc, got);
    for (int i = 0; i < kRows; ++i)
        REQUIRE(got[static_cast<size_t>(i)] == Approx(ref[static_cast<size_t>(i)]).margin(1e-4));

    // The single FP32 -> FP16 cast.
    __half *d_half = nullptr;
    REQUIRE(cudaMalloc(&d_half, sizeof(__half) * kRows) == cudaSuccess);
    REQUIRE(RunInklingCastToHalf(Stream0(), d_acc, d_half, kRows) ==
            cudaSuccess);
    Sync();
    std::vector<__half> hgot(static_cast<size_t>(kRows));
    FromDevice(d_half, hgot);
    for (int i = 0; i < kRows; ++i)
        REQUIRE(F(hgot[static_cast<size_t>(i)]) ==
                Approx(ref[static_cast<size_t>(i)])
                    .margin(std::fabs(ref[static_cast<size_t>(i)]) * 2e-3 + 1e-3));

    // Logit softcap: tanh(x/30)*30, and it must bound the magnitude.
    std::vector<float> logits(16);
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = static_cast<float>(i) * 20.0f - 100.0f;
    float *d_logits = ToDevice(logits);
    float *d_sc = nullptr;
    REQUIRE(cudaMalloc(&d_sc, sizeof(float) * logits.size()) == cudaSuccess);
    REQUIRE(RunInklingLogitSoftcap(Stream0(), d_logits, d_sc,
                                   static_cast<int>(logits.size()), 30.0f) ==
            cudaSuccess);
    Sync();
    std::vector<float> sgot(logits.size());
    FromDevice(d_sc, sgot);
    for (size_t i = 0; i < logits.size(); ++i) {
        const double want = Softcap(logits[i], 30.0);
        REQUIRE(sgot[i] == Approx(want).margin(1e-4));
        REQUIRE(std::fabs(sgot[i]) <= 30.0f);
    }
    cudaFree(d_slots);
    cudaFree(d_y);
    cudaFree(d_acc);
    cudaFree(d_half);
    cudaFree(d_logits);
    cudaFree(d_sc);
}

TEST_CASE("inkling attention gate multiplies before the output projection",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    constexpr int64_t kN = 24;
    const auto x = HPattern(kN, 0.5f, 1);
    const auto gate = HPattern(kN, 1.5f, 2);
    std::vector<float> ref(static_cast<size_t>(kN));
    for (int64_t i = 0; i < kN; ++i) {
        const double gv = F(gate[static_cast<size_t>(i)]);
        ref[static_cast<size_t>(i)] = static_cast<float>(
            static_cast<double>(F(x[static_cast<size_t>(i)])) *
            (1.0 / (1.0 + std::exp(-gv))));
    }
    __half *d_x = ToDevice(x);
    __half *d_g = ToDevice(gate);
    __half *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, sizeof(__half) * kN) == cudaSuccess);
    REQUIRE(RunInklingGateMul(Stream0(), d_x, d_g, d_out, kN) == cudaSuccess);
    Sync();
    std::vector<__half> got(static_cast<size_t>(kN));
    FromDevice(d_out, got);
    for (int64_t i = 0; i < kN; ++i)
        REQUIRE(F(got[static_cast<size_t>(i)]) ==
                Approx(ref[static_cast<size_t>(i)])
                    .margin(std::fabs(ref[static_cast<size_t>(i)]) * 2e-3 + 2e-3));
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_out);
}
