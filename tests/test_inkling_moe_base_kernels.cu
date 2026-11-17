// Inkling (base) numeric kernels (Model Ingestion Protocol stages 5/6).
//
// Every case compares the device kernel against an INDEPENDENT host reference
// written from the reference prose (transformers models/inkling), not against
// the kernel's own output. The deltas that break a copied inkling2 decoder are
// pinned here: 1/head_dim scale, no sinks, static conv WITHOUT normalizer,
// joint routed+shared router renorm, plain SwiGLU.
//
// Label `inference` (kernels stage); SKIPs (4) when no CUDA device is present.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/inkling_moe_base_kernels.h"
#include "lsxcommon/lsx_kv_cache.h"

using namespace lsxcommon::inkling::base;
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

cudaStream_t S0() { return nullptr; }

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
T *Dev(const std::vector<T> &h)
{
    void *d = nullptr;
    REQUIRE(cudaMalloc(&d, h.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return static_cast<T *>(d);
}

template <typename T>
std::vector<T> Host(const T *d, size_t n)
{
    std::vector<T> h(n);
    REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return h;
}

void Free(void *p) { cudaFree(p); }

}  // namespace

TEST_CASE("inkling base head norm matches the host reference",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int rows = 6, dim = 128;
    const float eps = 1e-6f;
    const auto x = FPattern(rows * dim, 0.1f, 1);
    auto wh = HPattern(dim, 0.05f, 2);
    for (auto &w : wh) w = H(0.9f + 0.1f * F(w));  // weights near 1
    std::vector<float> want(static_cast<size_t>(rows * dim));
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < dim; ++i)
            ss += double(x[r * dim + i]) * x[r * dim + i];
        const float inv = 1.0f / std::sqrt(float(ss / dim) + eps);
        for (int i = 0; i < dim; ++i)
            want[r * dim + i] = x[r * dim + i] * inv * F(wh[i]);
    }
    auto *dx = Dev(x);
    auto *dw = Dev(wh);
    float *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, want.size() * 4) == cudaSuccess);
    REQUIRE(RunHeadNorm(S0(), dx, dw, dout, rows, dim, eps) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = Host(dout, want.size());
    for (size_t i = 0; i < want.size(); ++i)
        REQUIRE(got[i] == Approx(want[i]).epsilon(1e-4).margin(1e-5));
    Free(dx); Free(dw); Free(dout);
}

TEST_CASE("inkling base relative bias gathers by backward distance",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int heads = 4, d_rel = 16, extent = 32, q_pos = 40, kv_len = 41;
    const auto rel = FPattern(heads * d_rel, 0.05f, 3);
    const auto proj = FPattern(d_rel * extent, 0.02f, 4);
    std::vector<float> want(static_cast<size_t>(heads * kv_len), 0.0f);
    for (int h = 0; h < heads; ++h)
        for (int j = 0; j < kv_len; ++j) {
            const int d = q_pos - j;
            if (d < 0 || d >= extent) continue;
            float v = 0;
            for (int c = 0; c < d_rel; ++c)
                v += rel[h * d_rel + c] * proj[c * extent + d];
            want[h * kv_len + j] = v;
        }
    auto *drel = Dev(rel);
    auto *dproj = Dev(proj);
    float *dbias = nullptr;
    REQUIRE(cudaMalloc(&dbias, want.size() * 4) == cudaSuccess);
    REQUIRE(RunRelBias(S0(), drel, dproj, dbias, heads, d_rel, extent, q_pos,
                       kv_len) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = Host(dbias, want.size());
    for (size_t i = 0; i < want.size(); ++i)
        REQUIRE(got[i] == Approx(want[i]).epsilon(1e-4).margin(1e-6));
    Free(drel); Free(dproj); Free(dbias);
}

TEST_CASE("inkling base attention: 1/d scale, window, bias, NO sinks",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    // Two kv widths (sliding 4 heads / full 2 heads analogue) and a window
    // boundary; the host reference is a plain softmax over the visible keys.
    for (int kv_heads : {2, 4}) {
        const int heads = 8, hd = 16, pos = 20;
        const int window = (kv_heads == 4) ? 8 : 0;
        const float scale = 1.0f / float(hd);  // NOT 1/sqrt(hd)
        const auto q = HPattern(heads * hd, 0.2f, 5);
        const auto k = HPattern((pos + 1) * kv_heads * hd, 0.2f, 6);
        const auto v = HPattern((pos + 1) * kv_heads * hd, 0.2f, 7);
        const auto bias = FPattern(heads * (pos + 1), 0.1f, 8);
        const auto geom = KvCacheGeometry::Make(
            KvCacheLayout::kPosMajor, pos + 1, kv_heads, hd, 1);
        std::vector<float> want(static_cast<size_t>(heads * hd));
        const int group = heads / kv_heads;
        for (int h = 0; h < heads; ++h) {
            const int kvh = h / group;
            std::vector<float> logits(static_cast<size_t>(pos + 1));
            float mx = -1e30f;
            for (int j = 0; j <= pos; ++j) {
                const bool visible = (window == 0) || (j > pos - window);
                if (!visible) { logits[j] = -1e30f; continue; }
                float dot = 0;
                for (int i = 0; i < hd; ++i)
                    dot += F(q[h * hd + i]) *
                           F(k[j * kv_heads * hd + kvh * hd + i]);
                logits[j] = dot * scale + bias[h * (pos + 1) + j];
                mx = std::max(mx, logits[j]);
            }
            float sum = 0;
            for (int j = 0; j <= pos; ++j) {
                logits[j] = (logits[j] <= -1e29f) ? 0.0f
                                                  : std::exp(logits[j] - mx);
                sum += logits[j];
            }
            for (int i = 0; i < hd; ++i) {
                float acc = 0;
                for (int j = 0; j <= pos; ++j)
                    acc += (logits[j] / sum) *
                           F(v[j * kv_heads * hd + kvh * hd + i]);
                want[h * hd + i] = acc;
            }
        }
        auto *dq = Dev(q);
        auto *dk = Dev(k);
        auto *dv = Dev(v);
        auto *dbias = Dev(bias);
        __half *dout = nullptr;
        REQUIRE(cudaMalloc(&dout, want.size() * 2) == cudaSuccess);
        REQUIRE(RunAttention(S0(), dq, dk, dv, geom, heads, pos, dbias, window,
                             scale, dout) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto got = Host(dout, want.size());
        for (size_t i = 0; i < want.size(); ++i)
            REQUIRE(F(got[i]) == Approx(want[i]).epsilon(2e-3).margin(2e-3));
        Free(dq); Free(dk); Free(dv); Free(dbias); Free(dout);
    }
}

TEST_CASE("inkling base short conv: conv(x)+x, static taps, no normalizer",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int channels = 64, k = 4, steps = 6;
    const auto w = FPattern(channels * k, 0.1f, 9);
    // Host ring reference over `steps` tokens.
    std::vector<float> state(static_cast<size_t>(k * channels), 0.0f);
    auto *dstate = Dev(state);
    auto *dw = Dev(w);
    for (int t = 0; t < steps; ++t) {
        const auto x = HPattern(channels, 0.3f, 10 + t);
        std::vector<float> want(static_cast<size_t>(channels));
        for (int c = 0; c < channels; ++c) {
            for (int j = 0; j + 1 < k; ++j)
                state[j * channels + c] = state[(j + 1) * channels + c];
            state[(k - 1) * channels + c] = F(x[c]);
            float acc = 0;
            for (int j = 0; j < k; ++j)
                acc += w[c * k + j] * state[j * channels + c];
            want[c] = acc + F(x[c]);  // NO 1/sqrt(k+1)
        }
        auto *dx = Dev(x);
        __half *dout = nullptr;
        REQUIRE(cudaMalloc(&dout, channels * 2) == cudaSuccess);
        REQUIRE(RunShortConv(S0(), dx, dstate, dw, dout, channels, k) ==
                cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        const auto got = Host(dout, static_cast<size_t>(channels));
        for (int c = 0; c < channels; ++c)
            REQUIRE(F(got[c]) == Approx(want[c]).epsilon(2e-3).margin(2e-3));
        Free(dx); Free(dout);
    }
    // The ring state itself must match the host ring after the last step.
    const auto got_state = Host(dstate, state.size());
    for (size_t i = 0; i < state.size(); ++i)
        REQUIRE(got_state[i] == Approx(state[i]).epsilon(1e-5));
    Free(dstate); Free(dw);
}

TEST_CASE("inkling base router: joint routed+shared renorm, bias selects only",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int E = 8, S = 2, K = 3, Hd = 64;
    const float route_scale = 8.0f, gs = 0.5f;  // non-unit global scale
    auto w = FPattern((E + S) * Hd, 0.05f, 11);
    // Make the shared rows strongly positive so their gammas are large.
    for (int s = 0; s < S; ++s)
        for (int i = 0; i < Hd; ++i)
            w[(E + s) * Hd + i] = 0.08f;
    auto bias = FPattern(E, 0.0f, 12);
    // A bias that FLIPS the selection: expert 7 gets a huge selection bias but
    // its raw score stays small — its weight must NOT include the bias.
    bias[7] = 5.0f;
    const auto x = HPattern(Hd, 0.1f, 13);

    // Host reference.
    std::vector<float> logits(static_cast<size_t>(E + S));
    for (int e = 0; e < E + S; ++e) {
        float dot = 0;
        for (int i = 0; i < Hd; ++i) dot += w[e * Hd + i] * F(x[i]);
        logits[e] = dot;
    }
    std::vector<int> sel;
    std::vector<bool> taken(static_cast<size_t>(E), false);
    for (int kk = 0; kk < K; ++kk) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; ++e) {
            if (taken[e]) continue;
            const float sc = 1.0f / (1.0f + std::exp(-logits[e])) + bias[e];
            if (sc > bv) { bv = sc; best = e; }
        }
        taken[best] = true;
        sel.push_back(best);
    }
    std::sort(sel.begin(), sel.end());
    // softmax over logsigmoid of the K selected + S shared logits.
    std::vector<float> ls;
    for (int e : sel) ls.push_back(-std::log(1.0f + std::exp(-logits[e])));
    for (int s = 0; s < S; ++s)
        ls.push_back(-std::log(1.0f + std::exp(-logits[E + s])));
    float mx = *std::max_element(ls.begin(), ls.end());
    float sum = 0;
    for (auto &v : ls) { v = std::exp(v - mx); sum += v; }
    std::vector<float> want_w;
    for (size_t i = 0; i < ls.size(); ++i)
        want_w.push_back(ls[i] / sum * route_scale * gs);

    auto *dw = Dev(w);
    auto *db = Dev(bias);
    auto *dx = Dev(x);
    RoutedSlot *dslots = nullptr;
    int32_t *dids = nullptr;
    float *dgammas = nullptr;
    REQUIRE(cudaMalloc(&dslots, K * sizeof(RoutedSlot)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dids, K * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgammas, S * 4) == cudaSuccess);
    REQUIRE(RunRouter(S0(), dw, db, dx, E, S, K, Hd, route_scale, gs, dslots,
                      dids, dgammas) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto slots = Host(dslots, static_cast<size_t>(K));
    const auto ids = Host(dids, static_cast<size_t>(K));
    const auto gammas = Host(dgammas, static_cast<size_t>(S));
    for (int kk = 0; kk < K; ++kk) {
        REQUIRE(ids[kk] == sel[kk]);
        REQUIRE(slots[kk].eid == sel[kk]);
        REQUIRE(slots[kk].coeff ==
                Approx(want_w[kk]).epsilon(1e-4).margin(1e-5));
    }
    for (int s = 0; s < S; ++s)
        REQUIRE(gammas[s] ==
                Approx(want_w[K + s]).epsilon(1e-4).margin(1e-5));
    // The weights sum to route_scale*gs over the K+S slots (joint renorm).
    float total = 0;
    for (const auto &sl : slots) total += sl.coeff;
    for (float g : gammas) total += g;
    REQUIRE(total == Approx(route_scale * gs).epsilon(1e-4));
    Free(dw); Free(db); Free(dx); Free(dslots); Free(dids); Free(dgammas);
}

TEST_CASE("inkling base SwiGLU is plain silu(g)*u (no softcaps)",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int64_t n = 1024;
    // Large magnitudes: a softcapped implementation would clamp these.
    const auto g = HPattern(n, 3.0f, 14);
    const auto u = HPattern(n, 4.0f, 15);
    std::vector<float> want(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float gf = F(g[i]);
        want[i] = (gf / (1.0f + std::exp(-gf))) * F(u[i]);
    }
    auto *dg = Dev(g);
    auto *du = Dev(u);
    __half *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, n * 2) == cudaSuccess);
    REQUIRE(RunSwiGLU(S0(), dg, du, dout, n) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto got = Host(dout, static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        REQUIRE(F(got[i]) == Approx(want[i]).epsilon(2e-3).margin(2e-2));
    Free(dg); Free(du); Free(dout);
}

TEST_CASE("inkling base expert and shared accumulation", "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int count = 3, rows = 64;
    const auto y = HPattern(count * rows, 0.2f, 16);
    std::vector<RoutedSlot> slots(static_cast<size_t>(count));
    for (int k = 0; k < count; ++k) {
        slots[k].eid = k;
        slots[k].coeff = 0.5f + 0.25f * k;
    }
    std::vector<float> want(static_cast<size_t>(rows), 0.0f);
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < count; ++k)
            want[i] += slots[k].coeff * F(y[k * rows + i]);
    auto *dy = Dev(y);
    auto *dslots = Dev(slots);
    float *dacc = nullptr;
    REQUIRE(cudaMalloc(&dacc, rows * 4) == cudaSuccess);
    REQUIRE(RunExpertAccumulate(S0(), dacc, dy, dslots, count, rows) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto got = Host(dacc, static_cast<size_t>(rows));
    for (int i = 0; i < rows; ++i)
        REQUIRE(got[i] == Approx(want[i]).epsilon(1e-4).margin(1e-5));
    // Shared accumulate adds gamma*y on top.
    const auto sy = HPattern(rows, 0.3f, 17);
    const float gamma = 0.7f;
    auto *dsy = Dev(sy);
    REQUIRE(RunSharedAccumulate(S0(), dacc, dsy, gamma, rows) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    got = Host(dacc, static_cast<size_t>(rows));
    for (int i = 0; i < rows; ++i)
        REQUIRE(got[i] ==
                Approx(want[i] + gamma * F(sy[i])).epsilon(1e-4).margin(1e-5));
    Free(dy); Free(dslots); Free(dacc); Free(dsy);
}

TEST_CASE("inkling base kv append addresses through the geometry",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int kv_heads = 4, hd = 16, cap = 16, pos = 5;
    const auto geom =
        KvCacheGeometry::Make(KvCacheLayout::kPosMajor, cap, kv_heads, hd, 1);
    const auto kn = HPattern(kv_heads * hd, 0.1f, 18);
    const auto vn = HPattern(kv_heads * hd, 0.1f, 19);
    std::vector<__half> zero(static_cast<size_t>(geom.Elements()), H(0.0f));
    auto *dk = Dev(zero);
    auto *dv = Dev(zero);
    auto *dkn = Dev(kn);
    auto *dvn = Dev(vn);
    REQUIRE(RunKvAppend(S0(), dkn, dvn, dk, dv, geom, pos) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto got_k = Host(dk, zero.size());
    const auto got_v = Host(dv, zero.size());
    for (int h = 0; h < kv_heads; ++h)
        for (int i = 0; i < hd; ++i) {
            const int64_t off = geom.Offset(0, h, pos, i);
            REQUIRE(F(got_k[off]) == F(kn[h * hd + i]));
            REQUIRE(F(got_v[off]) == F(vn[h * hd + i]));
        }
    Free(dk); Free(dv); Free(dkn); Free(dvn);
}

TEST_CASE("inkling base log scale multiplies q and bias by tau",
          "[inkling][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int64_t nq = 128, nrel = 64;
    const float tau = 1.21f;
    auto q = FPattern(nq, 0.1f, 20);
    auto rel = FPattern(nrel, 0.1f, 21);
    auto *dq = Dev(q);
    auto *drel = Dev(rel);
    REQUIRE(RunLogScale(S0(), dq, nq, drel, nrel, tau) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto gq = Host(dq, static_cast<size_t>(nq));
    const auto grel = Host(drel, static_cast<size_t>(nrel));
    for (int64_t i = 0; i < nq; ++i)
        REQUIRE(gq[i] == Approx(q[i] * tau).epsilon(1e-5));
    for (int64_t i = 0; i < nrel; ++i)
        REQUIRE(grel[i] == Approx(rel[i] * tau).epsilon(1e-5));
    Free(dq); Free(drel);
}