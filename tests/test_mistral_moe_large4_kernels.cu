// Mistral-Large-4 family kernels (Model Ingestion Protocol stage 5). Every
// case compares the device kernel against a host reference written
// independently in this file (external-oracle discipline).
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_large4_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

using namespace lsxcommon::mistral_large4;

namespace {

bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename T>
struct DevBuf {
    T *p = nullptr;
    explicit DevBuf(size_t n) { cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)); }
    ~DevBuf() { if (p) cudaFree(p); }
    T *get() { return p; }
};

std::vector<float> ToHostF32(const float *d, size_t n)
{
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

std::vector<__nv_bfloat16> ToHostBf16(const __nv_bfloat16 *d, size_t n)
{
    std::vector<__nv_bfloat16> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    return v;
}

float Bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

} // namespace

TEST_CASE("Mistral-Large-4 LeRoPE tables match the upstream YaRN oracle",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    // Oracle: transformers `_compute_yarn_parameters` for head_dim 64,
    // theta 1e4, factor 2048, original window 512, beta 32/1.
    const int rope_dim = 64, half = 32;
    std::vector<float> alpha(static_cast<size_t>(half), 0.0f);
    std::vector<float> cos_t, sin_t;
    REQUIRE(BuildLarge4RopeTables(4096, alpha, rope_dim, 1e4, 2048.0, 512.0,
                                  32.0, 1.0, cos_t, sin_t));
    // Angles from the oracle (alpha = 0): inv_freq = {1.0, 0.7498942, ...,
    // 6.511335e-08}.
    const struct { int pos, j; double angle; } kOracle[] = {
        {1, 0, 1.0}, {1, 1, 0.7498942017555237}, {1, 31, 6.511335470804624e-08},
        {100, 0, 100.0}, {100, 1, 74.98941802978516},
        {3000, 0, 3000.0}, {3000, 1, 2249.6826171875},
        {3000, 31, 0.00019534006423782557},
    };
    for (const auto &o : kOracle)
    {
        const size_t i = static_cast<size_t>(o.pos) * half + o.j;
        INFO("pos " << o.pos << " j " << o.j);
        // The oracle angles are float32-rounded, so a large position carries
        // up to ~1e-5 of angle error (and thus the same in cos/sin).
        REQUIRE(std::fabs(cos_t[i] - std::cos(o.angle)) < 5e-5);
        REQUIRE(std::fabs(sin_t[i] - std::sin(o.angle)) < 5e-5);
    }

    // LeRoPE: inv_freq *= exp(alpha). Oracle angles with
    // alpha[j] = 0.1 * ((j % 5) - 2).
    std::vector<float> alpha2(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j) alpha2[static_cast<size_t>(j)] = 0.1f * ((j % 5) - 2);
    REQUIRE(BuildLarge4RopeTables(4096, alpha2, rope_dim, 1e4, 2048.0, 512.0,
                                  32.0, 1.0, cos_t, sin_t));
    REQUIRE(std::fabs(cos_t[3000 * half + 0] - std::cos(2456.1923828125)) < 5e-4);
    REQUIRE(std::fabs(sin_t[3000 * half + 0] - std::sin(2456.1923828125)) < 5e-4);
    REQUIRE(std::fabs(cos_t[3000 * half + 1] - std::cos(2035.5970458984375)) < 5e-4);
    REQUIRE(std::fabs(cos_t[3000 * half + 31] -
                      std::cos(0.00017675099661573768)) < 1e-6);
    REQUIRE(std::fabs(cos_t[100 * half + 0] - std::cos(81.87307739257812)) < 1e-5);
}

TEST_CASE("Mistral-Large-4 interleaved RoPE matches the reference formula",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int heads = 2, nope = 8, rope = 8, head_dim = nope + rope;
    std::vector<float> alpha(static_cast<size_t>(rope / 2), 0.0f);
    std::vector<float> cos_t, sin_t;
    REQUIRE(BuildLarge4RopeTables(16, alpha, rope, 1e4, 1.0, 16.0, 32.0, 1.0,
                                  cos_t, sin_t));
    float *d_cos = nullptr, *d_sin = nullptr;
    REQUIRE(UploadLarge4RopeTables(nullptr, cos_t, sin_t, &d_cos, &d_sin) ==
            cudaSuccess);

    std::vector<float> host(static_cast<size_t>(heads * head_dim));
    for (size_t i = 0; i < host.size(); ++i)
        host[i] = 0.01f * static_cast<float>(static_cast<int>(i % 17) - 8);
    std::vector<float> want = host;
    const int pos = 5;
    for (int h = 0; h < heads; ++h)
        for (int j = 0; j < rope / 2; ++j)
        {
            const size_t b = static_cast<size_t>(h * head_dim + nope + 2 * j);
            const float c = cos_t[static_cast<size_t>(pos) * (rope / 2) + j];
            const float s = sin_t[static_cast<size_t>(pos) * (rope / 2) + j];
            want[b] = host[b] * c - host[b + 1] * s;
            want[b + 1] = host[b + 1] * c + host[b] * s;
        }
    DevBuf<float> d(host.size());
    cudaMemcpy(d.get(), host.data(), host.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4RopeInterleaved(d.get(), heads, head_dim, nope, rope,
                                        d_cos, d_sin, pos, nullptr) ==
            cudaSuccess);
    const auto got = ToHostF32(d.get(), host.size());
    for (size_t i = 0; i < got.size(); ++i) REQUIRE(got[i] == want[i]);

    // Single-vector variant (k_pe).
    std::vector<float> v(rope);
    for (int i = 0; i < rope; ++i) v[static_cast<size_t>(i)] = 0.1f * i;
    std::vector<float> wv = v;
    for (int j = 0; j < rope / 2; ++j)
    {
        const float c = cos_t[static_cast<size_t>(pos) * (rope / 2) + j];
        const float s = sin_t[static_cast<size_t>(pos) * (rope / 2) + j];
        wv[static_cast<size_t>(2 * j)] = v[static_cast<size_t>(2 * j)] * c - v[static_cast<size_t>(2 * j + 1)] * s;
        wv[static_cast<size_t>(2 * j + 1)] = v[static_cast<size_t>(2 * j + 1)] * c + v[static_cast<size_t>(2 * j)] * s;
    }
    DevBuf<float> dv(rope);
    cudaMemcpy(dv.get(), v.data(), rope * sizeof(float), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4RopeInterleavedVec(dv.get(), rope, d_cos, d_sin, pos,
                                           nullptr) == cudaSuccess);
    const auto gv = ToHostF32(dv.get(), rope);
    for (int i = 0; i < rope; ++i) REQUIRE(gv[static_cast<size_t>(i)] == wv[static_cast<size_t>(i)]);
    cudaFree(d_cos);
    cudaFree(d_sin);
}

TEST_CASE("Mistral-Large-4 ssmax scales q by s*log(pos+1)+b",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int heads = 3, dim = 4, pos = 7;
    std::vector<__nv_bfloat16> s(heads), b(heads);
    for (int h = 0; h < heads; ++h)
    {
        s[static_cast<size_t>(h)] = __float2bfloat16(0.1f * (h + 1));
        b[static_cast<size_t>(h)] = __float2bfloat16(0.5f - 0.2f * h);
    }
    std::vector<float> q(static_cast<size_t>(heads * dim));
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.25f * (i + 1);
    std::vector<float> want = q;
    for (int h = 0; h < heads; ++h)
    {
        const float lp = Bf(std::log(static_cast<float>(pos + 1)));
        const float sc = Bf(Bf(__bfloat162float(s[static_cast<size_t>(h)])) * lp +
                            __bfloat162float(b[static_cast<size_t>(h)]));
        for (int d = 0; d < dim; ++d)
            want[static_cast<size_t>(h * dim + d)] *= sc;
    }
    DevBuf<float> dq(q.size());
    DevBuf<__nv_bfloat16> ds(heads), db(heads);
    cudaMemcpy(dq.get(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ds.get(), s.data(), heads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(db.get(), b.data(), heads * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4SsmaxScale(dq.get(), heads, dim, ds.get(), db.get(), pos,
                                   nullptr) == cudaSuccess);
    const auto got = ToHostF32(dq.get(), q.size());
    for (size_t i = 0; i < got.size(); ++i)
        REQUIRE(std::fabs(got[i] - want[i]) <= 1e-5f * std::fabs(want[i]) + 1e-6f);
}

TEST_CASE("Mistral-Large-4 indexer LayerNorm matches the host reference",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int n = 32;
    std::vector<__nv_bfloat16> x(static_cast<size_t>(n)), w(static_cast<size_t>(n)),
        b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        x[static_cast<size_t>(i)] = __float2bfloat16(0.1f * static_cast<float>((i % 9) - 4));
        w[static_cast<size_t>(i)] = __float2bfloat16(1.0f + 0.01f * i);
        b[static_cast<size_t>(i)] = __float2bfloat16(0.05f * static_cast<float>(i % 3));
    }
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; ++i) mean += __bfloat162float(x[static_cast<size_t>(i)]);
    mean /= n;
    for (int i = 0; i < n; ++i)
    {
        const double d = __bfloat162float(x[static_cast<size_t>(i)]) - mean;
        var += d * d;
    }
    var /= n;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + 1e-6));
    std::vector<float> want(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        want[static_cast<size_t>(i)] =
            (__bfloat162float(x[static_cast<size_t>(i)]) -
             static_cast<float>(mean)) *
                inv * __bfloat162float(w[static_cast<size_t>(i)]) +
            __bfloat162float(b[static_cast<size_t>(i)]);

    DevBuf<__nv_bfloat16> dx(n), dw(n), db(n);
    DevBuf<float> dy(n);
    cudaMemcpy(dx.get(), x.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dw.get(), w.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(db.get(), b.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4LayerNorm(dx.get(), dw.get(), db.get(), n, 1e-6f, dy.get(),
                                  nullptr) == cudaSuccess);
    const auto got = ToHostF32(dy.get(), n);
    for (int i = 0; i < n; ++i)
        REQUIRE(std::fabs(got[static_cast<size_t>(i)] - want[static_cast<size_t>(i)]) <
                2e-4f);
}

TEST_CASE("Mistral-Large-4 router matches softmax+top-k+renormalize",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int E = 16, H = 24, K = 4;
    std::vector<__nv_bfloat16> w(static_cast<size_t>(E * H));
    std::vector<float> x(static_cast<size_t>(H));
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(0.05f * static_cast<float>((i % 23) - 11));
    for (int i = 0; i < H; ++i) x[static_cast<size_t>(i)] = 0.1f * static_cast<float>((i % 7) - 3);

    std::vector<float> logits(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e)
    {
        double acc = 0.0;
        for (int k = 0; k < H; ++k)
            acc += __bfloat162float(w[static_cast<size_t>(e * H + k)]) * x[static_cast<size_t>(k)];
        logits[static_cast<size_t>(e)] = static_cast<float>(acc);
    }
    double mx = logits[0];
    for (float v : logits) mx = std::max(mx, static_cast<double>(v));
    double den = 0.0;
    for (auto &v : logits)
    {
        v = std::exp(static_cast<float>(static_cast<double>(v) - mx));
        den += v;
    }
    for (auto &v : logits) v = static_cast<float>(static_cast<double>(v) / den);
    std::vector<int> order(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (logits[static_cast<size_t>(a)] != logits[static_cast<size_t>(b)])
            return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
        return a < b;
    });
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += logits[static_cast<size_t>(order[static_cast<size_t>(i)])];

    DevBuf<__nv_bfloat16> dw(w.size());
    DevBuf<float> dx(x.size());
    DevBuf<lsxcommon::moe::MoERoutedSlot> slots(static_cast<size_t>(K));
    cudaMemcpy(dw.get(), w.data(), w.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dx.get(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4Router(dw.get(), dx.get(), E, K, H, slots.get(), nullptr) ==
            cudaSuccess);
    std::vector<lsxcommon::moe::MoERoutedSlot> host(static_cast<size_t>(K));
    cudaMemcpy(host.data(), slots.get(), K * sizeof(lsxcommon::moe::MoERoutedSlot),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < K; ++i)
    {
        REQUIRE(host[static_cast<size_t>(i)].eid == order[static_cast<size_t>(i)]);
        const float want = logits[static_cast<size_t>(order[static_cast<size_t>(i)])] / sum;
        REQUIRE(std::fabs(host[static_cast<size_t>(i)].coeff - want) < 2e-5f);
    }
}

TEST_CASE("Mistral-Large-4 indexer scores and exact top-k match the host",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int heads = 4, dim = 16, n = 3000;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(heads * dim));
    std::vector<float> wts(static_cast<size_t>(heads));
    std::vector<__nv_bfloat16> k(static_cast<size_t>(n * dim));
    for (auto &v : q) v = dist(rng);
    for (auto &v : wts) v = dist(rng);
    for (auto &v : k) v = __float2bfloat16(dist(rng));
    const float scale = 0.08838834764831845f;

    std::vector<float> want(static_cast<size_t>(n));
    for (int t = 0; t < n; ++t)
    {
        double acc = 0.0;
        for (int h = 0; h < heads; ++h)
        {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d)
                dot += q[static_cast<size_t>(h * dim + d)] *
                       __bfloat162float(k[static_cast<size_t>(t * dim + d)]);
            acc += wts[static_cast<size_t>(h)] *
                   std::max(static_cast<double>(0.0),
                            static_cast<double>(static_cast<float>(dot) * scale));
        }
        want[static_cast<size_t>(t)] = static_cast<float>(acc);
    }

    DevBuf<float> dq(q.size()), dw(wts.size()), ds(static_cast<size_t>(n));
    DevBuf<__nv_bfloat16> dk(k.size());
    cudaMemcpy(dq.get(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw.get(), wts.data(), wts.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk.get(), k.data(), k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4IndexerScores(dq.get(), dk.get(), dw.get(), heads, dim,
                                      scale, n, ds.get(), nullptr) == cudaSuccess);
    const auto got = ToHostF32(ds.get(), static_cast<size_t>(n));
    for (int t = 0; t < n; ++t)
        REQUIRE(std::fabs(got[static_cast<size_t>(t)] - want[static_cast<size_t>(t)]) <
                1e-3f);

    const int kk = 2048;
    DevBuf<int32_t> idx(static_cast<size_t>(kk));
    DevBuf<int> cnt(1);
    REQUIRE(LaunchLarge4TopkSelect(ds.get(), n, kk, idx.get(), cnt.get(), nullptr) ==
            cudaSuccess);
    int count = 0;
    cudaMemcpy(&count, cnt.get(), sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(count == kk);
    std::vector<int32_t> sel(static_cast<size_t>(kk));
    cudaMemcpy(sel.data(), idx.get(), kk * sizeof(int32_t), cudaMemcpyDeviceToHost);
    // Ascending, unique, and exactly the top-k multiset.
    for (int i = 1; i < kk; ++i)
        REQUIRE(sel[static_cast<size_t>(i)] > sel[static_cast<size_t>(i - 1)]);
    std::vector<float> sorted = want;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float thr = sorted[static_cast<size_t>(kk - 1)];
    for (int i = 0; i < kk; ++i)
        REQUIRE(want[static_cast<size_t>(sel[static_cast<size_t>(i)])] >= thr);
    int above = 0;
    for (float v : want) above += v > thr;
    REQUIRE(above <= kk);

    // n <= k is the identity.
    REQUIRE(LaunchLarge4TopkSelect(ds.get(), 100, kk, idx.get(), cnt.get(),
                                   nullptr) == cudaSuccess);
    cudaMemcpy(&count, cnt.get(), sizeof(int), cudaMemcpyDeviceToHost);
    REQUIRE(count == 100);
}

TEST_CASE("Mistral-Large-4 absorbed MLA equals expand-then-attend",
          "[mistral_large4][kernels]")
{
    if (!GpuAvailable()) std::exit(4);
    const int heads = 2, kn = 4, v_head = 4, kv_lora = 8, qk_head = kn + 4;
    const int pos = 3;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    // wkv_b rows: heads x (kn + v_head), cols kv_lora.
    std::vector<__nv_bfloat16> wkv(static_cast<size_t>(heads * (kn + v_head) * kv_lora));
    for (auto &v : wkv) v = __float2bfloat16(dist(rng));
    // q is [heads][qk_head]: nope in the first kn lanes, rope in the rest.
    std::vector<float> q(static_cast<size_t>(heads * qk_head));
    for (auto &v : q) v = dist(rng);
    std::vector<float> latent(static_cast<size_t>(pos * kv_lora));
    for (auto &v : latent) v = dist(rng);
    std::vector<float> probs(static_cast<size_t>(heads * pos));
    for (auto &v : probs) v = std::fabs(dist(rng));
    for (int h = 0; h < heads; ++h)
    {
        double s = 0.0;
        for (int t = 0; t < pos; ++t) s += probs[static_cast<size_t>(h * pos + t)];
        for (int t = 0; t < pos; ++t)
            probs[static_cast<size_t>(h * pos + t)] = static_cast<float>(probs[static_cast<size_t>(h * pos + t)] / s);
    }

    // Host reference: ctx[h] = sum_t p_t latent_t; out[h] = Wv_h ctx[h].
    std::vector<float> ctx(static_cast<size_t>(heads * kv_lora), 0.0f);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < pos; ++t)
            for (int c = 0; c < kv_lora; ++c)
                ctx[static_cast<size_t>(h * kv_lora + c)] +=
                    probs[static_cast<size_t>(h * pos + t)] * latent[static_cast<size_t>(t * kv_lora + c)];
    std::vector<float> want(static_cast<size_t>(heads * v_head));
    for (int h = 0; h < heads; ++h)
        for (int v = 0; v < v_head; ++v)
        {
            double acc = 0.0;
            const int row = h * (kn + v_head) + kn + v;
            for (int c = 0; c < kv_lora; ++c)
                acc += __bfloat162float(wkv[static_cast<size_t>(row * kv_lora + c)]) *
                       ctx[static_cast<size_t>(h * kv_lora + c)];
            want[static_cast<size_t>(h * v_head + v)] = static_cast<float>(acc);
        }

    DevBuf<__nv_bfloat16> dw(wkv.size());
    DevBuf<float> dctx(ctx.size()), dout(static_cast<size_t>(heads * v_head));
    cudaMemcpy(dw.get(), wkv.data(), wkv.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dctx.get(), ctx.data(), ctx.size() * sizeof(float), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4ValueProject(dw.get(), dctx.get(), heads, kn, v_head,
                                     kv_lora, dout.get(), nullptr) == cudaSuccess);
    const auto got = ToHostF32(dout.get(), static_cast<size_t>(heads * v_head));
    for (size_t i = 0; i < got.size(); ++i)
        REQUIRE(std::fabs(got[i] - want[i]) < 2e-3f);

    // Quarter of absorption: q_abs[h][c] = sum_d q_nope[h][d]*Wk_h[d][c], and
    // the score equals the explicit expanded dot product.
    // (Computed on the host from the same inputs for the first head only.)
    DevBuf<float> dq(q.size()), dabs(static_cast<size_t>(heads * kv_lora));
    cudaMemcpy(dq.get(), q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    REQUIRE(LaunchLarge4AbsorbQuery(dw.get(), dq.get(), heads, qk_head, kn,
                                    v_head, kv_lora, dabs.get(), nullptr) ==
            cudaSuccess);
    const auto absq = ToHostF32(dabs.get(), static_cast<size_t>(heads * kv_lora));
    for (int h = 0; h < heads; ++h)
        for (int c = 0; c < kv_lora; ++c)
        {
            double dot_abs = 0.0, dot_exp = 0.0;
            for (int d = 0; d < kn; ++d)
            {
                const int row = h * (kn + v_head) + d;
                const float wk = __bfloat162float(wkv[static_cast<size_t>(row * kv_lora + c)]);
                dot_abs += q[static_cast<size_t>(h * qk_head + d)] * wk;
                dot_exp += q[static_cast<size_t>(h * qk_head + d)] *
                           __bfloat162float(wkv[static_cast<size_t>(row * kv_lora + c)]);
            }
            REQUIRE(std::fabs(absq[static_cast<size_t>(h * kv_lora + c)] -
                              static_cast<float>(dot_abs)) < 1e-3f);
            (void)dot_exp;
        }
}
