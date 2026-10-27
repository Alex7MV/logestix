// MiniMax-M3 numeric kernels (GPU parity against host references).
//
// Covers the model-specific arithmetic that no shared kernel provides:
// the sigmoid+bias router, SwiGLU-OAI, Gemma per-head norm, partial RoPE, the
// MSA indexer block selection, masked decode attention, and the routed
// accumulate/combine epilogue. Each case compares against a host reference
// written from the HF modeling code, not from our kernel.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "lsxcommon/minimax_moe_kernels.h"

using namespace lsxcommon::minimax;

namespace {

struct DevBuf {
    void *p = nullptr;
    explicit DevBuf(size_t bytes) { REQUIRE(cudaMalloc(&p, bytes) == cudaSuccess); }
    ~DevBuf() { if (p) cudaFree(p); }
    template <typename T> T *as() { return static_cast<T *>(p); }
};

void H2D(void *dst, const void *src, size_t bytes)
{
    REQUIRE(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
}
void D2H(void *dst, const void *src, size_t bytes)
{
    REQUIRE(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
}

std::vector<float> RandomF32(int64_t n, uint32_t seed, float scale = 1.0f)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(static_cast<size_t>(n));
    for (auto &x : v) x = d(rng);
    return v;
}

std::vector<__half> ToHalf(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

float SigmoidHost(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

TEST_CASE("MiniMax router: sigmoid + routing bias + renormalized top-k",
          "[minimax][kernels]")
{
    const int E = 8, K = 3, H = 32;
    auto w = RandomF32(static_cast<int64_t>(E) * H, 7, 0.5f);
    auto bias = RandomF32(E, 11, 0.2f);
    auto xf = RandomF32(H, 13, 1.0f);
    auto x = ToHalf(xf);

    DevBuf dw(sizeof(float) * static_cast<size_t>(E) * H);
    DevBuf db(sizeof(float) * static_cast<size_t>(E));
    DevBuf dx(sizeof(__half) * static_cast<size_t>(H));
    DevBuf dslots(sizeof(MinimaxRoutedSlot) * static_cast<size_t>(K));
    H2D(dw.p, w.data(), sizeof(float) * w.size());
    H2D(db.p, bias.data(), sizeof(float) * bias.size());
    H2D(dx.p, x.data(), sizeof(__half) * x.size());
    REQUIRE(RunMinimaxRouter(nullptr, dw.as<float>(), db.as<float>(),
                             dx.as<__half>(), 1, E, K, H,
                             dslots.as<MinimaxRoutedSlot>()) ==
            cudaSuccess);
    std::vector<MinimaxRoutedSlot> got(static_cast<size_t>(K));
    D2H(got.data(), dslots.p, sizeof(MinimaxRoutedSlot) * K);

    // Host reference (modeling_minimax_m3_vl.MiniMaxM3VLTopKRouter).
    std::vector<float> score(E), choice(E);
    for (int e = 0; e < E; ++e) {
        double acc = 0.0;
        for (int k = 0; k < H; ++k)
            acc += static_cast<double>(w[static_cast<size_t>(e) * H + k]) *
                   static_cast<double>(__half2float(x[static_cast<size_t>(k)]));
        score[static_cast<size_t>(e)] = SigmoidHost(static_cast<float>(acc));
        choice[static_cast<size_t>(e)] = score[static_cast<size_t>(e)] +
                                         bias[static_cast<size_t>(e)];
    }
    std::vector<int> idx(E);
    for (int e = 0; e < E; ++e) idx[static_cast<size_t>(e)] = e;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return choice[static_cast<size_t>(a)] > choice[static_cast<size_t>(b)];
    });
    std::vector<int> sel(idx.begin(), idx.begin() + K);
    std::sort(sel.begin(), sel.end());  // kernel stores ascending ids
    float sum = 0.0f;
    for (int e : sel) sum += score[static_cast<size_t>(e)];
    for (int k = 0; k < K; ++k) {
        CHECK(got[static_cast<size_t>(k)].eid == sel[static_cast<size_t>(k)]);
        CHECK(got[static_cast<size_t>(k)].coeff ==
              Catch::Approx(score[static_cast<size_t>(sel[static_cast<size_t>(k)])] /
                            sum)
                  .epsilon(1e-5));
    }
}

TEST_CASE("MiniMax SwiGLU-OAI matches the GPT-OSS-style reference",
          "[minimax][kernels]")
{
    const int n = 64;
    auto gf = RandomF32(n, 21, 9.0f);   // spans the clamp bound (7.0)
    auto uf = RandomF32(n, 22, 9.0f);
    auto g = ToHalf(gf);
    auto u = ToHalf(uf);
    DevBuf dg(sizeof(__half) * n), du(sizeof(__half) * n), dout(sizeof(__half) * n);
    H2D(dg.p, g.data(), sizeof(__half) * n);
    H2D(du.p, u.data(), sizeof(__half) * n);
    REQUIRE(RunMinimaxSwiGLUOai(nullptr, dg.as<__half>(), du.as<__half>(),
                                dout.as<__half>(), n, 1.702f, 7.0f) ==
            cudaSuccess);
    std::vector<__half> out(n);
    D2H(out.data(), dout.p, sizeof(__half) * n);
    for (int i = 0; i < n; ++i) {
        const float gv = fminf(__half2float(g[static_cast<size_t>(i)]), 7.0f);
        const float uv = fminf(fmaxf(__half2float(u[static_cast<size_t>(i)]),
                                     -7.0f), 7.0f);
        const float want = (uv + 1.0f) * gv * SigmoidHost(1.702f * gv);
        CHECK(__half2float(out[static_cast<size_t>(i)]) ==
              Catch::Approx(want).epsilon(2e-3).margin(2e-3));
    }
}

TEST_CASE("MiniMax per-head Gemma norm uses the (1 + w) scale", "[minimax][kernels]")
{
    const int rows = 2, heads = 3, HD = 8;
    auto xf = RandomF32(rows * heads * HD, 31, 1.0f);
    auto wf = RandomF32(HD, 32, 0.5f);
    auto x = ToHalf(xf);
    auto w = ToHalf(wf);
    DevBuf dx(sizeof(__half) * x.size()), dw(sizeof(__half) * w.size()),
        dout(sizeof(__half) * x.size());
    H2D(dx.p, x.data(), sizeof(__half) * x.size());
    H2D(dw.p, w.data(), sizeof(__half) * w.size());
    REQUIRE(RunMinimaxHeadNorm(nullptr, dx.as<__half>(), dw.as<__half>(),
                               dout.as<__half>(), rows, heads, HD, 1e-6f) ==
            cudaSuccess);
    std::vector<__half> out(x.size());
    D2H(out.data(), dout.p, sizeof(__half) * out.size());
    for (int r = 0; r < rows * heads; ++r) {
        double ss = 0.0;
        for (int d = 0; d < HD; ++d) {
            const double v = __half2float(x[static_cast<size_t>(r) * HD + d]);
            ss += v * v;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(ss / HD + 1e-6));
        for (int d = 0; d < HD; ++d) {
            const float v = __half2float(x[static_cast<size_t>(r) * HD + d]);
            const float want = v * inv * (1.0f + __half2float(w[static_cast<size_t>(d)]));
            CHECK(__half2float(out[static_cast<size_t>(r) * HD + d]) ==
                  Catch::Approx(want).epsilon(2e-3).margin(2e-3));
        }
    }
}

TEST_CASE("MiniMax partial RoPE rotates only the first rotary_dim channels",
          "[minimax][kernels]")
{
    const int heads = 2, HD = 16, rot = 8, pos = 5;
    const float theta = 5000000.0f;
    auto xf = RandomF32(heads * HD, 41, 1.0f);
    auto x = ToHalf(xf);
    DevBuf dx(sizeof(__half) * x.size());
    H2D(dx.p, x.data(), sizeof(__half) * x.size());
    REQUIRE(RunMinimaxPartialRope(nullptr, dx.as<__half>(), nullptr, 1, heads, 0,
                                  HD, rot, theta, pos) == cudaSuccess);
    std::vector<__half> out(x.size());
    D2H(out.data(), dx.p, sizeof(__half) * out.size());
    const int half = rot / 2;
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < half; ++j) {
            const float inv_freq =
                1.0f / std::pow(theta, static_cast<float>(j) / half);
            const float ang = static_cast<float>(pos) * inv_freq;
            const float c = std::cos(ang), s = std::sin(ang);
            const float x1 = __half2float(x[static_cast<size_t>(h) * HD + j]);
            const float x2 = __half2float(x[static_cast<size_t>(h) * HD + j + half]);
            CHECK(__half2float(out[static_cast<size_t>(h) * HD + j]) ==
                  Catch::Approx(x1 * c - x2 * s).epsilon(3e-3).margin(3e-3));
            CHECK(__half2float(out[static_cast<size_t>(h) * HD + j + half]) ==
                  Catch::Approx(x1 * s + x2 * c).epsilon(3e-3).margin(3e-3));
        }
        for (int d = rot; d < HD; ++d)
            CHECK(out[static_cast<size_t>(h) * HD + d] ==
                  x[static_cast<size_t>(h) * HD + d]);
    }
}

TEST_CASE("MiniMax indexer selects the strongest block plus the local block",
          "[minimax][kernels]")
{
    // 4 index heads x 8 dims; 32 keys in blocks of 8 => 4 blocks. Query
    // position 20 (block 2): the local selection is block 2, and the crafted
    // key alignment must add block 0 (or the strongest non-local one).
    const int IH = 4, ID = 8, block = 8, kv = 32, pos = 20, topk = 2, local = 1;
    std::vector<__half> q(static_cast<size_t>(IH) * ID);
    std::vector<__half> k(static_cast<size_t>(kv) * ID);
    for (int h = 0; h < IH; ++h) {
        for (int d = 0; d < ID; ++d)
            q[static_cast<size_t>(h) * ID + d] = __float2half(d == 0 ? 1.0f : 0.0f);
    }
    // All keys are orthogonal (0) except one key in block 0 which aligns with
    // q head 0 and one in block 3 (future, must never win).
    for (auto &v : k) v = __float2half(0.0f);
    k[0 * ID + 0] = __float2half(3.0f);   // block 0, key 0 -> score 3
    k[31 * ID + 0] = __float2half(10.0f); // block 3, key 31 -> future
    DevBuf dq(sizeof(__half) * q.size()), dk(sizeof(__half) * k.size());
    const int num_blocks = (kv + block - 1) / block;
    DevBuf dmask(static_cast<size_t>(IH) * num_blocks);
    H2D(dq.p, q.data(), sizeof(__half) * q.size());
    H2D(dk.p, k.data(), sizeof(__half) * k.size());
    REQUIRE(RunMinimaxIndexerSelect(nullptr, dq.as<__half>(), dk.as<__half>(),
                                    kv, pos, IH, ID, block, topk, local,
                                    dmask.as<uint8_t>()) == cudaSuccess);
    std::vector<uint8_t> mask(static_cast<size_t>(IH) * num_blocks);
    D2H(mask.data(), dmask.p, mask.size());
    for (int h = 0; h < IH; ++h) {
        const uint8_t *m = mask.data() + static_cast<size_t>(h) * num_blocks;
        CHECK(m[2] == 1);   // local block always kept
        CHECK(m[3] == 0);   // future block never kept
        if (h == 0)
            CHECK(m[0] == 1);  // strongest (score 3) non-local block
        CHECK(m[0] + m[1] + m[2] + m[3] == topk);
    }
}

TEST_CASE("MiniMax masked attention equals a host softmax over allowed keys",
          "[minimax][kernels]")
{
    const int heads = 2, kv_heads = 1, HD = 8, kv = 8, block = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    auto qf = RandomF32(heads * HD, 51, 1.0f);
    auto kf = RandomF32(kv * kv_heads * HD, 52, 1.0f);
    auto vf = RandomF32(kv * kv_heads * HD, 53, 1.0f);
    auto q = ToHalf(qf), k = ToHalf(kf), v = ToHalf(vf);
    DevBuf dq(sizeof(__half) * q.size()), dk(sizeof(__half) * k.size()),
        dv(sizeof(__half) * v.size());
    DevBuf dout(sizeof(__half) * static_cast<size_t>(heads) * HD);
    DevBuf dsc(sizeof(float) * static_cast<size_t>(heads) * kv);
    H2D(dq.p, q.data(), sizeof(__half) * q.size());
    H2D(dk.p, k.data(), sizeof(__half) * k.size());
    H2D(dv.p, v.data(), sizeof(__half) * v.size());

    for (int masked = 0; masked < 2; ++masked) {
        const int num_blocks = (kv + block - 1) / block;
        std::vector<uint8_t> mask(static_cast<size_t>(heads) * num_blocks, 1);
        if (masked) {
            // Head 0 keeps blocks {0,2}; head 1 keeps {1,3}.
            for (int b = 0; b < num_blocks; ++b) {
                mask[static_cast<size_t>(0) * num_blocks + b] = (b % 2 == 0);
                mask[static_cast<size_t>(1) * num_blocks + b] = (b % 2 == 1);
            }
        }
        DevBuf dmask(mask.size());
        H2D(dmask.p, mask.data(), mask.size());
        REQUIRE(RunMinimaxMaskedAttn(
                    nullptr, dq.as<__half>(), dk.as<__half>(), dv.as<__half>(),
                    kv, heads, kv_heads, HD, scale,
                    /*pos_stride=*/kv_heads * HD,
                    /*head_stride=*/static_cast<int64_t>(HD), dmask.as<uint8_t>(),
                    num_blocks, block, /*group_heads=*/1, /*score_stride=*/kv,
                    dsc.as<float>(), dout.as<__half>()) == cudaSuccess);
        std::vector<__half> out(static_cast<size_t>(heads) * HD);
        D2H(out.data(), dout.p, sizeof(__half) * out.size());
        for (int h = 0; h < heads; ++h) {
            std::vector<float> sc(static_cast<size_t>(kv));
            float m = -INFINITY;
            for (int s = 0; s < kv; ++s) {
                if (!mask[static_cast<size_t>(h) * num_blocks + s / block]) {
                    sc[static_cast<size_t>(s)] = -INFINITY;
                    continue;
                }
                double acc = 0.0;
                for (int d = 0; d < HD; ++d)
                    acc += __half2float(q[static_cast<size_t>(h) * HD + d]) *
                           __half2float(k[static_cast<size_t>(s) * HD + d]);
                sc[static_cast<size_t>(s)] = static_cast<float>(acc) * scale;
                m = std::max(m, sc[static_cast<size_t>(s)]);
            }
            double l = 0.0;
            for (int s = 0; s < kv; ++s)
                if (sc[static_cast<size_t>(s)] > -INFINITY)
                    l += std::exp(sc[static_cast<size_t>(s)] - m);
            for (int d = 0; d < HD; ++d) {
                double acc = 0.0;
                for (int s = 0; s < kv; ++s) {
                    if (sc[static_cast<size_t>(s)] == -INFINITY) continue;
                    const double p = std::exp(sc[static_cast<size_t>(s)] - m) / l;
                    acc += p * __half2float(v[static_cast<size_t>(s) * HD + d]);
                }
                CHECK(__half2float(out[static_cast<size_t>(h) * HD + d]) ==
                      Catch::Approx(acc).epsilon(3e-3).margin(3e-3));
            }
        }
    }
}

TEST_CASE("MiniMax MoE epilogue: coeff accumulation then routed_scale + shared",
          "[minimax][kernels]")
{
    const int H = 16;
    auto a0 = RandomF32(H, 61, 1.0f);
    auto yf0 = RandomF32(H, 62, 1.0f);
    auto yf1 = RandomF32(H, 63, 1.0f);
    auto sharedf = RandomF32(H, 64, 1.0f);
    auto a0h = ToHalf(a0), y0 = ToHalf(yf0), y1 = ToHalf(yf1);
    auto shared = ToHalf(sharedf);
    DevBuf dacc(sizeof(float) * H), dy0(sizeof(__half) * H),
        dy1(sizeof(__half) * H), dshared(sizeof(__half) * H),
        dout(sizeof(__half) * H);
    H2D(dacc.p, a0h.data(), sizeof(float) * H);  // placeholder, overwritten
    std::vector<float> f(a0h.begin(), a0h.end());
    H2D(dacc.p, f.data(), sizeof(float) * H);
    H2D(dy0.p, y0.data(), sizeof(__half) * H);
    H2D(dy1.p, y1.data(), sizeof(__half) * H);
    H2D(dshared.p, shared.data(), sizeof(__half) * H);
    REQUIRE(RunMinimaxRoutedAccumulate(nullptr, dacc.as<float>(),
                                       dy0.as<__half>(), 0.25f, H) ==
            cudaSuccess);
    REQUIRE(RunMinimaxRoutedAccumulate(nullptr, dacc.as<float>(),
                                       dy1.as<__half>(), 0.75f, H) ==
            cudaSuccess);
    REQUIRE(RunMinimaxMoECombine(nullptr, dacc.as<float>(),
                                 dshared.as<__half>(), dout.as<__half>(), 1, H,
                                 2.0f) == cudaSuccess);
    std::vector<__half> out(H);
    D2H(out.data(), dout.p, sizeof(__half) * H);
    for (int i = 0; i < H; ++i) {
        const float want =
            __half2float(shared[static_cast<size_t>(i)]) +
            2.0f * (f[static_cast<size_t>(i)] +
                    0.25f * __half2float(y0[static_cast<size_t>(i)]) +
                    0.75f * __half2float(y1[static_cast<size_t>(i)]));
        CHECK(__half2float(out[static_cast<size_t>(i)]) ==
              Catch::Approx(want).epsilon(3e-3).margin(3e-3));
    }
}
