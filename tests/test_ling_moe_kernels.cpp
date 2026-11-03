// Ling-3.1-flash kernel stage: every device kernel against a host reference
// (label `fast`, GPU resource lock). Shapes are tiny; the point is the math
// contract, not throughput.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/ling_moe_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::ling;

namespace {

bool HasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct Dev {
    template <typename T>
    T *alloc(const std::vector<T> &h)
    {
        T *d = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d),
                           h.size() * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        return d;
    }
    template <typename T>
    T *alloc(size_t n)
    {
        T *d = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d), n * sizeof(T)) ==
                cudaSuccess);
        REQUIRE(cudaMemset(d, 0, n * sizeof(T)) == cudaSuccess);
        return d;
    }
    template <typename T>
    std::vector<T> get(const T *d, size_t n)
    {
        std::vector<T> h(n);
        REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(T),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        return h;
    }
    void free(void *p)
    {
        if (p) cudaFree(p);
    }
};

std::vector<__half> H(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

std::vector<float> F(const std::vector<__half> &v)
{
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = __half2float(v[i]);
    return f;
}

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

TEST_CASE("Ling RMSNorm matches the host reference", "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int Hd = 16, rows = 2;
    std::vector<float> x(rows * Hd), w(Hd);
    for (int i = 0; i < rows * Hd; ++i) x[i] = 0.1f * (i % 11) - 0.3f;
    for (int i = 0; i < Hd; ++i) w[i] = 0.5f + 0.1f * i;
    Dev d;
    auto *dx = d.alloc(H(x));
    auto *dw = d.alloc(H(w));
    auto *dy = d.alloc<__half>(rows * Hd);
    REQUIRE(RunLingRmsNorm(nullptr, dx, dw, dy, rows, Hd, 1e-6f) ==
            cudaSuccess);
    auto got = F(d.get(dy, rows * Hd));
    for (int r = 0; r < rows; ++r) {
        float ss = 0;
        for (int i = 0; i < Hd; ++i) ss += x[r * Hd + i] * x[r * Hd + i];
        const float inv = 1.0f / std::sqrt(ss / Hd + 1e-6f);
        for (int i = 0; i < Hd; ++i)
            CHECK(got[r * Hd + i] ==
                  Catch::Approx(x[r * Hd + i] * inv * w[i]).margin(4e-3));
    }
    d.free(dx);
    d.free(dw);
    d.free(dy);
}

TEST_CASE("Ling conv decode applies silu and shifts history",
          "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int C = 4, K = 3, taps = K - 1;
    std::vector<float> hist(taps * C), w(C * K), cur(C);
    for (int i = 0; i < taps * C; ++i) hist[i] = 0.05f * (i + 1);
    for (int i = 0; i < C * K; ++i) w[i] = 0.1f * ((i % 5) - 2);
    for (int i = 0; i < C; ++i) cur[i] = 0.2f * (i + 1);
    Dev d;
    auto *dh = d.alloc(H(hist));
    auto *dc = d.alloc(H(cur));
    auto *dw = d.alloc(H(w));
    auto *dy = d.alloc<__half>(C);
    REQUIRE(RunLingConvDecodeSilu(nullptr, dh, dc, dw, dy, C, K, true) ==
            cudaSuccess);
    auto got = F(d.get(dy, C));
    for (int c = 0; c < C; ++c) {
        float acc = 0;
        for (int i = 0; i < taps; ++i) acc += hist[i * C + c] * w[c * K + i];
        acc += cur[c] * w[c * K + taps];
        const float want = acc * sigmoidf(acc);
        CHECK(got[c] == Catch::Approx(want).margin(4e-3));
    }
    auto shifted = F(d.get(dh, taps * C));
    for (int c = 0; c < C; ++c) {
        CHECK(shifted[0 * C + c] == Catch::Approx(hist[1 * C + c]).margin(4e-3));
        CHECK(shifted[1 * C + c] == Catch::Approx(cur[c]).margin(4e-3));
    }
    d.free(dh);
    d.free(dc);
    d.free(dw);
    d.free(dy);
}

TEST_CASE("Ling KDA gating and delta step match the host reference",
          "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int NH = 2, D = 4;
    std::vector<float> b(NH), f(NH * D), alog(NH), dtb(NH * D), q(NH * D),
        k(NH * D), v(NH * D), state(NH * D * D);
    for (int i = 0; i < NH; ++i) {
        b[i] = 0.3f * (i + 1);
        alog[i] = 0.2f * (i + 1);
    }
    for (int i = 0; i < NH * D; ++i) {
        f[i] = 0.05f * (i % 7) - 0.1f;
        dtb[i] = 0.01f * i;
        q[i] = 0.1f * (i % 5) - 0.2f;
        k[i] = 0.15f * (i % 3) - 0.1f;
        v[i] = 0.2f * (i % 4) - 0.3f;
    }
    for (int i = 0; i < NH * D * D; ++i)
        state[i] = 0.01f * ((i % 9) - 4);
    Dev d;
    auto *db = d.alloc(H(b));
    auto *df = d.alloc(H(f));
    auto *da = d.alloc(alog);
    auto *dd = d.alloc(dtb);
    auto *dq = d.alloc(H(q));
    auto *dk = d.alloc(H(k));
    auto *dv = d.alloc(H(v));
    auto *dbeta = d.alloc<float>(NH);
    auto *ddecay = d.alloc<float>(NH * D);
    auto *dstate = d.alloc(state);
    auto *dout = d.alloc<__half>(NH * D);
    REQUIRE(RunLingKdaGating(nullptr, db, df, da, dd, dbeta, ddecay, NH, D,
                             -5.0f) == cudaSuccess);
    REQUIRE(RunLingKdaStep(nullptr, dq, dk, dv, dbeta, ddecay, dstate, dout,
                           NH, D) == cudaSuccess);

    // Host reference (FP32, same order).
    std::vector<float> hb(NH), hdecay(NH * D), hstate = state, hout(NH * D);
    for (int i = 0; i < NH; ++i) hb[i] = sigmoidf(b[i]);
    for (int i = 0; i < NH * D; ++i) {
        const int h = i / D;
        const float g = f[i] + dtb[i];
        hdecay[i] = std::exp(-5.0f * sigmoidf(std::exp(alog[h]) * g));
    }
    for (int h = 0; h < NH; ++h)
        for (int dd = 0; dd < D; ++dd) {
            float col[D];
            for (int l = 0; l < D; ++l)
                col[l] = hstate[(h * D + l) * D + dd] * hdecay[h * D + l];
            float kv = 0;
            for (int l = 0; l < D; ++l) kv += col[l] * k[h * D + l];
            const float delta = hb[h] * (v[h * D + dd] - kv);
            for (int l = 0; l < D; ++l) col[l] += k[h * D + l] * delta;
            float acc = 0;
            for (int l = 0; l < D; ++l) acc += col[l] * q[h * D + l];
            hout[h * D + dd] = acc;
            for (int l = 0; l < D; ++l)
                hstate[(h * D + l) * D + dd] = col[l];
        }
    auto got_beta = d.get(dbeta, NH);
    auto got_decay = d.get(ddecay, NH * D);
    auto got_out = F(d.get(dout, NH * D));
    auto got_state = d.get(dstate, NH * D * D);
    for (int i = 0; i < NH; ++i)
        CHECK(got_beta[i] == Catch::Approx(hb[i]).margin(1e-4));
    for (int i = 0; i < NH * D; ++i) {
        CHECK(got_decay[i] == Catch::Approx(hdecay[i]).margin(1e-5));
        CHECK(got_out[i] == Catch::Approx(hout[i]).margin(2e-3));
    }
    for (int i = 0; i < NH * D * D; ++i)
        CHECK(got_state[i] == Catch::Approx(hstate[i]).margin(2e-3));
    d.free(db);
    d.free(df);
    d.free(da);
    d.free(dd);
    d.free(dq);
    d.free(dk);
    d.free(dv);
    d.free(dbeta);
    d.free(ddecay);
    d.free(dstate);
    d.free(dout);
}

TEST_CASE("Ling L2 norm and gated RMSNorm", "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int NH = 2, D = 4;
    std::vector<float> q(NH * D), k(NH * D), core(NH * D), gate(NH * D),
        w(D);
    for (int i = 0; i < NH * D; ++i) {
        q[i] = 0.1f * (i + 1);
        k[i] = 0.2f * (i % 3 + 1);
        core[i] = 0.3f * (i % 5) - 0.2f;
        gate[i] = 0.4f * (i % 4) - 0.3f;
    }
    for (int i = 0; i < D; ++i) w[i] = 0.7f + 0.1f * i;
    Dev d;
    auto *dq = d.alloc(H(q));
    auto *dk = d.alloc(H(k));
    const float qs = 1.0f / std::sqrt(static_cast<float>(D));
    REQUIRE(RunLingL2NormQK(nullptr, dq, dk, NH, D, qs) == cudaSuccess);
    auto gq = F(d.get(dq, NH * D));
    auto gk = F(d.get(dk, NH * D));
    for (int h = 0; h < NH; ++h) {
        float sq = 0, sk = 0;
        for (int i = 0; i < D; ++i) {
            sq += q[h * D + i] * q[h * D + i];
            sk += k[h * D + i] * k[h * D + i];
        }
        for (int i = 0; i < D; ++i) {
            CHECK(gq[h * D + i] ==
                  Catch::Approx(q[h * D + i] / std::sqrt(sq + 1e-6f) * qs)
                      .margin(4e-3));
            CHECK(gk[h * D + i] ==
                  Catch::Approx(k[h * D + i] / std::sqrt(sk + 1e-6f))
                      .margin(4e-3));
        }
    }
    auto *dc = d.alloc(H(core));
    auto *dg = d.alloc(H(gate));
    auto *dw = d.alloc(H(w));
    auto *do_ = d.alloc<__half>(NH * D);
    REQUIRE(RunLingRmsNormGated(nullptr, dc, dg, dw, do_, NH, D, 1e-6f) ==
            cudaSuccess);
    auto go = F(d.get(do_, NH * D));
    for (int h = 0; h < NH; ++h) {
        float ss = 0;
        for (int i = 0; i < D; ++i) ss += core[h * D + i] * core[h * D + i];
        const float inv = 1.0f / std::sqrt(ss / D + 1e-6f);
        for (int i = 0; i < D; ++i) {
            const float want = core[h * D + i] * inv * w[i] *
                               sigmoidf(gate[h * D + i]);
            CHECK(go[h * D + i] == Catch::Approx(want).margin(2e-3));
        }
    }
    d.free(dq);
    d.free(dk);
    d.free(dc);
    d.free(dg);
    d.free(dw);
    d.free(do_);
}

TEST_CASE("Ling interleaved RoPE rotates adjacent pairs", "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int NH = 2, HD = 6, ROPE = 4;
    std::vector<float> q(NH * HD, 0.25f), k(ROPE, 0.5f);
    Dev d;
    auto *dq = d.alloc(H(q));
    auto *dk = d.alloc(H(k));
    REQUIRE(RunLingRopeInterleaved(nullptr, dq, dk, NH, HD, ROPE, 3, 10000.0) ==
            cudaSuccess);
    auto gq = F(d.get(dq, NH * HD));
    auto gk = F(d.get(dk, ROPE));
    // First pair of head 0 uses freq theta^0 = 1, angle = 3 rad.
    const float c = std::cos(3.0f), s = std::sin(3.0f);
    CHECK(gq[HD - ROPE + 0] ==
          Catch::Approx(0.25f * c - 0.25f * s).margin(2e-3));
    CHECK(gq[HD - ROPE + 1] ==
          Catch::Approx(0.25f * c + 0.25f * s).margin(2e-3));
    // Pass-through prefix untouched.
    CHECK(gq[0] == Catch::Approx(0.25f).margin(2e-3));
    CHECK(gk[0] == Catch::Approx(0.5f * c - 0.5f * s).margin(2e-3));
    CHECK(gk[1] == Catch::Approx(0.5f * c + 0.5f * s).margin(2e-3));
    d.free(dq);
    d.free(dk);
}

TEST_CASE("Ling group-limited router selects by bias and renormalizes",
          "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int E = 16, Hd = 8, K = 4, G = 4, TG = 2;
    std::vector<float> gate_f(E * Hd), x(Hd), bias(E);
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < Hd; ++i)
            gate_f[e * Hd + i] =
                0.05f * std::sin(0.7f * e + 0.3f * i) + 0.01f * e;
    for (int i = 0; i < Hd; ++i) x[i] = 0.2f * std::cos(0.5f * i);
    for (int e = 0; e < E; ++e) bias[e] = 0.0f;
    // Push group 1 (experts 4..7) above the rest.
    for (int e = 4; e < 8; ++e) bias[e] = 0.5f;

    // Gate is stored BF16 in the engine; feed the same bits back.
    std::vector<uint16_t> gate_bf(E * Hd);
    for (size_t i = 0; i < gate_f.size(); ++i) {
        const float v = gate_f[i];
        uint32_t u;
        std::memcpy(&u, &v, 4);
        gate_bf[i] = static_cast<uint16_t>(u >> 16);
    }
    std::vector<__half> gate_h(gate_bf.size());
    for (size_t i = 0; i < gate_bf.size(); ++i)
        gate_h[i] = __ushort_as_half(gate_bf[i]);
    Dev d;
    auto *dg = d.alloc(gate_h);
    auto *db = d.alloc(bias);
    auto *dx = d.alloc(H(x));
    auto *ds = d.alloc<lsxcommon::moe::MoERoutedSlot>(K);
    REQUIRE(RunLingRouterTopK(nullptr, reinterpret_cast<const uint8_t *>(dg), db, dx, E, K, G, TG, Hd, 2.5f,
                             ds) == cudaSuccess);
    auto slots = d.get(ds, K);
    // Host reference over the same BF16-decoded gate.
    std::vector<float> sc(E), ch(E);
    for (int e = 0; e < E; ++e) {
        float dot = 0;
        for (int i = 0; i < Hd; ++i) {
            uint32_t u = static_cast<uint32_t>(gate_bf[e * Hd + i]) << 16;
            float gv;
            std::memcpy(&gv, &u, 4);
            dot += gv * x[i];
        }
        sc[e] = sigmoidf(dot);
        ch[e] = sc[e] + bias[e];
    }
    const int gs = E / G;
    float gsc[G];
    for (int g = 0; g < G; ++g) {
        float b0 = -1e30f, b1 = -1e30f;
        for (int j = 0; j < gs; ++j) {
            const float v = ch[g * gs + j];
            if (v > b0) {
                b1 = b0;
                b0 = v;
            } else if (v > b1) {
                b1 = v;
            }
        }
        gsc[g] = b0 + b1;
    }
    bool sel[G] = {};
    for (int kk = 0; kk < TG; ++kk) {
        int best = -1;
        float bv = -1e30f;
        for (int g = 0; g < G; ++g)
            if (!sel[g] && gsc[g] > bv) {
                bv = gsc[g];
                best = g;
            }
        sel[best] = true;
    }
    int ce[K];
    float cw[K];
    for (int kk = 0; kk < K; ++kk) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; ++e) {
            if (!sel[e / gs]) continue;
            bool taken = false;
            for (int j = 0; j < kk; ++j)
                if (ce[j] == e) taken = true;
            if (taken) continue;
            if (ch[e] > bv) {
                bv = ch[e];
                best = e;
            }
        }
        ce[kk] = best;
        cw[kk] = sc[best];
    }
    float sum = 0;
    for (int kk = 0; kk < K; ++kk) sum += cw[kk];
    for (int kk = 0; kk < K; ++kk) {
        CHECK(slots[kk].eid == ce[kk]);
        CHECK(slots[kk].coeff ==
              Catch::Approx(cw[kk] / sum * 2.5f).margin(1e-4));
    }
    d.free(dg);
    d.free(db);
    d.free(dx);
    d.free(ds);
}

TEST_CASE("Ling MLA attention matches softmax over the cache",
          "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int NH = 2, KD = 6, VD = 4, POS = 3, CAP = 4;
    const int HD = KD + VD;
    const auto geom = lsxcommon::KvCacheGeometry::Make(
        lsxcommon::KvCacheLayout::kPosMajor, CAP, NH, HD, 1);
    std::vector<float> q(NH * KD), kv(geom.Elements());
    for (int h = 0; h < NH; ++h)
        for (int i = 0; i < KD; ++i)
            q[h * KD + i] = 0.1f * ((h + 1) * (i % 3) - 0.2f);
    for (int h = 0; h < NH; ++h)
        for (int p = 0; p < POS; ++p)
            for (int i = 0; i < HD; ++i)
                kv[geom.Offset(0, h, p, i)] =
                    0.05f * ((h * 7 + p * 3 + i) % 5) - 0.1f;
    Dev d;
    auto *dq = d.alloc(H(q));
    auto *dkv = d.alloc(H(kv));
    auto *dsc = d.alloc<float>(static_cast<size_t>(NH) * CAP);
    auto *dout = d.alloc<__half>(NH * VD);
    const float scale = 1.0f / std::sqrt(static_cast<float>(KD));
    REQUIRE(RunLingMlaAttention(nullptr, dq, dkv, dsc, geom, 0, POS, NH, KD, VD,
                               scale, dout) == cudaSuccess);
    auto go = F(d.get(dout, NH * VD));
    for (int h = 0; h < NH; ++h) {
        std::vector<float> s(POS), p(POS);
        float m = -1e30f;
        for (int pos = 0; pos < POS; ++pos) {
            float dot = 0;
            for (int i = 0; i < KD; ++i)
                dot += q[h * KD + i] * kv[geom.Offset(0, h, pos, i)];
            s[pos] = dot * scale;
            m = std::max(m, s[pos]);
        }
        float sum = 0;
        for (int pos = 0; pos < POS; ++pos) {
            p[pos] = std::exp(s[pos] - m);
            sum += p[pos];
        }
        for (int dd = 0; dd < VD; ++dd) {
            float acc = 0;
            for (int pos = 0; pos < POS; ++pos)
                acc += p[pos] / sum *
                       kv[geom.Offset(0, h, pos, KD + dd)];
            CHECK(go[h * VD + dd] == Catch::Approx(acc).margin(3e-3));
        }
    }
    d.free(dq);
    d.free(dkv);
    d.free(dsc);
    d.free(dout);
}

TEST_CASE("Ling SwiGLU, embed and combine epilogues", "[ling-moe-kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int n = 8, V = 3, Hd = 4;
    std::vector<float> g(n), u(n), tbl(V * Hd), resid(Hd), shared(Hd),
        routed(Hd);
    for (int i = 0; i < n; ++i) {
        g[i] = 0.2f * (i % 5) - 0.3f;
        u[i] = 0.1f * (i % 3) - 0.1f;
    }
    for (int i = 0; i < V * Hd; ++i) tbl[i] = 0.01f * i;
    for (int i = 0; i < Hd; ++i) {
        resid[i] = 0.1f * i;
        shared[i] = -0.05f * i;
        routed[i] = 0.3f * i;
    }
    Dev d;
    auto *dg = d.alloc(H(g));
    auto *du = d.alloc(H(u));
    auto *do_ = d.alloc<__half>(n);
    REQUIRE(RunLingSwiGLU(nullptr, dg, du, do_, n) == cudaSuccess);
    auto go = F(d.get(do_, n));
    for (int i = 0; i < n; ++i)
        CHECK(go[i] == Catch::Approx(g[i] * sigmoidf(g[i]) * u[i])
                            .margin(2e-3));
    auto *dt = d.alloc(H(tbl));
    auto *de = d.alloc<__half>(Hd);
    REQUIRE(RunLingEmbedToken(nullptr, dt, 2, de, Hd) == cudaSuccess);
    auto ge = F(d.get(de, Hd));
    for (int i = 0; i < Hd; ++i)
        CHECK(ge[i] == Catch::Approx(tbl[2 * Hd + i]).margin(4e-3));
    auto *dr = d.alloc(H(resid));
    auto *dsh = d.alloc(H(shared));
    auto *drt = d.alloc(routed);
    auto *dco = d.alloc<__half>(Hd);
    REQUIRE(RunLingAddRoutedShared(nullptr, dr, dsh, drt, dco, Hd) ==
            cudaSuccess);
    auto gc = F(d.get(dco, Hd));
    for (int i = 0; i < Hd; ++i)
        CHECK(gc[i] ==
              Catch::Approx(resid[i] + shared[i] + routed[i]).margin(2e-3));
    d.free(dg);
    d.free(du);
    d.free(do_);
    d.free(dt);
    d.free(de);
    d.free(dr);
    d.free(dsh);
    d.free(drt);
    d.free(dco);
}
