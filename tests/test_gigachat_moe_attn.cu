// TDD — GigaChat3.5 MLA decode (Plan 2b Task 2). Bit-exact interleaved YaRN
// RoPE, then full MLA-block parity against an F32 host reference. NVFP4 weights
// are built on the E2M1 grid with unit per-block scales so dequant is exact.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_device_math.h"
#include "lsxcommon/gigachat_moe_inference_attn.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace lsxcommon::gigachat;

namespace {

float R16(float v) { return __half2float(__float2half(v)); }

const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float GridVal(int i)
{
    const float m = kE2m1[static_cast<size_t>(i) % 8];
    return ((i / 8) % 2) ? -m : m;
}

uint8_t E2m1Nibble(float v)
{
    const float a = std::fabs(v);
    int code = 0;
    if (a > 5.0f) code = 7;
    else if (a > 3.5f) code = 6;
    else if (a > 2.5f) code = 5;
    else if (a > 1.75f) code = 4;
    else if (a > 1.25f) code = 3;
    else if (a > 0.75f) code = 2;
    else if (a > 0.25f) code = 1;
    return static_cast<uint8_t>(code | (v < 0.0f ? 8 : 0));
}

struct Nvfp4Host {
    std::vector<uint8_t> packed, scales;
};

Nvfp4Host PackNvfp4(const std::vector<float> &w, int N, int K)
{
    Nvfp4Host o;
    o.packed.assign(static_cast<size_t>(N) * K / 2, 0);
    o.scales.assign(static_cast<size_t>(N) * K / 16, 0x38);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; k += 2)
        {
            const uint8_t lo = E2m1Nibble(w[static_cast<size_t>(n) * K + k]);
            const uint8_t hi =
                E2m1Nibble(w[static_cast<size_t>(n) * K + k + 1]);
            o.packed[static_cast<size_t>(n) * K / 2 + k / 2] =
                static_cast<uint8_t>(lo | (hi << 4));
        }
    return o;
}

template <typename T>
T *DevCopy(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, v.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

lsxcommon::lsx_nvfp4::LsxNvfp4Weight DevNvfp4(const std::vector<float> &w, int N, int K)
{
    Nvfp4Host p = PackNvfp4(w, N, K);
    lsxcommon::lsx_nvfp4::LsxNvfp4Weight out;
    out.rows = N;
    out.cols = K;
    out.global_scale = 1.0f;
    out.packed_bytes = static_cast<int64_t>(p.packed.size());
    out.scale_bytes = static_cast<int64_t>(p.scales.size());
    out.d_packed = DevCopy(p.packed);
    out.d_scales = DevCopy(p.scales);
    return out;
}

GcGatedNorm DevNorm(const std::vector<float> &w, const std::vector<float> &up,
                    const std::vector<float> &down, int dim, int rank)
{
    std::vector<__nv_bfloat16> bw(w.size()), bu(up.size()), bd(down.size());
    for (size_t i = 0; i < w.size(); ++i) bw[i] = __float2bfloat16_rn(w[i]);
    for (size_t i = 0; i < up.size(); ++i) bu[i] = __float2bfloat16_rn(up[i]);
    for (size_t i = 0; i < down.size(); ++i) bd[i] = __float2bfloat16_rn(down[i]);
    std::vector<uint8_t> uw(w.size() * 2), uu(up.size() * 2), ud(down.size() * 2);
    std::memcpy(uw.data(), bw.data(), uw.size());
    std::memcpy(uu.data(), bu.data(), uu.size());
    std::memcpy(ud.data(), bd.data(), ud.size());
    GcGatedNorm n;
    n.dim = dim;
    n.rank = rank;
    n.d_weight = DevCopy(uw);
    n.d_gate_up = DevCopy(uu);
    n.d_gate_down = DevCopy(ud);
    return n;
}

std::vector<float> GemvRef(const std::vector<float> &W,
                           const std::vector<float> &x, int N, int K)
{
    std::vector<float> y(static_cast<size_t>(N), 0.0f);
    for (int n = 0; n < N; ++n)
    {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += W[static_cast<size_t>(n) * K + k] * x[static_cast<size_t>(k)];
        y[static_cast<size_t>(n)] = acc;
    }
    return y;
}

std::vector<float> NormRef(const std::vector<float> &x,
                           const std::vector<float> &w,
                           const std::vector<float> &up,
                           const std::vector<float> &down, int dim, int rank,
                           float eps)
{
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    std::vector<float> xn(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) xn[i] = x[i] * inv;
    std::vector<float> gu(static_cast<size_t>(rank), 0.0f);
    for (int r = 0; r < rank; ++r)
    {
        float a = 0.0f;
        for (int i = 0; i < dim; ++i)
            a += up[static_cast<size_t>(r) * dim + i] * xn[i];
        gu[r] = a / (1.0f + std::exp(-a));
    }
    std::vector<float> g(static_cast<size_t>(dim), 0.0f);
    for (int i = 0; i < dim; ++i)
    {
        float a = 0.0f;
        for (int r = 0; r < rank; ++r)
            a += down[static_cast<size_t>(i) * rank + r] * gu[r];
        g[i] = a;
    }
    std::vector<float> out(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
        out[i] = xn[i] * (1.0f + w[i]) * 2.0f / (1.0f + std::exp(-g[i]));
    return out;
}

void RopeRef(std::vector<float> &q, int heads, int qk, int nope,
             std::vector<float> &k, const std::vector<float> &c,
             const std::vector<float> &s)
{
    const int half = static_cast<int>(c.size());
    for (int h = 0; h < heads; ++h)
        for (int j = 0; j < half; ++j)
        {
            const int i0 = h * qk + nope + 2 * j;
            const float a = q[i0];
            const float b = q[i0 + 1];
            q[i0] = a * c[j] - b * s[j];
            q[i0 + 1] = a * s[j] + b * c[j];
        }
    for (int j = 0; j < half; ++j)
    {
        const float a = k[2 * j];
        const float b = k[2 * j + 1];
        k[2 * j] = a * c[j] - b * s[j];
        k[2 * j + 1] = a * s[j] + b * c[j];
    }
}

} // namespace

TEST_CASE("gigachat MLA interleaved rope is bit-exact", "[gigachat][attn]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    const int heads = 3, qk = 16, nope = 8, rope = 8, half = rope / 2;
    std::vector<float> q(static_cast<size_t>(heads) * qk), k(rope);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = 0.125f * static_cast<float>(static_cast<int>(i % 23) - 11);
    for (int i = 0; i < rope; ++i) k[i] = 0.2f * static_cast<float>(i - 3);
    std::vector<float> c(half), s(half);
    for (int j = 0; j < half; ++j)
    {
        c[j] = std::cos(0.37f * j + 0.1f);
        s[j] = std::sin(0.37f * j + 0.1f);
    }
    std::vector<float> qr = q, kr = k;
    RopeRef(qr, heads, qk, nope, kr, c, s);

    float *dq = DevCopy(q);
    float *dk = DevCopy(k);
    float *dc = DevCopy(c);
    float *ds = DevCopy(s);
    REQUIRE(RunGigaChatRopeQk(dq, dk, heads, qk, nope, rope, dc, ds, nullptr) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> hq(q.size()), hk(k.size());
    REQUIRE(cudaMemcpy(hq.data(), dq, hq.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(hk.data(), dk, hk.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (size_t i = 0; i < hq.size(); ++i) CHECK(hq[i] == qr[i]);
    for (size_t i = 0; i < hk.size(); ++i) CHECK(hk[i] == kr[i]);
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dc);
    cudaFree(ds);
}

TEST_CASE("gigachat MLA decode matches the host reference",
          "[gigachat][attn]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);

    const int hidden = 32, heads = 2, q_lora = 8, kv_lora = 8, nope = 8,
              rope = 8, vh = 8;
    const int qk = nope + rope, kvw = kv_lora + rope;
    const int kvb = heads * (nope + vh), hv = heads * vh;
    const GcMlaDims d{hidden, heads, q_lora, kv_lora, nope, rope, vh};

    auto gen = [](int n, int seed) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) v[i] = GridVal(i * 5 + seed);
        return v;
    };
    // Weights [N, K].
    std::vector<float> w_qa = gen(q_lora * hidden, 1);
    std::vector<float> w_qb = gen(heads * qk * q_lora, 2);
    std::vector<float> w_kva = gen(kvw * hidden, 3);
    std::vector<float> w_kvb = gen(kvb * kv_lora, 4);
    std::vector<float> w_gate = gen(hv * hidden, 5);
    std::vector<float> w_o = gen(hidden * hv, 6);
    // Norm triples: gate_up [rank, dim], gate_down [dim, rank].
    const int rank = 2;
    std::vector<float> qn_w = gen(q_lora, 7);
    std::vector<float> qn_up = gen(rank * q_lora, 8);
    std::vector<float> qn_down = gen(q_lora * rank, 9);
    std::vector<float> kn_w = gen(kv_lora, 10);
    std::vector<float> kn_up = gen(rank * kv_lora, 11);
    std::vector<float> kn_down = gen(kv_lora * rank, 12);

    GcLayerWeights lw;
    lw.q_a = DevNvfp4(w_qa, q_lora, hidden);
    lw.q_b = DevNvfp4(w_qb, heads * qk, q_lora);
    lw.kv_a = DevNvfp4(w_kva, kvw, hidden);
    lw.kv_b = DevNvfp4(w_kvb, kvb, kv_lora);
    lw.attn_gate = DevNvfp4(w_gate, hv, hidden);
    lw.o_proj = DevNvfp4(w_o, hidden, hv);
    lw.q_a_norm = DevNorm(qn_w, qn_up, qn_down, q_lora, rank);
    lw.kv_a_norm = DevNorm(kn_w, kn_up, kn_down, kv_lora, rank);

    const int capacity = 4;
    const int half = rope / 2;
    std::vector<float> rc(static_cast<size_t>(capacity) * half),
        rs(static_cast<size_t>(capacity) * half);
    for (int p = 0; p < capacity; ++p)
        for (int j = 0; j < half; ++j)
        {
            rc[static_cast<size_t>(p) * half + j] = std::cos(0.23f * (p + 1) + j);
            rs[static_cast<size_t>(p) * half + j] = std::sin(0.23f * (p + 1) + j);
        }
    std::vector<float> x0(static_cast<size_t>(hidden)),
        x1(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
    {
        x0[i] = 0.25f * static_cast<float>((i % 7) - 3);
        x1[i] = 0.2f * static_cast<float>((i % 5) - 2);
    }

    const float eps = 1e-6f, scale = 0.55f;
    float *kx = DevCopy(x0), *vx = DevCopy(x1);
    GcMlaScratch sc;
    REQUIRE(sc.Alloc(d, capacity));
    float *k_cache = nullptr, *v_cache = nullptr;
    REQUIRE(cudaMalloc(&k_cache, static_cast<size_t>(capacity) * heads * qk * 4) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&v_cache, static_cast<size_t>(capacity) * heads * vh * 4) ==
            cudaSuccess);
    float *d_rc = DevCopy(rc), *d_rs = DevCopy(rs);
    float *out0 = nullptr, *out1 = nullptr;
    REQUIRE(cudaMalloc(&out0, hidden * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&out1, hidden * 4) == cudaSuccess);
    REQUIRE(RunGigaChatMlaDecode(kx, lw, d, eps, scale, d_rc, d_rs, 0, capacity,
                                 k_cache, v_cache, sc, out0, nullptr) ==
            cudaSuccess);
    REQUIRE(RunGigaChatMlaDecode(vx, lw, d, eps, scale, d_rc, d_rs, 1, capacity,
                                 k_cache, v_cache, sc, out1, nullptr) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> hout(static_cast<size_t>(hidden));
    REQUIRE(cudaMemcpy(hout.data(), out1, hidden * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // ── Host reference. Mirrors each F16 rounding the device applies. ──
    auto token = [&](const std::vector<float> &x, int p,
                     std::vector<float> &q, std::vector<float> &k,
                     std::vector<float> &v, std::vector<float> &gate) {
        std::vector<float> x16(static_cast<size_t>(hidden));
        for (int i = 0; i < hidden; ++i) x16[i] = R16(x[i]);
        std::vector<float> qa = GemvRef(w_qa, x16, q_lora, hidden);
        for (float &z : qa) z = R16(z);
        std::vector<float> qan = NormRef(qa, qn_w, qn_up, qn_down, q_lora, rank,
                                         eps);
        const float aq = std::sqrt(static_cast<float>(hidden) / q_lora);
        std::vector<float> qas(static_cast<size_t>(q_lora));
        for (int i = 0; i < q_lora; ++i) qas[i] = R16(qan[i] * aq);
        std::vector<float> qb = GemvRef(w_qb, qas, heads * qk, q_lora);
        for (float &z : qb) z = R16(z);

        std::vector<float> kv = GemvRef(w_kva, x16, kvw, hidden);
        for (float &z : kv) z = R16(z);
        std::vector<float> kp(kv.begin(), kv.begin() + kv_lora);
        std::vector<float> krot(kv.begin() + kv_lora, kv.end());
        std::vector<float> kpn = NormRef(kp, kn_w, kn_up, kn_down, kv_lora,
                                         rank, eps);
        const float akv = std::sqrt(static_cast<float>(hidden) / kv_lora);
        std::vector<float> kps(static_cast<size_t>(kv_lora));
        for (int i = 0; i < kv_lora; ++i) kps[i] = R16(kpn[i] * akv);
        std::vector<float> kvb_f = GemvRef(w_kvb, kps, kvb, kv_lora);
        for (float &z : kvb_f) z = R16(z);

        const size_t ro = static_cast<size_t>(p) * half;
        std::vector<float> c(half), s(half);
        for (int j = 0; j < half; ++j)
        {
            c[j] = rc[ro + j];
            s[j] = rs[ro + j];
        }
        RopeRef(qb, heads, qk, nope, krot, c, s);
        q = qb;
        k.assign(static_cast<size_t>(heads) * qk, 0.0f);
        v.assign(static_cast<size_t>(heads) * vh, 0.0f);
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < nope; ++i)
            {
                k[static_cast<size_t>(h) * qk + i] =
                    kvb_f[static_cast<size_t>(h) * (nope + vh) + i];
                v[static_cast<size_t>(h) * vh + i] =
                    kvb_f[static_cast<size_t>(h) * (nope + vh) + nope + i];
            }
            for (int i = 0; i < rope; ++i)
                k[static_cast<size_t>(h) * qk + nope + i] = krot[i];
        }
        gate = GemvRef(w_gate, x16, hv, hidden);
        for (float &z : gate) z = R16(z);
    };

    std::vector<float> q0, k0, v0, g0, q1, k1, v1, g1;
    token(x0, 0, q0, k0, v0, g0);
    token(x1, 1, q1, k1, v1, g1);

    std::vector<float> ref(static_cast<size_t>(hidden), 0.0f);
    std::vector<float> attn(static_cast<size_t>(hv), 0.0f);
    for (int h = 0; h < heads; ++h)
    {
        float sc[2];
        for (int t = 0; t < 2; ++t)
        {
            const std::vector<float> &kt = (t == 0) ? k0 : k1;
            float a = 0.0f;
            for (int i = 0; i < qk; ++i)
                a += q1[static_cast<size_t>(h) * qk + i] *
                     kt[static_cast<size_t>(h) * qk + i];
            sc[t] = a * scale;
        }
        const float m = std::fmax(sc[0], sc[1]);
        const float e0 = std::exp(sc[0] - m), e1 = std::exp(sc[1] - m);
        const float p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
        for (int i = 0; i < vh; ++i)
        {
            const float o = p0 * v0[static_cast<size_t>(h) * vh + i] +
                            p1 * v1[static_cast<size_t>(h) * vh + i];
            const float g = g1[static_cast<size_t>(h) * vh + i];
            attn[static_cast<size_t>(h) * vh + i] =
                R16(o / (1.0f + std::exp(-g)));
        }
    }
    ref = GemvRef(w_o, attn, hidden, hv);
    for (float &z : ref) z = R16(z);

    for (int i = 0; i < hidden; ++i)
        CHECK(hout[static_cast<size_t>(i)] ==
              Catch::Approx(ref[static_cast<size_t>(i)]).margin(1e-3f));

    sc.Free();
    cudaFree(k_cache);
    cudaFree(v_cache);
    cudaFree(out0);
    cudaFree(out1);
    cudaFree(kx);
    cudaFree(vx);
    cudaFree(d_rc);
    cudaFree(d_rs);
}
