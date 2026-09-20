#pragma once

// Host F32 reference for the GigaChat3.5 whole-stack synthetic parity test.
// Pure host (std:: only; included from .cu translation units, so no abseil
// containers). Shares the NVFP4 codec with the device so dequant is exact; the
// arithmetic is independent of the device kernels and mirrors every F16/BF16
// rounding the pipeline applies.
#include "gigachat_moe_constants.h"
#include "gigachat_moe_device_math.h"
#include "nvfp4_codec.h"
#include "lsxcommon/gigachat_moe_inference_attn.h"
#include "lsxcommon/schema.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace gigachatref {

using lsxcommon::DenseNvfp4BatchViews;
using lsxcommon::DenseNvfp4View;
using lsxcommon::TensorBatchViews;
using lsxcommon::TensorView;
using GcGdnDims = lsxcommon::gigachat::GcGdnDims;
using GcMlaDims = lsxcommon::gigachat::GcMlaDims;

// ── F16 / BF16 rounding (mirrors the device conversions) ───────────────────
inline float R16(float v) { return __half2float(__float2half(v)); }
inline float RBf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// ── F32 primitives ─────────────────────────────────────────────────────────
inline std::vector<float> GemvRef(const std::vector<float> &W,
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

// ZeroCenteredGatedNorm: xn = x * rsqrt(mean(x^2)+eps);
// gate = gate_down @ silu(gate_up @ xn); out = xn*(1+w)*2*sigmoid(gate).
inline std::vector<float> NormRef(const std::vector<float> &x,
                                  const std::vector<float> &w,
                                  const std::vector<float> &up,
                                  const std::vector<float> &down, int dim,
                                  int rank, float eps)
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

// Interleaved RoPE over adjacent pairs: q [heads, qk] at offset `nope`, k [rope].
inline void RopeRef(std::vector<float> &q, int heads, int qk, int nope,
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

// ── Arrow readers ──────────────────────────────────────────────────────────
inline const TensorView *FindTv(const TensorBatchViews &b,
                                std::string_view suffix)
{
    const TensorView *r = nullptr;
    for (const auto &t : b.tensors)
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            r = &t;
    return r;
}

inline const DenseNvfp4View *FindNv(const DenseNvfp4BatchViews &b,
                                    std::string_view suffix)
{
    const DenseNvfp4View *r = nullptr;
    for (const auto &t : b.records)
        if (t.name.size() >= suffix.size() &&
            t.name.compare(t.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0)
            r = &t;
    return r;
}

inline std::vector<float> Bf16ToF32(const uint8_t *p, int64_t n)
{
    std::vector<float> o(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        __nv_bfloat16 b;
        std::memcpy(&b, p + 2 * i, sizeof(b));
        o[static_cast<size_t>(i)] = __bfloat162float(b);
    }
    return o;
}

inline std::vector<float> F32Copy(const uint8_t *p, int64_t n)
{
    std::vector<float> o(static_cast<size_t>(n));
    std::memcpy(o.data(), p, static_cast<size_t>(n) * sizeof(float));
    return o;
}

// Reads a small tensor that may be either F32 or BF16 (e.g. the router
// correction bias, which the real checkpoint stores as BF16).
inline std::vector<float> ReadF32OrBf16(const TensorView &t)
{
    int64_t n = 1;
    for (int64_t d = 0; d < t.rank; ++d) n *= t.shape[d];
    if (t.dtype == "F32") return F32Copy(t.data, n);
    return Bf16ToF32(t.data, n);
}

inline std::vector<float> DequantNvfp4(const uint8_t *packed,
                                       const uint8_t *scales, float gs,
                                       int64_t n)
{
    std::vector<float> w(static_cast<size_t>(n));
    for (int64_t e = 0; e < n; ++e)
    {
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0x0F);
        const float scl =
            lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(scales[e >> 4]) * gs;
        w[static_cast<size_t>(e)] = lsxcommon::gpu::nvfp4::E2M1Value(nib) * scl;
    }
    return w;
}

struct RefNorm {
    int dim = 0;
    int rank = 0;
    std::vector<float> w, up, down;
};

struct RefNvfp4 {
    int rows = 0;
    int cols = 0;
    std::vector<float> w;  // [rows*cols] dequantized F32
};

inline RefNorm ReadNorm(const TensorBatchViews &raw, const std::string &prefix)
{
    const TensorView *w = FindTv(raw, prefix + ".weight");
    const TensorView *gu = FindTv(raw, prefix + ".gate_up_projection.weight");
    const TensorView *gd = FindTv(raw, prefix + ".gate_down_projection.weight");
    RefNorm n;
    n.dim = static_cast<int>(w->shape[0]);
    n.rank = static_cast<int>(gu->shape[0]);
    n.w = Bf16ToF32(w->data, n.dim);
    n.up = Bf16ToF32(gu->data, static_cast<int64_t>(n.rank) * n.dim);
    n.down = Bf16ToF32(gd->data, static_cast<int64_t>(n.dim) * n.rank);
    return n;
}

inline RefNvfp4 ReadNv(const DenseNvfp4BatchViews &nv, std::string_view suffix)
{
    const DenseNvfp4View *r = FindNv(nv, suffix);
    RefNvfp4 o;
    o.rows = static_cast<int>(r->shape[0]);
    o.cols = static_cast<int>(r->shape[1]);
    o.w = DequantNvfp4(r->packed, r->scales, r->global_scale,
                       static_cast<int64_t>(o.rows) * o.cols);
    return o;
}

struct RefMla {
    RefNvfp4 q_a, q_b, kv_a, kv_b, gate, o_proj;
    RefNorm q_a_norm, kv_a_norm;
};

inline void MlaDecodeRef(const RefMla &m, const GcMlaDims &d, float eps,
                         float scale, const std::vector<float> &rc,
                         const std::vector<float> &rs, int pos,
                         std::vector<float> &k_cache,
                         std::vector<float> &v_cache,
                         const std::vector<float> &x, std::vector<float> &out)
{
    const int hidden = d.hidden, heads = d.heads, q_lora = d.q_lora,
              kv_lora = d.kv_lora, nope = d.qk_nope, rope = d.rope,
              vh = d.v_head;
    const int qk = nope + rope, kvw = kv_lora + rope, hv = heads * vh,
              kvb = heads * (nope + vh), half = rope / 2;
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x[i]);

    std::vector<float> qa = GemvRef(m.q_a.w, x16, q_lora, hidden);
    for (float &z : qa) z = R16(z);
    std::vector<float> qan =
        NormRef(qa, m.q_a_norm.w, m.q_a_norm.up, m.q_a_norm.down, q_lora,
                m.q_a_norm.rank, eps);
    const float aq = std::sqrt(static_cast<float>(hidden) / q_lora);
    std::vector<float> qas(static_cast<size_t>(q_lora));
    for (int i = 0; i < q_lora; ++i) qas[i] = R16(qan[i] * aq);
    std::vector<float> qb = GemvRef(m.q_b.w, qas, heads * qk, q_lora);
    for (float &z : qb) z = R16(z);

    std::vector<float> kv = GemvRef(m.kv_a.w, x16, kvw, hidden);
    for (float &z : kv) z = R16(z);
    std::vector<float> kp(kv.begin(), kv.begin() + kv_lora);
    std::vector<float> krot(kv.begin() + kv_lora, kv.end());
    std::vector<float> kpn =
        NormRef(kp, m.kv_a_norm.w, m.kv_a_norm.up, m.kv_a_norm.down, kv_lora,
                m.kv_a_norm.rank, eps);
    const float akv = std::sqrt(static_cast<float>(hidden) / kv_lora);
    std::vector<float> kps(static_cast<size_t>(kv_lora));
    for (int i = 0; i < kv_lora; ++i) kps[i] = R16(kpn[i] * akv);
    std::vector<float> kvb_w = GemvRef(m.kv_b.w, kps, kvb, kv_lora);
    for (float &z : kvb_w) z = R16(z);

    std::vector<float> c(static_cast<size_t>(half)),
        s(static_cast<size_t>(half));
    for (int j = 0; j < half; ++j)
    {
        c[static_cast<size_t>(j)] = rc[static_cast<size_t>(pos) * half + j];
        s[static_cast<size_t>(j)] = rs[static_cast<size_t>(pos) * half + j];
    }
    RopeRef(qb, heads, qk, nope, krot, c, s);

    for (int h = 0; h < heads; ++h)
    {
        for (int i = 0; i < nope; ++i)
        {
            k_cache[(static_cast<size_t>(pos) * heads + h) * qk + i] =
                kvb_w[static_cast<size_t>(h) * (nope + vh) + i];
            v_cache[(static_cast<size_t>(pos) * heads + h) * vh + i] =
                kvb_w[static_cast<size_t>(h) * (nope + vh) + nope + i];
        }
        for (int i = 0; i < rope; ++i)
            k_cache[(static_cast<size_t>(pos) * heads + h) * qk + nope + i] =
                krot[static_cast<size_t>(i)];
    }

    std::vector<float> gate = GemvRef(m.gate.w, x16, hv, hidden);
    for (float &z : gate) z = R16(z);

    std::vector<float> attn(static_cast<size_t>(hv), 0.0f);
    for (int h = 0; h < heads; ++h)
    {
        std::vector<float> sc(static_cast<size_t>(pos) + 1);
        for (int t = 0; t <= pos; ++t)
        {
            float a = 0.0f;
            for (int i = 0; i < qk; ++i)
                a += qb[static_cast<size_t>(h) * qk + i] *
                     k_cache[(static_cast<size_t>(t) * heads + h) * qk + i];
            sc[static_cast<size_t>(t)] = a * scale;
        }
        float mx = sc[0];
        for (int t = 1; t <= pos; ++t)
            mx = std::fmax(mx, sc[static_cast<size_t>(t)]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t)
        {
            sc[static_cast<size_t>(t)] =
                std::exp(sc[static_cast<size_t>(t)] - mx);
            sum += sc[static_cast<size_t>(t)];
        }
        for (int i = 0; i < vh; ++i)
        {
            float o = 0.0f;
            for (int t = 0; t <= pos; ++t)
                o += sc[static_cast<size_t>(t)] / sum *
                     v_cache[(static_cast<size_t>(t) * heads + h) * vh + i];
            const float g = gate[static_cast<size_t>(h) * vh + i];
            attn[static_cast<size_t>(h) * vh + i] =
                R16(o / (1.0f + std::exp(-g)));
        }
    }
    out = GemvRef(m.o_proj.w, attn, hidden, hv);
    for (float &z : out) z = R16(z);
}

struct RefGdn {
    RefNvfp4 in_qkvz, out_proj;
    std::vector<float> in_ba, conv, A_log, dt_bias, norm;
    int conv_dim = 0, conv_k = 0, ba_rows = 0, v_heads = 0;
};

inline void GdnDecodeRef(const RefGdn &g, const GcGdnDims &d, float eps,
                         float onorm_scale, std::vector<float> &conv_state,
                         std::vector<float> &rec_state,
                         const std::vector<float> &x, std::vector<float> &out)
{
    const int hidden = d.hidden, k_heads = d.k_heads, v_heads = d.v_heads,
              hk = d.hk, hv = d.hv, conv_k = d.conv_k;
    const int key_dim = k_heads * hk, value_dim = v_heads * hv;
    const int conv_dim = 2 * key_dim + value_dim, ba_dim = 2 * v_heads;
    const int qkvz_dim = 2 * key_dim + 2 * value_dim, rep = v_heads / k_heads;
    const int space = 2 * hk + 2 * hv * rep;

    std::vector<float> x16(static_cast<size_t>(hidden)),
        xb(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i)
    {
        x16[i] = R16(x[i]);
        xb[i] = RBf16(x[i]);
    }
    std::vector<float> qkvz = GemvRef(g.in_qkvz.w, x16, qkvz_dim, hidden);
    for (float &f : qkvz) f = R16(f);
    std::vector<float> ba = GemvRef(g.in_ba, xb, ba_dim, hidden);

    std::vector<float> mixed(static_cast<size_t>(conv_dim), 0.0f);
    std::vector<float> z(static_cast<size_t>(value_dim), 0.0f);
    for (int gh = 0; gh < k_heads; ++gh)
    {
        const float *base = &qkvz[static_cast<size_t>(gh) * space];
        for (int i = 0; i < hk; ++i)
        {
            mixed[gh * hk + i] = base[i];
            mixed[key_dim + gh * hk + i] = base[hk + i];
        }
        for (int j = 0; j < hv * rep; ++j)
        {
            mixed[2 * key_dim + gh * hv * rep + j] = base[2 * hk + j];
            z[gh * hv * rep + j] = base[2 * hk + hv * rep + j];
        }
    }

    std::vector<float> beta(static_cast<size_t>(v_heads));
    std::vector<float> gv(static_cast<size_t>(v_heads));
    for (int h = 0; h < v_heads; ++h)
    {
        const int kh = h / rep;
        const float b = ba[kh * 2 * rep + (h % rep)];
        const float a = ba[kh * 2 * rep + rep + (h % rep)];
        beta[static_cast<size_t>(h)] = 1.0f / (1.0f + std::exp(-b));
        const float t = a + g.dt_bias[static_cast<size_t>(h)];
        const float sp = t > 20.0f ? t : std::log1p(std::exp(t));
        gv[static_cast<size_t>(h)] =
            -std::exp(g.A_log[static_cast<size_t>(h)]) * sp;
    }

    std::vector<float> conv_out(static_cast<size_t>(conv_dim));
    for (int dd = 0; dd < conv_dim; ++dd)
    {
        float acc = g.conv[static_cast<size_t>(dd) * conv_k + (conv_k - 1)] *
                    mixed[static_cast<size_t>(dd)];
        for (int j = 0; j < conv_k - 1; ++j)
            acc += g.conv[static_cast<size_t>(dd) * conv_k + j] *
                   conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j];
        for (int j = 0; j < conv_k - 2; ++j)
            conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j] =
                conv_state[static_cast<size_t>(dd) * (conv_k - 1) + j + 1];
        conv_state[static_cast<size_t>(dd) * (conv_k - 1) + (conv_k - 2)] =
            mixed[static_cast<size_t>(dd)];
        conv_out[static_cast<size_t>(dd)] = acc / (1.0f + std::exp(-acc));
    }

    std::vector<float> q(static_cast<size_t>(v_heads) * hk),
        kk(static_cast<size_t>(v_heads) * hk),
        vv(static_cast<size_t>(value_dim));
    const float qscale = 1.0f / std::sqrt(static_cast<float>(hk));
    for (int h = 0; h < v_heads; ++h)
    {
        const int kh = h / rep;
        double sq = 0.0, sk = 0.0;
        for (int i = 0; i < hk; ++i)
        {
            const float qv = conv_out[kh * hk + i];
            const float kv = conv_out[key_dim + kh * hk + i];
            sq += static_cast<double>(qv) * qv;
            sk += static_cast<double>(kv) * kv;
        }
        const float iq = 1.0f / std::sqrt(static_cast<float>(sq) + 1e-6f);
        const float ik = 1.0f / std::sqrt(static_cast<float>(sk) + 1e-6f);
        for (int i = 0; i < hk; ++i)
        {
            q[static_cast<size_t>(h) * hk + i] =
                conv_out[kh * hk + i] * iq * qscale;
            kk[static_cast<size_t>(h) * hk + i] =
                conv_out[key_dim + kh * hk + i] * ik;
        }
        for (int j = 0; j < hv; ++j)
            vv[static_cast<size_t>(h) * hv + j] =
                conv_out[2 * key_dim + h * hv + j];
    }

    std::vector<float> o(static_cast<size_t>(value_dim));
    for (int h = 0; h < v_heads; ++h)
    {
        const size_t sb = static_cast<size_t>(h) * hk * hv;
        const float decay = std::exp(gv[static_cast<size_t>(h)]);
        for (int i = 0; i < hk; ++i)
            for (int j = 0; j < hv; ++j)
                rec_state[sb + static_cast<size_t>(i) * hv + j] *= decay;
        for (int j = 0; j < hv; ++j)
        {
            float kvmem = 0.0f;
            for (int i = 0; i < hk; ++i)
                kvmem += rec_state[sb + static_cast<size_t>(i) * hv + j] *
                         kk[static_cast<size_t>(h) * hk + i];
            const float delta =
                (vv[static_cast<size_t>(h) * hv + j] - kvmem) *
                beta[static_cast<size_t>(h)];
            for (int i = 0; i < hk; ++i)
                rec_state[sb + static_cast<size_t>(i) * hv + j] +=
                    kk[static_cast<size_t>(h) * hk + i] * delta;
            float acc = 0.0f;
            for (int i = 0; i < hk; ++i)
                acc += rec_state[sb + static_cast<size_t>(i) * hv + j] *
                       q[static_cast<size_t>(h) * hk + i];
            o[static_cast<size_t>(h) * hv + j] = acc;
        }
    }

    std::vector<float> onorm(static_cast<size_t>(value_dim));
    for (int h = 0; h < v_heads; ++h)
    {
        double ss = 0.0;
        for (int j = 0; j < hv; ++j)
            ss += static_cast<double>(o[static_cast<size_t>(h) * hv + j]) *
                  o[static_cast<size_t>(h) * hv + j];
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / hv) + eps);
        for (int j = 0; j < hv; ++j)
        {
            const float xn = o[static_cast<size_t>(h) * hv + j] * inv;
            const float zz = z[static_cast<size_t>(h) * hv + j];
            onorm[static_cast<size_t>(h) * hv + j] =
                xn * (1.0f + g.norm[static_cast<size_t>(j)]) * onorm_scale /
                (1.0f + std::exp(-zz));
        }
    }
    for (float &f : onorm) f = R16(f);
    out = GemvRef(g.out_proj.w, onorm, hidden, value_dim);
    for (float &f : out) f = R16(f);
}

struct RefExpert {
    RefNvfp4 gate, up, down;
};

struct RefMoe {
    std::vector<float> router;       // [E, hidden]
    std::vector<float> router_bias;  // [E]
    int num_experts = 0, top_k = 0, intermediate = 0;
    RefNvfp4 shared_gate, shared_up, shared_down;
    std::vector<RefExpert> experts;  // [E]
};

struct RefDense {
    RefNvfp4 gate, up, down;
};

struct RefRoute {
    std::vector<int> idx;
    std::vector<float> w;
};

// sigmoid router: logits = x.W^T; scores = sigmoid(logits);
// choice = scores + bias; top-k (lowest index wins ties);
// w = scores[idx] / (sum + 1e-20) * routed_scaling_factor.
inline RefRoute RouterTopKRef(const std::vector<float> &x,
                              const std::vector<float> &W,
                              const std::vector<float> &bias, int E, int hidden,
                              int k, float routed_scale)
{
    std::vector<float> scores(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e)
    {
        float a = 0.0f;
        for (int i = 0; i < hidden; ++i)
            a += W[static_cast<size_t>(e) * hidden + i] *
                 x[static_cast<size_t>(i)];
        scores[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-a));
    }
    std::vector<int> order(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float ca =
            scores[static_cast<size_t>(a)] + bias[static_cast<size_t>(a)];
        const float cb =
            scores[static_cast<size_t>(b)] + bias[static_cast<size_t>(b)];
        if (ca != cb) return ca > cb;
        return a < b;
    });
    RefRoute r;
    r.idx.assign(order.begin(), order.begin() + k);
    r.w.assign(static_cast<size_t>(k), 0.0f);
    float sum = 0.0f;
    for (int j = 0; j < k; ++j)
    {
        r.w[static_cast<size_t>(j)] =
            scores[static_cast<size_t>(r.idx[static_cast<size_t>(j)])];
        sum += r.w[static_cast<size_t>(j)];
    }
    for (int j = 0; j < k; ++j)
        r.w[static_cast<size_t>(j)] =
            r.w[static_cast<size_t>(j)] / (sum + 1e-20f) * routed_scale;
    return r;
}

inline std::vector<float> MoeRef(const RefMoe &m,
                                 const std::vector<float> &x_norm, int hidden,
                                 float routed_scale, float swiglu_limit)
{
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x_norm[i]);
    const RefRoute r = RouterTopKRef(x_norm, m.router, m.router_bias,
                                     m.num_experts, hidden, m.top_k,
                                     routed_scale);
    const int I = m.intermediate;
    std::vector<float> mlp(static_cast<size_t>(hidden), 0.0f);
    for (int j = 0; j < m.top_k; ++j)
    {
        const RefExpert &e =
            m.experts[static_cast<size_t>(r.idx[static_cast<size_t>(j)])];
        std::vector<float> gt = GemvRef(e.gate.w, x16, I, hidden);
        for (float &f : gt) f = R16(f);
        std::vector<float> up = GemvRef(e.up.w, x16, I, hidden);
        for (float &f : up) f = R16(f);
        std::vector<float> mid(static_cast<size_t>(I));
        for (int i = 0; i < I; ++i)
            mid[static_cast<size_t>(i)] =
                R16(lsxcommon::gigachat::dev::SwigluLimitElement(
                    gt[static_cast<size_t>(i)], up[static_cast<size_t>(i)],
                    swiglu_limit));
        std::vector<float> dn = GemvRef(e.down.w, mid, hidden, I);
        for (float &f : dn) f = R16(f);
        for (int i = 0; i < hidden; ++i)
            mlp[static_cast<size_t>(i)] +=
                r.w[static_cast<size_t>(j)] * dn[static_cast<size_t>(i)];
    }
    std::vector<float> sg = GemvRef(m.shared_gate.w, x16, I, hidden);
    for (float &f : sg) f = R16(f);
    std::vector<float> su = GemvRef(m.shared_up.w, x16, I, hidden);
    for (float &f : su) f = R16(f);
    std::vector<float> sm(static_cast<size_t>(I));
    for (int i = 0; i < I; ++i)
        sm[static_cast<size_t>(i)] =
            R16(lsxcommon::gigachat::dev::SwigluLimitElement(
                sg[static_cast<size_t>(i)], su[static_cast<size_t>(i)],
                swiglu_limit));
    std::vector<float> sd = GemvRef(m.shared_down.w, sm, hidden, I);
    for (float &f : sd) f = R16(f);
    for (int i = 0; i < hidden; ++i)
        mlp[static_cast<size_t>(i)] += sd[static_cast<size_t>(i)];
    return mlp;
}

inline std::vector<float> DenseMlpRef(const RefDense &dm,
                                      const std::vector<float> &x_norm,
                                      int hidden, int inter,
                                      float swiglu_limit)
{
    std::vector<float> x16(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) x16[i] = R16(x_norm[i]);
    std::vector<float> gt = GemvRef(dm.gate.w, x16, inter, hidden);
    for (float &f : gt) f = R16(f);
    std::vector<float> up = GemvRef(dm.up.w, x16, inter, hidden);
    for (float &f : up) f = R16(f);
    std::vector<float> mid(static_cast<size_t>(inter));
    for (int i = 0; i < inter; ++i)
        mid[static_cast<size_t>(i)] =
            R16(lsxcommon::gigachat::dev::SwigluLimitElement(
                gt[static_cast<size_t>(i)], up[static_cast<size_t>(i)],
                swiglu_limit));
    std::vector<float> dn = GemvRef(dm.down.w, mid, hidden, inter);
    for (float &f : dn) f = R16(f);
    return dn;
}

struct RefLayer {
    bool full_attention = false, moe = false;
    RefNorm input_norm, post_attn_norm, post_self_attn_norm, post_ffw_norm;
    RefMla mla;
    RefGdn gdn;
    RefMoe moe_w;
    RefDense dense;
    std::vector<float> k_cache, v_cache, conv_state, rec_state;
};

struct RefModel {
    int hidden = 0, vocab = 0, capacity = 2;
    std::vector<float> embed, lm_head;
    RefNorm final_norm;
    std::vector<RefLayer> layers;
    GcMlaDims mla_d;
    GcGdnDims gdn_d;
    float eps = 1e-6f;
    float attn_scale = 0.0f;
    float routed_scale = 2.5f;
    float swiglu_limit = 10.0f;
    int num_experts = 0, top_k = 0, moe_intermediate = 0, first_k_dense = 0;
    std::vector<float> rc, rs;  // YaRN tables [capacity, rope/2]
};

inline std::vector<float> GatedNormRef(const std::vector<float> &x,
                                       const RefNorm &n, float eps)
{
    return NormRef(x, n.w, n.up, n.down, n.dim, n.rank, eps);
}

inline std::vector<float> RunModelRef(RefModel &m,
                                      const std::vector<int> &tokens)
{
    for (RefLayer &l : m.layers)
    {
        std::fill(l.k_cache.begin(), l.k_cache.end(), 0.0f);
        std::fill(l.v_cache.begin(), l.v_cache.end(), 0.0f);
        std::fill(l.conv_state.begin(), l.conv_state.end(), 0.0f);
        std::fill(l.rec_state.begin(), l.rec_state.end(), 0.0f);
    }
    std::vector<float> logits;
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos)
    {
        std::vector<float> h(static_cast<size_t>(m.hidden));
        for (int i = 0; i < m.hidden; ++i)
            h[static_cast<size_t>(i)] =
                RBf16(m.embed[static_cast<size_t>(
                                   tokens[static_cast<size_t>(pos)]) *
                                   m.hidden +
                               i]);
        for (RefLayer &L : m.layers)
        {
            const std::vector<float> res = h;
            const std::vector<float> ni = GatedNormRef(h, L.input_norm, m.eps);
            std::vector<float> attn;
            if (L.full_attention)
                MlaDecodeRef(L.mla, m.mla_d, m.eps, m.attn_scale, m.rc, m.rs,
                             pos, L.k_cache, L.v_cache, ni, attn);
            else
                GdnDecodeRef(L.gdn, m.gdn_d, m.eps,
                             lsxcommon::gigachat::frozen::
                                 kLinearSigmoidGateScale,
                             L.conv_state, L.rec_state, ni, attn);
            const std::vector<float> g1 =
                GatedNormRef(attn, L.post_self_attn_norm, m.eps);
            h.resize(static_cast<size_t>(m.hidden));
            for (int i = 0; i < m.hidden; ++i)
                h[static_cast<size_t>(i)] =
                    res[static_cast<size_t>(i)] + g1[static_cast<size_t>(i)];

            const std::vector<float> res2 = h;
            const std::vector<float> np =
                GatedNormRef(h, L.post_attn_norm, m.eps);
            const std::vector<float> mlp =
                L.moe ? MoeRef(L.moe_w, np, m.hidden, m.routed_scale,
                               m.swiglu_limit)
                      : DenseMlpRef(L.dense, np, m.hidden, L.dense.gate.rows,
                                    m.swiglu_limit);
            const std::vector<float> g2 =
                GatedNormRef(mlp, L.post_ffw_norm, m.eps);
            for (int i = 0; i < m.hidden; ++i)
                h[static_cast<size_t>(i)] = res2[static_cast<size_t>(i)] +
                                            g2[static_cast<size_t>(i)];
        }
        const std::vector<float> fn = GatedNormRef(h, m.final_norm, m.eps);
        std::vector<float> xb(static_cast<size_t>(m.hidden));
        for (int i = 0; i < m.hidden; ++i)
            xb[static_cast<size_t>(i)] = RBf16(fn[static_cast<size_t>(i)]);
        logits = GemvRef(m.lm_head, xb, m.vocab, m.hidden);
    }
    return logits;
}

} // namespace gigachatref
