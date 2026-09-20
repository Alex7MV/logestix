#pragma once

// Host F32 reference for the GigaChat3.5 whole-stack synthetic parity test.
// Pure host (std:: only; included from .cu translation units, so no abseil
// containers). Shares the NVFP4 codec with the device so dequant is exact; the
// arithmetic is independent of the device kernels and mirrors every F16/BF16
// rounding the pipeline applies.
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

} // namespace gigachatref
