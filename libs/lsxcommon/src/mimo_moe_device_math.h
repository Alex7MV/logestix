#pragma once

// MiMo-V2.6-Flash-RL host reference math (family-owned). Mirrors the pinned
// formulas in the ingestion design; used by kernel-parity tests only. Host-only
// (never include from a .cu).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lsxcommon {
namespace mimo {
namespace ref {

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float Silu(float g) { return g / (1.0f + std::exp(-g)); }

// out[j] = (x[j] / sqrt(mean(x^2) + eps)) * w[j]
inline void RmsNorm(const float *x, const float *w, int n, float eps, float *out)
{
    float ss = 0.0f;
    for (int j = 0; j < n; ++j) ss += x[j] * x[j];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int j = 0; j < n; ++j) out[j] = x[j] * inv * w[j];
}

// sigmoid-router (noaux_tc, n_group=1): select top-k over
// sigmoid(logits)+bias; weights are the un-biased sigmoid at those indices,
// normalized by their sum.
inline void RouterTopK(const float *x, const float *gate_w, const float *bias,
                       int hidden, int num_experts, int top_k,
                       std::vector<int> &idx, std::vector<float> &w)
{
    std::vector<float> scores(num_experts);
    std::vector<float> choice(num_experts);
    for (int e = 0; e < num_experts; ++e)
    {
        float dot = 0.0f;
        for (int k = 0; k < hidden; ++k) dot += x[k] * gate_w[e * hidden + k];
        scores[e] = Sigmoid(dot);
        choice[e] = scores[e] + bias[e];
    }
    idx.assign(num_experts, 0);
    for (int i = 0; i < num_experts; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return choice[a] > choice[b]; });
    idx.resize(top_k);
    w.assign(top_k, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < top_k; ++i) { w[i] = scores[idx[i]]; sum += w[i]; }
    for (int i = 0; i < top_k; ++i) w[i] /= (sum + 1e-20f);
}

inline void SwiGLU(const float *gate, const float *up, int64_t n, float *out)
{
    for (int64_t i = 0; i < n; ++i) out[i] = Silu(gate[i]) * up[i];
}

// ── Attention reference (prefill, no cache) ─────────────────────────────────
struct AttnShape {
    int heads = 64;
    int kv_heads = 4;
    int head_dim = 192;
    int v_head_dim = 128;
    int rope_dim = 64;
    bool is_swa = false;
    int window = 128;
    float value_scale = 0.707f;
    double rope_theta = 1e7;
    float scale = 0.07216878364870322f;
    int q_size = 12288;
    int k_size = 768;
    int v_size = 512;
};

inline void RopeHalf(std::vector<float> &v, int pos, int rope_dim, double theta)
{
    const int half = rope_dim / 2;
    for (int d = 0; d < half; ++d)
    {
        const double freq = std::pow(theta, -2.0 * d / rope_dim);
        const double ang = pos * freq;
        const float c = static_cast<float>(std::cos(ang));
        const float s = static_cast<float>(std::sin(ang));
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * c - x2 * s;
        v[d + half] = x2 * c + x1 * s;
    }
}

// qkv: [seq, q_size + 2*kv_size] row-major per token (v unscaled, rope raw).
// out: [seq, heads * v_head_dim].
inline void AttentionRef(const std::vector<float> &qkv, int seq,
                         const AttnShape &p, const float *sink, float *out)
{
    const int group = p.heads / p.kv_heads;
    const int rows = p.q_size + 2 * p.k_size;
    for (int i = 0; i < seq; ++i)
    {
        const float *row = qkv.data() + static_cast<int64_t>(i) * rows;
        for (int h = 0; h < p.heads; ++h)
        {
            std::vector<float> q(row + h * p.head_dim,
                                 row + h * p.head_dim + p.head_dim);
            RopeHalf(q, i, p.rope_dim, p.rope_theta);
            const int kh = h / group;
            const int j0 = p.is_swa ? std::max(0, i - p.window + 1) : 0;
            // scores over keys j0..i plus the sink column.
            std::vector<double> s;
            std::vector<int> key_idx;
            for (int j = j0; j <= i; ++j)
            {
                const float *krow = qkv.data() + static_cast<int64_t>(j) * rows;
                std::vector<float> k(
                    krow + p.q_size + kh * p.head_dim,
                    krow + p.q_size + kh * p.head_dim + p.head_dim);
                RopeHalf(k, j, p.rope_dim, p.rope_theta);
                double dot = 0.0;
                for (int d = 0; d < p.head_dim; ++d) dot += q[d] * k[d];
                s.push_back(dot * p.scale);
                key_idx.push_back(j);
            }
            const bool has_sink = sink != nullptr;
            if (has_sink) s.push_back(sink[h]);
            const double mx = *std::max_element(s.begin(), s.end());
            double l = 0.0;
            std::vector<double> pr(s.size());
            for (size_t t = 0; t < s.size(); ++t) { pr[t] = std::exp(s[t] - mx); l += pr[t]; }
            float *o = out + (static_cast<int64_t>(i) * p.heads + h) * p.v_head_dim;
            for (int d = 0; d < p.v_head_dim; ++d) o[d] = 0.0f;
            for (size_t t = 0; t + (has_sink ? 1u : 0u) < s.size(); ++t)
            {
                const float pj = static_cast<float>(pr[t] / l);
                const float *vrow = qkv.data() + static_cast<int64_t>(key_idx[t]) * rows;
                const float *vv = vrow + p.q_size + p.k_size + kh * p.v_head_dim;
                for (int d = 0; d < p.v_head_dim; ++d)
                    o[d] += pj * vv[d] * p.value_scale;
            }
        }
    }
}

} // namespace ref
} // namespace mimo
} // namespace lsxcommon
