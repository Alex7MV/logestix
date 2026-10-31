#pragma once

// Host reference for the Beam numeric contract (kernel parity oracle).
// Pure host code: no CUDA, no Arrow, safe to include from tests.
//
// Mirrors transformers/models/beam/modeling_beam.py:
//   - plain RMSNorm (weight form), per-head q/k norm
//   - rotate-half RoPE on the full head_dim
//   - full-attention NoPE SSMax query scale (1 + (pos+1)/2048)^0.1
//   - sliding window = window tokens INCLUDING the current one
//   - group-limited sigmoid router with an FP32 correction bias
//   - depth-scaled residual add (both operands rounded before the add)
//   - per-head sigmoid attention gate (+ one bias per head)
//   - final logit softcap
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lsxcommon::beam::ref {

inline void RmsNorm(const float *x, const float *w, float *out, int n,
                    float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

// HeadNorm: rows*heads independent RMSNorms over head_dim.
inline void HeadNorm(const float *x, const float *w, float *out, int rows,
                     int heads, int head_dim, float eps)
{
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h)
            RmsNorm(x + static_cast<int64_t>(r) * heads * head_dim +
                        h * head_dim,
                    w, out + static_cast<int64_t>(r) * heads * head_dim +
                               h * head_dim,
                    head_dim, eps);
}

// rotate_half: pair (j, j + head_dim/2).
inline void Rope(float *q, int heads, int head_dim, int pos, float theta)
{
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float *v = q + h * head_dim;
        for (int j = 0; j < half; ++j) {
            const float inv =
                1.0f / std::pow(theta, static_cast<float>(j) / half);
            const float ang = static_cast<float>(pos) * inv;
            const float c = std::cos(ang), s = std::sin(ang);
            const float x1 = v[j], x2 = v[j + half];
            v[j] = x1 * c - x2 * s;
            v[j + half] = x1 * s + x2 * c;
        }
    }
}

inline float SsmScale(int pos)
{
    return std::exp(0.1f * std::log1p((static_cast<float>(pos) + 1.0f) / 2048.0f));
}

inline void SsmScaleQ(float *q, int heads, int head_dim, int pos)
{
    const float s = SsmScale(pos);
    for (int i = 0; i < heads * head_dim; ++i) q[i] *= s;
}

inline int WindowStart(int pos, int window)
{
    if (window <= 0) return 0;
    return std::max(0, pos + 1 - window);
}

inline void AttnGate(float *attn, const float *gate, const float *gate_bias,
                     int heads, int head_dim)
{
    for (int h = 0; h < heads; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const int i = h * head_dim + d;
            attn[i] *= 1.0f / (1.0f + std::exp(-(gate[i] + gate_bias[h])));
        }
}

inline void DepthResidual(const float *branch, const float *residual,
                          float *out, int n, int idx)
{
    const float bs = 1.0f / std::sqrt(static_cast<float>(idx));
    const float rs = std::sqrt(static_cast<float>(idx - 1) / idx);
    for (int i = 0; i < n; ++i) out[i] = branch[i] * bs + residual[i] * rs;
}

inline float Softcap(float v)
{
    return std::tanh(v / 30.0f) * 30.0f;
}

struct RouterResult {
    std::vector<int32_t> ids;      // ascending
    std::vector<float> weights;    // renormalized plain sigmoid * routed scale
};

// Group-limited sigmoid router: logits over ALL experts, sigmoid, add the
// correction bias for CHOICE, group score = sum of the two best corrected
// scores per group, keep topk_group groups, top-k by corrected score, then the
// PLAIN sigmoid scores of the selected experts renormalized, times the routed
// scale. Ties resolve to the lower expert id.
inline RouterResult Router(const float *logits, const float *bias, int E,
                           int K, int G, int TG, float routed_scale)
{
    std::vector<float> score(E), choice(E);
    for (int e = 0; e < E; ++e) {
        score[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        choice[e] = score[e] + bias[e];
    }
    const int per = E / G;
    std::vector<float> gs(G, 0.0f);
    for (int g = 0; g < G; ++g) {
        float a = -std::numeric_limits<float>::infinity();
        float b = -std::numeric_limits<float>::infinity();
        for (int e = g * per; e < (g + 1) * per; ++e) {
            const float v = choice[e];
            if (v > a) {
                b = a;
                a = v;
            } else if (v > b) {
                b = v;
            }
        }
        gs[g] = a + b;
    }
    std::vector<int> order(G);
    for (int g = 0; g < G; ++g) order[g] = g;
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
        if (gs[x] != gs[y]) return gs[x] > gs[y];
        return x < y;
    });
    std::vector<char> group_keep(G, 0);
    for (int i = 0; i < TG; ++i) group_keep[order[i]] = 1;

    std::vector<char> taken(E, 0);
    std::vector<int> sel;
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float best_v = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < E; ++e) {
            if (taken[e] || !group_keep[e / per]) continue;
            if (best < 0 || choice[e] > best_v) {
                best_v = choice[e];
                best = e;
            }
        }
        if (best < 0) break;
        taken[best] = 1;
        sel.push_back(best);
    }
    std::sort(sel.begin(), sel.end());
    double sum = 0.0;
    for (int e : sel) sum += score[e];
    RouterResult r;
    for (int e : sel) {
        r.ids.push_back(e);
        r.weights.push_back(
            sum > 0.0 ? static_cast<float>(score[e] / sum) * routed_scale : 0.0f);
    }
    return r;
}

} // namespace lsxcommon::beam::ref
