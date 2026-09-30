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

} // namespace ref
} // namespace mimo
} // namespace lsxcommon
