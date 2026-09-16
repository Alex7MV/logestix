#pragma once

// Host-only, device-independent reference math for the Nemotron-3.5 (NemotronH)
// hybrid. Pinned from transformers 5.15.1
// models/nemotron_h/modeling_nemotron_h.py and
// models/zamba2/modeling_zamba2.py::Zamba2RMSNormGated.
//
// This header is reachable from .cu translation units: it must stay free of
// abseil containers and of any CUDA/Arrow dependency.

#include "nemotron_moe_constants.h"

#include <cmath>
#include <cstddef>

namespace lsxcommon::nemotron_moe {
namespace ref {

inline float Silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float Softplus(float x)
{
    // numerically stable log(1 + e^x)
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float Relu2(float x)
{
    const float r = x > 0.0f ? x : 0.0f;
    return r * r;
}

// gated RMSNorm (Zamba2RMSNormGated): h = x * silu(gate); per-group rmsnorm;
// final elementwise weight. Layouts are flat: x/gate/weight are [n].
inline void GatedRmsNorm(const float *x, const float *gate, const float *weight,
                         int n, int group_size, float eps, float *out)
{
    const int groups = n / group_size;
    for (int g = 0; g < groups; ++g) {
        const int base = g * group_size;
        double ss = 0.0;
        for (int i = 0; i < group_size; ++i) {
            const float h = x[base + i] * Silu(gate[base + i]);
            ss += static_cast<double>(h) * static_cast<double>(h);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / group_size) + eps);
        for (int i = 0; i < group_size; ++i) {
            const float h = x[base + i] * Silu(gate[base + i]);
            out[base + i] = weight[base + i] * h * inv;
        }
    }
}

// One decode step of the Mamba-2 selective state-space recurrence.
//   xs     [heads, head_dim]
//   B, C   [groups, state_size]
//   D      [heads]
//   dt     [heads]            (raw in_proj slice)
//   dt_bias[heads], A_log[heads]
//   state  [heads, head_dim, state_size] (in/out carried across steps)
//   out    [heads, head_dim]
inline void SelectiveStateUpdate(const float *state_in, const float *xs,
                                 const float *B, const float *C,
                                 const float *D, const float *dt,
                                 const float *dt_bias, const float *A_log,
                                 int heads, int head_dim, int groups,
                                 int state_size, float *state_out, float *out)
{
    const int heads_per_group = heads / groups;
    for (int h = 0; h < heads; ++h) {
        float dth = Softplus(dt[h] + dt_bias[h]);
        if (dth < frozen::kDtMin) dth = frozen::kDtMin;
        const float A = -std::exp(A_log[h]);
        const float dA = std::exp(dth * A);
        const int g = h / heads_per_group;
        const float *Bg = B + static_cast<std::size_t>(g) * state_size;
        const float *Cg = C + static_cast<std::size_t>(g) * state_size;
        for (int d = 0; d < head_dim; ++d) {
            const float xsd = xs[h * head_dim + d];
            const std::size_t st_base =
                (static_cast<std::size_t>(h) * head_dim + d) * state_size;
            float acc = 0.0f;
            for (int s = 0; s < state_size; ++s) {
                const float st = state_in[st_base + s] * dA +
                                 dth * Bg[s] * xsd;
                state_out[st_base + s] = st;
                acc += st * Cg[s];
            }
            out[h * head_dim + d] = acc + xsd * D[h];
        }
    }
}

// Router: sigmoid scores, e_score_correction_bias selects the top-k only,
// normalized weights (norm_topk_prob) scaled by routed_scaling_factor.
// `n_group == 1` in this checkpoint, so the group stage is the identity.
inline void RouterTopK(const float *logits, const float *e_score_bias, int n,
                       int k, float scale, float *weights, int *indices)
{
    // sigmoid scores
    float s[frozen::kNumRouted];
    float choice[frozen::kNumRouted];
    for (int e = 0; e < n; ++e) {
        s[e] = 1.0f / (1.0f + std::exp(-logits[e]));
        choice[e] = s[e] + e_score_bias[e];
    }
    for (int t = 0; t < k; ++t) {
        int best = -1;
        for (int e = 0; e < n; ++e) {
            bool taken = false;
            for (int p = 0; p < t; ++p)
                if (indices[p] == e) taken = true;
            if (taken) continue;
            if (best < 0 || choice[e] > choice[best]) best = e;
        }
        indices[t] = best;
    }
    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
        weights[t] = s[indices[t]];
        sum += weights[t];
    }
    const float denom = sum + 1e-20f;
    for (int t = 0; t < k; ++t) weights[t] = weights[t] / denom * scale;
}

// Non-gated relu2 expert MLP: out = down @ relu2(up @ x).
inline void ExpertMlpRelu2(const float *x, const float *up, const float *down,
                           int h, int i, float *out)
{
    float act[frozen::kMoeIntermediate];
    for (int r = 0; r < i; ++r) {
        float acc = 0.0f;
        const float *row = up + static_cast<std::size_t>(r) * h;
        for (int c = 0; c < h; ++c) acc += row[c] * x[c];
        act[r] = Relu2(acc);
    }
    for (int r = 0; r < h; ++r) {
        float acc = 0.0f;
        const float *row = down + static_cast<std::size_t>(r) * i;
        for (int c = 0; c < i; ++c) acc += row[c] * act[c];
        out[r] = acc;
    }
}

} // namespace ref
} // namespace lsxcommon::nemotron_moe
