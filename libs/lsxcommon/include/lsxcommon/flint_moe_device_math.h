#pragma once

// Host reference for the Flint attention/router arithmetic. Plain C++ (no
// CUDA), shared by the kernel parity tests and by the CPU oracle tooling: the
// learned-sink denominator, the sliding-window bounds and the sigmoid top-k
// selection are the three places where a "looks right" kernel decodes silently
// wrong tokens, so each gets an executable definition outside the .cu.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace lsxcommon::flint {
namespace ref {

// First cached position a query at `pos` may attend to. window <= 0 = full.
inline int WindowStart(int pos, int window)
{
    return window > 0 ? std::max(0, pos - window + 1) : 0;
}

// Softmax over `nk` QK scores PLUS the learned sink, where the sink contributes
// no value. `weights` receives nk normalized probabilities that sum to
// 1 - p_sink (the sink's own share is deliberately not returned: it multiplies
// no value vector). Reproduces the vendor's FA4 fix:
//   row_max = max(scores..., sink)
//   row_sum = sum exp(score - row_max) + exp(sink - row_max)
inline void SinkSoftmaxWeights(const float *scores, int nk, float sink,
                               std::vector<float> &weights)
{
    weights.assign(static_cast<size_t>(nk), 0.0f);
    float m = sink;
    for (int i = 0; i < nk; ++i) m = std::max(m, scores[i]);
    float sum = std::exp(sink - m);
    for (int i = 0; i < nk; ++i) sum += std::exp(scores[i] - m);
    for (int i = 0; i < nk; ++i)
        weights[static_cast<size_t>(i)] = std::exp(scores[i] - m) / sum;
}

// Reference decode attention over a pos-major cache.
// q: [heads, head_dim]; k_cache/v_cache: [capacity, kv_heads, head_dim].
// out: [heads * head_dim]. Values are FP32 here (the kernel rounds to FP16).
inline void DecodeAttention(const float *q, int heads, int kv_heads,
                            int head_dim, int window, int pos, float scale,
                            const float *sink, const float *k_cache,
                            const float *v_cache, float *out)
{
    const int j0 = WindowStart(pos, window);
    const int nk = pos - j0 + 1;
    std::vector<float> scores(static_cast<size_t>(nk));
    std::vector<float> weights;
    for (int h = 0; h < heads; ++h) {
        const int kh = h / (heads / kv_heads);
        for (int j = j0; j <= pos; ++j) {
            const float *kj =
                k_cache + (static_cast<int64_t>(j) * kv_heads + kh) * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d)
                dot += static_cast<double>(q[h * head_dim + d]) * kj[d];
            scores[static_cast<size_t>(j - j0)] =
                static_cast<float>(dot) * scale;
        }
        SinkSoftmaxWeights(scores.data(), nk, sink[h], weights);
        for (int d = 0; d < head_dim; ++d) out[h * head_dim + d] = 0.0f;
        for (int j = j0; j <= pos; ++j) {
            const float p = weights[static_cast<size_t>(j - j0)];
            const float *vj =
                v_cache + (static_cast<int64_t>(j) * kv_heads + kh) * head_dim;
            for (int d = 0; d < head_dim; ++d)
                out[h * head_dim + d] += p * vj[d];
        }
    }
}

// Sigmoid top-k routing with renormalized weights, ids ascending.
// The checkpoint carries no expert bias (selection == plain sigmoid top-k).
inline void Router(const float *router_w, const float *x, int E, int K, int H,
                   std::vector<int32_t> &ids, std::vector<float> &coeffs)
{
    std::vector<float> score(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) {
        double dot = 0.0;
        for (int k = 0; k < H; ++k)
            dot += static_cast<double>(router_w[static_cast<int64_t>(e) * H + k]) *
                   x[k];
        score[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-static_cast<float>(dot)));
    }
    std::vector<char> taken(static_cast<size_t>(E), 0);
    ids.assign(static_cast<size_t>(K), -1);
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float best_v = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < E; ++e) {
            if (taken[static_cast<size_t>(e)]) continue;
            if (score[static_cast<size_t>(e)] > best_v) {
                best_v = score[static_cast<size_t>(e)];
                best = e;
            }
        }
        taken[static_cast<size_t>(best)] = 1;
        ids[static_cast<size_t>(k)] = best;
    }
    std::sort(ids.begin(), ids.end());
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += score[static_cast<size_t>(ids[static_cast<size_t>(k)])];
    coeffs.assign(static_cast<size_t>(K), 0.0f);
    for (int k = 0; k < K; ++k)
        coeffs[static_cast<size_t>(k)] =
            score[static_cast<size_t>(ids[static_cast<size_t>(k)])] /
            (sum + 1e-20f);
}

// Half-split rotary (HF rotate_half pairing j <-> j + head_dim/2).
inline void RopeHalf(float *v, int head_dim, float theta, int pos)
{
    const int half = head_dim / 2;
    for (int j = 0; j < half; ++j) {
        const double inv = std::pow(static_cast<double>(theta),
                                    static_cast<double>(j) / half);
        const double ang = static_cast<double>(pos) / inv;
        const float c = static_cast<float>(std::cos(ang));
        const float s = static_cast<float>(std::sin(ang));
        const float x1 = v[j], x2 = v[j + half];
        v[j] = x1 * c - x2 * s;
        v[j + half] = x1 * s + x2 * c;
    }
}

} // namespace ref
} // namespace lsxcommon::flint
