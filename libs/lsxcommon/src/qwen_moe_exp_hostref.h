#pragma once

// Host-side bit-exact reference sequences for qwen4_exp parity tests.
// Mirrors HF modeling_qwen4_exp.py float32 intermediates. Host-only: no CUDA
// includes, safe for test TUs and for headers reachable from .cu TUs.

#include <cmath>
#include <cstddef>
#include <vector>

namespace lsxcommon::qwen_moe_exp {
namespace hostref {

// HF RMSNorm float32 replica: x.float() * rsqrt(mean(x^2 per group) + eps)
// * (1 + weight). When `weight` is null the (1 + 0) centering is a no-op.
// `n` must be divisible by `group`.
inline void GroupRmsNorm(const float *x, int n, int group, double eps,
                         const float *weight, float *out)
{
    const int groups = n / group;
    for (int g = 0; g < groups; ++g) {
        double acc = 0.0;
        for (int i = 0; i < group; ++i) {
            const double v = x[g * group + i];
            acc += v * v;
        }
        const double inv = 1.0 / std::sqrt(acc / group + eps);
        for (int i = 0; i < group; ++i) {
            const double w = weight ? (1.0 + weight[g * group + i]) : 1.0;
            out[g * group + i] =
                static_cast<float>(x[g * group + i] * inv * w);
        }
    }
}

// Qwen4ExpTextGatedResidual.forward float32 replica.
//
// hyper_input: [hc, H] flattened; down_w: [lowrank, hc*H] row-major;
// up_w: [hc*H, lowrank] row-major; inject_w: [hc, hc*H] row-major.
// Outputs: mixed_out [H] (mean over streams of gated branch),
// inject_out [hc] (2 * sigmoid(W_inj @ normed / hc)).
inline void GatedResidualHost(const float *hyper_input, const float *down_w,
                              const float *up_w, const float *inject_w,
                              int H, int hc, int lowrank, double eps,
                              float *mixed_out, float *inject_out)
{
    const int n = H * hc;
    std::vector<float> normed(n), dn(lowrank), up(n);
    GroupRmsNorm(hyper_input, n, H, eps, /*weight=*/nullptr, normed.data());
    // down: [lowrank] = W_dn @ normed; silu(v / hc) = v / hc / (1 + e^(-v/hc))
    for (int r = 0; r < lowrank; ++r) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += down_w[r * n + i] * normed[i];
        const double v = acc / hc;
        dn[r] = static_cast<float>(v / (1.0 + std::exp(-v)));
    }
    // up: [n] = W_up @ dn; sigmoid
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int r = 0; r < lowrank; ++r) acc += up_w[i * lowrank + r] * dn[r];
        up[i] = 1.0f / (1.0f + std::exp(-acc));
    }
    // mixed = mean over hc branches of (sigmoid_gate ⊙ normed)
    for (int h = 0; h < H; ++h) {
        double acc = 0.0;
        for (int c = 0; c < hc; ++c)
            acc += static_cast<double>(up[c * H + h]) * normed[c * H + h];
        mixed_out[h] = static_cast<float>(acc / hc);
    }
    // injection weights: 2 * sigmoid(W_inj @ normed / hc)
    if (inject_w) {
        for (int c = 0; c < hc; ++c) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i)
                acc += inject_w[c * n + i] * normed[i];
            inject_out[c] =
                static_cast<float>(2.0 / (1.0 + std::exp(-acc / hc)));
        }
    }
}

// PLE gate (HF Qwen4ExpTextPLELayer): s = sum_c key_c · query_c / sqrt(H);
// gate = sign(s) * sqrt(|s| clamped to 1e-6).
inline float PleGateHost(const float *key, const float *query, int H, int hc)
{
    double acc = 0.0;
    for (int c = 0; c < hc; ++c)
        for (int h = 0; h < H; ++h)
            acc += static_cast<double>(key[c * H + h]) * query[c * H + h];
    const double s = acc / std::sqrt(static_cast<double>(H));
    const double a = std::fmax(1e-6, std::fabs(s));
    return static_cast<float>(std::sqrt(a) * (s < 0 ? -1.0 : 1.0));
}

} // namespace hostref
} // namespace lsxcommon::qwen_moe_exp
