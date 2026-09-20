#pragma once

// Pinned host/device reference for GigaChat3.5 gating math, transcribed from
// the shipped modeling_gigachat3_5.py (ZeroCenteredGatedNorm / GatedDeltaNet
// output norm / swiglu_limit / gated attention). Single source of truth for
// the device kernels in this family.
#include <cstdint>

#ifndef __CUDACC__
#include <cmath>
#define LSX_HD inline
#else
#define LSX_HD __host__ __device__ inline
#endif

namespace lsxcommon::gigachat::dev {

LSX_HD float Sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }
LSX_HD float Silu(float v) { return v * Sigmoid(v); }

// ZeroCenteredGatedNorm: `x_norm * (1+w) * 2.0 * sigmoid(gate)` where the
// gating weight is 2.0 (config `layernorm_gating_weight`). Callers pass the
// already-computed `gate = W_down . silu(W_up . x_norm)` and the already
// RMS-normalized `x_norm`.
LSX_HD float ZeroCenteredGatedNormElement(float x_norm, float w,
                                                 float gate)
{
    return x_norm * (1.0f + w) * 2.0f * Sigmoid(gate);
}

// GatedDeltaNet output norm (`gated_rmsnorm_sigmoid_zero_centered`):
// `x_norm * (1+w) * sigmoid(z) * linear_sigmoid_gate_scale` (scale = 2.0).
LSX_HD float GdnOutputNormElement(float x_norm, float w, float z,
                                         float /*eps*/, float scale)
{
    return x_norm * (1.0f + w) * scale * Sigmoid(z);
}

// swiglu_limit: `gate = clamp(gate, max=L)`, `up = clamp(up, -L, L)`;
// returns `silu(gate) * up`. `L <= 0` disables clipping.
LSX_HD float SwigluLimitElement(float g, float u, float limit)
{
    if (limit > 0.0f)
    {
        g = fminf(g, limit);
        u = fmaxf(fminf(u, limit), -limit);
    }
    return Silu(g) * u;
}

// Gated attention: `y * sigmoid(gate_proj(x))`.
LSX_HD float SigmoidGateElement(float y, float gate)
{
    return y * Sigmoid(gate);
}

} // namespace lsxcommon::gigachat::dev
