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
//
// `w_offset` selects the norm variant: 1.0 for GigaChat3.5's
// ZeroCenteredGatedNorm (scale `1+w`), 0.0 for GigaChat4's GatedNorm (scale
// `w`). The gate itself is the same rank-16 SiLU->sigmoid projection in both.
LSX_HD float GatedNormElement(float x_norm, float w, float gate,
                              float w_offset)
{
    return x_norm * (w_offset + w) * 2.0f * Sigmoid(gate);
}

// Legacy 3.5 spelling: zero-centred (`1+w`).
LSX_HD float ZeroCenteredGatedNormElement(float x_norm, float w,
                                                 float gate)
{
    return GatedNormElement(x_norm, w, gate, 1.0f);
}

// GatedDeltaNet output norm (`gated_rmsnorm_sigmoid[_zero_centered]`):
// `x_norm * (offset+w) * sigmoid(z) * linear_sigmoid_gate_scale` (scale = 2.0).
// `w_offset` is 1.0 for the 3.5 checkpoint and 0.0 for GigaChat4.
LSX_HD float GdnOutputNormElementOffset(float x_norm, float w, float z,
                                        float /*eps*/, float scale,
                                        float w_offset)
{
    return x_norm * (w_offset + w) * scale * Sigmoid(z);
}

// Legacy 3.5 spelling: zero-centred (`1+w`).
LSX_HD float GdnOutputNormElement(float x_norm, float w, float z,
                                         float eps, float scale)
{
    return GdnOutputNormElementOffset(x_norm, w, z, eps, scale, 1.0f);
}

// Final logit soft cap (`final_logit_softcapping`): `cap * tanh(x / cap)`.
// `cap <= 0` is the identity (no cap), never a division by zero.
LSX_HD float LogitSoftcapElement(float x, float cap)
{
    if (cap <= 0.0f) return x;
    return cap * tanhf(x / cap);
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
