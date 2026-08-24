// lsx_kernels_act.cuh — unified SwiGLU activation kernels (Phase A).
// Replaces per-model copies in qwen_moe_inference.cu, kimi_inference.cu,
// deepseek_inference_moe.cu (swiglu_clamp / swiglu_clamp_f32f),
// glm_inference.cu (GlmSwigluClampKernel) and glm_expert_kernels.cuh
// (GlmRealSwigluKernel).
//
// Canonical form: out = clamp(g) * sigmoid(clamp(g)) * clamp(u), where the
// gate uses a one-sided max clamp and up a two-sided clamp when Clamp=true.
//
// NOT unified here (documented deviations):
//   - qwen_inference.cu dense swiglu uses the silu_device form
//     x*e/(e+1) — numerically different rounding from the ratio form; kept
//     family-local (single site).
//   - deepseek silu_mul_kernel clamps asymmetrically (gate max-only on the
//     OTHER operand); single site, kept family-local.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "lsx_kernels_common.cuh"

namespace lsxcommon {
namespace kernels {

// Division form: out = (g / (1 + e^-g)) * u
// [qwen_moe, kimi, glm clamp/real]. Multiplication order preserved.
template <typename T, ExpFlavor F, bool Clamp>
__global__ void SwigluKernel(const T* __restrict__ g_in,
                             const T* __restrict__ u_in, T* __restrict__ out,
                             int64_t n, float limit)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    float g = detail::CastIn(g_in[i]);
    float u = detail::CastIn(u_in[i]);
    if (Clamp) {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
    }
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, ExpF<F>(-g)));
    out[i] = detail::CastOut<T>(__fmul_rn(s, u));
}

// Precomputed-sigmoid form: out = (1 / (1 + e^-g)) * g * u
// [deepseek swiglu_clamp / swiglu_clamp_f32f]. Multiplication order preserved.
template <typename T, ExpFlavor F, bool Clamp>
__global__ void SwigluSigKernel(const T* __restrict__ g_in,
                                const T* __restrict__ u_in,
                                T* __restrict__ out, int64_t n, float limit)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    float g = detail::CastIn(g_in[i]);
    float u = detail::CastIn(u_in[i]);
    if (Clamp) {
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
    }
    const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, ExpF<F>(-g)));
    out[i] = detail::CastOut<T>(__fmul_rn(sig, __fmul_rn(g, u)));
}

} // namespace kernels
} // namespace lsxcommon
