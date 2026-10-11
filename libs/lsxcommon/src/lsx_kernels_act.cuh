// lsx_kernels_act.cuh — unified SwiGLU activation kernels (Phase A).
// Replaces per-model copies in qwen_moe_inference.cu, kimi_moe_inference.cu,
// deepseek_moe_inference_moe.cu (swiglu_clamp / swiglu_clamp_f32f),
// glm_moe_inference.cu (GlmSwigluClampKernel) and glm_moe_expert_kernels.cuh
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

// ── Overflow-proof SwiGLU → FP16 activation ────────────────────────────────
// silu(g)*u is evaluated in FP32 and stored into the FP16 activation buffer
// under a power-of-two prescale `inv_alpha`; `alpha` is then folded back into
// the down projection's output, so the accumulated result is unchanged.
//
// Why: models with massive activations overflow the FP16 ceiling (65504) in
// the PRODUCT even when gate and up each fit -- Mistral-Small-4-119B layer 32
// expert 67 reaches |silu(g)*u| = 2.5e5. The inf propagated through the down
// GEMM into the residual stream and NaN'd the whole stack in bit-exact mode
// (the TC path quantizes from FP32 and is immune). Used by the scalar MoE
// branches in lsx_moe_inference.cu and by Mistral's ungated shared expert.
//
// When every product fits, alpha == 1 and both kernels are exact no-ops: the
// value expression is the same F32 silu the scalar SwigluKernel computes, so
// the result is bit-identical.
// Leaves 4x headroom below the FP16 ceiling of 65504 for the scaled value.
constexpr float kSwiFp16Headroom = 16384.0f;

__global__ inline void SwigluF32Kernel(const __half *__restrict__ g,
                                       const __half *__restrict__ u,
                                       float *__restrict__ sw, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __half2float(g[i]);
    const float uf = __half2float(u[i]);
    sw[i] = __fdiv_rn(gf, __fadd_rn(1.0f, ExpF<ExpFlavor::Precise>(-gf))) * uf;
}

__global__ inline void SwigluAbsMaxKernel(const float *__restrict__ sw,
                                          float *__restrict__ absmax, int n)
{
    __shared__ float red[256];
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, fabsf(sw[i]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *absmax = red[0];
}

// alpha = smallest power of two with absmax/alpha <= kSwiFp16Headroom.
__global__ inline void SwigluAlphaKernel(const float *__restrict__ absmax,
                                         float *__restrict__ alpha,
                                         float *__restrict__ inv_alpha)
{
    if (threadIdx.x || blockIdx.x) return;
    const float a = *absmax;
    int k = 0;
    if (a > kSwiFp16Headroom)
    {
        float t = kSwiFp16Headroom;
        while (t < a && k < 60) { t *= 2.0f; ++k; }
    }
    *alpha = ldexpf(1.0f, k);
    *inv_alpha = ldexpf(1.0f, -k);
}

__global__ inline void SwigluStoreScaledKernel(const float *__restrict__ sw,
                                               const float *__restrict__ inv_alpha,
                                               __half *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(sw[i] * (*inv_alpha));
}

} // namespace kernels
} // namespace lsxcommon
