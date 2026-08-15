#pragma once

// Device-side math for Qwen3.5-9B, matching transformers modeling_qwen3_5.py.
// Compiled only within the CUDA OBJECT library (lsxcommon_qwen_cu).

namespace lsxcommon::qwen::dev {

// l2norm over `dim` elements (eps k = 1e-6), matching the FLA-style l2norm.
__device__ __forceinline__ float l2norm_dot(const float *v, int dim)
{
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) acc = fmaf(v[i], v[i], acc);
    return acc;
}

__device__ __forceinline__ void l2norm_inplace(float *v, int dim, float eps)
{
    float inv = rsqrtf(l2norm_dot(v, dim) + eps);
    for (int i = 0; i < dim; ++i) v[i] = v[i] * inv;
}

// softplus: ln(1 + e^x), stable for x > 0.
__device__ __forceinline__ float softplus_device(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

__device__ __forceinline__ float sigmoid_device(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_device(float x)
{
    const float e = __expf(x);
    return x * e / (e + 1.0f);
}

} // namespace lsxcommon::qwen::dev