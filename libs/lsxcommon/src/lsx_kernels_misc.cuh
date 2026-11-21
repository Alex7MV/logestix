// lsx_kernels_misc.cuh — small neutral kernels shared across families
// (dedup audit 2026-11-20, residual groups). Every body was byte-identical
// (modulo the family name prefix) in its former copies; all arithmetic uses
// explicit *_rn intrinsics or pure copies, so the result is independent of
// the including TU's -fmad setting.
//
// Templated on a dummy Tag (default void) for weak linkage: a plain
// __global__ in a header multiply defines its host stub when two TUs link
// into one binary (the RmsNormTreeKernel pattern). Call sites use <>.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon {
namespace kernels {

namespace detail {
// Deterministic block-wide sum (fixed tree order) — the inkling pair's
// BlockSum, byte-identical in both TUs.
__device__ __forceinline__ float MiscBlockSum(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] =
                __fadd_rn(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    return scratch[0];
}
} // namespace detail

// FP8 E4M3 tile-128 dequant to BF16 (gigachat/ling/mimo dequant copies).
template <typename Tag = void>
__global__ void Fp8Tile128ToBf16Kernel(const uint8_t *__restrict__ fp8,
                                       const float *__restrict__ scale,
                                       __nv_bfloat16 *__restrict__ out,
                                       long long numel, int cols,
                                       int scale_cols)
{
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                        threadIdx.x;
    if (i >= numel) return;
    const int r = static_cast<int>(i / cols);
    const int c = static_cast<int>(i % cols);
    const int tile = (r >> 7) * scale_cols + (c >> 7);
    const __half h = __half(__nv_cvt_fp8_to_halfraw(fp8[i], __NV_E4M3));
    out[i] = __float2bfloat16(__half2float(h) * scale[tile]);
}

// Multi-token embedding gather (beam/flint/hy embed copies): grid.y = tokens,
// grid.x strides the hidden dim.
template <typename Tag = void>
__global__ void EmbedGatherRowsKernel(const __half *__restrict__ table,
                                      const int32_t *__restrict__ ids,
                                      __half *__restrict__ out, int hidden)
{
    const int t = blockIdx.y;
    const int id = ids[t];
    const int64_t src = static_cast<int64_t>(id) * hidden;
    const int64_t dst = static_cast<int64_t>(t) * hidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += blockDim.x * gridDim.x)
        out[dst + i] = table[src + i];
}

// acc += add (flint/inkling×2 add_f32 copies).
template <typename Tag = void>
__global__ void AddF32Kernel(float *__restrict__ acc,
                             const float *__restrict__ add, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    acc[i] = __fadd_rn(acc[i], add[i]);
}

// Learned relative-position bias (inkling pair's rel_bias copies).
template <typename Tag = void>
__global__ void RelBiasKernel(const float *__restrict__ rel_states,
                              const float *__restrict__ proj,
                              float *__restrict__ bias, int d_rel,
                              int rel_extent, int q_pos, int kv_len)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float *r = rel_states + static_cast<int64_t>(h) * d_rel;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        const int d = q_pos - j;
        float val = 0.0f;
        if (d >= 0 && d < rel_extent) {
            for (int c = 0; c < d_rel; ++c)
                val = __fadd_rn(val, __fmul_rn(r[c],
                                               proj[static_cast<int64_t>(c) *
                                                        rel_extent + d]));
        }
        bias[static_cast<int64_t>(h) * kv_len + j] = val;
    }
}

// RMSNorm with F32 input, FP16 weight/output (inkling pair's norm_f32in).
// Launch with blockDim.x floats of dynamic shared memory.
template <typename Tag = void>
__global__ void NormF32InKernel(const float *__restrict__ x,
                                const __half *__restrict__ w,
                                __half *__restrict__ out, int hidden,
                                float eps)
{
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float *xr = x + static_cast<int64_t>(row) * hidden;
    __half *orow = out + static_cast<int64_t>(row) * hidden;
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
        ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
    ss = detail::MiscBlockSum(ss, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += blockDim.x)
        orow[i] = __float2half(__fmul_rn(__fmul_rn(xr[i], inv),
                                         __half2float(w[i])));
}


// ── inkling-pair residual copies (byte-identical, *_rn arithmetic) ──────

template <typename Tag = void>
__global__ void LogScaleKernel(float *__restrict__ q, int64_t nq,
                               float *__restrict__ rel, int64_t nrel,
                               float tau)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < nq) q[i] = __fmul_rn(q[i], tau);
    if (i < nrel) rel[i] = __fmul_rn(rel[i], tau);
}

template <typename Tag = void>
__global__ void ScaleToHalfKernel(const float *__restrict__ src,
                                  __half *__restrict__ dst, int64_t n,
                                  float scale)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__fmul_rn(src[i], scale));
}

// Single-token embedding row gather (inkling pair): one block, id on device.
template <typename Tag = void>
__global__ void EmbedGatherRowKernel(const __half *__restrict__ table,
                                     const int32_t *__restrict__ d_id,
                                     __half *__restrict__ out, int hidden)
{
    const int64_t row = d_id[0];
    const __half *src = table + row * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[i] = src[i];
}

// repeat hidden -> hc streams (glm-flash / qwen-exp repeat_hc copies): pure
// FP16 replication, no arithmetic.
template <typename Tag = void>
__global__ void RepeatHcKernel(const __half *__restrict__ in, int H,
                               int hc, __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    const __half v = in[i];
    for (int c = 0; c < hc; ++c) out[c * H + i] = v;
}

} // namespace kernels
} // namespace lsxcommon