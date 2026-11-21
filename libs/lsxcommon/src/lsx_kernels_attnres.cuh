// lsx_kernels_attnres.cuh — neutral AttnRes depth mixer (dedup phase 3).
//
// One implementation shared by alice (MaxSrc=64) and kimi-k3 (MaxSrc=16):
// per-source RMS-dot score, softmax over the depth sources, weighted
// combine. The family launchers keep their own n_src guards (alice >64,
// kimi-k3 <=0 || >16) — the neutral kernel keeps no guard.
//
// Moved from alice_moe_kernels.cu / kimi_moe_k3_kernels.cu (the bodies were
// identical modulo the shared logits array size and whitespace). Two
// deliberate, documented deltas:
//   1. `logits[64]`/`logits[16]` -> `logits[MaxSrc]` (template parameter;
//      each family instantiates its own size, smem layout preserved).
//   2. Every +,*,/ is an explicit __fadd_rn/__fmul_rn/__fdiv_rn intrinsic:
//      the family TUs are -fmad=false, but a header can be included by a
//      TU with contraction enabled — the arithmetic must be flag-
//      independent (same rule as lsx_kernels_*.cuh). rsqrtf/__expf/fmaxf
//      are already intrinsics and stay untouched.
// The block reduction carries the trailing __syncthreads() of the kimi-k3
// variant: it only adds a barrier before the caller's next phase, never a
// value change (alice's next use of scratch is behind its own barrier).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace kernels {
namespace detail {

__device__ __forceinline__ float AttnResBlockSum(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] =
                __fadd_rn(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

} // namespace detail

// Launch <<<1, 256>>>; the caller guards n_src (family caps stay family-side).
template <int MaxSrc>
__global__ void AttnResDepthMixKernel(const __half *__restrict__ sources,
                                      int n_src,
                                      const __half *__restrict__ norm_w,
                                      const __half *__restrict__ proj_w,
                                      __half *__restrict__ out, int hidden,
                                      float eps)
{
    __shared__ float scratch[256];
    __shared__ float logits[MaxSrc];
    for (int s = 0; s < n_src; ++s) {
        const __half *p = sources + static_cast<size_t>(s) * hidden;
        float sq = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __half2float(p[i]);
            sq = __fadd_rn(sq, __fmul_rn(v, v));
        }
        const float rstd = rsqrtf(__fadd_rn(
            __fdiv_rn(detail::AttnResBlockSum(sq, scratch),
                      static_cast<float>(hidden)),
            eps));
        float dot = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __fmul_rn(__fmul_rn(__half2float(p[i]), rstd),
                                      __half2float(norm_w[i]));
            dot = __fadd_rn(dot, __fmul_rn(v, __half2float(proj_w[i])));
        }
        const float total = detail::AttnResBlockSum(dot, scratch);
        if (threadIdx.x == 0) logits[s] = total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = logits[0];
        for (int s = 1; s < n_src; ++s) mx = fmaxf(mx, logits[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_src; ++s) {
            logits[s] = __expf(__fadd_rn(logits[s], -mx));
            sum = __fadd_rn(sum, logits[s]);
        }
        for (int s = 0; s < n_src; ++s)
            logits[s] = __fdiv_rn(logits[s], sum);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_src; ++s)
            acc = __fadd_rn(
                acc, __fmul_rn(logits[s],
                               __half2float(sources[static_cast<size_t>(s) *
                                                        hidden + i])));
        out[i] = __float2half(acc);
    }
}

} // namespace kernels
} // namespace lsxcommon