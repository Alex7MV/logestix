// Xing4.0-29B-A4B kernels (stages 5-6). Mirrors modeling_xing4_0.py:
// Xing4_0UnweightedRMSNorm / Xing4_0HyperConnection / Xing4_0TopkRouter.
//
// Numerics at kOff: exp/rsqrt resolve to the model-neutral glibc replicas
// (ExpfGlibc / RsqrtfGlibc) so the block matches the BF16 host reference, and
// every divide is IEEE __fdiv_rn (never __frcp_rn, whose reciprocal-then-
// multiply reassociation would drift from the host on the gate values that pick
// the expert set). The only reductions (the mHC sum-of-squares and the router
// dot products) are deterministic and atomic-free: they feed final_logits.

#include "lsxcommon/xing_moe_kernels.h"

#include "xing_moe_device_math.h"

#include <cuda_bf16.h>

#include <cmath>

namespace lsxcommon::xing {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxHcMult = 8;
constexpr int kMaxMix = (2 + kMaxHcMult) * kMaxHcMult;   // 80
constexpr int kMaxExperts = 256;
// Shared layout inside the mHC kernel: [0, hc_mult*hidden) streams,
// then kThreads reduction slots, then kMaxHcMult pre gates and kMaxMix comb
// slots. Offsets are compile-time so nothing aliases.
constexpr int kCombBase = kMaxHcMult;

__device__ __forceinline__ float XingSigmoid(float x)
{
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, device::XingExpfGlibc(-x)));
}

__global__ void xing_unweighted_rmsnorm4_kernel(
    const float *__restrict__ x, float *__restrict__ out, int hc_mult,
    int hidden, float eps)
{
    extern __shared__ float red[];
    const int n = hc_mult * hidden;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const float v = x[j];
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float inv =
        device::XingRsqrtfGlibc(__fadd_rn(__fdiv_rn(red[0], static_cast<float>(n)), eps));
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        out[j] = __fmul_rn(x[j], inv);
}

// One token, one block. hc_fn (24 x 14336 = 688 KiB per block) is streamed
// from global; the 4 streams themselves are staged in shared memory because the
// collapse and the mix both read them again.
__global__ void xing_hyper_conn_kernel(
    const float *__restrict__ streams,
    const __nv_bfloat16 *__restrict__ hc_fn,
    const __nv_bfloat16 *__restrict__ hc_base, const float *__restrict__ hc_scale,
    int hc_mult, int hidden, float eps, int sinkhorn_iters, float clamp_min,
    float clamp_max, float post_gain, float *__restrict__ collapsed,
    float *__restrict__ post, float *__restrict__ comb)
{
    extern __shared__ float smem[];
    float *sh = smem;                        // hc_mult * hidden
    float *red = smem + hc_mult * hidden;    // kThreads
    __shared__ float pre_gate[kMaxHcMult];
    __shared__ float comb_buf[kMaxMix];
    const int n = hc_mult * hidden;
    const int mix = (2 + hc_mult) * hc_mult;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const float v = streams[j];
        sh[j] = v;
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float inv =
        device::XingRsqrtfGlibc(__fadd_rn(__fdiv_rn(red[0], static_cast<float>(n)), eps));
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        sh[j] = __fmul_rn(sh[j], inv);
    __syncthreads();

    // hc_fn @ flat, split into pre(hc_mult) / post(hc_mult) / comb(hc_mult^2).
    // `red` is reused: the norm reduction it held was already folded into `inv`,
    // and each row needs a full BLOCK reduction (writing every thread's partial
    // to one slot would keep only whichever thread landed last).
    for (int k = 0; k < mix; ++k)
    {
        const __nv_bfloat16 *row = hc_fn + static_cast<size_t>(k) * n;
        float acc = 0.0f;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(row[j]), sh[j]));
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                red[threadIdx.x] =
                    __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) comb_buf[k] = red[0];
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int d = 0; d < hc_mult; ++d)
        {
            const float pre_b = __bfloat162float(hc_base[d]);
            const float post_b = __bfloat162float(hc_base[hc_mult + d]);
            pre_gate[d] = XingSigmoid(__fadd_rn(
                __fmul_rn(comb_buf[d], hc_scale[0]), pre_b));
            post[d] = __fmul_rn(post_gain,
                                XingSigmoid(__fadd_rn(
                                    __fmul_rn(comb_buf[hc_mult + d],
                                              hc_scale[1]),
                                    post_b)));
        }
        const int off = 2 * hc_mult;
        float mx = -INFINITY;
        for (int i = 0; i < hc_mult * hc_mult; ++i)
        {
            const float lg = fminf(
                fmaxf(__fadd_rn(__fmul_rn(comb_buf[off + i], hc_scale[2]),
                               __bfloat162float(hc_base[off + i])),
                      clamp_min),
                clamp_max);
            comb_buf[kCombBase + i] = lg;
            mx = fmaxf(mx, lg);
        }
        for (int i = 0; i < hc_mult * hc_mult; ++i)
            comb_buf[kCombBase + i] =
                device::XingExpfGlibc(comb_buf[kCombBase + i] - mx);
        for (int it = 0; it < sinkhorn_iters; ++it)
        {
            for (int r = 0; r < hc_mult; ++r)
            {
                float s = 0.0f;
                for (int c = 0; c < hc_mult; ++c)
                    s += comb_buf[kCombBase + r * hc_mult + c];
                s = __fdiv_rn(1.0f, __fadd_rn(s, eps));
                for (int c = 0; c < hc_mult; ++c)
                    comb_buf[kCombBase + r * hc_mult + c] *= s;
            }
            for (int c = 0; c < hc_mult; ++c)
            {
                float s = 0.0f;
                for (int r = 0; r < hc_mult; ++r)
                    s += comb_buf[kCombBase + r * hc_mult + c];
                s = __fdiv_rn(1.0f, __fadd_rn(s, eps));
                for (int r = 0; r < hc_mult; ++r)
                    comb_buf[kCombBase + r * hc_mult + c] *= s;
            }
        }
        for (int i = 0; i < hc_mult * hc_mult; ++i) comb[i] = comb_buf[kCombBase + i];
    }
    __syncthreads();

    // The collapse multiplies the RAW streams, not the normalised ones: the
    // unweighted RMSNorm only feeds hc_fn, while upstream collapses with
    // `pre * hidden_streams` (Xing4_0HyperConnection.forward). Reading `sh`
    // here would silently normalise twice — caught by the host-parity test.
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    {
        float acc = 0.0f;
        for (int d = 0; d < hc_mult; ++d)
            acc = __fadd_rn(acc, __fmul_rn(pre_gate[d], streams[d * hidden + j]));
        collapsed[j] = acc;
    }
}

__global__ void xing_mix_kernel(const float *__restrict__ streams,
                                const float *__restrict__ out,
                                const float *__restrict__ post,
                                const float *__restrict__ comb, int hc_mult,
                                int hidden, float *__restrict__ hidden_new)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    // Read the whole column into registers BEFORE the first write: the decode
    // loop mixes in place, and writing stream d while later iterations still
    // read stream d corrupts the residual (column isolation is not enough).
    float in[kMaxHcMult];
    for (int k = 0; k < hc_mult; ++k) in[k] = streams[k * hidden + c];
    const float o = out[c];
    for (int d = 0; d < hc_mult; ++d)
    {
        float acc = __fmul_rn(post[d], o);
        for (int k = 0; k < hc_mult; ++k)
            acc = __fadd_rn(acc, __fmul_rn(comb[d * hc_mult + k], in[k]));
        hidden_new[d * hidden + c] = acc;
    }
}

// One block: thread 0 owns the bias-corrected top-k (deterministic tie-break by
// lowest id) while every thread cooperates on the expert dot products.
__global__ void xing_router_kernel(const __nv_bfloat16 *__restrict__ weight,
                                   const float *__restrict__ bias,
                                   const __nv_bfloat16 *__restrict__ x,
                                   int num_experts, int hidden, int top_k,
                                   bool norm_topk_prob, float routed_scale,
                                   float *__restrict__ out_weights,
                                   int32_t *__restrict__ out_experts)
{
    __shared__ float score[kMaxExperts];
    __shared__ bool taken[kMaxExperts];
    __shared__ float w[16];
    __shared__ int ids[16];

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        const __nv_bfloat16 *row = weight + static_cast<size_t>(e) * hidden;
        float acc = 0.0f;
        for (int i = 0; i < hidden; ++i)
            acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(row[i]),
                                           __bfloat162float(x[i])));
        score[e] = XingSigmoid(acc);
        taken[e] = false;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float wsum = 0.0f;
        for (int k = 0; k < top_k; ++k)
        {
            int best = -1;
            float best_score = 0.0f;
            for (int e = 0; e < num_experts; ++e)
            {
                if (taken[e]) continue;
                // noaux_tc: the bias only steers the CHOICE, the weight stays
                // the raw sigmoid score.
                const float s = __fadd_rn(score[e], bias[e]);
                if (best < 0 || s > best_score)
                {
                    best = e;
                    best_score = s;
                }
            }
            taken[best] = true;
            ids[k] = best;
            w[k] = score[best];
            wsum = __fadd_rn(wsum, score[best]);
        }
        const float denom =
            norm_topk_prob ? __fdiv_rn(1.0f, __fadd_rn(wsum, 1e-20f)) : 1.0f;
        for (int k = 0; k < top_k; ++k)
        {
            out_weights[k] = __fmul_rn(__fmul_rn(w[k], denom), routed_scale);
            out_experts[k] = ids[k];
        }
        // Zero-fill the unused lanes so the caller's 16-wide buffers are fully
        // defined even though this family selects 4.
        for (int i = top_k; i < 16; ++i)
        {
            out_weights[i] = 0.0f;
            out_experts[i] = -1;
        }
    }
}

}  // namespace

cudaError_t LaunchXingUnweightedRmsNorm4(const float *x, float *out,
                                          int hc_mult, int hidden, float eps,
                                          cudaStream_t stream)
{
    if (hc_mult <= 0 || hc_mult > kMaxHcMult || hidden <= 0)
        return cudaErrorInvalidValue;
    const size_t smem = static_cast<size_t>(kThreads) * sizeof(float);
    xing_unweighted_rmsnorm4_kernel<<<1, kThreads, smem, stream>>>(x, out,
                                                                   hc_mult, hidden,
                                                                   eps);
    return cudaGetLastError();
}

cudaError_t LaunchXingHyperConnection(
    const float *streams, const __nv_bfloat16 *hc_fn,
    const __nv_bfloat16 *hc_base, const float *hc_scale, int hc_mult,
    int hidden, float eps, int sinkhorn_iters, float clamp_min, float clamp_max,
    float post_gain, float *collapsed, float *post, float *comb,
    cudaStream_t stream)
{
    if (hc_mult <= 0 || hc_mult > kMaxHcMult || hidden <= 0)
        return cudaErrorInvalidValue;
    if ((2 + hc_mult) * hc_mult > kMaxMix) return cudaErrorInvalidValue;
    const size_t smem =
        (static_cast<size_t>(hc_mult) * hidden + kThreads) * sizeof(float);
    // 4 x 3584 streams + the reduction slots is ~56 KiB, above the 48 KiB a
    // block gets by DEFAULT on sm_120. Without this opt-in the launch fails with
    // cudaErrorInvalidValue - which is exactly what the full-geometry decode
    // test hit while the small-geometry kernel test stayed green.
    if (smem > 48 * 1024)
    {
        static bool opted = false;
        if (!opted)
        {
            const cudaError_t a = cudaFuncSetAttribute(
                xing_hyper_conn_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem));
            if (a != cudaSuccess) return a;
            opted = true;
        }
    }
    xing_hyper_conn_kernel<<<1, kThreads, smem, stream>>>(
        streams, hc_fn, hc_base, hc_scale, hc_mult, hidden, eps, sinkhorn_iters,
        clamp_min, clamp_max, post_gain, collapsed, post, comb);
    return cudaGetLastError();
}

cudaError_t LaunchXingMix(const float *streams, const float *out,
                          const float *post, const float *comb, int hc_mult,
                          int hidden, float *hidden_new, cudaStream_t stream)
{
    if (hc_mult <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const int blocks = (hidden + kThreads - 1) / kThreads;
    xing_mix_kernel<<<blocks, kThreads, 0, stream>>>(streams, out, post, comb,
                                                     hc_mult, hidden, hidden_new);
    return cudaGetLastError();
}

cudaError_t LaunchXingRouter(const __nv_bfloat16 *weight, const float *bias,
                             const __nv_bfloat16 *x, int num_experts, int hidden,
                             int top_k, bool norm_topk_prob,
                             float routed_scale, float *out_weights,
                             int32_t *out_experts, cudaStream_t stream)
{
    if (num_experts <= 0 || num_experts > kMaxExperts || hidden <= 0 ||
        top_k <= 0 || top_k > 16)
        return cudaErrorInvalidValue;
    // A top_k larger than the expert count has no k-th candidate: the selection
    // loop would emit expert id -1 and the caller's resolve step would reject
    // the whole layer. Clamp instead, and zero the lanes we do not fill.
    const int k = std::min(top_k, num_experts);
    xing_router_kernel<<<1, kThreads, 0, stream>>>(
        weight, bias, x, num_experts, hidden, k, norm_topk_prob,
        routed_scale, out_weights, out_experts);
    return cudaGetLastError();
}

}  // namespace lsxcommon::xing