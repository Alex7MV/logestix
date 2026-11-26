// GigaChat3.5 elementwise gating kernels (Plan 2a): swiglu_limit MLP
// activation, gated-attention sigmoid, and the sigmoid top-k router.
#include "gigachat_moe_kernels.h"

#include "gigachat_moe_device_math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace lsxcommon::gigachat {
namespace {

constexpr int kMaxTopK = 16;

__global__ void swiglu_limit_kernel(const float *__restrict__ g,
                                    const float *__restrict__ u, float limit,
                                    int n, float *__restrict__ mid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mid[i] = dev::SwigluLimitElement(g[i], u[i], limit);
}

__global__ void sigmoid_gate_kernel(float *__restrict__ y,
                                    const float *__restrict__ gate, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = dev::SigmoidGateElement(y[i], gate[i]);
}

constexpr int kMaxRouterExperts = 1024;

// Module scratch for the two-kernel router. Single-context, single-stream
// engine (numeric contract §7: no batch composition to be invariant to), so
// no per-call allocation and no cross-stream race.
__device__ float g_router_score[kMaxRouterExperts];
__device__ float g_router_choice[kMaxRouterExperts];

// Device addresses of the two scratch arrays.
//
// Taking the address of a __device__ variable in host code does NOT give its
// device address on this toolchain (MSVC host compiler + nvcc 13.4): the
// expression evaluates to a HOST pointer, 0x7ff6... on Windows, and the
// kernel then faults on its first write. Verified directly -- the address
// compute-sanitizer reported as the fault target matched what host code saw,
// while cudaGetSymbolAddress returned the real 0x1307... device address.
// Verified too that this is independent of the variable's linkage: the same
// happens with the arrays at namespace scope rather than in an anonymous
// namespace.
//
// So resolve the device address explicitly, once, and reuse it. This is also
// what the runtime requires before the symbol can be used in a launch.
namespace {

cudaError_t RouterScratchDev(float **score, float **choice)
{
    static float *cached_score = nullptr;
    static float *cached_choice = nullptr;
    if (!cached_score || !cached_choice) {
        void *s = nullptr, *c = nullptr;
        const cudaError_t e1 = cudaGetSymbolAddress(&s, g_router_score);
        const cudaError_t e2 = cudaGetSymbolAddress(&c, g_router_choice);
        if (e1 != cudaSuccess) return e1;
        if (e2 != cudaSuccess) return e2;
        cached_score = static_cast<float *>(s);
        cached_choice = static_cast<float *>(c);
    }
    *score = cached_score;
    *choice = cached_choice;
    return cudaSuccess;
}

} // namespace

// Multi-SM router gate: one block per expert, block-strided dot over hidden,
// fixed warp/block reduction tree. score = sigmoid(dot); choice = score+bias.
__global__ void router_score_kernel(const float *__restrict__ x,
                                    const __nv_bfloat16 *__restrict__ w,
                                    const float *__restrict__ bias, int E, int H,
                                    float *__restrict__ score,
                                    float *__restrict__ choice)
{
    const int e = blockIdx.x;
    const __nv_bfloat16 *wr = w + static_cast<size_t>(e) * H;
    float a = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        a += __bfloat162float(wr[i]) * x[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a += __shfl_down_sync(0xffffffffu, a, off);
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5,
              nw = (blockDim.x + 31) >> 5;
    if (lane == 0) red[warp] = a;
    __syncthreads();
    if (warp == 0)
    {
        a = (lane < nw) ? red[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0)
        {
            const float s = dev::Sigmoid(a);
            score[e] = s;
            choice[e] = s + (bias ? bias[e] : 0.0f);
        }
    }
}

// Block-parallel deterministic top-k over the E scratch scores: `top_k`
// rounds of a masked argmax (ties -> lower index) with a fixed reduction
// tree. Replaces the former single-thread O(top_k^2 * E) scan.
__global__ void router_select_kernel(const float *__restrict__ score,
                                     const float *__restrict__ choice_in,
                                     int n, int top_k, float scale,
                                     int *__restrict__ out_idx,
                                     float *__restrict__ out_w)
{
    extern __shared__ float smem[];
    float *sc = smem;                // [n] scores
    float *ch = smem + n;            // [n] choice
    float *r_val = smem + 2 * n;     // [blockDim]
    int *r_idx = reinterpret_cast<int *>(r_val + blockDim.x);
    __shared__ int s_used[kMaxRouterExperts];
    __shared__ int s_idx[kMaxTopK];
    __shared__ float s_w[kMaxTopK];
    const int t = threadIdx.x;
    for (int i = t; i < n; i += blockDim.x)
    {
        sc[i] = score[i];
        ch[i] = choice_in[i];
        s_used[i] = 0;
    }
    __syncthreads();
    for (int k = 0; k < top_k; ++k)
    {
        int lbest = -1;
        float lbv = -3.4e38f;
        for (int i = t; i < n; i += blockDim.x)
        {
            if (s_used[i]) continue;
            const float v = ch[i];
            if (lbest < 0 || v > lbv || (v == lbv && i < lbest))
            {
                lbv = v;
                lbest = i;
            }
        }
        r_val[t] = lbv;
        r_idx[t] = lbest;
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1)
        {
            if (t < off)
            {
                const float v2 = r_val[t + off];
                const int i2 = r_idx[t + off];
                const float v1 = r_val[t];
                const int i1 = r_idx[t];
                if (i2 >= 0 && (i1 < 0 || v2 > v1 || (v2 == v1 && i2 < i1)))
                {
                    r_val[t] = v2;
                    r_idx[t] = i2;
                }
            }
            __syncthreads();
        }
        if (t == 0 && r_idx[0] >= 0)
        {
            s_used[r_idx[0]] = 1;
            s_idx[k] = r_idx[0];
            s_w[k] = sc[r_idx[0]];
        }
        __syncthreads();
    }
    if (t == 0)
    {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) sum += s_w[k];
        const float denom = sum + 1e-20f;
        for (int k = 0; k < top_k; ++k)
        {
            out_idx[k] = s_idx[k];
            out_w[k] = s_w[k] / denom * scale;
        }
    }
}

// Fallback for num_experts > kMaxRouterExperts: one block, thread-0
// sequential selection (lowest index wins ties).
__global__ void router_topk_single_kernel(const float *__restrict__ x,
                                          const __nv_bfloat16 *__restrict__ w,
                                          const float *__restrict__ bias, int E,
                                          int H, int top_k, float scale,
                                          int *__restrict__ out_idx,
                                          float *__restrict__ out_w)
{
    extern __shared__ float smem[];  // scores[E] + choice[E]
    float *scores = smem;
    float *choice = smem + E;

    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        const __nv_bfloat16 *wr = w + static_cast<size_t>(e) * H;
        float a = 0.0f;
        int i = 0;
        // 8-way unroll exposes independent weight loads to the scheduler while
        // the single-accumulator adds keep the exact left-to-right order
        // (bit-identical to the former scalar loop under -fmad=false).
#pragma unroll 8
        for (; i + 8 <= H; i += 8)
        {
            const float v0 = __bfloat162float(wr[i + 0]);
            const float v1 = __bfloat162float(wr[i + 1]);
            const float v2 = __bfloat162float(wr[i + 2]);
            const float v3 = __bfloat162float(wr[i + 3]);
            const float v4 = __bfloat162float(wr[i + 4]);
            const float v5 = __bfloat162float(wr[i + 5]);
            const float v6 = __bfloat162float(wr[i + 6]);
            const float v7 = __bfloat162float(wr[i + 7]);
            a += v0 * x[i + 0];
            a += v1 * x[i + 1];
            a += v2 * x[i + 2];
            a += v3 * x[i + 3];
            a += v4 * x[i + 4];
            a += v5 * x[i + 5];
            a += v6 * x[i + 6];
            a += v7 * x[i + 7];
        }
        for (; i < H; ++i) a += __bfloat162float(wr[i]) * x[i];
        const float s = dev::Sigmoid(a);
        scores[e] = s;
        choice[e] = s + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int idx[kMaxTopK];
        for (int k = 0; k < top_k; ++k)
        {
            int best = -1;
            float bv = -3.4e38f;
            for (int e = 0; e < E; ++e)
            {
                bool used = false;
                for (int j = 0; j < k; ++j)
                    if (idx[j] == e)
                    {
                        used = true;
                        break;
                    }
                if (!used && choice[e] > bv)
                {
                    bv = choice[e];
                    best = e;
                }
            }
            idx[k] = best;
        }
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) sum += scores[idx[k]];
        const float denom = sum + 1e-20f;
        for (int k = 0; k < top_k; ++k)
        {
            out_idx[k] = idx[k];
            out_w[k] = scores[idx[k]] / denom * scale;
        }
    }
}

int GridFor(int n, int threads) { return (n + threads - 1) / threads; }

// Final logit soft cap (GigaChat4 `final_logit_softcapping`). One pass, in
// place, F32: the value the sampler and final_logits see is post-cap.
__global__ void softcap_kernel(float *__restrict__ x, int n, float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = dev::LogitSoftcapElement(x[i], cap);
}

} // namespace

cudaError_t RunSwigluLimit(const float *gate, const float *up, float limit,
                           int n, float *mid, cudaStream_t stream)
{
    if (!gate || !up || !mid || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    const int threads = 256;
    swiglu_limit_kernel<<<GridFor(n, threads), threads, 0, stream>>>(
        gate, up, limit, n, mid);
    return cudaGetLastError();
}

cudaError_t RunGigaChatSoftcap(float *logits, int n, float cap,
                               cudaStream_t stream)
{
    if (!logits || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    if (cap <= 0.0f) return cudaSuccess;  // no cap: identity
    const int threads = 256;
    softcap_kernel<<<GridFor(n, threads), threads, 0, stream>>>(logits, n, cap);
    return cudaGetLastError();
}

cudaError_t RunSigmoidGate(float *y, const float *gate, int n,
                           cudaStream_t stream)
{
    if (!y || !gate || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    const int threads = 256;
    sigmoid_gate_kernel<<<GridFor(n, threads), threads, 0, stream>>>(y, gate, n);
    return cudaGetLastError();
}

cudaError_t RunGigaChatRouterTopK(const float *x, const uint8_t *w,
                                  const float *bias, int num_experts, int hidden,
                                  int top_k, float routed_scaling_factor,
                                  int *out_idx, float *out_weights,
                                  cudaStream_t stream)
{
    if (!x || !w || !out_idx || !out_weights || num_experts <= 0 || hidden <= 0 ||
        top_k <= 0 || top_k > kMaxTopK)
        return cudaErrorInvalidValue;
    const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(w);
    float *dev_score = nullptr, *dev_choice = nullptr;
    const cudaError_t sym = RouterScratchDev(&dev_score, &dev_choice);
    if (sym != cudaSuccess) return sym;
    if (num_experts <= kMaxRouterExperts)
    {
        // Multi-SM gate + block-parallel selection (was a single-SM scan).
        constexpr int kBlock = 256;
        router_score_kernel<<<num_experts, kBlock, 0, stream>>>(
            x, wb, bias, num_experts, hidden, dev_score, dev_choice);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        const size_t smem =
            (static_cast<size_t>(2) * num_experts + 2 * kBlock) * sizeof(float);
        router_select_kernel<<<1, kBlock, smem, stream>>>(
            dev_score, dev_choice, num_experts, top_k,
            routed_scaling_factor, out_idx, out_weights);
        return cudaGetLastError();
    }
    int threads = num_experts < 256 ? num_experts : 256;
    if (threads < 1) threads = 1;
    const size_t smem = static_cast<size_t>(2) * num_experts * sizeof(float);
    router_topk_single_kernel<<<1, threads, smem, stream>>>(
        x, wb, bias, num_experts, hidden, top_k, routed_scaling_factor, out_idx,
        out_weights);
    return cudaGetLastError();
}

} // namespace lsxcommon::gigachat
