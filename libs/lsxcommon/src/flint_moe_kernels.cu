// Recursive Flint numeric kernels: plain RMSNorm (row and per-head), full
// rotary, the sigmoid top-k router (no bias, renormalized, no routed scale) and
// the routed accumulation / FP32 shared combine. Model-neutral primitives
// (RMSNorm tree, SwiGLU, GEMV) come from the shared layers; only the family's
// own arithmetic lives here.
#include "lsxcommon/flint_moe_kernels.h"

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_norm.cuh"

#include <algorithm>
#include <cmath>

namespace lsxcommon::flint {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float SigmoidFlint(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void flint_router_kernel(const float *__restrict__ w,
                                    const __half *__restrict__ x, int E, int K,
                                    int H, FlintRoutedSlot *__restrict__ out,
                                    int32_t *__restrict__ out_ids)
{
    extern __shared__ float smem[];
    float *score = smem;          // [E]
    __half *xs = reinterpret_cast<__half *>(score + E);   // [H]

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const __half *xt = x + static_cast<int64_t>(t) * H;
    for (int k = tid; k < H; k += blockDim.x) xs[k] = xt[k];
    __syncthreads();

    // One warp per expert: 32 lanes stride the hidden dim and reduce with shfl.
    for (int e = warp; e < E; e += warps) {
        const float *we = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = lane; k < H; k += 32) acc += we[k] * __half2float(xs[k]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) score[e] = SigmoidFlint(acc);
    }
    __syncthreads();

    if (tid == 0) {
        bool taken[512];
        int sel[32];
        for (int i = 0; i < E && i < 512; ++i) taken[i] = false;
        for (int k = 0; k < K && k < 32; ++k) {
            int best = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < E; ++e) {
                if (e < 512 && taken[e]) continue;
                if (score[e] > best_v) {   // ties: lower id wins
                    best_v = score[e];
                    best = e;
                }
            }
            if (best < 0) break;
            taken[best] = true;
            sel[k] = best;
        }
        for (int i = 1; i < K; ++i) {
            const int v = sel[i];
            int j = i - 1;
            while (j >= 0 && sel[j] > v) {
                sel[j + 1] = sel[j];
                --j;
            }
            sel[j + 1] = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += score[sel[k]];
        FlintRoutedSlot *dst = out + static_cast<int64_t>(t) * K;
        for (int k = 0; k < K; ++k) {
            const float norm = sum > 0.0f ? score[sel[k]] / (sum + 1e-20f) : 0.0f;
            dst[k].eid = sel[k];
            dst[k].coeff = norm;
            if (out_ids) out_ids[k] = sel[k];
        }
    }
}

// Full rotary: pair (j, j + half) with the HF rotate_half convention.
__global__ void flint_rope_kernel(__half *__restrict__ x, int head_dim,
                                  float theta, int pos)
{
    const int half = head_dim / 2;
    const int64_t row = blockIdx.x;   // (token, head)
    const int j = threadIdx.x;
    if (j >= half) return;
    const int64_t base = row * head_dim;
    const float inv_freq =
        1.0f / powf(theta, static_cast<float>(j) / static_cast<float>(half));
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float x1 = __half2float(x[base + j]);
    const float x2 = __half2float(x[base + j + half]);
    x[base + j] = __float2half(x1 * c - x2 * s);
    x[base + j + half] = __float2half(x1 * s + x2 * c);
}

__global__ void flint_embed_kernel(const __half *__restrict__ table,
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

__global__ void flint_expert_accumulate_kernel(
    float *__restrict__ acc, const __half *__restrict__ y,
    const FlintRoutedSlot *__restrict__ slots, int count, int hidden)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float v = 0.0f;
    for (int k = 0; k < count; ++k)
        v += slots[k].coeff *
             __half2float(y[static_cast<int64_t>(k) * hidden + i]);
    acc[i] = v;
}

__global__ void flint_moe_combine_kernel(const float *__restrict__ acc,
                                         const __half *__restrict__ shared,
                                         __half *__restrict__ out, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(acc[i] + __half2float(shared[i]));
}

} // namespace

cudaError_t RunFlintRouter(cudaStream_t stream, const float *router_w,
                           const __half *x, int seq, int E, int K, int H,
                           FlintRoutedSlot *d_slots, int32_t *d_ids)
{
    if (seq <= 0 || E <= 0 || K <= 0 || K > E || H <= 0)
        return cudaErrorInvalidValue;
    if (E > 512 || K > 32) return cudaErrorInvalidValue;
    const size_t shmem =
        (kThreads + static_cast<size_t>(E)) * sizeof(float) +
        static_cast<size_t>(H) * sizeof(__half);
    flint_router_kernel<<<seq, kThreads, shmem, stream>>>(
        router_w, x, E, K, H, d_slots, d_ids);
    return cudaGetLastError();
}

cudaError_t RunFlintRope(cudaStream_t stream, __half *q, __half *k, int seq,
                         int q_heads, int k_heads, int head_dim, float theta,
                         int pos_base)
{
    if (seq <= 0 || head_dim <= 0 || (head_dim & 1) != 0)
        return cudaErrorInvalidValue;
    const int half = head_dim / 2;
    for (int t = 0; t < seq; ++t) {
        const int pos = pos_base + t;
        if (q && q_heads > 0)
            flint_rope_kernel<<<q_heads, half, 0, stream>>>(
                q + static_cast<int64_t>(t) * q_heads * head_dim, head_dim,
                theta, pos);
        if (k && k_heads > 0)
            flint_rope_kernel<<<k_heads, half, 0, stream>>>(
                k + static_cast<int64_t>(t) * k_heads * head_dim, head_dim,
                theta, pos);
    }
    return cudaGetLastError();
}

cudaError_t RunFlintNorm(cudaStream_t stream, const __half *x, const __half *w,
                         __half *out, int rows, int hidden, float eps)
{
    if (rows <= 0 || hidden <= 0) return cudaSuccess;
    const int threads = 256;
    const size_t shmem =
        static_cast<size_t>(threads) * sizeof(float) +
        static_cast<size_t>(std::min<int64_t>(hidden * sizeof(__half),
                                              48 * 1024));
    lsxcommon::kernels::RmsNormTreeKernel<__half,
                                          lsxcommon::kernels::NormWeight::Direct>
        <<<rows, threads, shmem, stream>>>(x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunFlintHeadNorm(cudaStream_t stream, const __half *x,
                             const __half *w, __half *out, int rows, int heads,
                             int head_dim, float eps)
{
    if (rows <= 0 || heads <= 0 || head_dim <= 0) return cudaSuccess;
    int threads = 1;
    while (threads < head_dim && threads < 512) threads <<= 1;
    const size_t shmem =
        static_cast<size_t>(threads) * sizeof(float) +
        static_cast<size_t>(std::min<int64_t>(head_dim * sizeof(__half),
                                              48 * 1024));
    lsxcommon::kernels::RmsNormTreeKernel<__half,
                                          lsxcommon::kernels::NormWeight::Direct>
        <<<rows * heads, threads, shmem, stream>>>(x, w, out, rows * heads,
                                                   head_dim, eps);
    return cudaGetLastError();
}

cudaError_t RunFlintEmbed(cudaStream_t stream, const __half *table,
                          const int32_t *ids, __half *out, int seq, int hidden)
{
    if (seq <= 0 || hidden <= 0) return cudaSuccess;
    const int bx = (hidden + kThreads - 1) / kThreads;
    dim3 grid(bx, seq);
    flint_embed_kernel<<<grid, kThreads, 0, stream>>>(table, ids, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunFlintExpertAccumulate(cudaStream_t stream, float *acc,
                                     const __half *y,
                                     const FlintRoutedSlot *slots, int count,
                                     int hidden)
{
    if (count <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const int blocks = (hidden + kThreads - 1) / kThreads;
    flint_expert_accumulate_kernel<<<blocks, kThreads, 0, stream>>>(
        acc, y, slots, count, hidden);
    return cudaGetLastError();
}

cudaError_t RunFlintMoECombine(cudaStream_t stream, const float *acc,
                               const __half *shared, __half *out, int rows,
                               int hidden)
{
    if (rows <= 0 || hidden <= 0) return cudaSuccess;
    const int64_t n = static_cast<int64_t>(rows) * hidden;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    flint_moe_combine_kernel<<<blocks, kThreads, 0, stream>>>(acc, shared, out,
                                                              n);
    return cudaGetLastError();
}

} // namespace lsxcommon::flint
