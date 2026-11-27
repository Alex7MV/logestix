// Beam-1.0 numeric kernels. Model-neutral primitives (RMSNorm tree, NVFP4
// GEMV, the routed-expert FFN) come from the shared layers; this TU owns only
// the family's arithmetic: per-head q/k norm, rotate-half RoPE, the NoPE SSMax
// query scale, the gated sliding-window decode attention, the group-limited
// sigmoid router, the FP32 SwiGLU product, the depth-scaled residual add, the
// embedding lookup and the final-logit softcap.
#include "lsxcommon/beam_moe_kernels.h"
#include "lsx_kernels_misc.cuh"

#include "beam_moe_constants.h"

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_norm.cuh"

#include <algorithm>
#include <cmath>

namespace lsxcommon::beam {
using namespace frozen;
namespace {

constexpr int kThreads = 256;
constexpr int kMaxHeadDim = 256;

__device__ __forceinline__ float SigmoidBeam(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// One warp per query head. Lane d holds head_dim/32 consecutive floats, so the
// warp reduction is deterministic (fixed shfl tree) and needs no block sync.
__global__ void beam_attn_decode_kernel(const __half *__restrict__ q,
                                        const __half *__restrict__ kc,
                                        const __half *__restrict__ vc,
                                        const __half *__restrict__ gate,
                                        const __half *__restrict__ gate_bias,
                                        __half *__restrict__ out, int NH,
                                        int NKV, int HD, int pos, int start,
                                        float scale)
{
    const int warps = blockDim.x >> 5;
    const int h = blockIdx.x * warps + (threadIdx.x >> 5);
    if (h >= NH) return;
    const int lane = threadIdx.x & 31;
    const int per = HD / 32;   // head_dim is a multiple of 32 in this family
    const int64_t row_stride = static_cast<int64_t>(NKV) * HD;
    const int kvh = h / (NH / NKV);

    float qv[8];
    for (int i = 0; i < per; ++i)
        qv[i] = __half2float(q[static_cast<int64_t>(h) * HD + lane * per + i]);

    // Pass 1: row max.
    float mx = -INFINITY;
    for (int row = start; row <= pos; ++row) {
        const __half *kr = kc + static_cast<int64_t>(row) * row_stride + kvh * HD;
        float p = 0.0f;
        for (int i = 0; i < per; ++i)
            p += qv[i] * __half2float(kr[lane * per + i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p += __shfl_down_sync(0xffffffffu, p, off);
        p = __shfl_sync(0xffffffffu, p, 0) * scale;
        mx = fmaxf(mx, p);
    }
    // Pass 2: denominator.
    float sum = 0.0f;
    for (int row = start; row <= pos; ++row) {
        const __half *kr = kc + static_cast<int64_t>(row) * row_stride + kvh * HD;
        float p = 0.0f;
        for (int i = 0; i < per; ++i)
            p += qv[i] * __half2float(kr[lane * per + i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p += __shfl_down_sync(0xffffffffu, p, off);
        p = __shfl_sync(0xffffffffu, p, 0) * scale;
        sum += expf(p - mx);
    }
    // Pass 3: weighted value.
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int row = start; row <= pos; ++row) {
        const __half *kr = kc + static_cast<int64_t>(row) * row_stride + kvh * HD;
        const __half *vr = vc + static_cast<int64_t>(row) * row_stride + kvh * HD;
        float p = 0.0f;
        for (int i = 0; i < per; ++i)
            p += qv[i] * __half2float(kr[lane * per + i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p += __shfl_down_sync(0xffffffffu, p, off);
        p = __shfl_sync(0xffffffffu, p, 0) * scale;
        const float w = expf(p - mx);
        for (int i = 0; i < per; ++i)
            acc[i] += w * __half2float(vr[lane * per + i]);
    }
    const float inv = 1.0f / sum;
    const float gb = __half2float(gate_bias[h]);
    const int64_t base = static_cast<int64_t>(h) * HD + lane * per;
    for (int i = 0; i < per; ++i) {
        const float gv = __half2float(gate[base + i]);
        out[base + i] = __float2half(acc[i] * inv * SigmoidBeam(gv + gb));
    }
}

__global__ void beam_rope_kernel(__half *__restrict__ x, int head_dim,
                                 float theta, int pos)
{
    const int half = head_dim / 2;
    const int64_t row = blockIdx.x;   // (token, head)
    const int j = threadIdx.x;
    if (j >= half) return;
    const int64_t base = row * head_dim;
    const float inv =
        1.0f / powf(theta, static_cast<float>(j) / static_cast<float>(half));
    const float ang = static_cast<float>(pos) * inv;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float x1 = __half2float(x[base + j]);
    const float x2 = __half2float(x[base + j + half]);
    x[base + j] = __float2half(x1 * c - x2 * s);
    x[base + j + half] = __float2half(x1 * s + x2 * c);
}

__global__ void beam_ssmax_kernel(__half *__restrict__ q, int64_t n, int pos)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float s = expf(0.1f * log1pf((static_cast<float>(pos) + 1.0f) /
                                       kSsmBase));
    q[i] = __float2half(__half2float(q[i]) * s);
}

__global__ void beam_router_kernel(const float *__restrict__ w,
                                   const float *__restrict__ bias,
                                   const __half *__restrict__ x, int E, int K,
                                   int H, int G, int TG, float routed_scale,
                                   BeamRoutedSlot *__restrict__ out)
{
    extern __shared__ float smem[];
    float *score = smem;                                  // [E] (choice values)
    float *plain = score + E;                             // [E]
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const __half *xt = x + static_cast<int64_t>(t) * H;

    // One warp per expert: 32 lanes stride the hidden dim and reduce with shfl.
    for (int e = warp; e < E; e += warps) {
        const float *we = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = lane; k < H; k += 32)
            acc += we[k] * __half2float(xt[k]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            const float s = SigmoidBeam(acc);
            plain[e] = s;
            score[e] = s + bias[e];
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int per = E / G;
        bool taken[512];
        float gscore[64];
        int gorder[64];
        bool gkeep[64];
        for (int i = 0; i < E && i < 512; ++i) taken[i] = false;
        for (int g = 0; g < G && g < 64; ++g) {
            float a = -INFINITY, b = -INFINITY;
            for (int e = g * per; e < (g + 1) * per && e < E; ++e) {
                const float v = score[e];
                if (v > a) {
                    b = a;
                    a = v;
                } else if (v > b) {
                    b = v;
                }
            }
            gscore[g] = a + b;
            gorder[g] = g;
            gkeep[g] = false;
        }
        // Stable insertion sort of the groups by descending score (lower group
        // id first on ties).
        for (int i = 1; i < G && i < 64; ++i) {
            const int v = gorder[i];
            int j = i - 1;
            while (j >= 0 && (gscore[gorder[j]] < gscore[v] ||
                              (gscore[gorder[j]] == gscore[v] &&
                               gorder[j] > v))) {
                gorder[j + 1] = gorder[j];
                --j;
            }
            gorder[j + 1] = v;
        }
        for (int i = 0; i < TG && i < G && i < 64; ++i) gkeep[gorder[i]] = true;

        int sel[32];
        int nsel = 0;
        for (int k = 0; k < K && k < 32; ++k) {
            int best = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < E && e < 512; ++e) {
                if (taken[e] || !gkeep[e / per]) continue;
                if (best < 0 || score[e] > best_v) {
                    best_v = score[e];
                    best = e;
                }
            }
            if (best < 0) break;
            taken[best] = true;
            sel[nsel++] = best;
        }
        // Ascending ids: the expert accumulation order must be deterministic.
        for (int i = 1; i < nsel; ++i) {
            const int v = sel[i];
            int j = i - 1;
            while (j >= 0 && sel[j] > v) {
                sel[j + 1] = sel[j];
                --j;
            }
            sel[j + 1] = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < nsel; ++k)
            sum += isnan(plain[sel[k]]) ? 0.0f : plain[sel[k]];
        BeamRoutedSlot *dst = out + static_cast<int64_t>(t) * K;
        for (int k = 0; k < K; ++k) {
            if (k < nsel) {
                const float sc = isnan(plain[sel[k]]) ? 0.0f : plain[sel[k]];
                dst[k].eid = sel[k];
                dst[k].coeff =
                    (sum > 0.0f ? sc / (sum + 1e-20f) : 0.0f) * routed_scale;
            } else {
                dst[k].eid = 0;
                dst[k].coeff = 0.0f;
            }
        }
    }
}

__global__ void beam_swiglu_kernel(const __half *__restrict__ gate,
                                   const __half *__restrict__ up,
                                   __half *__restrict__ out, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float s = g / (1.0f + expf(-g));
    out[i] = __float2half(s * u);
}

__global__ void beam_depth_residual_kernel(const __half *__restrict__ branch,
                                           const __half *__restrict__ residual,
                                           __half *__restrict__ out, int n,
                                           float bs, float rs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __half b = __float2half(__half2float(branch[i]) * bs);
    const __half r = __float2half(__half2float(residual[i]) * rs);
    out[i] = __hadd(b, r);
}

__global__ void beam_softcap_kernel(float *__restrict__ logits, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits[i] = tanhf(logits[i] / kSoftcap) * kSoftcap;
}

} // namespace

cudaError_t RunBeamNorm(cudaStream_t stream, const __half *x, const __half *w,
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

cudaError_t RunBeamHeadNorm(cudaStream_t stream, const __half *x,
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

cudaError_t RunBeamRope(cudaStream_t stream, __half *q, __half *k, int seq,
                        int q_heads, int k_heads, int head_dim, float theta,
                        int pos)
{
    if (seq <= 0 || head_dim <= 0 || (head_dim & 1) != 0)
        return cudaErrorInvalidValue;
    const int half = head_dim / 2;
    for (int t = 0; t < seq; ++t) {
        const int p = pos + t;
        if (q && q_heads > 0)
            beam_rope_kernel<<<q_heads, half, 0, stream>>>(
                q + static_cast<int64_t>(t) * q_heads * head_dim, head_dim,
                theta, p);
        if (k && k_heads > 0)
            beam_rope_kernel<<<k_heads, half, 0, stream>>>(
                k + static_cast<int64_t>(t) * k_heads * head_dim, head_dim,
                theta, p);
    }
    return cudaGetLastError();
}

cudaError_t RunBeamSsmScale(cudaStream_t stream, __half *q, int heads,
                            int head_dim, int pos)
{
    if (!q || heads <= 0 || head_dim <= 0) return cudaSuccess;
    const int64_t n = static_cast<int64_t>(heads) * head_dim;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    beam_ssmax_kernel<<<blocks, kThreads, 0, stream>>>(q, n, pos);
    return cudaGetLastError();
}

cudaError_t RunBeamAttnDecode(cudaStream_t stream, const __half *q,
                              const __half *k_cache, const __half *v_cache,
                              const __half *gate, const __half *gate_bias,
                              __half *out, int heads, int kv_heads,
                              int head_dim, int pos, int window, float scale)
{
    if (!q || !k_cache || !v_cache || !gate || !gate_bias || !out ||
        heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim % 32) != 0 || heads % kv_heads)
        return cudaErrorInvalidValue;
    const int start = window > 0 ? std::max(0, pos + 1 - window) : 0;
    const int warps_per_block = 8;
    const int grid = (heads + warps_per_block - 1) / warps_per_block;
    beam_attn_decode_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        q, k_cache, v_cache, gate, gate_bias, out, heads, kv_heads, head_dim,
        pos, start, scale);
    return cudaGetLastError();
}

cudaError_t RunBeamRouter(cudaStream_t stream, const float *router_w,
                          const float *router_bias, const __half *x, int seq,
                          int E, int K, int H, int n_group, int topk_group,
                          float routed_scale, BeamRoutedSlot *d_slots)
{
    if (!router_w || !router_bias || !x || !d_slots || seq <= 0 || E <= 0 ||
        K <= 0 || K > E || H <= 0 || n_group <= 0 || n_group > 64 ||
        topk_group <= 0 || topk_group > n_group || E % n_group != 0)
        return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(2 * E) * sizeof(float);
    beam_router_kernel<<<seq, kThreads, shmem, stream>>>(
        router_w, router_bias, x, E, K, H, n_group, topk_group, routed_scale,
        d_slots);
    return cudaGetLastError();
}

cudaError_t RunBeamSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    beam_swiglu_kernel<<<blocks, kThreads, 0, stream>>>(gate, up, out, n);
    return cudaGetLastError();
}

cudaError_t RunBeamDepthResidual(cudaStream_t stream, const __half *branch,
                                 const __half *residual, __half *out, int n,
                                 int idx)
{
    if (n <= 0) return cudaSuccess;
    if (idx <= 0) return cudaErrorInvalidValue;
    const float bs = 1.0f / sqrtf(static_cast<float>(idx));
    const float rs = sqrtf(static_cast<float>(idx - 1) / static_cast<float>(idx));
    const int blocks = (n + kThreads - 1) / kThreads;
    beam_depth_residual_kernel<<<blocks, kThreads, 0, stream>>>(
        branch, residual, out, n, bs, rs);
    return cudaGetLastError();
}

cudaError_t RunBeamSoftcap(cudaStream_t stream, float *logits, int n)
{
    if (n <= 0) return cudaSuccess;
    const int blocks = (n + kThreads - 1) / kThreads;
    beam_softcap_kernel<<<blocks, kThreads, 0, stream>>>(logits, n);
    return cudaGetLastError();
}

cudaError_t RunBeamEmbed(cudaStream_t stream, const __half *table,
                         const int32_t *ids, __half *out, int seq, int hidden)
{
    if (seq <= 0 || hidden <= 0) return cudaSuccess;
    const int bx = (hidden + kThreads - 1) / kThreads;
    dim3 grid(bx, seq);
    lsxcommon::kernels::EmbedGatherRowsKernel<><<<grid, kThreads, 0, stream>>>(table, ids, out, hidden);
    return cudaGetLastError();
}

} // namespace lsxcommon::beam
