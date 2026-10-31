#include "lsxcommon/step_moe_kernels.h"

#include "step_moe_constants.h"

#include <cmath>
#include <cstring>

namespace lsxcommon::step {
namespace {

constexpr int kThreads = 256;
constexpr float kNegInf = -3.402823466e+38f;

inline int Blocks(int64_t n)
{
    return static_cast<int>((n + kThreads - 1) / kThreads);
}

__device__ __forceinline__ float Bf16ToF32(const __nv_bfloat16 v)
{
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 F32ToBf16(const float v)
{
    return __float2bfloat16_rn(v);
}

// ── norms ──────────────────────────────────────────────────────────────────

__global__ void step_norm_f32_to_half_kernel(const float *__restrict__ x,
                                             const float *__restrict__ w,
                                             __half *__restrict__ out, int n,
                                             float eps)
{
    __shared__ float sh[32];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += x[i] * x[i];
    // block reduction (deterministic tree over 32 partials)
    float v = ss;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < (blockDim.x + 31) / 32) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_down_sync(0xffffffff, t, off);
        if (tid == 0) sh[0] = rsqrtf(t / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = __float2half_rn(x[i] * inv * (w[i] + 1.0f));
}

__global__ void step_norm_f32_to_bf16_narrowed_kernel(
    const float *__restrict__ x, const float *__restrict__ w,
    __nv_bfloat16 *__restrict__ out, int n, float eps)
{
    __shared__ float sh[32];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = Bf16ToF32(F32ToBf16(x[i]));
        ss += v * v;
    }
    float v = ss;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < (blockDim.x + 31) / 32) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_down_sync(0xffffffff, t, off);
        if (tid == 0) sh[0] = rsqrtf(t / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = tid; i < n; i += blockDim.x)
        out[i] = F32ToBf16(Bf16ToF32(F32ToBf16(x[i])) * inv * (w[i] + 1.0f));
}

__global__ void step_head_norm_kernel(const __half *__restrict__ x,
                                      const float *__restrict__ w,
                                      __half *__restrict__ out, int heads,
                                      int head_dim, float eps)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const __half *src = x + static_cast<int64_t>(h) * head_dim;
    __half *dst = out + static_cast<int64_t>(h) * head_dim;
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = __half2float(src[i]);
        ss += v * v;
    }
    float v = ss;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if ((tid & 31) == 0) sh[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        float t = (tid < (blockDim.x + 31) / 32) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_down_sync(0xffffffff, t, off);
        if (tid == 0)
            sh[0] = rsqrtf(t / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = tid; i < head_dim; i += blockDim.x)
        dst[i] = __float2half_rn(__half2float(src[i]) * inv * (w[i] + 1.0f));
    (void)heads;
}

// ── partial NEOX RoPE ──────────────────────────────────────────────────────

__global__ void step_rope_kernel(__half *__restrict__ x, int heads,
                                 int head_dim, int span, float theta, int pos)
{
    const int h = blockIdx.x;
    const int half = span / 2;
    const int j = threadIdx.x;
    if (j >= half) return;
    __half *base = x + static_cast<int64_t>(h) * head_dim;
    // inv_freq[j] = 1 / theta^(2j/span)
    const float inv = 1.0f / powf(theta, 2.0f * static_cast<float>(j) /
                                              static_cast<float>(span));
    const float ang = static_cast<float>(pos) * inv;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float a = __half2float(base[j]);
    const float b = __half2float(base[j + half]);
    base[j] = __float2half_rn(a * c - b * s);
    base[j + half] = __float2half_rn(a * s + b * c);
    (void)heads;
}

// ── windowed causal GQA decode attention ──────────────────────────────────

__global__ void step_attn_window_kernel(const __half *__restrict__ q,
                                        const __half *__restrict__ k,
                                        const __half *__restrict__ v,
                                        __half *__restrict__ out, int pos,
                                        int heads, int kv_heads, int head_dim,
                                        float scale, int window)
{
    // One block per query head; 4 warps split the visible position range.
    // Each lane owns head_dim/32 value dims (head_dim % 32 == 0, <= 256).
    constexpr int kWarps = 4;
    constexpr int kPerLane = 8;
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hpg = heads / kv_heads;
    const int kv = h / hpg;
    const int nper = head_dim / 32;
    const int lo = window > 0 ? max(0, pos + 1 - window) : 0;

    __shared__ float s_wm[kWarps];
    __shared__ float s_wl[kWarps];
    __shared__ float s_wacc[kWarps][32 * kPerLane];

    float acc[kPerLane];
    for (int i = 0; i < kPerLane; ++i) acc[i] = 0.0f;
    float m = kNegInf;
    float l = 0.0f;

    const __half *qh = q + static_cast<int64_t>(h) * head_dim;
    const __half *kh_base = k + kv * head_dim;
    const __half *vh_base = v + kv * head_dim;

    for (int s = lo + warp; s <= pos; s += kWarps) {
        const __half *kh = kh_base + static_cast<int64_t>(s) * kv_heads * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < nper; ++i) {
            const int d = lane + 32 * i;
            dot += __half2float(qh[d]) * __half2float(kh[d]);
        }
        float r = dot;
        for (int off = 16; off > 0; off >>= 1)
            r += __shfl_down_sync(0xffffffff, r, off);
        const float score = __shfl_sync(0xffffffff, r, 0) * scale;
        const float nm = fmaxf(m, score);
        const float rescale = (m == kNegInf) ? 0.0f : expf(m - nm);
        const float p = expf(score - nm);
        m = nm;
        l = l * rescale + p;
        const __half *vh = vh_base + static_cast<int64_t>(s) * kv_heads * head_dim;
        for (int i = 0; i < nper; ++i) {
            const int d = lane + 32 * i;
            acc[i] = acc[i] * rescale + p * __half2float(vh[d]);
        }
    }

    if (lane == 0) {
        s_wm[warp] = m;
        s_wl[warp] = l;
    }
    for (int i = 0; i < nper; ++i)
        s_wacc[warp][lane + 32 * i] = acc[i];
    __syncthreads();

    if (warp == 0) {
        // Merge the four warp-local online-softmax states in fixed order.
        float gm = kNegInf;
        for (int w = 0; w < kWarps; ++w) gm = fmaxf(gm, s_wm[w]);
        float gl = 0.0f;
        for (int w = 0; w < kWarps; ++w)
            gl += (s_wm[w] == kNegInf) ? 0.0f : expf(s_wm[w] - gm) * s_wl[w];
        const float denom = gl > 0.0f ? gl : 1.0f;
        for (int i = 0; i < nper; ++i) {
            const int d = lane + 32 * i;
            float a = 0.0f;
            for (int w = 0; w < kWarps; ++w) {
                const float wgt = (s_wm[w] == kNegInf) ? 0.0f : expf(s_wm[w] - gm);
                a += wgt * s_wacc[w][d];
            }
            out[static_cast<int64_t>(h) * head_dim + d] =
                __float2half_rn(a / denom);
        }
    }
}

// ── per-head output gate ───────────────────────────────────────────────────

__global__ void step_head_gate_kernel(__half *__restrict__ attn,
                                      const __half *__restrict__ g, int heads,
                                      int head_dim)
{
    const int64_t n = static_cast<int64_t>(heads) * head_dim;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int h = static_cast<int>(i / head_dim);
        const float gate = 1.0f / (1.0f + expf(-__half2float(g[h])));
        attn[i] = __float2half_rn(__half2float(attn[i]) * gate);
    }
}

// ── sigmoid + selection-only-bias router ───────────────────────────────────

__global__ void step_router_kernel(const __nv_bfloat16 *__restrict__ gate,
                                   const float *__restrict__ bias,
                                   const __half *__restrict__ x, int E, int K,
                                   int H, float routed_scale,
                                   StepRoutedSlot *__restrict__ slots,
                                   int32_t *__restrict__ ids)
{
    extern __shared__ float sh[];
    float *score = sh;               // [E] sigmoid + bias
    int *sel = reinterpret_cast<int *>(sh + E);  // [K]
    const int tid = threadIdx.x;
    for (int e = tid; e < E; e += blockDim.x) {
        const __nv_bfloat16 *row = gate + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = 0; k < H; ++k)
            acc += Bf16ToF32(row[k]) * __half2float(x[k]);
        const float p = 1.0f / (1.0f + expf(-acc));
        score[e] = p + bias[e];
    }
    __syncthreads();
    if (tid == 0) {
        // Deterministic top-k: strict greater keeps the lower expert index on
        // ties; emitted in descending score order (torch.topk semantics).
        for (int i = 0; i < K; ++i) {
            int best = -1;
            float bestv = kNegInf;
            for (int e = 0; e < E; ++e) {
                bool already = false;
                for (int j = 0; j < i; ++j)
                    if (sel[j] == e) {
                        already = true;
                        break;
                    }
                if (already) continue;
                const float v = score[e];
                if (v > bestv) {
                    bestv = v;
                    best = e;
                }
            }
            sel[i] = best;
        }
        float denom = 0.0f;
        for (int i = 0; i < K; ++i) {
            const int e = sel[i];
            const float w = score[e] - bias[e];
            denom += w;
        }
        const float inv = 1.0f / (denom + 1e-20f);
        for (int i = 0; i < K; ++i) {
            const int e = sel[i];
            const float w = score[e] - bias[e];
            slots[i].eid = e;
            slots[i].coeff = w * inv * routed_scale;
            if (ids) ids[i] = e;
        }
    }
}

// ── clamped SwiGLU ─────────────────────────────────────────────────────────

__global__ void step_swiglu_clamp_kernel(const __half *__restrict__ gate,
                                         const __half *__restrict__ up,
                                         __half *__restrict__ out, int64_t n,
                                         float limit)
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float g = __half2float(gate[i]);
        const float u = __half2float(up[i]);
        const float silu = g / (1.0f + expf(-g));
        const float a = fminf(silu, limit);          // no lower clamp
        const float b = fminf(fmaxf(u, -limit), limit);
        out[i] = __float2half_rn(a * b);
    }
}

// ── residual combiners ─────────────────────────────────────────────────────

__global__ void step_combine_kernel(const float *__restrict__ routed,
                                    const __half *__restrict__ shared,
                                    float *__restrict__ residual, int64_t n)
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        residual[i] += Bf16ToF32(F32ToBf16(routed[i])) +
                       __half2float(shared[i]);
}

__global__ void step_add_half_kernel(const __half *__restrict__ y,
                                     float *__restrict__ residual, int64_t n)
{
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        residual[i] += __half2float(y[i]);
}

__global__ void step_embed_kernel(const __nv_bfloat16 *__restrict__ table,
                                  const int32_t *__restrict__ ids,
                                  float *__restrict__ out, int n)
{
    const int64_t row = static_cast<int64_t>(ids[0]) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = Bf16ToF32(table[row + i]);
}

} // namespace

cudaError_t RunStepNormF32ToHalf(cudaStream_t stream, const float *x,
                                 const float *w, __half *out, int n, float eps)
{
    if (n <= 0) return cudaSuccess;
    step_norm_f32_to_half_kernel<<<1, kThreads, 0, stream>>>(x, w, out, n, eps);
    return cudaGetLastError();
}

cudaError_t RunStepNormF32ToBf16Narrowed(cudaStream_t stream, const float *x,
                                         const float *w, __nv_bfloat16 *out,
                                         int n, float eps)
{
    if (n <= 0) return cudaSuccess;
    step_norm_f32_to_bf16_narrowed_kernel<<<1, kThreads, 0, stream>>>(x, w, out,
                                                                     n, eps);
    return cudaGetLastError();
}

cudaError_t RunStepHeadNorm(cudaStream_t stream, const __half *x,
                            const float *w, __half *out, int heads, int head_dim,
                            float eps)
{
    if (heads <= 0 || head_dim <= 0) return cudaSuccess;
    step_head_norm_kernel<<<heads, kThreads, 0, stream>>>(x, w, out, heads,
                                                          head_dim, eps);
    return cudaGetLastError();
}

cudaError_t RunStepPartialRope(cudaStream_t stream, __half *q, int q_heads,
                               __half *k, int k_heads, int head_dim, int span,
                               float theta, int pos)
{
    if (span <= 0 || (span & 1)) return cudaErrorInvalidValue;
    const int half = span / 2;
    if (q && q_heads > 0)
        step_rope_kernel<<<q_heads, half, 0, stream>>>(q, q_heads, head_dim,
                                                       span, theta, pos);
    if (k && k_heads > 0)
        step_rope_kernel<<<k_heads, half, 0, stream>>>(k, k_heads, head_dim,
                                                       span, theta, pos);
    return cudaGetLastError();
}

cudaError_t RunStepWindowedAttnDecode(cudaStream_t stream, const __half *q,
                                      const __half *k_cache,
                                      const __half *v_cache, __half *out,
                                      int pos, int heads, int kv_heads,
                                      int head_dim, float scale, int window)
{
    if (pos < 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0)
        return cudaErrorInvalidValue;
    if (head_dim % 32 != 0 || head_dim / 32 > 8)
        return cudaErrorInvalidValue;
    step_attn_window_kernel<<<heads, 128, 0, stream>>>(
        q, k_cache, v_cache, out, pos, heads, kv_heads, head_dim, scale, window);
    return cudaGetLastError();
}

cudaError_t RunStepApplyHeadGate(cudaStream_t stream, __half *attn,
                                 const __half *g, int heads, int head_dim)
{
    const int64_t n = static_cast<int64_t>(heads) * head_dim;
    if (n <= 0) return cudaSuccess;
    step_head_gate_kernel<<<Blocks(n), kThreads, 0, stream>>>(attn, g, heads,
                                                              head_dim);
    return cudaGetLastError();
}

cudaError_t RunStepRouter(cudaStream_t stream, const __nv_bfloat16 *gate,
                          const float *bias, const __half *x, int E, int K,
                          int H, float routed_scale, StepRoutedSlot *d_slots,
                          int32_t *d_ids)
{
    if (E <= 0 || K <= 0 || K > 64 || K > E || H <= 0)
        return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(E) * sizeof(float) +
                         static_cast<size_t>(K) * sizeof(int) + 64;
    step_router_kernel<<<1, kThreads, shmem, stream>>>(
        gate, bias, x, E, K, H, routed_scale, d_slots, d_ids);
    return cudaGetLastError();
}

cudaError_t RunStepSwigluClamp(cudaStream_t stream, const __half *gate,
                               const __half *up, __half *out, int64_t n,
                               float limit)
{
    if (n <= 0) return cudaSuccess;
    step_swiglu_clamp_kernel<<<Blocks(n), kThreads, 0, stream>>>(gate, up, out,
                                                                 n, limit);
    return cudaGetLastError();
}

cudaError_t RunStepCombineF32(cudaStream_t stream, const float *routed_acc,
                              const __half *shared, float *residual, int n)
{
    if (n <= 0) return cudaSuccess;
    step_combine_kernel<<<Blocks(n), kThreads, 0, stream>>>(routed_acc, shared,
                                                            residual, n);
    return cudaGetLastError();
}

cudaError_t RunStepAddHalfToF32(cudaStream_t stream, const __half *y,
                                float *residual, int n)
{
    if (n <= 0) return cudaSuccess;
    step_add_half_kernel<<<Blocks(n), kThreads, 0, stream>>>(y, residual, n);
    return cudaGetLastError();
}

cudaError_t RunStepEmbedBf16(cudaStream_t stream, const __nv_bfloat16 *table,
                             const int32_t *ids, float *out, int n)
{
    if (n <= 0) return cudaSuccess;
    step_embed_kernel<<<1, kThreads, 0, stream>>>(table, ids, out, n);
    return cudaGetLastError();
}

} // namespace lsxcommon::step
