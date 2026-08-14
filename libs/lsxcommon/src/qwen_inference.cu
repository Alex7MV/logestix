#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace lsxcommon {
namespace qwen {

namespace {

__device__ __forceinline__ float silu_dev(float x)
{
    const float e = __expf(x);
    return x * e / (e + 1.0f);
}

__device__ __forceinline__ float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// ── RMSNorm, offset-weight form: out = rmsnorm(x) * (1 + w) ────────────────
__global__ void rmsnorm_offset_kernel(const __half *x, const __half *w,
                                      __half *out, int rows, int hidden,
                                      double eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half *xr = x + (int64_t)row * hidden;
    __half *orow = out + (int64_t)row * hidden;

    extern __shared__ float sarray[];
    float *squares = sarray;
    float val = __half2float(xr[threadIdx.x]);
    squares[threadIdx.x] = val * val;
    __syncthreads();
    for (int s = hidden / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(squares[0] / (float)hidden + (float)eps);
    float wf = __half2float(w[threadIdx.x]);
    orow[threadIdx.x] = __float2half(val * inv * (1.0f + wf));
}

// ── RMSNormGated: out = rmsnorm(x) * head * silu(z) ────────────────────────
__global__ void rmsnorm_gated_kernel(const __half *x, const __half *z,
                                     const __half *whead, __half *out,
                                     int rows, int dim, double eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half *xr = x + (int64_t)row * dim;
    const __half *zr = z + (int64_t)row * dim;
    extern __shared__ float sarray[];
    float *squares = sarray;
    float val = __half2float(xr[threadIdx.x]);
    squares[threadIdx.x] = val * val;
    __syncthreads();
    for (int s = dim / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(squares[0] / (float)dim + (float)eps);
    float wf = __half2float(whead[threadIdx.x]);
    float g = __half2float(zr[threadIdx.x]);
    out[(int64_t)row * dim + threadIdx.x] = __float2half(val * inv * wf * silu_dev(g));
}

// ── RoPE cos/sin (partial rotary, dims = rotary_dim) ───────────────────────
__global__ void rope_cos_sin_kernel(const int32_t *pos, __half *cos_buf,
                                    __half *sin_buf, int seq, int rotary_dim,
                                    double theta)
{
    // Layout: cos_buf/sin_buf hold [seq][rotary_dim/2]; entry (t,d) =
    // cos/sin of the pair at rotary index 2d.
    const int64_t total = (int64_t)seq * (rotary_dim / 2);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        const int t = (int)(idx / (rotary_dim / 2));
        const int d = (int)(idx % (rotary_dim / 2));
        double inv = 1.0 / std::pow(theta, (double)(2 * d) / (double)rotary_dim);
        double ang = (double)pos[t] * inv;
        cos_buf[idx] = __float2half((float)std::cos(ang));
        sin_buf[idx] = __float2half((float)std::sin(ang));
    }
}

__device__ __forceinline__ void rope_pair(__half2 &v, float cosv, float sinv)
{
    float2 f = __half22float2(v);
    v = __floats2half2_rn(f.x * cosv - f.y * sinv, f.x * sinv + f.y * cosv);
}

__global__ void partial_rope_kernel(const __half *q, const __half *k,
                                    const __half *cos_buf, const __half *sin_buf,
                                    __half *q_out, __half *k_out,
                                    int seq, int heads, int head_dim,
                                    int rotary_dim)
{
    const int64_t total = (int64_t)seq * heads * (rotary_dim / 2);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        const int p = (int)(idx % (rotary_dim / 2));
        const int h = (int)((idx / (rotary_dim / 2)) % heads);
        const int t = (int)(idx / ((int64_t)(rotary_dim / 2) * heads));

        const float cosv = __half2float(cos_buf[t * (rotary_dim / 2) + p]);
        const float sinv = __half2float(sin_buf[t * (rotary_dim / 2) + p]);
        const int64_t base = ((int64_t)t * heads + h) * head_dim + (int64_t)p * 2;
        __half2 qv = *reinterpret_cast<const __half2*>(q + base);
        __half2 kv = *reinterpret_cast<const __half2*>(k + base);
        rope_pair(qv, cosv, sinv);
        rope_pair(kv, cosv, sinv);
        *reinterpret_cast<__half2*>(q_out + base) = qv;
        *reinterpret_cast<__half2*>(k_out + base) = kv;
    }
    // Pass-through dims (rotary_dim .. head_dim) for each (t,h) row.
    const int64_t npt = (int64_t)seq * heads * (head_dim - rotary_dim);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < npt; idx += (int64_t)blockDim.x * gridDim.x) {
        const int pk = (int)(idx % (head_dim - rotary_dim));
        const int rh = (int)(idx / (head_dim - rotary_dim));   // (t*heads+h)
        const int64_t base = (int64_t)rh * head_dim + rotary_dim;
        q_out[base + pk] = q[base + pk];
        k_out[base + pk] = k[base + pk];
    }
}

// ── Embed ───────────────────────────────────────────────────────────────────
__global__ void embed_kernel(const __half *embed, const int32_t *ids,
                             __half *out, int seq, int hidden)
{
    const int64_t total = (int64_t)seq * hidden;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t)blockDim.x * gridDim.x) {
        const int t = (int)(i / hidden);
        const int h = (int)(i % hidden);
        out[i] = embed[ids[t] * (int64_t)hidden + h];
    }
}

// ── GatedDeltaNet causal conv1d (depthwise, kernel per channel) ─────────────
// mixed: [seq, conv_dim]; w: [conv_dim, kernel]; out: [seq, conv_dim].
// causal: out[t][c] = sum_{k=0..kernel-1} mixed[t-k][c] * w[c][k]  (skip t-k<0)
__global__ void qwen_conv1d_kernel(const __half *mixed, const __half *w,
                                   __half *out, int seq, int conv_dim,
                                   int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int tt = t - k;
            if (tt < 0) continue;
            acc += __half2float(w[c * kernel + k]) *
                   __half2float(mixed[(int64_t)tt * conv_dim + c]);
        }
        out[(int64_t)t * conv_dim + c] = __float2half(acc);
    }
}

// ── Gated delta rule (recurrent, per head, sequential over seq) ─────────────
// One block per v-head; threads = VD. Iterates tokens 0..seq-1 maintaining
// the state S[KD][VD] in shared memory, exactly matching
// torch_recurrent_gated_delta_rule ordering:
//   S = S*exp(g_t); kv=S·k_t; Δ=(v_t - kv)*β_t; S = S + k_t⊗Δ; out_t = (S·q_t)*scale
// q,k already l2-normalized; q already scaled by (1/sqrt(KD)).
template<int KD>
__global__ void gated_delta_kernel(
    const __half *q, const __half *k, const __half *v,
    const float *g, const float *beta,
    __half *out, int seq, int n_v_heads, int VD_, float scale)
{
    __shared__ float S[KD][64];
    const int head = blockIdx.x;
    if (head >= n_v_heads) return;

    for (int i = threadIdx.x; i < KD * VD_; i += blockDim.x)
        reinterpret_cast<float*>(&S[0][0])[i] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq; ++t) {
        const __half *qtt = q + ((int64_t)t * n_v_heads + head) * KD;
        const __half *ktt = k + ((int64_t)t * n_v_heads + head) * KD;
        const __half *vtt = v + ((int64_t)t * n_v_heads + head) * VD_;
        const float decay = __expf(g[t * n_v_heads + head]);
        const float beta_f = beta[t * n_v_heads + head];

        for (int d = threadIdx.x; d < VD_; d += blockDim.x) {
            for (int kk = 0; kk < KD; ++kk)
                S[kk][d] *= decay;
        }
        __syncthreads();

        __shared__ float kv_mem[64];
        for (int d = threadIdx.x; d < VD_; d += blockDim.x) {
            float acc = 0.0f;
            for (int kk = 0; kk < KD; ++kk)
                acc += S[kk][d] * __half2float(ktt[kk]);
            kv_mem[d] = acc;
        }
        __syncthreads();

        __shared__ float delta[64];
        for (int d = threadIdx.x; d < VD_; d += blockDim.x) {
            delta[d] = (__half2float(vtt[d]) - kv_mem[d]) * beta_f;
        }
        __syncthreads();
        // S[k][d] += k[k] * delta[d]
        for (int d = threadIdx.x; d < VD_; d += blockDim.x) {
            for (int kk = 0; kk < KD; ++kk)
                S[kk][d] += __half2float(ktt[kk]) * delta[d];
        }
        __syncthreads();

        // out_t[d] = (sum_dk S[kk][d]*q_t[kk]) * scale
        for (int d = threadIdx.x; d < VD_; d += blockDim.x) {
            float acc = 0.0f;
            for (int kk = 0; kk < KD; ++kk)
                acc += S[kk][d] * __half2float(qtt[kk]);
            out[(int64_t)t * n_v_heads * VD_ + head * VD_ + d] =
                __float2half(acc * scale);
        }
        __syncthreads();
    }
}

// ── Full attention softmax core (per block per (t, head)) ──────────────────
__global__ void full_attn_softmax_kernel(
    const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int heads, int head_dim, int kv_heads, float scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);

    const __half *qrow = q + ((int64_t)t * heads + h) * head_dim;
    const __half *kbase = k + (int64_t)kvh * head_dim;
    const __half *vbase = v + (int64_t)kvh * head_dim;

    extern __shared__ float slogits[];
    float m = -1e30f;
    for (int j = 0; j <= t; ++j) {
        const __half *krow = kbase + (int64_t)j * (int64_t)kv_heads * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qrow[d]) * __half2float(krow[d]);
        const float lg = dot * scale;
        slogits[j] = lg;
        m = fmaxf(m, lg);
    }
    __syncthreads();
    float sum = 0.0f;
    for (int j = 0; j <= t; ++j) {
        const float e = __expf(slogits[j] - m);
        slogits[j] = e;
        sum += e;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) {
            const __half *vrow = vbase + (int64_t)j * (int64_t)kv_heads * head_dim;
            acc += (slogits[j] / sum) * __half2float(vrow[d]);
        }
        out[((int64_t)t * heads + h) * head_dim + d] = __float2half(acc);
    }
}

} // namespace

// ── Public numeric kernels ──────────────────────────────────────────────────

cudaError_t RunQwenRMSNorm(cudaStream_t stream, const __half *x, const __half *w,
                           __half *out, int rows, int hidden, double eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    rmsnorm_offset_kernel<<<rows, hidden, hidden * sizeof(float), stream>>>(
        x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunQwenRopeCosSin(cudaStream_t stream, const int32_t *pos, int seq,
                              int rotary_dim, double theta, __half *cos_buf,
                              __half *sin_buf)
{
    const int threads = 256;
    const int64_t total = (int64_t)seq * (rotary_dim / 2);
    const int blocks = (int)((total + threads - 1) / threads);
    rope_cos_sin_kernel<<<blocks, threads, 0, stream>>>(
        pos, cos_buf, sin_buf, seq, rotary_dim, theta);
    return cudaGetLastError();
}

cudaError_t RunQwenPartialRope(cudaStream_t stream, const __half *q,
                               const __half *k, const __half *cos_buf,
                               const __half *sin_buf, __half *q_out,
                               __half *k_out, int seq, int heads,
                               int head_dim, int rotary_dim)
{
    const int threads = 256;
    const int64_t total = (int64_t)seq * heads * (rotary_dim / 2);
    const int blocks = (int)((total + threads - 1) / threads);
    partial_rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos_buf, sin_buf, q_out, k_out, seq, heads, head_dim, rotary_dim);
    return cudaGetLastError();
}

cudaError_t RunQwenEmbed(cudaStream_t stream, const __half *embed,
                         const int32_t *ids, __half *out, int seq, int hidden)
{
    const int threads = 256;
    const int blocks = (int)(((int64_t)seq * hidden + threads - 1) / threads);
    embed_kernel<<<blocks, threads, 0, stream>>>(embed, ids, out, seq, hidden);
    return cudaGetLastError();
}

} // namespace qwen
} // namespace lsxcommon