#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/shared_experts.h"

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
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_dev(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// ── RMSNorm, offset-weight form: out = rmsnorm(x) * (1 + w) ────────────────
// grid-stride reduction: each thread processes multiple elements when
// hidden > blockDim.x (supports arbitrary hidden up to 4096+).
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
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / (float)hidden + (float)eps);
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float wf = __half2float(w[j]);
        orow[j] = __float2half(__half2float(xr[j]) * inv * (1.0f + wf));
    }
}

// ── RMSNormGated: out = rmsnorm(x) * head * silu(z) ────────────────────────
// grid-stride reduction (same pattern).
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
    float sum = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / (float)dim + (float)eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float wf = __half2float(whead[j]);
        float g = __half2float(zr[j]);
        out[(int64_t)row * dim + j] = __float2half(__half2float(xr[j]) * inv * wf * silu_dev(g));
    }
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

__global__ void partial_rope_kernel(const __half *q, const __half *k,
                                    const __half *cos_buf, const __half *sin_buf,
                                    __half *q_out, __half *k_out,
                                    int seq, int q_heads, int k_heads,
                                    int head_dim, int rotary_dim)
{
    // HF Qwen3_5 rotate_half pairing: for j in [0, rot/2) the pair is
    // (j, j + rot/2); cos/sin index j. cos/sin layout: [seq, rot/2].
    const int64_t half = rotary_dim / 2;
    // q rotation
    {
        const int64_t total = (int64_t)seq * q_heads * half;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
            const int j = (int)(idx % half);
            const int h = (int)((idx / half) % q_heads);
            const int t = (int)(idx / (half * q_heads));
            const float c = __half2float(cos_buf[t * half + j]);
            const float s = __half2float(sin_buf[t * half + j]);
            const int64_t row = ((int64_t)t * q_heads + h) * head_dim;
            const float x1 = __half2float(q[row + j]);
            const float x2 = __half2float(q[row + j + half]);
            q_out[row + j]        = __float2half(x1 * c - x2 * s);
            q_out[row + j + half] = __float2half(x2 * c + x1 * s);
        }
    }
    // q pass-through
    {
        const int64_t npt = (int64_t)seq * q_heads * (head_dim - rotary_dim);
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < npt; idx += (int64_t)blockDim.x * gridDim.x) {
            const int pk = (int)(idx % (head_dim - rotary_dim));
            const int rh = (int)(idx / (head_dim - rotary_dim));
            const int64_t base = (int64_t)rh * head_dim + rotary_dim;
            q_out[base + pk] = q[base + pk];
        }
    }
    // k rotation
    {
        const int64_t total = (int64_t)seq * k_heads * half;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
            const int j = (int)(idx % half);
            const int h = (int)((idx / half) % k_heads);
            const int t = (int)(idx / (half * k_heads));
            const float c = __half2float(cos_buf[t * half + j]);
            const float s = __half2float(sin_buf[t * half + j]);
            const int64_t row = ((int64_t)t * k_heads + h) * head_dim;
            const float x1 = __half2float(k[row + j]);
            const float x2 = __half2float(k[row + j + half]);
            k_out[row + j]        = __float2half(x1 * c - x2 * s);
            k_out[row + j + half] = __float2half(x2 * c + x1 * s);
        }
    }
    // k pass-through
    {
        const int64_t npt = (int64_t)seq * k_heads * (head_dim - rotary_dim);
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < npt; idx += (int64_t)blockDim.x * gridDim.x) {
            const int pk = (int)(idx % (head_dim - rotary_dim));
            const int rh = (int)(idx / (head_dim - rotary_dim));
            const int64_t base = (int64_t)rh * head_dim + rotary_dim;
            k_out[base + pk] = k[base + pk];
        }
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
            // HF causal conv1d: padding=kernel-1 both sides + right trim ==
            // F.conv1d cross-correlation == kernel reversal vs. this loop's
            // index, hence (kernel-1-k).
            acc += __half2float(w[c * kernel + (kernel - 1 - k)]) *
                   __half2float(mixed[(int64_t)tt * conv_dim + c]);
        }
        out[(int64_t)t * conv_dim + c] = __float2half(silu_dev(acc));
    }
}

// ── Full attention softmax core (per block per (t, head)) ──────────────────
// Thread 0 builds the logit row into shared memory; after the barrier every
// thread treats slogits as READ-ONLY (any read-modify-write there is a race:
// it corrupted softmax weights -> 0/0 -> NaN, which poisoned the residual
// stream and made generation non-deterministic).
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
    if (threadIdx.x == 0) {
        for (int j = 0; j <= t; ++j) {
            const __half *krow = kbase + (int64_t)j * (int64_t)kv_heads * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                dot += __half2float(qrow[d]) * __half2float(krow[d]);
            slogits[j] = dot * scale;
        }
    }
    __syncthreads();

    float m = -1e30f;
    for (int j = 0; j <= t; ++j) m = fmaxf(m, slogits[j]);
    float sum = 0.0f;
    for (int j = 0; j <= t; ++j) sum += __expf(slogits[j] - m);
    const float inv_sum = 1.0f / sum;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j <= t; ++j) {
            const __half *vrow = vbase + (int64_t)j * (int64_t)kv_heads * head_dim;
            acc += (__expf(slogits[j] - m) * inv_sum) * __half2float(vrow[d]);
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
    const int threads = std::min(hidden, 256);
    rmsnorm_offset_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
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
                               __half *k_out, int seq, int q_heads,
                               int k_heads, int head_dim, int rotary_dim)
{
    const int threads = 256;
    const int64_t work = std::max(
        (int64_t)seq * std::max(q_heads, k_heads) * (rotary_dim / 2),
        (int64_t)seq * std::max(q_heads, k_heads) * (head_dim - rotary_dim));
    const int blocks = (int)((work + threads - 1) / threads);
    partial_rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos_buf, sin_buf, q_out, k_out, seq, q_heads, k_heads,
        head_dim, rotary_dim);
    return cudaGetLastError();
}

cudaError_t RunQwenPartialRope(cudaStream_t stream, const __half *q,
                               const __half *k, const __half *cos_buf,
                               const __half *sin_buf, __half *q_out,
                               __half *k_out, int seq, int heads,
                               int head_dim, int rotary_dim)
{
    return RunQwenPartialRope(stream, q, k, cos_buf, sin_buf, q_out, k_out,
                              seq, heads, heads, head_dim, rotary_dim);
}

cudaError_t RunQwenEmbed(cudaStream_t stream, const __half *embed,
                         const int32_t *ids, __half *out, int seq, int hidden)
{
    const int threads = 256;
    const int blocks = (int)(((int64_t)seq * hidden + threads - 1) / threads);
    embed_kernel<<<blocks, threads, 0, stream>>>(embed, ids, out, seq, hidden);
    return cudaGetLastError();
}

// ── Inference kernels (NVFP4 decode + dense pipeline) ───────────────────────

namespace {

// E2M1 decode table (indexed by 4-bit nibble, sign bit 3).
__constant__ float qwen_e2m1_tab[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float qwen_e4m3_scale(uint8_t e4m3)
{
    uint32_t e4 = (e4m3 >> 3) & 0x0F;
    if (e4 == 0) return 0.0f;
    return __uint_as_float((e4 - 7 + 127) << 23);
}

// Fused NVFP4 dense GEMM: y[t][n] = sum_k x[t*K+k] * W[n*K+k], where W is the
// [N, K] row-major weight with per-16 E4M3 block scales + global_scale.
// x: [seq, K] FP16; y: [seq, N] FP16 (FP32 accumulate).
__global__ void qwen_nvfp4_gemm_kernel(
    const __half *x, const uint8_t *packed, const uint8_t *scales,
    float global_scale, __half *y, int seq, int K, int N)
{
    const int64_t total = (int64_t)seq * N;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int n = (int)(idx % N);
    const int t = (int)(idx / N);

    const __half *xrow = x + (int64_t)t * K;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // NVFP4 layout: element e lives at byte e>>1, low nibble if even,
        // high nibble if odd; scale is per-16 elements (e>>4).
        const int64_t e = (int64_t)n * K + k;
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        const float scl = qwen_e4m3_scale(scales[e >> 4]) * global_scale;
        acc += __half2float(xrow[k]) * qwen_e2m1_tab[nib] * scl;
    }
    y[idx] = __float2half(acc);
}

// Output gate: y = x * sigmoid(gate)  (per-feature).
__global__ void qwen_out_gate_kernel(const __half *x, const __half *gate,
                                     __half *y, int seq, int N)
{
    const int64_t total = (int64_t)seq * N;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    y[idx] = __float2half(__half2float(x[idx]) * sigmoid_dev(__half2float(gate[idx])));
}

// SwiGLU activation: y[i] = silu(a[i]) * b[i].
__global__ void qwen_swiglu_kernel(const __half *a, const __half *b,
                                   __half *y, int64_t n)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = __float2half(silu_dev(__half2float(a[idx])) * __half2float(b[idx]));
}

// elementwise: dst[i] += src[i].
__global__ void qwen_add_kernel(const __half *src, __half *dst, int64_t n)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
}

cudaError_t LaunchNvfp4Gemm(const QwenNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream)
{
    if (!w.d_packed || w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
    const int K = w.cols;   // input
    const int N = w.rows;   // output
    const int threads = 256;
    const int64_t total = (int64_t)seq * N;
    const int blocks = (int)((total + threads - 1) / threads);
    qwen_nvfp4_gemm_kernel<<<blocks, threads, 0, stream>>>(
        x, w.d_packed, w.d_scales, w.global_scale, y, seq, K, N);
    return cudaGetLastError();
}

cudaError_t LaunchNorm1d(const __half *x, const __half *w, __half *y,
                         int rows, int hidden, double eps, cudaStream_t stream)
{
    const int threads = std::min(hidden, 256);
    rmsnorm_offset_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
        x, w, y, rows, hidden, eps);
    return cudaGetLastError();
}

// Splits q_proj output [seq, NH*HD*2] into query (first half) and gate (second
// half), reshaping query to [seq, NH, HD], and applies per-head RMSNorm.
// d_k_norm unused (key norm done separately); primal q_norm applied here.
__global__ void qwen_split_query_kernel(
    const __half *d_qc, __half *d_gate, __half *d_q,
    const __half *d_k_norm, int seq, int NH, int HD, const __half *d_q_norm,
    const __half *d_k_norm2, float eps)
{
    const int64_t idx = (int64_t)blockIdx.x;      // t*NH + h
    const int h = (int)(idx % NH);
    const int t = (int)(idx / NH);
    const int d = threadIdx.x;
    if (d >= HD) return;

    const int64_t qc_base = idx * 2 * HD;
    const __half qv = d_qc[qc_base + d];
    const __half gv = d_qc[qc_base + HD + d];

    d_q[idx * HD + d] = qv;
    d_gate[idx * HD + d] = gv;

    // RMSNorm over this head's HD dims (offset form).
    extern __shared__ float ssum[];
    const __half *qrow = d_q + idx * HD;
    float v = __half2float(qrow[d]);
    ssum[d] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (d < s) ssum[d] += ssum[d + s];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] / (float)HD + eps);
    float wf = __half2float(d_q_norm[d]);
    d_q[idx * HD + d] = __float2half(v * inv * (1.0f + wf));
    (void)d_k_norm; (void)d_k_norm2;
}

// Per-head RMSNorm offset form on [seq, NH, HD] (in place).
__global__ void qwen_head_norm_kernel(const __half *x, const __half *w,
                                      __half *y, int seq, int NH, int HD,
                                      float eps)
{
    const int64_t idx = (int64_t)blockIdx.x;      // t*NH + h
    const int d = threadIdx.x;
    if (d >= HD) return;
    extern __shared__ float ssum[];
    const __half *row = x + idx * HD;
    float v = __half2float(row[d]);
    ssum[d] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (d < s) ssum[d] += ssum[d + s];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] / (float)HD + eps);
    float wf = __half2float(w[d]);
    y[idx * HD + d] = __float2half(v * inv * (1.0f + wf));
}

// Converts a/b (half) projections into g/beta (float).
//   a_in -> a[h]; b_in -> b[h]; g = -exp(A_log[h])*softplus(a+dt[h]),
//   beta = sigmoid(b). h = e % nv.
__global__ void qwen_prep_gating_kernel(
    const __half *a_in, const __half *b_in, const float *A_log,
    const float *dt_bias, float *g, float *beta, int64_t total, int nv)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int h = (int)(idx % nv);
    float a = __half2float(a_in[idx]);
    float b = __half2float(b_in[idx]);
    beta[idx] = sigmoid_dev(b);
    g[idx] = -__expf(A_log[h]) * softplus_dev(a + dt_bias[h]);
}

// L2-normalize one (t, head) row of kd elements in place within a strided
// buffer (row stride = `stride`, head offset = `head_off`), then multiply by
// extra_scale (1/sqrt(kd) for q, 1 for k). Mirrors HF l2norm(eps=1e-6).
// Grid: (seq, nk); one thread per element. blockDim is a power of 2 >= kd.
__global__ void qwen_l2norm_rows_kernel(__half *base, int seq, int nk, int kd,
                                        int stride, int head_off,
                                        float extra_scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int l = threadIdx.x;
    __shared__ float vals[256];
    __shared__ float red[256];
    __shared__ float inv;
    __half *row = base + (int64_t)t * stride + head_off + (int64_t)h * kd;
    float v = 0.0f;
    if (l < kd) v = __half2float(row[l]);
    vals[l] = v;
    red[l] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        __syncthreads();
    }
    if (l == 0) inv = rsqrtf(red[0] + 1e-6f) * extra_scale;
    __syncthreads();
    if (l < kd) row[l] = __float2half(vals[l] * inv);
}

// Gated delta rule, block per (t, v_head), thread per value-dim d.
// q and k arrive already l2-normalised (q pre-scaled by 1/sqrt(kd)) in the
// strided conv buffer (row stride = `stride`); v is read from the same buffer.
// Columns d are independent -> one thread owns the whole column S[0..kd, d].
// Ordering mirrors HF torch_recurrent_gated_delta_rule:
//   S=0; for tt in 0..t: S*=exp(g); kv=sum_l S[l]*ktt[l]; Δ=(v-kv)β;
//        S[l]+=ktt[l]*Δ;  out=sum_l S[l]*q[l].
__global__ void qwen_gated_delta_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, __half *__restrict__ out,
    int seq, int nk, int kd, int nv, int vd, int r, int stride)
{
    const int t = blockIdx.x;
    const int vh = blockIdx.y;
    const int kh = vh / r;         // key head shared by r value heads
    const int d = threadIdx.x;     // value-dim column [0, vd)
    if (d >= vd) return;

    // Registers hold this thread's column [l in 0..kd).
    float col[256];
    for (int l = 0; l < kd; ++l) col[l] = 0.0f;

    const __half *qrow = q + (int64_t)t * stride + (int64_t)kh * kd;

    for (int tt = 0; tt <= t; ++tt) {
        const float decay = __expf(g[tt * nv + vh]);
        const float betaf = beta[tt * nv + vh];
        const __half *ktt = k + (int64_t)tt * stride + (int64_t)kh * kd;
        const __half *vtt = v + (int64_t)tt * stride + (int64_t)vh * vd;

        for (int l = 0; l < kd; ++l) col[l] *= decay;
        float kv = 0.0f;
        for (int l = 0; l < kd; ++l) kv += col[l] * __half2float(ktt[l]);
        const float delta = (__half2float(vtt[d]) - kv) * betaf;
        for (int l = 0; l < kd; ++l) col[l] += __half2float(ktt[l]) * delta;
    }

    float acc = 0.0f;
    for (int l = 0; l < kd; ++l) acc += col[l] * __half2float(qrow[l]);
    out[((int64_t)t * nv + vh) * vd + d] = __float2half(acc);
}

} // namespace

cudaError_t RunQwenMLP(cudaStream_t stream,
                       const QwenLayerWeights &lw, int seq,
                       const __half *d_in, __half *d_out, __half *d_scratch)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int H = lw.mlp_gate.cols;       // input dim
    const int I = lw.mlp_gate.rows;       // intermediate
    if (H <= 0 || I <= 0) return cudaErrorInvalidValue;

    __half *d_gate = d_scratch;                 // [seq,I]
    __half *d_up   = d_scratch + (int64_t)seq * I;

    cudaError_t e;
    if (!lw.mlp_gate.d_packed) return cudaErrorInvalidValue;
    e = LaunchNvfp4Gemm(lw.mlp_gate, d_in, seq, d_gate, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.mlp_up,   d_in, seq, d_up, stream);   if (e) return e;
    const int64_t n_act = (int64_t)seq * I;
    const int threads = 256;
    const int blocks = (int)((n_act + threads - 1) / threads);
    qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_gate, n_act);
    e = cudaGetLastError(); if (e) return e;
    e = LaunchNvfp4Gemm(lw.mlp_down, d_gate, seq, d_out, stream);
    return e;
}

// ── Full attention layer. ───────────────────────────────────────────────────
// scratch (halves), sized by caller (see generation/arena):
//   d_qc  : [seq, NH*HD*2]  raw q_proj output (query + gate)
//   d_q   : [seq, NH*HD]    query (post-norm, post-rope)
//   d_gate: [seq, NH*HD]    output gate
//   d_k   : [seq, NKV*HD]   key (post-norm, post-rope)
//   d_v   : [seq, NKV*HD]   value
//   d_attn: [seq, NH*HD]    attention output (pre-o_proj)
//   d_cos : [seq, rotary_dim]
//   d_sin : [seq, rotary_dim]
cudaError_t RunQwenFullAttention(cudaStream_t stream,
                                 const QwenLayerWeights &lw,
                                 const QwenTopology &top, int seq,
                                 const __half *d_normed, __half *d_out,
                                 __half *d_qc, __half *d_q, __half *d_gate,
                                 __half *d_k, __half *d_v, __half *d_attn,
                                 __half *d_cos, __half *d_sin,
                                 int32_t *d_pos)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int NH = top.num_attention_heads;
    const int HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int H = top.hidden_size;
    const int rotary_dim = (int)(HD * top.partial_rotary_factor);
    if (NH <= 0 || HD <= 0 || NKV <= 0 || H <= 0) return cudaErrorInvalidValue;

    cudaError_t e;
    // 1. q_proj: [seq, H] -> [seq, NH*HD*2].
    e = LaunchNvfp4Gemm(lw.q, d_normed, seq, d_qc, stream); if (e) return e;
    // 2. k,v projections.
    e = LaunchNvfp4Gemm(lw.k, d_normed, seq, d_k, stream);  if (e) return e;
    e = LaunchNvfp4Gemm(lw.v, d_normed, seq, d_v, stream);  if (e) return e;

    // 3. Split query + gate, reshape to [seq, NH, HD], per-head RMSNorm(q).
    {
        const int threads = std::min(HD, 512);
        qwen_split_query_kernel<<<seq * NH, threads, HD * sizeof(float),
                                  stream>>>(
            d_qc, d_gate, d_q, nullptr, seq, NH, HD, lw.d_q_norm,
            lw.d_k_norm, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 4. per-head RMSNorm(k) into d_k itself.
    {
        const int threads = std::min(HD, 512);
        qwen_head_norm_kernel<<<seq * NKV, threads, HD * sizeof(float),
                                  stream>>>(
            d_k, lw.d_k_norm, d_k, seq, NKV, HD, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 5. RoPE on q,k (first rotary_dim dims). k has NKV heads, q has NH heads.
    e = RunQwenRopeCosSin(stream, d_pos, seq, rotary_dim, top.rope_theta,
                          d_cos, d_sin); if (e) return e;
    e = RunQwenPartialRope(stream, d_q, d_k, d_cos, d_sin, d_q, d_k,
                           seq, NH, NKV, HD, rotary_dim); if (e) return e;

    // 6. Scaled dot-product attention (GQA, causal) -> d_attn [seq, NH, HD].
    {
        const float scale = 1.0f / sqrtf((float)HD);
        const dim3 grid(seq, NH);
        const int threads = std::min(HD, 512);
        // shared holds up to (seq+1) logits; cap at threads to bound memory.
        const size_t smem = (static_cast<size_t>(seq) + 1) * sizeof(float);
        full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
            d_q, d_k, d_v, d_attn, seq, NH, HD, NKV, scale);
        e = cudaGetLastError(); if (e) return e;
    }

    // 7. Output gate: attn * sigmoid(gate).
    {
        const int64_t n = (int64_t)seq * NH * HD;
        const int threads = 256;
        qwen_out_gate_kernel<<<(int)((n + threads - 1) / threads), threads,
                               0, stream>>>(d_attn, d_gate, d_attn, seq, NH * HD);
        e = cudaGetLastError(); if (e) return e;
    }

    // 8. o_proj: [seq, NH*HD] -> [seq, H].
    e = LaunchNvfp4Gemm(lw.o, d_attn, seq, d_out, stream);
    return e;
}

// ── Linear attention (GatedDeltaNet) layer ─────────────────────────────────
// scratch (halves unless noted):
//   d_qkv : [seq, KD*2+VD]  in_proj_qkv output
//   d_z   : [seq, VD]       in_proj_z output (RMSNormGated gate)
//   d_g   : [seq, NVR] float decay (-exp(A)*softplus(a+dt))  [reused as fp32]
//   d_beta: [seq, NVR] float beta = sigmoid(b)
//   d_conv: [seq, KD*2+VD]  conv1d output (= qkv post-conv)
//   d_core: [seq, NVR*VD]   gated-delta output (pre-RMSNormGated)
//   d_gated:[seq, NVR*VD]   RMSNormGated output
// NVR = linear_num_value_heads.
cudaError_t RunQwenLinearAttention(cudaStream_t stream,
                                   const QwenLayerWeights &lw,
                                   const QwenTopology &top, int seq,
                                   const __half *d_normed, __half *d_out,
                                   __half *d_qkv, __half *d_z, __half *d_ab,
                                   float *d_g, float *d_beta,
                                   __half *d_conv, __half *d_core,
                                   __half *d_gated)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int nk = top.linear_num_key_heads;
    const int kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads;
    const int vd = top.linear_value_head_dim;
    const int KD = nk * kd;
    const int VD = nv * vd;
    const int S = KD * 2 + VD;
    if (nk <= 0 || kd <= 0 || nv <= 0 || vd <= 0) return cudaErrorInvalidValue;

    cudaError_t e;
    // 1. qkv, z projections via NVFP4 GEMM.
    e = LaunchNvfp4Gemm(lw.qkv, d_normed, seq, d_qkv, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.z,   d_normed, seq, d_z,   stream);   if (e) return e;

    // 2. conv1d over S channels.
    if (lw.d_conv1d && lw.conv_dim == S) {
        const int threads = 128;
        qwen_conv1d_kernel<<<(S + threads - 1) / threads, threads, 0, stream>>>(
            d_qkv, lw.d_conv1d, d_conv, seq, S, lw.conv_kernel);
        e = cudaGetLastError(); if (e) return e;
    } else {
        cudaMemcpyAsync(d_conv, d_qkv, (int64_t)seq * S * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. a,b projections into d_ab [seq*nv*2].
    __half *a_buf = d_ab;
    __half *b_buf = d_ab + (int64_t)seq * nv;
    e = LaunchNvfp4Gemm(lw.a, d_normed, seq, a_buf, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.b, d_normed, seq, b_buf, stream); if (e) return e;
    {
        const int threads = 256;
        const int64_t total = (int64_t)seq * nv;
        qwen_prep_gating_kernel<<<(int)((total + threads - 1) / threads), threads,
                                  0, stream>>>(
            a_buf, b_buf, lw.d_A_log, lw.d_dt_bias, d_g, d_beta, total, nv);
        e = cudaGetLastError(); if (e) return e;
    }

    // 4. l2-normalize q,k rows in place (HF: l2norm eps=1e-6, q scaled by
    //    1/sqrt(kd)), then run the gated delta rule. q/k/v are slices of the
    //    strided conv buffer [seq, S], so the recurrence must use stride S.
    {
        if (kd > 256) return cudaErrorInvalidValue;
        int nthreads = 1;
        while (nthreads < kd) nthreads <<= 1;
        const float q_scale = 1.0f / sqrtf((float)kd);
        const dim3 norm_grid(seq, nk);
        qwen_l2norm_rows_kernel<<<norm_grid, nthreads, 0, stream>>>(
            d_conv, seq, nk, kd, S, 0, q_scale);
        e = cudaGetLastError(); if (e) return e;
        qwen_l2norm_rows_kernel<<<norm_grid, nthreads, 0, stream>>>(
            d_conv, seq, nk, kd, S, KD, 1.0f);
        e = cudaGetLastError(); if (e) return e;

        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const int r = nv / nk;
        // one block per (t, v_head)
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, S);
        e = cudaGetLastError(); if (e) return e;
    }

    // 5. RMSNormGated(core, z, norm) -> d_gated.
    {
        const int threads = std::min(vd, 256);
        rmsnorm_gated_kernel<<<seq * nv, threads, vd * sizeof(float), stream>>>(
            d_core, d_z, lw.d_attn_norm, d_gated, seq * nv, vd,
            top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 6. out_proj: [seq, VD] -> [seq, H].
    e = LaunchNvfp4Gemm(lw.out, d_gated, seq, d_out, stream);
    return e;
}
// ── Generation loop ─────────────────────────────────────────────────────────
// Greedy argmax over the last row of the logits matrix [seq, V].
__global__ void qwen_argmax_kernel(const __half *logits, int seq, int V,
                                   int32_t *out)
{
    const int64_t base = (int64_t)(seq - 1) * V;
    float best = -1e30f;
    int32_t bidx = -1;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float lv = __half2float(logits[base + v]);
        if (lv > best) { best = lv; bidx = v; }
    }
    __shared__ float sb[256];
    __shared__ int32_t si[256];
    sb[threadIdx.x] = best;
    si[threadIdx.x] = bidx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sb[threadIdx.x + s] > sb[threadIdx.x]) {
                sb[threadIdx.x] = sb[threadIdx.x + s];
                si[threadIdx.x] = si[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = si[0] >= 0 ? si[0] : 0;
}

cudaError_t RunQwenGeneration(cudaStream_t stream,
                              const QwenPinnedWeights &pw,
                              const std::vector<int32_t> &prompt, int max_new,
                              int eos_id,
                              QwenGenerationResult &result)
{
    const QwenTopology &top = pw.top;
    const int H = top.hidden_size, I = top.intermediate_size;
    const int NH = top.num_attention_heads, HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    const int V = top.vocab_size;
    const int rotary_dim = (int)(HD * top.partial_rotary_factor);
    result = QwenGenerationResult{};
    if (H <= 0 || I <= 0 || V <= 0 || prompt.empty() || max_new <= 0) {
        result.error_msg = "invalid generation inputs";
        return cudaErrorInvalidValue;
    }
    if (eos_id < 0) {
        result.error_msg = "invalid eos_id";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.head.d_packed) {
        result.error_msg = "embed/head weights missing";
        return cudaErrorInvalidValue;
    }
    const int max_seq = (int)prompt.size() + max_new;

    // ── arena (all halves unless noted; single cudaMalloc) ──
    size_t half_sz[] = {
        (size_t)max_seq * H,             // d_act
        (size_t)max_seq * H,             // d_normed (input)
        (size_t)max_seq * H,             // d_attn_out
        (size_t)max_seq * H,             // d_post_in
        (size_t)max_seq * H,             // d_mlp_out
        (size_t)max_seq * I,             // d_gate
        (size_t)max_seq * I,             // d_up
        (size_t)max_seq * NH * HD * 2,   // d_qc
        (size_t)max_seq * NH * HD,       // d_q
        (size_t)max_seq * NH * HD,       // d_qgate
        (size_t)max_seq * NKV * HD,      // d_k
        (size_t)max_seq * NKV * HD,      // d_v
        (size_t)max_seq * NH * HD,       // d_attn
        (size_t)max_seq * rotary_dim,    // d_cos
        (size_t)max_seq * rotary_dim,    // d_sin
        (size_t)max_seq * S,             // d_qkv
        (size_t)max_seq * VD,            // d_z
        (size_t)max_seq * nv * 2,        // d_ab
        (size_t)max_seq * S,             // d_conv
        (size_t)max_seq * VD,            // d_core
        (size_t)max_seq * VD,            // d_gated
        (size_t)V * H,                   // d_embed_half
        (size_t)V * H,                   // d_head_half
        (size_t)max_seq * V,             // d_logits
        (size_t)max_seq * 2,             // d_ids (int32: 2 halves each)
        (size_t)max_seq * 2,             // d_pos (int32: 2 halves each)
    };
    size_t float_sz[] = {
        (size_t)max_seq * nv,            // d_gf
        (size_t)max_seq * nv,            // d_betaf
    };
    size_t half_bytes = 0;
    for (auto s : half_sz) half_bytes += ((s + 7) / 8) * 8 * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    size_t float_bytes = 0;
    for (auto s : float_sz) float_bytes += s * sizeof(float);
    float_bytes = (float_bytes + 15) & ~static_cast<size_t>(15);

    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, half_bytes + float_bytes);
    if (e != cudaSuccess) { result.error_msg = "arena alloc failed"; return e; }

    auto adv = [&](__half *&p, size_t hs) {
        __half *r = p;
        p += ((hs + 7) / 8) * 8;   // align next buffer to 16 bytes (8 halves)
        return r;
    };
    __half *ph = reinterpret_cast<__half*>(base);
    __half *d_act = adv(ph, half_sz[0]);
    __half *d_normed = adv(ph, half_sz[1]);
    __half *d_attn_out = adv(ph, half_sz[2]);
    __half *d_postin = adv(ph, half_sz[3]);
    __half *d_mlp_out = adv(ph, half_sz[4]);
    __half *d_gate = adv(ph, half_sz[5]);
    __half *d_up = adv(ph, half_sz[6]);
    __half *d_qc = adv(ph, half_sz[7]);
    __half *d_q = adv(ph, half_sz[8]);
    __half *d_qgate = adv(ph, half_sz[9]);
    __half *d_k = adv(ph, half_sz[10]);
    __half *d_v = adv(ph, half_sz[11]);
    __half *d_attn = adv(ph, half_sz[12]);
    __half *d_cos = adv(ph, half_sz[13]);
    __half *d_sin = adv(ph, half_sz[14]);
    __half *d_qkv = adv(ph, half_sz[15]);
    __half *d_z = adv(ph, half_sz[16]);
    __half *d_ab = adv(ph, half_sz[17]);
    __half *d_conv = adv(ph, half_sz[18]);
    __half *d_core = adv(ph, half_sz[19]);
    __half *d_gated = adv(ph, half_sz[20]);
    __half *d_embed_h = adv(ph, half_sz[21]);
    __half *d_head_h = adv(ph, half_sz[22]);
    __half *d_logits = adv(ph, half_sz[23]);
    int32_t *d_ids = reinterpret_cast<int32_t*>(adv(ph, half_sz[24]));
    int32_t *d_pos = reinterpret_cast<int32_t*>(adv(ph, half_sz[25]));
    float *pf = reinterpret_cast<float*>((uint8_t*)base + half_bytes);
    float *d_gf = pf; pf += float_sz[0];
    float *d_betaf = pf; pf += float_sz[1];

    // Dequant embed + head to half for gather / logits GEMM.
    e = lsxcommon::gpu::dequant_shared_nvfp4(stream, pw.embed.d_packed, pw.embed.packed_bytes,
                             pw.embed.d_scales, pw.embed.scale_bytes,
                             pw.embed.global_scale, d_embed_h,
                             (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant embed"; cudaFree(base); return e; }
    e = lsxcommon::gpu::dequant_shared_nvfp4(stream, pw.head.d_packed, pw.head.packed_bytes,
                             pw.head.d_scales, pw.head.scale_bytes,
                             pw.head.global_scale, d_head_h,
                             (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant head"; cudaFree(base); return e; }

    int32_t *d_argmax = nullptr;
    e = cudaMalloc(&d_argmax, sizeof(int32_t));
    if (e != cudaSuccess) { result.error_msg="argmax alloc"; cudaFree(base); return e; }

    // Wall-clock timing (stream-delta): t_start at entry, t_ttft at the first
    // generated token, t_stop after the final decode row.
    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_ttft);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, stream);
    auto fail = [&](cudaError_t err, std::string msg) -> cudaError_t {
        result.error_msg = std::move(msg);
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        cudaFree(d_argmax);
        cudaFree(base);
        return err;
    };

    std::vector<int32_t> ids = prompt;
    std::vector<int32_t> pos(max_seq);
    for (int i = 0; i < max_seq; ++i) pos[i] = i;
    const double eps = top.rms_norm_eps;
    const int threads = 256;

    for (int step = 0; step < max_new; ++step) {
        const int seq = (int)ids.size();
        std::vector<int32_t> posb(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) posb[i] = (int)i;
        cudaMemcpyAsync(d_ids, ids.data(), ids.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_pos, posb.data(), posb.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);

        e = RunQwenEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
        if (e != cudaSuccess) return fail(e, "embed");

        for (int l = 0; l < (int)pw.layers.size(); ++l) {
            const QwenLayerWeights &lw = pw.layers[l];
            // input RMSNorm
            e = RunQwenRMSNorm(stream, d_act, lw.d_input_norm, d_normed,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, std::string("inputnorm l")+std::to_string(l));
            if (lw.is_linear) {
                e = RunQwenLinearAttention(stream, lw, top, seq,
                    d_normed, d_attn_out, d_qkv, d_z, d_ab,
                    d_gf, d_betaf, d_conv, d_core, d_gated);
            } else {
                e = RunQwenFullAttention(stream, lw, top, seq,
                    d_normed, d_attn_out, d_qc, d_q, d_qgate,
                    d_k, d_v, d_attn, d_cos, d_sin, d_pos);
            }
            if (e != cudaSuccess) return fail(e, std::string("attn l")+std::to_string(l));
            // residual: d_act = d_act + d_attn_out
            {
                const int64_t n = (int64_t)seq * H;
                qwen_add_kernel<<<(int)((n + threads - 1) / threads), threads,
                                  0, stream>>>(d_attn_out, d_act, n);
                e = cudaGetLastError(); if (e) return fail(e, std::string("res1 l")+std::to_string(l));
            }
            // post RMSNorm
            e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, std::string("postnorm l")+std::to_string(l));
            // MLP
            e = RunQwenMLP(stream, lw, seq, d_postin, d_mlp_out,
                           d_gate);
            if (e != cudaSuccess) return fail(e, std::string("mlp l")+std::to_string(l));
            // residual: d_act = d_act + d_mlp_out
            {
                const int64_t n = (int64_t)seq * H;
                qwen_add_kernel<<<(int)((n + threads - 1) / threads), threads,
                                  0, stream>>>(d_mlp_out, d_act, n);
                e = cudaGetLastError(); if (e) return fail(e, std::string("res2 l")+std::to_string(l));
            }
        }

        // final norm
        e = RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                           seq, H, eps);
        if (e != cudaSuccess) return fail(e, "finalnorm");
        // head logits: LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits)
        e = LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits, stream);
        if (e != cudaSuccess) return fail(e, "head");
        // argmax last row
        qwen_argmax_kernel<<<1, std::min(256, V), 0, stream>>>(
            d_logits, seq, V, d_argmax);
        e = cudaGetLastError();
        if (e != cudaSuccess) return fail(e, "argmax");

        // Publish first generated token id (async, synchronized via event below).
        int32_t tok = 0;
        cudaEvent_t tok_ready;
        cudaEventCreate(&tok_ready);
        cudaMemcpyAsync(&tok, d_argmax, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(tok_ready, stream);
        cudaEventSynchronize(tok_ready);
        cudaEventDestroy(tok_ready);
        if (step == 0) cudaEventRecord(t_ttft, stream);
        result.tokens.push_back(tok);
        if (tok == eos_id) {
            result.eos_terminated = true;
            break;
        }
        ids.push_back(tok);
    }

    // Publish last-row logits (final row) for the regression harness, while
    // the arena is still alive.
    {
        std::vector<float> hlast_f(V);
        int64_t base_idx = (int64_t)ids.size() - 1;
        if (base_idx < 0) base_idx = 0;
        base_idx *= V;
        std::vector<__half> hlast(V);
        cudaMemcpy(hlast.data(), d_logits + base_idx, V * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i)
            hlast_f[i] = __half2float(hlast[i]);
        result.final_logits = std::move(hlast_f);
    }

    // Publish timing and residency before tearing down the arena.
    cudaEventRecord(t_stop, stream);
    cudaEventSynchronize(t_stop);
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    result.resident_bytes = pw.resident_bytes + half_bytes
                            + float_bytes + sizeof(int32_t);
    cudaFree(d_argmax);
    cudaFree(base);
    result.ok = true;
    return cudaSuccess;
}

} // namespace qwen
} // namespace lsxcommon