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

__device__ __forceinline__ float softplus_dev(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
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
    const uint8_t *wrow = packed + (int64_t)n * K;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int idx2 = n * K + k;
        const uint8_t nib = (idx2 & 1) ? (wrow[k] >> 4) : (wrow[k] & 0x0F);
        const float scl = qwen_e4m3_scale(scales[idx2 >> 4]) * global_scale;
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
    rmsnorm_offset_kernel<<<rows, hidden, hidden * sizeof(float), stream>>>(
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

// Gated delta rule, block per (t, v_head), thread per value-dim d.
// Columns d are mutually independent across tt (kv[d] uses only column d of S,
// and updates only column d) -> one thread owns the whole column [kd]. Only the
// q/k l2-norm reduction is shared across the block.
// Ordering mirrors HF torch_recurrent_gated_delta_rule:
//   q,k l2-normalised (eps 1e-6), q scaled by 1/sqrt(kd);
//   S=0; for tt in 0..t: S*=exp(g); kv=sum_l S[l]*ktt[l]; Δ=(v-kv)β;
//        S[l]+=ktt[l]*Δ;  out=sum_l S[l]*q[l].
__global__ void qwen_gated_delta_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, __half *__restrict__ out,
    int seq, int nk, int kd, int nv, int vd, int r, float scale)
{
    const int t = blockIdx.x;
    const int vh = blockIdx.y;
    const int kh = vh / r;         // key head shared by r value heads
    const int d = threadIdx.x;     // value-dim column [0, vd)
    if (d >= vd) return;

    // Registers hold this thread's column [l in 0..kd).
    __shared__ float qn[256];
    __shared__ float kttn_s[256];

    const __half *qrow = q + ((int64_t)t * nk + kh) * kd;
    const __half *krow = k + ((int64_t)t * nk + kh) * kd;

    // Cooperative l2-norm of q,k over kd.
    __shared__ float qdot, kdot;
    float lq = 0.0f, lk = 0.0f;
    for (int l = d; l < kd; l += blockDim.x) {
        float qv = __half2float(qrow[l]);
        float kv2 = __half2float(krow[l]);
        lq += qv * qv; lk += kv2 * kv2;
    }
    // reduce across threads
    __shared__ float swap[256];
    if (d < blockDim.x) swap[d] = lq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) swap[d] += swap[d + s];
        __syncthreads();
    }
    if (d == 0) qdot = swap[0];
    __syncthreads();
    if (d < blockDim.x) swap[d] = lk;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) swap[d] += swap[d + s];
        __syncthreads();
    }
    if (d == 0) kdot = swap[0];
    __syncthreads();
    const float isq = rsqrtf(qdot + 1e-6f) * scale;
    const float isk = rsqrtf(kdot + 1e-6f);

    // Each thread keeps column array in registers.
    float col[256];
    for (int l = 0; l < kd; ++l) col[l] = 0.0f;
    if (d == 0) {
        for (int l = 0; l < kd; ++l) qn[l] = __half2float(qrow[l]) * isq;
    }
    __syncthreads();

    for (int tt = 0; tt <= t; ++tt) {
        const float decay = __expf(g[tt * nv + vh]);
        const float betaf = beta[tt * nv + vh];
        const __half *ktt = k + ((int64_t)tt * nk + kh) * kd;
        const __half *vtt = v + ((int64_t)tt * nv + vh) * vd;

        if (d == 0)
            for (int l = 0; l < kd; ++l) kttn_s[l] = __half2float(ktt[l]);
        __syncthreads();

        for (int l = 0; l < kd; ++l) col[l] *= decay;
        float kv = 0.0f;
        for (int l = 0; l < kd; ++l) kv += col[l] * kttn_s[l];
        const float delta = (__half2float(vtt[d]) - kv) * betaf;
        for (int l = 0; l < kd; ++l) col[l] += kttn_s[l] * delta;
        __syncthreads();
    }

    float acc = 0.0f;
    for (int l = 0; l < kd; ++l) acc += col[l] * qn[l];
    out[((int64_t)t * nv + vh) * vd + d] = __float2half(acc);
}

} // namespace

cudaError_t RunQwenMLP(cublasHandle_t cublas_h, cudaStream_t stream,
                       const QwenLayerWeights &lw, int seq,
                       const __half *d_in, __half *d_out, __half *d_scratch)
{
    (void)cublas_h;
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
cudaError_t RunQwenFullAttention(cublasHandle_t cublas_h, cudaStream_t stream,
                                 const QwenLayerWeights &lw,
                                 const QwenTopology &top, int seq,
                                 const __half *d_normed, __half *d_out,
                                 __half *d_qc, __half *d_q, __half *d_gate,
                                 __half *d_k, __half *d_v, __half *d_attn,
                                 __half *d_cos, __half *d_sin,
                                 int32_t *d_pos)
{
    (void)cublas_h;
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

    // 5. RoPE on q,k (first rotary_dim dims).
    e = RunQwenRopeCosSin(stream, d_pos, seq, rotary_dim, top.rope_theta,
                          d_cos, d_sin); if (e) return e;
    e = RunQwenPartialRope(stream, d_q, d_k, d_cos, d_sin, d_q, d_k,
                           seq, NH, HD, rotary_dim); if (e) return e;

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
cudaError_t RunQwenLinearAttention(cublasHandle_t cublas_h, cudaStream_t stream,
                                   const QwenLayerWeights &lw,
                                   const QwenTopology &top, int seq,
                                   const __half *d_normed, __half *d_out,
                                   __half *d_qkv, __half *d_z, __half *d_ab,
                                   float *d_g, float *d_beta,
                                   __half *d_conv, __half *d_core,
                                   __half *d_gated)
{
    (void)cublas_h;
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

    // 4. Split q/k/v from d_conv and run gated delta rule.
    {
        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const float scale = 1.0f / sqrtf((float)kd);
        const int r = nv / nk;
        // one block per (t, v_head)
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, scale);
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

} // namespace qwen
} // namespace lsxcommon