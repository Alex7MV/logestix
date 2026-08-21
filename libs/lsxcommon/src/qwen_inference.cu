#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/sampling.h"
#include "nvfp4_codec.h"

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
// causal: out[t][c] = sum_{k=0..kernel-1} mixed[kv_base+t-k][c] * w[c][k]
//         (skip rows < 0). kv_base offsets the history read for incremental
//         decode; kv_base == 0 keeps the transient full-recompute semantics.
__global__ void qwen_conv1d_kernel(const __half *mixed, const __half *w,
                                   __half *out, int seq, int conv_dim,
                                   int kernel, int kv_base)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int tt = kv_base + t - k;
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
    __half *out, int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int ctx = kv_base + t;
    const int nscores = ctx + 1;
    const int threads = blockDim.x;

    const __half *qrow = q + ((int64_t)t * heads + h) * head_dim;
    const __half *kbase = k + (int64_t)kvh * head_dim;
    const __half *vbase = v + (int64_t)kvh * head_dim;

    extern __shared__ float slogits[];      // dynamic: [nscores]
    __shared__ float sreduce[512];          // separate block-reduction scratch

    // 1. Scores, chunk-per-thread (parallel dots).
    for (int j = threadIdx.x; j < nscores; j += threads) {
        const __half *krow = kbase + (int64_t)j * kv_heads * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qrow[d]) * __half2float(krow[d]);
        slogits[j] = dot * scale;
    }
    __syncthreads();

    // 2. Global max via block reduction (scratch, NOT slogits).
    float m = -1e30f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        m = fmaxf(m, slogits[j]);
    sreduce[threadIdx.x] = m;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] = fmaxf(sreduce[threadIdx.x], sreduce[threadIdx.x + stride]);
        __syncthreads();
    }
    m = sreduce[0];
    __syncthreads();

    // 3. Softmax denominator via block reduction (one exp per score).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        sum += __expf(slogits[j] - m);
    sreduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / (sreduce[0] + 1e-10f);
    __syncthreads();

    // 4. Per-output-dim value pass: thread d sums all scores with the SHARED
    //    exp(s-m)*inv. Stride-d loop (idles when threads > head_dim).
    for (int d = threadIdx.x; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int j = 0; j < nscores; ++j) {
            const __half *vrow = vbase + (int64_t)j * kv_heads * head_dim;
            acc += __expf(slogits[j] - m) * inv * __half2float(vrow[d]);
        }
        out[((int64_t)t * heads + h) * head_dim + d] = __float2half(acc);
    }
}

} // namespace

// ── Public numeric kernels ──────────────────────────────────────────────────

// Block-wide chunked-parallel causal attention (GQA). Serves as the numeric
// entry for the standalone softmax-kernel launch (the high-level
// RunQwenFullAttention calls the same kernel with projections/RoPE/gate).
cudaError_t RunQwenFullAttnKernel(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale, int kv_base)
{
    if (!q || !k || !v || !out || seq <= 0 || heads <= 0 || head_dim <= 0 ||
        kv_heads <= 0 || kv_base < 0)
        return cudaErrorInvalidValue;
    const dim3 grid(seq, heads);
    const int threads = std::min(head_dim, 512);
    // dynamic shared = slogits[nscores]; the block REDUCTION scratch is a
    // static __shared__ (sreduce[512]) and needs no dynamic smem.
    const size_t smem = (static_cast<size_t>(kv_base) + seq + 1) * sizeof(float);
    full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
        q, k, v, out, seq, heads, head_dim, kv_heads, scale, kv_base);
    return cudaGetLastError();
}

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

// Shared NVFP4 decode primitives (nvfp4_codec.h). Aliased into this anonymous
// namespace so existing call sites stay unchanged.
using gpu::nvfp4::kE2M1Table;
using gpu::nvfp4::E4M3ScaleToFloat;

// Fused NVFP4 dense GEMM: y[t][n] = sum_k x[t*K+k] * W[n*K+k], where W is the
// [N, K] row-major weight with per-16 E4M3 block scales + global_scale.
// x: [seq, K] FP16; y: [seq, N] FP16 (FP32 accumulate).
__global__ void qwen_nvfp4_gemm_kernel(
    const __half *x, const uint8_t *packed, const uint8_t *scales,
    float global_scale, __half *y, int seq, int K, int N, bool acc_in)
{
    const int64_t total = (int64_t)seq * N;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int n = (int)(idx % N);
    const int t = (int)(idx / N);

    const __half *xrow = x + (int64_t)t * K;

    float acc = acc_in ? __half2float(y[idx]) : 0.0f;
    for (int k = 0; k < K; ++k) {
        // NVFP4 layout: element e lives at byte e>>1, low nibble if even,
        // high nibble if odd; scale is per-16 elements (e>>4).
        const int64_t e = (int64_t)n * K + k;
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        const float scl = E4M3ScaleToFloat(scales[e >> 4]) * global_scale;
        acc += __half2float(xrow[k]) * kE2M1Table[nib] * scl;
    }
    y[idx] = __float2half(acc);
}

// Tile NVFP4 dense GEMM: block computes an [BM x BN] output tile for one token
// tile. X rows and the NVFP4-dequantized weight rows are staged in shared
// memory once per K-block; E8M0 scales are decoded once per 16 elements.
// One thread owns exactly one output element (threads == BM * BN), so `acc` is
// a single register per thread — no per-thread array, no bank-pressure.
template <int BM, int BN, int BK>
__global__ void qwen_nvfp4_gemm_tile_kernel(
    const __half *x, const uint8_t *packed, const uint8_t *scales,
    float global_scale, __half *y, int seq, int K, int N, bool acc_in)
{
    static_assert(BM * BN == 256, "one thread per output element (256)");

    __shared__ __half sX[BM][BK];
    __shared__ __half sW[BN][BK];
    const int m0 = blockIdx.x * BM;      // token tile
    const int n0 = blockIdx.y * BN;      // output-column tile
    const int bi = threadIdx.x;          // 0 .. BM*BN-1
    const int bm = bi / BN;              // token index within tile
    const int bn = bi % BN;              // output-column index within tile

    const int t = m0 + bm;
    const int n = n0 + bn;
    const bool in_bounds = (t < seq) && (n < N);

    float acc = acc_in && in_bounds ? __half2float(y[(int64_t)t * N + n]) : 0.0f;

    for (int kb = 0; kb < K; kb += BK) {
        // W tile: element e = (n0+bn)*K + kb+kk (weight row-major [n, K]).
        // K-tail (K % BK != 0) and N-tail are zero-padded: the accumulate
        // loop below runs the full BK width unmasked.
        for (int i = bi; i < BN * BK; i += BM * BN) {
            const int w_bn = i / BK, kk = i % BK;
            const int n = n0 + w_bn;
            float wv = 0.0f;
            if (n < N && kb + kk < K) {
                const int64_t e = (int64_t)n * K + (kb + kk);
                const uint8_t byte = packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl = E4M3ScaleToFloat(scales[e >> 4]) * global_scale;
                wv = kE2M1Table[nib] * scl;
            }
            sW[w_bn][kk] = __float2half(wv);
        }
        // X tile: x[t][kb+kk] for rows [m0, m0+BM). K-tail zero-padded.
        for (int i = bi; i < BM * BK; i += BM * BN) {
            const int x_bm = i / BK, kk = i % BK;
            const int tt = m0 + x_bm;
            sX[x_bm][kk] = (tt < seq && kb + kk < K)
                               ? x[(int64_t)tt * K + (kb + kk)]
                               : __float2half(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
            acc += __half2float(sX[bm][kk]) * __half2float(sW[bn][kk]);

        __syncthreads();
    }

    if (in_bounds)
        y[(int64_t)t * N + n] = __float2half(acc);
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
// A non-null `col_seed` (per-linear-layer slice of the persistent state)
// initialises col to the previous call's accumulator (incremental decode) and
// writes the final state back from the last row's block (t == seq-1).
__global__ void qwen_gated_delta_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, __half *__restrict__ out,
    int seq, int nk, int kd, int nv, int vd, int r, int stride,
    float *__restrict__ col_seed)
{
    const int t = blockIdx.x;
    const int vh = blockIdx.y;
    const int kh = vh / r;         // key head shared by r value heads
    const int d = threadIdx.x;     // value-dim column [0, vd)
    if (d >= vd) return;

    const int seed_off = (vh * vd + d) * kd;

    // Registers hold this thread's column [l in 0..kd).
    float col[256];
    for (int l = 0; l < kd; ++l)
        col[l] = col_seed ? col_seed[seed_off + l] : 0.0f;

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

    if (col_seed && t == seq - 1) {
        for (int l = 0; l < kd; ++l) col_seed[seed_off + l] = col[l];
    }

    float acc = 0.0f;
    for (int l = 0; l < kd; ++l) acc += col[l] * __half2float(qrow[l]);
    out[((int64_t)t * nv + vh) * vd + d] = __float2half(acc);
}

// ── Fused decoder (seq=1) full-attention segments ─────────────────────────
// Stage 1: per-head q/k/v+gate (inline NVFP4 dequant to registers) + per-head
// RMSNorm(q)/(k) + inline RoPE (rotate-half pairing, mirrors
// partial_rope_kernel) + append k/v to persistent KV row kv_base.
__global__ void fused_qkv_kernel(
    const __half* __restrict__ x,          // input token [H] (already input-normed)
    const QwenNvfp4Weight qw, const QwenNvfp4Weight kw, const QwenNvfp4Weight vw,
    const __half* __restrict__ q_norm, const __half* __restrict__ k_norm,
    double theta, int rotary_dim, int kv_base,
    int NH, int NKV, int HD, int H, float eps,
    __half* __restrict__ kv_k, __half* __restrict__ kv_v,
    __half* __restrict__ q_gl, __half* __restrict__ gate_gl)
{
    __shared__ float sq[512];  // pair staging for RoPE q (rotary_dim<=512 assumed)
    __shared__ float sk[512];
    __shared__ float red[512];
    const int h = blockIdx.x;
    const int kvh = h / (NH / NKV);
    const int d = threadIdx.x;
    if (d >= HD) return;

    // 1. q/k/v/gate projections for head h (inline NVFP4 per element).
    const int qcol = h*HD + d;             // query column
    const int gcol = NH*HD + h*HD + d;     // gate column (second NH*HD half)
    const int kcol = kvh*HD + d;           // key/value column
    float qv = 0.0f, gv = 0.0f, kv_ = 0.0f, vv = 0.0f;
    for (int k = 0; k < H; ++k) {
        const float xv = __half2float(x[k]);
        {
            const int64_t e = (int64_t)qcol * H + k;
            const uint8_t b = qw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            qv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(qw.d_scales[e>>4]) * qw.global_scale);
        }
        {
            const int64_t e = (int64_t)gcol * H + k;
            const uint8_t b = qw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            gv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(qw.d_scales[e>>4]) * qw.global_scale);
        }
        {
            const int64_t e = (int64_t)kcol * H + k;
            const uint8_t b = kw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            kv_ += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(kw.d_scales[e>>4]) * kw.global_scale);
        }
        {
            const int64_t e = (int64_t)kcol * H + k;
            const uint8_t b = vw.d_packed[e>>1];
            const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
            vv += xv * kE2M1Table[nib] * (E4M3ScaleToFloat(vw.d_scales[e>>4]) * vw.global_scale);
        }
    }
    // 2. per-head RMSNorm(q) and RMSNorm(k) — offset form.
    red[d] = qv*qv; __syncthreads();
    for (int st = HD/2; st>0; st>>=1){ if(d<st) red[d]+=red[d+st]; __syncthreads(); }
    const float qinv = rsqrtf(red[0]/(float)HD + eps);
    qv *= qinv * (1.0f + __half2float(q_norm[d]));
    red[d] = kv_*kv_; __syncthreads();
    for (int st = HD/2; st>0; st>>=1){ if(d<st) red[d]+=red[d+st]; __syncthreads(); }
    const float kinv = rsqrtf(red[0]/(float)HD + eps);
    kv_ *= kinv * (1.0f + __half2float(k_norm[d]));

    // 3. inline RoPE (rotate-half pairing) on first rotary_dim dims.
    //    Each thread computes ONLY its own dim `d`, reading the partner from
    //    the synced sq/sk staging (mirrors partial_rope_kernel).
    sq[d] = qv; sk[d] = kv_; __syncthreads();
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        double inv = 1.0 / std::pow(theta, (double)(2*(d % half))/(double)rotary_dim);
        double ang = (double)kv_base * inv;
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        if (d < half) {
            qv = sq[d]*c - sq[d+half]*s;      // role j,   out[j]   = x1*c - x2*s
            kv_ = sk[d]*c - sk[d+half]*s;
        } else {
            qv = sq[d]*c + sq[d-half]*s;      // role j+half, out[j+half] = x2*c + x1*s
            kv_ = sk[d]*c + sk[d-half]*s;
        }
    }
    __syncthreads();

    // 4. append k/v to persistent KV row kv_base, publish q/gate.
    kv_k[((int64_t)kv_base * NKV + kvh) * HD + d] = __float2half(kv_);
    kv_v[((int64_t)kv_base * NKV + kvh) * HD + d] = __float2half(vv);
    q_gl[(int64_t)h * HD + d]    = __float2half(qv);
    gate_gl[(int64_t)h * HD + d] = __float2half(gv);
}

// Stage 2: block-wide online-softmax causal GQA attention over attended KV rows
// [0..kv_base], per-head output gate (sigmoid). Reuses the validated
// sreduce[512] static + dynamic slogits reduction.
__global__ void fused_attn_kernel(
    const __half* __restrict__ q_gl, const __half* __restrict__ gate_gl,
    const __half* __restrict__ kv_k, const __half* __restrict__ kv_v,
    __half* __restrict__ out_h, int kv_base, int NH, int NKV, int HD, float scale)
{
    __shared__ float sreduce[512];
    extern __shared__ float slogits[];
    const int h = blockIdx.x;
    const int kvh = h / (NH / NKV);
    const int d = threadIdx.x;
    const int nscores = kv_base + 1;
    if (d >= HD) return;

    const __half* qrow = q_gl + (int64_t)h * HD;

    // scores (parallel chunks)
    for (int s = d; s < nscores; s += HD) {
        const __half* krow = kv_k + ((int64_t)s * NKV + kvh) * HD;
        float dot = 0.0f;
        for (int dd = 0; dd < HD; ++dd)
            dot += __half2float(qrow[dd]) * __half2float(krow[dd]);
        slogits[s] = dot * scale;
    }
    __syncthreads();
    // max block-reduce (sreduce)
    float m = -1e30f;
    for (int s = d; s < nscores; s += HD) m = fmaxf(m, slogits[s]);
    sreduce[d] = m; __syncthreads();
    for (int st = blockDim.x/2; st>0; st>>=1){ if(d<st) sreduce[d]=fmaxf(sreduce[d],sreduce[d+st]); __syncthreads(); }
    m = sreduce[0]; __syncthreads();
    // sum block-reduce
    float sumd = 0.0f;
    for (int s = d; s < nscores; s += HD) sumd += __expf(slogits[s] - m);
    sreduce[d] = sumd; __syncthreads();
    for (int st = blockDim.x/2; st>0; st>>=1){ if(d<st) sreduce[d]+=sreduce[d+st]; __syncthreads(); }
    const float inv = 1.0f / (sreduce[0] + 1e-10f); __syncthreads();
    // value + gate
    float acc = 0.0f;
    for (int s = 0; s < nscores; ++s) {
        const __half* vrow = kv_v + ((int64_t)s * NKV + kvh) * HD;
        acc += __expf(slogits[s] - m) * inv * __half2float(vrow[d]);
    }
    const float g = __half2float(gate_gl[h*HD + d]);
    out_h[(int64_t)h * HD + d] = __float2half(acc * (1.0f / (1.0f + expf(-g))));
}

// Stage 3: o_proj. out[n] = sum_k out_h[k] * dequant(o[n*K + k]) over ALL input
// rows k (K = NH*HD), exactly matching RunQwenNvfp4Gemm's [N,K] row-major
// layout. One thread per output column n; full-K FP32 dot, plain store.
__global__ void fused_o_proj_kernel(
    const __half* __restrict__ out_h,        // [NH*HD]
    const uint8_t* __restrict__ o_packed, const uint8_t* __restrict__ o_scales,
    float o_gs, float* __restrict__ out_acc, // [H], zeroed by caller
    int K, int H)                            // K = NH*HD input rows, H output cols
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float oh = __half2float(out_h[k]);
        const int64_t e = (int64_t)n * K + k;
        const uint8_t b = o_packed[e>>1];
        const uint8_t nib = (e&1) ? (b>>4) : (b&0xF);
        const float scl = E4M3ScaleToFloat(o_scales[e>>4]) * o_gs;
        acc += oh * kE2M1Table[nib] * scl;
    }
    out_acc[n] = acc;
}

// Convert a float accumulator row [H] to half out [H].
__global__ void qwen_f2h_acc_kernel(const float *in, __half *out, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

// ── Fused linear-attention front stage A: raw qkv NVFP4 GEMM into the
// persistent d_qkv history. One block per token row t; blockDim = pow2(kd).
// Own launch: the conv stage (stage B) reads history rows written by other
// tokens of the same call, which requires kernel-launch ordering.
__global__ void fused_linear_qkv_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const QwenNvfp4Weight qkv,              // rows=H, cols=S
    int seq, int nk, int kd, int nv, int vd, int H, int kv_base,
    __half * __restrict__ d_qkv)            // persistent raw history
{
    const int t = blockIdx.x;
    const int S = nk * kd * 2 + nv * vd;
    const int N = S;
    const int K = H;
    const int nth = blockDim.x;

    for (int c = threadIdx.x; c < N; c += nth) {
        float acc = 0.0f;
        const int64_t cbase = (int64_t)c * K;
        for (int k = 0; k < K; ++k) {
            const int64_t e = cbase + k;
            const uint8_t byte = qkv.d_packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl = E4M3ScaleToFloat(qkv.d_scales[e >> 4]) * qkv.global_scale;
            // Tile-GEMM parity: dequantized weight is rounded to half before
            // the product (mirrors qwen_nvfp4_gemm_tile_kernel sW staging).
            const __half wv = __float2half(kE2M1Table[nib] * scl);
            acc += __half2float(x[(int64_t)t * K + k]) * __half2float(wv);
        }
        d_qkv[((int64_t)kv_base + t) * S + c] = __float2half(acc);
    }
}

// ── Fused linear-attention front stage B: causal conv1d+silu + l2norm(q,k)
// for blocks [0, seq); z + a + b NVFP4 GEMMs + prep_gating for blocks
// [seq, 2*seq). The two halves are data-independent (conv reads d_qkv from
// stage A; z/a/b read d_normed directly), so they share one launch.
__global__ void fused_linear_conv_zab_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const QwenNvfp4Weight z,
    const QwenNvfp4Weight a, const QwenNvfp4Weight b,
    const float * __restrict__ A_log, const float * __restrict__ dt_bias,
    const __half * __restrict__ conv_w,     // [S, kernel]
    int seq, int nk, int kd, int nv, int vd, int H, int kernel, int kv_base,
    const __half * __restrict__ d_qkv,      // raw history from stage A
    __half * __restrict__ d_conv,           // [seq, S]
    __half * __restrict__ d_z, __half * __restrict__ d_ab,
    float * __restrict__ d_g, float * __restrict__ d_beta)
{
    const int VD = nv * vd;
    if (blockIdx.x < (unsigned)seq) {
        // ---- conv1d + silu + l2norm(q,k) for token blockIdx.x ----
        const int t = blockIdx.x;
        const int S = nk * kd * 2 + nv * vd;
        const int KD = nk * kd;
        for (int c = threadIdx.x; c < S; c += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < kernel; ++k) {
                const int tt = kv_base + t - k;
                if (tt < 0) continue;
                const float wv = __half2float(conv_w[(int64_t)c * kernel + (kernel - 1 - k)]);
                const float mv = __half2float(d_qkv[(int64_t)tt * S + c]);
                acc += wv * mv;
            }
            d_conv[(int64_t)t * S + c] = __float2half(silu_dev(acc));
        }
        __syncthreads();

        __shared__ float svals[256], sred[256], sinv;
        const float q_scale = 1.0f / sqrtf((float)kd);
        for (int h = 0; h < nk; ++h) {
            for (int slice = 0; slice < 2; ++slice) {
                const int head_off = (slice == 0) ? 0 : KD;
                const float extra = (slice == 0) ? q_scale : 1.0f;
                const int lane = threadIdx.x;
                const int c = head_off + h * kd + lane;
                float v = 0.0f;
                if (lane < kd) v = __half2float(d_conv[(int64_t)t * S + c]);
                svals[lane] = v; sred[lane] = v * v;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (lane < s) sred[lane] += sred[lane + s];
                    __syncthreads();
                }
                if (lane == 0) sinv = rsqrtf(sred[0] + 1e-6f) * extra;
                __syncthreads();
                if (lane < kd) d_conv[(int64_t)t * S + c] = __float2half(svals[lane] * sinv);
                __syncthreads();
            }
        }
    } else {
        // ---- z + a + b projections + prep gating for token blockIdx.x-seq ----
        const int t = blockIdx.x - (unsigned)seq;
        const int nth = blockDim.x;
        const __half *xrow = x + (int64_t)t * H;

        for (int c = threadIdx.x; c < VD; c += nth) {
            float acc = 0.0f;
            for (int k = 0; k < H; ++k) {
                const int64_t e = (int64_t)c * H + k;
                const uint8_t byte = z.d_packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl = E4M3ScaleToFloat(z.d_scales[e >> 4]) * z.global_scale;
                const __half wv = __float2half(kE2M1Table[nib] * scl);
                acc += __half2float(xrow[k]) * __half2float(wv);
            }
            d_z[(int64_t)t * VD + c] = __float2half(acc);
        }

        __half *a_buf = d_ab;
        __half *b_buf = d_ab + (int64_t)seq * nv;
        for (int c = threadIdx.x; c < nv; c += nth) {
            float acca = 0.0f, accb = 0.0f;
            for (int k = 0; k < H; ++k) {
                const int64_t ea = (int64_t)c * H + k;
                const uint8_t ba = a.d_packed[ea >> 1];
                const uint8_t na = (ea & 1) ? (ba >> 4) : (ba & 0x0F);
                const float sa = E4M3ScaleToFloat(a.d_scales[ea >> 4]) * a.global_scale;
                acca += __half2float(xrow[k]) *
                        __half2float(__float2half(kE2M1Table[na] * sa));
                const int64_t eb = (int64_t)c * H + k;
                const uint8_t bb = b.d_packed[eb >> 1];
                const uint8_t nb = (eb & 1) ? (bb >> 4) : (bb & 0x0F);
                const float sb = E4M3ScaleToFloat(b.d_scales[eb >> 4]) * b.global_scale;
                accb += __half2float(xrow[k]) *
                        __half2float(__float2half(kE2M1Table[nb] * sb));
            }
            a_buf[(int64_t)t * nv + c] = __float2half(acca);
            b_buf[(int64_t)t * nv + c] = __float2half(accb);
        }
        __syncthreads();

        for (int c = threadIdx.x; c < nv; c += nth) {
            const float a_v = __half2float(a_buf[(int64_t)t * nv + c]);
            const float b_v = __half2float(b_buf[(int64_t)t * nv + c]);
            d_beta[(int64_t)t * nv + c] = sigmoid_dev(b_v);
            d_g[(int64_t)t * nv + c] = -__expf(A_log[c]) * softplus_dev(a_v + dt_bias[c]);
        }
    }
}

} // namespace

// ── Fused linear-attention (GatedDeltaNet) front-end entry ──────────────────
// Fused F1 + F2 replace the split front of RunQwenLinearAttention; the
// gated-delta recurrence, RMSNormGated and out_proj tail are unchanged.
cudaError_t RunQwenFusedLinearAttnKernel(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base /* = 0 */, float *d_gd_state /* = nullptr */)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int nk = top.linear_num_key_heads;
    const int kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads;
    const int vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    if (nk <= 0 || kd <= 0 || nv <= 0 || vd <= 0) return cudaErrorInvalidValue;
    if (kd > 256) return cudaErrorInvalidValue;  // shared arrays sized [256]
    if (lw.qkv.cols <= 0) return cudaErrorInvalidValue;
    cudaError_t e;

    // Stage A: raw qkv GEMM into the persistent history (own launch — stage B
    // conv reads rows written by other tokens of this call).
    int nthreads = 1;
    while (nthreads < kd) nthreads <<= 1;
    fused_linear_qkv_kernel<<<seq, nthreads, 0, stream>>>(
        d_normed, lw.qkv, seq, nk, kd, nv, vd, lw.qkv.cols, kv_base, d_qkv);
    e = cudaGetLastError(); if (e) return e;

    // Stage B: conv1d+silu+l2norm(q,k) ∥ z+a+b+prep_gating (one launch).
    fused_linear_conv_zab_kernel<<<2 * seq, nthreads, 0, stream>>>(
        d_normed, lw.z, lw.a, lw.b, lw.d_A_log, lw.d_dt_bias, lw.d_conv1d,
        seq, nk, kd, nv, vd, lw.z.cols, lw.conv_kernel, kv_base,
        d_qkv, d_conv, d_z, d_ab, d_g, d_beta);
    e = cudaGetLastError(); if (e) return e;

    // Tail (unchanged, mirrors RunQwenLinearAttention stages 4-6).
    {
        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const int r = nv / nk;
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, S, d_gd_state);
        e = cudaGetLastError(); if (e) return e;
    }
    {
        const int threads = std::min(vd, 256);
        rmsnorm_gated_kernel<<<seq * nv, threads, vd * sizeof(float), stream>>>(
            d_core, d_z, lw.d_attn_norm, d_gated, seq * nv, vd,
            top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }
    return RunQwenNvfp4Gemm(stream, d_gated, lw.out.d_packed, lw.out.d_scales,
                            lw.out.global_scale, d_out, seq, lw.out.cols,
                            lw.out.rows);
}

// ── Fused decoder (seq=1) stage wrappers ────────────────────────────────────
cudaError_t RunQwenFusedQkv(
    cudaStream_t stream, const __half *x,
    const QwenNvfp4Weight &qw, const QwenNvfp4Weight &kw,
    const QwenNvfp4Weight &vw,
    const __half *q_norm, const __half *k_norm,
    double theta, int rotary_dim, int kv_base,
    int NH, int NKV, int HD, int H, float eps,
    const __half *input_norm,
    __half *kv_k, __half *kv_v, __half *q_gl, __half *gate_gl)
{
    if (!x || !qw.d_packed || !qw.d_scales || !kw.d_packed || !kw.d_scales ||
        !vw.d_packed || !vw.d_scales || !q_norm || !k_norm || !input_norm ||
        !kv_k || !kv_v || !q_gl || !gate_gl ||
        NH <= 0 || NKV <= 0 || HD <= 0 || H <= 0 || kv_base < 0 ||
        rotary_dim < 0 || rotary_dim > 512)
        return cudaErrorInvalidValue;
    __half *d_tmp;
    if (cudaMalloc(&d_tmp, (size_t)H * 2) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    const int th = std::min(H, 256);
    rmsnorm_offset_kernel<<<1, th, th * sizeof(float), stream>>>(
        x, input_norm, d_tmp, 1, H, (double)eps);
    cudaError_t e = cudaGetLastError();
    if (!e) {
        fused_qkv_kernel<<<NH, HD, 0, stream>>>(
            d_tmp, qw, kw, vw, q_norm, k_norm, theta, rotary_dim, kv_base,
            NH, NKV, HD, H, eps, kv_k, kv_v, q_gl, gate_gl);
        e = cudaGetLastError();
    }
    cudaFree(d_tmp);
    return e;
}

cudaError_t RunQwenFusedAttn(
    cudaStream_t stream,
    const __half *q_gl, const __half *gate_gl,
    const __half *kv_k, const __half *kv_v, __half *out_h,
    int kv_pos, int NH, int NKV, int HD, float scale)
{
    if (!q_gl || !gate_gl || !kv_k || !kv_v || !out_h ||
        NH <= 0 || NKV <= 0 || HD <= 0 || HD > 512 || kv_pos < 0)
        return cudaErrorInvalidValue;
    const size_t smem = ((size_t)kv_pos + 1) * sizeof(float);
    fused_attn_kernel<<<NH, HD, smem, stream>>>(
        q_gl, gate_gl, kv_k, kv_v, out_h, kv_pos, NH, NKV, HD, scale);
    return cudaGetLastError();
}

cudaError_t RunQwenFusedOProj(
    cudaStream_t stream, const __half *out_h,
    const QwenNvfp4Weight &o, float *out_acc, __half *out,
    int NH, int HD, int H)
{
    if (!out_h || !o.d_packed || !o.d_scales || !out_acc || !out ||
        NH <= 0 || HD <= 0 || H <= 0)
        return cudaErrorInvalidValue;
    cudaMemsetAsync(out_acc, 0, (size_t)H * sizeof(float), stream);
    cudaError_t e = cudaGetLastError();
    if (!e) {
        const int blocks = (int)((H + HD - 1) / HD);
        fused_o_proj_kernel<<<blocks, HD, 0, stream>>>(
            out_h, o.d_packed, o.d_scales, o.global_scale, out_acc,
            NH * HD, H);
        e = cudaGetLastError();
    }
    if (!e) {
        const int blocks = (int)((H + 255) / 256);
        qwen_f2h_acc_kernel<<<blocks, 256, 0, stream>>>(out_acc, out, H);
        e = cudaGetLastError();
    }
    return e;
}

// Public numeric entry: NVFP4 dense GEMM (y = x @ dequant(W)) with FP32
// accumulate. Dispatches to the shared-memory tile kernel for N>=8 && K>=16,
// else the per-element kernel (tiny shapes). External linkage so tests can
// call it directly.
cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc /* = false */)
{
    if (!x || !packed || !scales || !y || seq <= 0 || K <= 0 || N <= 0)
        return cudaErrorInvalidValue;
    if (N < 8 || K < 16) {
        const int threads = 256;
        const int64_t total = (int64_t)seq * N;
        const int blocks = (int)((total + threads - 1) / threads);
        qwen_nvfp4_gemm_kernel<<<blocks, threads, 0, stream>>>(
            x, packed, scales, global_scale, y, seq, K, N, acc);
        return cudaGetLastError();
    }
    constexpr int BM = 16, BN = 16, BK = 32;
    const int gx = (int)((seq + BM - 1) / BM);
    const int gy = (int)((N  + BN - 1) / BN);
    qwen_nvfp4_gemm_tile_kernel<BM, BN, BK><<<dim3(gx, gy), BM * BN, 0, stream>>>(
        x, packed, scales, global_scale, y, seq, K, N, acc);
    return cudaGetLastError();
}

cudaError_t LaunchNvfp4Gemm(const QwenNvfp4Weight &w, const __half *x, int seq,
                            __half *y, cudaStream_t stream)
{
    if (!w.d_packed || w.rows <= 0 || w.cols <= 0) return cudaErrorInvalidValue;
    return RunQwenNvfp4Gemm(stream, x, w.d_packed, w.d_scales,
                            w.global_scale, y, seq, w.cols, w.rows, /*acc=*/false);
}

// Fused decoder (seq=1) full-attention layer. Task-1 version composes the
// existing split launches (bit-identical reference); Tasks 2-4 replace the
// stages with the fused kernels. Numeric: matches the split path.
cudaError_t RunQwenFusedDecodeAttnKernel(
    cudaStream_t stream, const __half *x, const QwenLayerWeights &lw,
    const QwenTopology &top, int kv_base,
    __half *kv_k, __half *kv_v, __half *out)
{
    if (!x || !out || !kv_k || !kv_v || kv_base < 0 || !lw.loaded)
        return cudaErrorInvalidValue;
    if (!lw.d_input_norm || !lw.d_q_norm || !lw.d_k_norm)
        return cudaErrorInvalidValue;
    const int NH = top.num_attention_heads, HD = top.head_dim,
              NKV = top.num_key_value_heads, H = top.hidden_size;
    if (NH <= 0 || HD <= 0 || NKV <= 0 || H <= 0) return cudaErrorInvalidValue;
    if (HD > 512) return cudaErrorInvalidValue;
    const int rot = (int)(HD * top.partial_rotary_factor);
    const float scale = 1.0f / std::sqrt((float)HD);

    __half *d_q, *d_gate, *d_out_h;
    float *d_out_acc;
    if (cudaMalloc(&d_q,       (size_t)NH * HD * 2) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(&d_gate,    (size_t)NH * HD * 2) != cudaSuccess)
        { cudaFree(d_q); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_out_h,   (size_t)NH * HD * 2) != cudaSuccess)
        { cudaFree(d_q); cudaFree(d_gate); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_out_acc, (size_t)H * 4)       != cudaSuccess)
        { cudaFree(d_q); cudaFree(d_gate); cudaFree(d_out_h); return cudaErrorMemoryAllocation; }

    cudaError_t e = RunQwenFusedQkv(
        stream, x, lw.q, lw.k, lw.v, lw.d_q_norm, lw.d_k_norm,
        top.rope_theta, rot, kv_base, NH, NKV, HD, H,
        (float)top.rms_norm_eps, lw.d_input_norm, kv_k, kv_v, d_q, d_gate);
    if (!e)
        e = RunQwenFusedAttn(stream, d_q, d_gate, kv_k, kv_v, d_out_h,
                             kv_base, NH, NKV, HD, scale);
    if (!e)
        e = RunQwenFusedOProj(stream, d_out_h, lw.o, d_out_acc, out,
                              NH, HD, H);

    cudaFree(d_q); cudaFree(d_gate); cudaFree(d_out_h); cudaFree(d_out_acc);
    return e;
}

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

// Fused MLP layer. Takes d_in as the ALREADY post-RMSNormed input stream and
// accumulates the down-projection INTO d_resid (the residual stream) via the
// GEMM's acc=true path, absorbing the separate qwen_add residual. This mirrors
// RunQwenMLP except the final down-GEMM writes resid (d_resid) rather than a
// separate output buffer that the caller then adds. `top` is reserved for
// future use (geometry is read from `lw`).
cudaError_t RunQwenFusedMLP(cudaStream_t stream, const QwenLayerWeights &lw,
                            const QwenTopology &top, int seq,
                            const __half *d_in, __half *d_resid, __half *d_scratch)
{
    (void)top;
    if (!lw.loaded || !d_in || !d_resid || !d_scratch || seq <= 0)
        return cudaErrorInvalidValue;
    const int H = lw.mlp_gate.cols;   // input hidden dim
    const int I = lw.mlp_gate.rows;   // intermediate
    if (H <= 0 || I <= 0) return cudaErrorInvalidValue;
    if (!lw.mlp_gate.d_packed || !lw.mlp_up.d_packed || !lw.mlp_down.d_packed)
        return cudaErrorInvalidValue;

    __half *d_gate = d_scratch;                // [seq,I]
    __half *d_up   = d_scratch + (int64_t)seq * I;

    cudaError_t e;
    e = RunQwenNvfp4Gemm(stream, d_in, lw.mlp_gate.d_packed, lw.mlp_gate.d_scales,
                         lw.mlp_gate.global_scale, d_gate, seq, H, I,
                         /*acc=*/false);
    if (e) return e;
    e = RunQwenNvfp4Gemm(stream, d_in, lw.mlp_up.d_packed, lw.mlp_up.d_scales,
                         lw.mlp_up.global_scale, d_up, seq, H, I,
                         /*acc=*/false);
    if (e) return e;
    const int64_t n_act = (int64_t)seq * I;
    const int threads = 256;
    const int blocks = (int)((n_act + threads - 1) / threads);
    qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_gate, n_act);
    e = cudaGetLastError();
    if (e) return e;
    return RunQwenNvfp4Gemm(stream, d_gate, lw.mlp_down.d_packed,
                            lw.mlp_down.d_scales, lw.mlp_down.global_scale,
                            d_resid, seq, I, H, /*acc=*/true);
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
                                 int32_t *d_pos, int kv_base)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int NH = top.num_attention_heads;
    const int HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int H = top.hidden_size;
    const int rotary_dim = (int)(HD * top.partial_rotary_factor);
    if (NH <= 0 || HD <= 0 || NKV <= 0 || H <= 0) return cudaErrorInvalidValue;

    // kv rows are appended at [kv_base..kv_base+seq) when kv_base > 0
    // (persistent cache) or [0..seq) when kv_base == 0 (transient arena).
    __half *k_row0 = d_k + (int64_t)kv_base * NKV * HD;
    __half *v_row0 = d_v + (int64_t)kv_base * NKV * HD;

    cudaError_t e;
    // 1. q_proj: [seq, H] -> [seq, NH*HD*2].
    e = LaunchNvfp4Gemm(lw.q, d_normed, seq, d_qc, stream); if (e) return e;
    // 2. k,v projections.
    e = LaunchNvfp4Gemm(lw.k, d_normed, seq, k_row0, stream);  if (e) return e;
    e = LaunchNvfp4Gemm(lw.v, d_normed, seq, v_row0, stream);  if (e) return e;

    // 3. Split query + gate, reshape to [seq, NH, HD], per-head RMSNorm(q).
    {
        const int threads = std::min(HD, 512);
        qwen_split_query_kernel<<<seq * NH, threads, HD * sizeof(float),
                                  stream>>>(
            d_qc, d_gate, d_q, nullptr, seq, NH, HD, lw.d_q_norm,
            lw.d_k_norm, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 4. per-head RMSNorm(k) into k_row0 itself.
    {
        const int threads = std::min(HD, 512);
        qwen_head_norm_kernel<<<seq * NKV, threads, HD * sizeof(float),
                                  stream>>>(
            k_row0, lw.d_k_norm, k_row0, seq, NKV, HD, (float)top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 5. RoPE on q,k (first rotary_dim dims). k has NKV heads, q has NH heads.
    e = RunQwenRopeCosSin(stream, d_pos, seq, rotary_dim, top.rope_theta,
                          d_cos, d_sin); if (e) return e;
    e = RunQwenPartialRope(stream, d_q, k_row0, d_cos, d_sin, d_q, k_row0,
                           seq, NH, NKV, HD, rotary_dim); if (e) return e;

    // 6. Scaled dot-product attention (GQA, causal) -> d_attn [seq, NH, HD].
    {
        const float scale = 1.0f / sqrtf((float)HD);
        const dim3 grid(seq, NH);
        const int threads = std::min(HD, 512);
        // shared holds up to (kv_base+seq+1) logits; cap at threads to bound
        // memory.
        const size_t smem =
            (static_cast<size_t>(kv_base) + static_cast<size_t>(seq) + 1) *
            sizeof(float);
        full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
            d_q, d_k, d_v, d_attn, seq, NH, HD, NKV, scale, kv_base);
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
                                   __half *d_gated, int kv_base,
                                   float *d_gd_state)
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
    // 1. qkv, z projections via NVFP4 GEMM. For incremental decode `d_qkv`
    //    points at the layer's persistent cache base (rows [kv_base..kv_base+seq)
    //    of the cache); the conv1d below then reads the full history.
    e = LaunchNvfp4Gemm(lw.qkv, d_normed, seq,
                        d_qkv + (int64_t)kv_base * S, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.z,   d_normed, seq, d_z,   stream);   if (e) return e;

    // 2. conv1d over S channels (reads history rows [kv_base-kernel+1..kv_base+seq)).
    if (lw.d_conv1d && lw.conv_dim == S) {
        const int threads = 128;
        qwen_conv1d_kernel<<<(S + threads - 1) / threads, threads, 0, stream>>>(
            d_qkv, lw.d_conv1d, d_conv, seq, S, lw.conv_kernel, kv_base);
        e = cudaGetLastError(); if (e) return e;
    } else {
        cudaMemcpyAsync(d_conv, d_qkv + (int64_t)kv_base * S,
                        (int64_t)seq * S * sizeof(__half),
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
            nk, kd, nv, vd, r, S, d_gd_state);
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
// ── QwenKvCache (spec item 15: persistent incremental session state) ───────
// Buffers are per-layer-sliced: full-attention layers own [max_total][NKV*HD]
// k/v halves; linear layers own [max_total][KD*2+VD] raw qkv history halves
// (feed for the causal conv1d) plus a [nv*vd*kd] float GatedDeltaNet col
// accumulator. Allocate() sets up a fresh session (kv_rows == 0, col zeroed);
// EnsureCapacity() grows geometrically and preserves the written rows.

cudaError_t QwenKvCache::Allocate(const QwenPinnedWeights &pw, int max_total_in,
                                  cudaStream_t stream)
{
    Free();
    const QwenTopology &top = pw.top;
    const int NKV = top.num_key_value_heads, HD = top.head_dim;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    num_full_layers = 0;
    num_linear_layers = 0;
    for (const auto &lw : pw.layers) {
        if (lw.is_linear) ++num_linear_layers;
        else ++num_full_layers;
    }
    if (max_total_in <= 0 || num_full_layers <= 0 || num_linear_layers <= 0)
        return cudaErrorInvalidValue;

    max_total = max_total_in;
    const size_t kvk_bytes =
        (size_t)num_full_layers * (size_t)max_total * NKV * HD * sizeof(__half);
    const size_t kvv_bytes = kvk_bytes;
    const size_t qkv_bytes =
        (size_t)num_linear_layers * (size_t)max_total * S * sizeof(__half);
    const size_t col_bytes =
        (size_t)num_linear_layers * (size_t)nv * vd * kd * sizeof(float);

    if (cudaMalloc(&d_kv_k, kvk_bytes) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(&d_kv_v, kvv_bytes) != cudaSuccess) { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_qkv, qkv_bytes) != cudaSuccess)   { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_col, col_bytes) != cudaSuccess)   { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMemsetAsync(d_col, 0, col_bytes, stream) != cudaSuccess) {
        Free(); return cudaErrorInvalidValue;
    }
    kv_rows = 0;
    last_tok = -1;
    cache_bytes = (int64_t)(kvk_bytes + kvv_bytes + qkv_bytes + col_bytes);
    return cudaSuccess;
}

cudaError_t QwenKvCache::EnsureCapacity(const QwenPinnedWeights &pw, int needed,
                                        cudaStream_t stream)
{
    if (needed <= max_total) return cudaSuccess;
    if (max_total <= 0) return cudaErrorInvalidValue;
    const QwenTopology &top = pw.top;
    const int NKV = top.num_key_value_heads, HD = top.head_dim;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    const int new_max = std::max(2 * max_total, needed);
    const size_t kvk_bytes =
        (size_t)num_full_layers * (size_t)new_max * NKV * HD * sizeof(__half);
    const size_t kvv_bytes = kvk_bytes;
    const size_t qkv_bytes =
        (size_t)num_linear_layers * (size_t)new_max * S * sizeof(__half);
    const size_t col_bytes =
        (size_t)num_linear_layers * (size_t)nv * vd * kd * sizeof(float);

    __half *n_k = nullptr, *n_v = nullptr, *n_q = nullptr;
    float *n_c = nullptr;
    if (cudaMalloc(&n_k, kvk_bytes) != cudaSuccess) return cudaErrorMemoryAllocation;
    if (cudaMalloc(&n_v, kvv_bytes) != cudaSuccess) { cudaFree(n_k); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&n_q, qkv_bytes) != cudaSuccess) { cudaFree(n_k); cudaFree(n_v); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&n_c, col_bytes) != cudaSuccess) {
        cudaFree(n_k); cudaFree(n_v); cudaFree(n_q);
        return cudaErrorMemoryAllocation;
    }

    // Preserve the written rows [0..kv_rows) per layer (col state is row-free).
    const size_t row_k = (size_t)NKV * HD * sizeof(__half);
    const size_t row_q = (size_t)S * sizeof(__half);
    for (int li = 0; li < num_full_layers; ++li) {
        cudaMemcpyAsync(n_k + (size_t)li * new_max * NKV * HD,
                        d_kv_k + (size_t)li * max_total * NKV * HD,
                        (size_t)kv_rows * row_k, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(n_v + (size_t)li * new_max * NKV * HD,
                        d_kv_v + (size_t)li * max_total * NKV * HD,
                        (size_t)kv_rows * row_k, cudaMemcpyDeviceToDevice, stream);
    }
    for (int li = 0; li < num_linear_layers; ++li) {
        cudaMemcpyAsync(n_q + (size_t)li * new_max * S,
                        d_qkv + (size_t)li * max_total * S,
                        (size_t)kv_rows * row_q, cudaMemcpyDeviceToDevice, stream);
    }
    cudaMemcpyAsync(n_c, d_col, col_bytes, cudaMemcpyDeviceToDevice, stream);

    // Drain the copy stream before freeing the old buffers so no in-flight
    // D2D read outlives its source allocation.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaFree(n_k); cudaFree(n_v); cudaFree(n_q); cudaFree(n_c);
        return cudaErrorInvalidValue;
    }

    cudaFree(d_kv_k);
    cudaFree(d_kv_v);
    cudaFree(d_qkv);
    cudaFree(d_col);
    d_kv_k = n_k;
    d_kv_v = n_v;
    d_qkv = n_q;
    d_col = n_c;
    max_total = new_max;
    cache_bytes = (int64_t)(kvk_bytes + kvv_bytes + qkv_bytes + col_bytes);
    return cudaSuccess;
}

void QwenKvCache::Free()
{
    cudaFree(d_kv_k);
    cudaFree(d_kv_v);
    cudaFree(d_qkv);
    cudaFree(d_col);
    d_kv_k = nullptr;
    d_kv_v = nullptr;
    d_qkv = nullptr;
    d_col = nullptr;
    kv_rows = 0;
    last_tok = -1;
    max_total = 0;
    num_full_layers = 0;
    num_linear_layers = 0;
    cache_bytes = 0;
}

// ── Generation loop ─────────────────────────────────────────────────────────
// Greedy argmax over the last row of the logits matrix [seq, V] is delegated to
// the injected lsxcommon::sample::ISampler (kF16), replacing the former fused
// qwen_argmax_kernel.

cudaError_t RunQwenGeneration(cudaStream_t stream,
                              const QwenPinnedWeights &pw,
                              const std::vector<int32_t> &prompt, int max_new,
                              int eos_id,
                              QwenGenerationResult &result,
                              lsxcommon::sample::ISampler &sampler,
                              QwenKvCache *kv)
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

    // ── incremental session cache (spec item 15) ──
    // kv != nullptr selects the one-token-per-step append-only path: a fresh
    // session allocates a cache for prompt.size()+max_new rows (zeroed
    // GatedDeltaNet state); a continuation grows it if the remaining budget
    // does not fit. kv == nullptr keeps the full-recompute path untouched.
    const bool is_incr = (kv != nullptr);
    if (is_incr) {
        cudaError_t ce;
        if (!kv->allocated())
            ce = kv->Allocate(pw, max_seq, stream);
        else
            ce = kv->EnsureCapacity(pw, kv->kv_rows + max_new, stream);
        if (ce != cudaSuccess) {
            result.error_msg = "kv cache alloc failed";
            return ce;
        }
    }

    // ── arena (all halves unless noted; single cudaMalloc) ──
    size_t half_sz[] = {
        (size_t)max_seq * H,             // d_act
        (size_t)max_seq * H,             // d_normed (input)
        (size_t)max_seq * H,             // d_attn_out
        (size_t)max_seq * H,             // d_post_in
        2 * (size_t)max_seq * I,         // d_gate (fused-MLP scratch [2*seq,I])
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
        (size_t)1,                       // d_maxv_scratch (sampler max-value out)
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
    __half *d_gate = adv(ph, half_sz[4]);
    __half *d_qc = adv(ph, half_sz[5]);
    __half *d_q = adv(ph, half_sz[6]);
    __half *d_qgate = adv(ph, half_sz[7]);
    __half *d_k = adv(ph, half_sz[8]);
    __half *d_v = adv(ph, half_sz[9]);
    __half *d_attn = adv(ph, half_sz[10]);
    __half *d_cos = adv(ph, half_sz[11]);
    __half *d_sin = adv(ph, half_sz[12]);
    __half *d_qkv = adv(ph, half_sz[13]);
    __half *d_z = adv(ph, half_sz[14]);
    __half *d_ab = adv(ph, half_sz[15]);
    __half *d_conv = adv(ph, half_sz[16]);
    __half *d_core = adv(ph, half_sz[17]);
    __half *d_gated = adv(ph, half_sz[18]);
    __half *d_embed_h = adv(ph, half_sz[19]);
    __half *d_head_h = adv(ph, half_sz[20]);
    __half *d_logits = adv(ph, half_sz[21]);
    int32_t *d_ids = reinterpret_cast<int32_t*>(adv(ph, half_sz[22]));
    int32_t *d_pos = reinterpret_cast<int32_t*>(adv(ph, half_sz[23]));
    float *pf = reinterpret_cast<float*>((uint8_t*)base + half_bytes);
    float *d_gf = pf; pf += float_sz[0];
    float *d_betaf = pf; pf += float_sz[1];
    float *d_maxv_scratch = pf; pf += float_sz[2];

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

    if (!is_incr) {
        // ── full-recompute path (unchanged) ──
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
                    e = RunQwenFusedLinearAttnKernel(stream, lw, top, seq,
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
                // fused MLP (down-GEMM accumulates into d_act residual)
                e = RunQwenFusedMLP(stream, lw, top, seq, d_postin, d_act,
                                    d_gate);
                if (e != cudaSuccess) return fail(e, std::string("mlp l")+std::to_string(l));
            }

            // final norm
            e = RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, "finalnorm");
            // head logits: LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits)
            e = LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits, stream);
            if (e != cudaSuccess) return fail(e, "head");
            // argmax last row
            {
                const int last_row = seq - 1;
                e = sampler.Sample(stream,
                                   d_logits + static_cast<size_t>(last_row) * V, V,
                                   lsxcommon::sample::LogitType::kF16,
                                   d_argmax, d_maxv_scratch);
                if (e != cudaSuccess) return fail(e, "argmax");
            }

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
    } else {
        // ── incremental path: one token per step at kv->kv_rows ──
        // The first prompt.size()-1 steps replay the prompt rows one at a time
        // (their argmax samples are discarded); from prompt.size()-1 on every
        // sample is a generated token. Full-attn k/v rows are appended to the
        // persistent cache; the GatedDeltaNet col accumulator is seeded from
        // and written back to kv->d_col per linear layer.
        const int prompt_steps = (int)prompt.size();
        const int total_steps = prompt_steps + max_new - 1;
        int32_t cur = prompt[0];
        int32_t tok = -1;
        int generated = 0;
        for (int step = 0; step < total_steps; ++step) {
            const int seq = 1;
            const int kv_base = kv->kv_rows;
            cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_pos, &kv_base, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);

            e = RunQwenEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
            if (e != cudaSuccess) return fail(e, "embed");

            int full_idx = 0, lin_idx = 0;
            for (int l = 0; l < (int)pw.layers.size(); ++l) {
                const QwenLayerWeights &lw = pw.layers[l];
                // input RMSNorm
                e = RunQwenRMSNorm(stream, d_act, lw.d_input_norm, d_normed,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(e, std::string("inputnorm l")+std::to_string(l));
                if (lw.is_linear) {
                    e = RunQwenFusedLinearAttnKernel(stream, lw, top, seq,
                        d_normed, d_attn_out,
                        kv->d_qkv + (int64_t)lin_idx * kv->max_total * S,
                        d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                        kv_base,
                        kv->d_col + (int64_t)lin_idx * nv * vd * kd);
                    ++lin_idx;
                } else {
                    e = RunQwenFullAttention(stream, lw, top, seq,
                        d_normed, d_attn_out, d_qc, d_q, d_qgate,
                        kv->d_kv_k + (int64_t)full_idx * kv->max_total * NKV * HD,
                        kv->d_kv_v + (int64_t)full_idx * kv->max_total * NKV * HD,
                        d_attn, d_cos, d_sin, d_pos, kv_base);
                    ++full_idx;
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
                // fused MLP (down-GEMM accumulates into d_act residual)
                e = RunQwenFusedMLP(stream, lw, top, seq, d_postin, d_act,
                                    d_gate);
                if (e != cudaSuccess) return fail(e, std::string("mlp l")+std::to_string(l));
            }

            // final norm
            e = RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, "finalnorm");
            // head logits
            e = LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits, stream);
            if (e != cudaSuccess) return fail(e, "head");

            kv->kv_rows += seq;

            if (step >= prompt_steps - 1) {
                e = sampler.Sample(stream, d_logits, V,
                                   lsxcommon::sample::LogitType::kF16,
                                   d_argmax, d_maxv_scratch);
                if (e != cudaSuccess) return fail(e, "argmax");
                int32_t gtok = 0;
                cudaEvent_t tok_ready;
                cudaEventCreate(&tok_ready);
                cudaMemcpyAsync(&gtok, d_argmax, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);
                cudaEventDestroy(tok_ready);
                tok = gtok;
                if (generated == 0) cudaEventRecord(t_ttft, stream);
                result.tokens.push_back(tok);
                ++generated;
                if (tok == eos_id) {
                    result.eos_terminated = true;
                    break;
                }
                cur = tok;
            } else {
                cur = prompt[step + 1];
            }
        }
        kv->last_tok = tok;
    }

    // Publish last-row logits (final row) for the regression harness, while
    // the arena is still alive.
    {
        std::vector<float> hlast_f(V);
        // Incremental path always produces the final row at index 0 (seq == 1).
        int64_t base_idx = is_incr ? 0 : (int64_t)ids.size() - 1;
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
                            + float_bytes + sizeof(int32_t)
                            + (is_incr ? kv->Bytes() : 0);
    cudaFree(d_argmax);
    cudaFree(base);
    result.ok = true;
    return cudaSuccess;
}

} // namespace qwen
} // namespace lsxcommon