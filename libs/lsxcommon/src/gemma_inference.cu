#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "lsxcommon/gemma4_inference.h"
#include "lsxcommon/gemma4_inference_load.h"
#include "lsxcommon/shared_experts.h"

namespace lsxcommon {
namespace gemma {

namespace {

__device__ __forceinline__ unsigned int bf16_bits_wide(unsigned short bits)
{
    return static_cast<unsigned int>(bits) << 16;
}

__global__ void gemma_bf16_to_half_inplace_kernel(unsigned short *buf,
                                                  long long n)
{
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int wide = bf16_bits_wide(buf[i]);
    float f = __uint_as_float(wide);
    __half h = __float2half(f);
    buf[i] = __half_as_ushort(h);
}

} // namespace

cudaError_t GemmaBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                   cudaStream_t stream)
{
    if (!d_buf || numel <= 0) return cudaErrorInvalidValue;
    if (reinterpret_cast<uintptr_t>(d_buf) % 16 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    long long grid = (numel + kBlock - 1) / kBlock;
    if (grid > 2147483647LL) grid = 2147483647LL;
    gemma_bf16_to_half_inplace_kernel<<<static_cast<int>(grid), kBlock, 0,
                                        stream>>>(reinterpret_cast<unsigned short *>(d_buf),
                                                  numel);
    return cudaGetLastError();
}

// ── RMSNorm (offset-weight form) ────────────────────────────────────────────

__global__ void gemma_rmsnorm_kernel(const __half *__restrict__ x,
                                     const __half *__restrict__ w,
                                     __half *out, int rows, int hidden,
                                     float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half *xr = x + static_cast<int64_t>(row) * hidden;
    __half *orow = out + static_cast<int64_t>(row) * hidden;
    extern __shared__ float squares[];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / static_cast<float>(hidden) + eps);
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float wf = __half2float(w[j]);
        orow[j] = __float2half(__half2float(xr[j]) * inv * (1.0f + wf));
    }
}

cudaError_t RunGemmaRMSNorm(const __half *d_x, const __half *d_w, __half *d_out,
                            int rows, int hidden, float eps,
                            cudaStream_t stream)
{
    if (!d_x || !d_w || !d_out || rows <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    gemma_rmsnorm_kernel<<<rows, kBlock, kBlock * sizeof(float), stream>>>(
        d_x, d_w, d_out, rows, hidden, eps);
    return cudaGetLastError();
}

// ── RoPE ────────────────────────────────────────────────────────────────────

__global__ void gemma_rope_cos_sin_kernel(const int32_t *__restrict__ pos,
                                          __half *__restrict__ cos_buf,
                                          __half *__restrict__ sin_buf,
                                          int seq, int rotary_dim, double theta)
{
    const int64_t total = static_cast<int64_t>(seq) * (rotary_dim / 2);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int t = static_cast<int>(idx / (rotary_dim / 2));
        const int d = static_cast<int>(idx % (rotary_dim / 2));
        double inv =
            1.0 / pow(theta, static_cast<double>(2 * d) / rotary_dim);
        double ang = static_cast<double>(pos[t]) * inv;
        cos_buf[idx] = __float2half(static_cast<float>(cos(ang)));
        sin_buf[idx] = __float2half(static_cast<float>(sin(ang)));
    }
}

// HF rotate_half pairing: pair (j, j+rot/2); cos/sin layout [seq, rot/2].
__global__ void gemma_partial_rope_kernel(const __half *__restrict__ x,
                                          const __half *__restrict__ cos_buf,
                                          const __half *__restrict__ sin_buf,
                                          __half *__restrict__ out, int seq,
                                          int heads, int head_dim,
                                          int rotary_dim)
{
    const int half = rotary_dim / 2;

    // Rotation.
    const int64_t total =
        static_cast<int64_t>(seq) * heads * static_cast<int64_t>(half);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < total; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int j = static_cast<int>(idx % half);
        const int h = static_cast<int>((idx / half) % heads);
        const int t = static_cast<int>(idx / (static_cast<int64_t>(half) * heads));
        const float c = __half2float(cos_buf[t * half + j]);
        const float s = __half2float(sin_buf[t * half + j]);
        const int64_t row = (static_cast<int64_t>(t) * heads + h) * head_dim;
        const float x1 = __half2float(x[row + j]);
        const float x2 = __half2float(x[row + j + half]);
        out[row + j] = __float2half(x1 * c - x2 * s);
        out[row + j + half] = __float2half(x2 * c + x1 * s);
    }

    // Pass-through tail.
    const int64_t npt = static_cast<int64_t>(seq) * heads *
                        (head_dim - rotary_dim);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         idx < npt; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int pk = static_cast<int>(idx % (head_dim - rotary_dim));
        const int rh =
            static_cast<int>(idx / (head_dim - rotary_dim));
        const int64_t base =
            static_cast<int64_t>(rh) * head_dim + rotary_dim;
        out[base + pk] = x[base + pk];
    }
}

cudaError_t RunGemmaRopeCosSin(const int32_t *d_pos, __half *d_cos,
                               __half *d_sin, int seq, int rotary_dim,
                               double theta, cudaStream_t stream)
{
    if (!d_pos || !d_cos || !d_sin || seq <= 0 || rotary_dim <= 0 ||
        rotary_dim % 2 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    const int64_t total =
        static_cast<int64_t>(seq) * (rotary_dim / 2);
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    gemma_rope_cos_sin_kernel<<<grid, kBlock, 0, stream>>>(d_pos, d_cos, d_sin,
                                                           seq, rotary_dim,
                                                           theta);
    return cudaGetLastError();
}

cudaError_t RunGemmaPartialRope(const __half *d_x, const __half *d_cos,
                                const __half *d_sin, __half *d_out, int seq,
                                int heads, int head_dim, int rotary_dim,
                                cudaStream_t stream)
{
    if (!d_x || !d_cos || !d_sin || !d_out || seq <= 0 || heads <= 0 ||
        head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
        rotary_dim % 2 != 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    gemma_partial_rope_kernel<<<1024, kBlock, 0, stream>>>(
        d_x, d_cos, d_sin, d_out, seq, heads, head_dim, rotary_dim);
    return cudaGetLastError();
}

// ── Attention ───────────────────────────────────────────────────────────────

// Two-pass softmax (shared-max then sum) with a fixed slot order; output pass
// iterates slots sequentially per (d). Out-of-band slots never enter any
// reduction (band predicate checked before scoring).
__global__ void gemma_attention_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, __half *__restrict__ out, int seq,
    int total_slots, int pos_base, int q_heads, int kv_heads, int head_dim,
    int window, float scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= seq || h >= q_heads) return;

    const int kvh = h / (q_heads / kv_heads);
    const __half *qr = q + (static_cast<int64_t>(t) * q_heads + h) * head_dim;
    __half *orr = out + (static_cast<int64_t>(t) * q_heads + h) * head_dim;
    const int last = pos_base + t;
    const int first =
        (window > 0) ? (last - window + 1 > 0 ? last - window + 1 : 0) : 0;
    const int n_valid = last - first + 1;
    if (n_valid <= 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) orr[d] = __float2half(0.f);
        return;
    }

    extern __shared__ float red[];
    __shared__ float s_max, s_sum;

    // Pass A: global max score.
    float local_max = -1e30f;
    for (int s = first + threadIdx.x; s <= last; s += blockDim.x) {
        const __half *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qr[d]) * __half2float(kr[d]);
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        for (int i = 1; i < blockDim.x; ++i) m = fmaxf(m, red[i]);
        s_max = m;
    }
    __syncthreads();

    // Pass B: normalizer.
    float local_sum = 0.f;
    for (int s = first + threadIdx.x; s <= last; s += blockDim.x) {
        const __half *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qr[d]) * __half2float(kr[d]);
        local_sum += expf(dot * scale - s_max);
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float z = 0.f;
        for (int i = 0; i < blockDim.x; ++i) z += red[i];
        s_sum = z;
    }
    __syncthreads();

    // Output pass: sequential slot order per dim (deterministic).
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int s = first; s <= last; ++s) {
            const __half *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
            float dot = 0.f;
            for (int dd = 0; dd < head_dim; ++dd)
                dot += __half2float(qr[dd]) * __half2float(kr[dd]);
            const float p = expf(dot * scale - s_max) / s_sum;
            acc += p * __half2float(v[(static_cast<int64_t>(s) * kv_heads + kvh) *
                                       head_dim +
                                       d]);
        }
        orr[d] = __float2half(acc);
    }
}

cudaError_t RunGemmaAttention(const __half *d_q, const __half *d_k,
                              const __half *d_v, __half *d_out, int seq,
                              int total_slots, int pos_base, int q_heads,
                              int kv_heads, int head_dim, int window,
                              float scale, cudaStream_t stream)
{
    if (!d_q || !d_k || !d_v || !d_out || seq <= 0 || total_slots <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || pos_base < 0)
        return cudaErrorInvalidValue;
    if (q_heads % kv_heads != 0) return cudaErrorInvalidValue;
    if (pos_base + seq > total_slots) return cudaErrorInvalidValue;
    constexpr int kBlock = 128;
    dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(q_heads));
    gemma_attention_kernel<<<grid, kBlock, kBlock * sizeof(float), stream>>>(
        d_q, d_k, d_v, d_out, seq, total_slots, pos_base, q_heads, kv_heads,
        head_dim, window, scale);
    return cudaGetLastError();
}

// ── Dense MLP + MoE ─────────────────────────────────────────────────────────

__device__ __forceinline__ float gelu_tanh_dev(float v)
{
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v * (1.0f + tanhf(k0 * (v + 0.044715f * v * v * v)));
}

// out[t,c] = sum_i down[c,i] * gelu(gate[i]@x_t) * (up[i]@x_t)
__global__ void gemma_dense_mlp_kernel(const __half *__restrict__ gate,
                                       const __half *__restrict__ up,
                                       const __half *__restrict__ down,
                                       int64_t I, int64_t H,
                                       const __half *__restrict__ x,
                                       __half *__restrict__ out, int seq)
{
    // grid: (seq, H); one thread per output element.
    const int t = blockIdx.x;
    const int c = blockIdx.y;
    if (t >= seq || c >= H) return;
    const __half *xr = x + static_cast<int64_t>(t) * H;
    float acc = 0.f;
    for (int64_t i = 0; i < I; ++i) {
        float g = 0.f, u = 0.f;
        for (int64_t d = 0; d < H; ++d) {
            float xv = __half2float(xr[d]);
            g += xv * __half2float(gate[i * H + d]);
            u += xv * __half2float(up[i * H + d]);
        }
        acc += gelu_tanh_dev(g) * u *
               __half2float(down[c * I + i]);
    }
    out[static_cast<int64_t>(t) * H + c] = __float2half(acc);
}

cudaError_t RunGemmaDenseMLP(const __half *d_gate, const __half *d_up,
                             const __half *d_down, int64_t I, int64_t H,
                             const __half *d_x, __half *d_out, int seq,
                             cudaStream_t stream)
{
    if (!d_gate || !d_up || !d_down || !d_x || !d_out || I <= 0 || H <= 0 ||
        seq <= 0)
        return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(H));
    gemma_dense_mlp_kernel<<<grid, 1, 0, stream>>>(d_gate, d_up, d_down, I, H,
                                                   d_x, d_out, seq);
    return cudaGetLastError();
}

// One block per token: computes logits, deterministic insertion top-k with
// lower-index tie-break, softmax over the selected logits.
__global__ void gemma_router_topk_kernel(
    const __half *__restrict__ proj, const float *__restrict__ per_expert_scale,
    const __half *__restrict__ x, int E, int H, int K,
    int32_t *__restrict__ sel, float *__restrict__ weights)
{
    const int t = blockIdx.x;
    if (t >= gridDim.x) return;
    const __half *xr = x + static_cast<int64_t>(t) * H;

    extern __shared__ float sh[];
    float *logits = sh;             // [E]

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < H; ++d)
            dot += __half2float(proj[static_cast<int64_t>(e) * H + d]) *
                   __half2float(xr[d]);
        logits[e] = dot;
    }
    __syncthreads();

    // Thread 0: insertion top-k, lower index wins ties.
    if (threadIdx.x == 0) {
        int idx[16];   // top_k <= 16 supported
        float val[16];
        int n = 0;
        for (int e = 0; e < E; ++e) {
            float v = logits[e];
            int pos = n;
            while (pos > 0 &&
                   (val[pos - 1] < v ||
                    (val[pos - 1] == v && idx[pos - 1] > e))) {
                if (pos < K) { val[pos] = val[pos - 1]; idx[pos] = idx[pos - 1]; }
                --pos;
            }
            if (pos < K) {
                val[pos] = v;
                idx[pos] = e;
                if (n < K) ++n;
            }
        }
        float m = val[0];
        for (int i = 1; i < K; ++i) m = fmaxf(m, val[i]);
        float z = 0.f;
        for (int i = 0; i < K; ++i) {
            z += expf(val[i] - m);
        }
        for (int i = 0; i < K; ++i) {
            sel[static_cast<int64_t>(t) * K + i] = idx[i];
            float w = expf(val[i] - m) / z;
            if (per_expert_scale) w *= per_expert_scale[idx[i]];
            weights[static_cast<int64_t>(t) * K + i] = w;
        }
    }
}

cudaError_t RunGemmaRouterTopK(const __half *d_router_proj,
                               const float *d_per_expert_scale,
                               const __half *d_x, int num_experts,
                               int hidden, int top_k, int32_t *d_sel_experts,
                               float *d_sel_weights, int seq,
                               cudaStream_t stream)
{
    if (!d_router_proj || !d_x || !d_sel_experts || !d_sel_weights ||
        num_experts <= 0 || hidden <= 0 || top_k <= 0 || top_k > 16 ||
        seq <= 0)
        return cudaErrorInvalidValue;
    gemma_router_topk_kernel<<<seq, 128,
                               static_cast<size_t>(num_experts) *
                                   sizeof(float),
                               stream>>>(d_router_proj, d_per_expert_scale,
                                         d_x, num_experts, hidden, top_k,
                                         d_sel_experts, d_sel_weights);
    return cudaGetLastError();
}

// out[t,h] = router_scale[h] * sum_k w_tk * expert_outs[k][t,h]
__global__ void gemma_moe_combine_kernel(__half *__restrict__ out,
                                         const __half *const *__restrict__ eos,
                                         const float *__restrict__ w,
                                         const float *__restrict__ rscale,
                                         int H, int K)
{
    const int t = blockIdx.x;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += w[static_cast<int64_t>(t) * K + k] *
                   __half2float(eos[k][static_cast<int64_t>(t) * H + h]);
        if (rscale) acc *= rscale[h];
        out[static_cast<int64_t>(t) * H + h] = __float2half(acc);
    }
}

cudaError_t RunGemmaMoECombine(const __half *const *d_expert_outs,
                               const float *d_weights,
                               const float *d_router_scale, __half *d_out,
                               int hidden, int k, cudaStream_t stream)
{
    if (!d_expert_outs || !d_weights || !d_out || hidden <= 0 || k <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    gemma_moe_combine_kernel<<<1, kBlock, 0, stream>>>(
        d_out, d_expert_outs, d_weights, d_router_scale, hidden, k);
    return cudaGetLastError();
}

// ── Generation building blocks ──────────────────────────────────────────────

__global__ void gemma_embed_kernel(const __half *__restrict__ embed,
                                   const int32_t *__restrict__ ids,
                                   __half *__restrict__ out, int hidden)
{
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        out[static_cast<int64_t>(t) * hidden + d] =
            embed[static_cast<int64_t>(ids[t]) * hidden + d];
}

// out[t,r] = sum_d W[r,d] * x[t,d]   (W [rows, cols] row-major)
__global__ void gemma_matmul_kernel(const __half *__restrict__ w,
                                    const __half *__restrict__ x,
                                    __half *__restrict__ out, int rows,
                                    int cols, int seq)
{
    const int r = blockIdx.y;
    if (r >= rows) return;
    for (int t = blockIdx.x; t < seq; t += gridDim.x) {
        const __half *xr = x + static_cast<int64_t>(t) * cols;
        float acc = 0.f;
        for (int d = 0; d < cols; ++d)
            acc += __half2float(w[static_cast<int64_t>(r) * cols + d]) *
                   __half2float(xr[d]);
        out[static_cast<int64_t>(t) * rows + r] = __float2half(acc);
    }
}

static cudaError_t Matmul(const __half *w, const __half *x, __half *out,
                          int rows, int cols, int seq, cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned>(std::min(seq, 1024)),
              static_cast<unsigned>(rows));
    gemma_matmul_kernel<<<grid, 1, 0, stream>>>(w, x, out, rows, cols, seq);
    return cudaGetLastError();
}

// Per-head RMSNorm with (1+w) offset over packed [seq, heads*dim].
__global__ void gemma_head_norm_kernel(const __half *__restrict__ x,
                                       const __half *__restrict__ w,
                                       __half *__restrict__ out, int seq,
                                       int heads, int dim, float eps)
{
    extern __shared__ float squares[];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int base = h * dim;
    const __half *xr = x + static_cast<int64_t>(t) * heads * dim + base;
    __half *orr = out + static_cast<int64_t>(t) * heads * dim + base;

    float sum = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / static_cast<float>(dim) + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        orr[j] = __float2half(__half2float(xr[j]) * inv *
                              (1.0f + __half2float(w[j])));
}

// dst[t,d] += scalar * src[t,d]
__global__ void gemma_axpy_kernel(__half *__restrict__ dst,
                                  const __half *__restrict__ src,
                                  float scalar, int64_t numel)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel)
        dst[i] = __float2half(__half2float(dst[i]) +
                              scalar * __half2float(src[i]));
}

// dst[t,d] *= scale[d]
__global__ void gemma_hadamard_kernel(__half *__restrict__ dst,
                                      const float *__restrict__ scale,
                                      int hidden, int seq)
{
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        dst[static_cast<int64_t>(t) * hidden + d] = __float2half(
            __half2float(dst[static_cast<int64_t>(t) * hidden + d]) *
            scale[d]);
}

// logits[v] = dot(embed[v], x); then softcap tanh(l/cap)*cap; block-0 argmax.
__global__ void gemma_logits_argmax_kernel(const __half *__restrict__ embed,
                                           const __half *__restrict__ x,
                                           int vocab, int hidden, float cap,
                                           int32_t *__restrict__ d_argmax,
                                           float *__restrict__ d_maxlogit,
                                           float *__restrict__ d_logits)
{
    // One thread per vocab row (strided).
    float local_best = -1e30f;
    int local_arg = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const __half *row = embed + static_cast<int64_t>(v) * hidden;
        float acc = 0.f;
        for (int d = 0; d < hidden; ++d)
            acc += __half2float(row[d]) * __half2float(x[d]);
        if (cap > 0.f) acc = tanhf(acc / cap) * cap;
        d_logits[v] = acc;
        if (acc > local_best) { local_best = acc; local_arg = v; }
    }
    __shared__ float best[256];
    __shared__ int barg[256];
    best[threadIdx.x] = local_best;
    barg[threadIdx.x] = local_arg;
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = best[0];
        int a = barg[0];
        for (int i = 1; i < blockDim.x && i < 256; ++i)
            if (best[i] > b) { b = best[i]; a = barg[i]; }
        *d_argmax = a;
        if (d_maxlogit) *d_maxlogit = b;
    }
}

// ── Generation loop ─────────────────────────────────────────────────────────

namespace {

// File-scope expert db reference used inside ForwardLayer (set by
// RunGemmaGeneration; single-threaded generation contract).
GemmaExpertDb *g_expert_db = nullptr;

struct GenWorkspace {
    __half *d_x = nullptr;        // [seq0][H]
    __half *d_normed = nullptr;   // [seq0][H]
    __half *d_tmp = nullptr;      // [seq0][q_rows_max]
    __half *d_qr = nullptr;       // [seq0][q_rows_max]
    __half *d_kr = nullptr;       // [seq0][kv_rows_max]
    __half *d_vr = nullptr;
    __half *d_cos = nullptr;      // [seq0][rot_half]
    __half *d_sin = nullptr;
    int32_t *d_pos = nullptr;     // [seq0]
    float *d_logits = nullptr;    // [vocab]
    int32_t *d_argmax = nullptr;
    int32_t *d_sel = nullptr;     // [seq0*K]
    float *d_selw = nullptr;      // [seq0*K]
    int32_t *h_sel = nullptr;     // pinned
    float *h_selw = nullptr;      // pinned
    int32_t *h_arg = nullptr;     // pinned
    uint8_t *d_p1 = nullptr, *d_s1 = nullptr, *d_p2 = nullptr;
    uint8_t *d_s2 = nullptr, *d_p3 = nullptr, *d_s3 = nullptr;
    __half *d_w1 = nullptr, *d_w3 = nullptr, *d_w2 = nullptr;
    __half *d_eout = nullptr;     // [1][H]
    __half *d_acc = nullptr;      // [seq0][H]

    size_t seq0 = 0, q_rows_max = 0, kv_rows_max = 0, rot_half_max = 0;
    int K = 0;

    void Free()
    {
        auto F = [](void **p) { if (*p) { cudaFree(*p); *p = nullptr; } };
        F((void **)&d_x); F((void **)&d_normed); F((void **)&d_tmp);
        F((void **)&d_qr); F((void **)&d_kr); F((void **)&d_vr);
        F((void **)&d_cos); F((void **)&d_sin); F((void **)&d_pos);
        F((void **)&d_logits); F((void **)&d_argmax);
        F((void **)&d_sel); F((void **)&d_selw);
        F((void **)&d_p1); F((void **)&d_s1); F((void **)&d_p2);
        F((void **)&d_s2); F((void **)&d_p3); F((void **)&d_s3);
        F((void **)&d_w1); F((void **)&d_w3); F((void **)&d_w2);
        F((void **)&d_eout); F((void **)&d_acc);
        if (h_sel) { cudaFreeHost(h_sel); h_sel = nullptr; }
        if (h_selw) { cudaFreeHost(h_selw); h_selw = nullptr; }
        if (h_arg) { cudaFreeHost(h_arg); h_arg = nullptr; }
    }

    ~GenWorkspace() { Free(); }
};

// Forward `seq` tokens (hidden states in ws.d_x) through one layer; writes
// k/v rows at [pos_base .. pos_base+seq) of the layer cache.
cudaError_t ForwardLayer(cudaStream_t stream, const GemmaPinnedWeights &pw,
                         const GemmaLayerWeights &lw, int layer_idx,
                         GenWorkspace &ws, int seq, int pos_base, __half *dk,
                         __half *dv)
{
    const auto &top = pw.top;
    const int64_t H = top.hidden_size;
    const bool full =
        layer_idx < static_cast<int>(top.layer_types.size()) &&
        top.layer_types[static_cast<size_t>(layer_idx)] == "full_attention";
    const int heads = top.num_attention_heads;
    const int head_dim = full ? top.global_head_dim : top.head_dim;
    const int kv_heads =
        full ? top.num_global_key_value_heads : top.num_key_value_heads;
    const int q_rows = heads * head_dim;
    const int rotary_dim =
        full
            ? static_cast<int>(top.partial_rotary_factor * head_dim)
            : head_dim;
    const int rot_half = rotary_dim / 2;
    const float theta = full ? top.rope_theta_full : top.rope_theta_sliding;
    const int window = full ? 0 : top.sliding_window;
    const int64_t kv_stride = kv_heads * head_dim;
    float layer_scalar = 1.0f;
    if (lw.d_layer_scalar) {
        cudaError_t e = cudaMemcpyAsync(&layer_scalar, lw.d_layer_scalar,
                                        sizeof(float), cudaMemcpyDeviceToHost,
                                        stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
    }

    // 1) input RMSNorm.
    cudaError_t e = RunGemmaRMSNorm(ws.d_x, lw.d_input_norm, ws.d_normed, seq,
                                    static_cast<int>(H), top.rms_norm_eps,
                                    stream);
    if (e != cudaSuccess) return e;

    // 2) q/k/v projections. Full-attention layers tie V to K
    //    (attention_k_eq_v): no v_proj weights exist, so v := k.
    const bool v_tied_to_k = (lw.d_v == nullptr);
    e = Matmul(lw.d_q, ws.d_normed, ws.d_qr, q_rows, static_cast<int>(H), seq,
               stream);
    if (e != cudaSuccess) return e;
    e = Matmul(lw.d_k, ws.d_normed, ws.d_kr, kv_heads * head_dim,
               static_cast<int>(H), seq, stream);
    if (e != cudaSuccess) return e;
    if (!v_tied_to_k) {
        e = Matmul(lw.d_v, ws.d_normed, ws.d_vr, kv_heads * head_dim,
                   static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) return e;
    }

    // 3) QK-norms (per-head, offset weights).
    {
        dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(heads));
        gemma_head_norm_kernel<<<grid, 128, 128 * sizeof(float), stream>>>(
            ws.d_qr, lw.d_q_norm, ws.d_qr, seq, heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    {
        dim3 grid(static_cast<unsigned>(seq),
                  static_cast<unsigned>(kv_heads));
        gemma_head_norm_kernel<<<grid, 128, 128 * sizeof(float), stream>>>(
            ws.d_kr, lw.d_k_norm, ws.d_kr, seq, kv_heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    // Full-attention layers tie V to K (attention_k_eq_v): values are the
    // NORMALIZED, UNROTATED keys. Snapshot them before RoPE.
    if (v_tied_to_k) {
        e = cudaMemcpyAsync(ws.d_vr, ws.d_kr,
                            static_cast<size_t>(kv_heads * head_dim) * seq * 2,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }

    // 4) RoPE.
    {
        std::vector<int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i)
            pos[static_cast<size_t>(i)] = pos_base + i;
        e = cudaMemcpyAsync(ws.d_pos, pos.data(), sizeof(int32_t) * seq,
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = RunGemmaRopeCosSin(ws.d_pos, ws.d_cos, ws.d_sin, seq, rotary_dim,
                               theta, stream);
        if (e != cudaSuccess) return e;
        e = RunGemmaPartialRope(ws.d_qr, ws.d_cos, ws.d_sin, ws.d_qr, seq,
                                heads, head_dim, rotary_dim, stream);
        if (e != cudaSuccess) return e;
        e = RunGemmaPartialRope(ws.d_kr, ws.d_cos, ws.d_sin, ws.d_kr, seq,
                                kv_heads, head_dim, rotary_dim, stream);
        if (e != cudaSuccess) return e;
    }

    // 5) Append k/v rows to the cache (D2D, stream-ordered). For tied-V
    // layers dv rows hold the unrotated normalized keys.
    const size_t row_bytes = static_cast<size_t>(kv_stride) * 2;
    e = cudaMemcpyAsync(dk + static_cast<int64_t>(pos_base) * kv_stride,
                        ws.d_kr, row_bytes * seq, cudaMemcpyDeviceToDevice,
                        stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(dv + static_cast<int64_t>(pos_base) * kv_stride,
                        ws.d_vr, row_bytes * seq, cudaMemcpyDeviceToDevice,
                        stream);
    if (e != cudaSuccess) return e;

    // 6) Attention over the cache (band mask for sliding layers).
    e = RunGemmaAttention(ws.d_qr, dk, dv, ws.d_tmp, seq,
                          static_cast<int>(pos_base + seq), pos_base, heads,
                          kv_heads, head_dim, window,
                          1.0f / std::sqrt(static_cast<float>(head_dim)),
                          stream);
    if (e != cudaSuccess) return e;

    // 7) o_proj + sandwich post-norm + residual.
    e = Matmul(lw.d_o, ws.d_tmp, ws.d_qr, static_cast<int>(H), q_rows, seq,
               stream);
    if (e != cudaSuccess) return e;
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_norm, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, layer_scalar, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }

    // 8) Dense MLP branch (sandwich norms).
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    e = RunGemmaDenseMLP(lw.d_mlp_gate, lw.d_mlp_up, lw.d_mlp_down,
                         top.intermediate_size, H, ws.d_normed, ws.d_qr, seq,
                         stream);
    if (e != cudaSuccess) return e;
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, layer_scalar, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }

    // 9) MoE branch (sandwich norms): router -> staged NVFP4 experts.
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn2, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;

    const int E = top.num_experts;
    const int K = top.top_k_experts;
    const int64_t MI = top.moe_intermediate_size;
    e = RunGemmaRouterTopK(lw.d_router_proj, lw.d_per_expert_scale,
                           ws.d_normed, E, static_cast<int>(H), K, ws.d_sel,
                           ws.d_selw, seq, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(ws.h_sel, ws.d_sel, sizeof(int32_t) * seq * K,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(ws.h_selw, ws.d_selw, sizeof(float) * seq * K,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return e;

    e = cudaMemsetAsync(ws.d_acc, 0, sizeof(__half) * seq * H, stream);
    if (e != cudaSuccess) return e;

    for (int t = 0; t < seq; ++t) {
        for (int k = 0; k < K; ++k) {
            const int expert =
                ws.h_sel[static_cast<size_t>(t) * K + k];
            const float weight =
                ws.h_selw[static_cast<size_t>(t) * K + k];
            ensure_gemma_expert_resident(*g_expert_db, layer_idx, expert,
                                         stream);
            const auto &ent =
                g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                         static_cast<size_t>(E) +
                                     static_cast<size_t>(expert)];
            if (!ent.host_w1 || !ent.host_w2) continue;

            e = cudaMemcpyAsync(ws.d_p1, ent.host_w1,
                                static_cast<size_t>(ent.w1_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(ws.d_s1, ent.host_w1_scales,
                                static_cast<size_t>(ent.w1_scales_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(ws.d_p2, ent.host_w2,
                                static_cast<size_t>(ent.w2_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(ws.d_s2, ent.host_w2_scales,
                                static_cast<size_t>(ent.w2_scales_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(ws.d_p3, ent.host_w3,
                                static_cast<size_t>(ent.w3_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaMemcpyAsync(ws.d_s3, ent.host_w3_scales,
                                static_cast<size_t>(ent.w3_scales_bytes),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;

            e = gpu::dequant_shared_nvfp4(stream, ws.d_p1, ent.w1_bytes,
                                          ws.d_s1, ent.w1_scales_bytes,
                                          ent.w1_global_scale, ws.d_w1,
                                          MI * H);
            if (e != cudaSuccess) return e;
            e = gpu::dequant_shared_nvfp4(stream, ws.d_p2, ent.w2_bytes,
                                          ws.d_s2, ent.w2_scales_bytes,
                                          ent.w2_global_scale, ws.d_w2,
                                          H * MI);
            if (e != cudaSuccess) return e;
            e = gpu::dequant_shared_nvfp4(stream, ws.d_p3, ent.w3_bytes,
                                          ws.d_s3, ent.w3_scales_bytes,
                                          ent.w3_global_scale, ws.d_w3,
                                          MI * H);
            if (e != cudaSuccess) return e;

            e = RunGemmaDenseMLP(ws.d_w1, ws.d_w3, ws.d_w2, MI, H,
                                 ws.d_normed + static_cast<int64_t>(t) * H,
                                 ws.d_eout, /*seq=*/1, stream);
            if (e != cudaSuccess) return e;

            gemma_axpy_kernel<<<static_cast<unsigned>((H + 255) / 256), 256, 0,
                                stream>>>(
                ws.d_acc + static_cast<int64_t>(t) * H, ws.d_eout, weight, H);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
        }
    }

    // Channel scale (router.scale) + sandwich post-norm + residual.
    gemma_hadamard_kernel<<<static_cast<unsigned>(seq), 256, 0, stream>>>(
        ws.d_acc, lw.d_router_scale, static_cast<int>(H), seq);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    e = RunGemmaRMSNorm(ws.d_acc, lw.d_post_ffn2, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, layer_scalar, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace

cudaError_t RunGemmaGeneration(cudaStream_t stream, const GemmaPinnedWeights &pw,
                               GemmaExpertDb &db,
                               const std::vector<int32_t> &input_ids,
                               int max_gen_tokens, int eos_id, float softcap,
                               GemmaGenerationResult &out, GemmaKvCache &kv)
{
    out = GemmaGenerationResult{};
    // Clear any sticky context error from prior async work so per-call
    // cudaGetLastError() checks reflect only this call's launches.
    (void)cudaGetLastError();
    const auto &top = pw.top;
    if (!top.error_msg.empty()) {
        out.error_msg = "topology: " + top.error_msg;
        return cudaErrorInvalidValue;
    }
    const int64_t H = top.hidden_size;
    const int64_t V = pw.vocab_size;
    const int K = top.top_k_experts;
    const int64_t MI = top.moe_intermediate_size;
    if (input_ids.empty() || max_gen_tokens <= 0 || H <= 0 || V <= 0 ||
        pw.layers.empty() || !kv.allocated())
        return cudaErrorInvalidValue;

    const int P = static_cast<int>(input_ids.size());
    if (kv.max_total < P + max_gen_tokens) {
        out.error_msg = "kv cache capacity too small";
        return cudaErrorInvalidValue;
    }

    size_t q_rows_max = 0, kv_rows_max = 0;
    for (const auto &lw : pw.layers) {
        q_rows_max = std::max(q_rows_max, static_cast<size_t>(lw.q_rows));
        kv_rows_max = std::max(kv_rows_max, static_cast<size_t>(lw.kv_rows));
    }
    const int head_dim_max = std::max(top.head_dim, top.global_head_dim);
    const size_t rot_half_max = static_cast<size_t>(head_dim_max / 2);
    const size_t S = static_cast<size_t>(P);

    GenWorkspace ws;
    ws.seq0 = S;
    ws.q_rows_max = q_rows_max;
    ws.kv_rows_max = kv_rows_max;
    ws.rot_half_max = rot_half_max;
    ws.K = K;

    cudaError_t e = cudaSuccess;
    auto A = [&](void **p, size_t bytes) { return cudaMalloc(p, bytes); };
    if ((e = A((void **)&ws.d_x, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_normed, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmp, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_qr, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_kr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_vr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_cos, S * rot_half_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_sin, S * rot_half_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_pos, S * 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_logits, V * sizeof(float))) != cudaSuccess ||
        (e = A((void **)&ws.d_argmax, 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_sel, S * K * 4)) != cudaSuccess ||
        (e = A((void **)&ws.d_selw, S * K * sizeof(float))) != cudaSuccess ||
        (e = A((void **)&ws.d_p1, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s1, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_p2, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s2, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_p3, MI * H)) != cudaSuccess ||
        (e = A((void **)&ws.d_s3, MI * H / 16 + 16)) != cudaSuccess ||
        (e = A((void **)&ws.d_w1, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_w3, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_w2, MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eout, H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_acc, S * H * 2)) != cudaSuccess ||
        cudaMallocHost(&ws.h_sel, S * K * 4) != cudaSuccess ||
        cudaMallocHost(&ws.h_selw, S * K * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&ws.h_arg, 4) != cudaSuccess) {
        out.error_msg = "workspace alloc failed";
        return e != cudaSuccess ? e : cudaErrorMemoryAllocation;
    }

    g_expert_db = &db;

    // ── Prefill ─────────────────────────────────────────────────────────────
    e = cudaMemcpyAsync(ws.d_pos, input_ids.data(), sizeof(int32_t) * P,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { g_expert_db = nullptr; return e; }
    {
        dim3 grid(static_cast<unsigned>(P));
        gemma_embed_kernel<<<grid, 256, 0, stream>>>(
            pw.d_embed, ws.d_pos, ws.d_x, static_cast<int>(H));
        e = cudaGetLastError();
        if (e != cudaSuccess) { g_expert_db = nullptr; return e; }
    }
    cudaEvent_t ev0, ev_ttft, ev_end;
    cudaEventCreateWithFlags(&ev0, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_ttft, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_end, cudaEventDisableTiming);
    cudaEventRecord(ev0, stream);

    for (size_t l = 0; l < pw.layers.size(); ++l) {
        e = ForwardLayer(stream, pw, pw.layers[l], static_cast<int>(l), ws, P,
                         0, kv.dk[l], kv.dv[l]);
        if (e != cudaSuccess) {
            out.error_msg = "prefill layer " + std::to_string(l) + ": " +
                            cudaGetErrorString(e);
            g_expert_db = nullptr;
            return e;
        }
    }

    auto FinalNormArgmax = [&]() -> cudaError_t {
        cudaError_t le = RunGemmaRMSNorm(
            ws.d_x + static_cast<int64_t>(P - 1) * H, pw.d_final_norm,
            ws.d_normed, 1, static_cast<int>(H), top.rms_norm_eps, stream);
        if (le != cudaSuccess) return le;
        gemma_logits_argmax_kernel<<<1, 256, 0, stream>>>(
            pw.d_embed, ws.d_normed, static_cast<int>(V),
            static_cast<int>(H), softcap, ws.d_argmax, nullptr, ws.d_logits);
        return cudaGetLastError();
    };

    e = FinalNormArgmax();
    if (e != cudaSuccess) {
        out.error_msg = "prefill logits";
        g_expert_db = nullptr;
        return e;
    }
    e = cudaMemcpyAsync(ws.h_arg, ws.d_argmax, 4, cudaMemcpyDeviceToHost,
                        stream);
    if (e != cudaSuccess) { g_expert_db = nullptr; return e; }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) { g_expert_db = nullptr; return e; }
    cudaEventRecord(ev_ttft, stream);
    cudaEventSynchronize(ev_ttft);
    float ttft_ms = 0.f;
    cudaEventElapsedTime(&ttft_ms, ev0, ev_ttft);
    out.ttft_ms = ttft_ms;

    int32_t tok = ws.h_arg[0];
    out.tokens.push_back(tok);

    // ── Decode ──────────────────────────────────────────────────────────────
    for (int step = 1; step < max_gen_tokens; ++step) {
        if (tok == eos_id) {
            out.eos_terminated = true;
            break;
        }
        const int pos = P + step - 1;
        e = cudaMemcpyAsync(ws.d_pos, &tok, 4, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) break;
        gemma_embed_kernel<<<1, 256, 0, stream>>>(pw.d_embed, ws.d_pos,
                                                  ws.d_x,
                                                  static_cast<int>(H));
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        bool failed = false;
        for (size_t l = 0; l < pw.layers.size(); ++l) {
            e = ForwardLayer(stream, pw, pw.layers[l], static_cast<int>(l),
                             ws, /*seq=*/1, pos, kv.dk[l], kv.dv[l]);
            if (e != cudaSuccess) { failed = true; break; }
        }
        if (failed) break;
        e = RunGemmaRMSNorm(ws.d_x, pw.d_final_norm, ws.d_normed, 1,
                            static_cast<int>(H), top.rms_norm_eps, stream);
        if (e != cudaSuccess) break;
        gemma_logits_argmax_kernel<<<1, 256, 0, stream>>>(
            pw.d_embed, ws.d_normed, static_cast<int>(V),
            static_cast<int>(H), softcap, ws.d_argmax, nullptr, ws.d_logits);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        e = cudaMemcpyAsync(ws.h_arg, ws.d_argmax, 4, cudaMemcpyDeviceToHost,
                            stream);
        if (e != cudaSuccess) break;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) break;
        tok = ws.h_arg[0];
        out.tokens.push_back(tok);
    }

    // Capture the last-step logits (D2H).
    out.final_logits.resize(static_cast<size_t>(V));
    e = cudaMemcpyAsync(out.final_logits.data(), ws.d_logits,
                        V * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);

    cudaEventRecord(ev_end, stream);
    cudaEventSynchronize(ev_end);
    float total_ms = 0.f;
    cudaEventElapsedTime(&total_ms, ev0, ev_end);
    out.total_ms = total_ms;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev_ttft);
    cudaEventDestroy(ev_end);
    g_expert_db = nullptr;
    out.ok = (e == cudaSuccess);
    if (e != cudaSuccess && out.error_msg.empty())
        out.error_msg = "decode: " + std::string(cudaGetErrorString(e));
    return e;
}

} // namespace gemma
} // namespace lsxcommon

