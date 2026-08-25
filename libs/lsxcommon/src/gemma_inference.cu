#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lsxcommon/gemma4_inference.h"
#include "lsxcommon/gemma4_inference_load.h"
#include "lsxcommon/expert_device_cache.h"

#include "nvfp4_codec.h"
#include "lsx_kernels_quant.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace gemma {

__device__ __forceinline__ float gelu_tanh_dev(float v)
{
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v * (1.0f + tanhf(k0 * (v + 0.044715f * v * v * v)));
}

// NVFP4 elementwise dequant: unified kernels::Nvfp4DequantKernel<__nv_bfloat16>
// (lsx_kernels_quant.cuh), byte-equal to the former gemma_dequant_nvfp4_kernel.

static cudaError_t DequantNvfp4(cudaStream_t stream, const uint8_t *d_packed,
                                int64_t packed_bytes, const uint8_t *d_scales,
                                int64_t scales_bytes, float global_scale,
                                __nv_bfloat16 *d_out, int64_t num_elements)
{
    if (!d_packed || !d_scales || !d_out || packed_bytes <= 0 ||
        scales_bytes <= 0 || num_elements <= 0)
        return cudaErrorInvalidValue;
    if (packed_bytes * 2 != num_elements) return cudaErrorInvalidValue;
    if (num_elements / 2 < scales_bytes) return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((num_elements + kBlock - 1) / kBlock);
    lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
        <<<grid, kBlock, 0, stream>>>(d_packed, d_scales, global_scale,
                                      d_out, num_elements);
    return cudaGetLastError();
}

// ── RMSNorm (offset-weight form) ────────────────────────────────────────────

// RMSNorm (direct weight): unified kernels::RmsNormTreeKernel.
// RMSNorm (no weight): unified kernels::RmsNormTreeKernel<None>.

static cudaError_t RunGemmaRmsNormPlain(const __nv_bfloat16 *d_x,
                                        __nv_bfloat16 *d_out, int rows,
                                        int hidden, float eps,
                                        cudaStream_t stream)
{
    if (!d_x || !d_out || rows <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    ::lsxcommon::kernels::RmsNormTreeKernel<__nv_bfloat16, ::lsxcommon::kernels::NormWeight::None><<<rows, kBlock, kBlock * sizeof(float), stream>>>(
        d_x, nullptr, d_out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunGemmaRMSNorm(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_w, __nv_bfloat16 *d_out,
                            int rows, int hidden, float eps,
                            cudaStream_t stream)
{
    if (!d_x || !d_w || !d_out || rows <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    ::lsxcommon::kernels::RmsNormTreeKernel<__nv_bfloat16, ::lsxcommon::kernels::NormWeight::Direct><<<rows, kBlock, kBlock * sizeof(float), stream>>>(
        d_x, d_w, d_out, rows, hidden, eps);
    return cudaGetLastError();
}

// ── RoPE ────────────────────────────────────────────────────────────────────

__global__ void gemma_rope_cos_sin_kernel(const int32_t *__restrict__ pos,
                                          __nv_bfloat16 *__restrict__ cos_buf,
                                          __nv_bfloat16 *__restrict__ sin_buf,
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
        cos_buf[idx] = __float2bfloat16(static_cast<float>(cos(ang)));
        sin_buf[idx] = __float2bfloat16(static_cast<float>(sin(ang)));
    }
}

// HF rotate_half pairing: pair (j, j+rot/2); cos/sin layout [seq, rot/2].
__global__ void gemma_partial_rope_kernel(const __nv_bfloat16 *__restrict__ x,
                                          const __nv_bfloat16 *__restrict__ cos_buf,
                                          const __nv_bfloat16 *__restrict__ sin_buf,
                                          __nv_bfloat16 *__restrict__ out, int seq,
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
        const float c = __bfloat162float(cos_buf[t * half + j]);
        const float s = __bfloat162float(sin_buf[t * half + j]);
        const int64_t row = (static_cast<int64_t>(t) * heads + h) * head_dim;
        const float x1 = __bfloat162float(x[row + j]);
        const float x2 = __bfloat162float(x[row + j + half]);
        out[row + j] = __float2bfloat16(x1 * c - x2 * s);
        out[row + j + half] = __float2bfloat16(x2 * c + x1 * s);
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

cudaError_t RunGemmaRopeCosSin(const int32_t *d_pos, __nv_bfloat16 *d_cos,
                               __nv_bfloat16 *d_sin, int seq, int rotary_dim,
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

cudaError_t RunGemmaPartialRope(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_cos,
                                const __nv_bfloat16 *d_sin, __nv_bfloat16 *d_out, int seq,
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
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ out, int seq,
    int total_slots, int pos_base, int q_heads, int kv_heads, int head_dim,
    int window, float scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= seq || h >= q_heads) return;

    const int kvh = h / (q_heads / kv_heads);
    const __nv_bfloat16 *qr = q + (static_cast<int64_t>(t) * q_heads + h) * head_dim;
    __nv_bfloat16 *orr = out + (static_cast<int64_t>(t) * q_heads + h) * head_dim;
    const int last = pos_base + t;
    const int first =
        (window > 0) ? (last - window + 1 > 0 ? last - window + 1 : 0) : 0;
    const int n_valid = last - first + 1;
    if (n_valid <= 0) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) orr[d] = __float2bfloat16(0.f);
        return;
    }

    extern __shared__ float red[];
    __shared__ float s_max, s_sum;

    // Pass A: global max score.
    float local_max = -1e30f;
    for (int s = first + threadIdx.x; s <= last; s += blockDim.x) {
        const __nv_bfloat16 *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
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
        const __nv_bfloat16 *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += __bfloat162float(qr[d]) * __bfloat162float(kr[d]);
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
            const __nv_bfloat16 *kr = k + (static_cast<int64_t>(s) * kv_heads + kvh) * head_dim;
            float dot = 0.f;
            for (int dd = 0; dd < head_dim; ++dd)
                dot += __bfloat162float(qr[dd]) * __bfloat162float(kr[dd]);
            const float p = expf(dot * scale - s_max) / s_sum;
            acc += p * __bfloat162float(v[(static_cast<int64_t>(s) * kv_heads + kvh) *
                                       head_dim +
                                       d]);
        }
        orr[d] = __float2bfloat16(acc);
    }
}

cudaError_t RunGemmaAttention(const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k,
                              const __nv_bfloat16 *d_v, __nv_bfloat16 *d_out, int seq,
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

// out[t,c] = sum_i down[c,i] * gelu(gate[i]@x_t) * (up[i]@x_t)
__global__ void gemma_dense_mlp_kernel(const __nv_bfloat16 *__restrict__ gate,
                                       const __nv_bfloat16 *__restrict__ up,
                                       const __nv_bfloat16 *__restrict__ down,
                                       int64_t I, int64_t H,
                                       const __nv_bfloat16 *__restrict__ x,
                                       __nv_bfloat16 *__restrict__ out, int seq)
{
    // grid: (seq, H); one thread per output element.
    const int t = blockIdx.x;
    const int c = blockIdx.y;
    if (t >= seq || c >= H) return;
    const __nv_bfloat16 *xr = x + static_cast<int64_t>(t) * H;
    float acc = 0.f;
    for (int64_t i = 0; i < I; ++i) {
        float g = 0.f, u = 0.f;
        for (int64_t d = 0; d < H; ++d) {
            float xv = __bfloat162float(xr[d]);
            g += xv * __bfloat162float(gate[i * H + d]);
            u += xv * __bfloat162float(up[i * H + d]);
        }
        acc += gelu_tanh_dev(g) * u *
               __bfloat162float(down[c * I + i]);
    }
    out[static_cast<int64_t>(t) * H + c] = __float2bfloat16(acc);
}

cudaError_t RunGemmaDenseMLP(const __nv_bfloat16 *d_gate, const __nv_bfloat16 *d_up,
                             const __nv_bfloat16 *d_down, int64_t I, int64_t H,
                             const __nv_bfloat16 *d_x, __nv_bfloat16 *d_out, int seq,
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

// Gemma4 router (authentic): softmax over ALL expert logits, top-k on the
// probabilities, renormalize to sum 1, then scale by per_expert_scale[e].
// One block per token; deterministic lower-index top-k tie-break.
__global__ void gemma_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ proj,
    const float *__restrict__ per_expert_scale, const __nv_bfloat16 *__restrict__ x,
    int E, int H, int K, int32_t *__restrict__ sel, float *__restrict__ weights)
{
    const int t = blockIdx.x;
    if (t >= gridDim.x) return;
    const __nv_bfloat16 *xr = x + static_cast<int64_t>(t) * H;

    extern __shared__ float sh[];
    float *logits = sh;             // [E]

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < H; ++d)
            dot += __bfloat162float(proj[static_cast<int64_t>(e) * H + d]) *
                   __bfloat162float(xr[d]);
        logits[e] = dot;
    }
    __syncthreads();

    // Thread 0: full softmax -> top-k on probabilities -> renorm -> scale.
    if (threadIdx.x == 0) {
        float m = logits[0];
        for (int e = 1; e < E; ++e) m = fmaxf(m, logits[e]);
        float z = 0.f;
        for (int e = 0; e < E; ++e) z += expf(logits[e] - m);
        float probs[128];
        for (int e = 0; e < E; ++e) probs[e] = expf(logits[e] - m) / z;

        int idx[16];
        float val[16];
        int n = 0;
        for (int e = 0; e < E; ++e) {
            float v = probs[e];
            int pos = n;
            while (pos > 0 &&
                   (val[pos - 1] < v || (val[pos - 1] == v && idx[pos - 1] > e))) {
                if (pos < K) { val[pos] = val[pos - 1]; idx[pos] = idx[pos - 1]; }
                --pos;
            }
            if (pos < K) {
                val[pos] = v;
                idx[pos] = e;
                if (n < K) ++n;
            }
        }
        float s = 0.f;
        for (int i = 0; i < K; ++i) s += val[i];
        for (int i = 0; i < K; ++i) {
            sel[static_cast<int64_t>(t) * K + i] = idx[i];
            float w = val[i] / s;
            if (per_expert_scale) w *= per_expert_scale[idx[i]];
            weights[static_cast<int64_t>(t) * K + i] = w;
        }
    }
}

cudaError_t RunGemmaRouterTopK(const __nv_bfloat16 *d_router_proj,
                               const float *d_per_expert_scale,
                               const __nv_bfloat16 *d_x, int num_experts,
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
__global__ void gemma_moe_combine_kernel(__nv_bfloat16 *__restrict__ out,
                                         const __nv_bfloat16 *const *__restrict__ eos,
                                         const float *__restrict__ w,
                                         const float *__restrict__ rscale,
                                         int H, int K)
{
    const int t = blockIdx.x;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += w[static_cast<int64_t>(t) * K + k] *
                   __bfloat162float(eos[k][static_cast<int64_t>(t) * H + h]);
        if (rscale) acc *= rscale[h];
        out[static_cast<int64_t>(t) * H + h] = __float2bfloat16(acc);
    }
}

cudaError_t RunGemmaMoECombine(const __nv_bfloat16 *const *d_expert_outs,
                               const float *d_weights,
                               const float *d_router_scale, __nv_bfloat16 *d_out,
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

__global__ void gemma_embed_kernel(const __nv_bfloat16 *__restrict__ embed,
                                   const int32_t *__restrict__ ids,
                                   __nv_bfloat16 *__restrict__ out, int hidden)
{
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        out[static_cast<int64_t>(t) * hidden + d] =
            embed[static_cast<int64_t>(ids[t]) * hidden + d];
}

// out[t,r] = sum_d W[r,d] * x[t,d]   (W [rows, cols] row-major)
// cuBLAS FP16 GEMM (FP32 compute): row-major trick via column-major views —
//   C_cm[rows x seq] = W_cmᵀ[rows x cols] · X_cm[cols x seq]
__global__ void gemma_matmul_kernel_unused_guard() {}

static cudaError_t Matmul(cublasHandle_t cublas, const __nv_bfloat16 *w,
                          const __nv_bfloat16 *x, __nv_bfloat16 *out, int rows, int cols,
                          int seq, cudaStream_t stream)
{
    // Compute type FP32 => alpha/beta must be host floats.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetStream(cublas, stream);
    cublasStatus_t st = cublasGemmEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, seq, cols, &alpha, w,
        CUDA_R_16BF, cols, x, CUDA_R_16BF, cols, &beta, out, CUDA_R_16BF,
        rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// h[t,i] = gelu_tanh(g[t,i]) * u[t,i]
__global__ void gemma_geglu_kernel(const __nv_bfloat16 *__restrict__ g,
                                   const __nv_bfloat16 *__restrict__ u,
                                   __nv_bfloat16 *__restrict__ h, int64_t numel)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel)
        h[i] = __float2bfloat16(gelu_tanh_dev(__bfloat162float(g[i])) *
                            __bfloat162float(u[i]));
}

// Per-head RMSNorm over packed [seq, heads*dim]; Gemma4 w-form when w
// non-null, plain (with_scale=False) when w null (v_norm).
__global__ void gemma_head_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                       const __nv_bfloat16 *__restrict__ w,
                                       __nv_bfloat16 *__restrict__ out, int seq,
                                       int heads, int dim, float eps)
{
    extern __shared__ float squares[];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int base = h * dim;
    const __nv_bfloat16 *xr = x + static_cast<int64_t>(t) * heads * dim + base;
    __nv_bfloat16 *orr = out + static_cast<int64_t>(t) * heads * dim + base;

    float sum = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = __bfloat162float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / static_cast<float>(dim) + eps);
    // Gemma4 head norms: output = rms(h) * w (w-form q/k_norm). When `w` is
    // null, plain RMSNorm (v_norm, with_scale=False).
    if (w) {
        for (int j = threadIdx.x; j < dim; j += blockDim.x)
            orr[j] = __float2bfloat16(__bfloat162float(xr[j]) * inv *
                                      __bfloat162float(w[j]));
    } else {
        for (int j = threadIdx.x; j < dim; j += blockDim.x)
            orr[j] = __float2bfloat16(__bfloat162float(xr[j]) * inv);
    }
}

// dst *= scalar (elementwise)
__global__ void gemma_scale_kernel(__nv_bfloat16 *__restrict__ dst, float scalar,
                                   int64_t numel)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel)
        dst[i] = __float2bfloat16(__bfloat162float(dst[i]) * scalar);
}

// dst[t,d] += scalar * src[t,d]
__global__ void gemma_axpy_kernel(__nv_bfloat16 *__restrict__ dst,
                                  const __nv_bfloat16 *__restrict__ src,
                                  float scalar, int64_t numel)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel)
        dst[i] = __float2bfloat16(__bfloat162float(dst[i]) +
                              scalar * __bfloat162float(src[i]));
}

// dst[t,d] *= scale[d] * extra  (router input scaling)
__global__ void gemma_scale_vec_kernel(__nv_bfloat16 *__restrict__ dst,
                                       const float *__restrict__ scale,
                                       float extra, int hidden, int seq)
{
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        dst[static_cast<int64_t>(t) * hidden + d] = __float2bfloat16(
            __bfloat162float(dst[static_cast<int64_t>(t) * hidden + d]) *
            scale[d] * extra);
}

// dst[t,d] *= scale[d]
__global__ void gemma_hadamard_kernel(__nv_bfloat16 *__restrict__ dst,
                                      const float *__restrict__ scale,
                                      int hidden, int seq)
{
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x)
        dst[static_cast<int64_t>(t) * hidden + d] = __float2bfloat16(
            __bfloat162float(dst[static_cast<int64_t>(t) * hidden + d]) *
            scale[d]);
}

// logits[v] = dot(embed[v], x); then softcap tanh(l/cap)*cap; block-0 argmax.
__global__ void gemma_logits_argmax_kernel(const __nv_bfloat16 *__restrict__ embed,
                                           const __nv_bfloat16 *__restrict__ x,
                                           int vocab, int hidden, float cap,
                                           int32_t *__restrict__ d_argmax,
                                           float *__restrict__ d_maxlogit,
                                           float *__restrict__ d_logits)
{
    // One thread per vocab row (strided).
    float local_best = -1e30f;
    int local_arg = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const __nv_bfloat16 *row = embed + static_cast<int64_t>(v) * hidden;
        float acc = 0.f;
        for (int d = 0; d < hidden; ++d)
            acc += __bfloat162float(row[d]) * __bfloat162float(x[d]);
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

// Public generation-context lifecycle (declared in gemma4_inference.h with
// external linkage; consumed by GemmaBaseModel). Must live OUTSIDE any
// anonymous namespace: an internal-linkage definition here mangles to
// _GLOBAL__N__... and leaves gemma_base_model.cpp's call unresolved.
GemmaGenContext *CreateGemmaGenContext() { return new GemmaGenContext(); }
void DestroyGemmaGenContext(GemmaGenContext *ctx) { delete ctx; }

namespace {

// Debug aid (LSX_TEST_GEMMA_DEBUG_SYNC=1): sync+check after every stage of
// ForwardLayer to pinpoint asynchronous faults on real checkpoints.
bool GemmaDebugSync(cudaStream_t stream, const char *stage, int layer,
                    cudaError_t &e)
{
    static const bool on =
        std::getenv("LSX_TEST_GEMMA_DEBUG_SYNC") != nullptr;
    if (!on) return false;
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        printf("[gemma-dbg] fault after stage='%s' layer=%d: %s\n", stage,
               layer, cudaGetErrorString(e));
        fflush(stdout);
        return true;
    }
    return false;
}

// Debug switches (NaN bisection on real checkpoints).
bool GemmaSkipBranch(const char *name){
    static const bool no_attn =
        std::getenv("LSX_TEST_GEMMA_NO_ATTN") != nullptr;
    static const bool no_dense =
        std::getenv("LSX_TEST_GEMMA_NO_DENSE") != nullptr;
    static const bool no_moe =
        std::getenv("LSX_TEST_GEMMA_NO_MOE") != nullptr;
    if (std::strcmp(name, "attn") == 0) return no_attn;
    if (std::strcmp(name, "dense") == 0) return no_dense;
    return no_moe; // "moe"
}

// File-scope expert db reference used inside ForwardLayer (set by
// RunGemmaGeneration; single-threaded generation contract).
GemmaExpertDb *g_expert_db = nullptr;

// Capacity-grown workspace reuse: reallocates only when a request exceeds
// every recorded capacity. Same allocation layout as the previous per-call
// path; contents are scratch and never carry values between calls.
cudaError_t EnsureWorkspace(GemmaGenContext &ctx, int64_t H, int64_t V,
                            int64_t MI, int64_t I_max, size_t S,
                            size_t q_rows_max, size_t kv_rows_max,
                            size_t rot_half_max, int K)
{
    GenWorkspace &ws = ctx.ws;
    if (ctx.ws_ready && ctx.seq0_cap >= S && ctx.q_rows_cap >= q_rows_max &&
        ctx.kv_rows_cap >= kv_rows_max && ctx.rot_half_cap >= rot_half_max &&
        ctx.K_cap >= K)
        return cudaSuccess;
    ws.Free();
    cudaError_t e = cudaSuccess;
    auto A = [&](void **p, size_t bytes) { return cudaMalloc(p, bytes); };
    if ((e = A((void **)&ws.d_x, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_normed, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmp, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_qr, S * q_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_kr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_vr, S * kv_rows_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmp2, S * kv_rows_max * 2)) != cudaSuccess ||
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
        (e = A((void **)&ws.d_gg, S * I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_gu, S * I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eg, MI * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_eu, MI * 2)) != cudaSuccess ||
        cudaMallocHost(&ws.h_sel, S * K * 4) != cudaSuccess ||
        cudaMallocHost(&ws.h_selw, S * K * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&ws.h_arg, 4) != cudaSuccess) {
        return e != cudaSuccess ? e : cudaErrorMemoryAllocation;
    }
    ctx.seq0_cap = S; ctx.q_rows_cap = q_rows_max;
    ctx.kv_rows_cap = kv_rows_max; ctx.rot_half_cap = rot_half_max;
    ctx.K_cap = K;
    ctx.ws_ready = true;
    return cudaSuccess;
}

GemmaGenContext *g_gen_ctx = nullptr;

// LRU device slot cache for routed experts: [slots] × (pack|scales), one
// slot per flat expert id. Strides derive from the uniform expert geometry
// (+16B scale slack, matching the ws.d_s* scratch sizing).
cudaError_t EnsureGemmaExpertCache(GemmaGenContext &ctx,
                                   const GemmaPinnedWeights &pw,
                                   const GemmaExpertDb &db)
{
    if (ctx.ec.slots > 0) return cudaSuccess;
    const int64_t MIH = static_cast<int64_t>(pw.top.moe_intermediate_size) *
                        pw.top.hidden_size;
    const int64_t pack = MIH / 2 * 3;
    const int64_t scale = MIH / 16 + 16;
    int slots = db.num_layers * db.experts_per_layer;
    if (const char *env = std::getenv("LSX_GEMMA_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env);
        if (v > 0) slots = std::min(slots, v);
    }
    slots = std::min(slots, 1024);
    if (slots <= 0) return cudaErrorInvalidValue;

    ctx.pack_stride = pack;
    ctx.scale_stride = scale;
    ctx.ec.slots = slots;
    ctx.ec.total_experts = db.num_layers * db.experts_per_layer;
    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, ctx.ec.DeviceBytes());
    if (e != cudaSuccess) return e;
    ctx.ec.d_base = static_cast<uint8_t *>(base);
    ctx.ec.d_pack = ctx.ec.d_base;
    ctx.ec.d_scale = ctx.ec.d_base +
                     static_cast<int64_t>(slots) * ctx.pack_stride;
    ctx.ec.h_slot_of = new int32_t[static_cast<size_t>(ctx.ec.total_experts)];
    ctx.ec.h_slot_owner = new int32_t[static_cast<size_t>(slots)];
    ctx.ec.h_lru = new int32_t[static_cast<size_t>(slots)];
    for (int i = 0; i < ctx.ec.total_experts; ++i) ctx.ec.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        ctx.ec.h_slot_owner[i] = -1;
        ctx.ec.h_lru[i] = i;
    }
    return cudaSuccess;
}

// Forward `seq` tokens (hidden states in ws.d_x) through one layer; writes
// k/v rows at [pos_base .. pos_base+seq) of the layer cache.
cudaError_t ForwardLayer(cudaStream_t stream, cublasHandle_t cublas,
                         const GemmaPinnedWeights &pw,
                         const GemmaLayerWeights &lw, int layer_idx,
                         GenWorkspace &ws, int seq, int pos_base, __nv_bfloat16 *dk,
                         __nv_bfloat16 *dv)
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
    // layer_scalar is a constant weight: take the host mirror uploaded at
    // Load instead of a device readback + full stream sync per layer per
    // forward (bit-identical value).
    float layer_scalar = 1.0f;
    if (layer_idx >= 0 &&
        static_cast<size_t>(layer_idx) < pw.h_layer_scalar.size())
        layer_scalar = pw.h_layer_scalar[static_cast<size_t>(layer_idx)];
    else if (lw.d_layer_scalar) {
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
    //    (attention_k_eq_v): no v_proj weights exist, so v := raw k proj.
    const bool v_tied_to_k = (lw.d_v == nullptr);
    e = Matmul(cublas, lw.d_q, ws.d_normed, ws.d_qr, q_rows,
               static_cast<int>(H), seq, stream);
    if (e != cudaSuccess) return e;
    e = Matmul(cublas, lw.d_k, ws.d_normed, ws.d_kr, kv_heads * head_dim,
               static_cast<int>(H), seq, stream);
    if (e != cudaSuccess) return e;
    if (!v_tied_to_k) {
        e = Matmul(cublas, lw.d_v, ws.d_normed, ws.d_vr, kv_heads * head_dim,
                   static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) return e;
    } else {
        // v := raw k_proj output (before k_norm) — value = v_norm(raw k).
        e = cudaMemcpyAsync(ws.d_vr, ws.d_kr,
                            static_cast<size_t>(kv_heads * head_dim) * seq * 2,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    if (GemmaDebugSync(stream, "qkv", layer_idx, e)) return e;

    // 3) QK-norms (w-form) + v_norm (plain, with_scale=False).
    {
        dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(heads));
        gemma_head_norm_kernel<<<grid, 128, 128 * sizeof(float), stream>>>(
            ws.d_qr, lw.d_q_norm, ws.d_qr, seq, heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        dim3 grid2(static_cast<unsigned>(seq),
                   static_cast<unsigned>(kv_heads));
        gemma_head_norm_kernel<<<grid2, 128, 128 * sizeof(float), stream>>>(
            ws.d_kr, lw.d_k_norm, ws.d_kr, seq, kv_heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        // v_norm: plain RMSNorm (with_scale=False, no weights) on values.
        gemma_head_norm_kernel<<<grid2, 128, 128 * sizeof(float), stream>>>(
            ws.d_vr, nullptr, ws.d_vr, seq, kv_heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    if (GemmaDebugSync(stream, "qknorm-vnorm", layer_idx, e)) return e;

    // 4) RoPE on q and k only. Keys: k_normed then rotated; values keep the
    //    v_normed (plain) form, never rotated.
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
    if (!GemmaSkipBranch("attn")) {
        e = RunGemmaAttention(ws.d_qr, dk, dv, ws.d_tmp, seq,
                              static_cast<int>(pos_base + seq), pos_base,
                              heads, kv_heads, head_dim, window,
                              // HF Gemma4TextAttention uses scaling = 1.0:
                              // q/k are per-head RMS-normalized, so there is
                              // no additional 1/sqrt(head_dim) factor.
                              1.0f,
                              stream);
        if (e != cudaSuccess) return e;
    } else {
        e = cudaMemcpyAsync(ws.d_tmp, ws.d_qr, sizeof(__nv_bfloat16) * seq * q_rows,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    if (GemmaDebugSync(stream, "attn", layer_idx, e)) return e;

    // 7) o_proj + sandwich post-norm + residual.
    e = Matmul(cublas, lw.d_o, ws.d_tmp, ws.d_qr, static_cast<int>(H), q_rows,
               seq, stream);
    if (e != cudaSuccess) return e;
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_norm, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }

    // 8) Dense MLP branch: n = rms(x, pre_ffn); d = mlp(n); d1 = rms(d, post_ffn_1).
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t I = top.intermediate_size;
        const int64_t numel = static_cast<int64_t>(seq) * I;
        e = Matmul(cublas, lw.d_mlp_gate, ws.d_normed, ws.d_gg,
                   static_cast<int>(I), static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) return e;
        e = Matmul(cublas, lw.d_mlp_up, ws.d_normed, ws.d_gu,
                   static_cast<int>(I), static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) return e;
        gemma_geglu_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                             stream>>>(ws.d_gg, ws.d_gu, ws.d_gg, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = Matmul(cublas, lw.d_mlp_down, ws.d_gg, ws.d_qr, static_cast<int>(H),
                   static_cast<int>(I), seq, stream);
        if (e != cudaSuccess) return e;
    }
    // d1 = post_ffn_1(mlp_out)  -> keeps ws.d_qr
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_ffn1, ws.d_qr, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;

    // 9) Router: rn = rms_plain(x) * router_scale * H^-0.5; proj; softmax-all;
    //    top-k on probs; renorm; scale by per_expert_scale.
    const int E = top.num_experts;
    const int K = top.top_k_experts;
    const int64_t MI = top.moe_intermediate_size;
    e = RunGemmaRmsNormPlain(ws.d_x, ws.d_tmp2, seq, static_cast<int>(H),
                             top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const float root = 1.0f / std::sqrt(static_cast<float>(H));
        gemma_scale_vec_kernel<<<static_cast<unsigned>(seq), 256, 0, stream>>>(
            ws.d_tmp2, lw.d_router_scale, root, static_cast<int>(H), seq);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    e = RunGemmaRouterTopK(lw.d_router_proj, lw.d_per_expert_scale, ws.d_tmp2,
                           E, static_cast<int>(H), K, ws.d_sel, ws.d_selw,
                           seq, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(ws.h_sel, ws.d_sel, sizeof(int32_t) * seq * K,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(ws.h_selw, ws.d_selw, sizeof(float) * seq * K,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return e;

    // MoE: h2 = rms(x, pre_ffn2); moe(h2, sel, w); d2 = rms(moe, post_ffn2).
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn2, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemsetAsync(ws.d_acc, 0, sizeof(__nv_bfloat16) * seq * H, stream);
    if (e != cudaSuccess) return e;

    for (int t = 0; t < seq; ++t) {
        for (int k = 0; k < K; ++k) {
            const int expert =
                ws.h_sel[static_cast<size_t>(t) * K + k];
            const float weight =
                ws.h_selw[static_cast<size_t>(t) * K + k];
            const auto &ent =
                g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                         static_cast<size_t>(E) +
                                     static_cast<size_t>(expert)];
            if (!ent.host_w1 || !ent.host_w2) continue;

            gpu::ExpertDeviceCache &ec = g_gen_ctx->ec;
            const int flat = layer_idx * E + expert;
            int slot = ec.Lookup(flat);
            const bool miss = slot < 0;
            if (miss) slot = ec.Insert(flat);
            uint8_t *slot_pack = ec.d_pack +
                static_cast<int64_t>(slot) * g_gen_ctx->pack_stride;
            uint8_t *slot_scale = ec.d_scale +
                static_cast<int64_t>(slot) * g_gen_ctx->scale_stride;
            if (miss) {
                // Stream-ordered writes into the slot: safe against prior
                // same-stream readers of an evicted victim slot. Host mmaps
                // never evict (max_open=7000 > total experts), so no munmap
                // barrier is needed here.
                uint8_t *pk = slot_pack;
                uint8_t *sc = slot_scale;
                e = cudaMemcpyAsync(pk, ent.host_w1,
                                    static_cast<size_t>(ent.w1_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                pk += ent.w1_bytes;
                e = cudaMemcpyAsync(pk, ent.host_w2,
                                    static_cast<size_t>(ent.w2_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                pk += ent.w2_bytes;
                e = cudaMemcpyAsync(pk, ent.host_w3,
                                    static_cast<size_t>(ent.w3_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                e = cudaMemcpyAsync(sc, ent.host_w1_scales,
                                    static_cast<size_t>(ent.w1_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                sc += ent.w1_scales_bytes;
                e = cudaMemcpyAsync(sc, ent.host_w2_scales,
                                    static_cast<size_t>(ent.w2_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
                sc += ent.w2_scales_bytes;
                e = cudaMemcpyAsync(sc, ent.host_w3_scales,
                                    static_cast<size_t>(ent.w3_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                if (e != cudaSuccess) return e;
            }

            e = DequantNvfp4(stream, slot_pack, ent.w1_bytes, slot_scale,
                             ent.w1_scales_bytes, ent.w1_global_scale,
                             ws.d_w1, MI * H);
            if (e != cudaSuccess) return e;
            e = DequantNvfp4(stream, slot_pack + ent.w1_bytes, ent.w2_bytes,
                             slot_scale + ent.w1_scales_bytes,
                             ent.w2_scales_bytes, ent.w2_global_scale,
                             ws.d_w2, H * MI);
            if (e != cudaSuccess) return e;
            e = DequantNvfp4(stream,
                             slot_pack + ent.w1_bytes + ent.w2_bytes,
                             ent.w3_bytes,
                             slot_scale + ent.w1_scales_bytes +
                                 ent.w2_scales_bytes,
                             ent.w3_scales_bytes, ent.w3_global_scale,
                             ws.d_w3, MI * H);
            if (e != cudaSuccess) return e;

            // Expert FFN: out_e[t] = w2 @ (gelu(w1 @ x) * (w3 @ x)) —
            // cuBLAS GEMMs + elementwise GeGLU.
            const __nv_bfloat16 *x_t =
                ws.d_normed + static_cast<int64_t>(t) * H;
            e = Matmul(cublas, ws.d_w1, x_t, ws.d_eg, static_cast<int>(MI),
                       static_cast<int>(H), 1, stream);
            if (e != cudaSuccess) return e;
            e = Matmul(cublas, ws.d_w3, x_t, ws.d_eu, static_cast<int>(MI),
                       static_cast<int>(H), 1, stream);
            if (e != cudaSuccess) return e;
            gemma_geglu_kernel<<<static_cast<unsigned>((MI + 255) / 256), 256,
                                 0, stream>>>(ws.d_eg, ws.d_eu, ws.d_eg, MI);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
            e = Matmul(cublas, ws.d_w2, ws.d_eg, ws.d_eout, static_cast<int>(H),
                       static_cast<int>(MI), 1, stream);
            if (e != cudaSuccess) return e;

            gemma_axpy_kernel<<<static_cast<unsigned>((H + 255) / 256), 256, 0,
                                stream>>>(
                ws.d_acc + static_cast<int64_t>(t) * H, ws.d_eout, weight, H);
            e = cudaGetLastError();
            if (e != cudaSuccess) return e;
            if (GemmaDebugSync(stream, "moe-expert", layer_idx, e)) return e;
        }
    }

    // d2 = post_ffn2(moe_out); combine ff = d1 + d2; ff = post_ffn(ff);
    // hidden += ff; hidden *= layer_scalar.
    e = RunGemmaRMSNorm(ws.d_acc, lw.d_post_ffn2, ws.d_tmp2, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_tmp2, ws.d_qr, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    e = RunGemmaRMSNorm(ws.d_tmp2, lw.d_post_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) return e;
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    gemma_scale_kernel<<<static_cast<unsigned>((seq * H + 255) / 256), 256, 0,
                         stream>>>(ws.d_x, layer_scalar,
                                   static_cast<int64_t>(seq) * H);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return cudaSuccess;
}

} // namespace

cudaError_t RunGemmaGeneration(cudaStream_t stream, const GemmaPinnedWeights &pw,
                               GemmaExpertDb &db,
                               const std::vector<int32_t> &input_ids,
                               int max_gen_tokens, int eos_id, float softcap,
                               GemmaGenerationResult &out, GemmaKvCache &kv,
                               GemmaGenContext *ctx)
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
    const int64_t I_max =
        std::max(top.intermediate_size, top.moe_intermediate_size);
    if (input_ids.empty() || max_gen_tokens <= 0 || H <= 0 || V <= 0 ||
        pw.layers.empty() || !kv.allocated())
        return cudaErrorInvalidValue;

    GemmaGenContext local_ctx;
    GemmaGenContext &c = ctx ? *ctx : local_ctx;
    if (!c.cublas &&
        cublasCreate(&c.cublas) != CUBLAS_STATUS_SUCCESS) {
        out.error_msg = "cublas create failed";
        return cudaErrorInvalidValue;
    }
    cublasHandle_t cublas = c.cublas;

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

    cudaError_t e = EnsureWorkspace(c, H, V, MI, I_max, S, q_rows_max,
                                    kv_rows_max, rot_half_max, K);
    if (e != cudaSuccess) {
        out.error_msg = "workspace alloc failed";
        return e;
    }
    GenWorkspace &ws = c.ws;
    ws.seq0 = S;
    ws.q_rows_max = q_rows_max;
    ws.kv_rows_max = kv_rows_max;
    ws.rot_half_max = rot_half_max;
    ws.K = K;

    e = EnsureGemmaExpertCache(c, pw, db);
    if (e != cudaSuccess) {
        out.error_msg = "expert cache alloc failed";
        return e;
    }
    g_gen_ctx = &c;

    g_expert_db = &db;

    // ── Prefill ─────────────────────────────────────────────────────────────
    e = cudaMemcpyAsync(ws.d_pos, input_ids.data(), sizeof(int32_t) * P,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) { g_expert_db = nullptr; g_gen_ctx = nullptr; return e; }
    {
        dim3 grid(static_cast<unsigned>(P));
        gemma_embed_kernel<<<grid, 256, 0, stream>>>(
            pw.d_embed, ws.d_pos, ws.d_x, static_cast<int>(H));
        e = cudaGetLastError();
        if (e != cudaSuccess) { g_expert_db = nullptr; g_gen_ctx = nullptr; return e; }
        // Gemma scales embeddings by sqrt(hidden_size).
        const float normalizer = std::sqrt(static_cast<float>(H));
        const int64_t numel = static_cast<int64_t>(P) * H;
        gemma_scale_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256,
                             0, stream>>>(ws.d_x, normalizer, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { g_expert_db = nullptr; g_gen_ctx = nullptr; return e; }
    }
    // Wall-clock timing via std::chrono: the previous cudaEvent-based TTFT
    // measurement returned a sticky "invalid resource handle" from
    // cudaEventElapsedTime on cudaEventDisableTiming events, which the decode
    // loop's error checks then misread as a generation failure after the
    // first token.
    const auto t0 = std::chrono::steady_clock::now();

    for (size_t l = 0; l < pw.layers.size(); ++l) {
        e = ForwardLayer(stream, cublas, pw, pw.layers[l],
                         static_cast<int>(l), ws, P, 0, kv.dk[l], kv.dv[l]);
        if (e != cudaSuccess) {
            out.error_msg = "prefill layer " + std::to_string(l) + ": " +
                            cudaGetErrorString(e);
        g_expert_db = nullptr;
        g_gen_ctx = nullptr;
            g_gen_ctx = nullptr;
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
    if (e != cudaSuccess) { g_expert_db = nullptr; g_gen_ctx = nullptr; return e; }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) { g_expert_db = nullptr; g_gen_ctx = nullptr; return e; }
    const double ttft_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count();
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
        fflush(stdout);
        e = cudaMemcpyAsync(ws.d_pos, &tok, 4, cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) break;
        gemma_embed_kernel<<<1, 256, 0, stream>>>(pw.d_embed, ws.d_pos,
                                                  ws.d_x,
                                                  static_cast<int>(H));
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        {
            const int64_t numel = H;
            gemma_scale_kernel<<<static_cast<unsigned>((numel + 255) / 256),
                                 256, 0, stream>>>(
                ws.d_x, std::sqrt(static_cast<float>(H)), numel);
            e = cudaGetLastError();
            if (e != cudaSuccess) break;
        }
        bool failed = false;
        for (size_t l = 0; l < pw.layers.size(); ++l) {
            e = ForwardLayer(stream, cublas, pw, pw.layers[l],
                             static_cast<int>(l), ws, /*seq=*/1, pos,
                             kv.dk[l], kv.dv[l]);
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

    out.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count();

    g_expert_db = nullptr;
    g_gen_ctx = nullptr;
    out.ok = (e == cudaSuccess);
    if (e != cudaSuccess && out.error_msg.empty())
        out.error_msg = "decode: " + std::string(cudaGetErrorString(e));
    return e;
}

} // namespace gemma
} // namespace lsxcommon












