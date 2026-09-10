#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lsxcommon/gemma_moe_inference.h"
#include "lsxcommon/qwen_moe_inference.h"
#include "qwen_reduce.cuh"
#include "lsxcommon/gemma_moe_load.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/log.h"

#include "nvfp4_codec.h"
#include "lsx_kernels_quant.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace gemma {

// Round-21 helpers: definitions sit in the batched-GEMV chain section at
// the file tail; forward-declared here for the decode driver.
__global__ void gemma_bf16_to_half_kernel(const __nv_bfloat16 *__restrict__ src,
                                          __half *__restrict__ dst, int n);
__global__ void gemma_f32_to_bf16_kernel(const float *__restrict__ src,
                                         __nv_bfloat16 *__restrict__ dst,
                                         int n);
__global__ void gemma_gemv_zero_f32_kernel(float *__restrict__ dst, int n);
__global__ void gemma_gemv_gather_table_kernel(
    const int32_t *__restrict__ d_sel, const float *__restrict__ d_selw,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ flat_meta,
    int K, lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ tbl);
__global__ void gemma_embed_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float gs, const int32_t *__restrict__ ids,
    __nv_bfloat16 *__restrict__ out, int hidden);
__global__ void gemma_dense_mlp_act_kernel(const __half *__restrict__ g,
                                           const __half *__restrict__ u,
                                           __half *__restrict__ h, int n);
__global__ void gemma_f16_to_bf16_kernel(const __half *__restrict__ src,
                                         __nv_bfloat16 *__restrict__ dst,
                                         int n);
cudaError_t RunRoutedGemvBatchSeg(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *slots_dev, int k_slots,
    int64_t off_gate_p, int64_t off_gate_s,
    int64_t off_up_p, int64_t off_up_s,
    int64_t off_down_p, int64_t off_down_s,
    int n_gate_up, int k_gate_up,
    int n_down, int k_down,
    bool geglu,
    const __half *d_x,
    __half *ygu, __half *hbuf, float *acc_f32);
cudaError_t RunGemmaF32ToBf16(cudaStream_t stream, const float *src,
                              __nv_bfloat16 *dst, int64_t n);

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

cudaError_t RunGemmaSingleDequant(cudaStream_t stream, const uint8_t *packed,
                                  int64_t packed_bytes, const uint8_t *scales,
                                  int64_t scales_bytes, float global_scale,
                                  __nv_bfloat16 *out, int64_t num_elements)
{
    return DequantNvfp4(stream, packed, packed_bytes, scales, scales_bytes,
                        global_scale, out, num_elements);
}

namespace {
struct DequantBatchParams {
    const uint8_t *packs[16];
    const uint8_t *scales[16];
    float global_scales[16];
};
} // namespace

// Gather-dequant: K experts in one launch. Element mapping replicates
// Nvfp4DequantKernel exactly (same nibble order, scale block index and
// decode LUTs), so rows are BIT-IDENTICAL to per-expert dequants.
__global__ void gemma_dequant_gather_kernel(DequantBatchParams p,
                                            int64_t elems_per_expert, int k,
                                            __nv_bfloat16 *__restrict__ out)
{
    const int64_t total = static_cast<int64_t>(k) * elems_per_expert;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int e = static_cast<int>(i / elems_per_expert);
        const int64_t j = i - static_cast<int64_t>(e) * elems_per_expert;
        const uint8_t byte = p.packs[e][j >> 1];
        const int nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
        out[i] = __float2bfloat16(
            gpu::nvfp4::E2M1Value(nib) *
            gpu::nvfp4::E4M3ScaleToFloat(p.scales[e][j >> 4]) *
            p.global_scales[e]);
    }
}

cudaError_t RunGemmaBatchedDequant(cudaStream_t stream,
                                   const GemmaDequantBatch &batch,
                                   int64_t num_elements_per_expert, int k,
                                   __nv_bfloat16 *out_rows)
{
    if (!out_rows || num_elements_per_expert <= 0 || k <= 0 || k > 16)
        return cudaErrorInvalidValue;
    for (int e = 0; e < k; ++e)
        if (!batch.packs[e] || !batch.scales[e])
            return cudaErrorInvalidValue;
    DequantBatchParams params{};
    for (int e = 0; e < k; ++e) {
        params.packs[e] = batch.packs[e];
        params.scales[e] = batch.scales[e];
        params.global_scales[e] = batch.global_scales[e];
    }
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(k) * num_elements_per_expert;
    const unsigned grid = static_cast<unsigned>(
        (total + kBlock - 1) / kBlock);
    gemma_dequant_gather_kernel<<<grid, kBlock, 0, stream>>>(
        params, num_elements_per_expert, k, out_rows);
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
    ::lsxcommon::kernels::RmsNormTreeKernel<__nv_bfloat16, ::lsxcommon::kernels::NormWeight::None><<<rows, kBlock,
        kBlock * sizeof(float) + std::min(hidden * int(sizeof(__nv_bfloat16)), 48 * 1024), stream>>>(
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
    ::lsxcommon::kernels::RmsNormTreeKernel<__nv_bfloat16, ::lsxcommon::kernels::NormWeight::Direct><<<rows, kBlock,
        kBlock * sizeof(float) + std::min(hidden * int(sizeof(__nv_bfloat16)), 48 * 1024), stream>>>(
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

// ── Split-KV decode attention (flash-decode) ───────────────────────────────
// Port of the qwen split-KV design to the gemma BF16 geometry with the
// sliding-window band predicate. Stage 1: grid (heads, nchunks); each block
// owns a fixed 512-slot KV chunk INSIDE the band, computes chunk scores into
// shared memory, block-reduces the local max m_c and denominator l_c, and
// emits the local-max-weighted value partial (FP32). Stage 2: grid (heads);
// global max via fmaxf (order-free), denominator and weighted value walk
// chunks in ASCENDING order (fixed sequence -> run-to-run bit-identical).
namespace {

constexpr int kGemmaSplitChunk = 512;

__global__ void gemma_split_kv_partials_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, float *__restrict__ d_part,
    float *__restrict__ d_st, int q_heads, int kv_heads, int head_dim,
    float scale, int pos_base, int window, int nchunks, int chunk)
{
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int kvh = h / (q_heads / kv_heads);
    const int last = pos_base;
    const int first =
        (window > 0) ? (last - window + 1 > 0 ? last - window + 1 : 0) : 0;
    const int nscores = last - first + 1;
    const int begin = first + c * chunk;
    const int end = min(begin + chunk, first + nscores);
    const int rows = end - begin;
    const int threads = blockDim.x;
    if (rows <= 0) return;

    const __nv_bfloat16 *qrow = q + static_cast<int64_t>(h) * head_dim;
    const __nv_bfloat16 *kbase_p =
        k + static_cast<int64_t>(kvh) * head_dim;
    const __nv_bfloat16 *vbase_p =
        v + static_cast<int64_t>(kvh) * head_dim;

    extern __shared__ float slogits[];      // dynamic: [rows]
    __shared__ float sreduce[512];

    // 1. Chunk scores.
    for (int j = threadIdx.x; j < rows; j += threads) {
        const __nv_bfloat16 *krow =
            kbase_p + static_cast<int64_t>(begin + j) * kv_heads * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d)
            dot += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
        slogits[j] = dot * scale;
    }
    __syncthreads();

    // 2. Chunk max (warp-shuffle block reduction, order-free).
    float m = -1e30f;
    for (int j = threadIdx.x; j < rows; j += threads)
        m = fmaxf(m, slogits[j]);
    m = qwen_block_max_shfl(m);
    __syncthreads();

    // 3. Chunk denominator (one exp per score).
    float sum = 0.f;
    for (int j = threadIdx.x; j < rows; j += threads)
        sum += __expf(slogits[j] - m);
    sreduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float l_c = sreduce[0];
    __syncthreads();

    // 4. Local-max-weighted value partial: one thread per output dim,
    //    ascending j inside the chunk.
    {
        const int d = threadIdx.x;          // blockDim == head_dim (entry)
        float acc = 0.f;
        for (int j = 0; j < rows; ++j) {
            const __nv_bfloat16 *vrow =
                vbase_p +
                static_cast<int64_t>(begin + j) * kv_heads * head_dim;
            acc += __expf(slogits[j] - m) * __bfloat162float(vrow[d]);
        }
        d_part[(static_cast<int64_t>(h) * nchunks + c) * head_dim + d] = acc;
    }
    if (threadIdx.x == 0) {
        d_st[static_cast<int64_t>(h * nchunks + c) * 2 + 0] = m;
        d_st[static_cast<int64_t>(h * nchunks + c) * 2 + 1] = l_c;
    }
}

__global__ void gemma_split_kv_combine_kernel(
    const float *__restrict__ d_part, const float *__restrict__ d_st,
    __nv_bfloat16 *__restrict__ out, int q_heads, int head_dim, int nchunks)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;              // blockDim == head_dim (entry)
    const int64_t base = static_cast<int64_t>(h) * nchunks;
    __shared__ float sreduce[512];

    // Global max: fmaxf is order-free (deterministic).
    float m = -1e30f;
    for (int c = d; c < nchunks; c += blockDim.x)
        m = fmaxf(m, d_st[(base + c) * 2 + 0]);
    m = qwen_block_max_shfl(m);
    __syncthreads();

    // Denominator over rescaled chunk sums (fixed chunk-strided order).
    float den = 0.f;
    for (int c = d; c < nchunks; c += blockDim.x)
        den += d_st[(base + c) * 2 + 1] * __expf(d_st[(base + c) * 2 + 0] - m);
    sreduce[d] = den;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) sreduce[d] += sreduce[d + stride];
        __syncthreads();
    }
    const float inv = 1.0f / (sreduce[0] + 1e-10f);
    __syncthreads();

    // Weighted value: ascending chunk order (deterministic).
    float acc = 0.f;
    for (int c = 0; c < nchunks; ++c)
        acc += __expf(d_st[(base + c) * 2 + 0] - m) *
               d_part[(base + c) * head_dim + d];
    out[static_cast<int64_t>(h) * head_dim + d] = __float2bfloat16(acc * inv);
}

} // namespace

cudaError_t RunGemmaAttentionSplitKV(
    const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v, __nv_bfloat16 *d_out, int seq,
    int total_slots, int pos_base, int q_heads, int kv_heads, int head_dim,
    int window, float scale, cudaStream_t stream)
{
    if (!d_q || !d_k || !d_v || !d_out || seq != 1 || q_heads <= 0 ||
        head_dim <= 0 || head_dim > 512 || kv_heads <= 0 || pos_base < 0 ||
        pos_base >= total_slots)
        return cudaErrorInvalidValue;
    if (q_heads % kv_heads != 0) return cudaErrorInvalidValue;
    // Block reductions and the one-thread-per-dim value pass assume a
    // power-of-two blockDim (== head_dim); GQA mapping requires kv_heads to
    // divide q_heads exactly.
    if ((head_dim & (head_dim - 1)) != 0 || kv_heads > q_heads)
        return cudaErrorInvalidValue;

    const int last = pos_base;
    const int first =
        (window > 0) ? (last - window + 1 > 0 ? last - window + 1 : 0) : 0;
    const int nscores = last - first + 1;
    if (nscores <= 0) return cudaErrorInvalidValue;

    const int nchunks = (nscores + kGemmaSplitChunk - 1) / kGemmaSplitChunk;
    float *d_part = nullptr, *d_st = nullptr;
    cudaError_t e = cudaMallocAsync(
        &d_part,
        static_cast<size_t>(q_heads) * nchunks * head_dim * sizeof(float),
        stream);
    if (e != cudaSuccess) return e;
    e = cudaMallocAsync(&d_st,
                        static_cast<size_t>(q_heads) * nchunks * 2 *
                            sizeof(float),
                        stream);
    if (e != cudaSuccess) {
        cudaFreeAsync(d_part, stream);
        return e;
    }

    {
        const dim3 grid(q_heads, nchunks);
        const size_t smem =
            static_cast<size_t>(std::min(kGemmaSplitChunk, nscores)) *
            sizeof(float);
        gemma_split_kv_partials_kernel<<<grid, head_dim, smem, stream>>>(
            d_q, d_k, d_v, d_part, d_st, q_heads, kv_heads, head_dim, scale,
            pos_base, window, nchunks, kGemmaSplitChunk);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) {
        gemma_split_kv_combine_kernel<<<q_heads, head_dim, 0, stream>>>(
            d_part, d_st, d_out, q_heads, head_dim, nchunks);
        e = cudaGetLastError();
    }
    cudaFreeAsync(d_part, stream);
    cudaFreeAsync(d_st, stream);
    return e;
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

// NVFP4 tied-table embedding gather: row `ids[t]` dequantized on the fly.
__global__ void gemma_embed_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float gs, const int32_t *__restrict__ ids,
    __nv_bfloat16 *__restrict__ out, int hidden)
{
    const int t = blockIdx.x;
    const int token = ids[t];
    const int64_t row_packed = static_cast<int64_t>(token) * (hidden >> 1);
    const int64_t row_scales = static_cast<int64_t>(token) * (hidden >> 4);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const uint8_t byte = packed[row_packed + (d >> 1)];
        const int nib = (d & 1) ? (byte >> 4) : (byte & 0xF);
        const float w = gpu::nvfp4::E2M1Value(nib) *
                        gpu::nvfp4::E4M3ScaleToFloat(scales[row_scales + (d >> 4)]) *
                        gs;
        out[static_cast<int64_t>(t) * hidden + d] = __float2bfloat16(w);
    }
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

// Batched decode (seq==1): geGLU over all K experts' [K][MI] projections.
__global__ void gemma_geglu_batched_kernel(const __nv_bfloat16 *__restrict__ g,
                                           const __nv_bfloat16 *__restrict__ u,
                                           __nv_bfloat16 *__restrict__ h,
                                           int64_t numel)
{
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel)
        h[i] = __float2bfloat16(gelu_tanh_dev(__bfloat162float(g[i])) *
                                __bfloat162float(u[i]));
}

// Batched combine: dst[h] = sum_k (w[k] * eout[k][h]) — per-k bf16 rounding
// replicates the sequential axpy chain of the per-expert path.
__global__ void gemma_combine_batched_kernel(__nv_bfloat16 *__restrict__ acc,
                                             const __nv_bfloat16 *__restrict__ eout,
                                             const float *__restrict__ w,
                                             int k, int H)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= H) return;
    float a = __bfloat162float(acc[d]);
    for (int i = 0; i < k; ++i) {
        a += w[i] * __bfloat162float(
                        eout[static_cast<int64_t>(i) * H + d]);
    }
    acc[d] = __float2bfloat16(a);
}

// cuBLAS FP32-compute batched GEMM over @p batch rows (same op/modes as the
// scalar Matmul wrapper): C_b[rows x seq] = W_bᵀ · X (X shared when x_stride==0).
static cudaError_t MatmulBatched(cublasHandle_t cublas,
                                 const __nv_bfloat16 *a,
                                 long long stride_a, const __nv_bfloat16 *x,
                                 long long stride_x, __nv_bfloat16 *out,
                                 long long stride_out, int rows, int cols,
                                 int batch, cudaStream_t stream)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetStream(cublas, stream);
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols, &alpha, a,
        CUDA_R_16BF, cols, stride_a, x, CUDA_R_16BF, cols, stride_x, &beta,
        out, CUDA_R_16BF, rows, stride_out, batch, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
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
// ── Two-phase tiled version ────────────────────────────────────────────────
// The old single-block variant (<<<1,256>>>) serialized the whole lm-head
// GEMV on ONE SM (~262k vocab rows × 2816 = ~740M MACs, ~290 ms/token).
// Phase 1 spreads rows across the SM array with vectorized loads along the
// hidden dim while keeping the STRICT ascending-d accumulation order per
// row (bit-identical logits); phase 2 reduces block winners deterministically
// in ascending block order.
constexpr int kLogitsMaxBlocks = 4096;

__global__ void gemma_logits_rows_kernel(
    const __nv_bfloat16 *__restrict__ embed, const __nv_bfloat16 *__restrict__ x,
    int vocab, int hidden, float cap,
    float *__restrict__ d_logits,
    float *__restrict__ blk_best, int *__restrict__ blk_idx)
{
    const bool vec_ok =
        ((hidden & 7) == 0) &&
        ((reinterpret_cast<uintptr_t>(embed) |
          reinterpret_cast<uintptr_t>(x)) & 15) == 0;

    float local_best = -1e30f;
    int local_arg = -1;
    const int64_t span =
        (static_cast<int64_t>(gridDim.x)) * blockDim.x;
    // One-or-more full vocab rows per thread, ascending v order.
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < vocab; i += span) {
        const __nv_bfloat16 *row = embed + i * hidden;
        float acc = 0.f;
        if (vec_ok) {
            // Four bf16 elements per load, added in ascending-d order
            // (identical rounding sequence as the scalar walk).
            const int h4 = hidden >> 3;
            const uint4 *r4 = reinterpret_cast<const uint4 *>(row);
            const uint4 *x4 = reinterpret_cast<const uint4 *>(x);
            for (int q = 0; q < h4; ++q) {
                const uint4 rb = r4[q];
                const uint4 xb = x4[q];
                const __nv_bfloat16 *rp =
                    reinterpret_cast<const __nv_bfloat16 *>(&rb);
                const __nv_bfloat16 *xp =
                    reinterpret_cast<const __nv_bfloat16 *>(&xb);
                #pragma unroll
                for (int u = 0; u < 8; ++u)
                    acc += __bfloat162float(rp[u]) *
                           __bfloat162float(xp[u]);
            }
            for (int d = (h4 << 3); d < hidden; ++d)
                acc += __bfloat162float(row[d]) *
                       __bfloat162float(x[d]);
        } else {
            for (int d = 0; d < hidden; ++d)
                acc += __bfloat162float(row[d]) *
                       __bfloat162float(x[d]);
        }
        if (cap > 0.f) acc = tanhf(acc / cap) * cap;
        if (d_logits) d_logits[i] = acc;
        blk_best[i] = acc;
        blk_idx[i] = static_cast<int>(i);
        if (acc > local_best) { local_best = acc; local_arg = static_cast<int>(i); }
    }
    // Per-block tree winner (RMW isolation: shared staging, one writer).
    __shared__ float s_best[256];
    __shared__ int s_idx[256];
    s_best[threadIdx.x] = local_best;
    s_idx[threadIdx.x] = local_arg;
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = s_best[0];
        int a = s_idx[0];
        for (int j = 1; j < blockDim.x && j < 256; ++j)
            if (s_best[j] > b ||
                (s_best[j] == b && s_idx[j] < a))
                { b = s_best[j]; a = s_idx[j]; }
        blk_best[vocab + blockIdx.x] = b;
        blk_idx[vocab + blockIdx.x] = a;
    }
}

// NVFP4 tied-table rows variant: each vocab row dot decodes E2M1/E4M3
// chunk-wise (same contract as GemvDotRow: 32-element chunks, scale blocks
// sb = 2c / 2c+1, ascending-d accumulation); softcap + winner plumbing
// identical to the BF16 kernel.
__global__ void gemma_logits_rows_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float gs, const __nv_bfloat16 *__restrict__ x,
    int vocab, int hidden, float cap,
    float *__restrict__ d_logits,
    float *__restrict__ blk_best, int *__restrict__ blk_idx)
{
    float local_best = -1e30f;
    int local_arg = -1;
    const int64_t span =
        (static_cast<int64_t>(gridDim.x)) * blockDim.x;
    const int chunks = hidden >> 5;
    const __nv_bfloat16 *xf = x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < vocab; i += span) {
        const uint8_t *row_p = packed + i * (hidden >> 1);
        const uint8_t *row_s = scales + i * (hidden >> 4);
        float acc = 0.f;
        for (int c = 0; c < chunks; ++c) {
            const uint8_t *pb = row_p + c * 16;
            const float sc0 = gpu::nvfp4::E4M3ScaleToFloat(row_s[2 * c]);
            const float sc1 = gpu::nvfp4::E4M3ScaleToFloat(row_s[2 * c + 1]);
            const int xb = c * 32;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                const float scl = (b < 8) ? sc0 : sc1;
                acc += __bfloat162float(xf[xb + 2 * b]) *
                       gpu::nvfp4::E2M1Value(pb[b] & 0xF) * scl;
                acc += __bfloat162float(xf[xb + 2 * b + 1]) *
                       gpu::nvfp4::E2M1Value(pb[b] >> 4) * scl;
            }
        }
        if (cap > 0.f) acc = tanhf(acc / cap) * cap;
        if (d_logits) d_logits[i] = acc;
        blk_best[i] = acc;
        blk_idx[i] = static_cast<int>(i);
        if (acc > local_best) { local_best = acc; local_arg = static_cast<int>(i); }
    }
    __shared__ float s_best[256];
    __shared__ int s_idx[256];
    s_best[threadIdx.x] = local_best;
    s_idx[threadIdx.x] = local_arg;
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = s_best[0];
        int a = s_idx[0];
        for (int j = 1; j < blockDim.x && j < 256; ++j)
            if (s_best[j] > b ||
                (s_best[j] == b && s_idx[j] < a))
                { b = s_best[j]; a = s_idx[j]; }
        blk_best[vocab + blockIdx.x] = b;
        blk_idx[vocab + blockIdx.x] = a;
    }
}

__global__ void gemma_argmax_reduce_kernel(const float *__restrict__ blk_best,
                                           const int *__restrict__ blk_idx,
                                           int vocab, int grid,
                                           int32_t *__restrict__ d_argmax,
                                           float *__restrict__ d_maxlogit)
{
    __shared__ float s_best[256];
    __shared__ int s_idx[256];
    float b = -1e30f;
    int a = -1;
    for (int g = threadIdx.x; g < grid; g += blockDim.x) {
        const float v = blk_best[vocab + g];
        const int idx = blk_idx[vocab + g];
        if (v > b || (v == b && idx < a)) { b = v; a = idx; }
    }
    s_best[threadIdx.x] = b;
    s_idx[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        float bb = s_best[0];
        int aa = s_idx[0];
        for (int i = 1; i < blockDim.x && i < 256; ++i)
            if (s_best[i] > bb ||
                (s_best[i] == bb && s_idx[i] < aa))
                { bb = s_best[i]; aa = s_idx[i]; }
        *d_argmax = aa;
        if (d_maxlogit) *d_maxlogit = bb;
    }
}

cudaError_t RunGemmaLogitsArgmax(cudaStream_t stream,
                                 const __nv_bfloat16 *d_embed,
                                        const __nv_bfloat16 *d_x, int vocab,
                                        int hidden, float cap,
                                        int32_t *d_argmax, float *d_maxlogit,
                                        float *d_logits,
                                        float *d_blkbest, int *d_blkidx)
{
    if (!d_embed || !d_x || !d_argmax || !d_blkbest || !d_blkidx ||
        !d_logits || vocab <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    unsigned grid = static_cast<unsigned>(
        (static_cast<int64_t>(vocab) + kBlock - 1) / kBlock);
    if (grid > kLogitsMaxBlocks) grid = kLogitsMaxBlocks;
    gemma_logits_rows_kernel<<<grid, kBlock, 0, stream>>>(
        d_embed, d_x, vocab, hidden, cap, d_logits, d_blkbest, d_blkidx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    gemma_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        d_blkbest, d_blkidx, vocab, static_cast<int>(grid), d_argmax,
        d_maxlogit);
    return cudaGetLastError();
}

cudaError_t RunGemmaLogitsArgmaxNvfp4(
    cudaStream_t stream, const uint8_t *packed, const uint8_t *scales,
    float gs, const __nv_bfloat16 *d_x, int vocab, int hidden, float cap,
    int32_t *d_argmax, float *d_maxlogit, float *d_logits, float *d_blkbest,
    int *d_blkidx)
{
    if (!packed || !scales || !d_x || !d_argmax || !d_blkbest || !d_blkidx ||
        !d_logits || vocab <= 0 || hidden <= 0)
        return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    unsigned grid = static_cast<unsigned>(
        (static_cast<int64_t>(vocab) + kBlock - 1) / kBlock);
    if (grid > kLogitsMaxBlocks) grid = kLogitsMaxBlocks;
    gemma_logits_rows_nvfp4_kernel<<<grid, kBlock, 0, stream>>>(
        packed, scales, gs, d_x, vocab, hidden, cap, d_logits, d_blkbest,
        d_blkidx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    gemma_argmax_reduce_kernel<<<1, 256, 0, stream>>>(
        d_blkbest, d_blkidx, vocab, static_cast<int>(grid), d_argmax,
        d_maxlogit);
    return cudaGetLastError();
}

// ── Generation loop ─────────────────────────────────────────────────────────

// Public generation-context lifecycle (declared in gemma_moe_inference.h with
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
        (e = A((void **)&ws.d_bw1, (size_t)K * MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_bw3, (size_t)K * MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_bw2, (size_t)K * MI * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_beg, (size_t)K * MI * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_beu, (size_t)K * MI * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_beo, (size_t)K * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_bselw, (size_t)K * sizeof(float))) !=
            cudaSuccess ||
        (e = A((void **)&ws.d_lg_best,
               (static_cast<size_t>(V) + 4096) * sizeof(float))) != cudaSuccess ||
        (e = A((void **)&ws.d_lg_idx,
               (static_cast<size_t>(V) + 4096) * sizeof(int32_t))) !=
            cudaSuccess ||
        (e = A((void **)&ws.d_xh, S * H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_mlp_g, I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_mlp_u, I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_mlp_h, I_max * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_mlp_y, H * 2)) != cudaSuccess ||
        (e = A((void **)&ws.d_tmpf, S * q_rows_max * 2)) != cudaSuccess ||
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

// Copies one expert's pack (w1|w2|w3) and scales (s1|s2|s3) blocks into a
// device cache slot. Shared by the hot-path miss staging and the load-time
// prewarm. Stream-ordered; host mmaps never evict (max_open=7000 > pool), so
// no munmap barrier is required. When @p copy_stream is null the caller's
// @p compute_stream receives the uploads inline.
cudaError_t UploadGemmaExpertIntoSlot(cudaStream_t compute_stream,
                                      cudaStream_t copy_stream,
                                      uint8_t *slot_pack, uint8_t *slot_scale,
                                      const gemma::GemmaExpertEntry &ent)
{
    cudaStream_t s = copy_stream ? copy_stream : compute_stream;
    uint8_t *pk = slot_pack;
    uint8_t *sc = slot_scale;
    cudaError_t e = cudaMemcpyAsync(pk, ent.host_w1,
                                    static_cast<size_t>(ent.w1_bytes),
                                    cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    pk += ent.w1_bytes;
    e = cudaMemcpyAsync(pk, ent.host_w2, static_cast<size_t>(ent.w2_bytes),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    pk += ent.w2_bytes;
    e = cudaMemcpyAsync(pk, ent.host_w3, static_cast<size_t>(ent.w3_bytes),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(sc, ent.host_w1_scales,
                        static_cast<size_t>(ent.w1_scales_bytes),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    sc += ent.w1_scales_bytes;
    e = cudaMemcpyAsync(sc, ent.host_w2_scales,
                        static_cast<size_t>(ent.w2_scales_bytes),
                        cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return e;
    sc += ent.w2_scales_bytes;
    e = cudaMemcpyAsync(sc, ent.host_w3_scales,
                        static_cast<size_t>(ent.w3_scales_bytes),
                        cudaMemcpyHostToDevice, s);
    return e;
}

// LRU device slot cache for routed experts: [slots] × (pack|scales), one
// slot per flat expert id. Strides derive from the uniform expert geometry
// (+16B scale slack, matching the ws.d_s* scratch sizing).
//
// Residency policy (qwen 17348a1 analogy): target EVERY expert resident,
// bounded by free VRAM with a 6 GiB headroom reserve (ComputeGemmaResident-
// Slots); explicit LSX_GEMMA_EXPERT_CACHE_SLOTS overrides. The allocation
// shrinks by halves on failure instead of failing the lifecycle. On success
// the first min(slots, total) experts are PREWARMED onto the cache so steady
// state starts fully warm instead of paying cold misses during early tokens.
cudaError_t EnsureGemmaExpertCache(GemmaGenContext &ctx,
                                   const GemmaPinnedWeights &pw,
                                   const GemmaExpertDb &db,
                                   cudaStream_t stream)
{
    if (ctx.ec.slots > 0) return cudaSuccess;
    const int64_t MIH = static_cast<int64_t>(pw.top.moe_intermediate_size) *
                        pw.top.hidden_size;
    // Slot layout is w1|w2|w3 packed and s1|s2|s3 scales, so both strides
    // must hold ALL THREE blocks (mirrors the qwen-moe cache sizing; the
    // old single-block scale stride overflowed on every expert upload).
    const int64_t pack = MIH / 2 * 3;
    const int64_t scale = MIH / 16 * 3 + 16;
    const int total = db.num_layers * db.experts_per_layer;

    int explicit_slots = -1;
    if (const char *env = std::getenv("LSX_GEMMA_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env);
        if (v > 0) explicit_slots = v;
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes = 0;
        total_bytes = 0;
    }
    int slots = gemma::ComputeGemmaResidentSlots(
        free_bytes, static_cast<size_t>(pack + scale), total, explicit_slots);
    // Graceful LRU floor when nothing fits (qwen parity).
    if (slots <= 0) slots = 4;
    log::info("[gemma-infer] expert cache: free=" +
              std::to_string(free_bytes / (1024 * 1024)) + "MiB per_expert=" +
              std::to_string(static_cast<size_t>(pack + scale) / (1024 * 1024)) +
              "MiB total=" + std::to_string(total) + " slots=" +
              std::to_string(slots));

    ctx.pack_stride = pack;
    ctx.scale_stride = scale;
    ctx.ec.pack_stride = pack;
    ctx.ec.scale_stride = scale;
    ctx.ec.total_experts = total;

    // Shrink-retry: an optimistic estimate degrades to a smaller LRU instead
    // of failing the lifecycle.
    void *base = nullptr;
    cudaError_t e = cudaSuccess;
    while (slots >= 4) {
        ctx.ec.slots = slots;
        base = nullptr;
        e = cudaMalloc(&base, ctx.ec.DeviceBytes());
        if (e == cudaSuccess && base && ctx.ec.DeviceBytes() != 0) break;
        if (base) cudaFree(base);
        base = nullptr;
        slots /= 2;
    }
    if (!base || slots < 4) return e != cudaSuccess ? e : cudaErrorMemoryAllocation;
    ctx.ec.d_base = static_cast<uint8_t *>(base);
    ctx.ec.d_pack = ctx.ec.d_base;
    ctx.ec.d_scale = ctx.ec.d_base +
                     static_cast<int64_t>(ctx.ec.slots) * ctx.pack_stride;
    ctx.ec.h_slot_of = new int32_t[static_cast<size_t>(ctx.ec.total_experts)];
    ctx.ec.h_slot_owner = new int32_t[static_cast<size_t>(ctx.ec.slots)];
    ctx.ec.h_lru = new int32_t[static_cast<size_t>(ctx.ec.slots)];
    for (int i = 0; i < ctx.ec.total_experts; ++i) ctx.ec.h_slot_of[i] = -1;
    for (int i = 0; i < ctx.ec.slots; ++i) {
        ctx.ec.h_slot_owner[i] = -1;
        ctx.ec.h_lru[i] = i;
    }

    // Router-handoff fence: host busy-polls this event per layer instead of a
    // full cudaStreamSynchronize (pure-GPU-idle wakeup latency, qwen parity).
    if (!ctx.ev_router &&
        cudaEventCreateWithFlags(&ctx.ev_router,
                                 cudaEventDisableTiming) != cudaSuccess)
        return cudaErrorInvalidValue;

    // Prewarm: fill every affordable slot with its expert, MRU-first in flat
    // id order (matches ExpertStore warm ordering). Enqueued on the caller's
    // generation stream — all later Infer work is ordered behind it.
    const int warmed = std::min(ctx.ec.slots, total);
    for (int flat = 0; flat < warmed; ++flat) {
        const int layer = flat / db.experts_per_layer;
        const int expert = flat % db.experts_per_layer;
        const auto &ent =
            db.entries[static_cast<size_t>(flat)];
        if (!ent.host_w1 || !ent.host_w2) continue;
        const int slot = ctx.ec.Insert(flat);
        (void)layer;
        (void)expert;
        e = UploadGemmaExpertIntoSlot(
            stream, nullptr,
            ctx.ec.d_pack + static_cast<int64_t>(slot) * pack,
            ctx.ec.d_scale + static_cast<int64_t>(slot) * scale, ent);
        if (e != cudaSuccess) return e;
    }

    // No-handshake routing: upload the per-flat-expert meta mirror when the
    // cache is (or can be) fully resident — no evictions then, so dev_slot
    // values in the mirror stay valid for the whole generation.
    ctx.no_handshake = (ctx.ec.slots >= total);
    if (ctx.no_handshake) {
        std::vector<lsxcommon::qwen_moe::QwenMoeGvSlotDev> meta(
            static_cast<size_t>(total));
        for (int flat = 0; flat < total; ++flat) {
            const auto &ent = db.entries[static_cast<size_t>(flat)];
            meta[static_cast<size_t>(flat)].dev_slot = ctx.ec.h_slot_of[flat];
            meta[static_cast<size_t>(flat)].gs1 = ent.w1_global_scale;
            meta[static_cast<size_t>(flat)].gs3 = ent.w3_global_scale;
            meta[static_cast<size_t>(flat)].gs2 = ent.w2_global_scale;
            meta[static_cast<size_t>(flat)].coeff = 0.f;
        }
        e = cudaMalloc(&ctx.d_flat_meta,
                       sizeof(lsxcommon::qwen_moe::QwenMoeGvSlotDev) *
                           static_cast<size_t>(total));
        if (e != cudaSuccess) return e;
        ctx.d_flat_meta_count = total;
        e = cudaMemcpyAsync(ctx.d_flat_meta, meta.data(),
                            sizeof(meta[0]) * static_cast<size_t>(total),
                            cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
        e = cudaMalloc(&ctx.d_tbl,
                       sizeof(lsxcommon::qwen_moe::QwenMoeGvSlotDev) * 16);
        if (e != cudaSuccess) return e;
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
        if (e != cudaSuccess) { std::printf("[gemma-chk] L788: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L790: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }

    // 1) input RMSNorm.
    cudaError_t e = RunGemmaRMSNorm(ws.d_x, lw.d_input_norm, ws.d_normed, seq,
                                    static_cast<int>(H), top.rms_norm_eps,
                                    stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L797: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

    // 2) q/k/v projections. Full-attention layers tie V to K
    //    (attention_k_eq_v): no v_proj weights exist, so v := raw k proj.
    static const bool attn_bf16 =
        std::getenv("LSX_GEMMA_ATTN_BF16") != nullptr;
    const bool attn_nv =
        lw.attn_nvfp4 && !attn_bf16 &&
        (q_rows & 31) == 0 && (kv_heads * head_dim & 31) == 0 && (H & 31) == 0;
    const bool v_tied_to_k =
        attn_nv ? (lw.attn_v.d_packed == nullptr) : (lw.d_v == nullptr);
    if (attn_nv) {
        // NVFP4 weight-streaming GEMVs; outputs cast to BF16 so every
        // downstream kernel (QK-norms, RoPE, cache writes) stays unchanged.
        // Row loop: the GEMV core is single-row; decode (seq==1) is the hot
        // path, prefill reuses it per token.
        gemma_bf16_to_half_kernel<<<static_cast<unsigned>(
                                        (seq * H + 255) / 256),
                                    256, 0, stream>>>(
            ws.d_normed, ws.d_xh, static_cast<int>(seq * H));
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L802nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        for (int t = 0; t < seq; ++t) {
            const __half *xt = ws.d_xh + static_cast<int64_t>(t) * H;
            e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                stream, xt, lw.attn_q.d_packed, lw.attn_q.d_scales,
                lw.attn_q.global_scale, ws.d_mlp_g, static_cast<int>(H),
                q_rows, false);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L804nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = RunGemmaF16ToBf16(stream, ws.d_mlp_g,
                                  ws.d_qr + static_cast<int64_t>(t) * q_rows,
                                  q_rows);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L805nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                stream, xt, lw.attn_k.d_packed, lw.attn_k.d_scales,
                lw.attn_k.global_scale, ws.d_mlp_u, static_cast<int>(H),
                kv_heads * head_dim, false);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L807nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = RunGemmaF16ToBf16(stream, ws.d_mlp_u,
                                  ws.d_kr + static_cast<int64_t>(t) * kv_heads * head_dim,
                                  kv_heads * head_dim);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L808nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            if (!v_tied_to_k) {
                e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                    stream, xt, lw.attn_v.d_packed, lw.attn_v.d_scales,
                    lw.attn_v.global_scale, ws.d_mlp_h, static_cast<int>(H),
                    kv_heads * head_dim, false);
                if (e != cudaSuccess) { std::printf("[gemma-chk] L811nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
                e = RunGemmaF16ToBf16(stream, ws.d_mlp_h,
                                      ws.d_vr + static_cast<int64_t>(t) * kv_heads * head_dim,
                                      kv_heads * head_dim);
                if (e != cudaSuccess) { std::printf("[gemma-chk] L812nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            } else {
                e = cudaMemcpyAsync(ws.d_vr + static_cast<int64_t>(t) * kv_heads * head_dim,
                                    ws.d_kr + static_cast<int64_t>(t) * kv_heads * head_dim,
                                    static_cast<size_t>(kv_heads * head_dim) * 2,
                                    cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) { std::printf("[gemma-chk] L817: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }
        }
    } else {
    e = Matmul(cublas, lw.d_q, ws.d_normed, ws.d_qr, q_rows,
               static_cast<int>(H), seq, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L804: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    e = Matmul(cublas, lw.d_k, ws.d_normed, ws.d_kr, kv_heads * head_dim,
               static_cast<int>(H), seq, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L807: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    if (!v_tied_to_k) {
        e = Matmul(cublas, lw.d_v, ws.d_normed, ws.d_vr, kv_heads * head_dim,
                   static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L811: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    } else {
        // v := raw k_proj output (before k_norm) — value = v_norm(raw k).
        e = cudaMemcpyAsync(ws.d_vr, ws.d_kr,
                            static_cast<size_t>(kv_heads * head_dim) * seq * 2,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L817: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    } // attn_nv else (legacy BF16 projections)
    if (GemmaDebugSync(stream, "qkv", layer_idx, e)) return e;

    // 3) QK-norms (w-form) + v_norm (plain, with_scale=False).
    {
        dim3 grid(static_cast<unsigned>(seq), static_cast<unsigned>(heads));
        gemma_head_norm_kernel<<<grid, 128, 128 * sizeof(float), stream>>>(
            ws.d_qr, lw.d_q_norm, ws.d_qr, seq, heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L828: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        dim3 grid2(static_cast<unsigned>(seq),
                   static_cast<unsigned>(kv_heads));
        gemma_head_norm_kernel<<<grid2, 128, 128 * sizeof(float), stream>>>(
            ws.d_kr, lw.d_k_norm, ws.d_kr, seq, kv_heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L835: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        // v_norm: plain RMSNorm (with_scale=False, no weights) on values.
        gemma_head_norm_kernel<<<grid2, 128, 128 * sizeof(float), stream>>>(
            ws.d_vr, nullptr, ws.d_vr, seq, kv_heads, head_dim,
            top.rms_norm_eps);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L841: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
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
        if (e != cudaSuccess) { std::printf("[gemma-chk] L853: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunGemmaRopeCosSin(ws.d_pos, ws.d_cos, ws.d_sin, seq, rotary_dim,
                               theta, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L856: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunGemmaPartialRope(ws.d_qr, ws.d_cos, ws.d_sin, ws.d_qr, seq,
                                heads, head_dim, rotary_dim, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L859: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunGemmaPartialRope(ws.d_kr, ws.d_cos, ws.d_sin, ws.d_kr, seq,
                                kv_heads, head_dim, rotary_dim, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L862: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }

    // 5) Append k/v rows to the cache (D2D, stream-ordered). For tied-V
    // layers dv rows hold the unrotated normalized keys.
    const size_t row_bytes = static_cast<size_t>(kv_stride) * 2;
    e = cudaMemcpyAsync(dk + static_cast<int64_t>(pos_base) * kv_stride,
                        ws.d_kr, row_bytes * seq, cudaMemcpyDeviceToDevice,
                        stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L871: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    e = cudaMemcpyAsync(dv + static_cast<int64_t>(pos_base) * kv_stride,
                        ws.d_vr, row_bytes * seq, cudaMemcpyDeviceToDevice,
                        stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L875: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

    // 6) Attention over the cache (band mask for sliding layers).
    if (!GemmaSkipBranch("attn")) {
        // Decode fast-path (seq==1): split-KV flash-decode. The legacy
        // one-block-per-head kernel is memory-latency bound — its per-step
        // cost grows linearly with the attended span (measured: 21.5 t/s at
        // ctx 64 vs 4.5 t/s at ctx 1024). The chunked path keeps the
        // per-step cost flat (qwen RunQwenFullAttnKernelSplitKV parity,
        // tolerance-envelope contract class; run-to-run bit-identical).
        // Prefill (seq > 1) keeps the legacy kernel: its 2D grid over
        // (t, h) already provides the parallelism decode lacks.
        if (seq == 1) {
            e = RunGemmaAttentionSplitKV(
                ws.d_qr, dk, dv, ws.d_tmp, seq,
                static_cast<int>(pos_base + seq), pos_base, heads, kv_heads,
                head_dim, window,
                // HF Gemma4TextAttention scaling = 1.0 (q/k per-head
                // RMS-normalized; see legacy call below).
                1.0f, stream);
        } else {
            e = RunGemmaAttention(ws.d_qr, dk, dv, ws.d_tmp, seq,
                                  static_cast<int>(pos_base + seq), pos_base,
                                  heads, kv_heads, head_dim, window,
                                  1.0f, stream);
        }
        if (e != cudaSuccess) { std::printf("[gemma-chk] L887: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    } else {
        e = cudaMemcpyAsync(ws.d_tmp, ws.d_qr, sizeof(__nv_bfloat16) * seq * q_rows,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L891: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    if (GemmaDebugSync(stream, "attn", layer_idx, e)) return e;

    // 7) o_proj + sandwich post-norm + residual.
    if (attn_nv) {
        gemma_bf16_to_half_kernel<<<static_cast<unsigned>(
                                        ((seq * q_rows) + 255) / 256),
                                    256, 0, stream>>>(
            ws.d_tmp, ws.d_tmpf, static_cast<int>(seq * q_rows));
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L897nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        for (int t = 0; t < seq; ++t) {
            e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                stream, ws.d_tmpf + static_cast<int64_t>(t) * q_rows,
                lw.attn_o.d_packed, lw.attn_o.d_scales,
                lw.attn_o.global_scale, ws.d_mlp_g, q_rows,
                static_cast<int>(H), false);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L898nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = RunGemmaF16ToBf16(stream, ws.d_mlp_g,
                                  ws.d_qr + static_cast<int64_t>(t) * H, H);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L899nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        }
    } else {
    e = Matmul(cublas, lw.d_o, ws.d_tmp, ws.d_qr, static_cast<int>(H), q_rows,
               seq, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L898: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_norm, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L901: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L907: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }

    // 8) Dense MLP branch: n = rms(x, pre_ffn); d = mlp(n); d1 = rms(d, post_ffn_1).
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L913: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    {
        const int64_t I = top.intermediate_size;
        static const bool mlp_bf16 =
            std::getenv("LSX_GEMMA_MLP_BF16") != nullptr;
        if (lw.mlp_nvfp4 && !mlp_bf16 && ((I & 31) == 0) && ((H & 31) == 0)) {
            // NVFP4 weight-streaming chain (decode fast path): FP16
            // activations, one GEMV per projection + geGLU, all through the
            // shared GemvDotRow contract.
            gemma_bf16_to_half_kernel<<<static_cast<unsigned>(
                                            (seq * H + 255) / 256),
                                        256, 0, stream>>>(
                ws.d_normed, ws.d_xh, static_cast<int>(seq * H));
            e = cudaGetLastError();
            if (e != cudaSuccess) { std::printf("[gemma-chk] L916nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            // Row-by-row weight-streaming chain (seq==1 decode is the hot
            // path; prefill reuses the same cores per row for correctness).
            for (int t = 0; t < seq; ++t) {
                e = RunGemmaDenseMlpNvfp4Chain(
                    stream, lw.mlp_gate.d_packed, lw.mlp_gate.d_scales,
                    lw.mlp_gate.global_scale, lw.mlp_up.d_packed,
                    lw.mlp_up.d_scales, lw.mlp_up.global_scale,
                    lw.mlp_down.d_packed, lw.mlp_down.d_scales,
                    lw.mlp_down.global_scale, ws.d_xh + static_cast<int64_t>(t) * H,
                    ws.d_mlp_g, ws.d_mlp_u, ws.d_mlp_h,
                    ws.d_mlp_y + static_cast<int64_t>(t) * H,
                    static_cast<int>(H), static_cast<int>(I));
                if (e != cudaSuccess) { std::printf("[gemma-chk] L918nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }
            e = RunGemmaF16ToBf16(stream, ws.d_mlp_y, ws.d_qr,
                                  static_cast<int64_t>(seq) * H);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L920nv: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        } else {
        const int64_t numel = static_cast<int64_t>(seq) * I;
        e = Matmul(cublas, lw.d_mlp_gate, ws.d_normed, ws.d_gg,
                   static_cast<int>(I), static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L919: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = Matmul(cublas, lw.d_mlp_up, ws.d_normed, ws.d_gu,
                   static_cast<int>(I), static_cast<int>(H), seq, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L922: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        gemma_geglu_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                             stream>>>(ws.d_gg, ws.d_gu, ws.d_gg, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L926: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = Matmul(cublas, lw.d_mlp_down, ws.d_gg, ws.d_qr, static_cast<int>(H),
                   static_cast<int>(I), seq, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L929: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        }
    }
    // d1 = post_ffn_1(mlp_out)  -> keeps ws.d_qr
    e = RunGemmaRMSNorm(ws.d_qr, lw.d_post_ffn1, ws.d_qr, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L934: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

    // 9) Router: rn = rms_plain(x) * router_scale * H^-0.5; proj; softmax-all;
    //    top-k on probs; renorm; scale by per_expert_scale.
    const int E = top.num_experts;
    const int K = top.top_k_experts;
    const int64_t MI = top.moe_intermediate_size;
    e = RunGemmaRmsNormPlain(ws.d_x, ws.d_tmp2, seq, static_cast<int>(H),
                             top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L943: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    {
        const float root = 1.0f / std::sqrt(static_cast<float>(H));
        gemma_scale_vec_kernel<<<static_cast<unsigned>(seq), 256, 0, stream>>>(
            ws.d_tmp2, lw.d_router_scale, root, static_cast<int>(H), seq);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L949: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    e = RunGemmaRouterTopK(lw.d_router_proj, lw.d_per_expert_scale, ws.d_tmp2,
                           E, static_cast<int>(H), K, ws.d_sel, ws.d_selw,
                           seq, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L954: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    // No-handshake decode (seq==1, fully-resident cache): the GEMV chain
    // composes its slot table straight from d_sel/d_selw + the device meta
    // mirror — no host readback, no busy-poll stall.
    const bool no_hs = g_gen_ctx->no_handshake &&
                       g_gen_ctx->d_flat_meta != nullptr && seq == 1;
    if (!no_hs) {
        e = cudaMemcpyAsync(ws.h_sel, ws.d_sel, sizeof(int32_t) * seq * K,
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L957: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = cudaMemcpyAsync(ws.h_selw, ws.d_selw, sizeof(float) * seq * K,
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] L960: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        // Router handoff: busy-poll a dedicated event instead of draining the
        // whole stream — with 30 layers × steps handshakes per generation, host
        // wake latency after full sync is pure GPU idle time (qwen parity).
        cudaEventRecord(g_gen_ctx->ev_router, stream);
        {
            cudaError_t q;
            while ((q = cudaEventQuery(g_gen_ctx->ev_router)) == cudaErrorNotReady) {}
            if (q != cudaSuccess) { std::printf("[gemma-chk] L962: %s\n", cudaGetErrorString(q)); std::fflush(stdout); return q; }
        }
    }

    // MoE: h2 = rms(x, pre_ffn2); moe(h2, sel, w); d2 = rms(moe, post_ffn2).
    e = RunGemmaRMSNorm(ws.d_x, lw.d_pre_ffn2, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L967: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    e = cudaMemsetAsync(ws.d_acc, 0, sizeof(__nv_bfloat16) * seq * H, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L969: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

    gpu::ExpertDeviceCache &ec = g_gen_ctx->ec;
    static const bool inline_uploads =
        std::getenv("LSX_GEMMA_INLINE_UPLOADS") != nullptr;
    // Batched decode fast-path is the DEFAULT: gathered dequant + batched
    // GEMMs (or the direct NVFP4 GEMV chain) instead of the legacy
    // per-expert dequant->GEMV->axpy chain (~9*K device ops per layer per
    // token). LSX_GEMMA_MOE_BATCH=0 restores the legacy sequential path;
    // LSX_GEMMA_GEMV=0 disables the direct weight-streaming GEMV route
    // while keeping the batched dequant path.
    static const int moe_batch_mode = []() {
        const char *e = std::getenv("LSX_GEMMA_MOE_BATCH");
        return e ? std::atoi(e) : 1;
    }();
    static const bool kGemvRoute = []() {
        const char *e = std::getenv("LSX_GEMMA_GEMV");
        return e ? std::atoi(e) != 0 : true;
    }();
    const bool batched_path =
        (seq == 1 && K <= 16 && K >= 1 && moe_batch_mode != 0) ||
        (seq == 1 && K <= 16 && K >= 1 && kGemvRoute);
    const bool batch_gemms = moe_batch_mode != 2;

    if (no_hs) {
        // Device-composed slot table -> direct weight-streaming GEMV chain.
        // Experts are all resident, so no staging and no host participation.
        // The flat meta mirror was uploaded at cache setup; dev_slot entries
        // point at prewarmed slots. GeGLU (gelu_tanh) matches the routed
        // chain; accumulation is FP32 ascending-k, result cast to d_acc.
        gemma_gemv_gather_table_kernel<<<1, 32, 0, stream>>>(
            ws.d_sel, ws.d_selw, g_gen_ctx->d_flat_meta, K,
            g_gen_ctx->d_tbl);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] nohs1: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        const auto &ent0 =
            g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                 static_cast<size_t>(E)];
        __half *xh2 = nullptr, *ygu2 = nullptr, *hbuf2 = nullptr;
        float *accf2 = nullptr;
        e = cudaMallocAsync(&xh2, sizeof(__half) * H, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] nohs2: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = cudaMallocAsync(&ygu2, sizeof(__half) * K * 2 * MI, stream);
        if (e != cudaSuccess) { cudaFreeAsync(xh2, stream); std::printf("[gemma-chk] nohs2b: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = cudaMallocAsync(&hbuf2, sizeof(__half) * K * MI, stream);
        if (e != cudaSuccess) { cudaFreeAsync(xh2, stream); cudaFreeAsync(ygu2, stream); std::printf("[gemma-chk] nohs2c: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = cudaMallocAsync(&accf2, sizeof(float) * H, stream);
        if (e != cudaSuccess) { cudaFreeAsync(xh2, stream); cudaFreeAsync(ygu2, stream); cudaFreeAsync(hbuf2, stream); std::printf("[gemma-chk] nohs2d: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        gemma_bf16_to_half_kernel<<<(unsigned)((H + 255) / 256), 256, 0,
                                    stream>>>(ws.d_normed, xh2,
                                              static_cast<int>(H));
        gemma_gemv_zero_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
            accf2, H);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] nohs2e: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunRoutedGemvBatchSeg(
            stream, ec.d_pack, ec.d_scale, g_gen_ctx->pack_stride,
            g_gen_ctx->scale_stride, g_gen_ctx->d_tbl, K,
            /*gate*/ 0, 0,
            /*up*/ static_cast<int64_t>(ent0.w1_bytes + ent0.w2_bytes),
            static_cast<int64_t>(ent0.w1_scales_bytes +
                                 ent0.w2_scales_bytes),
            /*down*/ static_cast<int64_t>(ent0.w1_bytes),
            static_cast<int64_t>(ent0.w1_scales_bytes),
            static_cast<int>(MI), static_cast<int>(H),
            static_cast<int>(H), static_cast<int>(MI),
            /*geglu=*/true, xh2, ygu2, hbuf2, accf2);
        if (e == cudaSuccess)
            e = RunGemmaF32ToBf16(stream, accf2, ws.d_acc, H);
        cudaFreeAsync(xh2, stream);
        cudaFreeAsync(ygu2, stream);
        cudaFreeAsync(hbuf2, stream);
        cudaFreeAsync(accf2, stream);
        if (e != cudaSuccess) { std::printf("[gemma-chk] nohs3: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    } else if (batched_path) {
        // ── Batched decode fast-path: one gathered dequant + three batched
        // GEMMs + one geGLU + one combine instead of the per-expert launch
        // chain (~9*K -> ~7 device ops per layer per token). Gather rows are
        // bit-identical to sequential dequants (unit test "[gemma][moe]
        // [batch]"); combine replicates the sequential axpy rounding order.
        GemmaDequantBatch g1{}, g2{}, g3{};
        float w_compact[16];
        int gemv_devslots[16];
        int filled = 0;

        for (int k = 0; k < K; ++k) {
            const int expert = ws.h_sel[static_cast<size_t>(k)];
            const auto &ent =
                g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                         static_cast<size_t>(E) +
                                     static_cast<size_t>(expert)];
            if (!ent.host_w1 || !ent.host_w2) {
                std::printf("[gemma-chk] missing routed expert k=%d\n", k);
                std::fflush(stdout);
                return cudaErrorInvalidValue;
            }
            const int flat = layer_idx * E + expert;
            int slot = ec.Lookup(flat);
            const bool miss = slot < 0;
            if (miss) {
                slot = ec.Insert(flat);
            }
            uint8_t *slot_pack = ec.d_pack +
                static_cast<int64_t>(slot) * g_gen_ctx->pack_stride;
            uint8_t *slot_scale = ec.d_scale +
                static_cast<int64_t>(slot) * g_gen_ctx->scale_stride;
            if (miss) {
                // Miss staging runs INLINE on the generation stream. The
                // earlier cross-stream copy-channel variant raced over long
                // sessions (nondeterministic token corruption) and bought
                // little after the lm-head fix made logits/argmax no longer
                // dominant — keep the pipeline single-stream & deterministic.
                e = UploadGemmaExpertIntoSlot(stream, nullptr, slot_pack,
                                              slot_scale, ent);
                if (e != cudaSuccess) { std::printf("[gemma-chk] L1002: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }

            // Slot layout mirrors pack strides: w1|w2|w3 packed blocks,
            // s1|s2|s3 scale blocks (uniform geometry across experts).
            g1.packs[filled] = slot_pack;
            g1.scales[filled] = slot_scale;
            g1.global_scales[filled] = ent.w1_global_scale;
            g2.packs[filled] = slot_pack + ent.w1_bytes;
            g2.scales[filled] = slot_scale + ent.w1_scales_bytes;
            g2.global_scales[filled] = ent.w2_global_scale;
            g3.packs[filled] =
                slot_pack + ent.w1_bytes + ent.w2_bytes;
            g3.scales[filled] =
                slot_scale + ent.w1_scales_bytes + ent.w2_scales_bytes;
            g3.global_scales[filled] = ent.w3_global_scale;
            w_compact[filled] = ws.h_selw[static_cast<size_t>(k)];
            gemv_devslots[filled] = slot;
            ++filled;
        }

        // ── Round-21 direct streaming chain: skips the dequant-to-BF16
        // detour and streams packed NVFP4 straight through the GEMV cores;
        // identical ascending-k reduce contract.
        const bool gemv_shapes_ok =
            ((MI & 31) == 0) && ((H & 31) == 0);
        bool gemv_used = false;
        if (kGemvRoute && gemv_shapes_ok && filled == K &&
            static_cast<int>(H) % 32 == 0) {
            lsxcommon::qwen_moe::QwenMoeGvSlotDev tbl[16];
            for (int k = 0; k < filled; ++k) {
                const int expert = ws.h_sel[static_cast<size_t>(k)];
                const auto &ent =
                    g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                             static_cast<size_t>(E) +
                                         static_cast<size_t>(expert)];
                tbl[k].dev_slot = gemv_devslots[k];
                tbl[k].gs1 = ent.w1_global_scale;
                tbl[k].gs3 = ent.w3_global_scale;
                tbl[k].gs2 = ent.w2_global_scale;
                tbl[k].coeff = w_compact[k];
            }
            const auto &ent0 =
                g_expert_db->entries[static_cast<size_t>(layer_idx) *
                                     static_cast<size_t>(E)];
            lsxcommon::qwen_moe::QwenMoeGvSlotDev *d_tbl = nullptr;
            __half *xh = nullptr, *ygu = nullptr, *hbuf = nullptr;
            float *accf = nullptr;
            bool pool_ok =
                cudaMallocAsync(&d_tbl,
                                sizeof(lsxcommon::qwen_moe::QwenMoeGvSlotDev) *
                                    filled,
                                stream) == cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&xh),
                                sizeof(__half) * H, stream) ==
                    cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&ygu),
                                sizeof(__half) * filled * 2 * MI,
                                stream) == cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&hbuf),
                                sizeof(__half) * filled * MI, stream) ==
                    cudaSuccess &&
                cudaMallocAsync(&accf, sizeof(float) * H, stream) ==
                    cudaSuccess;
            if (pool_ok) {
                cudaMemcpyAsync(d_tbl, tbl, sizeof(*tbl) * filled,
                                cudaMemcpyHostToDevice, stream);
                gemma_bf16_to_half_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    ws.d_normed, xh, H);
                gemma_gemv_zero_f32_kernel<<<(H + 255) / 256, 256, 0,
                                             stream>>>(accf, H);
                e = RunRoutedGemvBatchSeg(
                    stream, ec.d_pack, ec.d_scale, g_gen_ctx->pack_stride,
                    g_gen_ctx->scale_stride, d_tbl, filled,
                    /*gate*/ 0, 0,
                    /*up*/
                    static_cast<int64_t>(ent0.w1_bytes + ent0.w2_bytes),
                    static_cast<int64_t>(ent0.w1_scales_bytes +
                                         ent0.w2_scales_bytes),
                    /*down*/ static_cast<int64_t>(ent0.w1_bytes),
                    static_cast<int64_t>(ent0.w1_scales_bytes),
                    /*N_gate_up=*/static_cast<int>(MI),
                    /*K_gate_up=*/static_cast<int>(H),
                    /*N_down=*/static_cast<int>(H),
                    /*K_down=*/static_cast<int>(MI),
                    /*geglu=*/true, xh, ygu, hbuf, accf);
                if (e == cudaSuccess)
                    gemma_f32_to_bf16_kernel<<<(H + 255) / 256, 256, 0,
                                               stream>>>(accf, ws.d_acc, H);
                cudaFreeAsync(d_tbl, stream);
                cudaFreeAsync(xh, stream);
                cudaFreeAsync(ygu, stream);
            cudaFreeAsync(hbuf, stream);
            cudaFreeAsync(accf, stream);
            static int gv_hits = 0;
                if (++gv_hits == 1 && std::getenv("LSX_GEMV_TRACE"))
                    std::fprintf(stderr,
                                 "[GEMMA-GEMV] chain engaged (layers x "
                                 "steps)\n");
                if (e != cudaSuccess) return e;
                // MoE result is in ws.d_acc: fall through to the shared
                // sandwich epilogue (post_ffn2 -> d1+d2 -> post_ffn ->
                // residual -> layer_scalar). The former early return skipped
                // the whole FFN epilogue on every layer -- root cause of the
                // GEMV route producing wrong tokens.
                gemv_used = true;
                e = cudaGetLastError();
                if (e != cudaSuccess) return e;
            }
            // Pool scratch unavailable -> legacy batched/sequential flow.
        }

        if (!gemv_used) {
        const int64_t MIH = MI * H;
        e = RunGemmaBatchedDequant(stream, g1, MIH, filled,
                                   ws.d_bw1);                      // [K][MI*H]
        if (e != cudaSuccess) { std::printf("[gemma-chk] gdq1: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunGemmaBatchedDequant(stream, g3, MIH, filled,
                                   ws.d_bw3);                      // [K][MI*H]
        if (e != cudaSuccess) { std::printf("[gemma-chk] gdq3: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        e = RunGemmaBatchedDequant(stream, g2, H * MI, filled,
                                   ws.d_bw2);                      // [K][H*MI]
        if (e != cudaSuccess) { std::printf("[gemma-chk] gdq2: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

        {
            // (Re)publish selected weights every step: tiny pinned H2D.
            e = cudaMemcpyAsync(ws.d_bselw, w_compact,
                                sizeof(float) * static_cast<size_t>(filled),
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] wcp: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        }

        // gate/up projections over all experts sharing the same token x.
        if (batch_gemms) {
            e = MatmulBatched(cublas, ws.d_bw1, MIH, ws.d_normed, 0,
                              ws.d_beg, MI, static_cast<int>(MI),
                              static_cast<int>(H), filled, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] bgem1: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = MatmulBatched(cublas, ws.d_bw3, MIH, ws.d_normed, 0,
                              ws.d_beu, MI, static_cast<int>(MI),
                              static_cast<int>(H), filled, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] bgem3: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        } else {
            for (int r = 0; r < filled; ++r) {
                e = Matmul(cublas, ws.d_bw1 + static_cast<int64_t>(r) * MIH,
                           ws.d_normed,
                           ws.d_beg + static_cast<int64_t>(r) * MI,
                           static_cast<int>(MI), static_cast<int>(H), 1,
                           stream);
                if (e != cudaSuccess) { std::printf("[gemma-chk] sgem1: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
                e = Matmul(cublas, ws.d_bw3 + static_cast<int64_t>(r) * MIH,
                           ws.d_normed,
                           ws.d_beu + static_cast<int64_t>(r) * MI,
                           static_cast<int>(MI), static_cast<int>(H), 1,
                           stream);
                if (e != cudaSuccess) { std::printf("[gemma-chk] sgem3: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }
        }
        {
            const int64_t numel = static_cast<int64_t>(filled) * MI;
            gemma_geglu_batched_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256,
                                         0, stream>>>(ws.d_beg, ws.d_beu,
                                                      ws.d_beg, numel);
            e = cudaGetLastError();
            if (e != cudaSuccess) { std::printf("[gemma-chk] bgeglu: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        }
        // Down projection: [K][H] outputs.
        if (batch_gemms) {
            e = MatmulBatched(cublas, ws.d_bw2, H * MI, ws.d_beg, MI,
                              ws.d_beo, H, static_cast<int>(H),
                              static_cast<int>(MI), filled, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] bgem2: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        } else {
            for (int r = 0; r < filled; ++r) {
                e = Matmul(cublas, ws.d_bw2 + static_cast<int64_t>(r) * MI * H,
                           ws.d_beg + static_cast<int64_t>(r) * MI,
                           ws.d_beo + static_cast<int64_t>(r) * H,
                           static_cast<int>(H), static_cast<int>(MI), 1,
                           stream);
                if (e != cudaSuccess) { std::printf("[gemma-chk] sgem2: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }
        }
        // Weighted accumulate in ascending-k order (axpy parity).
        gemma_combine_batched_kernel<<<static_cast<unsigned>((H + 255) / 256),
                                       256, 0, stream>>>(
            ws.d_acc, ws.d_beo, ws.d_bselw, filled, H);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] bcomb: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
        }
    } else {
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

            const int flat = layer_idx * E + expert;
            int slot = ec.Lookup(flat);
            const bool miss = slot < 0;
            if (miss) slot = ec.Insert(flat);
            uint8_t *slot_pack = ec.d_pack +
                static_cast<int64_t>(slot) * g_gen_ctx->pack_stride;
            uint8_t *slot_scale = ec.d_scale +
                static_cast<int64_t>(slot) * g_gen_ctx->scale_stride;
            if (miss) {
                // Inline staging on the generation stream (deterministic;
                // see the legacy-path note above re: copy-channel removal).
                e = UploadGemmaExpertIntoSlot(stream, nullptr, slot_pack,
                                              slot_scale, ent);
                if (e != cudaSuccess) { std::printf("[gemma-chk] L1002: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            }

            e = DequantNvfp4(stream, slot_pack, ent.w1_bytes, slot_scale,
                             ent.w1_scales_bytes, ent.w1_global_scale,
                             ws.d_w1, MI * H);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1032: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = DequantNvfp4(stream, slot_pack + ent.w1_bytes, ent.w2_bytes,
                             slot_scale + ent.w1_scales_bytes,
                             ent.w2_scales_bytes, ent.w2_global_scale,
                             ws.d_w2, H * MI);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1037: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = DequantNvfp4(stream,
                             slot_pack + ent.w1_bytes + ent.w2_bytes,
                             ent.w3_bytes,
                             slot_scale + ent.w1_scales_bytes +
                                 ent.w2_scales_bytes,
                             ent.w3_scales_bytes, ent.w3_global_scale,
                             ws.d_w3, MI * H);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1045: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

            // Expert FFN: out_e[t] = w2 @ (gelu(w1 @ x) * (w3 @ x)) —
            // cuBLAS GEMMs + elementwise GeGLU.
            const __nv_bfloat16 *x_t =
                ws.d_normed + static_cast<int64_t>(t) * H;
            e = Matmul(cublas, ws.d_w1, x_t, ws.d_eg, static_cast<int>(MI),
                       static_cast<int>(H), 1, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1053: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = Matmul(cublas, ws.d_w3, x_t, ws.d_eu, static_cast<int>(MI),
                       static_cast<int>(H), 1, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1056: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            gemma_geglu_kernel<<<static_cast<unsigned>((MI + 255) / 256), 256,
                                 0, stream>>>(ws.d_eg, ws.d_eu, ws.d_eg, MI);
            e = cudaGetLastError();
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1060: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            e = Matmul(cublas, ws.d_w2, ws.d_eg, ws.d_eout, static_cast<int>(H),
                       static_cast<int>(MI), 1, stream);
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1063: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }

            gemma_axpy_kernel<<<static_cast<unsigned>((H + 255) / 256), 256, 0,
                                stream>>>(
                ws.d_acc + static_cast<int64_t>(t) * H, ws.d_eout, weight, H);
            e = cudaGetLastError();
            if (e != cudaSuccess) { std::printf("[gemma-chk] L1069: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
            if (GemmaDebugSync(stream, "moe-expert", layer_idx, e)) return e;
        }
    }
    } // batched_path

    // d2 = post_ffn2(moe_out); combine ff = d1 + d2; ff = post_ffn(ff);
    // hidden += ff; hidden *= layer_scalar.
    e = RunGemmaRMSNorm(ws.d_acc, lw.d_post_ffn2, ws.d_tmp2, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L1078: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_tmp2, ws.d_qr, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L1084: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    e = RunGemmaRMSNorm(ws.d_tmp2, lw.d_post_ffn, ws.d_normed, seq,
                        static_cast<int>(H), top.rms_norm_eps, stream);
    if (e != cudaSuccess) { std::printf("[gemma-chk] L1088: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    {
        const int64_t numel = static_cast<int64_t>(seq) * H;
        gemma_axpy_kernel<<<static_cast<unsigned>((numel + 255) / 256), 256, 0,
                            stream>>>(ws.d_x, ws.d_normed, 1.0f, numel);
        e = cudaGetLastError();
        if (e != cudaSuccess) { std::printf("[gemma-chk] L1094: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
    }
    gemma_scale_kernel<<<static_cast<unsigned>((seq * H + 255) / 256), 256, 0,
                         stream>>>(ws.d_x, layer_scalar,
                                   static_cast<int64_t>(seq) * H);
    e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("[gemma-chk] L1100: %s\n", cudaGetErrorString(e)); std::fflush(stdout); return e; }
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

    e = EnsureGemmaExpertCache(c, pw, db, stream);
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
        const bool head_nv = pw.embed_nvfp4.d_packed != nullptr;
        if (head_nv)
            gemma_embed_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                pw.embed_nvfp4.d_packed, pw.embed_nvfp4.d_scales,
                pw.embed_nvfp4.global_scale, ws.d_pos, ws.d_x,
                static_cast<int>(H));
        else
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
        if (pw.embed_nvfp4.d_packed != nullptr)
            return RunGemmaLogitsArgmaxNvfp4(
                stream, pw.head_nvfp4.d_packed, pw.head_nvfp4.d_scales,
                pw.head_nvfp4.global_scale, ws.d_normed, static_cast<int>(V),
                static_cast<int>(H), softcap, ws.d_argmax, nullptr,
                ws.d_logits, ws.d_lg_best, ws.d_lg_idx);
        return RunGemmaLogitsArgmax(stream, pw.d_embed, ws.d_normed,
                                    static_cast<int>(V), static_cast<int>(H),
                                    softcap, ws.d_argmax, nullptr,
                                    ws.d_logits, ws.d_lg_best, ws.d_lg_idx);
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
        if (pw.embed_nvfp4.d_packed != nullptr)
            gemma_embed_nvfp4_kernel<<<1, 256, 0, stream>>>(
                pw.embed_nvfp4.d_packed, pw.embed_nvfp4.d_scales,
                pw.embed_nvfp4.global_scale, ws.d_pos, ws.d_x,
                static_cast<int>(H));
        else
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
        if (pw.embed_nvfp4.d_packed != nullptr)
            e = RunGemmaLogitsArgmaxNvfp4(
                stream, pw.head_nvfp4.d_packed, pw.head_nvfp4.d_scales,
                pw.head_nvfp4.global_scale, ws.d_normed, static_cast<int>(V),
                static_cast<int>(H), softcap, ws.d_argmax, nullptr,
                ws.d_logits, ws.d_lg_best, ws.d_lg_idx);
        else
            e = RunGemmaLogitsArgmax(stream, pw.d_embed, ws.d_normed,
                                     static_cast<int>(V), static_cast<int>(H),
                                     softcap, ws.d_argmax, nullptr,
                                     ws.d_logits, ws.d_lg_best,
                                     ws.d_lg_idx);
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

#include "nvfp4_gemv_math.cuh"

// ── Round 21: batched weight-streaming GEMV chain for gemma routed experts ──
// Same warp-slice streaming contract as the qwen_moe batched chain
// (nvfp4_gemv_math.cuh is the shared dot-math source of truth). The staged
// expert slot keeps three SEPARATED slabs (w1 gate | w2 down | w3 up packed,
// s1|s2|s3 scales) so the pair stage takes independent base offsets for
// gate and up. Activation applies geGLU (gelu_tanh(g)*u, bit-matched to the
// BF16 pipeline kernel above) in FP16 buffers; the weighted reduce adds into
// an FP32 accumulator ascending-k. Down partials use a transient pool
// allocation per call (k_slots*H*2 bytes, ~32 KB at gemma geometry).

namespace lsxcommon {

namespace nvfp4gemv {

constexpr int kGemvWarps3 = 8;

inline int PickKpSeg(int cols, int rows)
{
    for (int cand : {8, 4, 2}) {
        if (cols / cand >= 1024 &&
            static_cast<int64_t>(rows) * cand >= 1024)
            return cand;
    }
    return 1;
}

template <int KP>
__global__ void gemma_gemv_pair_batch_kernel(
    const __half *__restrict__ x_row,
    const uint8_t *__restrict__ pack_base,
    const uint8_t *__restrict__ scale_base,
    int64_t pack_stride, int64_t scale_stride,
    int64_t off_gate_p, int64_t off_gate_s,
    int64_t off_up_p, int64_t off_up_s,
    const qwen_moe::QwenMoeGvSlotDev *__restrict__ slots,
    int k_slots, int n_rows, int cols, __half *__restrict__ ygu)
{
    constexpr int ROWS = kGemvWarps3 / KP;
    __shared__ float sp[kGemvWarps3];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rflat = blockIdx.x * ROWS + (warp / KP);
    const int total = k_slots * 2 * n_rows;
    float acc = 0.0f;
    if (rflat < total) {
        const int s = rflat / (2 * n_rows);
        const int r = rflat - s * 2 * n_rows;
        const bool gate = r < n_rows;
        const qwen_moe::QwenMoeGvSlotDev slot = slots[s];
        const int rr = gate ? r : r - n_rows;
        acc = GemvDotRow(
                  x_row,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      (gate ? off_gate_p : off_up_p) +
                      static_cast<int64_t>(rr) * (cols >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      (gate ? off_gate_s : off_up_s) +
                      static_cast<int64_t>(rr) * (cols >> 4),
                  (KP > 1 ? (cols / KP) * (warp % KP) : 0),
                  (cols / KP) >> 5,
                  lane) *
              (gate ? slot.gs1 : slot.gs3);
    }
    if (lane == 0) sp[warp] = acc;
    __syncthreads();
    if (lane == 0 && (warp % KP) == 0 && rflat < total) {
        const int wbase = (warp / KP) * KP;
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KP; ++j) t += sp[wbase + j];
        ygu[rflat] = __float2half(t);
    }
}

template <int KP>
__global__ void gemma_gemv_down_batch_kernel(
    const __half *__restrict__ hbuf,
    const uint8_t *__restrict__ pack_base,
    const uint8_t *__restrict__ scale_base,
    int64_t pack_stride, int64_t scale_stride,
    int64_t off_down_p, int64_t off_down_s,
    const qwen_moe::QwenMoeGvSlotDev *__restrict__ slots,
    int k_slots, int n_rows, int cols, __half *__restrict__ ypart)
{
    constexpr int ROWS = kGemvWarps3 / KP;
    __shared__ float sp[kGemvWarps3];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rflat = blockIdx.x * ROWS + (warp / KP);
    const int total = k_slots * n_rows;
    float acc = 0.0f;
    if (rflat < total) {
        const int s = rflat / n_rows;
        const int r = rflat - s * n_rows;
        const qwen_moe::QwenMoeGvSlotDev slot = slots[s];
        acc = GemvDotRow(
                  hbuf + static_cast<int64_t>(s) * cols,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      off_down_p + static_cast<int64_t>(r) * (cols >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      off_down_s + static_cast<int64_t>(r) * (cols >> 4),
                  (KP > 1 ? (cols / KP) * (warp % KP) : 0),
                  (cols / KP) >> 5,
                  lane) *
              slot.gs2;
    }
    if (lane == 0) sp[warp] = acc;
    __syncthreads();
    if (lane == 0 && (warp % KP) == 0 && rflat < total) {
        const int wbase = (warp / KP) * KP;
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KP; ++j) t += sp[wbase + j];
        ypart[rflat] = __float2half(t);
    }
}

template <bool GEGLU>
__global__ void gemma_gemv_act_batch_kernel(const __half *__restrict__ ygu,
                                            int n_half, int N,
                                            __half *__restrict__ hbuf)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;
    const int s = i / N;
    const int j = i - s * N;
    const float g =
        __half2float(ygu[static_cast<int64_t>(s) * 2 * N + j]);
    const float u =
        __half2float(ygu[static_cast<int64_t>(s) * 2 * N + N + j]);
    hbuf[i] = __float2half(GEGLU ? lsxcommon::gemma::gelu_tanh_dev(g) * u
                                 : (g / (1.0f + expf(-g))) * u);
}

__global__ void gemma_gemv_reduce_batch_kernel(
    const __half *__restrict__ ypart,
    const qwen_moe::QwenMoeGvSlotDev *__restrict__ slots, int k_slots,
    int n_elems, float *__restrict__ acc_f32)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    float t = acc_f32[i];
    for (int s = 0; s < k_slots; ++s)
        t += slots[s].coeff *
             __half2float(ypart[static_cast<int64_t>(s) * n_elems + i]);
    acc_f32[i] = t;
}
} // namespace nvfp4gemv

namespace gemma {

__global__ void gemma_bf16_to_half_kernel(
    const __nv_bfloat16 *__restrict__ src, __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__bfloat162float(src[i]));
}

__global__ void gemma_f32_to_bf16_kernel(const float *__restrict__ src,
                                         __nv_bfloat16 *__restrict__ dst,
                                         int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

__global__ void gemma_gemv_zero_f32_kernel(float *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = 0.0f;
}

cudaError_t RunRoutedGemvBatchSeg(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *slots_dev, int k_slots,
    int64_t off_gate_p, int64_t off_gate_s,
    int64_t off_up_p, int64_t off_up_s,
    int64_t off_down_p, int64_t off_down_s,
    int n_gate_up, int k_gate_up,
    int n_down, int k_down,
    bool geglu,
    const __half *d_x,
    __half *ygu, __half *hbuf, float *acc_f32)
{
    using namespace lsxcommon::nvfp4gemv;
    constexpr int WARPS = kGemvWarps3;
    if (!d_pack_base || !d_scale_base || !slots_dev || !d_x || !ygu ||
        !hbuf || !acc_f32 || k_slots <= 0 || k_slots > 16 ||
        n_gate_up <= 0 || k_gate_up <= 0 || n_down <= 0 || k_down <= 0 ||
        (k_gate_up & 31) != 0 || (k_down & 31) != 0)
        return cudaErrorInvalidValue;

    // Pair (gate | up).
    cudaError_t e;
    {
        const int rows = k_slots * 2 * n_gate_up;
        const int kp = PickKpSeg(k_gate_up, rows);
        auto go = [&](auto tag) -> cudaError_t {
            constexpr int KPc = decltype(tag)::value;
            const int ROWSp = WARPS / KPc;
            gemma_gemv_pair_batch_kernel<KPc>
                <<<(rows + ROWSp - 1) / ROWSp, WARPS * 32, 0, stream>>>(
                    d_x, d_pack_base, d_scale_base, pack_stride,
                    scale_stride, off_gate_p, off_gate_s, off_up_p,
                    off_up_s, slots_dev, k_slots, n_gate_up, k_gate_up,
                    ygu);
            return cudaGetLastError();
        };
        switch (kp) {
        case 8:  e = go(std::integral_constant<int, 8>{}); break;
        case 4:  e = go(std::integral_constant<int, 4>{}); break;
        case 2:  e = go(std::integral_constant<int, 2>{}); break;
        default: e = go(std::integral_constant<int, 1>{}); break;
        }
        if (e != cudaSuccess) return e;
    }

    // Activation.
    {
        const int n_act = k_slots * n_gate_up;
        auto go = [&](auto tag) -> cudaError_t {
            constexpr bool G = decltype(tag)::value;
            gemma_gemv_act_batch_kernel<G>
                <<<(n_act + 255) / 256, 256, 0, stream>>>(
                    ygu, n_act, n_gate_up, hbuf);
            return cudaGetLastError();
        };
        e = geglu ? go(std::bool_constant<true>{})
                  : go(std::bool_constant<false>{});
        if (e != cudaSuccess) return e;
    }

    // Down -> per-slot partials -> weighted reduce into fp32 acc.
    __half *ypart = nullptr;
    e = cudaMallocAsync(&ypart,
                        static_cast<size_t>(k_slots) * n_down * sizeof(__half),
                        stream);
    if (e != cudaSuccess) return e;
    {
        const int rows = k_slots * n_down;
        const int kp = PickKpSeg(k_down, rows);
        auto go = [&](auto tag) -> cudaError_t {
            constexpr int KPc = decltype(tag)::value;
            const int ROWSp = WARPS / KPc;
            gemma_gemv_down_batch_kernel<KPc>
                <<<(rows + ROWSp - 1) / ROWSp, WARPS * 32, 0, stream>>>(
                    hbuf, d_pack_base, d_scale_base, pack_stride,
                    scale_stride, off_down_p, off_down_s, slots_dev,
                    k_slots, n_down, k_down, ypart);
            return cudaGetLastError();
        };
        switch (kp) {
        case 8:  e = go(std::integral_constant<int, 8>{}); break;
        case 4:  e = go(std::integral_constant<int, 4>{}); break;
        case 2:  e = go(std::integral_constant<int, 2>{}); break;
        default: e = go(std::integral_constant<int, 1>{}); break;
        }
        if (e == cudaSuccess) {
            gemma_gemv_reduce_batch_kernel<<<(n_down + 255) / 256, 256, 0,
                                             stream>>>(ypart, slots_dev,
                                                       k_slots, n_down,
                                                       acc_f32);
            e = cudaGetLastError();
        }
    }
    cudaFreeAsync(ypart, stream);
    return e == cudaSuccess ? cudaGetLastError() : e;
}

// No-handshake gather: composes the routed-GEMV slot table from the device
// routing results (d_sel/d_selw) and the per-flat-expert meta mirror. Fully
// on device — no host readback, no LRU-lookup stall.
__global__ void gemma_gemv_gather_table_kernel(
    const int32_t *__restrict__ d_sel, const float *__restrict__ d_selw,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ flat_meta,
    int K, lsxcommon::qwen_moe::QwenMoeGvSlotDev *__restrict__ tbl)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int expert = d_sel[k];
    lsxcommon::qwen_moe::QwenMoeGvSlotDev v = flat_meta[expert];
    v.coeff = d_selw[k];
    tbl[k] = v;
}

// ── Dense-MLP NVFP4 weight-streaming chain (Task 3) ────────────────────────
// y = down @ (gelu_tanh(gate @ x) * (up @ x)). Reuses the shared GemvDotRow
// decode contract and the geGLU act kernel from the routed chain; single-
// row GEMVs via RunLsxNvfp4GemmSeq1GemmV (K % 32 == 0 enforced upstream by
// the converter shape check).
__global__ void gemma_f16_to_bf16_kernel(const __half *__restrict__ src,
                                         __nv_bfloat16 *__restrict__ dst,
                                         int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__half2float(src[i]));
}

// Dense-MLP geGLU over separate gate/up buffers (unlike the routed chain's
// interleaved [2I] ygu layout): h[i] = gelu_tanh(g[i]) * u[i].
__global__ void gemma_dense_mlp_act_kernel(const __half *__restrict__ g,
                                           const __half *__restrict__ u,
                                           __half *__restrict__ h, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = __half2float(g[i]);
    h[i] = __float2half(gelu_tanh_dev(gv) * __half2float(u[i]));
}

cudaError_t RunGemmaF16ToBf16(cudaStream_t stream, const __half *src,
                              __nv_bfloat16 *dst, int64_t n)
{
    if (!src || !dst || n <= 0) return cudaErrorInvalidValue;
    gemma_f16_to_bf16_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                               stream>>>(src, dst, n);
    return cudaGetLastError();
}

cudaError_t RunGemmaF32ToBf16(cudaStream_t stream, const float *src,
                              __nv_bfloat16 *dst, int64_t n)
{
    if (!src || !dst || n <= 0) return cudaErrorInvalidValue;
    gemma_f32_to_bf16_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                               stream>>>(src, dst, n);
    return cudaGetLastError();
}

} // namespace gemma
} // namespace lsxcommon

namespace lsxcommon {
namespace gemma {

cudaError_t RunGemmaDenseMlpNvfp4Chain(
    cudaStream_t stream, const uint8_t *gate_packed, const uint8_t *gate_scales,
    float gate_gs, const uint8_t *up_packed, const uint8_t *up_scales,
    float up_gs, const uint8_t *down_packed, const uint8_t *down_scales,
    float down_gs, const __half *x, __half *g_out, __half *u_out,
    __half *h_out, __half *y, int H, int I)
{
    if (!gate_packed || !gate_scales || !up_packed || !up_scales ||
        !down_packed || !down_scales || !x || !g_out || !u_out || !h_out ||
        !y || H <= 0 || I <= 0 || (H & 31) != 0 || (I & 31) != 0)
        return cudaErrorInvalidValue;

    cudaError_t e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, x, gate_packed, gate_scales, gate_gs, g_out, H, I, false);
    if (e != cudaSuccess) return e;
    e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, x, up_packed, up_scales, up_gs, u_out, H, I, false);
    if (e != cudaSuccess) return e;
    {
        const int n_act = I;
        gemma_dense_mlp_act_kernel<<<(n_act + 255) / 256, 256, 0, stream>>>(
            g_out, u_out, h_out, n_act);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    e = lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
        stream, h_out, down_packed, down_scales, down_gs, y, I, H, false);
    return e == cudaSuccess ? cudaGetLastError() : e;
}

} // namespace gemma
} // namespace lsxcommon
