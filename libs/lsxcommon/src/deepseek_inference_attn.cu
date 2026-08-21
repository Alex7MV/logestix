// deepseek_inference_attn.cu — Hyper-Connections + Multi-head Latent Attention (MLA)
// for DeepSeek-V4-Flash.  Mirrors inference/model.py (Attention, Block, ParallelHead)
// and inference/kernel.py (hc_split_sinkhorn, sparse_attn).
//
// Pure-MLA layers only (compress_ratio == 0): full causal sliding-window
// attention with a learnable attn_sink.  CSA (compressor/indexer) is out of scope.

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/deepseek_inference_csa.h"
#include "lsxcommon/deepseek_norm_kernels.h"
#include "lsxcommon/log.h"
#include "deepseek_inference_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstring>
#include <cmath>
#include <sstream>

namespace lsxcommon {
namespace gpu {

namespace {

// ── FP8 device helpers (mirror safetensors_gpu.cu) ───────────────────────────

// Decode semantics per cuda_fp8.hpp __nv_cvt_fp8_to_halfraw (__NV_E4M3):
// 0x7F/0xFF -> canonical NaN, e == 0 && m != 0 -> subnormal m*2^-9.
__device__ float dev_f8e4m3_to_float(uint8_t b)
{
    if (!b || b == 0x80) return 0.0f;
    if ((b & 0x7F) == 0x7F) return __int_as_float(0x7FC00000);
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    if (e == 0) {
        float v = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
        return s ? -v : v;
    }
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

__device__ float dev_f8e8m0_to_float(uint8_t b)
{
    if (!b) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// ── KV act_quant (FP8_E4M3 inplace, block_size=64) ───────────────────────────
// Reference line 506: act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
// Applies FP8_E4M3 quantization+dequantization (fused) to the non-RoPE dims
// of the KV vector. The RoPE dims (last 64) are left untouched.
// This matches QAT behavior and prevents precision drift in KV cache.
__global__ void kv_act_quant_kernel(
    __half* d_kv,   // [head_dim] in-place FP8 quantize+dequant for first 448 dims
    int nope_dim)    // head_dim - rope_head_dim = 448
{
    constexpr int BLK = 64;
    const int blk = blockIdx.x;
    const int base = blk * BLK;
    if (base >= nope_dim) return;
    const int n = min(BLK, nope_dim - base);

    // Phase 1: amax
    __shared__ float s_amax;
    if (threadIdx.x == 0) s_amax = 0.0f;
    __syncthreads();

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = fabsf(__half2float(d_kv[base + i]));
        local_max = fmaxf(local_max, v);
    }
    __shared__ float s_reduce[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    if (lane == 0) s_reduce[warp] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_reduce[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, offset));
        if (threadIdx.x == 0) s_amax = v;
    }
    __syncthreads();

    // Phase 2: quantize to FP8_E4M3 then dequant back (inplace)
    // ue8m0: round scale UP to next power of 2
    float linear_scale = fmaxf(s_amax, 1e-4f) / 448.0f;
    int log2_s = 0;
    float vs = linear_scale;
    while (vs > 1.0f) { vs *= 0.5f; log2_s++; }
    while (vs <= 0.5f && log2_s > -149) { vs *= 2.0f; log2_s--; }
    float scale = exp2f(static_cast<float>(log2_s));
    if (scale < linear_scale) scale *= 2.0f;
    float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(d_kv[base + i]) * inv_scale;
        v = fmaxf(-448.0f, fminf(448.0f, v));
        // FP8_E4M3 quantization: sign(1) + exp(4, bias=7) + mant(3)
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        uint32_t sign = (bits >> 31) & 1;
        int32_t  exp32 = (bits >> 23) & 0xFF;
        uint32_t mant3 = (bits >> 20) & 0x7;
        int e4 = exp32 - 127 + 7;
        if (e4 < 1) e4 = 0;
        if (e4 > 15) { e4 = 15; mant3 = 7; }
        if (e4 == 0) {
            d_kv[base + i] = __float2half(0.0f);
        } else {
            uint32_t out_bits = (sign << 31) | (static_cast<uint32_t>(e4 + 120) << 23) | (mant3 << 20);
            float result;
            std::memcpy(&result, &out_bits, sizeof(result));
            // Dequant back: result is the FP8 value, multiply by scale
            d_kv[base + i] = __float2half(result * scale);
        }
    }
}

// ── F8_E4M3 → FP16 dequant with per-128x128-tile E8M0 scales ─────────────────
__global__ void dequant_f8_to_half_kernel(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
    __half* __restrict__ out, int64_t n,
    int64_t C, int64_t SC, int64_t block_r, int64_t block_c)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        int64_t r = i / C;
        int64_t c = i % C;
        int64_t sr = r / block_r;
        int64_t sc = c / block_c;
        float v = dev_f8e4m3_to_float(w[i]) *
                  dev_f8e8m0_to_float(scales[sr * SC + sc]);
        out[i] = __float2half(v);
    }
}

// ── HC pre: mixes → sinkhorn → y = Σ pre[j]·x[j] ─────────────────────────────
// x: [hc_mult·dim] FP32 (HC state in float for precision);
// writes y [dim] FP16, post [hc_mult], comb [hc_mult·hc_mult] FP32.
__global__ void hc_pre_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    __half* __restrict__ y,
    float* __restrict__ post,
    float* __restrict__ comb,
    float* __restrict__ pre)
{
    constexpr int N = kHcDim;         // 16384
    constexpr int M = kHcMix;         // 24
    constexpr int H = kHcMult;        // 4
    constexpr int D = kHiddenDim;     // 4096
    constexpr int kThreads = 256;
    __shared__ float s_reduce[kThreads];
    __shared__ float s_mixes[M];
    __shared__ float s_pre[H], s_post[H], s_comb[H * H];

    const int tid = threadIdx.x;

    // rsqrt = rsqrt(mean(x^2) + eps)
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += kThreads) {
        float v = x[i];
        sum_sq += v * v;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    float rsqrt = rsqrtf(s_reduce[0] / static_cast<float>(N) + kRmsEps);

    // mixes[m] = (hc_fn[m] · x) * rsqrt
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        const float* row = hc_fn + m * N;
        for (int i = tid; i < N; i += kThreads)
            acc += row[i] * x[i];
        s_reduce[tid] = acc;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_reduce[tid] += s_reduce[tid + s];
            __syncthreads();
        }
        if (tid == 0) s_mixes[m] = s_reduce[0] * rsqrt;
        __syncthreads();
    }

    // sinkhorn (hc_split_sinkhorn_kernel)
    if (tid == 0) {
        for (int j = 0; j < H; ++j)
            s_pre[j] = 1.0f / (1.0f + expf(-(s_mixes[j] * hc_scale[0] + hc_base[j]))) + kRmsEps;
        for (int j = 0; j < H; ++j)
            s_post[j] = 2.0f / (1.0f + expf(-(s_mixes[j + H] * hc_scale[1] + hc_base[j + H])));
        for (int j = 0; j < H; ++j)
            for (int k = 0; k < H; ++k)
                s_comb[j * H + k] = s_mixes[j * H + k + 2 * H] * hc_scale[2] +
                                    hc_base[j * H + k + 2 * H];

        // row softmax
        for (int j = 0; j < H; ++j) {
            float mx = -1e30f;
            for (int k = 0; k < H; ++k) mx = fmaxf(mx, s_comb[j * H + k]);
            for (int k = 0; k < H; ++k) s_comb[j * H + k] = expf(s_comb[j * H + k] - mx);
        }
        for (int j = 0; j < H; ++j) {
            float rs = 0.0f;
            for (int k = 0; k < H; ++k) rs += s_comb[j * H + k];
            for (int k = 0; k < H; ++k) s_comb[j * H + k] = s_comb[j * H + k] / rs + kRmsEps;
        }
        // column normalize
        for (int k = 0; k < H; ++k) {
            float cs = 0.0f;
            for (int j = 0; j < H; ++j) cs += s_comb[j * H + k];
            for (int j = 0; j < H; ++j) s_comb[j * H + k] = s_comb[j * H + k] / (cs + kRmsEps);
        }
        // sinkhorn iterations (kSinkhornIters - 1)
        for (int it = 1; it < kSinkhornIters; ++it) {
            for (int j = 0; j < H; ++j) {
                float rs = 0.0f;
                for (int k = 0; k < H; ++k) rs += s_comb[j * H + k];
                for (int k = 0; k < H; ++k) s_comb[j * H + k] = s_comb[j * H + k] / (rs + kRmsEps);
            }
            for (int k = 0; k < H; ++k) {
                float cs = 0.0f;
                for (int j = 0; j < H; ++j) cs += s_comb[j * H + k];
                for (int j = 0; j < H; ++j) s_comb[j * H + k] = s_comb[j * H + k] / (cs + kRmsEps);
            }
        }
        for (int j = 0; j < H; ++j) post[j] = s_post[j];
        for (int j = 0; j < H; ++j) pre[j] = s_pre[j];
        for (int j = 0; j < H * H; ++j) comb[j] = s_comb[j];
    }
    __syncthreads();

    // y[d] = Σ_j pre[j] * x[j·D + d]
    for (int i = tid; i < D; i += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < H; ++j)
            acc += s_pre[j] * x[j * D + i];
        y[i] = __float2half(acc);
    }
}

// ── HC post: y[j] = post[j]·x + Σ_k comb[j,k]·residual[k] ────────────────────
// x is now FP32 (module output stays in FP32 — no FP16 conversion)
__global__ void hc_post_kernel(
    const float* __restrict__ x,      // [dim] FP32 (module output)
    const float* __restrict__ residual, // [hc_mult·dim] FP32 (HC state)
    const float* __restrict__ post,    // [hc_mult]
    const float* __restrict__ comb,    // [hc_mult·hc_mult]
    float* __restrict__ y)            // [hc_mult·dim] FP32
{
    constexpr int D = kHiddenDim;
    constexpr int H = kHcMult;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    for (int j = 0; j < H; ++j) {
        // HF ref: y[j] = post[j]*x[i] + sum_i comb[i,j]*residual[i]  (verified broadcast)
        float acc = post[j] * x[i];
        for (int k = 0; k < H; ++k)
            acc += comb[k * H + j] * residual[k * D + i];
        y[j * D + i] = acc;
    }
}

// ── HC head: y = Σ_j sigmoid(mixes[j]·scale + base + eps)·x[j] ───────────────
__global__ void hc_head_kernel(
    const float* __restrict__ x,          // [hc_mult·dim] FP32
    const float* __restrict__ hc_fn,       // [hc_mult, hc_mult·dim] FP32
    const float* __restrict__ hc_scale,    // [1]
    const float* __restrict__ hc_base,     // [hc_mult]
    __half* __restrict__ y)                // [dim] FP16
{
    constexpr int N = kHcDim;
    constexpr int H = kHcMult;
    constexpr int D = kHiddenDim;
    constexpr int kThreads = 256;
    __shared__ float s_reduce[kThreads];
    __shared__ float s_mixes[H];
    __shared__ float s_pre[H];
    const int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += kThreads) {
        float v = x[i];
        sum_sq += v * v;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    float rsqrt = rsqrtf(s_reduce[0] / static_cast<float>(N) + kRmsEps);

    for (int m = 0; m < H; ++m) {
        float acc = 0.0f;
        const float* row = hc_fn + m * N;
        for (int i = tid; i < N; i += kThreads)
            acc += row[i] * x[i];
        s_reduce[tid] = acc;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_reduce[tid] += s_reduce[tid + s];
            __syncthreads();
        }
        if (tid == 0) s_mixes[m] = s_reduce[0] * rsqrt;
        __syncthreads();
    }
    if (tid == 0) {
        for (int j = 0; j < H; ++j)
            s_pre[j] = 1.0f / (1.0f + expf(-(s_mixes[j] * hc_scale[0] + hc_base[j]))) + kRmsEps;
    }
    __syncthreads();
    for (int i = tid; i < D; i += kThreads) {
        float acc = 0.0f;
        for (int j = 0; j < H; ++j)
            acc += s_pre[j] * x[j * D + i];
        y[i] = __float2half(acc);
    }
}

// ── RoPE with optional YaRN scaling (applied on the last rope_head_dim dims) ──────────
// yarn_enabled: 0 = plain RoPE (MLA layers), 1 = YaRN with factor=16 (CSA layers)
__global__ void rope_apply_kernel(
    __half* __restrict__ data, int rows, int row_stride,
    int offset, int rope_dim, float pos, float theta, int yarn_enabled)
{
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kYarnFactor = 16.0f;
    constexpr float kYarnBetaFast = 32.0f;
    constexpr float kYarnBetaSlow = 1.0f;
    constexpr float kYarnOrigMax = 65536.0f;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pairs = rope_dim / 2;
    const int total = rows * n_pairs;
    if (idx >= total) return;
    const int h = idx / n_pairs;
    const int i = idx % n_pairs;

    float freq = 1.0f / powf(theta, static_cast<float>(2 * i) / rope_dim);

    if (yarn_enabled) {
        // YaRN: dimension-based smooth scaling (matching reference model.py)
        constexpr float kPi2 = 2.0f * 3.14159265358979323846f;
        constexpr float kYarnFactor = 16.0f;
        constexpr float kYarnBetaFast = 32.0f;
        constexpr float kYarnBetaSlow = 1.0f;
        constexpr float kYarnOrigMax = 65536.0f;

        // find_correction_dim: dim * log(max_seq_len / (num_rot * 2*pi)) / (2 * log(base))
        float log_base = logf(theta);
        float corr_fast = static_cast<float>(rope_dim)
            * logf(kYarnOrigMax / (kYarnBetaFast * kPi2)) / (2.0f * log_base);
        float corr_slow = static_cast<float>(rope_dim)
            * logf(kYarnOrigMax / (kYarnBetaSlow * kPi2)) / (2.0f * log_base);
        int low = max(0, static_cast<int>(floorf(corr_fast)));
        int high = min(n_pairs - 1, static_cast<int>(ceilf(corr_slow)));

        // linear_ramp: (i - low) / (high - low), clamped [0,1]
        float ramp = 0.0f;
        if (high > low)
            ramp = fmaxf(0.0f, fminf(1.0f,
                static_cast<float>(i - low) / static_cast<float>(high - low)));
        float smooth = 1.0f - ramp;
        freq = freq / kYarnFactor * (1.0f - smooth) + freq * smooth;
    }

    const float angle = pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    if (yarn_enabled >= 2) s = -s;  // inverse

    __half* row = data + h * row_stride + offset;
    const float x0 = __half2float(row[2 * i]);
    const float x1 = __half2float(row[2 * i + 1]);
    row[2 * i] = __float2half(x0 * c - x1 * s);
    row[2 * i + 1] = __float2half(x0 * s + x1 * c);
}

// ── Per-head RMS norm on q (like q *= rsqrt(mean(q^2) + eps)) ────────────────
__global__ void per_head_rms_kernel(__half* __restrict__ q, int n_heads, int head_dim)
{
    constexpr int kThreads = 128;
    __shared__ float s_reduce[kThreads];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = q + h * head_dim;

    float sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += kThreads) {
        float v = __half2float(row[d]);
        sum_sq += v * v;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    const float r = rsqrtf(s_reduce[0] / static_cast<float>(head_dim) + kRmsEps);
    for (int d = tid; d < head_dim; d += kThreads)
        row[d] = __float2half(__half2float(row[d]) * r);
}

// ── Expand single hidden vector to hc_mult copies ────────────────────────────
__global__ void expand_hc_kernel(const __half* __restrict__ src,
                                 float* __restrict__ dst)
{
    constexpr int D = kHiddenDim;
    constexpr int H = kHcMult;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    const float v = __half2float(src[i]);
    for (int j = 0; j < H; ++j)
        dst[j * D + i] = v;
}

// ── Causal sliding-window attention (sparse_attn for pure-MLA) ───────────────
// q: [n_heads, head_dim] FP16; kv_cache: [window, head_dim] FP16;
// sink: [n_heads] FP32; o: [n_heads, head_dim] FP16.
// slot order (decode): newest-first circular window; causal (start_pos excludes future).
__global__ void mla_attn_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ kv_cache,
    const float* __restrict__ sink,
    __half* __restrict__ o,
    int start_pos, int window, float scale)
{
    constexpr int kThreads = 128;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* qrow = q + h * kHeadDim;
    __half* orow = o + h * kHeadDim;

    const int n_attend = min(start_pos + 1, window);

    // First pass: scores → max (shared so all threads see it)
    __shared__ float s_maxv;
    if (tid == 0) s_maxv = -1e30f;
    __syncthreads();
    __shared__ float s_acc[kThreads];
    for (int t = 0; t < n_attend; ++t) {
        const int slot = (start_pos - t + window) % window;
        const __half* krow = kv_cache + slot * kHeadDim;
        float acc = 0.0f;
        for (int d = tid; d < kHeadDim; d += kThreads)
            acc += __half2float(qrow[d]) * __half2float(krow[d]);
        s_acc[tid] = acc;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_acc[tid] += s_acc[tid + s];
            __syncthreads();
        }
        if (tid == 0) s_maxv = fmaxf(s_maxv, s_acc[0] * scale);
        __syncthreads();
    }
    const float maxv = s_maxv;

    // Second pass: probs + output (denominator includes sink).
    // Per-slot probabilities are cached so the denominator reuses them
    // (ascending-t order preserved) instead of recomputing every score.
    __shared__ float s_acc2[kThreads];
    __shared__ float s_probs[kAttnWindow];
    float acc_o[kHeadDim / kThreads];  // 512/128 = 4 slices per thread
    for (int i = 0; i < kHeadDim / kThreads; ++i) acc_o[i] = 0.0f;

    for (int t = 0; t < n_attend; ++t) {
        const int slot = (start_pos - t + window) % window;
        const __half* krow = kv_cache + slot * kHeadDim;
        float acc = 0.0f;
        for (int d = tid; d < kHeadDim; d += kThreads)
            acc += __half2float(qrow[d]) * __half2float(krow[d]);
        s_acc2[tid] = acc;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_acc2[tid] += s_acc2[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            const float p = expf(s_acc2[0] * scale - maxv);
            s_acc2[0] = p;
            if (t < kAttnWindow) s_probs[t] = p;
        }
        __syncthreads();
        const float p = s_acc2[0];
        // accumulate into thread-local slices
        for (int d = tid; d < kHeadDim; d += kThreads) {
            const int slice = d / kThreads;
            acc_o[slice] += p * __half2float(krow[d]);
        }
    }
    // final normalize: denom = Σ p over slots + exp(sink - maxv)
    __shared__ float s_denom_final;
    if (tid == 0) {
        float d2 = expf(sink[h] - maxv);
        const int n_cached = min(n_attend, kAttnWindow);
        for (int t = 0; t < n_cached; ++t)
            d2 += s_probs[t];
        s_denom_final = 1.0f / d2;
    }
    __syncthreads();
    const float inv_denom = s_denom_final;
    for (int d = tid; d < kHeadDim; d += kThreads)
        orow[d] = __float2half(acc_o[d / kThreads] * inv_denom);
}

// ── FP32 GEMM: out[m] = W[m,k] @ x[k] (W FP32 row-major, x FP16) ────────────
// ── FP16 GEMM: out[m] = W[m,k] @ x[k] (W row-major) ──────────────────────────
static cudaError_t gemm_f16_1d(cublasHandle_t cublas_h, cudaStream_t stream,
                               const __half* d_w, int m, int k,
                               const __half* d_x, __half* d_out)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        m, 1, k, &alpha,
        d_w, CUDA_R_16F, k,
        d_x, CUDA_R_16F, k,
        &beta, d_out, CUDA_R_16F, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

// Clamp FP32→FP16 kernel: prevents overflow when dequanted weights exceed FP16 range
static __global__ void clamp_f32_to_f16_kernel(const float* src, __half* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = fmaxf(-65504.0f, fminf(65504.0f, v));
    dst[i] = __float2half(v);
}

// FP32 dequant kernel for F8_E4M3 × F8_E8M0
static __global__ void dequant_f8_to_f32_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales,
    float* __restrict__ out, int64_t n,
    int64_t C, int64_t SC, int64_t block_r, int64_t block_c)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        int64_t r = i / C;
        int64_t c = i % C;
        int64_t sr = r / block_r;
        int64_t sc = c / block_c;
        float sv = 0.0f;
        uint8_t sb = scales[sr * SC + sc];
        if (sb != 0) sv = __uint_as_float(static_cast<uint32_t>(sb) << 23);
        uint8_t b = w[i];
        out[i] = dev_f8e4m3_to_float(b) * sv;
    }
}

// half_to_float_kernel / float_to_half_kernel come from
// deepseek_inference_load.cuh (included above).

// FP32×FP32→FP32 GEMM then convert → FP16 output
static cudaError_t gemm_f32f32_f16(cublasHandle_t cublas_h, cudaStream_t stream,
                                const float* d_w, int m, int k,
                                const float* d_x, __half* d_out,
                                float* d_out_f32)
{
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        m, 1, k, &alpha,
        d_w, CUDA_R_32F, k,
        d_x, CUDA_R_32F, k,
        &beta, d_out_f32, CUDA_R_32F, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    // Convert FP32 output → FP16 (with clamp to prevent overflow)
    int block = 256;
    int grid = (m + block - 1) / block;
    clamp_f32_to_f16_kernel<<<grid, block, 0, stream>>>(d_out_f32, d_out, m);
    return cudaGetLastError();
}

// f8_gemm_1d: dequant F8→FP32, convert FP16 act→FP32, FP32×FP32→FP32→FP16
static cudaError_t f8_gemm_1d(cublasHandle_t cublas_h, cudaStream_t stream,
                              const uint8_t* d_w, const uint8_t* d_scale,
                              int rows, int cols,
                              const __half* d_x, __half* d_out,
                              float* d_scratch)
{
    int64_t n = static_cast<int64_t>(rows) * cols;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    // Step 1: dequant F8 → FP32 weights
    dequant_f8_to_f32_kernel<<<grid, block, 0, stream>>>(
        d_w, d_scale, d_scratch, n, cols, cols / 128, 128, 128);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    // Step 2: convert FP16 activation → FP32 (after weights in scratch)
    float* d_x_f32 = d_scratch + n;
    int grid2 = (cols + block - 1) / block;
    half_to_float_kernel<<<grid2, block, 0, stream>>>(d_x, d_x_f32, cols);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    // Step 3: FP32×FP32→FP32 GEMM, output at d_x_f32 + cols (after activation)
    float* d_out_f32 = d_x_f32 + cols;
    return gemm_f32f32_f16(cublas_h, stream, d_scratch, rows, cols, d_x_f32, d_out, d_out_f32);
}

// ── RMSNorm bridge (half → float → rmsnorm_float → half) ─────────────────────
// half_to_float_kernel / float_to_half_kernel are defined once in
// deepseek_inference_load.cuh (included above) and shared across this
// pipeline's .cu files.

static cudaError_t run_rmsnorm_attn(
    cudaStream_t stream, const __half* d_input, const __half* d_weight,
    __half* d_output, float* d_scratch, float* d_gamma_scratch, int dim, float eps)
{
    int block = 256;
    int grid = (dim + block - 1) / block;
    half_to_float_kernel<<<grid, block, 0, stream>>>(d_input, d_scratch, dim);
    half_to_float_kernel<<<grid, block, 0, stream>>>(d_weight, d_gamma_scratch, dim);
    cudaError_t err = rmsnorm_float(stream, 1, dim, d_scratch,
                                    d_gamma_scratch, eps, d_scratch);
    if (err != cudaSuccess) return err;
    // Clamp FP32→FP16 (prevent overflow from gamma multiplication)
    clamp_f32_to_f16_kernel<<<grid, block, 0, stream>>>(d_scratch, d_output, dim);
    return cudaGetLastError();
}

} // anonymous namespace

// ── Public: HC pre ───────────────────────────────────────────────────────────
cudaError_t run_hc_pre(cudaStream_t stream,
                       const float* d_state,          // [hc_mult·dim] FP32
                       const float* d_fn,              // [mix_hc, hc_mult·dim]
                       const float* d_scale,           // [3]
                       const float* d_base,            // [mix_hc]
                       __half* d_single,               // [dim] FP16
                       float* d_post, float* d_comb,   // [hc_mult], [hc_mult·hc_mult]
                       float* d_pre)                   // [hc_mult]
{
    hc_pre_kernel<<<1, 256, 0, stream>>>(d_state, d_fn, d_scale, d_base,
                                         d_single, d_post, d_comb, d_pre);
    return cudaGetLastError();
}

cudaError_t run_hc_post(cudaStream_t stream,
                        const float* d_x,              // [dim] FP32
                        const float* d_residual,       // [hc_mult·dim] FP32
                        const float* d_post, const float* d_comb,
                        float* d_y)                    // [hc_mult·dim] FP32
{
    hc_post_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
        d_x, d_residual, d_post, d_comb, d_y);
    return cudaGetLastError();
}

cudaError_t run_hc_head(cudaStream_t stream,
                        const float* d_state,         // [hc_mult·dim] FP32
                        const float* d_fn,             // [hc_mult, hc_mult·dim]
                        const float* d_scale,          // [1]
                        const float* d_base,           // [hc_mult]
                        __half* d_single)              // [dim] FP16
{
    hc_head_kernel<<<1, 256, 0, stream>>>(d_state, d_fn, d_scale, d_base, d_single);
    return cudaGetLastError();
}

cudaError_t run_expand_hc(cudaStream_t stream,
                          const __half* d_single, float* d_state)
{
    expand_hc_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(d_single, d_state);
    return cudaGetLastError();
}

// ── Public: MLA forward (one token) ──────────────────────────────────────────
// O-projection shared by MLA and CSA attention (identical math in both:
// o[kAttnHeads,kHeadDim] -> view [kOGroups, gd]; per-group wo_a dequant +
// FP32 GEMM -> olora; then wo_b @ olora -> the final [kHiddenDim] output).
// wo_a.weight [8192,4096] row-major, groups are contiguous 1024-row blocks.
// Previously hand-copied into run_csa_attention with its cudaGetLastError()/
// cublasStatus_t checks silently dropped (an "error handling desync" between
// the two copies) — extracting one checked implementation fixes that for
// both call sites.
static cudaError_t run_o_projection(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights &lw,
    __half *d_o, __half *d_olora, __half *d_out,
    float *d_dequant)
{
    int64_t n = static_cast<int64_t>(lw.wo_a_rows) * lw.wo_a_cols;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    // Use FP32 dequant kernel via reinterpret_cast (data is same size in bytes)
    dequant_f8_to_f32_kernel<<<grid, block, 0, stream>>>(
        lw.d_wo_a, lw.d_wo_a_scales, d_dequant, n,
        lw.wo_a_cols, lw.wo_a_cols / 128, 128, 128);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // O-projection: FP32×FP32→FP32→FP16
    const int gd = (kAttnHeads / kOGroups) * kHeadDim;  // 4096
    cublasSetStream(cublas_h, stream);
    float alpha = 1.0f, beta = 0.0f;
    for (int g = 0; g < kOGroups; ++g) {
        // Convert FP16 d_o → FP32 (after weights in scratch)
        float* d_o_f32 = d_dequant + n;
        int g2 = (gd + block - 1) / block;
        half_to_float_kernel<<<g2, block, 0, stream>>>(d_o + g * gd, d_o_f32, gd);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        // FP32×FP32→FP32 output, then convert → FP16
        float* d_olora_f32 = d_o_f32 + gd;
        cublasStatus_t st = cublasGemmEx(
            cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            kOloraRank, 1, gd, &alpha,
            d_dequant + g * kOloraRank * gd, CUDA_R_32F, gd,
            d_o_f32, CUDA_R_32F, gd,
            &beta, d_olora_f32, CUDA_R_32F, kOloraRank,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
        int g3 = (kOloraRank + block - 1) / block;
        clamp_f32_to_f16_kernel<<<g3, block, 0, stream>>>(d_olora_f32, d_olora + g * kOloraRank, kOloraRank);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // out = wo_b @ olora → [kHiddenDim]
    return f8_gemm_1d(cublas_h, stream, lw.d_wo_b, lw.d_wo_b_scales,
                      lw.wo_b_rows, lw.wo_b_cols, d_olora, d_out, d_dequant);
}

cudaError_t run_mla_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights& lw,
    const __half* d_x,
    __half* d_kv_cache_layer,
    int start_pos,
    __half* d_q, __half* d_qr, __half* d_qr2,
    __half* d_kv, __half* d_kv2,
    __half* d_o, __half* d_olora, __half* d_out,
    float* d_dequant,
    float* d_norm_scratch, float* d_norm_gamma_scratch)
{
    if (!lw.d_wq_a || !lw.d_wq_b || !lw.d_wkv || !lw.d_wo_a || !lw.d_wo_b) {
        log::error("[attn] layer missing attention weights");
        return cudaErrorInvalidValue;
    }
    // Mandatory per-layer companions of the weights above; the loader marks
    // them optional, but a nullptr here becomes an in-kernel illegal address.
    if (!lw.d_q_norm || !lw.d_kv_norm || !lw.d_attn_sink) {
        log::error("[attn] layer missing q_norm/kv_norm/attn_sink");
        return cudaErrorInvalidValue;
    }

    // qr = wq_a @ x
    cudaError_t e = f8_gemm_1d(cublas_h, stream, lw.d_wq_a, lw.d_wq_a_scales,
                               lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    // qr = q_norm(qr)
    e = run_rmsnorm_attn(stream, d_qr, lw.d_q_norm, d_qr2,
                         d_norm_scratch, d_norm_gamma_scratch, kQloraRank, kRmsEps);
    if (e != cudaSuccess) return e;
    // q = wq_b @ qr2 → [32768]
    e = f8_gemm_1d(cublas_h, stream, lw.d_wq_b, lw.d_wq_b_scales,
                   lw.wq_b_rows, lw.wq_b_cols, d_qr2, d_q, d_dequant);
    if (e != cudaSuccess) return e;
    // per-head RMS + RoPE on last 64 dims
    per_head_rms_kernel<<<kAttnHeads, 128, 0, stream>>>(d_q, kAttnHeads, kHeadDim);
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        int block = 128;
        rope_apply_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
            d_q, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta,
            lw.compress_ratio > 0 ? 1 : 0);
    }

    // kv = wkv @ x → [512]
    e = f8_gemm_1d(cublas_h, stream, lw.d_wkv, lw.d_wkv_scales,
                   lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_kv, lw.d_kv_norm, d_kv2,
                         d_norm_scratch, d_norm_gamma_scratch, kHeadDim, kRmsEps);
    if (e != cudaSuccess) return e;
    {
        int total = kRopeHeadDim / 2;
        int block = 64;
        rope_apply_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
            d_kv2, 1, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta,
            lw.compress_ratio > 0 ? 1 : 0);
    }

    // write kv into circular cache slot (with FP8 act_quant on non-RoPE dims)
    {
        // Apply act_quant(kv[..., :-rd], 64, inplace=True) — FP8 quantize+dequant
        int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
        kv_act_quant_kernel<<<n_blocks_q, 256, 0, stream>>>(d_kv2, kHeadDim - kRopeHeadDim);
    }
    const int slot = start_pos % kAttnWindow;
    e = cudaMemcpyAsync(d_kv_cache_layer + slot * kHeadDim, d_kv2,
                        static_cast<size_t>(kHeadDim) * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    // attention
    {
        float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
        mla_attn_kernel<<<kAttnHeads, 128, 0, stream>>>(
            d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
            start_pos, kAttnWindow, scale);
    }

    // inverse RoPE on o[..., -64:]
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        int block = 128;
        rope_apply_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
            d_o, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta,
            lw.compress_ratio > 0 ? 3 : 2);
    }

    // o-projection: o[64,512] → view [8,4096]; per-group wo_a[1024,4096] → [8,1024]
    // wo_a.weight [8192,4096] row-major, groups are contiguous 1024-row blocks.
    e = run_o_projection(cublas_h, stream, lw, d_o, d_olora, d_out, d_dequant);
    if (e != cudaSuccess) return e;

    return cudaSuccess;
}

// ── CSA: sparse attention kernel (window + compressed KV) ───────────────────
// One block per head. Attends to window positions AND compressed KV positions.
// The kv_cache is laid out as: [window_slots, head_dim] followed by
// [compressed_slots, head_dim] in a single contiguous buffer.
__global__ void csa_sparse_attn_kernel(
    const __half* __restrict__ q,         // [n_heads, head_dim]
    const __half* __restrict__ kv_cache, // [total_slots, head_dim] (window + compressed)
    const float*  __restrict__ sink,      // [n_heads]
    __half* __restrict__ o,               // [n_heads, head_dim]
    int start_pos,                         // absolute position (for circular window)
    int window,                             // window size
    int n_attend,                           // number of window positions (≤ window)
    int n_compressed,                      // number of compressed positions
    float scale)
{
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int kThreads = 128;
    constexpr int kHD = 512;

    // Use s_acc as 4-slice thread-local accumulator (like mla_attn_kernel)
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_part[kThreads];
    float acc_o[kHD / kThreads];  // 4 slices per thread
    for (int i = 0; i < kHD / kThreads; ++i) acc_o[i] = 0.0f;

    if (tid == 0) { s_max = -1e30f; s_sum = 0.0f; }
    __syncthreads();

    const __half* qrow = q + h * kHD;
    const int total = n_attend + n_compressed;

    // Pass 1: find max score
    for (int t = 0; t < total; ++t) {
        int slot;
        if (t < n_attend)
            slot = (start_pos - t + window) % window;
        else
            slot = window + (t - n_attend);
        const __half* krow = kv_cache + slot * kHD;
        float partial = 0.0f;
        for (int d = tid; d < kHD; d += kThreads)
            partial += __half2float(qrow[d]) * __half2float(krow[d]);
        s_part[tid] = partial;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_part[tid] += s_part[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float sc = s_part[0] * scale;
            if (sc > s_max) s_max = sc;
        }
        __syncthreads();
    }
    // Include sink
    if (tid == 0) {
        float sk = sink[h];
        if (sk > s_max) s_max = sk;
    }
    __syncthreads();
    const float maxv = s_max;

    // Pass 2: probs + weighted accumulation (thread-local slices, like mla_attn)
    for (int t = 0; t < total; ++t) {
        int slot;
        if (t < n_attend)
            slot = (start_pos - t + window) % window;
        else
            slot = window + (t - n_attend);
        const __half* krow = kv_cache + slot * kHD;
        float partial = 0.0f;
        for (int d = tid; d < kHD; d += kThreads)
            partial += __half2float(qrow[d]) * __half2float(krow[d]);
        s_part[tid] = partial;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) s_part[tid] += s_part[tid + s];
            __syncthreads();
        }
        // Thread 0 computes weight and updates sum
        if (tid == 0) {
            float p = expf(s_part[0] * scale - maxv);
            s_part[0] = p;  // reuse s_part[0] as weight storage
            s_sum += p;
        }
        __syncthreads();
        const float p = s_part[0];  // all threads read weight
        for (int d = tid; d < kHD; d += kThreads) {
            const int slice = d / kThreads;
            acc_o[slice] += p * __half2float(krow[d]);
        }
    }
    // Sink contribution (just adds to denominator, no output contribution)
    if (tid == 0) s_sum += expf(sink[h] - maxv);
    __syncthreads();

    const float inv = 1.0f / (s_sum + 1e-9f);
    __half* orow = o + h * kHD;
    for (int d = tid; d < kHD; d += kThreads) {
        const int slice = d / kThreads;
        orow[d] = __float2half(acc_o[slice] * inv);
    }
}

// ── CSA: apply RMSNorm to compressed KV ──────────────────────────────────────
__global__ void compress_kv_norm_kernel(
    __half* d_kv,                    // [head_dim] compressed KV (in-place)
    const __half* d_weight,          // [head_dim] norm weight
    int head_dim)
{
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    __shared__ float s_partial[32];
    float vsum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __half2float(d_kv[i]);
        vsum += v * v;
    }
    vsum += __shfl_down_sync(0xffffffffu, vsum, 16);
    vsum += __shfl_down_sync(0xffffffffu, vsum, 8);
    vsum += __shfl_down_sync(0xffffffffu, vsum, 4);
    vsum += __shfl_down_sync(0xffffffffu, vsum, 2);
    vsum += __shfl_down_sync(0xffffffffu, vsum, 1);
    if (lane == 0) s_partial[warp] = vsum;
    __syncthreads();
    if (warp == 0) {
        float wsum = (lane < (blockDim.x / 32)) ? s_partial[lane] : 0.0f;
        wsum += __shfl_down_sync(0xffffffffu, wsum, 16);
        wsum += __shfl_down_sync(0xffffffffu, wsum, 8);
        wsum += __shfl_down_sync(0xffffffffu, wsum, 4);
        wsum += __shfl_down_sync(0xffffffffu, wsum, 2);
        wsum += __shfl_down_sync(0xffffffffu, wsum, 1);
        if (lane == 0) s_partial[0] = wsum;
    }
    __syncthreads();
    float s_sum = s_partial[0];
    float inv = rsqrtf(s_sum / static_cast<float>(head_dim) + kRmsEps);
    for (int i = tid; i < head_dim; i += blockDim.x)
        d_kv[i] = __float2half(__half2float(d_kv[i]) * inv * __half2float(d_weight[i]));
}

// ── CSA: Compressor step kernel (gated KV pooling) ──────────────────────────
// Matches reference Compressor.forward (model.py:316-377).
// State layout: [ratio*coff, coff*head_dim] FP32.
//   overlap (ratio==4, coff==2): token `sp` is written to row `ratio + sp%ratio`.
//     Rows [0,ratio) hold the previous block (first-half dims) = overlapping window;
//     Rows [ratio,2ratio) hold the current window (second-half dims).
//     On compress: pool over 2*ratio rows (row r<ratio -> col i; row r>=ratio -> col d+i),
//     then shift rows [ratio,2ratio) into [0,ratio) for the next round.
//   non-overlap (ratio==128, coff==1): token written to `sp%ratio`, pool over ratio rows.
__global__ void compressor_step_kernel(
    const __half *d_kv, const __half *d_score,
    const float  *d_ape,
    float *d_kv_state, float *d_score_state,
    __half *d_compressed_kv,
    int ratio, int coff, int head_dim, int start_pos)
{
    const int d = head_dim;
    const int out_dim = coff * d;
    const int tid = threadIdx.x;
    const int slot = start_pos % ratio;
    const int store_row = (coff == 2) ? (ratio + slot) : slot;

    // Store current token KV + score (score += ape[sp%ratio], like reference)
    for (int i = tid; i < out_dim; i += blockDim.x) {
        d_kv_state[store_row * out_dim + i] = __half2float(d_kv[i]);
        d_score_state[store_row * out_dim + i] = __half2float(d_score[i]) + d_ape[slot * out_dim + i];
    }
    __syncthreads();

    const bool should_compress = ((start_pos + 1) % ratio) == 0;
    if (!should_compress) return;

    if (coff == 1) {
        // Non-overlapping compression: pool over `ratio` rows.
        for (int i = tid; i < d; i += blockDim.x) {
            float max_s = -1e30f;
            for (int r = 0; r < ratio; ++r)
                max_s = fmaxf(max_s, d_score_state[r * out_dim + i]);
            float sum_exp = 0.0f;
            for (int r = 0; r < ratio; ++r)
                sum_exp += expf(d_score_state[r * out_dim + i] - max_s);
            float compressed = 0.0f;
            for (int r = 0; r < ratio; ++r)
                compressed += d_kv_state[r * out_dim + i] * (expf(d_score_state[r * out_dim + i] - max_s) / (sum_exp + 1e-9f));
            d_compressed_kv[i] = __float2half(compressed);
        }
    } else {
        // Overlapping compression (ratio==4): pool over 2*ratio rows.
        const int nrows = 2 * ratio;
        for (int i = tid; i < d; i += blockDim.x) {
            float max_s = -1e30f;
            for (int r = 0; r < nrows; ++r) {
                const int c = (r < ratio) ? i : (d + i);
                max_s = fmaxf(max_s, d_score_state[r * out_dim + c]);
            }
            float sum_exp = 0.0f;
            for (int r = 0; r < nrows; ++r) {
                const int c = (r < ratio) ? i : (d + i);
                sum_exp += expf(d_score_state[r * out_dim + c] - max_s);
            }
            float compressed = 0.0f;
            for (int r = 0; r < nrows; ++r) {
                const int c = (r < ratio) ? i : (d + i);
                compressed += d_kv_state[r * out_dim + c] * (expf(d_score_state[r * out_dim + c] - max_s) / (sum_exp + 1e-9f));
            }
            d_compressed_kv[i] = __float2half(compressed);
        }
        __syncthreads();
        // Shift: rows [ratio,2*ratio) -> [0,ratio) for next overlap window.
        for (int i = tid; i < ratio * out_dim; i += blockDim.x) {
            const int r = i / out_dim;
            const int c = i % out_dim;
            d_kv_state[r * out_dim + c] = d_kv_state[(r + ratio) * out_dim + c];
            d_score_state[r * out_dim + c] = d_score_state[(r + ratio) * out_dim + c];
        }
    }
}

// ── Public: CSA forward (one token, layers 2-41) ────────────────────────────
// Uses the SAME Q/KV/O-projection as MLA, plus the compressor step.
// For now, uses mla_attn_kernel for attention (window-only, no compressed positions).
// The compressor state is maintained for future use when sparse_attn is added.
cudaError_t run_csa_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights& lw,
    const __half* d_x,
    __half* d_kv_cache_layer,
    CSAState& csa_state,
    int start_pos,
    __half* d_q, __half* d_qr, __half* d_qr2,
    __half* d_kv, __half* d_kv2,
    __half* d_o, __half* d_olora, __half* d_out,
    float* d_dequant,
    float* d_norm_scratch, float* d_norm_gamma_scratch)
{
    if (!lw.d_wq_a || !lw.d_wq_b || !lw.d_wkv || !lw.d_wo_a || !lw.d_wo_b) {
        log::error("[csa] layer missing attention weights");
        return cudaErrorInvalidValue;
    }
    // Mandatory per-layer companions of the weights above; a nullptr here
    // becomes an in-kernel illegal address.
    if (!lw.d_q_norm || !lw.d_kv_norm || !lw.d_attn_sink) {
        log::error("[csa] layer missing q_norm/kv_norm/attn_sink");
        return cudaErrorInvalidValue;
    }

    // ── Q projection (identical to MLA) ──────────────────────────────────
    cudaError_t e = f8_gemm_1d(cublas_h, stream, lw.d_wq_a, lw.d_wq_a_scales,
                               lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_qr, lw.d_q_norm, d_qr2,
                         d_norm_scratch, d_norm_gamma_scratch, kQloraRank, kRmsEps);
    if (e != cudaSuccess) return e;
    e = f8_gemm_1d(cublas_h, stream, lw.d_wq_b, lw.d_wq_b_scales,
                   lw.wq_b_rows, lw.wq_b_cols, d_qr2, d_q, d_dequant);
    if (e != cudaSuccess) return e;
    per_head_rms_kernel<<<kAttnHeads, 128, 0, stream>>>(d_q, kAttnHeads, kHeadDim);
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        rope_apply_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            d_q, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // ── KV projection (identical to MLA) ──────────────────────────────────
    e = f8_gemm_1d(cublas_h, stream, lw.d_wkv, lw.d_wkv_scales,
                   lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_kv, lw.d_kv_norm, d_kv2,
                         d_norm_scratch, d_norm_gamma_scratch, kHeadDim, kRmsEps);
    if (e != cudaSuccess) return e;
    {
        int total = kRopeHeadDim / 2;
        rope_apply_kernel<<<(total + 63) / 64, 64, 0, stream>>>(
            d_kv2, 1, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // Store KV in sliding-window cache (with FP8 act_quant on non-RoPE dims)
    {
        int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
        kv_act_quant_kernel<<<n_blocks_q, 256, 0, stream>>>(d_kv2, kHeadDim - kRopeHeadDim);
    }
    const int slot = start_pos % kAttnWindow;
    e = cudaMemcpyAsync(d_kv_cache_layer + slot * kHeadDim, d_kv2,
        static_cast<size_t>(kHeadDim) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

// ── Compressor step (gated KV pooling) ───────────────────────────────
    // Use d_qr2 (1024 elements) for wkv output, d_qr (1024) for wgate output
    // Both are large enough for comp_out = coff*head_dim = 1024 (ratio=4) or 512 (ratio=128)
    if (lw.d_comp_wkv && lw.d_comp_wgate && lw.d_comp_ape) {
        int coff = lw.comp_coff;
        int comp_out = coff * kHeadDim;
        {
            float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas_h, stream);
            cublasStatus_t s1 = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                comp_out, 1, kHiddenDim, &alpha,
                lw.d_comp_wkv, CUDA_R_16F, kHiddenDim,
                d_x, CUDA_R_16F, kHiddenDim, &beta,
                d_qr2, CUDA_R_16F, comp_out,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (s1 != CUBLAS_STATUS_SUCCESS) {
                log::error("[csa] compressor wkv GEMM failed");
                return cudaErrorInvalidValue;
            }
            cublasStatus_t s2 = cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                comp_out, 1, kHiddenDim, &alpha,
                lw.d_comp_wgate, CUDA_R_16F, kHiddenDim,
                d_x, CUDA_R_16F, kHiddenDim, &beta,
                d_qr, CUDA_R_16F, comp_out,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (s2 != CUBLAS_STATUS_SUCCESS) {
                log::error("[csa] compressor wgate GEMM failed");
                return cudaErrorInvalidValue;
            }
        }
        bool should_compress = ((start_pos + 1) % lw.compress_ratio) == 0;
        __half* d_compressed_slot =
            csa_state.d_compressed_kv + csa_state.n_compressed * kHeadDim;
        compressor_step_kernel<<<1, 256, 0, stream>>>(
            d_qr2, d_qr, lw.d_comp_ape,
            csa_state.d_comp_kv_state, csa_state.d_comp_score_state,
            d_compressed_slot,
            lw.compress_ratio, coff, kHeadDim, start_pos);
        if (should_compress) {
            // Apply RMSNorm to compressed KV
            if (lw.d_comp_norm) {
                compress_kv_norm_kernel<<<1, 256, 0, stream>>>(
                    d_compressed_slot, lw.d_comp_norm, kHeadDim);
            }
            // Apply RoPE to last 64 dims of compressed KV
            int comp_pos = start_pos + 1 - lw.compress_ratio;
            {
                int total = kRopeHeadDim / 2;
                rope_apply_kernel<<<(total + 63) / 64, 64, 0, stream>>>(
                    d_compressed_slot, 1, kHeadDim, kHeadDim - kRopeHeadDim,
                    kRopeHeadDim, static_cast<float>(comp_pos), lw.rope_theta, 1);
            }
            // Main compressor (rotate=False) applies FP8 act_quant to non-RoPE dims
            // (reference model.py:372 act_quant(kv[..., :-rd], 64, ..., True)).
            {
                int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
                kv_act_quant_kernel<<<n_blocks_q, 256, 0, stream>>>(d_compressed_slot, kHeadDim - kRopeHeadDim);
            }
            csa_state.n_compressed++;
        }
    }

    // ── Build combined KV cache: window + compressed ─────────────────────
    // Copy compressed KV slots to end of kv_cache_layer buffer.
    // The buffer is [kAttnWindow + max_compressed, head_dim].
    // Window slots: [0, kAttnWindow)
    // Compressed slots: [kAttnWindow, kAttnWindow + n_compressed)
    int n_compressed = csa_state.n_compressed;
    if (n_compressed > 0) {
        size_t bytes = static_cast<size_t>(n_compressed) * kHeadDim * sizeof(__half);
        e = cudaMemcpyAsync(
            d_kv_cache_layer + static_cast<size_t>(kAttnWindow) * kHeadDim,
            csa_state.d_compressed_kv,
            bytes, cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }

    // ── Sparse attention over window + compressed positions ─────────────
    {
        int n_attend = std::min(start_pos + 1, kAttnWindow);
        float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
        if (n_compressed > 0) {
            csa_sparse_attn_kernel<<<kAttnHeads, 128, 0, stream>>>(
                d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
                start_pos, kAttnWindow, n_attend, n_compressed, scale);
        } else {
            mla_attn_kernel<<<kAttnHeads, 128, 0, stream>>>(
                d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
                start_pos, kAttnWindow, scale);
        }
    }

    // ── Inverse RoPE on output ───────────────────────────────────────────
    {
        int total = kAttnHeads * (kRopeHeadDim / 2);
        rope_apply_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            d_o, kAttnHeads, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 3);
    }

    // ── O-projection (identical to MLA — FP32×FP32→FP32→FP16) ────────────
    e = run_o_projection(cublas_h, stream, lw, d_o, d_olora, d_out, d_dequant);
    return e;
}

} // namespace gpu
} // namespace lsxcommon
