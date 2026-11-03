// deepseek_moe_inference_attn.cu — Hyper-Connections + Multi-head Latent Attention (MLA)
// for DeepSeek-V4-Flash.  Mirrors inference/model.py (Attention, Block, ParallelHead)
// and inference/kernel.py (hc_split_sinkhorn, sparse_attn).
//
// Pure-MLA layers only (compress_ratio == 0): full causal sliding-window
// attention with a learnable attn_sink.  CSA (compressor/indexer) is out of scope.

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/deepseek_moe_pro_inference_csa.h"
#include "lsxcommon/deepseek_moe_pro_norm_kernels.h"
#include "lsxcommon/shared_experts.h"
#include "lsx_kernels_quant.cuh"
#include "lsxcommon/log.h"
#include "deepseek_moe_pro_load.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <sstream>

namespace lsxcommon {
namespace deepseek_pro {
namespace gpu {
using lsxcommon::gpu::dequant_shared_f8_half;

namespace {

// ── FP8 device helpers (mirror safetensors_gpu.cu) ───────────────────────────
// The signed F8_E4M3 decode lives once in lsx_kernels_quant.cuh
// ::F8E4m3ToFloat; the unsigned NVFP4 block scale is the separate
// nvfp4_scale.h::E4m3ToFloat. The two differ in signedness and are not
// interchangeable.


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
        float v = kernels::F8E4m3ToFloat(w[i]) *
                  kernels::E8m0ToFloat(scales[sr * SC + sc]);
        out[i] = __float2half(v);
    }
}

// ── HC pre: mixes → sinkhorn → y = Σ pre[j]·x[j] ─────────────────────────────
// x: [hc_mult·dim] FP32 (HC state in float for precision);
// writes y [dim] FP16, post [hc_mult], comb [hc_mult·hc_mult] FP32.
//
// Blackwell perf: the old single-block (<<<1,256>>>) version kept ~147 of
// ~148 SMs idle while streaming hc_fn [24×16384] three times per launch.
// Split into two deterministic phases:
//   Phase A (GPU-wide): grid ((M+1), kHcTiles) — each block reduces one tile
//     of one mix row (or the Σx² row) to a single partial float.
//   Phase B (single block): reduce partials in fixed ascending-tile order,
//     sinkhorn (thread-0 serial, unchanged math), write post/comb/pre.
//   Phase C: y epilogue over D elements across many blocks.
// Every reduction order is fixed → run-to-run deterministic.
namespace {

constexpr int kHcPartThreads = 256;
constexpr int kHcTiles = 32;   // tiles per row; partials = (M+1)*kHcTiles floats

__global__ void __launch_bounds__(kHcPartThreads)
hc_mix_partials_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hc_fn,
    int M, int N, int tiles_per_row,
    float* __restrict__ partials)      // [(M+1) * tiles_per_row]
{
    const int m = blockIdx.x;          // 0..M-1 rows; M = Σx² row
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int tile = (N + tiles_per_row - 1) / tiles_per_row;
    const int begin = b * tile;
    const int end = min(begin + tile, N);

    float acc = 0.0f;
    if (m < M) {
        const float* row = hc_fn + static_cast<int64_t>(m) * N;
        for (int i = begin + tid; i < end; i += blockDim.x)
            acc += row[i] * x[i];
    } else {
        for (int i = begin + tid; i < end; i += blockDim.x) {
            const float v = x[i];
            acc += v * v;
        }
    }
    __shared__ float s_warp[kHcPartThreads / 32];
    const float total = lsx_block_reduce_sum(acc, s_warp);
    if (tid == 0) partials[m * tiles_per_row + b] = total;
}

// Reduce partials (fixed ascending order), apply rsqrt + scale/base, and run
// the thread-0 sinkhorn exactly as before. M == kHcMix, H == kHcMult.
__global__ void __launch_bounds__(256)
hc_pre_finalize_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    int M, int H, int N, int tiles_per_row,
    float* __restrict__ post,
    float* __restrict__ comb,
    float* __restrict__ pre,
    float rms_eps)
{
    constexpr int D = kHiddenDim;
    __shared__ float s_red[32];
    __shared__ float s_mixes[64];
    const int tid = threadIdx.x;

    // rsqrt = rsqrt(mean(x^2) + eps)
    float ssq = 0.0f;
    for (int b = tid; b < tiles_per_row; b += blockDim.x)
        ssq += partials[M * tiles_per_row + b];
    const float ssq_tot = lsx_block_reduce_sum(ssq, s_red);
    const float rsqrt_v = rsqrtf(ssq_tot / static_cast<float>(N) + rms_eps);

    // mixes[m] = (Σ_b partials[m][b]) * rsqrt — ascending-b order per row
    if (tid < M) {
        float acc = 0.0f;
        const float* pr = partials + tid * tiles_per_row;
        for (int b = 0; b < tiles_per_row; ++b) acc += pr[b];
        s_mixes[tid] = acc * rsqrt_v;
    }
    __syncthreads();

    // Sinkhorn (hc_split_sinkhorn_kernel semantics). Bit-exact parallel
    // form: the serial thread-0 loop was latency-bound (~39 us at 1 block);
    // each row/col phase is element-independent given fixed ascending
    // accumulation order, so per-row (tid = row j < H) and per-col
    // (tid = col k < H) threads replicate the exact serial arithmetic —
    // same loads, same order, same per-element ops — with barriers between
    // phases replacing the program order.
    __shared__ float s_pre[8];
    __shared__ float s_post[8];
    __shared__ float s_comb[16];  // [H*H], H = kHcMult = 4
    if (tid < H)
        s_pre[tid] =
            1.0f / (1.0f + expf(-(s_mixes[tid] * hc_scale[0] + hc_base[tid]))) +
            kRmsEps;
    if (tid < H)
        s_post[tid] =
            2.0f /
            (1.0f + expf(-(s_mixes[tid + H] * hc_scale[1] + hc_base[tid + H])));
    if (tid < H) {
        for (int k = 0; k < H; ++k)
            s_comb[tid * H + k] =
                s_mixes[tid * H + k + 2 * H] * hc_scale[2] +
                hc_base[tid * H + k + 2 * H];
    }
    __syncthreads();
    // row softmax
    if (tid < H) {
        float mx = -1e30f;
        for (int k = 0; k < H; ++k) mx = fmaxf(mx, s_comb[tid * H + k]);
        for (int k = 0; k < H; ++k)
            s_comb[tid * H + k] = expf(s_comb[tid * H + k] - mx);
    }
    __syncthreads();
    if (tid < H) {
        float rs = 0.0f;
        for (int k = 0; k < H; ++k) rs += s_comb[tid * H + k];
        for (int k = 0; k < H; ++k)
            s_comb[tid * H + k] = s_comb[tid * H + k] / rs + kRmsEps;
    }
    __syncthreads();
    // column normalize
    if (tid < H) {
        float cs = 0.0f;
        for (int j = 0; j < H; ++j) cs += s_comb[j * H + tid];
        for (int j = 0; j < H; ++j)
            s_comb[j * H + tid] = s_comb[j * H + tid] / (cs + kRmsEps);
    }
    __syncthreads();
    // sinkhorn iterations (kSinkhornIters - 1)
    for (int it = 1; it < kSinkhornIters; ++it) {
        if (tid < H) {
            float rs = 0.0f;
            for (int k = 0; k < H; ++k) rs += s_comb[tid * H + k];
            for (int k = 0; k < H; ++k)
                s_comb[tid * H + k] = s_comb[tid * H + k] / (rs + kRmsEps);
        }
        __syncthreads();
        if (tid < H) {
            float cs = 0.0f;
            for (int j = 0; j < H; ++j) cs += s_comb[j * H + tid];
            for (int j = 0; j < H; ++j)
                s_comb[j * H + tid] = s_comb[j * H + tid] / (cs + kRmsEps);
        }
        __syncthreads();
    }
    if (tid < H) post[tid] = s_post[tid];
    if (tid < H) pre[tid] = s_pre[tid];
    for (int idx = tid; idx < H * H; idx += blockDim.x)
        comb[idx] = s_comb[idx];
}

// y[d] = Σ_j pre[j] * x[j·D + d]
__global__ void hc_combine_kernel(
    const float* __restrict__ x,
    const float* __restrict__ pre,
    float* __restrict__ y)
{
    constexpr int D = kHiddenDim;
    constexpr int H = kHcMult;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float acc = 0.0f;
    for (int j = 0; j < H; ++j)
        acc += pre[j] * x[j * D + i];
    // V4-Pro keeps the HC residual in FP32 and its norm can exceed FP16's
    // 65504 max (the source runs BF16/FP8 with a 3e38 range); the combined
    // vector feeds an RMSNorm, which is scale-invariant, but must be stored
    // in FP32 or it becomes inf before the norm sees it.
    y[i] = acc;
}

} // anonymous namespace

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
// Two-phase variant of hc_pre (M = H = kHcMult rows, no sinkhorn/post/comb).
// pre[] is staged through the scratch tail so the combine kernel can read it.
__global__ void __launch_bounds__(kHcPartThreads)
hc_head_partials_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hc_fn,
    int M, int N, int tiles_per_row,
    float* __restrict__ partials)      // [(M+1) * tiles_per_row]
{
    const int m = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int tile = (N + tiles_per_row - 1) / tiles_per_row;
    const int begin = b * tile;
    const int end = min(begin + tile, N);

    float acc = 0.0f;
    if (m < M) {
        const float* row = hc_fn + static_cast<int64_t>(m) * N;
        for (int i = begin + tid; i < end; i += blockDim.x)
            acc += row[i] * x[i];
    } else {
        for (int i = begin + tid; i < end; i += blockDim.x) {
            const float v = x[i];
            acc += v * v;
        }
    }
    __shared__ float s_warp[kHcPartThreads / 32];
    const float total = lsx_block_reduce_sum(acc, s_warp);
    if (tid == 0) partials[m * tiles_per_row + b] = total;
}

__global__ void __launch_bounds__(256)
hc_head_finalize_kernel(
    const float* __restrict__ partials,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    int M, int H, int N, int tiles_per_row,
    float* __restrict__ pre)           // [H]
{
    __shared__ float s_red[32];
    __shared__ float s_mixes[64];
    const int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int b = tid; b < tiles_per_row; b += blockDim.x)
        ssq += partials[M * tiles_per_row + b];
    const float ssq_tot = lsx_block_reduce_sum(ssq, s_red);
    const float rsqrt_v = rsqrtf(ssq_tot / static_cast<float>(N) + kRmsEps);

    if (tid < M) {
        float acc = 0.0f;
        const float* pr = partials + tid * tiles_per_row;
        for (int b = 0; b < tiles_per_row; ++b) acc += pr[b];
        s_mixes[tid] = acc * rsqrt_v;
    }
    __syncthreads();
    if (tid == 0) {
        for (int j = 0; j < H; ++j)
            pre[j] = 1.0f / (1.0f + expf(-(s_mixes[j] * hc_scale[0] + hc_base[j]))) + kRmsEps;
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
__global__ void __launch_bounds__(128)
per_head_rms_kernel(__half* __restrict__ q, int n_heads, int head_dim)
{
    constexpr int kThreads = 128;
    __shared__ float s_red[32];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    __half* row = q + h * head_dim;

    float sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += kThreads) {
        float v = __half2float(row[d]);
        sum_sq += v * v;
    }
    const float total = lsx_block_reduce_sum(sum_sq, s_red);
    const float r = rsqrtf(total / static_cast<float>(head_dim) + kRmsEps);
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
//
// Blackwell perf vs the original double-pass version:
//   • QK scores are computed ONCE and cached in dynamic shared memory — the
//     second pass no longer recomputes every dot product (2× less QK work).
//   • Reductions use warp shuffles (2 barriers per slot instead of ~8) via
//     lsx_block_reduce_sum — the old tree walk issued ~2000 __syncthreads()
//     per head per token.
//   • 256-thread blocks (8 warps) double per-SM warp occupancy.
__global__ void __launch_bounds__(256)
mla_attn_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ kv_cache,
    const float* __restrict__ sink,
    __half* __restrict__ o,
    int start_pos, int window, float scale)
{
    constexpr int kThreads = 256;
    constexpr int kHD = kHeadDim;      // 512
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* qrow = q + h * kHD;
    __half* orow = o + h * kHD;

    const int n_attend = min(start_pos + 1, window);

    extern __shared__ float s_smem[];
    float* s_scores = s_smem;             // [n_attend] scaled QK scores
    float* s_warp = s_scores + n_attend;  // [kThreads/32] reduction scratch

    // Pass 1: scores (cached, scaled)
    for (int t = 0; t < n_attend; ++t) {
        const int slot = (start_pos - t + window) % window;
        const __half* krow = kv_cache + slot * kHD;
        float acc = 0.0f;
        for (int d = tid; d < kHD; d += kThreads)
            acc += __half2float(qrow[d]) * __half2float(krow[d]);
        const float tot = lsx_block_reduce_sum(acc, s_warp);
        if (tid == 0) s_scores[t] = tot * scale;
    }
    __syncthreads();

    // Block max over cached scores + sink (fixed order → deterministic).
    float mx = -1e30f;
    for (int t = 0; t < n_attend; ++t) mx = fmaxf(mx, s_scores[t]);
    mx = fmaxf(mx, sink[h]);
    const float maxv = mx;

    // Pass 2: probs + output accumulation from cached scores (no QK recompute).
    float acc_o[kHD / kThreads];  // 2 slices per thread
    for (int i = 0; i < kHD / kThreads; ++i) acc_o[i] = 0.0f;
    float psum = 0.0f;
    for (int t = 0; t < n_attend; ++t) {
        const int slot = (start_pos - t + window) % window;
        const __half* krow = kv_cache + slot * kHD;
        const float p = expf(s_scores[t] - maxv);
        psum += p;
        for (int d = tid; d < kHD; d += kThreads)
            acc_o[d / kThreads] += p * __half2float(krow[d]);
    }
    // final normalize: denom = Σ p over slots + exp(sink - maxv).
    // p is uniform across the block (read from shared s_scores), so every
    // thread's psum equals the full Σp — the block reduction yields
    // blockDim.x · Σp and must be divided back down.
    const float denom = lsx_block_reduce_sum(psum, s_warp)
                            / static_cast<float>(kThreads)
                        + expf(sink[h] - maxv);
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < kHD; d += kThreads)
        orow[d] = __float2half(acc_o[d / kThreads] * inv_denom);
}

// ── FP32 GEMM: out[m] = W[m,k] @ x[k] (W FP32 row-major, x FP16) ────────────
// ── FP16 GEMM: out[m] = W[m,k] @ x[k] (W row-major) ──────────────────────────

// Clamp FP32→FP16 kernel: prevents overflow when dequanted weights exceed FP16 range
static __global__ void clamp_f32_to_f16_kernel(const float* __restrict__ src,
                                                __half* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = fmaxf(-65504.0f, fminf(65504.0f, v));
    dst[i] = __float2half(v);
}

// f8_gemm_1d: dequant F8_E4M3×E8M0 → FP16 (vectorized kernel from
// shared_experts.cu), then a CUDA_R_16F tensor-core GEMM with FP32
// accumulate/output and a clamped FP16 store. E4M3 × E8M0 products are
// exactly representable in FP16 for in-range scales, so weight precision is
// unchanged while the weight-materialization traffic halves and the GEMM
// runs on the 5th-gen Tensor Cores instead of the FP32 CUDA-core path.
// d_scratch needs (n halves + rows floats) — strictly less than the old
// (n + cols + rows) floats footprint.
static cudaError_t f8_gemm_1d(cudaStream_t stream,
                              const uint8_t* d_w, const uint8_t* d_scale,
                              int rows, int cols,
                              const __half* d_x, __half* d_out,
                              float* d_scratch)
{
    const int64_t n = static_cast<int64_t>(rows) * cols;
    __half* d_w_half = reinterpret_cast<__half*>(d_scratch);
    cudaError_t e = dequant_shared_f8_half(stream, d_w, n, d_scale, 0,
                                           d_w_half, n, cols);
    if (e != cudaSuccess) return e;
    float* d_out_f32 = reinterpret_cast<float*>(d_w_half + n);
    float alpha = 1.0f, beta = 0.0f;
    if (gemv::RunLxGemmVec(stream, d_w_half, d_x, 1, rows, cols,
                           gemv::LxDtype::kF16, gemv::LxDtype::kF16, alpha,
                           d_out_f32, gemv::LxDtype::kF32,
                           beta) != cudaSuccess)
        return cudaErrorUnknown;
    int block = 256;
    int grid = (rows + block - 1) / block;
    clamp_f32_to_f16_kernel<<<grid, block, 0, stream>>>(d_out_f32, d_out, rows);
    return cudaGetLastError();
}

// ── RMSNorm bridge (half → float → rmsnorm_float → half) ─────────────────────
// half_to_float_kernel / float_to_half_kernel are defined once in
// deepseek_moe_load.cuh (included above) and shared across this
// pipeline's .cu files.

static cudaError_t run_rmsnorm_attn(
    cudaStream_t stream, const __half* d_input, const __half* d_weight,
    __half* d_output, float* /*d_scratch*/, float* /*d_gamma_scratch*/,
    int dim, float eps)
{
    // Fused single-launch FP16 RMSNorm.
    rmsnorm_half_fused_kernel<<<1, 256, 0, stream>>>(d_input, d_weight,
                                                      d_output, dim, eps);
    return cudaGetLastError();
}

} // anonymous namespace

// ── Public: HC pre ───────────────────────────────────────────────────────────
cudaError_t run_hc_pre(cudaStream_t stream,
                       const float* d_state,          // [hc_mult·dim] FP32
                       const float* d_fn,              // [mix_hc, hc_mult·dim]
                       const float* d_scale,           // [3]
                       const float* d_base,            // [mix_hc]
                       float* d_single,                // [dim] FP32
                       float* d_post, float* d_comb,   // [hc_mult], [hc_mult·hc_mult]
                       float* d_pre,                   // [hc_mult]
                       float* d_scratch,               // ≥ (kHcMix+1)*kHcTiles floats
                       float rms_eps)                  // input RMSNorm eps
{
    constexpr int M = kHcMix;
    constexpr int H = kHcMult;
    dim3 grid(M + 1, kHcTiles);
    hc_mix_partials_kernel<<<grid, kHcPartThreads, 0, stream>>>(
        d_state, d_fn, M, kHcDim, kHcTiles, d_scratch);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    hc_pre_finalize_kernel<<<1, 256, 0, stream>>>(
        d_scratch, d_scale, d_base, M, H, kHcDim, kHcTiles,
        d_post, d_comb, d_pre, rms_eps);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    hc_combine_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
        d_state, d_pre, d_single);
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
                        float* d_single,               // [dim] FP32
                        float* d_scratch)              // ≥ (kHcMult+1)*kHcTiles + kHcMult floats
{
    constexpr int M = kHcMult;
    constexpr int H = kHcMult;
    float* const d_pre = d_scratch + (M + 1) * kHcTiles;
    dim3 grid(M + 1, kHcTiles);
    hc_head_partials_kernel<<<grid, kHcPartThreads, 0, stream>>>(
        d_state, d_fn, M, kHcDim, kHcTiles, d_scratch);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    hc_head_finalize_kernel<<<1, 256, 0, stream>>>(
        d_scratch, d_scale, d_base, M, H, kHcDim, kHcTiles, d_pre);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    hc_combine_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
        d_state, d_pre, d_single);
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
// GEMM -> olora; then wo_b @ olora -> the final [kHiddenDim] output).
// wo_a.weight [8192,4096] row-major, groups are contiguous 1024-row blocks.
// Previously hand-copied into run_csa_attention with its cudaGetLastError()/
// cublasStatus_t checks silently dropped (an "error handling desync" between
// the two copies) — extracting one checked implementation fixes that for
// both call sites.
//
// Blackwell perf: wo_a is dequanted ONCE to FP16 and all 8 group GEMMs run
// as a single cublasGemmStridedBatchedEx (16F inputs, 32F accumulate) —
// 3 launches instead of the old 24 (per-group dequant + convert + GEMM +
// clamp × 8), with proper tensor-core GEMM shapes.
static cudaError_t run_o_projection(
    cudaStream_t stream,
    const LayerWeights &lw,
    __half *d_o, __half *d_olora, __half *d_out,
    float *d_dequant)
{
    int64_t n = static_cast<int64_t>(lw.wo_a_rows) * lw.wo_a_cols;
    __half* d_w_half = reinterpret_cast<__half*>(d_dequant);
    cudaError_t err = dequant_shared_f8_half(stream, lw.d_wo_a, n,
        lw.d_wo_a_scales, 0, d_w_half, n, lw.wo_a_cols);
    if (err != cudaSuccess) return err;

    // O-projection: 8 batched FP16 TC GEMMs ([1024×4096] × [4096] each) →
    // FP32 olora scratch, then one clamped FP16 store.
    const int gd = (kAttnHeads / kOGroups) * kHeadDim;  // 4096
    float* d_olora_f32 = reinterpret_cast<float*>(d_w_half + n);
    const float alpha = 1.0f, beta = 0.0f;
    // Device-independent per-group GEMV (numeric-contract 1.1.1); the former
    // strided-batched cuBLAS call selected its algorithm by device.
    for (int b = 0; b < kOGroups; ++b) {
        const __half *wb = d_w_half + static_cast<int64_t>(b) * kOloraRank * gd;
        const __half *xb = d_o + static_cast<int64_t>(b) * gd;
        float *yb = d_olora_f32 + static_cast<int64_t>(b) * kOloraRank;
        const cudaError_t e = gemv::RunLxGemmVec(
            stream, wb, xb, 1, kOloraRank, gd, gemv::LxDtype::kF16,
            gemv::LxDtype::kF16, alpha, yb, gemv::LxDtype::kF32, beta);
        if (e != cudaSuccess) return e;
    }
    {
        int block = 256;
        int g3 = (kOGroups * kOloraRank + block - 1) / block;
        clamp_f32_to_f16_kernel<<<g3, block, 0, stream>>>(
            d_olora_f32, d_olora, kOGroups * kOloraRank);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // out = wo_b @ olora → [kHiddenDim]
    return f8_gemm_1d(stream, lw.d_wo_b, lw.d_wo_b_scales,
                      lw.wo_b_rows, lw.wo_b_cols, d_olora, d_out, d_dequant);
}

cudaError_t run_mla_attention(
    cudaStream_t stream,
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
    cudaError_t e = f8_gemm_1d(stream, lw.d_wq_a, lw.d_wq_a_scales,
                               lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    // qr = q_norm(qr)
    e = run_rmsnorm_attn(stream, d_qr, lw.d_q_norm, d_qr2,
                         d_norm_scratch, d_norm_gamma_scratch, kQloraRank, kRmsEps);
    if (e != cudaSuccess) return e;
    // q = wq_b @ qr2 → [32768]
    e = f8_gemm_1d(stream, lw.d_wq_b, lw.d_wq_b_scales,
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
    e = f8_gemm_1d(stream, lw.d_wkv, lw.d_wkv_scales,
                   lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_kv, lw.d_kv_norm, d_kv2,
                         d_norm_scratch, d_norm_gamma_scratch, kHeadDim, kRmsEps);
    if (e != cudaSuccess) return e;
    {
        int total = kRopeHeadDim / 2;
        int block = 32;   // total == 32: one full warp, no idle lanes
        rope_apply_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
            d_kv2, 1, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta,
            lw.compress_ratio > 0 ? 1 : 0);
    }

    // write kv into circular cache slot (with FP8 act_quant on non-RoPE dims)
    {
        // Apply act_quant(kv[..., :-rd], 64, inplace=True) — FP8 quantize+dequant
        int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
        kv_act_quant_kernel<<<n_blocks_q, 64, 0, stream>>>(d_kv2, kHeadDim - kRopeHeadDim);
    }
    const int slot = start_pos % kAttnWindow;
    e = cudaMemcpyAsync(d_kv_cache_layer + slot * kHeadDim, d_kv2,
                        static_cast<size_t>(kHeadDim) * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    // attention
    {
        float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
        const int n_attend = min(start_pos + 1, kAttnWindow);
        const size_t smem = static_cast<size_t>(n_attend + 8) * sizeof(float);
        mla_attn_kernel<<<kAttnHeads, 256, smem, stream>>>(
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
    e = run_o_projection(stream, lw, d_o, d_olora, d_out, d_dequant);
    if (e != cudaSuccess) return e;

    return cudaSuccess;
}

// ── CSA: sparse attention kernel (window + compressed KV) ───────────────────
// One block per head. Attends to window positions AND compressed KV positions.
// The kv_cache is laid out as: [window_slots, head_dim] followed by
// [compressed_slots, head_dim] in a single contiguous buffer.
// Same Blackwell optimizations as mla_attn_kernel: single QK pass with cached
// scores in dynamic shared memory + warp-shuffle reductions.
__global__ void __launch_bounds__(256)
csa_sparse_attn_kernel(
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
    constexpr int kThreads = 256;
    constexpr int kHD = kHeadDim;      // 512

    const int h = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_smem[];
    float* s_scores = s_smem;             // [n_attend + n_compressed]
    float* s_warp = s_scores + (n_attend + n_compressed);

    const __half* qrow = q + h * kHD;
    const int total = n_attend + n_compressed;

    // Pass 1: scores (cached, scaled)
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
        const float tot = lsx_block_reduce_sum(partial, s_warp);
        if (tid == 0) s_scores[t] = tot * scale;
    }
    __syncthreads();

    // Block max over cached scores + sink.
    float mx = -1e30f;
    for (int t = 0; t < total; ++t) mx = fmaxf(mx, s_scores[t]);
    mx = fmaxf(mx, sink[h]);
    const float maxv = mx;

    // Pass 2: probs + weighted accumulation from cached scores.
    float acc_o[kHD / kThreads];  // 2 slices per thread
    for (int i = 0; i < kHD / kThreads; ++i) acc_o[i] = 0.0f;
    float psum = 0.0f;
    for (int t = 0; t < total; ++t) {
        int slot;
        if (t < n_attend)
            slot = (start_pos - t + window) % window;
        else
            slot = window + (t - n_attend);
        const __half* krow = kv_cache + slot * kHD;
        const float p = expf(s_scores[t] - maxv);
        psum += p;
        for (int d = tid; d < kHD; d += kThreads)
            acc_o[d / kThreads] += p * __half2float(krow[d]);
    }
    // Sink contribution (denominator only, no output contribution).
    // p is uniform across the block — divide the block reduction by kThreads
    // (see mla_attn_kernel).
    const float denom = lsx_block_reduce_sum(psum, s_warp)
                            / static_cast<float>(kThreads)
                        + expf(sink[h] - maxv);
    const float inv = 1.0f / (denom + 1e-9f);
    __half* orow = o + h * kHD;
    for (int d = tid; d < kHD; d += kThreads)
        orow[d] = __float2half(acc_o[d / kThreads] * inv);
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
    cudaStream_t stream,
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
    cudaError_t e = f8_gemm_1d(stream, lw.d_wq_a, lw.d_wq_a_scales,
                               lw.wq_a_rows, lw.wq_a_cols, d_x, d_qr, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_qr, lw.d_q_norm, d_qr2,
                         d_norm_scratch, d_norm_gamma_scratch, kQloraRank, kRmsEps);
    if (e != cudaSuccess) return e;
    e = f8_gemm_1d(stream, lw.d_wq_b, lw.d_wq_b_scales,
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
    e = f8_gemm_1d(stream, lw.d_wkv, lw.d_wkv_scales,
                   lw.wkv_rows, lw.wkv_cols, d_x, d_kv, d_dequant);
    if (e != cudaSuccess) return e;
    e = run_rmsnorm_attn(stream, d_kv, lw.d_kv_norm, d_kv2,
                         d_norm_scratch, d_norm_gamma_scratch, kHeadDim, kRmsEps);
    if (e != cudaSuccess) return e;
    {
        int total = kRopeHeadDim / 2;
        rope_apply_kernel<<<(total + 31) / 32, 32, 0, stream>>>(
            d_kv2, 1, kHeadDim, kHeadDim - kRopeHeadDim,
            kRopeHeadDim, static_cast<float>(start_pos), lw.rope_theta, 1);
    }

    // Store KV in sliding-window cache (with FP8 act_quant on non-RoPE dims)
    {
        int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
        kv_act_quant_kernel<<<n_blocks_q, 64, 0, stream>>>(d_kv2, kHeadDim - kRopeHeadDim);
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
                    if (gemv::RunLxGemmVec(
                    stream, lw.d_comp_wkv, d_x, 1, comp_out, kHiddenDim,
                    gemv::LxDtype::kF16, gemv::LxDtype::kF16, alpha, d_qr2,
                    gemv::LxDtype::kF16, beta) != cudaSuccess) {
                log::error("[csa] compressor wkv GEMM failed");
                return cudaErrorInvalidValue;
            }
            if (gemv::RunLxGemmVec(
                    stream, lw.d_comp_wgate, d_x, 1, comp_out, kHiddenDim,
                    gemv::LxDtype::kF16, gemv::LxDtype::kF16, alpha, d_qr,
                    gemv::LxDtype::kF16, beta) != cudaSuccess) {
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
                rope_apply_kernel<<<(total + 31) / 32, 32, 0, stream>>>(
                    d_compressed_slot, 1, kHeadDim, kHeadDim - kRopeHeadDim,
                    kRopeHeadDim, static_cast<float>(comp_pos), lw.rope_theta, 1);
            }
            // Main compressor (rotate=False) applies FP8 act_quant to non-RoPE dims
            // (reference model.py:372 act_quant(kv[..., :-rd], 64, ..., True)).
            {
                int n_blocks_q = (kHeadDim - kRopeHeadDim + 63) / 64;
                kv_act_quant_kernel<<<n_blocks_q, 64, 0, stream>>>(d_compressed_slot, kHeadDim - kRopeHeadDim);
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
            const size_t smem = static_cast<size_t>(n_attend + n_compressed + 8)
                                * sizeof(float);
            csa_sparse_attn_kernel<<<kAttnHeads, 256, smem, stream>>>(
                d_q, d_kv_cache_layer, lw.d_attn_sink, d_o,
                start_pos, kAttnWindow, n_attend, n_compressed, scale);
        } else {
            const size_t smem = static_cast<size_t>(n_attend + 8) * sizeof(float);
            mla_attn_kernel<<<kAttnHeads, 256, smem, stream>>>(
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
    e = run_o_projection(stream, lw, d_o, d_olora, d_out, d_dequant);
    return e;
}

} // namespace gpu
} // namespace deepseek_pro
} // namespace lsxcommon
