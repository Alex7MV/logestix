// GLM-5.3-Flash CUDA kernels: KDA (conv1d + per-channel-decay gated delta),
// router top-8 (sigmoid noaux_tc), and the single-stream generation driver.
// Per-channel decay mirrors the FLA kimi-delta semantics: decay applies to
// the KEY dim (l) of the state column, unlike qwen GDN's scalar head decay.
#include "lsxcommon/glm_moe_flash_inference.h"

#include "glm_moe_flash_device_math.h"
#include "lsx_kernels_quant.cuh"
#include "glm_moe_flash_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "deepseek_moe_load.cuh"  // run_hc_pre/post/head, run_expand_hc (lsxcommon_cu)
#include "lsx_kernels_act.cuh"
#include "lsxcommon/qwen_moe_exp_inference.h"  // RunExpQsaAttention reuse

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace lsxcommon {
namespace glm_moe_flash {
namespace {
__device__ int flash_dbg_on = 0;

// Device-side buffer probe (bit of first N elems + NaN count) — temporary
// debugging aid, env-gated via a device flag set by the driver.
__global__ void flash_dbg_probe_kernel(const void *buf, int kind,
                                       int n_first, int64_t total)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // kind: 0=half, 1=float
    int nan = 0, inf = 0;
    float v0 = 0.f, v1 = 0.f;
    if (kind == 0) {
        const __half *p = static_cast<const __half *>(buf);
        v0 = __half2float(p[0]);
        v1 = __half2float(p[1]);
        for (int64_t i = 0; i < total; ++i) {
            const float f = __half2float(p[i]);
            if (isnan(f)) ++nan;
            if (isinf(f)) ++inf;
        }
    } else {
        const float *p = static_cast<const float *>(buf);
        v0 = p[0];
        v1 = p[1];
        for (int64_t i = 0; i < total; ++i) {
            if (isnan(p[i])) ++nan;
            if (isinf(p[i])) ++inf;
        }
    }
    if (flash_dbg_on)
        printf("[probe] kind=%d v0=%.6f v1=%.6f nan=%lld/%lld inf=%lld\n",
               kind, v0, v1, static_cast<long long>(nan),
               static_cast<long long>(total), static_cast<long long>(inf));
}


FLASH_HD float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + flash_expf_glibc(-x));
}

// BF16 (low 16 bits) -> f32: exact widening, no rounding.
FLASH_HD float bf16_bits_to_f32(uint32_t low16)
{
    const uint32_t bits = low16 << 16;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

} // namespace

// ── Conv1d test twin ─────────────────────────────────────────────────────
__global__ void flash_conv1d_f32_kernel(const float *__restrict__ x,
                                        const float *__restrict__ w,
                                        float *__restrict__ y, int channels,
                                        int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k)
        acc += w[c * kernel + k] * x[k * channels + c];
    y[c] = acc;
}

cudaError_t RunFlashConv1dF32(cudaStream_t stream, const float *x,
                              const float *w, float *y, int channels,
                              int kernel)
{
    const int threads = 128;
    const int blocks = (channels + threads - 1) / threads;
    flash_conv1d_f32_kernel<<<blocks, threads, 0, stream>>>(x, w, y, channels,
                                                            kernel);
    return cudaGetLastError();
}

// ── Gated delta test twin (per-channel decay) ─────────────────────────────
// Block per (t, v_head, d-column plane); thread per value-dim d. Registers
// hold the column S[l, d] across the time loop (l in 0..dim).
__global__ void flash_gated_delta_f32_kernel(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ beta,
    const float *__restrict__ gdecay, float *__restrict__ state_out,
    float *__restrict__ out, int seq, int heads, int dim)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= dim) return;

    float col[256];
    for (int l = 0; l < dim; ++l) col[l] = 0.0f;

    for (int tt = 0; tt <= t; ++tt) {
        // Per-channel decay along the key dim l.
        for (int l = 0; l < dim; ++l)
            col[l] *= gdecay[(tt * heads + h) * dim + l];
        float kv = 0.0f;
        for (int l = 0; l < dim; ++l) kv += col[l] * k[(tt * heads + h) * dim + l];
        const float delta = beta[tt * heads + h] * (v[(tt * heads + h) * dim + d] - kv);
        for (int l = 0; l < dim; ++l)
            col[l] += k[(tt * heads + h) * dim + l] * delta;
        float acc = 0.0f;
        for (int l = 0; l < dim; ++l) acc += col[l] * q[(tt * heads + h) * dim + l];
        out[(tt * heads + h) * dim + d] = acc;
    }

    // Write the final row's state: [H, D(l), D(d)].
    if (t == seq - 1) {
        for (int l = 0; l < dim; ++l)
            state_out[(h * dim + l) * dim + d] = col[l];
    }
}

cudaError_t RunFlashGatedDeltaF32(cudaStream_t stream, const float *q,
                                  const float *k, const float *v,
                                  const float *beta, const float *gdecay,
                                  int seq, int heads, int dim,
                                  float *state_out, float *out)
{
    if (dim > 256) return cudaErrorInvalidValue;
    const dim3 grid(seq, heads);
    int threads = 1;
    while (threads < dim) threads <<= 1;
    flash_gated_delta_f32_kernel<<<grid, threads, 0, stream>>>(
        q, k, v, beta, gdecay, state_out, out, seq, heads, dim);
    return cudaGetLastError();
}

// ── Router top-8 ─────────────────────────────────────────────────────────
// One block per token. HF Glm5NextTextTopkRouter semantics: choice score =
// sigmoid(dot(gate_e, x)) + bias_e (FP32 over BF16-decoded values); top-8 by
// choice score (tie-break lower expert index); coeff from the RAW sigmoid;
// renorm + scale 2.5. RMW-free per-slot writes: each
// selection round writes its slot once.
__global__ void flash_router_top8_kernel(
    const uint8_t *__restrict__ gate_bf16, const float *__restrict__ bias,
    const uint8_t *__restrict__ x_bf16, qwen_moe::QwenMoeRoutedSlot *__restrict__ slots,
    int hidden, int experts)
{
    const int t = blockIdx.x;
    extern __shared__ float s_scores[];  // [experts]
    // Score pass: one thread per expert.
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        float dot = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            const uint32_t gb =
                (static_cast<uint32_t>(gate_bf16[e * hidden * 2 + h * 2])
                 | (static_cast<uint32_t>(gate_bf16[e * hidden * 2 + h * 2 + 1])
                    << 8))
                << 16;
            const uint32_t xb =
                (static_cast<uint32_t>(x_bf16[t * hidden * 2 + h * 2])
                 | (static_cast<uint32_t>(x_bf16[t * hidden * 2 + h * 2 + 1])
                    << 8))
                << 16;
            float g, xv;
            memcpy(&g, &gb, 4);
            memcpy(&xv, &xb, 4);
            dot += g * xv;
        }
        s_scores[e] = sigmoid_dev(dot);
        s_scores[experts + e] = s_scores[e] + bias[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    // Thread-0 selection over the staged choice scores; HF semantics:
    // selection on sigmoid+bias, coeff from the RAW sigmoid score.
    constexpr int kTopK = 8;
    for (int kk = 0; kk < kTopK; ++kk) {
        int best = -1;
        float best_s = -1e30f;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int j = 0; j < kk; ++j)
                if (slots[t * kTopK + j].eid == e) taken = true;
            if (taken) continue;
            if (s_scores[experts + e] > best_s) {
                best_s = s_scores[experts + e];
                best = e;
            }
        }
        slots[t * kTopK + kk].eid = best;
        slots[t * kTopK + kk].coeff = s_scores[best];
    }
    float sum = 0.0f;
    for (int kk = 0; kk < kTopK; ++kk) sum += slots[t * kTopK + kk].coeff;
    for (int kk = 0; kk < kTopK; ++kk)
        slots[t * kTopK + kk].coeff =
            slots[t * kTopK + kk].coeff / sum * 2.5f;
}

cudaError_t RunFlashRouterTop8(cudaStream_t stream, const uint8_t *d_gate,
                               const float *d_bias, const uint8_t *d_x,
                               int tok, int hidden, int experts,
                               qwen_moe::QwenMoeRoutedSlot *d_slots)
{
    const int threads = experts <= 1024 ? experts : 1024;
    const size_t scratch = static_cast<size_t>(experts) * 2 * sizeof(float);
    flash_router_top8_kernel<<<tok, threads, scratch, stream>>>(
        d_gate, d_bias, d_x, d_slots, hidden, experts);
    return cudaGetLastError();
}

// ── Production kernels (decode path, seq == 1) ───────────────────────────

namespace {

// FP16 GEMV: y[n] = W[n, k] @ x[k], W row-major FP16, FP32 accumulate,
// FP16 store. One warp per row.
__global__ void flash_gemv_f16_kernel(const __half *__restrict__ w,
                                      const __half *__restrict__ x,
                                      __half *__restrict__ y, int k_dim,
                                      int n_dim)
{
    const int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (n >= n_dim) return;
    const int lane = threadIdx.x & 31;
    const __half *row = w + static_cast<int64_t>(n) * k_dim;
    float part = 0.f;
    for (int k = lane; k < k_dim; k += 32)
        part += __half2float(row[k]) * __half2float(x[k]);
    for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off);
    if (lane == 0) y[n] = __float2half(part);
}

cudaError_t FlashGemmV16(cudaStream_t stream, const __half *w,
                         const __half *x, __half *y, int k_dim, int n_dim)
{
    static const bool dbg = std::getenv("LSX_FLASH_GEMV_DEBUG") != nullptr;
    if (dbg)
        std::fprintf(stderr, "[gemv] w=%p x=%p y=%p k=%d n=%d\n", (const void *)w,
                     (const void *)x, (void *)y, k_dim, n_dim);
    const int warps = 8;
    const int blocks = (n_dim + warps - 1) / warps;
    flash_gemv_f16_kernel<<<blocks, warps * 32, 0, stream>>>(w, x, y, k_dim,
                                                             n_dim);
    return cudaGetLastError();
}

// Depthwise causal conv1d, seq=1: y[c] = sum_k w[c,k] * window[k, c] where
// window = [(K-1) history rows (oldest..newest), current row x].
// history: [(K-1), C] FP16 rows (oldest first), x: [C].
__global__ void flash_conv_decode_kernel(const __half *__restrict__ hist,
                                         const __half *__restrict__ x,
                                         const __half *__restrict__ w,
                                         __half *__restrict__ y, int channels,
                                         int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float acc = 0.0f;
    for (int k = 0; k < kernel - 1; ++k)
        acc += __half2float(w[c * kernel + k])
             * __half2float(hist[k * channels + c]);
    acc += __half2float(w[c * kernel + (kernel - 1)])
         * __half2float(x[c]);
    y[c] = __float2half(acc);
}

// SiLU in place: x * sigmoid(x) with the glibc exp replica.
__global__ void flash_silu_kernel(__half *__restrict__ x, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(x[i]);
    x[i] = __float2half(v / (1.0f + flash_expf_glibc(-v)));
}

// L2-normalize one head row (dim elements) in place; q gets extra scale.
__global__ void flash_l2norm_kernel(__half *row, int dim, float extra_scale)
{
    extern __shared__ float red[];
    // One block per head: block h normalizes row[h*dim .. h*dim+dim).
    row += static_cast<int64_t>(blockIdx.x) * dim;
    const int l = threadIdx.x;
    float v = 0.0f;
    if (l < dim) v = __half2float(row[l]);
    red[l] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        __syncthreads();
    }
    if (l == 0) red[0] = sqrtf(red[0] + 1e-6f);
    __syncthreads();
    if (l < dim)
        row[l] = __float2half(__half2float(row[l]) / red[0] * extra_scale);
}

// Gating prep: one thread per (h, l) over S_out = 3*8192? KDA uses
// decay over key dim only: threads = heads*dim (8192).
//   beta[h]    = sigmoid(b_proj[h])                       (per head)
//   dt[h,l]    = f_b[h,l] + dt_bias[h,l]
//   gdecay[h,l]= exp(-exp(A_log[h]) * softplus(dt[h,l]))
__global__ void flash_kda_gating_kernel(const __half *__restrict__ b_proj_out,
                                        const __half *__restrict__ f_out,
                                        const float *__restrict__ A_log,
                                        const float *__restrict__ dt_bias,
                                        float *__restrict__ beta,
                                        float *__restrict__ gdecay, int heads,
                                        int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * dim) return;
    const int h = idx / dim;
    const int l = idx % dim;
    if (l == 0) beta[h] = sigmoid_dev(__half2float(b_proj_out[h]));
    const float dt =
        __half2float(f_out[idx]) + dt_bias[idx];
    // HF Glm5NextTextForgetGate: safe lower bound path (linear_lower_bound
    // = -5.0): g_log = bound * sigmoid(exp(A_log) * g); decay = exp(g_log).
    constexpr float kGateLowerBound = -5.0f;
    gdecay[idx] = flash_expf_glibc(
        kGateLowerBound
        * sigmoid_dev(flash_expf_glibc(A_log[h]) * dt));
}

// Per-channel-decay delta step for ONE token (decode): block per head,
// thread per value-dim d; col[0..dim) in registers.
//   col[l] *= gdecay[h,l]
//   kv  = sum_l col[l]*k[l]
//   delta = beta[h]*(v[d] - kv)
//   col[l] += k[l]*delta
//   out[d] = sum_l col[l]*q[l]
// State seeded from / written back to d_state[h*dim*dim + l*dim + d].
__global__ void flash_kda_step_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, const float *__restrict__ beta,
    const float *__restrict__ gdecay, float *__restrict__ state,
    float *__restrict__ out, int heads, int dim)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= dim) return;
    const int64_t base = static_cast<int64_t>(h) * dim * dim;

    float col[128];
    for (int l = 0; l < dim; ++l)
        col[l] = state[base + static_cast<int64_t>(l) * dim + d]
               * gdecay[h * dim + l];
    float kv = 0.0f;
    for (int l = 0; l < dim; ++l)
        kv += col[l] * __half2float(k[h * dim + l]);
    const float delta =
        beta[h] * (__half2float(v[h * dim + d]) - kv);
    for (int l = 0; l < dim; ++l)
        col[l] += __half2float(k[h * dim + l]) * delta;
    for (int l = 0; l < dim; ++l)
        state[base + static_cast<int64_t>(l) * dim + d] = col[l];
    float acc = 0.0f;
    for (int l = 0; l < dim; ++l)
        acc += col[l] * __half2float(q[h * dim + l]);
    if (flash_dbg_on && h == 0 && d == 0 && blockIdx.x == 0)
        printf("[step-kernel] acc=%.6f col0=%.6f state0=%.6f gd0=%.6f q0=%.6f\n",
               acc, col[0], state[0], gdecay[0], __half2float(q[0]));
    out[h * dim + d] = acc;
}

// RMSNormGated per head: core[h,dim], gate[h,dim], norm weight [dim].
// out[h,l] = RMSNorm(core)[l] * sigmoid(gate[l])? — GLM KDA o_norm gate
// follows the qwen RMSNormGated: normed = core * rsqrt(mean(core^2)+eps);
// out = normed * sigmoid? — the qwen kernel uses the gate directly. We use
// out = normed[l] * gate[l] (gate post-sigmoid from the g projections).
__global__ void flash_rmsnorm_gated_kernel(const float *__restrict__ core,
                                           const __half *__restrict__ gate,
                                           const __half *__restrict__ norm_w,
                                           __half *__restrict__ out, int heads,
                                           int dim, float eps)
{
    extern __shared__ float red2[];
    const int h = blockIdx.x;
    const int l = threadIdx.x;
    float v = 0.0f;
    if (l < dim) v = core[h * dim + l];
    red2[l] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (l < s) red2[l] += red2[l + s];
        __syncthreads();
    }
    if (l == 0) red2[0] = rsqrtf(red2[0] / static_cast<float>(dim) + eps);
    __syncthreads();
    if (l < dim)
        out[h * dim + l] = __float2half(
            v * red2[0] * __half2float(gate[h * dim + l])
            * __half2float(norm_w[l]));
    if (flash_dbg_on && h == 0 && l == 0)
        printf("[gated] v0=%.8f inv=%.4f gate0=%.6f w0=%.6f out0=%.8f\n",
               v, red2[0], __half2float(gate[0]), __half2float(norm_w[0]),
               __half2float(out[0]));
}

// conv history shift: hist[0..K-2] rows; slide up, write current row at
// the newest slot. One thread per channel.
__global__ void flash_conv_shift_kernel(__half *__restrict__ hist,
                                        const __half *__restrict__ cur,
                                        int channels, int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int k = 0; k < kernel - 2; ++k)
        hist[k * channels + c] = hist[(k + 1) * channels + c];
    hist[(kernel - 2) * channels + c] = cur[c];
}

// fp16 vec add: y += x.
__global__ void flash_add_kernel(const __half *__restrict__ x,
                                 __half *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(__half2float(y[i]) + __half2float(x[i]));
}

// repeat hidden -> hc streams (FP16 x4)
__global__ void flash_repeat_hc_kernel(const __half *__restrict__ in, int H,
                                       int hc, __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    const __half v = in[i];
    for (int c = 0; c < hc; ++c) out[c * H + i] = v;
}

// Gating activation over [heads*dim] in place: sigmoid (default) or silu.
__global__ void flash_gate_act_kernel(__half *__restrict__ g, int n,
                                      bool silu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __half2float(g[i]);
        g[i] = silu
                   ? __float2half(v / (1.0f + flash_expf_glibc(-v)))
                   : __float2half(sigmoid_dev(v));
    }
}

// Conv history shift variant taking separate q/k/v raw rows. History
// layout: three contiguous planes of (kernel-1)*KD halves (q|k|v), each
// plane row-major [(kernel-1), KD] rows oldest..newest — matching the
// [k*channels + c] addressing of flash_conv_decode_kernel (channels = KD
// per plane), so decode must be launched with the plane base pointer.
__global__ void flash_conv_shift_kernel2(__half *__restrict__ hist,
                                         const __half *__restrict__ q,
                                         const __half *__restrict__ k,
                                         const __half *__restrict__ v,
                                         int channels, int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int KD = channels / 3;
    const int plane = c / KD;
    const int cc = c % KD;
    __half *base = hist + plane * (kernel - 1) * KD;
    const __half *cur = plane == 0 ? q : (plane == 1 ? k : v);
    for (int kk = 0; kk < kernel - 2; ++kk)
        base[kk * KD + cc] = base[(kk + 1) * KD + cc];
    base[(kernel - 2) * KD + cc] = cur[cc];
}

// acc[i] += coeff * y[i].
__global__ void flash_coeff_add_kernel(const __half *__restrict__ y,
                                       float coeff, float *__restrict__ acc,
                                       int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += coeff * __half2float(y[i]);
}

} // namespace

cudaError_t FlashKdaStateAlloc(FlashKdaState &st, int64_t max_ctx)
{
    (void) max_ctx;
    constexpr int kHeads = 64, kDim = 128, kConv = 4, kChannels = 3 * 8192;
    cudaError_t e = cudaMalloc(&st.d_state,
                               static_cast<size_t>(kHeads * kDim * kDim) * 4);
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&st.d_conv_hist,
                   static_cast<size_t>(kConv - 1) * kChannels * 2);
    if (e != cudaSuccess) {
        cudaFree(st.d_state);
        return e;
    }
    st.capacity_ctx = max_ctx;
    return cudaSuccess;
}

void FlashKdaStateFree(FlashKdaState &st)
{
    cudaFree(st.d_state);
    cudaFree(st.d_conv_hist);
    st = {};
}

cudaError_t RunFlashKdaDecode(cudaStream_t stream, FlashKdaState &st,
                              const FlashLayerWeights &lw, int hidden,
                              int heads, int dim, int conv_kernel,
                              const __half *d_x, __half *d_out,
                              void *d_scratch)
{
    const int KD = heads * dim;  // 8192
    __half *d_q = static_cast<__half *>(d_scratch);
    __half *d_k = d_q + KD;
    __half *d_v = d_k + KD;
    __half *d_qc = d_v + KD;            // [KD] post-conv q
    __half *d_kc = d_qc + KD;           // post-conv k
    __half *d_vc = d_kc + KD;           // post-conv v
    __half *d_b = d_vc + KD;            // [heads]
    __half *d_f = d_b + heads;          // [heads*dim]
    __half *d_g = d_f + heads * dim;    // [heads*dim]
    float *d_beta = reinterpret_cast<float *>(d_g + heads * dim);
    float *d_gdecay = d_beta + heads;
    float *d_attn_f = d_gdecay + heads * dim;  // [KD] FP32 delta out
    __half *d_gated =
        reinterpret_cast<__half *>(d_attn_f + KD);  // [KD]

    cudaError_t e;
    // 1. q/k/v GEMVs (raw projections).
    if ((e = lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.kda_q.d_packed,
                                    lw.kda_q.d_scales, lw.kda_q.global_scale,
                                    d_q, 1, hidden, KD, false))
        != cudaSuccess) return e;
    if ((e = lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.kda_k.d_packed,
                                    lw.kda_k.d_scales, lw.kda_k.global_scale,
                                    d_k, 1, hidden, KD, false))
        != cudaSuccess) return e;
    if ((e = lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.kda_v.d_packed,
                                    lw.kda_v.d_scales, lw.kda_v.global_scale,
                                    d_v, 1, hidden, KD, false))
        != cudaSuccess) return e;

    // 2. Depthwise causal conv over [hist(K-1) prev rows oldest..newest,
    //    current raw row] — the window EXCLUDES cur from hist; the shift
    //    appending cur happens AFTER the conv (below).
    {
        const int threads = 256;
        const int plane = (conv_kernel - 1) * KD;  // (K-1) rows x KD halves
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist, d_q, lw.d_kda_q_conv, d_qc, KD, conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist + plane, d_k, lw.d_kda_k_conv, d_kc, KD,
            conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist + 2 * plane, d_v, lw.d_kda_v_conv, d_vc, KD,
            conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 3. Shift the conv history: slide up, append the current raw rows.
    //    hist layout: [(K-1) rows, 3*KD channels] rows oldest..newest,
    //    each row = [q_raw, k_raw, v_raw].
    {
        const int threads = 256;
        const int C = 3 * KD;
        flash_conv_shift_kernel2<<<(C + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist, d_q, d_k, d_v, C, conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 3b. Post-conv activation (FLA short-conv applies silu AFTER the
    // depthwise conv, BEFORE the qk l2norm — HF causal_conv1d_update takes
    // activation=silu inside the conv; the recurrent kernel l2norms the
    // activated q/k). Toggle: default ON.
    {
        static const bool conv_silu = [] {
            const char *e = std::getenv("LSX_FLASH_CONV_SILU");
            return !e || std::string_view(e) != "0";
        }();
        if (conv_silu) {
            flash_silu_kernel<<<(3 * KD + 255) / 256, 256, 0, stream>>>(
                d_qc, 3 * KD);
            if ((e = cudaGetLastError()) != cudaSuccess) return e;
        }
    }

    // 4. L2-normalize q, k per head (q scaled 1/sqrt(dim)).
    {
        const int threads = 128;  // == dim
        static const float q_scale = [] {
            const char *e = std::getenv("LSX_FLASH_QSCALE");
            return e ? std::strtof(e, nullptr) : (1.0f / sqrtf(128.0f));
        }();
        flash_l2norm_kernel<<<heads, threads, threads * sizeof(float), stream>>>(
            d_qc, dim, q_scale);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_l2norm_kernel<<<heads, threads, threads * sizeof(float), stream>>>(
            d_kc, dim, 1.0f);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 5. Gating: beta per head; per-channel decay from dt = f(x) + bias.
    if ((e = FlashGemmV16(stream, lw.d_b_proj, d_x, d_b, hidden, heads))
        != cudaSuccess) return e;
    {
        // dt = f_b(f_a(x)): two chained GEMVs through the low-rank 128.
        __half *d_fmid = d_gated;  // reuse gated scratch [128] mid buffer
        if ((e = FlashGemmV16(stream, lw.d_f_a, d_x, d_fmid, hidden, 128))
            != cudaSuccess) return e;
        if ((e = FlashGemmV16(stream, lw.d_f_b, d_fmid, d_f, 128, KD))
            != cudaSuccess) return e;
        // gate g = g_b(g_a(x)) [KD]
        if ((e = FlashGemmV16(stream, lw.d_g_a, d_x, d_fmid, hidden, 128))
            != cudaSuccess) return e;
        if ((e = FlashGemmV16(stream, lw.d_g_b, d_fmid, d_g, 128, KD))
            != cudaSuccess) return e;
    }
    {
        const int threads = 256;
        flash_kda_gating_kernel<<<(heads * dim + threads - 1) / threads,
                                  threads, 0, stream>>>(
            d_b, d_f, lw.d_A_log, lw.d_dt_bias, d_beta, d_gdecay, heads, dim);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 6. Delta step (state seeded/written back in d_state).
    {
        flash_kda_step_kernel<<<heads, dim, 0, stream>>>(
            d_qc, d_kc, d_vc, d_beta, d_gdecay, st.d_state, d_attn_f, heads,
            dim);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 7. RMSNormGated(core=attn, gate=act(g), o_norm).
    static const bool gate_silu = [] {
        const char *e = std::getenv("LSX_FLASH_GATE");
        return e && std::string_view(e) == "silu";
    }();
    {
        const int threads = 128;
        flash_gate_act_kernel<<<(heads * dim + 255) / 256, 256, 0, stream>>>(
            d_g, heads * dim, gate_silu);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_rmsnorm_gated_kernel<<<heads, threads, threads * sizeof(float),
                                     stream>>>(
            d_attn_f, d_g, lw.d_o_norm, d_gated, heads, dim,
                       1e-5f);  // RMSNormGated eps = config.rms_norm_eps
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 8. o_proj: [H, KD] @ gated -> d_out.
    if ((e = lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_gated, lw.kda_o.d_packed,
                                    lw.kda_o.d_scales, lw.kda_o.global_scale,
                                    d_out, 1, KD, hidden, false))
        != cudaSuccess) return e;
    return cudaSuccess;
}

// ── MoE clamped-swiglu routed chain ──────────────────────────────────────


// ── DSA (deepseek_sparse_attention) decode path ──────────────────────────
namespace {

// Weighted RMSNorm over one n-length row (any n via stride): out =
// x * rsqrt(mean(x^2)+eps) * w. One block; shared reduction over blockDim,
// grid-stride accumulation for n > blockDim.
__global__ void flash_rmsnorm_weighted_kernel(const __half *__restrict__ x,
                                              const __half *__restrict__ w,
                                              __half *__restrict__ out, int n,
                                              float eps)
{
    extern __shared__ float red[];
    const int l = threadIdx.x;
    float ssq = 0.0f;
    for (int i = l; i < n; i += blockDim.x)
        ssq += __half2float(x[i]) * __half2float(x[i]);
    red[l] = ssq;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (l < s2) red[l] += red[l + s2];
        __syncthreads();
    }
    if (l == 0)
        red[0] = rsqrtf(red[0] / static_cast<float>(n) + eps);
    __syncthreads();
    const float inv = red[0];
    for (int i = l; i < n; i += blockDim.x)
        out[i] = __float2half(__half2float(x[i]) * inv * __half2float(w[i]));
    if (flash_dbg_on && threadIdx.x == 0 && n == 4096)
        printf("[norm] inv=%.6f out0=%.6f x0=%.6f\n", inv,
               __half2float(out[0]), __half2float(x[0]));
}

// BF16-row GEMV: y[n] = W[n,k] @ x[k], W row-major BF16 bits, FP32 accum.
__global__ void flash_gemv_bf16_kernel(const uint16_t *__restrict__ w,
                                       const __half *__restrict__ x,
                                       __half *__restrict__ y, int k_dim,
                                       int n_dim)
{
    const int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (n >= n_dim) return;
    const int lane = threadIdx.x & 31;
    const uint16_t *row = w + static_cast<int64_t>(n) * k_dim;
    float part = 0.f;
    for (int k = lane; k < k_dim; k += 32) {
        // BF16 bits -> F32 decode: exact widening.
        const uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
        float wf;
        memcpy(&wf, &bits, 4);
        part += wf * __half2float(x[k]);
    }
    for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off);
    if (lane == 0) y[n] = __float2half(part);
}

cudaError_t FlashGemmVBf16(cudaStream_t stream, const uint16_t *w,
                           const __half *x, __half *y, int k_dim, int n_dim)
{
    const int warps = 8;
    const int blocks = (n_dim + warps - 1) / warps;
    flash_gemv_bf16_kernel<<<blocks, warps * 32, 0, stream>>>(w, x, y, k_dim,
                                                              n_dim);
    return cudaGetLastError();
}

// MLA attention over the per-layer KV cache (no rope on the main path;
// qk/v head_dim 256). One block per head. Mirrors the GlmMlaAttnKernel
// numerics (FP32 softmax, single FP16 rounding at the end) with the
// glibc exp replica.

// KV append for one token: k_cache[t*heads + h)*D] = k[h*D + d] from the
// kv_b projection rows: kv_b output [heads*(2D)] = [k part | v part].
__global__ void flash_mla_store_kv_kernel(
    const __half *__restrict__ kvb, __half *__restrict__ k_cache,
    __half *__restrict__ v_cache, int64_t slot, int heads, int dim,
    int64_t cap)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int D2 = 2 * dim;
    if (idx >= heads * dim) return;
    const int h = idx / dim;
    const int d = idx % dim;
    k_cache[(slot * heads + h) * dim + d] = kvb[h * D2 + d];
    v_cache[(slot * heads + h) * dim + d] = kvb[h * D2 + dim + d];
}

__global__ void flash_mla_attn_kernel(const __half *__restrict__ q,
                                      const __half *__restrict__ k_cache,
                                      const __half *__restrict__ v_cache,
                                      __half *__restrict__ out, int cap,
                                      int total, int heads, float scale)
{
    const int h = blockIdx.x;
    const int D = 256;
    extern __shared__ float smem[];  // [D] q + [4096] scores + red[256]
    float *s_q = smem;               // [D]
    float *s_score = smem + D;       // [total] (cap <= 4096)
    float *s_red = smem + D + 4096;  // [blockDim]
    if (threadIdx.x < D)
        s_q[threadIdx.x] = __half2float(q[h * D + threadIdx.x]);
    __syncthreads();

    // Score pass: strided partial dots + block reduction (no warp
    // intrinsics — shared-memory tree, thread-0 finalize).
    for (int t = 0; t < total; ++t) {
        const __half *kr =
            k_cache + (static_cast<int64_t>(t) * heads + h) * D;
        float part = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            part += s_q[d] * __half2float(kr[d]);
        s_red[threadIdx.x] = part;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off)
                s_red[threadIdx.x] += s_red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) s_score[t] = s_red[0] * scale;
        __syncthreads();
    }
    // Softmax finalize (thread 0) + shared p.
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int t = 0; t < total; ++t) mx = fmaxf(mx, s_score[t]);
        float psum = 0.0f;
        for (int t = 0; t < total; ++t) {
            const float p = flash_expf_glibc(s_score[t] - mx);
            s_score[t] = p;
            psum += p;
        }
        s_red[0] = psum;
        if (flash_dbg_on)
            printf("[mla2] h=%d mx=%.4f psum=%.4f score0=%.4f q0=%.4f k0=%.4f\n",
                   h, mx, psum, s_score[0], s_q[0],
                   __half2float(k_cache[(0 * heads + h) * D]));
    }
    __syncthreads();
    const float psum = s_red[0];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < total; ++t) {
            const __half *vr =
                v_cache + (static_cast<int64_t>(t) * heads + h) * D;
            acc += (s_score[t] / psum) * __half2float(vr[d]);
        }
        out[h * D + d] = __float2half(acc);
    }
}

} // namespace


__global__ void flash_exp0_probe_kernel(float *out)
{
    out[0] = flash_expf_glibc(0.0f);
    out[1] = flash_expf_glibc(1.0f);
    out[2] = flash_expf_glibc(-1.0f);
    out[3] = expf(0.0f);
}

__global__ void flash_exp_dbg_kernel(float *out)
{
    // Exact copy of flash_expf_glibc's body for x=0 with step prints.
    const float x = 0.0f;
    const uint32_t ux = flash_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    out[7] = static_cast<float>(abstop);
    if (abstop >= 0x42bu) {
        out[6] = 666.0f;  // fast path taken!
        return;
    }
    const double xd = static_cast<double>(x);
    const double z = 0x1.71547652b82fep+5 * xd;
    out[5] = static_cast<float>(z);
    constexpr double kShiftD = 0x1.8p+52;
    const double kd =
        static_cast<double>(static_cast<uint64_t>(z + kShiftD)) - kShiftD;
    out[4] = static_cast<float>(kd);
    const uint64_t ki = static_cast<uint64_t>(static_cast<int64_t>(kd));
    const double r = z - kd;
    uint64_t t = kFlashExpfTab[ki % kFlashExpfN];
    t += ki << (52 - kFlashExpfTableBits);
    const double sv = flash_as_double(t);
    out[3] = static_cast<float>(sv);
    const double zz = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = zz * r2 + y;
    y = y * sv;
    out[0] = static_cast<float>(y);
}

cudaError_t RunFlashExp0Probe(cudaStream_t stream, float *d_out)
{
    flash_exp0_probe_kernel<<<1, 1, 0, stream>>>(d_out);
    flash_exp_dbg_kernel<<<1, 1, 0, stream>>>(d_out + 8);
    return cudaGetLastError();
}


cudaError_t RunFlashMlaAttnTest(cudaStream_t stream, const __half *d_q,
                                const __half *d_k, const __half *d_v,
                                __half *d_out, int heads, int dim,
                                float scale, size_t smem)
{
    flash_mla_attn_kernel<<<heads, 256, smem, stream>>>(d_q, d_k, d_v, d_out,
                                                        dim, 1, heads, scale);
    return cudaGetLastError();
}

// DSA decode scratch contract (caller arena, FP16 unless noted):
//   d_qr [q_lora], d_q [heads*qk], d_kvc [kv_lora], d_kvp [kv_lora],
//   d_kvb [heads*2*qk], d_attn [heads*qk], d_out [H]
cudaError_t RunFlashDsaDecode(cudaStream_t stream,
                              const FlashLayerWeights &lw, int hidden,
                              int heads, int qk_dim, int q_lora, int kv_lora,
                              __half *d_x, __half *d_qr, __half *d_q,
                              __half *d_kvc, __half *d_kvp, __half *d_kvb,
                              __half *d_attn, __half *d_out,
                              __half *d_kv_cache_k, __half *d_kv_cache_v,
                              int64_t slot, int64_t cap,
                              int /*indexer_budget*/)
{
    const int D = qk_dim;
    cudaError_t e;
    // 1. q_a (NVFP4) -> RMSNorm(q_a_norm) -> q_b (NVFP4).
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.q_a.d_packed, lw.q_a.d_scales,
                               lw.q_a.global_scale, d_qr, 1, hidden, q_lora,
                               false) != cudaSuccess)
        return cudaErrorUnknown;
    flash_rmsnorm_weighted_kernel<<<1, 1024, 1024 * sizeof(float), stream>>>(
        d_qr, lw.d_q_a_norm, d_qr, q_lora, 1e-5f);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_qr, lw.q_b.d_packed, lw.q_b.d_scales,
                               lw.q_b.global_scale, d_q, 1, q_lora, heads * D,
                               false) != cudaSuccess)
        return cudaErrorUnknown;
    // 2. kv_a (NVFP4) -> RMSNorm(kv_a_norm).
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.kv_a.d_packed, lw.kv_a.d_scales,
                               lw.kv_a.global_scale, d_kvc, 1, hidden,
                               kv_lora, false) != cudaSuccess)
        return cudaErrorUnknown;
    flash_rmsnorm_weighted_kernel<<<1, 512, 512 * sizeof(float), stream>>>(
        d_kvc, lw.d_kv_a_norm, d_kvc, kv_lora, 1e-5f);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    // 3. kv_b (BF16 raw) GEMV -> store k/v.
    if ((e = FlashGemmVBf16(stream, lw.d_kv_b, d_kvc, d_kvb, kv_lora,
                            heads * 2 * D)) != cudaSuccess) return e;
    flash_mla_store_kv_kernel<<<(heads * D + 255) / 256, 256, 0, stream>>>(
        d_kvb, d_kv_cache_k, d_kv_cache_v, slot, heads, D, cap);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    // 4. Attention (causal over slots 0..slot).
    {
        const float scale = 1.0f / sqrtf(static_cast<float>(D));
        const size_t smem = (2 * D + 1 + 4096) * sizeof(float);
        flash_mla_attn_kernel<<<heads, 256, smem, stream>>>(
            d_q, d_kv_cache_k + slot * heads * D,
            d_kv_cache_v + slot * heads * D, d_attn, cap, 1, heads, scale);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }
    // 5. o_proj (NVFP4).
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_attn, lw.o_big.d_packed,
                               lw.o_big.d_scales, lw.o_big.global_scale,
                               d_out, 1, heads * D, hidden, false)
        != cudaSuccess) return cudaErrorUnknown;
    return cudaSuccess;
}


// ── Driver helper kernels + constants ─────────────────────────────────────
namespace {

constexpr int KD_TOTAL = 64 * 128;  // KDA q/k/v channels

constexpr int KdaIndex(int layer)
{
    for (int i = 0; i < 34; ++i)
        if (frozen::kKdaLayerIds[i] == layer) return i;
    return -1;
}
constexpr int DsaIndex(int layer)
{
    for (int i = 0; i < 11; ++i)
        if (frozen::kDsaLayerIds[i] == layer) return i;
    return -1;
}

__global__ void bf16_to_f16_kernel(const uint8_t *__restrict__ src,
                                   __half *__restrict__ dst, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    if (i >= n) return;
    const uint32_t bits = (static_cast<uint32_t>(src[2 * i])
                           | (static_cast<uint32_t>(src[2 * i + 1]) << 8))
                          << 16;
    float f;
    memcpy(&f, &bits, 4);
    dst[i] = __float2half(f);
}

__global__ void f16_to_f32_kernel(const __half *__restrict__ src,
                                  float *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void f32_to_f16_kernel(const float *__restrict__ src,
                                  __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__device__ int flash_dbg_id = 0;


__global__ void flash_embed_row_kernel(const __half *__restrict__ table,
                                       const int32_t *__restrict__ id,
                                       __half *__restrict__ out, int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0) flash_dbg_id = *id;
    if (i == 0 && flash_dbg_on && flash_dbg_id < 0)
        printf("[emb] id=%d\n", flash_dbg_id);
    if (i < H) out[i] = table[static_cast<int64_t>(max(*id, 0)) * H + i];
}

// Mean over hc streams: out[i] = mean_c state[c*H + i].
__global__ void flash_mean_streams_kernel(const float *__restrict__ state,
                                          int H, int hc,
                                          __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) acc += state[c * H + i];
    out[i] = __float2half(acc / static_cast<float>(hc));
}

// BF16 lm_head GEMV: logits[v] = head[v,:] . x (FP32 accumulate).
__global__ void flash_gemv_bf16_logits_kernel(const uint8_t *__restrict__ head,
                                              const __half *__restrict__ x,
                                              float *__restrict__ logits,
                                              int H, int V)
{
    const int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const int lane = threadIdx.x & 31;
    const uint8_t *row = head + static_cast<int64_t>(v) * H * 2;
    float part = 0.f;
    for (int k = lane; k < H; k += 32) {
        const uint32_t bits = (static_cast<uint32_t>(row[2 * k])
                               | (static_cast<uint32_t>(row[2 * k + 1]) << 8))
                              << 16;
        float wf;
        memcpy(&wf, &bits, 4);
        part += wf * __half2float(x[k]);
    }
    for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off);
    if (lane == 0) logits[v] = part;
}



static bool flash_router_debug = std::getenv("LSX_TRACE_FLASH") != nullptr;

// Router top-8, FP16 activation variant: gate BF16 bits [E, H], x FP16.
//
// Perf form (profiled 2026-09: 1.1 ms -> ~15 us per call), split into two
// launches so expert scoring scales across SMs:
//   Phase 1 — one warp per expert; lanes consume uint4 chunks (8 bf16) with
//   32*8 element stride, fp32 partials folded by a fixed xor-butterfly.
//   Writes raw-sigmoid score + choice (score+bias) to global scratch.
//   Phase 2 — per token, 8 rounds of block-wide (val, idx) argmax on the
//   choice score; selected experts masked to -inf. Tie-break: lower expert
//   index wins (matches FlashRouterTop8Host linear scan).
__global__ void flash_router_score_f16_kernel(
    const uint8_t *__restrict__ gate_bf16, const float *__restrict__ bias,
    const __half *__restrict__ x, float *__restrict__ scratch,
    int hidden, int experts)
{
    // scratch layout: [tok][2 * experts]: raw scores then choice scores.
    // grid = (chunks, tok): chunk = 16-warp group of experts, token = y.
    float *s_out = scratch + static_cast<size_t>(blockIdx.y) * experts * 2;
    constexpr int kWarps = 16;
    const int warp = threadIdx.x >> 5;
    const int e = blockIdx.x * kWarps + warp;
    if (e >= experts) return;
    const int lane = threadIdx.x & 31;
    const uint8_t *row = gate_bf16 + static_cast<size_t>(e) * hidden * 2;
    float part = 0.0f;
    int h = lane * 8;
    for (; h + 8 <= hidden; h += 32 * 8) {
        const uint4 pkt = *reinterpret_cast<const uint4 *>(row +
                                                           h * 2);
        const float2 x01 = __half22float2(
            *reinterpret_cast<const __half2 *>(x + h));
        const float2 x23 = __half22float2(
            *reinterpret_cast<const __half2 *>(x + h + 2));
        const float2 x45 = __half22float2(
            *reinterpret_cast<const __half2 *>(x + h + 4));
        const float2 x67 = __half22float2(
            *reinterpret_cast<const __half2 *>(x + h + 6));
        const float g[8] = {
            bf16_bits_to_f32(pkt.x & 0xffff),
            bf16_bits_to_f32(pkt.x >> 16),
            bf16_bits_to_f32(pkt.y & 0xffff),
            bf16_bits_to_f32(pkt.y >> 16),
            bf16_bits_to_f32(pkt.z & 0xffff),
            bf16_bits_to_f32(pkt.z >> 16),
            bf16_bits_to_f32(pkt.w & 0xffff),
            bf16_bits_to_f32(pkt.w >> 16),
        };
        part += g[0] * x01.x + g[1] * x01.y;
        part += g[2] * x23.x + g[3] * x23.y;
        part += g[4] * x45.x + g[5] * x45.y;
        part += g[6] * x67.x + g[7] * x67.y;
    }
    for (; h < hidden; h += 32 * 8) {  // tail (hidden % 256 != 0)
        const int lim = h + 8 < hidden ? h + 8 : hidden;
        for (int i = h; i < lim; ++i) {
            const uint32_t bits =
                (static_cast<uint32_t>(row[i * 2]) |
                 (static_cast<uint32_t>(row[i * 2 + 1]) << 8)) << 16;
            part += bf16_bits_to_f32(bits) * __half2float(x[i]);
        }
    }
    // Warp fully converged here: e is warp-uniform.
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, off);
    if (lane == 0) {
        const float s = sigmoid_dev(part);
        s_out[e] = s;
        s_out[experts + e] = s + bias[e];
    }
}

// Phase 2: top-8 by choice score; each round a block-wide argmax.
// grid = tok, one block per token, 512 threads (16 warps).
__global__ void flash_router_top8_f16_kernel(
    const float *__restrict__ scratch, int experts,
    qwen_moe::QwenMoeRoutedSlot *__restrict__ slots)
{
    constexpr int kTopK = 8;
    constexpr int kWarps = 16;
    __shared__ float s_choice[1024];
    __shared__ float s_val[kWarps];
    __shared__ int s_idx[kWarps];
    const float *s_in = scratch + static_cast<size_t>(blockIdx.x) * experts * 2
                      + experts;
    const int t0 = threadIdx.x;
    if (t0 < experts) s_choice[t0] = s_in[t0];
    __syncthreads();
    for (int kk = 0; kk < kTopK; ++kk) {
        float my = -3.4e38f;
        int myi = -1;
        if (t0 < experts) {
            my = s_choice[t0];
            myi = t0;
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, my, off);
            const int oi = __shfl_xor_sync(0xffffffffu, myi, off);
            if (ov > my || (ov == my && oi < myi)) {
                my = ov;
                myi = oi;
            }
        }
        const int w = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) {
            s_val[w] = my;
            s_idx[w] = myi;
        }
        __syncthreads();
        if (t0 == 0) {
            float bv = -3.4e38f;
            int bi = -1;
            for (int i = 0; i < kWarps; ++i) {
                if (s_val[i] > bv || (s_val[i] == bv && s_idx[i] < bi)) {
                    bv = s_val[i];
                    bi = s_idx[i];
                }
            }
            slots[blockIdx.x * kTopK + kk].eid = bi;
            slots[blockIdx.x * kTopK + kk].coeff = s_in[bi];
            s_choice[bi] = -3.4e38f;  // mask for the next round
        }
        __syncthreads();
    }
    if (t0 == 0) {
        float sum = 0.0f;
        for (int kk = 0; kk < kTopK; ++kk)
            sum += slots[blockIdx.x * kTopK + kk].coeff;
        for (int kk = 0; kk < kTopK; ++kk)
            slots[blockIdx.x * kTopK + kk].coeff =
                slots[blockIdx.x * kTopK + kk].coeff / sum * 2.5f;
    }
}

} // namespace


cudaError_t RunFlashRouterTop8F16(cudaStream_t stream, const uint8_t *d_gate,
                                  const float *d_bias, const __half *d_x,
                                  int tok, int hidden, int experts,
                                  qwen_moe::QwenMoeRoutedSlot *d_slots)
{
    if (std::getenv("LSX_TRACE_FLASH")) {
        __half hx[4];
        cudaMemcpy(hx, d_x, sizeof(hx), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "[pre-rk] x0=%.6f x1=%.6f\n",
                     __half2float(hx[0]), __half2float(hx[1]));
    }
    const size_t scratch = static_cast<size_t>(experts) * 2 * sizeof(float);
    (void) scratch;
    // Grow-on-demand score scratch: [tok][2 * experts] floats (process-wide
    // singleton, same lifetime pattern as the fp16 embed cache).
    static float *d_rscratch = nullptr;
    static size_t rscratch_floats = 0;
    const size_t need = static_cast<size_t>(tok) * experts * 2;
    if (rscratch_floats < need) {
        if (d_rscratch) cudaFree(d_rscratch);
        if (cudaMalloc(&d_rscratch, need * sizeof(float)) != cudaSuccess)
            return cudaGetLastError();
        rscratch_floats = need;
    }
    const int chunks = (experts + 15) / 16;
    flash_router_score_f16_kernel<<<dim3(chunks, tok), 512, 0, stream>>>(
        d_gate, d_bias, d_x, d_rscratch, hidden, experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    flash_router_top8_f16_kernel<<<tok, 512, 0, stream>>>(
        d_rscratch, experts, d_slots);
    return cudaGetLastError();
}



cudaError_t RunFlashRoutedFFN(
    cudaStream_t stream, qwen_moe::QwenMoeExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer, int moe_layer,
    const qwen_moe::QwenMoeRoutedSlot *h_slots, int k_top,
    const __half *d_x, int hidden, int moe_inter, float *d_acc,
    __half *d_yg, __half *d_yu, __half *d_y2, std::string &error,
    cudaStream_t h2d_stream, cudaEvent_t *copy_ev, cudaEvent_t *slot_free_ev,
    __half *batch_ygu, __half *batch_hbuf, __half *batch_ypart,
    qwen_moe::QwenMoeGvSlotDev *d_gemv_tbl)
{
    error.clear();
    if (k_top > 16) {
        error = "routed slots exceed staging capacity";
        return cudaErrorInvalidValue;
    }
    const int store_layer = moe_layer - frozen::kMoeFirstLayer;
    int dev_slots[16];
    // Dual-stream staging: miss uploads queue on h2d_stream while the
    // compute stream keeps running; each expert's GEMM chain fences its own
    // upload via copy_ev. An eviction recycles a slot whose last reader is
    // the GEMM chain fenced by slot_free_ev[slot] — wait it on h2d_stream
    // before overwriting.
    for (int k = 0; k < k_top; ++k) {
        std::string ensure_err;
        bool missed = false;
        int evicted = -1;
        const int slot = qwen_moe::EnsureQwenMoeExpertResident(
            store, store_layer, h_slots[k].eid, cache, stream, ensure_err,
            h2d_stream, &missed, &evicted);
        if (slot < 0) {
            error = "ensure resident failed: " + ensure_err;
            return cudaErrorInvalidValue;
        }
        if (evicted >= 0 && slot_free_ev && slot_free_ev[evicted])
            cudaStreamWaitEvent(h2d_stream, slot_free_ev[evicted], 0);
        if (missed && h2d_stream && copy_ev && copy_ev[k]) {
            cudaEventRecord(copy_ev[k], h2d_stream);
            cudaStreamWaitEvent(stream, copy_ev[k], 0);
        }
        dev_slots[k] = slot;
    }
    const auto &recs = store.records[static_cast<size_t>(store_layer)];
    const auto &slab = store.h_slabs[static_cast<size_t>(store_layer)];
    const auto &rec0 = recs[static_cast<size_t>(h_slots[0].eid)];
    const int64_t pack_stride = cache.pack_stride;
    const int64_t scale_stride = cache.scale_stride;
    const uint8_t *pack_base = cache.d_pack;
    const uint8_t *scale_base = cache.d_scale;
    (void) slab;
    (void) rec0;

    // Batched routed chain: all k_top experts in 4 launches (w13 GEMV batch ->
    // elementwise swiglu -> down GEMV batch -> ordered coeff reduce) through
    // the shared model-neutral core. Same slot-major weight streaming as the
    // legacy per-slot chain but without the K*4 launch/drain overhead.
    // Miss uploads were already fenced on the compute stream in the staging
    // loop above, so the batch can read every slot. Nulls (ad-hoc callers)
    // keep the legacy path.
    if (batch_ygu && batch_hbuf && batch_ypart && d_gemv_tbl &&
        (hidden & 63) == 0 && (moe_inter & 63) == 0 && k_top <= 16) {
        qwen_moe::QwenMoeGvSlotDev tbl[16];
        for (int k = 0; k < k_top; ++k) {
            const auto &rec = recs[static_cast<size_t>(h_slots[k].eid)];
            tbl[k].dev_slot = dev_slots[k];
            tbl[k].gs1 = rec.gs1;
            tbl[k].gs3 = rec.gs3;
            tbl[k].gs2 = rec.gs2;
            tbl[k].coeff = h_slots[k].coeff;
        }
        if (cudaMemcpyAsync(d_gemv_tbl, tbl,
                            sizeof(qwen_moe::QwenMoeGvSlotDev) * k_top,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            error = "gemv slot table upload failed";
            return cudaErrorInvalidValue;
        }
        const int64_t w13_b =
            static_cast<int64_t>(rec0.w1_bytes + rec0.w3_bytes);
        const int64_t s13_b =
            static_cast<int64_t>(rec0.s1_bytes + rec0.s3_bytes);
        const cudaError_t be = qwen_moe::RunQwenMoeRoutedGemmVBatchCore(
            stream, pack_base, scale_base, pack_stride, scale_stride,
            d_gemv_tbl, k_top, w13_b, s13_b, hidden, moe_inter, d_x, batch_ygu,
            batch_hbuf, batch_ypart, d_acc);
        if (be != cudaSuccess) {
            error = "batched routed ffn failed";
            return be;
        }
        if (slot_free_ev) {
            for (int k = 0; k < k_top; ++k)
                if (slot_free_ev[dev_slots[k]])
                    cudaEventRecord(slot_free_ev[dev_slots[k]], stream);
        }
        return cudaGetLastError();
    }

    for (int k = 0; k < k_top; ++k) {
        const auto &rec = recs[static_cast<size_t>(h_slots[k].eid)];
        const uint8_t *pack =
            pack_base + static_cast<int64_t>(dev_slots[k]) * pack_stride;
        const uint8_t *scale =
            scale_base + static_cast<int64_t>(dev_slots[k]) * scale_stride;
        const uint8_t *w1 = pack;
        const uint8_t *w3 = pack + rec.w1_bytes;
        const uint8_t *w2 = pack + rec.w1_bytes + rec.w3_bytes;
        const uint8_t *s1 = scale;
        const uint8_t *s3 = scale + rec.s1_bytes;
        const uint8_t *s2 = scale + rec.s1_bytes + rec.s3_bytes;
        // Gate/up live adjacently in the LRU pack ([w1|w3|w2], scales
        // [s1|s3|s2]): one pair GEMV emits [gate | up] into d_yg[0..2I)
        // (the arena slot is kYMax >= 2I) — halves the w13 launch count
        // and doubles the per-launch weight stream. Bit-identical halves.
        if (lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(
                stream, d_x, w1, s1, rec.gs1, rec.gs3, d_yg, hidden,
                moe_inter, false) != cudaSuccess)
            { error = "w13 gemm failed"; return cudaErrorInvalidValue; }
        {
            const int blocks = (moe_inter + 255) / 256;
            ::lsxcommon::kernels::SwigluKernel<__half,
                ::lsxcommon::kernels::ExpFlavor::Glibc, true>
                <<<blocks, 256, 0, stream>>>(d_yg, d_yg + moe_inter, d_yg,
                                             moe_inter, 10.0f);
        }
        if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, w2, s2, rec.gs2, d_y2, 1,
                                   moe_inter, hidden, false) != cudaSuccess)
            { error = "w2 gemm failed"; return cudaErrorInvalidValue; }
        // acc += coeff * y2
        flash_coeff_add_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            d_y2, h_slots[k].coeff, d_acc, hidden);
        // Slot reads are done: fence for future evictions of this slot.
        if (slot_free_ev && slot_free_ev[dev_slots[k]])
            cudaEventRecord(slot_free_ev[dev_slots[k]], stream);
    }
    return cudaGetLastError();
}

cudaError_t RunFlashSharedFFN(cudaStream_t stream, const FlashLayerWeights &lw,
                              const __half *d_x, int hidden, int shared_inter,
                              __half *d_yg, __half *d_yu, __half *d_y2,
                              __half *d_out)
{
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.shared_w1.d_packed,
                               lw.shared_w1.d_scales, lw.shared_w1.global_scale,
                               d_yg, 1, hidden, shared_inter, false)
        != cudaSuccess) return cudaErrorUnknown;
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, lw.shared_w3.d_packed,
                               lw.shared_w3.d_scales, lw.shared_w3.global_scale,
                               d_yu, 1, hidden, shared_inter, false)
        != cudaSuccess) return cudaErrorUnknown;
    {
        const int blocks = (shared_inter + 255) / 256;
        ::lsxcommon::kernels::SwigluKernel<__half,
            ::lsxcommon::kernels::ExpFlavor::Glibc, true>
            <<<blocks, 256, 0, stream>>>(d_yg, d_yu, d_yg, shared_inter,
                                         10.0f);
    }
    if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, lw.shared_w2.d_packed,
                               lw.shared_w2.d_scales, lw.shared_w2.global_scale,
                               d_y2, 1, shared_inter, hidden, false)
        != cudaSuccess) return cudaErrorUnknown;
    flash_add_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(d_y2, d_out,
                                                               hidden);
    return cudaGetLastError();
}

// ── Generation driver ─────────────────────────────────────────────────────
// Full GLM-5.3-Flash decode pipeline. Production order per step:
//   embed -> hc expand (FP32 state x4) -> per layer
//     hc_pre(attn) -> [KDA|DSA attention] -> hc_post -> hc_pre(ffn) ->
//     MoE|dense MLP -> hc_post -> hc_head -> final norm -> lm_head.

cudaError_t RunGlmFlashGeneration(
    cudaStream_t stream, const FlashPinnedWeights &pw,
    qwen_moe::QwenMoeExpertStore &store, gpu::ExpertDeviceCache &cache,
    const std::vector<int32_t> &prompt, int max_new,
    const std::vector<int> &eos_ids, qwen::QwenGenerationResult &result,
    sample::ISampler &sampler)
{
    result = qwen::QwenGenerationResult{};
    using namespace frozen;
    const int H = kHidden;
    const int hc = kHcCount;
    const int HCH = H * hc;
    const int V = kVocab;
    const int E = kNumExperts;
    const int K = kTopK;
    const int I = kMoeIntermediate;
    const int SI = kMoeIntermediate;
    if (prompt.empty() || max_new <= 0 || pw.layers.empty()
        || !pw.d_embed_bf16 || !pw.d_head_bf16) {
        result.error_msg = "invalid glm flash generation inputs";
        return cudaErrorInvalidValue;
    }
    const int max_seq = static_cast<int>(prompt.size()) + max_new;

    // Persistent FP16 embed table (dequant once).
    static __half *d_embed_h = nullptr;
    static int64_t embed_elems = 0;
    if (!d_embed_h || embed_elems < static_cast<int64_t>(V) * H) {
        if (d_embed_h) cudaFree(d_embed_h);
        if (cudaMalloc(&d_embed_h,
                       static_cast<size_t>(V) * H * sizeof(__half))
            != cudaSuccess) return cudaGetLastError();
        embed_elems = static_cast<int64_t>(V) * H;
        // BF16 bits -> FP16 exact widening (both 8-bit mantissa formats).
        bf16_to_f16_kernel<<<static_cast<int>((embed_elems + 255) / 256), 256,
                             0, stream>>>(
            pw.d_embed_bf16, d_embed_h, embed_elems);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return cudaGetLastError();
    }

    // Per-layer KDA state (allocated once, static etalon style).
    static std::vector<FlashKdaState> kda_states;
    if (kda_states.empty()) {
        kda_states.resize(static_cast<size_t>(kKdaLayerCount));
        for (auto &st : kda_states)
            if (FlashKdaStateAlloc(st, max_seq + 64) != cudaSuccess)
                return cudaGetLastError();
    }
    // DSA KV caches: [dsa_layer][cap * heads * 256] x2.
    static std::vector<__half *> dsa_kv_k, dsa_kv_v;
    static int64_t dsa_cap = 0;
    if (dsa_cap < max_seq + 64) {
        for (auto *p : dsa_kv_k) cudaFree(p);
        for (auto *p : dsa_kv_v) cudaFree(p);
        dsa_kv_k.clear();
        dsa_kv_v.clear();
        dsa_cap = max_seq + 64;
        const size_t bytes = static_cast<size_t>(dsa_cap) * kHeads
                           * kQkHeadDim * sizeof(__half);
        for (int i = 0; i < kDsaLayerCount; ++i) {
            __half *a = nullptr, *b = nullptr;
            if (cudaMalloc(&a, bytes) != cudaSuccess
                || cudaMalloc(&b, bytes) != cudaSuccess)
                return cudaGetLastError();
            dsa_kv_k.push_back(a);
            dsa_kv_v.push_back(b);
        }
    }

    // Arena.
    static void *arena = nullptr;
    static size_t arena_cap = 0;
    size_t half_elems = 0;
    half_elems += H;                       // d_act
    half_elems += (size_t)2 * HCH;         // d_state (FP32? no: kept FP32 separately)
    half_elems += H;                       // d_single (hc pre out)
    half_elems += H;                       // d_act2 (module out f16)
    half_elems += kQLora;                  // d_qr
    half_elems += kHeads * kQkHeadDim;     // d_q
    half_elems += kKvLora;                 // d_kvc
    half_elems += kHeads * 2 * kQkHeadDim; // d_kvb
    half_elems += kHeads * kQkHeadDim;     // d_attn
    half_elems += 2 * KD_TOTAL;            // kda q/k raw
    half_elems += 2 * KD_TOTAL;            // kda v + qc
    half_elems += 2 * KD_TOTAL;            // kc + vc
    half_elems += kHeads;                  // d_b
    half_elems += 3 * KD_TOTAL;            // d_f + d_g + d_gated
    half_elems += KD_TOTAL;                // d_attn kda
    half_elems += 128;                     // f/g mid
    half_elems += 3 * I;                   // moe yg/yu/y2
    half_elems += 3 * I;                   // shared yg/yu/y2 + dense reuse
    // Dense-MLP layers (0-2) reuse d_yg/d_yu with kDenseIntermediate (12288)
    // rows — the slots must fit the largest of the two, or the dense GEMMs
    // stomp d_sy and the FP32 state region beyond.
    constexpr int kYMax = kMoeIntermediate > kDenseIntermediate
                              ? kMoeIntermediate
                              : kDenseIntermediate;
    half_elems += 3 * kYMax - 3 * I;       // dense-MLP headroom for yg/yu/y2
    // Batched routed-FFN scratch: slot-major [K*2I] gate/up, [K*I] swiglu,
    // [K*H] down partials (see RunFlashRoutedFFN batch branch).
    half_elems += (size_t) K * 2 * I;      // d_batch_ygu
    half_elems += (size_t) K * I;          // d_batch_hbuf
    half_elems += (size_t) K * H;          // d_batch_ypart
    size_t float_elems = HCH;              // d_state FP32 [hc*H]
    float_elems += H;                      // d_module_f32 (hc_post input)
    float_elems += (2 + hc) * hc;          // hc post/comb/pre scratch
    float_elems += 64;                     // misc
    float_elems += V;                      // logits
    float_elems += H;                      // moe acc
    const size_t half_bytes = (half_elems * 2 + 255) & ~size_t(255);
    const size_t float_bytes = (float_elems * 4 + 255) & ~size_t(255);
    const size_t need = half_bytes + float_bytes + 4096;
    if (arena_cap < need) {
        if (arena) cudaFree(arena);
        if (cudaMalloc(&arena, need) != cudaSuccess)
            return cudaGetLastError();
        arena_cap = need;
    }
    __half *ph = static_cast<__half *>(arena);
    auto adv = [&](size_t elems) {
        __half *r = ph;
        ph += ((elems + 7) / 8) * 8;
        return r;
    };
    __half *d_act = adv(H);
    __half *d_single = adv(H);
    __half *d_act2 = adv(H);
    __half *d_qr = adv(kQLora);
    __half *d_q = adv(kHeads * kQkHeadDim);
    __half *d_kvc = adv(kKvLora);
    __half *d_kvb = adv(kHeads * 2 * kQkHeadDim);
    __half *d_attn = adv(kHeads * kQkHeadDim);
    __half *d_kda_q = adv(KD_TOTAL);
    __half *d_kda_k = adv(KD_TOTAL);
    __half *d_kda_v = adv(KD_TOTAL);
    __half *d_kda_qc = adv(KD_TOTAL);
    __half *d_kda_kc = adv(KD_TOTAL);
    __half *d_kda_vc = adv(KD_TOTAL);
    __half *d_kda_b = adv(kHeads);
    __half *d_kda_f = adv(KD_TOTAL);
    __half *d_kda_g = adv(KD_TOTAL);
    __half *d_kda_gated = adv(KD_TOTAL);
    __half *d_kda_attn = adv(KD_TOTAL);
    __half *d_kda_mid = adv(128);
    __half *d_yg = adv(kYMax);
    __half *d_yu = adv(kYMax);
    __half *d_y2 = adv(kYMax);
    __half *d_sy = adv(kYMax);
    // Batched routed-FFN scratch (see RunFlashRoutedFFN). Slot-major layouts
    // consumed by qwen_moe::RunQwenMoeRoutedGemmVBatchCore.
    __half *d_batch_ygu = adv((size_t) K * 2 * I);
    __half *d_batch_hbuf = adv((size_t) K * I);
    __half *d_batch_ypart = adv((size_t) K * H);
    qwen_moe::QwenMoeGvSlotDev *d_gemv_tbl = nullptr;
    float *pf = reinterpret_cast<float *>(static_cast<uint8_t *>(arena)
                                          + half_bytes);
    float *d_state = pf; pf += HCH;
    float *d_module_f = pf; pf += H;
    float *d_hc_scr = pf; pf += 1024;  // >= (kHcMix+1)*kHcTiles = 800
    float *d_logits = pf; pf += V;
    float *d_moe_acc = pf; pf += H;
    float *d_hc_post = pf; pf += hc;
    float *d_hc_comb = pf; pf += hc * hc;
    float *d_hc_pre = pf; pf += hc;
    float *d_maxv = pf; pf += 64;

    // KDA packed scratch: (8*KD + heads) halves then (heads + heads*dim)
    // floats (contract of RunFlashKdaDecode).
    static __half *d_kda_scratch = nullptr;
    if (!d_kda_scratch) {
        const size_t half_cnt = static_cast<size_t>(9) * KD_TOTAL + kHeads;
        const size_t flt_cnt = static_cast<size_t>(2) * KD_TOTAL + kHeads;
        const size_t kda_bytes =
            half_cnt * 2 + ((flt_cnt * 4 + 255) & ~size_t(255));
        if (cudaMalloc(&d_kda_scratch, kda_bytes) != cudaSuccess)
            return cudaGetLastError();
    }

    int32_t *d_argmax = nullptr, *d_tokens = nullptr, *d_ids = nullptr;
    int32_t *h_ids_pinned = nullptr;
    cudaMallocHost(&h_ids_pinned, 4);
    if (cudaMalloc(&d_argmax, 4) != cudaSuccess
        || cudaMalloc(&d_tokens, sizeof(int32_t) * max_new) != cudaSuccess
        || cudaMalloc(&d_ids, 4) != cudaSuccess)
        return cudaGetLastError();
    qwen_moe::QwenMoeRoutedSlot *d_slots = nullptr;
    if (cudaMalloc(&d_slots, sizeof(qwen_moe::QwenMoeRoutedSlot) * K)
        != cudaSuccess) return cudaGetLastError();
    if (cudaMalloc(&d_gemv_tbl, sizeof(qwen_moe::QwenMoeGvSlotDev) * K)
        != cudaSuccess) return cudaGetLastError();

    // Dual-stream expert staging: a dedicated H2D stream overlaps miss
    // uploads with the compute stream; per-expert copy events fence each
    // GEMM chain, per-slot events fence evictions (see RunFlashRoutedFFN).
    // Perf knob: narrow-bus GPUs keep single-stream staging (the overlap
    // cannot pay for the extra sync on low-bandwidth links).
    cudaStream_t h2d_stream = nullptr;
    cudaEvent_t copy_ev[16] = {};
    std::vector<cudaEvent_t> slot_free_ev;
    const bool dual_stream =
        lsx_nvfp4::LsxDualStreamStaging() &&
        cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking) ==
            cudaSuccess;
    if (!dual_stream) h2d_stream = nullptr;
    for (int k = 0; k < 16; ++k) {
        if (cudaEventCreateWithFlags(&copy_ev[k], cudaEventDisableTiming)
            != cudaSuccess) return cudaGetLastError();
    }
    slot_free_ev.resize(static_cast<size_t>(cache.slots), nullptr);
    for (int s = 0; s < cache.slots; ++s) {
        if (cudaEventCreateWithFlags(&slot_free_ev[static_cast<size_t>(s)],
                                     cudaEventDisableTiming)
            != cudaSuccess) return cudaGetLastError();
    }

    {
        const int dbg = std::getenv("LSX_TRACE_FLASH") ? 1 : 0;
        cudaMemcpyToSymbolAsync(flash_dbg_on, &dbg, 4, 0,
                                cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // KDA state + conv history zeroing (fresh session). Stale conv history
    // from a previous call would poison the depthwise conv windows.
    for (auto &st : kda_states) {
        cudaMemsetAsync(st.d_state, 0,
                        static_cast<size_t>(64 * 128 * 128) * 4, stream);
        cudaMemsetAsync(st.d_conv_hist, 0,
                        static_cast<size_t>(3 * (kConvKernel - 1)
                                            * 64 * kHeadDim) * 2, stream);
    }
    // DSA KV caches: stale rows from a previous call would leak into the
    // next session (slot 0 reused). Zero once per call.
    for (int i = 0; i < kDsaLayerCount; ++i) {
        cudaMemsetAsync(dsa_kv_k[i], 0,
                        static_cast<size_t>(dsa_cap) * kHeads * kQkHeadDim
                            * sizeof(__half),
                        stream);
        cudaMemsetAsync(dsa_kv_v[i], 0,
                        static_cast<size_t>(dsa_cap) * kHeads * kQkHeadDim
                            * sizeof(__half),
                        stream);
    }

    const int prompt_steps = static_cast<int>(prompt.size());
    const int total_steps = prompt_steps + max_new - 1;
    int32_t cur = prompt[0];
    int generated = 0;
    cudaError_t step_err = cudaSuccess;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    if (t_start) cudaEventRecord(t_start, stream);

    for (int step = 0; step < total_steps; ++step) {
        const int kv_pos = step;  // 1 token/step for both KDA and DSA
        h_ids_pinned[0] = cur;
        if (std::getenv("LSX_TRACE_FLASH"))
            std::fprintf(stderr, "[step] step=%d cur=%d\n", step, cur);
        cudaMemcpyAsync(d_ids, h_ids_pinned, 4, cudaMemcpyHostToDevice,
                        stream);
        // Decode-loop contract: the token id feeds the very first kernel of
        // the step; a synchronous staging copy keeps the pinned cell stable
        // across steps (async H2D from a reused pinned slot races with the
        // host's next-step overwrite).
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            result.error_msg = "id upload sync";
            step_err = cudaGetLastError();
            break;
        }
        cudaError_t e;
        // Embed + expand to hc state.
        flash_embed_row_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
            d_embed_h, d_ids, d_act, H);
        if ((e = cudaGetLastError()) != cudaSuccess)
            { result.error_msg = "embed"; break; }
        gpu::run_expand_hc(stream, d_act, d_state);

        for (int l = 0; l < kLayers; ++l) {
            const FlashLayerWeights &lw = pw.layers[l];
            const bool is_kda = IsKdaLayer(l);
            if (std::getenv("LSX_TRACE_FLASH")) {
                static std::vector<float> hs;
                hs.resize(HCH);
                cudaMemcpyAsync(hs.data(), d_state, sizeof(float) * HCH,
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                float s0 = hs[0], sm = 0;
                int nan = 0;
                for (float v : hs) {
                    if (std::isnan(v)) ++nan;
                    sm += v;
                }
                std::fprintf(stderr,
                             "[stage] pre-l=%d moe=%d kda=%d s0=%.6f "
                             "mean=%.6f nan=%d\n",
                             l, (int) lw.is_moe, (int) lw.is_kda, s0,
                             sm / HCH, nan);
            }
            const int dsa_idx = DsaIndex(l);
            const int kda_idx = KdaIndex(l);
            // ── attn hc_pre ──
            gpu::run_hc_pre(stream, d_state, lw.d_hc_attn_fn, lw.d_hc_attn_scale,
                       lw.d_hc_attn_base, d_single, d_hc_post, d_hc_comb,
                       d_hc_pre, d_hc_scr, 1e-5f);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "hc_pre"; break; }
            // input norm (post_norm slot used as attn-input norm).
            flash_rmsnorm_weighted_kernel<<<1, 1024,
                                            1024 * sizeof(float), stream>>>(
                d_single, lw.d_input_norm, d_single, H, 1e-5f);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "in norm"; break; }
            // ── attention branch ──
            if (is_kda) {
                // Packed scratch contract (alloc below).
                e = RunFlashKdaDecode(stream, kda_states[kda_idx], lw, H,
                                      kHeads, kHeadDim, kConvKernel,
                                      d_single, d_act2, d_kda_scratch);
                if (e != cudaSuccess) { result.error_msg = "kda"; break; }
            } else {
                __half *kk = dsa_kv_k[dsa_idx];
                __half *vv = dsa_kv_v[dsa_idx];
                e = RunFlashDsaDecode(stream, lw, H, kHeads, kQkHeadDim,
                                      kQLora, kKvLora, d_single, d_qr, d_q,
                                      d_kvc, nullptr, d_kvb, d_attn, d_act2,
                                      kk, vv, kv_pos, dsa_cap, kIndexerTopk);
                if (e != cudaSuccess) { result.error_msg = "dsa"; break; }
            }
            if (std::getenv("LSX_TRACE_FLASH")) {
                std::fprintf(stderr, "[attn] post-l=%d\n", l);
            }
            if (std::getenv("LSX_TRACE_FLASH")) {
                std::vector<float> ha(H);
                cudaMemcpy(ha.data(), d_act2, sizeof(float) * H,
                           cudaMemcpyDeviceToHost);
                float hmean = 0;
                int nan = 0;
                for (float v : ha) {
                    if (std::isnan(v)) ++nan;
                    hmean += v;
                }
                std::fprintf(stderr, "[attn-out] l=%d mean=%.6f nan=%d\n",
                             l, hmean / H, nan);
            }
            // ── attn hc_post (module out FP32) ──
            f16_to_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                d_act2, d_module_f, H);
            gpu::run_hc_post(stream, d_module_f, d_state, d_hc_post, d_hc_comb,
                        d_state);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "hc_post"; break; }
            if (flash_dbg_on) {
                flash_dbg_probe_kernel<<<1, 1, 0, stream>>>(d_state, 1, 2,
                                                            HCH);
                flash_dbg_probe_kernel<<<1, 1, 0, stream>>>(d_act2, 0, 2, H);
            }
            if (std::getenv("LSX_TRACE_FLASH")) {
                std::vector<float> hst(HCH);
                cudaMemcpy(hst.data(), d_state, sizeof(float) * HCH,
                           cudaMemcpyDeviceToHost);
                int nan = 0;
                float hmean = 0;
                for (float v : hst) {
                    if (std::isnan(v)) ++nan;
                    hmean += v;
                }
                std::fprintf(stderr,
                             "[state2] l=%d mean=%.6f nan=%d\n", l,
                             hmean / HCH, nan);
            }
            // ── ffn hc_pre ──
            gpu::run_hc_pre(stream, d_state, lw.d_hc_ffn_fn, lw.d_hc_ffn_scale,
                       lw.d_hc_ffn_base, d_single, d_hc_post, d_hc_comb,
                       d_hc_pre, d_hc_scr, 1e-5f);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "hc_pre2"; break; }
            flash_rmsnorm_weighted_kernel<<<1, 1024,
                                            1024 * sizeof(float), stream>>>(
                d_single, lw.d_post_norm, d_single, H, 1e-5f);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "post norm"; break; }
            // ── MLP branch ──
            if (lw.is_moe) {
                if (std::getenv("LSX_TRACE_FLASH")) {
                    std::vector<float> hx(static_cast<size_t>(H));
                    cudaMemcpy(hx.data(), d_single, sizeof(float) * H,
                               cudaMemcpyDeviceToHost);
                    float hmean = 0;
                    int nan = 0;
                    for (float v : hx) {
                        if (std::isnan(v)) ++nan;
                        hmean += v;
                    }
                    std::fprintf(stderr,
                                 "[router-in] l=%d mean=%.6f nan=%d\n", l,
                                 hmean / H, nan);
                }
                // Router (FP16 activations, BF16 gate bits).
                qwen_moe::QwenMoeRoutedSlot h_slots[16];
                {
                    // Host-side route via bf16 gemv on device — reuse the
                    // f16 router variant below (RunFlashRouterTop8F16).
                    if (RunFlashRouterTop8F16(stream, lw.d_router,
                                              lw.d_router_bias, d_single, 1,
                                              H, E, d_slots) != cudaSuccess)
                        { result.error_msg = "router f16"; break; }
                    cudaMemcpyAsync(h_slots, d_slots, sizeof(h_slots[0]) * K,
                                    cudaMemcpyDeviceToHost, stream);
                    if (cudaStreamSynchronize(stream) != cudaSuccess)
                        { result.error_msg = "router sync"; break; }
                }
                cudaMemsetAsync(d_moe_acc, 0, sizeof(float) * H, stream);
                std::string ffn_err;
                if (RunFlashRoutedFFN(stream, store, cache, 0, l, h_slots, K,
                                      d_single, H, I, d_moe_acc, d_yg, d_yu,
                                      d_y2, ffn_err, h2d_stream, copy_ev,
                                      slot_free_ev.data(), d_batch_ygu,
                                      d_batch_hbuf, d_batch_ypart,
                                      d_gemv_tbl) != cudaSuccess)
                    { result.error_msg = "routed ffn: " + ffn_err; break; }
                // acc -> f16 into d_act2, add shared expert.
                f32_to_f16_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    d_moe_acc, d_act2, H);
                if (RunFlashSharedFFN(stream, lw, d_single, H, SI, d_yg, d_yu,
                                      d_y2, d_act2) != cudaSuccess)
                    { result.error_msg = "shared ffn"; break; }
            } else if (l < kMoeFirstLayer) {
                // Dense MLP: gate/up (NVFP4) -> clamped swiglu -> down.
                if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_single,
                                           lw.mlp_gate.d_packed,
                                           lw.mlp_gate.d_scales,
                                           lw.mlp_gate.global_scale, d_yg, 1,
                                           H, kDenseIntermediate, false)
                    != cudaSuccess)
                    { result.error_msg = "dense gate"; break; }
                if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_single,
                                           lw.mlp_up.d_packed,
                                           lw.mlp_up.d_scales,
                                           lw.mlp_up.global_scale, d_yu, 1,
                                           H, kDenseIntermediate, false)
                    != cudaSuccess)
                    { result.error_msg = "dense up"; break; }
                {
                    const int blocks = (kDenseIntermediate + 255) / 256;
                    ::lsxcommon::kernels::SwigluKernel<__half,
                        ::lsxcommon::kernels::ExpFlavor::Glibc, true>
                        <<<blocks, 256, 0, stream>>>(
                            d_yg, d_yu, d_yg, kDenseIntermediate, 10.0f);
                }
                if (lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, lw.mlp_down.d_packed,
                                           lw.mlp_down.d_scales,
                                           lw.mlp_down.global_scale, d_act2,
                                           1, kDenseIntermediate, H, false)
                    != cudaSuccess)
                    { result.error_msg = "dense down"; break; }
            } else {
                result.error_msg = "non-moe layer >= 3";
                e = cudaErrorInvalidValue;
                break;
            }
            if (std::getenv("LSX_TRACE_FLASH") && l >= 3) {
                static std::vector<float> hm;
                hm.resize(H);
                cudaMemcpy(hm.data(), d_module_f, sizeof(float) * H,
                           cudaMemcpyDeviceToHost);
                float hmean = 0;
                int nan = 0;
                for (float v : hm) {
                    if (std::isnan(v)) ++nan;
                    hmean += v;
                }
                std::fprintf(stderr,
                             "[ffn] post-l=%d mean=%.6f nan=%d\n", l,
                             hmean / H, nan);
            }
            // ── ffn hc_post ──
            f16_to_f32_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                d_act2, d_module_f, H);
            gpu::run_hc_post(stream, d_module_f, d_state, d_hc_post, d_hc_comb,
                        d_state);
            if ((e = cudaGetLastError()) != cudaSuccess)
                { result.error_msg = "hc_post2"; break; }
        }
        if (std::getenv("LSX_TRACE_FLASH"))
            std::fprintf(stderr, "[loopend] e=%d err='%s'\n", (int) e,
                         result.error_msg.c_str());
        if (e != cudaSuccess) {
            step_err = e;
            break;
        }

        // ── head: stream collapse -> final norm -> lm_head ──
        // GLM-5.3-Flash has no separate head hc (shared_head.norm is MTP).
        // Collapse variant A: plain mean. LSX_FLASH_HEAD=stream0 selects
        // the first stream; LSX_FLASH_HEAD=mean (default) the average.
        {
            static const int head_mode = [] {
                const char *e = std::getenv("LSX_FLASH_HEAD");
                return e && std::string_view(e) == "stream0" ? 1 : 0;
            }();
            if (head_mode == 1)
                cudaMemcpyAsync(d_act, d_state, H * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
            else
                flash_mean_streams_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
                    d_state, H, hc, d_act);
        }
        flash_rmsnorm_weighted_kernel<<<1, 1024, 1024 * sizeof(float),
                                        stream>>>(
            d_act, pw.d_final_norm, d_act, H, 1e-5f);
        if ((e = cudaGetLastError()) != cudaSuccess)
            { result.error_msg = std::string("final norm: ")
                               + cudaGetErrorString(e); break; }
        // lm_head: BF16 bits [V, H] @ act -> FP32 logits (bf16 gemv).
        flash_gemv_bf16_logits_kernel<<<(V + 7) / 8, 256, 0, stream>>>(
            pw.d_head_bf16, d_act, d_logits, H, V);
        if ((e = cudaGetLastError()) != cudaSuccess)
            { result.error_msg = "head"; break; }

        if (std::getenv("LSX_TRACE_FLASH")) {
            static std::vector<float> hlog;
            hlog.resize(static_cast<size_t>(V));
            cudaMemcpy(hlog.data(), d_logits, sizeof(float) * V,
                       cudaMemcpyDeviceToHost);
            int idx[4] = {0, 0, 0, 0};
            float val[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
            float hmean = 0, hmin = 1e30f, hmax = -1e30f;
            for (int i = 0; i < V; ++i) {
                hmean += hlog[i];
                hmin = std::fmin(hmin, hlog[i]);
                hmax = std::fmax(hmax, hlog[i]);
                for (int k = 0; k < 4; ++k)
                    if (hlog[i] > val[k]) {
                        for (int m = 3; m > k; --m) {
                            val[m] = val[m - 1];
                            idx[m] = idx[m - 1];
                        }
                        val[k] = hlog[i];
                        idx[k] = i;
                        break;
                    }
            }
            std::fprintf(stderr,
                         "[trace] step=%d mean=%.4f min=%.3f max=%.3f "
                         "top: %d:%.3f %d:%.3f %d:%.3f %d:%.3f\n",
                         step, hmean / V, hmin, hmax, idx[0], val[0],
                         idx[1], val[1], idx[2], val[2], idx[3], val[3]);
        }
        if (step >= prompt_steps - 1) {
            if (sampler.Sample(stream, d_logits, V,
                               lsxcommon::sample::LogitType::kF32, d_argmax,
                               d_maxv) != cudaSuccess)
                { result.error_msg = "sample"; break; }
            int32_t h_tok = -1;
            cudaMemcpyAsync(&h_tok, d_argmax,
                            4, cudaMemcpyDeviceToHost, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                { result.error_msg = "sample sync"; break; }
            result.tokens.push_back(h_tok);
            cur = h_tok;
            ++generated;
            bool eos_hit = false;
            for (int id : eos_ids)
                if (h_tok == id) eos_hit = true;
            if (eos_hit) break;
        } else {
            cur = prompt[step + 1];
        }
    }

    if (step_err != cudaSuccess) return step_err;
    if (generated > 0) {
        result.ok = true;
        std::vector<float> hlast(static_cast<size_t>(V));
        if (cudaMemcpy(hlast.data(), d_logits, sizeof(float) * V,
                       cudaMemcpyDeviceToHost) == cudaSuccess)
            result.final_logits = std::move(hlast);
        if (t_stop) {
            cudaEventRecord(t_stop, stream);
            cudaEventSynchronize(t_stop);
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, t_start, t_stop) == cudaSuccess)
                result.total_ms = ms;
        }
    }
    cudaStreamSynchronize(stream);
    cudaStreamSynchronize(h2d_stream);
    for (int k = 0; k < 16; ++k) cudaEventDestroy(copy_ev[k]);
    for (cudaEvent_t ev : slot_free_ev)
        if (ev) cudaEventDestroy(ev);
    cudaStreamDestroy(h2d_stream);
    cudaFreeHost(h_ids_pinned);
    cudaFree(d_argmax);
    cudaFree(d_tokens);
    cudaFree(d_ids);
    cudaFree(d_slots);
    cudaFree(d_gemv_tbl);
    if (t_start) cudaEventDestroy(t_start);
    if (t_stop) cudaEventDestroy(t_stop);
    return result.ok ? cudaSuccess : cudaErrorUnknown;
}


namespace {

__global__ void fp8_f32tile_to_bf16_kernel(const uint8_t *__restrict__ src,
                                           const float *__restrict__ scale_inv,
                                           uint16_t *__restrict__ dst,
                                           int64_t numel, int64_t cols,
                                           int64_t sc_tiles)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
    if (i >= numel) return;
    const int64_t r = i / cols;
    const int64_t c = i % cols;
    const float v = kernels::F8E4m3ToFloat(src[i])
                  * scale_inv[(r / 128) * sc_tiles + (c / 128)];
    // BF16 round-to-nearest-even from F32 (drop 16 low mantissa bits).
    uint32_t bits = __float_as_uint(v);
    bits += 0x7FFF + ((bits >> 16) & 1);
    dst[i] = static_cast<uint16_t>(bits >> 16);
}

struct DevGuard {
    uint8_t *d_fp8 = nullptr;
    float *d_scale = nullptr;
    uint16_t *d_out = nullptr;
    ~DevGuard() {
        cudaFree(d_fp8);
        cudaFree(d_scale);
        cudaFree(d_out);
    }
};

} // namespace

arrow::Status DequantFp8TilesToBf16Gpu(const std::vector<uint8_t> &fp8,
                                       const std::vector<int64_t> &shape,
                                       const std::vector<uint8_t> &scale_f32,
                                       const std::vector<int64_t> &scale_shape,
                                       std::vector<uint8_t> &bf16)
{
    if (shape.size() != 2)
        return arrow::Status::Invalid("FP8 tensor must be 2-D");
    if (scale_shape.size() != 2)
        return arrow::Status::Invalid("scale_inv must be 2-D");
    const int64_t rows = shape[0], cols = shape[1];
    const int64_t sr = scale_shape[0], sc = scale_shape[1];
    if (rows != sr * 128 || cols != sc * 128)
        return arrow::Status::Invalid(
            "scale tile grid mismatch: expect rows/128 x cols/128");
    const int64_t numel = rows * cols;
    if (static_cast<int64_t>(fp8.size()) != numel)
        return arrow::Status::Invalid("fp8 byte size != rows*cols");
    if (static_cast<int64_t>(scale_f32.size()) != sr * sc * 4)
        return arrow::Status::Invalid("scale byte size != SR*SC*4");

    DevGuard g;
    if (cudaMalloc(&g.d_fp8, fp8.size()) != cudaSuccess
        || cudaMalloc(&g.d_scale, scale_f32.size()) != cudaSuccess
        || cudaMalloc(&g.d_out, numel * 2) != cudaSuccess)
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    cudaStream_t stream = nullptr;  // legacy default stream (offline path)
    if (cudaMemcpyAsync(g.d_fp8, fp8.data(), fp8.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess
        || cudaMemcpyAsync(g.d_scale, scale_f32.data(), scale_f32.size(),
                           cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return arrow::Status::IOError("H2D copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    const int threads = 256;
    const int64_t grid64 = (numel + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<int64_t>(grid64, 65535LL * 16));
    fp8_f32tile_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        g.d_fp8, g.d_scale, g.d_out, numel, cols, sc);
    if (cudaGetLastError() != cudaSuccess)
        return arrow::Status::IOError("kernel launch failed: ",
                                      cudaGetErrorString(cudaGetLastError()));

    bf16.resize(static_cast<size_t>(numel) * 2);
    if (cudaMemcpyAsync(bf16.data(), g.d_out, bf16.size(),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return arrow::Status::IOError("D2H copy failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return arrow::Status::IOError("stream sync failed: ",
                                      cudaGetErrorString(cudaGetLastError()));
    return arrow::Status::OK();
}

} // namespace glm_moe_flash
} // namespace lsxcommon
