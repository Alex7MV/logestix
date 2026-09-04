// GLM-5.3-Flash CUDA kernels: KDA (conv1d + per-channel-decay gated delta),
// router top-8 (sigmoid noaux_tc), and the single-stream generation driver.
// Per-channel decay mirrors the FLA kimi-delta semantics: decay applies to
// the KEY dim (l) of the state column, unlike qwen GDN's scalar head decay.
#include "lsxcommon/glm_moe_flash_inference.h"

#include "glm_moe_flash_device_math.h"
#include "glm_moe_flash_constants.h"
#include "lsx_kernels_act.cuh"
#include "lsxcommon/qwen_moe_exp_inference.h"  // RunExpQsaAttention reuse

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

namespace lsxcommon {
namespace glm_moe_flash {
namespace {

FLASH_HD float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + flash_expf_glibc(-x));
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
// One block per token. Score = sigmoid(dot(gate_e, x)) + bias_e computed in
// FP32 over BF16-decoded values; top-8 via linear-scan selection (tie-break
// lower expert index); renorm + scale 2.5. RMW-free per-slot writes: each
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
                (static_cast<uint32_t>(gate_bf16[e * hidden + h * 2])
                 | (static_cast<uint32_t>(gate_bf16[e * hidden + h * 2 + 1])
                    << 8))
                << 16;
            const uint32_t xb =
                (static_cast<uint32_t>(x_bf16[t * hidden + h * 2])
                 | (static_cast<uint32_t>(x_bf16[t * hidden + h * 2 + 1])
                    << 8))
                << 16;
            float g, xv;
            memcpy(&g, &gb, 4);
            memcpy(&xv, &xb, 4);
            dot += g * xv;
        }
        const float s = sigmoid_dev(dot) + bias[e];
        s_scores[e] = s;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    // Thread-0 selection over the staged scores.
    constexpr int kTopK = 8;
    for (int kk = 0; kk < kTopK; ++kk) {
        int best = -1;
        float best_s = -1e30f;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int j = 0; j < kk; ++j)
                if (slots[t * kTopK + j].eid == e) taken = true;
            if (taken) continue;
            if (s_scores[e] > best_s) {
                best_s = s_scores[e];
                best = e;
            }
        }
        slots[t * kTopK + kk].eid = best;
        slots[t * kTopK + kk].coeff = best_s;
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
    const int threads = 256;
    const size_t scratch = static_cast<size_t>(experts) * sizeof(float);
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

// L2-normalize one head row (dim elements) in place; q gets extra scale.
__global__ void flash_l2norm_kernel(__half *row, int dim, float extra_scale)
{
    extern __shared__ float red[];
    const int l = threadIdx.x;
    float v = 0.0f;
    if (l < dim) v = __half2float(row[l]);
    red[l] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        __syncthreads();
    }
    if (l == 0) red[0] = rsqrtf(red[0] + 1e-6f);
    __syncthreads();
    if (l < dim)
        row[l] = __float2half(__half2float(row[l]) * red[0] * extra_scale);
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
    gdecay[idx] = flash_expf_glibc(
        -flash_expf_glibc(A_log[h]) * log1pf(flash_expf_glibc(dt)));
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
    __half *__restrict__ out, int heads, int dim)
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
    out[h * dim + d] = __float2half(acc);
}

// RMSNormGated per head: core[h,dim], gate[h,dim], norm weight [dim].
// out[h,l] = RMSNorm(core)[l] * sigmoid(gate[l])? — GLM KDA o_norm gate
// follows the qwen RMSNormGated: normed = core * rsqrt(mean(core^2)+eps);
// out = normed * sigmoid? — the qwen kernel uses the gate directly. We use
// out = normed[l] * gate[l] (gate post-sigmoid from the g projections).
__global__ void flash_rmsnorm_gated_kernel(const __half *__restrict__ core,
                                           const __half *__restrict__ gate,
                                           const __half *__restrict__ norm_w,
                                           __half *__restrict__ out, int heads,
                                           int dim, float eps)
{
    extern __shared__ float red2[];
    const int h = blockIdx.x;
    const int l = threadIdx.x;
    float v = 0.0f;
    if (l < dim) v = __half2float(core[h * dim + l]);
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

// Gating sigmoid over [heads*dim] in place.
__global__ void flash_gate_sigmoid_kernel(__half *__restrict__ g, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        g[i] = __float2half(sigmoid_dev(__half2float(g[i])));
}

// Conv history shift variant taking separate q/k/v raw rows.
__global__ void flash_conv_shift_kernel2(__half *__restrict__ hist,
                                         const __half *__restrict__ q,
                                         const __half *__restrict__ k,
                                         const __half *__restrict__ v,
                                         int channels, int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int KD = channels / 3;
    const __half *cur = c < KD ? q : (c < 2 * KD ? k - KD : v - 2 * KD);
    for (int kk = 0; kk < kernel - 2; ++kk)
        hist[kk * channels + c] = hist[(kk + 1) * channels + c];
    hist[(kernel - 2) * channels + c] = cur[c];
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
    __half *d_attn = d_g + heads * dim; // [KD] delta output
    __half *d_gated = d_attn + KD;      // [KD]
    float *d_beta = reinterpret_cast<float *>(d_gated + KD);
    float *d_gdecay = d_beta + heads;

    cudaError_t e;
    // 1. q/k/v GEMVs (raw projections).
    if ((e = FlashGemmV16(stream, lw.d_kda_q, d_x, d_q, hidden, KD))
        != cudaSuccess) return e;
    if ((e = FlashGemmV16(stream, lw.d_kda_k, d_x, d_k, hidden, KD))
        != cudaSuccess) return e;
    if ((e = FlashGemmV16(stream, lw.d_kda_v, d_x, d_v, hidden, KD))
        != cudaSuccess) return e;

    // 2. Shift the conv history: slide up, append the current raw rows.
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

    // 3. Depthwise causal conv over [hist(K-1) rows, current raw row].
    {
        const int threads = 256;
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist, d_q, lw.d_kda_q_conv, d_qc, KD, conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist + KD, d_k, lw.d_kda_k_conv, d_kc, KD, conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_conv_decode_kernel<<<(KD + threads - 1) / threads, threads, 0,
                                   stream>>>(
            st.d_conv_hist + 2 * KD, d_v, lw.d_kda_v_conv, d_vc, KD,
            conv_kernel);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 4. L2-normalize q, k per head (q scaled 1/sqrt(dim)).
    {
        const int threads = 128;  // == dim
        const float q_scale = 1.0f / sqrtf(static_cast<float>(dim));
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
            d_qc, d_kc, d_vc, d_beta, d_gdecay, st.d_state, d_attn, heads,
            dim);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 7. RMSNormGated(core=attn, gate=sigmoid(g), o_norm).
    {
        const int threads = 128;
        flash_gate_sigmoid_kernel<<<(heads * dim + 255) / 256, 256, 0,
                                    stream>>>(d_g, heads * dim);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        flash_rmsnorm_gated_kernel<<<heads, threads, threads * sizeof(float),
                                     stream>>>(
            d_attn, d_g, lw.d_o_norm, d_gated, heads, dim, 1e-6f);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    // 8. o_proj: [H, KD] @ gated -> d_out.
    if ((e = FlashGemmV16(stream, lw.d_kda_o, d_gated, d_out, KD, hidden))
        != cudaSuccess) return e;
    return cudaSuccess;
}

// ── MoE clamped-swiglu routed chain ──────────────────────────────────────

cudaError_t RunFlashRoutedFFN(
    cudaStream_t stream, qwen_moe::QwenMoeExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer, int moe_layer,
    const qwen_moe::QwenMoeRoutedSlot *h_slots, int k_top,
    const __half *d_x, int hidden, int moe_inter, float *d_acc,
    __half *d_yg, __half *d_yu, __half *d_y2, std::string &error)
{
    error.clear();
    if (k_top > 16) {
        error = "routed slots exceed staging capacity";
        return cudaErrorInvalidValue;
    }
    const int store_layer = moe_layer - frozen::kMoeFirstLayer;
    int dev_slots[16];
    for (int k = 0; k < k_top; ++k) {
        std::string ensure_err;
        const int slot = qwen_moe::EnsureQwenMoeExpertResident(
            store, store_layer, h_slots[k].eid, cache, stream, ensure_err);
        if (slot < 0) {
            error = "ensure resident failed: " + ensure_err;
            return cudaErrorInvalidValue;
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
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, w1, s1, rec.gs1, d_yg, 1,
                                   hidden, moe_inter, false) != cudaSuccess)
            { error = "w1 gemm failed"; return cudaErrorInvalidValue; }
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, w3, s3, rec.gs3, d_yu, 1,
                                   hidden, moe_inter, false) != cudaSuccess)
            { error = "w3 gemm failed"; return cudaErrorInvalidValue; }
        {
            const int blocks = (moe_inter + 255) / 256;
            ::lsxcommon::kernels::SwigluKernel<__half,
                ::lsxcommon::kernels::ExpFlavor::Glibc, true>
                <<<blocks, 256, 0, stream>>>(d_yg, d_yu, d_yg, moe_inter,
                                             10.0f);
        }
        if (qwen::RunQwenNvfp4Gemm(stream, d_yg, w2, s2, rec.gs2, d_y2, 1,
                                   moe_inter, hidden, false) != cudaSuccess)
            { error = "w2 gemm failed"; return cudaErrorInvalidValue; }
        // acc += coeff * y2
        flash_coeff_add_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
            d_y2, h_slots[k].coeff, d_acc, hidden);
    }
    return cudaGetLastError();
}

cudaError_t RunFlashSharedFFN(cudaStream_t stream, const FlashLayerWeights &lw,
                              const __half *d_x, int hidden, int shared_inter,
                              __half *d_yg, __half *d_yu, __half *d_y2,
                              __half *d_out)
{
    if (qwen::RunQwenNvfp4Gemm(stream, d_x, lw.shared_w1.d_packed,
                               lw.shared_w1.d_scales, lw.shared_w1.global_scale,
                               d_yg, 1, hidden, shared_inter, false)
        != cudaSuccess) return cudaErrorUnknown;
    if (qwen::RunQwenNvfp4Gemm(stream, d_x, lw.shared_w3.d_packed,
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
    if (qwen::RunQwenNvfp4Gemm(stream, d_yg, lw.shared_w2.d_packed,
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
    result.error_msg = "generation driver not yet implemented";
    return cudaErrorNotYetImplemented;
}

} // namespace glm_moe_flash
} // namespace lsxcommon
