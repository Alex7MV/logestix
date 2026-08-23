#include "kimi_inference_kernels.h"

#include <cuda_bf16.h>

namespace lsxcommon {
namespace kimi {

__global__ void kimi_int4_dequant_kernel(const int *__restrict__ packed,
                                         const uint16_t *__restrict__ scales,
                                         uint16_t *__restrict__ out,
                                         int cols)
{
    const int r = blockIdx.y;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (c0 >= cols) return;
    const int word = packed[static_cast<size_t>(r) * (cols >> 3) + (c0 >> 3)];
    const float s = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(
        &scales[static_cast<size_t>(r) * (cols >> 5) + (c0 >> 5)]));
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int nib = (word >> (4 * k)) & 0xF;
        const float v =
            static_cast<float>(nib > 7 ? nib - 16 : nib) * s;
        __nv_bfloat16 h(v);
        out[static_cast<size_t>(r) * cols + c0 + k] =
            *reinterpret_cast<uint16_t *>(&h);
    }
}

cudaError_t KimiInt4DequantToBf16(const int *d_packed,
                                  const uint16_t *d_scales_bf16,
                                  int rows, int cols,
                                  uint16_t *d_out_bf16,
                                  cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((cols / 8 + block.x - 1) / block.x, rows);
    kimi_int4_dequant_kernel<<<grid, block, 0, stream>>>(
        d_packed, d_scales_bf16, d_out_bf16, cols);
    return cudaGetLastError();
}

} // namespace kimi
} // namespace lsxcommon


// ── Router (noaux_tc sigmoid top-k) ────────────────────────────────────────

#include <limits>
#include <vector>

namespace lsxcommon {
namespace kimi {

void KimiRouterTopKHost(const float *scores, const float *bias, int n_routed,
                        int top_k, int *out_ids, float *out_weights)
{
    std::vector<char> chosen(static_cast<size_t>(n_routed), 0);
    for (int t = 0; t < top_k; ++t) {
        int best = -1;
        float best_key = -std::numeric_limits<float>::infinity();
        for (int e = 0; e < n_routed; ++e) {
            if (chosen[e]) continue;
            float s = 1.0f / (1.0f + std::exp(-scores[e]));
            float key = s + bias[e];
            // Strict > keeps the earliest (lowest-id) expert on ties.
            if (key > best_key) {
                best_key = key;
                best = e;
            }
        }
        out_ids[t] = best;
        chosen[best] = 1;
    }
    float sum = 0.0f;
    for (int t = 0; t < top_k; ++t)
        sum += 1.0f / (1.0f + std::exp(-scores[out_ids[t]]));
    for (int t = 0; t < top_k; ++t) {
        out_weights[t] =
            (1.0f / (1.0f + std::exp(-scores[out_ids[t]])) / sum) *
            kKimiRoutedScalingFactor;
    }
}

// One block, one thread per routed expert (block padded to 512). Per round
// every thread publishes its candidate into shared memory, the block reduces
// in a fixed order (lowest thread index wins ties), thread 0 commits the
// winner behind __syncthreads() barriers — no RMW races, fully deterministic.
constexpr int kRouterMaxBlock = 512;

__global__ void kimi_router_topk_kernel(const float *__restrict__ scores,
                                        const float *__restrict__ bias,
                                        int n_routed, int top_k,
                                        int *__restrict__ out_ids,
                                        float *__restrict__ out_weights)
{
    __shared__ float s_cand[kRouterMaxBlock];
    __shared__ int   s_cand_idx[kRouterMaxBlock];
    __shared__ char  s_chosen[kRouterMaxBlock];
    __shared__ int   s_winner;

    const int e = threadIdx.x;
    const bool active = e < n_routed;
    if (active)
        s_chosen[e] = 0;
    __syncthreads();

    for (int round = 0; round < top_k; ++round) {
        float cand = -INFINITY;
        int idx = -1;
        if (active && !s_chosen[e]) {
            float s = 1.0f / (1.0f + __expf(-scores[e]));
            cand = s + bias[e];
            idx = e;
        }
        s_cand[e] = cand;
        s_cand_idx[e] = idx;
        if (e == 0) s_winner = -1;
        __syncthreads();

        // Deterministic reduction: fixed scan order, first-max wins.
        if (e == 0) {
            float best = -INFINITY;
            int bi = -1;
            for (int i = 0; i < kRouterMaxBlock; ++i) {
                if (s_cand_idx[i] < 0) continue;
                if (s_cand[i] > best) {
                    best = s_cand[i];
                    bi = s_cand_idx[i];
                }
            }
            s_winner = bi;
        }
        __syncthreads();

        if (e == 0) {
            out_ids[round] = s_winner;
            if (s_winner >= 0) s_chosen[s_winner] = 1;
        }
        __syncthreads();
    }

    if (e == 0) {
        float sum = 0.0f;
        for (int t = 0; t < top_k; ++t)
            sum += 1.0f / (1.0f + __expf(-scores[out_ids[t]]));
        for (int t = 0; t < top_k; ++t)
            out_weights[t] =
                (1.0f / (1.0f + __expf(-scores[out_ids[t]])) / sum) *
                kKimiRoutedScalingFactor;
    }
}

cudaError_t KimiRouterTopK(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k, int *d_out_ids,
                           float *d_out_weights, cudaStream_t stream)
{
    kimi_router_topk_kernel<<<1, kRouterMaxBlock, 0, stream>>>(
        d_scores, d_bias, n_routed, top_k, d_out_ids, d_out_weights);
    return cudaGetLastError();
}

} // namespace kimi
} // namespace lsxcommon

// ── MLA attention (DeepSeek-V3 math, absorbed decode form) ─────────────────

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>

namespace lsxcommon {
namespace kimi {

namespace {

constexpr int kRopePairs = kKimiQkRope / 2;
// YaRN attention-scale multiplier: mscale / sqrt(qk_nope + qk_rope).
__device__ __host__ __forceinline__ float KimiAttnScale()
{
    return (0.1f * logf(kKimiYarnFactor) + 1.0f) *
           rsqrtf(static_cast<float>(kKimiQkNope + kKimiQkRope));
}

double YarnCorrectionDim(double num_rotations, double dim, double base,
                         double max_pos)
{
    return dim * std::log(max_pos / (num_rotations * 2.0 * M_PI)) /
           (2.0 * std::log(base));
}

std::vector<float> BuildYarnInvFreq()
{
    const double dim = kKimiQkRope;
    const double base = kKimiRopeTheta;
    const double max_pos = kKimiYarnOriginalMaxPos;
    const double low = std::floor(YarnCorrectionDim(32.0, dim, base, max_pos));
    const double high = std::ceil(YarnCorrectionDim(1.0, dim, base, max_pos));
    const double wl_low = 2.0 * M_PI * std::pow(base, 2.0 * low / dim);
    const double wl_high = 2.0 * M_PI * std::pow(base, 2.0 * high / dim);

    std::vector<float> inv_freq(kRopePairs);
    for (int i = 0; i < kRopePairs; ++i) {
        const double f0 = std::pow(base, -2.0 * i / dim);
        const double wavelen = 2.0 * M_PI / f0;
        double out;
        if (wavelen < wl_low) {
            out = f0;
        } else if (wavelen > wl_high) {
            out = f0 / kKimiYarnFactor;
        } else {
            const double alpha = (wavelen - wl_high) / (wl_low - wl_high);
            out = (1.0 - alpha) * (f0 / kKimiYarnFactor) + alpha * f0;
        }
        inv_freq[i] = static_cast<float>(out);
    }
    return inv_freq;
}

// Per-token postprocess of the kv_a output chunk [576]:
//   [0..512): RMSNorm(latent) with kv_a_layernorm weight
//   [512..576): yarn-rope the shared key branch in place
// and yarn-rope every head's rope part of q_all ([12288 per token]).
__global__ void kimi_kv_postprocess_kernel(__half *__restrict__ kvfull,
                                           __half *__restrict__ q_all,
                                           const __half *__restrict__ kv_a_norm,
                                           int seq,
                                           const float *__restrict__ inv_freq)
{
    const int token = blockIdx.x;
    __shared__ float s_ss;

    __half *c = kvfull + (size_t)token * kKimiKvaRows;
    if (threadIdx.x == 0) s_ss = 0.0f;
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kKimiKvLoraRank; i += blockDim.x) {
        float v = __half2float(c[i]);
        ss += v * v;
    }
    atomicAdd(&s_ss, ss);
    __syncthreads();
    const float inv =
        rsqrtf(s_ss / static_cast<float>(kKimiKvLoraRank) + kKimiRmsEps);
    for (int i = threadIdx.x; i < kKimiKvLoraRank; i += blockDim.x)
        c[i] = __float2half(__half2float(c[i]) *
                            __half2float(kv_a_norm[i]) * inv);
    __syncthreads();

    for (int pair = threadIdx.x; pair < kRopePairs; pair += blockDim.x) {
        __half *kr = c + kKimiKvLoraRank;
        float freq = inv_freq[pair];
        float cs = cosf(freq * static_cast<float>(token));
        float sn = sinf(freq * static_cast<float>(token));
        float x = __half2float(kr[pair]);
        float y = __half2float(kr[kRopePairs + pair]);
        kr[pair] = __float2half(x * cs - y * sn);
        kr[kRopePairs + pair] = __float2half(x * sn + y * cs);
    }

    __half *qtok = q_all +
        (size_t)token * (kKimiNumHeads * (kKimiQkNope + kKimiQkRope));
    for (int h = 0; h < kKimiNumHeads; ++h) {
        __half *qr = qtok + h * (kKimiQkNope + kKimiQkRope) + kKimiQkNope;
        for (int pair = threadIdx.x; pair < kRopePairs; pair += blockDim.x) {
            float freq = inv_freq[pair];
            float cs = cosf(freq * static_cast<float>(token));
            float sn = sinf(freq * static_cast<float>(token));
            float x = __half2float(qr[pair]);
            float y = __half2float(qr[kRopePairs + pair]);
            qr[pair] = __float2half(x * cs - y * sn);
            qr[kRopePairs + pair] = __float2half(x * sn + y * cs);
        }
    }
}

// Appends this turn's latents + roped keys to cache slots.
__global__ void kimi_mla_append_kernel(KimiMlaState *__restrict__ st,
                                       const __half *__restrict__ kvfull,
                                       int seq)
{
    __shared__ int s_slot;
    if (threadIdx.x == 0)
        s_slot = atomicAdd(&st->n_used, 1);
    __syncthreads();
    const int slot = s_slot;
    const int token = blockIdx.x;
    const int j = threadIdx.x;
    const __half *c = kvfull + (size_t)token * kKimiKvaRows;
    if (j < kKimiKvLoraRank) st->kv[slot][j] = c[j];
    else if (j < kKimiKvLoraRank + kKimiQkRope)
        st->k_rope[slot][j - kKimiKvLoraRank] =
            c[kKimiKvLoraRank + (j - kKimiKvLoraRank)];
}

// Residual add: d_out[t*H+i] = hidden + proj.
__global__ void kimi_residual_add_kernel(const __half *__restrict__ hidden,
                                         const __half *__restrict__ proj,
                                         __half *__restrict__ out, int n_total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    out[i] = __float2half(__half2float(hidden[i]) + __half2float(proj[i]));
}

// Fused absorbed attention: grid (seq, heads), block 512 threads.
__global__ void kimi_mla_attn_kernel(
    const __half *__restrict__ q_all,         // [12288 x seq] col-major fp16
    const __half *__restrict__ kv_b,          // [16384 x 512] row-major fp16
    const KimiMlaState *__restrict__ st,
    __half *__restrict__ attn_out,            // [seq x 7168] row-major fp16
    int seq)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t qbase =
        (size_t)token * (kKimiNumHeads * (kKimiQkNope + kKimiQkRope)) +
        head * (kKimiQkNope + kKimiQkRope);
    const size_t kvrow_k = (size_t)head * 256;
    const size_t kvrow_v = (size_t)head * 256 + kKimiQkNope;
    const int n_slots = st->n_used;

    __shared__ float s_qeff[kKimiKvLoraRank];
    __shared__ float s_exp[1024];
    __shared__ float s_m[kKimiKvLoraRank];
    __shared__ float s_red[kRouterMaxBlock + 1];
    __shared__ float s_blockmax;
    __shared__ float s_denom;

    // Phase A: q_eff[j] = Σ_i q_nope[i] * kvb_k[i][j]
    // (absorbed W_k^T·q; key rows of this head are kvb[kvrow_k + i]).
    for (int j = tid; j < kKimiKvLoraRank; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < kKimiQkNope; ++i)
            acc += __half2float(q_all[qbase + i]) *
                   __half2float(kv_b[(kvrow_k + i) * kKimiKvLoraRank + j]);
        s_qeff[j] = acc;
    }
    __syncthreads();

    // Phase B1: raw scores, deterministic block max.
    float local_max = -INFINITY;
    for (int s = tid; s < n_slots; s += blockDim.x) {
        const __half *c = st->kv[s];
        const __half *kr = st->k_rope[s];
        float acc = 0.0f, racc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += s_qeff[j] * __half2float(c[j]);
        for (int j = 0; j < kRopePairs * 2; ++j)
            racc += __half2float(q_all[qbase + kKimiQkNope + j]) *
                    __half2float(kr[j]);
        float score = (acc + racc) * KimiAttnScale();
        if (score > local_max) local_max = score;
    }
    s_red[tid] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = -INFINITY;
        for (int i = 0; i < blockDim.x; ++i) m = fmaxf(m, s_red[i]);
        s_blockmax = m;
    }
    __syncthreads();

    // Phase B2: exponentials + block sum.
    float local_sum = 0.0f;
    for (int s = tid; s < n_slots; s += blockDim.x) {
        const __half *c = st->kv[s];
        const __half *kr = st->k_rope[s];
        float acc = 0.0f, racc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += s_qeff[j] * __half2float(c[j]);
        for (int j = 0; j < kRopePairs * 2; ++j)
            racc += __half2float(q_all[qbase + kKimiQkNope + j]) *
                    __half2float(kr[j]);
        float ev = __expf((acc + racc) * KimiAttnScale() - s_blockmax);
        s_exp[s] = ev;
        local_sum += ev;
    }
    s_red[tid] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) z += s_red[i];
        s_denom = z;
    }
    __syncthreads();

    // Phase C: weighted latent mean m[j].
    for (int j = tid; j < kKimiKvLoraRank; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_slots; ++s)
            acc += s_exp[s] * __half2float(st->kv[s][j]);
        s_m[j] = acc / s_denom;
    }
    __syncthreads();

    // Phase D: out[j2] = dot(kvb_v[j2], m).
    for (int j2 = tid; j2 < kKimiHeadV; j2 += blockDim.x) {
        const __half *wrow = &kv_b[(kvrow_v + j2) * kKimiKvLoraRank];
        float acc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += __half2float(wrow[j]) * s_m[j];
        attn_out[(size_t)token * kKimiHidden +
                 head * kKimiHeadV + j2] = __float2half(acc);
    }
}

} // namespace

cudaError_t KimiMlaReset(KimiMlaState *d_state, cudaStream_t stream)
{
    return cudaMemsetAsync(d_state, 0, sizeof(KimiMlaState), stream);
}

namespace {

// RMSNorm of contiguous FP16 rows [seq x n] in place.
__global__ void kimi_rmsnorm_rows_kernel(__half *__restrict__ x,
                                         const __half *__restrict__ w, int n)
{
    const int token = blockIdx.x;
    __shared__ float s_ss;
    if (threadIdx.x == 0) s_ss = 0.0f;
    __syncthreads();
    __half *row = x + (size_t)token * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += __half2float(row[i]) * __half2float(row[i]);
    atomicAdd(&s_ss, ss);
    __syncthreads();
    const float inv = rsqrtf(s_ss / static_cast<float>(n) + kKimiRmsEps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row[i] =
            __float2half(__half2float(row[i]) * __half2float(w[i]) * inv);
}

} // namespace

cudaError_t RunKimiMlaAttention(cudaStream_t stream, int turn,
                                const KimiLayerWeights &lw, int seq,
                                const __half *d_hidden, __half *d_out,
                                KimiMlaState *d_state,
                                cublasHandle_t cublas_h)
{
    if (!lw.d_q_a_proj || !lw.d_kv_a_proj || !lw.d_o_proj || !d_state ||
        seq <= 0 || turn < 0)
        return cudaErrorInvalidValue;

    cublasSetStream(cublas_h, stream);
    const float alpha = 1.0f, beta0 = 0.0f;

    static const std::vector<float> kInvFreq = BuildYarnInvFreq();
    static float *d_inv_freq = nullptr;
    if (!d_inv_freq) {
        if (cudaMalloc(&d_inv_freq, kRopePairs * sizeof(float)) !=
            cudaSuccess)
            return cudaErrorMemoryAllocation;
        cudaMemcpy(d_inv_freq, kInvFreq.data(), kRopePairs * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    __half *d_qa = nullptr;     // [seq x 1536]
    __half *d_qall = nullptr;   // [seq x 12288]
    __half *d_kvfull = nullptr; // [seq x 576]
    __half *d_attn = nullptr;   // [seq x 7168]
    __half *d_oproj = nullptr;  // [seq x 7168]
    cudaError_t e = cudaSuccess;
#define KIMI_ALLOC(var, count)                                             \
    if (e == cudaSuccess)                                                  \
        e = cudaMallocAsync(reinterpret_cast<void **>(&var),               \
                            static_cast<size_t>(count) * sizeof(__half),   \
                            stream)
    KIMI_ALLOC(d_qa, 1536 * seq);
    KIMI_ALLOC(d_qall, 12288 * seq);
    KIMI_ALLOC(d_kvfull, 576 * seq);
    KIMI_ALLOC(d_attn, 7168 * seq);
    KIMI_ALLOC(d_oproj, 7168 * seq);
#undef KIMI_ALLOC
    if (e != cudaSuccess) return e;

    // 1. q_a projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiQLoraRank, seq, kKimiHidden, &alpha,
                 lw.d_q_a_proj, CUDA_R_16F, kKimiHidden,
                 d_hidden, CUDA_R_16F, kKimiHidden, &beta0,
                 d_qa, CUDA_R_16F, kKimiQLoraRank,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 2. RMSNorm(q_a).
    kimi_rmsnorm_rows_kernel<<<dim3(seq), 256, 0, stream>>>(
        d_qa, lw.d_q_a_norm, kKimiQLoraRank);

    // 3. q_b projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiQbRows, seq, kKimiQLoraRank, &alpha,
                 lw.d_q_b_proj, CUDA_R_16F, kKimiQLoraRank,
                 d_qa, CUDA_R_16F, kKimiQLoraRank, &beta0,
                 d_qall, CUDA_R_16F, kKimiQbRows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 4. kv_a projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiKvaRows, seq, kKimiHidden, &alpha,
                 lw.d_kv_a_proj, CUDA_R_16F, kKimiHidden,
                 d_hidden, CUDA_R_16F, kKimiHidden, &beta0,
                 d_kvfull, CUDA_R_16F, kKimiKvaRows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 5. Postprocess: latent norm + key/query yarn rope.
    kimi_kv_postprocess_kernel<<<dim3(seq), 256, 0, stream>>>(
        d_kvfull, d_qall, lw.d_kv_a_norm, seq, d_inv_freq);

    // 6. Append latents+keys to cache slots.
    kimi_mla_append_kernel<<<dim3(seq), kKimiKvLoraRank + kKimiQkRope, 0,
                             stream>>>(d_state, d_kvfull, seq);

    // 7. Fused absorbed attention.
    dim3 attn_grid(seq, kKimiNumHeads);
    kimi_mla_attn_kernel<<<attn_grid, 512,
                           0, stream>>>(d_qall, lw.d_kv_b_proj, d_state,
                                        d_attn, seq);

    // 8. o_proj.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiORows, seq, kKimiNumHeads * kKimiHeadV, &alpha,
                 lw.d_o_proj, CUDA_R_16F, kKimiNumHeads * kKimiHeadV,
                 d_attn, CUDA_R_16F, kKimiNumHeads * kKimiHeadV, &beta0,
                 d_oproj, CUDA_R_16F, kKimiORows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 9. Residual add.
    {
        int total = seq * kKimiHidden;
        int block = 256;
        int grid = (total + block - 1) / block;
        kimi_residual_add_kernel<<<grid, block, 0, stream>>>(
            d_hidden, d_oproj, d_out, total);
    }

    // 10. Free the stream-ordered workspace (async, ordered after use).
    cudaFreeAsync(d_qa, stream);
    cudaFreeAsync(d_qall, stream);
    cudaFreeAsync(d_kvfull, stream);
    cudaFreeAsync(d_attn, stream);
    cudaFreeAsync(d_oproj, stream);

    return cudaGetLastError();
}

} // namespace kimi
} // namespace lsxcommon
