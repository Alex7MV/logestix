#include "kimi_inference_kernels.h"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_quant.cuh"
#include "nvfp4_codec.h"

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
        // compressed-tensors pack-quantized symmetric int4: unsigned nibble
        // with implicit zero-point 8 (offset binary), low nibble first.
        const int nib = (word >> (4 * k)) & 0xF;
        const float v = static_cast<float>(nib - 8) * s;
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

size_t KimiExpertFfnStagingBytes()
{
    constexpr int I = kKimiMoeIntermediate;
    constexpr int Hd = kKimiHidden;
    const int64_t w13_el = static_cast<int64_t>(I) * Hd;
    auto align_up = [](size_t v) { return (v + 255) / 256 * 256; };
    size_t packed = align_up(w13_el / 2) * 2 + align_up(static_cast<size_t>(Hd) * I / 2);
    size_t scales = align_up(w13_el / 16) * 2 + align_up(static_cast<size_t>(Hd) * I / 16);
    size_t dq = w13_el * sizeof(__nv_bfloat16) * 3;
    size_t acts = (I * 3 + Hd) * sizeof(__nv_bfloat16);
    return align_up(packed + scales) + dq + acts;
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
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdlib>
#include <vector>

namespace lsxcommon {
namespace kimi {

namespace {

constexpr int kRopePairs = kKimiQkRope / 2;
// YaRN attention-scale: softmax_scale = q_head_dim^-0.5 * attn_factor.
// Kimi config sets mscale = mscale_all_dim = 1.0, so the HF factor
// yarn_get_mscale(f,mscale)/yarn_get_mscale(f,mscale_all_dim) is 1.0
// and the scale degenerates to the plain inverse sqrt (no mscale boost).
__device__ __host__ __forceinline__ float KimiAttnScale()
{
    return rsqrtf(static_cast<float>(kKimiQkNope + kKimiQkRope));
}

double YarnCorrectionDim(double num_rotations, double dim, double base,
                         double max_pos)
{
    return dim * std::log(max_pos / (num_rotations * 2.0 * M_PI)) /
           (2.0 * std::log(base));
}

std::vector<float> BuildYarnInvFreq()
{
    // Exact HF DeepseekV3RotaryEmbedding._set_cos_sin_cache math:
    //   ramp(i)   = clamp((i - low) / (high - low), 0, 1)
    //   inv_freq  = freq_inter * ramp + freq_extra * (1 - ramp)
    // interpolated linearly over the PAIR INDEX (not the wavelength).
    const double dim = kKimiQkRope;
    const double base = kKimiRopeTheta;
    const double max_pos = kKimiYarnOriginalMaxPos;
    // Kimi rope_scaling config: beta_fast = 32.0, beta_slow = 1.0
    // (NOT the DeepSeek default beta_slow = 128).
    const double low =
        std::floor(YarnCorrectionDim(32.0, dim, base, max_pos));
    const double high =
        std::ceil(YarnCorrectionDim(1.0, dim, base, max_pos));

    std::vector<float> inv_freq(kRopePairs);
    for (int i = 0; i < kRopePairs; ++i) {
        const double extra = std::pow(base, -2.0 * i / dim);
        const double inter = extra / kKimiYarnFactor;
        double ramp = (high != low)
                          ? (static_cast<double>(i) - low) / (high - low)
                          : 1.0;
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        inv_freq[i] = static_cast<float>(inter * ramp + extra * (1.0 - ramp));
    }
    return inv_freq;
}

// Per-token postprocess of the kv_a output chunk [576]:
//   [0..512): RMSNorm(latent) with kv_a_layernorm weight
//   [512..576): yarn-rope the shared key branch in place
// and yarn-rope every head's rope part of q_all ([12288 per token]).
__global__ void kimi_kv_postprocess_kernel(__nv_bfloat16 *__restrict__ kvfull,
                                           __nv_bfloat16 *__restrict__ q_all,
                                           const __nv_bfloat16 *__restrict__ kv_a_norm,
                                           int pos_offset,
                                           const float *__restrict__ inv_freq)
{
    const int token = blockIdx.x;
    const int abs_pos = pos_offset + token;
    __shared__ float s_ss;

    __nv_bfloat16 *c = kvfull + (size_t)token * kKimiKvaRows;
    // Deterministic block reduction (fixed-order partials + thread-0 commit):
    // float atomicAdd would introduce run-to-run rounding nondeterminism.
    __shared__ float s_part[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kKimiKvLoraRank; i += blockDim.x) {
        float v = __bfloat162float(c[i]);
        ss += v * v;
    }
    s_part[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) total += s_part[i];
        s_ss = total;
    }
    __syncthreads();
    const float inv =
        rsqrtf(s_ss / static_cast<float>(kKimiKvLoraRank) + kKimiRmsEps);
    for (int i = threadIdx.x; i < kKimiKvLoraRank; i += blockDim.x)
        c[i] = __float2bfloat16_rn(__bfloat162float(c[i]) *
                            __bfloat162float(kv_a_norm[i]) * inv);
    __syncthreads();

    for (int pair = threadIdx.x; pair < kRopePairs; pair += blockDim.x) {
        __nv_bfloat16 *kr = c + kKimiKvLoraRank;
        float freq = inv_freq[pair];
        float cs = cosf(freq * static_cast<float>(abs_pos));
        float sn = sinf(freq * static_cast<float>(abs_pos));
        float x = __bfloat162float(kr[pair]);
        float y = __bfloat162float(kr[kRopePairs + pair]);
        kr[pair] = __float2bfloat16_rn(x * cs - y * sn);
        kr[kRopePairs + pair] = __float2bfloat16_rn(x * sn + y * cs);
    }

    __nv_bfloat16 *qtok = q_all +
        (size_t)token * (kKimiNumHeads * (kKimiQkNope + kKimiQkRope));
    for (int h = 0; h < kKimiNumHeads; ++h) {
        __nv_bfloat16 *qr = qtok + h * (kKimiQkNope + kKimiQkRope) + kKimiQkNope;
        for (int pair = threadIdx.x; pair < kRopePairs; pair += blockDim.x) {
            float freq = inv_freq[pair];
            float cs = cosf(freq * static_cast<float>(abs_pos));
            float sn = sinf(freq * static_cast<float>(abs_pos));
            float x = __bfloat162float(qr[pair]);
            float y = __bfloat162float(qr[kRopePairs + pair]);
            qr[pair] = __float2bfloat16_rn(x * cs - y * sn);
            qr[kRopePairs + pair] = __float2bfloat16_rn(x * sn + y * cs);
        }
    }
}

// Appends this turn's latents + roped keys to cache slots.
__global__ void kimi_mla_append_kernel(KimiMlaState *__restrict__ st,
                                       const __nv_bfloat16 *__restrict__ kvfull,
                                       int seq, int base_slot)
{
    const int slot = base_slot + blockIdx.x;
    const int token = blockIdx.x;
    const int j = threadIdx.x;
    if (token == 0 && j == 0) st->n_used = base_slot + seq;
    const __nv_bfloat16 *c = kvfull + (size_t)token * kKimiKvaRows;
    if (j < kKimiKvLoraRank) st->kv[slot][j] = c[j];
    else if (j < kKimiKvLoraRank + kKimiQkRope)
        st->k_rope[slot][j - kKimiKvLoraRank] =
            c[kKimiKvLoraRank + (j - kKimiKvLoraRank)];
}

// Residual adds: unified kernels::AddOutKernel/AddInPlaceKernel.

// Fused absorbed attention: grid (seq, heads), block 512 threads.
__global__ void kimi_mla_attn_kernel(
    const __nv_bfloat16 *__restrict__ q_all,         // [12288 x seq] col-major fp16
    const __nv_bfloat16 *__restrict__ kv_b,          // [16384 x 512] row-major fp16
    const KimiMlaState *__restrict__ st,
    __nv_bfloat16 *__restrict__ attn_out,            // [seq x 7168] row-major fp16
    int seq, int base_slot)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    // Causal visibility: this token may attend to slots strictly below its
    // own chronological position (its own slot included).
    const int visible = base_slot + token + 1;

    const size_t qbase =
        (size_t)token * (kKimiNumHeads * (kKimiQkNope + kKimiQkRope)) +
        head * (kKimiQkNope + kKimiQkRope);
    const size_t kvrow_k = (size_t)head * 256;
    const size_t kvrow_v = (size_t)head * 256 + kKimiQkNope;
    const int n_slots = visible;

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
            acc += __bfloat162float(q_all[qbase + i]) *
                   __bfloat162float(kv_b[(kvrow_k + i) * kKimiKvLoraRank + j]);
        s_qeff[j] = acc;
    }
    __syncthreads();

    // Phase B1: raw scores, deterministic block max.
    float local_max = -INFINITY;
    for (int s = tid; s < n_slots; s += blockDim.x) {
        const __nv_bfloat16 *c = st->kv[s];
        const __nv_bfloat16 *kr = st->k_rope[s];
        float acc = 0.0f, racc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += s_qeff[j] * __bfloat162float(c[j]);
        for (int j = 0; j < kRopePairs * 2; ++j)
            racc += __bfloat162float(q_all[qbase + kKimiQkNope + j]) *
                    __bfloat162float(kr[j]);
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
        const __nv_bfloat16 *c = st->kv[s];
        const __nv_bfloat16 *kr = st->k_rope[s];
        float acc = 0.0f, racc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += s_qeff[j] * __bfloat162float(c[j]);
        for (int j = 0; j < kRopePairs * 2; ++j)
            racc += __bfloat162float(q_all[qbase + kKimiQkNope + j]) *
                    __bfloat162float(kr[j]);
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
            acc += s_exp[s] * __bfloat162float(st->kv[s][j]);
        s_m[j] = acc / s_denom;
    }
    __syncthreads();

    // Phase D: out[j2] = dot(kvb_v[j2], m).
    for (int j2 = tid; j2 < kKimiHeadV; j2 += blockDim.x) {
        const __nv_bfloat16 *wrow = &kv_b[(kvrow_v + j2) * kKimiKvLoraRank];
        float acc = 0.0f;
        for (int j = 0; j < kKimiKvLoraRank; ++j)
            acc += __bfloat162float(wrow[j]) * s_m[j];
        attn_out[(size_t)token * (kKimiNumHeads * kKimiHeadV) +
                 head * kKimiHeadV + j2] = __float2bfloat16_rn(acc);
    }
}

} // namespace

cudaError_t KimiMlaReset(KimiMlaState *d_state, cudaStream_t stream)
{
    return cudaMemsetAsync(d_state, 0, sizeof(KimiMlaState), stream);
}

namespace {

// RMSNorm of contiguous FP16 rows [seq x n] in place.
__global__ void kimi_rmsnorm_rows_kernel(__nv_bfloat16 *__restrict__ x,
                                          const __nv_bfloat16 *__restrict__ w, int n)
{
    const int token = blockIdx.x;
    __shared__ float s_ss;
    __shared__ float s_part[256];
    __nv_bfloat16 *row = x + (size_t)token * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += __bfloat162float(row[i]) * __bfloat162float(row[i]);
    s_part[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) total += s_part[i];
        s_ss = total;
    }
    __syncthreads();
    const float inv = rsqrtf(s_ss / static_cast<float>(n) + kKimiRmsEps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row[i] =
            __float2bfloat16_rn(__bfloat162float(row[i]) * __bfloat162float(w[i]) * inv);
}

} // namespace

cudaError_t RunKimiMlaAttention(cudaStream_t stream, int turn,
                                const KimiLayerWeights &lw, int seq,
                                const __nv_bfloat16 *d_hidden,
                                const __nv_bfloat16 *d_residual, __nv_bfloat16 *d_out,
                                KimiMlaState *d_state,
                                cublasHandle_t cublas_h, int base_slot,
                                int pos_offset)
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

    // Persistent workspace (single-consumer contract, like GlmMoEArena):
    // grown on demand, never freed until process exit. Avoids per-token
    // allocation churn entirely.
    constexpr size_t kMaxSeqChunk = 4096;
    const size_t need_elems =
        static_cast<size_t>(seq) *
        (1536 + 12288 + 576 + kKimiNumHeads * kKimiHeadV +
         kKimiHidden);
    (void)kMaxSeqChunk;
    static __nv_bfloat16 *ws = nullptr;
    static size_t ws_elems = 0;
    if (ws_elems < need_elems) {
        if (ws) cudaFree(ws);
        if (cudaMalloc(&ws, need_elems * sizeof(__nv_bfloat16)) != cudaSuccess) {
            ws = nullptr;
            ws_elems = 0;
            return cudaErrorMemoryAllocation;
        }
        ws_elems = need_elems;
    }
    __nv_bfloat16 *d_qa = ws;                 // [seq x 1536]
    __nv_bfloat16 *d_qall = d_qa + (size_t)seq * 1536;      // [seq x 12288]
    __nv_bfloat16 *d_kvfull = d_qall + (size_t)seq * 12288; // [seq x 576]
    __nv_bfloat16 *d_attn =
        d_kvfull + (size_t)seq * 576;                // [seq x 8192]
    __nv_bfloat16 *d_oproj =
        d_attn + (size_t)seq * (kKimiNumHeads * kKimiHeadV); // [seq x 7168]

    // 1. q_a projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiQLoraRank, seq, kKimiHidden, &alpha,
                 lw.d_q_a_proj, CUDA_R_16BF, kKimiHidden,
                 d_hidden, CUDA_R_16BF, kKimiHidden, &beta0,
                 d_qa, CUDA_R_16BF, kKimiQLoraRank,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

    // 2. RMSNorm(q_a).
    kimi_rmsnorm_rows_kernel<<<dim3(seq), 256, 0, stream>>>(
        d_qa, lw.d_q_a_norm, kKimiQLoraRank);

    // 3. q_b projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiQbRows, seq, kKimiQLoraRank, &alpha,
                 lw.d_q_b_proj, CUDA_R_16BF, kKimiQLoraRank,
                 d_qa, CUDA_R_16BF, kKimiQLoraRank, &beta0,
                 d_qall, CUDA_R_16BF, kKimiQbRows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

    // 4. kv_a projection.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiKvaRows, seq, kKimiHidden, &alpha,
                 lw.d_kv_a_proj, CUDA_R_16BF, kKimiHidden,
                 d_hidden, CUDA_R_16BF, kKimiHidden, &beta0,
                 d_kvfull, CUDA_R_16BF, kKimiKvaRows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

    // 5. Postprocess: latent norm + key/query yarn rope.
    kimi_kv_postprocess_kernel<<<dim3(seq), 256, 0, stream>>>(
        d_kvfull, d_qall, lw.d_kv_a_norm, pos_offset, d_inv_freq);

    // 6. Append latents+keys to cache slots (chronological, deterministic).
    kimi_mla_append_kernel<<<dim3(seq), kKimiKvLoraRank + kKimiQkRope, 0,
                             stream>>>(d_state, d_kvfull, seq, base_slot);


    // 7. Fused absorbed attention.
    dim3 attn_grid(seq, kKimiNumHeads);
    kimi_mla_attn_kernel<<<attn_grid, 512,
                           0, stream>>>(d_qall, lw.d_kv_b_proj, d_state,
                                        d_attn, seq, base_slot);

    // 8. o_proj.
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 kKimiORows, seq, kKimiNumHeads * kKimiHeadV, &alpha,
                 lw.d_o_proj, CUDA_R_16BF, kKimiNumHeads * kKimiHeadV,
                 d_attn, CUDA_R_16BF, kKimiNumHeads * kKimiHeadV, &beta0,
                 d_oproj, CUDA_R_16BF, kKimiORows,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);


    // 9. Residual add.
    {
        int total = seq * kKimiHidden;
        int block = 256;
        int grid = (total + block - 1) / block;
        ::lsxcommon::kernels::AddOutKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_oproj, d_residual, d_out, total);
    }

    return cudaGetLastError();
}

// ── Routed-expert NVFP4 FFN (SwiGLU) ───────────────────────────────────────

#include <cfloat>

namespace {

// NVFP4 elementwise dequant: unified kernels::Nvfp4DequantKernel
// (lsx_kernels_quant.cuh), byte-equal to the former kimi_nvfp4_dequant_kernel.

// SwiGLU: unified kernels::SwigluKernel (lsx_kernels_act.cuh).

// d_out += weight * y (elementwise over the hidden row).
__global__ void kimi_accum_kernel(const __nv_bfloat16 *__restrict__ y, float weight,
                                  __nv_bfloat16 *__restrict__ out, int64_t n)
{
    const int64_t i =
        blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16_rn(__bfloat162float(out[i]) +
                          weight * __bfloat162float(y[i]));
}

} // namespace

cudaError_t RunKimiExpertFfn(cublasHandle_t cublas_h, cudaStream_t stream,
                             const KimiExpertEntry &expert,
                             float expert_weight, const __nv_bfloat16 *d_x,
                             __nv_bfloat16 *d_out, void *d_staging,
                             size_t staging_bytes)
{
    if (!expert.host_w1 || !expert.host_w2 || !expert.host_w3 || !d_staging)
        return cudaErrorInvalidValue;

    constexpr int I = kKimiMoeIntermediate;
    constexpr int Hd = kKimiHidden;
    const int64_t w13_el = static_cast<int64_t>(I) * Hd;

    auto align_up = [](size_t v) {
        return (v + 255) / 256 * 256;
    };
    size_t off_p_w1 = 0;
    size_t off_p_w3 = off_p_w1 + align_up(expert.w1_bytes);
    size_t off_p_w2 = off_p_w3 + align_up(expert.w3_bytes);
    size_t off_s_w1 = off_p_w2 + align_up(expert.w2_bytes);
    size_t off_s_w3 = off_s_w1 + align_up(expert.w1_scales_bytes);
    size_t off_s_w2 = off_s_w3 + align_up(expert.w3_scales_bytes);
    size_t off_dq_gate = align_up(off_s_w2 + expert.w2_scales_bytes);
    size_t off_dq_up = off_dq_gate + w13_el * sizeof(__nv_bfloat16);
    size_t off_dq_down = off_dq_up + w13_el * sizeof(__nv_bfloat16);
    size_t off_act_g = off_dq_down + w13_el * sizeof(__nv_bfloat16);
    size_t off_act_u = off_act_g + I * sizeof(__nv_bfloat16);
    size_t off_act_h = off_act_u + I * sizeof(__nv_bfloat16);
    size_t off_y = off_act_h + I * sizeof(__nv_bfloat16);
    size_t need = off_y + Hd * sizeof(__nv_bfloat16);

    if (staging_bytes < need) return cudaErrorInvalidValue;

    uint8_t *base = static_cast<uint8_t *>(d_staging);
    uint8_t *p_w1 = base + off_p_w1;
    uint8_t *p_w3 = base + off_p_w3;
    uint8_t *p_w2 = base + off_p_w2;
    uint8_t *p_s1 = base + off_s_w1;
    uint8_t *p_s3 = base + off_s_w3;
    uint8_t *p_s2 = base + off_s_w2;
    __nv_bfloat16 *dq_gate = reinterpret_cast<__nv_bfloat16 *>(base + off_dq_gate);
    __nv_bfloat16 *dq_up = reinterpret_cast<__nv_bfloat16 *>(base + off_dq_up);
    __nv_bfloat16 *dq_down = reinterpret_cast<__nv_bfloat16 *>(base + off_dq_down);
    __nv_bfloat16 *act_g = reinterpret_cast<__nv_bfloat16 *>(base + off_act_g);
    __nv_bfloat16 *act_u = reinterpret_cast<__nv_bfloat16 *>(base + off_act_u);
    __nv_bfloat16 *act_h = reinterpret_cast<__nv_bfloat16 *>(base + off_act_h);
    __nv_bfloat16 *y = reinterpret_cast<__nv_bfloat16 *>(base + off_y);

    // H2D of the mmap'd payloads + dequant.
    cudaError_t e = cudaSuccess;
    e = cudaMemcpyAsync(p_w1, expert.host_w1, expert.w1_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p_w3, expert.host_w3, expert.w3_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p_w2, expert.host_w2, expert.w2_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p_s1, expert.host_w1_scales, expert.w1_scales_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p_s3, expert.host_w3_scales, expert.w3_scales_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(p_s2, expert.host_w2_scales, expert.w2_scales_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    auto dequant = [&](const uint8_t *packed, const uint8_t *scales,
                       float gscale, __nv_bfloat16 *dst, int64_t numel) {
        const int block = 256;
        const int64_t grid = (numel + block - 1) / block;
        ::lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
            <<<static_cast<unsigned>(grid), block, 0, stream>>>(
                packed, scales, gscale, dst, numel);
    };
    dequant(p_w1, p_s1, expert.w1_global_scale, dq_gate, w13_el);
    dequant(p_w3, p_s3, expert.w3_global_scale, dq_up, w13_el);
    dequant(p_w2, p_s2, expert.w2_global_scale, dq_down,
            static_cast<int64_t>(Hd) * I);

    cublasSetStream(cublas_h, stream);
    const float alpha = 1.0f, beta0 = 0.0f;

    // gate/up projections: [I x 1] = W(I x Hd)^T @ x(Hd).
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 I, 1, Hd, &alpha, dq_gate, CUDA_R_16BF, Hd,
                 d_x, CUDA_R_16BF, Hd, &beta0,
                 act_g, CUDA_R_16BF, I, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 I, 1, Hd, &alpha, dq_up, CUDA_R_16BF, Hd,
                 d_x, CUDA_R_16BF, Hd, &beta0,
                 act_u, CUDA_R_16BF, I, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

    ::lsxcommon::kernels::SwigluKernel<__nv_bfloat16, ::lsxcommon::kernels::ExpFlavor::Fast, false>
        <<<static_cast<unsigned>((I + 255) / 256), 256, 0, stream>>>(
            act_g, act_u, act_h, I, 0.f);

    // down projection: y(Hd) = W_down(Hd x I)^T @ h(I).
    cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                 Hd, 1, I, &alpha, dq_down, CUDA_R_16BF, I,
                 act_h, CUDA_R_16BF, I, &beta0,
                 y, CUDA_R_16BF, Hd, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

    kimi_accum_kernel<<<static_cast<unsigned>((Hd + 255) / 256), 256, 0,
                        stream>>>(y, expert_weight, d_out, Hd);


    return cudaGetLastError();
}

// ── Full generation loop (greedy decode, single stream) ────────────────────

namespace {

// Embed gather: unified kernels::EmbedGatherKernel (lsx_kernels_math.cuh).

// Out-of-place RMSNorm rows [seq x n]: dst = LN(src).
__global__ void kimi_rmsnorm_rows_op_kernel(const __nv_bfloat16 *__restrict__ src,
                                            const __nv_bfloat16 *__restrict__ w,
                                            __nv_bfloat16 *__restrict__ dst, int n)
{
    const int token = blockIdx.x;
    __shared__ float s_ss;
    __shared__ float s_part[256];
    const __nv_bfloat16 *srow = src + (size_t)token * n;
    __nv_bfloat16 *drow = dst + (size_t)token * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += __bfloat162float(srow[i]) * __bfloat162float(srow[i]);
    s_part[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) total += s_part[i];
        s_ss = total;
    }
    __syncthreads();
    const float inv = rsqrtf(s_ss / static_cast<float>(n) + kKimiRmsEps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        drow[i] =
            __float2bfloat16_rn(__bfloat162float(srow[i]) * __bfloat162float(w[i]) * inv);
}

// Argmax over one FP32 logits row; deterministic lowest-id tie-break.
__global__ void kimi_argmax_kernel(const float *__restrict__ logits, int n,
                                   int *__restrict__ best_id,
                                   float *__restrict__ best_val)
{
    __shared__ float s_val[kRouterMaxBlock];
    __shared__ int s_idx[kRouterMaxBlock];
    const int tid = threadIdx.x;
    float local = -INFINITY;
    int idx = -1;
    for (int i = tid; i < n; i += blockDim.x)
        if (logits[i] > local) {
            local = logits[i];
            idx = i;
        }
    s_val[tid] = local;
    s_idx[tid] = idx;
    __syncthreads();
    if (tid == 0) {
        float m = -INFINITY;
        int bi = -1;
        for (int i = 0; i < blockDim.x; ++i)
            if (s_idx[i] >= 0 && s_val[i] > m) {
                m = s_val[i];
                bi = s_idx[i];
            }
        *best_id = bi;
        *best_val = m;
    }
}

} // namespace

namespace {

cudaError_t KimiLayerStack(cublasHandle_t cublas_h, cudaStream_t stream,
                           const KimiPinnedWeights &pw, KimiExpertDb &db,
                           KimiMlaState *d_states, float *d_bias_all,
                           void *d_staging, size_t staging_bytes,
                           __nv_bfloat16 *d_scratch_a, __nv_bfloat16 *d_scratch_b,
                           __nv_bfloat16 *d_mlp_g, __nv_bfloat16 *d_mlp_u, __nv_bfloat16 *d_mlp_h,
                           float *d_scores, int *d_top_ids, float *d_top_w,
                           int turn, int base, int seq, int base_slot,
                           int pos_offset)
{
    cublasSetStream(cublas_h, stream);
    const float alpha = 1.0f, beta0 = 0.0f;

    for (int l = 0; l < kKimiNumLayers; ++l) {
        const KimiLayerWeights &lw = pw.layers[l];

        // 1. input_layernorm -> scratch_b.
        kimi_rmsnorm_rows_op_kernel<<<dim3(seq), 256, 0, stream>>>(
            d_scratch_a + (size_t)base * kKimiHidden, lw.d_input_norm,
            d_scratch_b + (size_t)base * kKimiHidden, kKimiHidden);

        // 2. MLA attention: out = normed + attn(normed) -> scratch_a.
        cudaError_t ae = RunKimiMlaAttention(
            stream, turn, lw, seq,
            d_scratch_b + (size_t)base * kKimiHidden,   // normed input
            d_scratch_a + (size_t)base * kKimiHidden,   // residual = x
            d_scratch_a + (size_t)base * kKimiHidden,   // out
            d_states + l, cublas_h, base_slot, pos_offset);
        if (ae != cudaSuccess) return ae;

        // 3. post_attention_layernorm -> scratch_b.
        kimi_rmsnorm_rows_op_kernel<<<dim3(seq), 256, 0, stream>>>(
            d_scratch_a + (size_t)base * kKimiHidden, lw.d_post_attn_norm,
            d_scratch_b + (size_t)base * kKimiHidden, kKimiHidden);

        if (lw.d_router_gate) {
            // ── MoE path ──
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                         kKimiNumRouted, seq, kKimiHidden, &alpha,
                         lw.d_router_gate, CUDA_R_16BF, kKimiHidden,
                         d_scratch_b + (size_t)base * kKimiHidden,
                         CUDA_R_16BF, kKimiHidden, &beta0,
                         d_scores, CUDA_R_32F, kKimiNumRouted,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);

            // Residual contract: scratch_a still holds x (post-attention);
            // expert FFNs accumulate their weighted outputs onto it.

            // Shared expert (weight 1.0).
            if ((int)pw.shared_host.size() > l &&
                pw.shared_host[l].w1_packed) {
                const PackedExpertView &v = pw.shared_host[l];
                KimiExpertEntry se;
                se.host_w1 = v.w1_packed;
                se.host_w2 = v.w2_packed;
                se.host_w3 = v.w3_packed;
                se.host_w1_scales = v.w1_scales;
                se.host_w2_scales = v.w2_scales;
                se.host_w3_scales = v.w3_scales;
                se.w1_global_scale = v.w1_global_scale;
                se.w2_global_scale = v.w2_global_scale;
                se.w3_global_scale = v.w3_global_scale;
                se.w1_bytes = v.w1_packed_bytes;
                se.w2_bytes = v.w2_packed_bytes;
                se.w3_bytes = v.w3_packed_bytes;
                se.w1_scales_bytes = v.w1_scale_bytes;
                se.w2_scales_bytes = v.w2_scale_bytes;
                se.w3_scales_bytes = v.w3_scale_bytes;
                for (int t = 0; t < seq; ++t) {
                    cudaError_t ee = RunKimiExpertFfn(
                        cublas_h, stream, se, 1.0f,
                        d_scratch_b + (size_t)(base + t) * kKimiHidden,
                        d_scratch_a + (size_t)(base + t) * kKimiHidden,
                        d_staging, staging_bytes);
                    if (ee != cudaSuccess) return ee;
                }
            }

            // Routed top-k per token row.
            for (int t = 0; t < seq; ++t) {
                const __nv_bfloat16 *row =
                    d_scratch_b + (size_t)(base + t) * kKimiHidden;
                __nv_bfloat16 *acc =
                    d_scratch_a + (size_t)(base + t) * kKimiHidden;
                cudaError_t re = KimiRouterTopK(
                    d_scores + static_cast<size_t>(t) * kKimiNumRouted,
                    d_bias_all + (size_t)l * kKimiNumRouted,
                    kKimiNumRouted, kKimiTopK, d_top_ids, d_top_w, stream);
                if (re != cudaSuccess) return re;
                int h_ids[kKimiTopK];
                float h_w[kKimiTopK];
                cudaMemcpyAsync(h_ids, d_top_ids, sizeof(h_ids),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_w, d_top_w, sizeof(h_w),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);   // ids needed on host
                for (int k = 0; k < kKimiTopK; ++k) {
                    ensure_kimi_expert_resident(db, l, h_ids[k], stream);
                    cudaError_t fe = RunKimiExpertFfn(
                        cublas_h, stream,
                        db.entries[(size_t)l * kKimiNumRouted + h_ids[k]],
                        h_w[k], row, acc, d_staging, staging_bytes);
                    if (fe != cudaSuccess) {
                        fprintf(stderr, "[trace] FFN l=%d e=%d err=%s\n",
                                l, h_ids[k], cudaGetErrorString(fe));
                        return fe;
                    }
                }
            }
        } else {
            // ── Dense MLP path (layer 0) ──
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                         kKimiDenseIntermediate, seq, kKimiHidden, &alpha,
                         lw.d_mlp_gate, CUDA_R_16BF, kKimiHidden,
                         d_scratch_b + (size_t)base * kKimiHidden,
                         CUDA_R_16BF, kKimiHidden, &beta0,
                         d_mlp_g, CUDA_R_16BF, kKimiDenseIntermediate,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                         kKimiDenseIntermediate, seq, kKimiHidden, &alpha,
                         lw.d_mlp_up, CUDA_R_16BF, kKimiHidden,
                         d_scratch_b + (size_t)base * kKimiHidden,
                         CUDA_R_16BF, kKimiHidden, &beta0,
                         d_mlp_u, CUDA_R_16BF, kKimiDenseIntermediate,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);
            {
                const int64_t n = (int64_t)seq * kKimiDenseIntermediate;
                ::lsxcommon::kernels::SwigluKernel<__nv_bfloat16, ::lsxcommon::kernels::ExpFlavor::Fast, false>
                <<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
                    d_mlp_g, d_mlp_u, d_mlp_h, n, 0.f);
            }
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                         kKimiHidden, seq, kKimiDenseIntermediate, &alpha,
                         lw.d_mlp_down, CUDA_R_16BF, kKimiDenseIntermediate,
                         d_mlp_h, CUDA_R_16BF, kKimiDenseIntermediate, &beta0,
                         d_mlp_g, CUDA_R_16BF, kKimiHidden,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);
            {
                int total = seq * kKimiHidden;
                int block = 256;
                int grid = (total + block - 1) / block;
                ::lsxcommon::kernels::AddInPlaceKernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                    d_mlp_g,
                    d_scratch_a + (size_t)base * kKimiHidden, total);
            }
        }
    }
    return cudaSuccess;
}

} // namespace

cudaError_t RunKimiGeneration(cublasHandle_t cublas_h, cudaStream_t stream,
                              const KimiPinnedWeights &pw, KimiExpertDb &db,
                              const std::vector<int> &prompt_ids,
                              int max_gen_tokens, KimiGenerationResult &gen)
{
    gen = {};
    if (!pw.d_embed || !pw.d_head ||
        pw.layers.size() != static_cast<size_t>(kKimiNumLayers)) {
        gen.error_msg = "weights not loaded / geometry mismatch";
        return cudaErrorInvalidValue;
    }
    if (prompt_ids.empty() || max_gen_tokens <= 0) {
        gen.error_msg = "empty prompt / bad max_gen_tokens";
        return cudaErrorInvalidValue;
    }

    cublasSetStream(cublas_h, stream);
    const auto t0 = std::chrono::steady_clock::now();

    KimiMlaState *d_states = nullptr;
    if (cudaMalloc(&d_states, kKimiNumLayers * sizeof(KimiMlaState)) !=
        cudaSuccess)
        return cudaErrorMemoryAllocation;
    for (int l = 0; l < kKimiNumLayers; ++l)
        KimiMlaReset(d_states + l, stream);

    float *d_bias_all = nullptr;
    if (cudaMalloc(&d_bias_all,
                   (size_t)kKimiNumLayers * kKimiNumRouted * sizeof(float)) !=
        cudaSuccess)
        return cudaErrorMemoryAllocation;
    for (int l = 0; l < kKimiNumLayers; ++l) {
        const KimiLayerWeights &lw = pw.layers[l];
        if (lw.h_gate_bias && lw.gate_bias_n == kKimiNumRouted)
            cudaMemcpyAsync(d_bias_all + (size_t)l * kKimiNumRouted,
                            lw.h_gate_bias, kKimiNumRouted * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
    }

    void *d_staging = nullptr;
    const size_t staging_bytes = KimiExpertFfnStagingBytes();
    if (cudaMalloc(&d_staging, staging_bytes) != cudaSuccess)
        return cudaErrorMemoryAllocation;

    std::vector<int> ids = prompt_ids;
    const int prefill_seq = static_cast<int>(ids.size());
    const int max_total = prefill_seq + max_gen_tokens;

    int *d_ids = nullptr;
    if (cudaMalloc(&d_ids, (size_t)max_total * sizeof(int)) !=
        cudaSuccess)
        return cudaErrorMemoryAllocation;
    cudaMemcpyAsync(d_ids, ids.data(), ids.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    __nv_bfloat16 *d_hidden = nullptr, *d_scratch_b = nullptr;
    __nv_bfloat16 *d_mlp_g = nullptr, *d_mlp_u = nullptr, *d_mlp_h = nullptr;
    float *d_scores = nullptr, *d_logits = nullptr, *d_bestv = nullptr;
    float *d_top_w = nullptr;
    int *d_top_ids = nullptr, *d_best = nullptr;
    cudaError_t e = cudaSuccess;
#define KIMI_ALLOC(var, count)                                             \
    if (e == cudaSuccess)                                                  \
        e = cudaMalloc(reinterpret_cast<void **>(&var),                    \
                       static_cast<size_t>(count) * sizeof(*var))
    KIMI_ALLOC(d_hidden, (size_t)max_total * kKimiHidden);
    KIMI_ALLOC(d_scratch_b, (size_t)max_total * kKimiHidden);
    KIMI_ALLOC(d_mlp_g, (size_t)max_total * kKimiDenseIntermediate);
    KIMI_ALLOC(d_mlp_u, (size_t)max_total * kKimiDenseIntermediate);
    KIMI_ALLOC(d_mlp_h, (size_t)max_total * kKimiDenseIntermediate);
    KIMI_ALLOC(d_scores, (size_t)max_total * kKimiNumRouted);
    KIMI_ALLOC(d_logits, kKimiVocab);
    KIMI_ALLOC(d_bestv, 1);
    KIMI_ALLOC(d_top_ids, kKimiTopK);
    KIMI_ALLOC(d_top_w, kKimiTopK);
    KIMI_ALLOC(d_best, 1);
#undef KIMI_ALLOC
    if (e != cudaSuccess) {
        gen.error_msg = "workspace alloc failed";
        return e;
    }

    int generated = 0;
    for (int step = 0; step <= max_gen_tokens; ++step) {
        if (step == 0) {
            ::lsxcommon::kernels::EmbedGatherKernel<__nv_bfloat16>
                <<<(int)((prefill_seq * kKimiHidden + 255) / 256), 256, 0,
                   stream>>>(pw.d_embed, d_ids, d_hidden, prefill_seq,
                             kKimiHidden);
        } else {
            ::lsxcommon::kernels::EmbedGatherKernel<__nv_bfloat16>
                <<<(int)((kKimiHidden + 255) / 256), 256, 0, stream>>>(
                    pw.d_embed, d_ids + prefill_seq + step - 1,
                    d_hidden + (size_t)(prefill_seq + step - 1) * kKimiHidden,
                    1, kKimiHidden);
        }

        const int base = (step == 0) ? 0 : (prefill_seq + step - 1);
        const int seq = (step == 0) ? prefill_seq : 1;
        const int base_slot = (step == 0) ? 0 : (prefill_seq + step - 1);
        e = KimiLayerStack(cublas_h, stream, pw, db, d_states, d_bias_all,
                           d_staging, staging_bytes, d_hidden, d_scratch_b,
                           d_mlp_g, d_mlp_u, d_mlp_h, d_scores, d_top_ids,
                           d_top_w, step, base, seq, base_slot, base);
        if (e != cudaSuccess) {
            gen.error_msg = std::string("layer stack failed: ") +
                            cudaGetErrorString(e);
            return e;
        }

        // Final norm (last row) -> logits -> argmax.
        kimi_rmsnorm_rows_op_kernel<<<dim3(1), 256, 0, stream>>>(
            d_hidden + (size_t)(base + seq - 1) * kKimiHidden,
            pw.d_final_norm, d_scratch_b, kKimiHidden);
        cublasSetStream(cublas_h, stream);
        const float alpha = 1.0f, beta0 = 0.0f;
        cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                     kKimiVocab, 1, kKimiHidden, &alpha,
                     pw.d_head, CUDA_R_16BF, kKimiHidden,
                     d_scratch_b, CUDA_R_16BF, kKimiHidden, &beta0,
                     d_logits, CUDA_R_32F, kKimiVocab,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        do {
            cudaError_t ce_ = cudaGetLastError();
            if (ce_ != cudaSuccess)
                fprintf(stderr,
                        "[cublas] err=%s at line %d\n",
                        cudaGetErrorString(ce_), __LINE__);
        } while (0);
        kimi_argmax_kernel<<<1, kRouterMaxBlock, 0, stream>>>(
            d_logits, kKimiVocab, d_best, d_bestv);


        int next_id = -1;
        float next_maxv = 0.0f;
        cudaMemcpyAsync(&next_id, d_best, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&next_maxv, d_bestv, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);   // greedy decode needs the id

        if (step == 0) {
            gen.ttft_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
        }
        if (next_id < 0) {
            gen.error_msg = "argmax failed";
            return cudaErrorUnknown;
        }

        if (step > 0 || true) {
            // Every loop iteration produces one token (prefill pass produces
            // the first sampled token from the full prompt).
            gen.tokens.push_back(next_id);
            ++generated;
            if (generated == 1) gen.first_maxv = next_maxv;
            ids.push_back(next_id);
            cudaMemcpyAsync(d_ids + prefill_seq + step, &next_id,
                            sizeof(int), cudaMemcpyHostToDevice, stream);
        }

        constexpr int kKimiEosId = 163586;
        if (next_id == kKimiEosId) {
            gen.eos_terminated = true;
            break;
        }
        if (generated >= max_gen_tokens) break;
    }

    gen.final_logits.resize(kKimiVocab);
    cudaMemcpy(gen.final_logits.data(), d_logits,
               kKimiVocab * sizeof(float), cudaMemcpyDeviceToHost);
    gen.kv_slots_used = prefill_seq + generated;
    gen.ok = true;
    gen.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    return cudaSuccess;
}

} // namespace kimi
} // namespace lsxcommon
