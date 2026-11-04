// Kimi-K3 device kernels. FP32 accumulation everywhere, fixed reduction
// geometry (run-to-run deterministic), thread-0 RMW behind barriers.
#include "lsxcommon/kimi_moe_k3_kernels.h"

#include "nvfp4_codec.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon::kimi_k3 {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxTopK = 64;

__device__ __forceinline__ float silu_f(float x)
{
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float sigmoid_f(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float block_sum(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// ── RMSNorm ─────────────────────────────────────────────────────────────────
__global__ void rmsnorm_kernel(const __half *__restrict__ x,
                               const __half *__restrict__ w,
                               __half *__restrict__ out, int n, float eps)
{
    __shared__ float scratch[kBlock];
    const int row = blockIdx.x;
    const __half *xr = x + static_cast<size_t>(row) * n;
    __half *orr = out + static_cast<size_t>(row) * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        sum += v * v;
    }
    const float rstd = rsqrtf(block_sum(sum, scratch) / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        orr[i] = __float2half(__half2float(xr[i]) * rstd * __half2float(w[i]));
}

__global__ void gated_rmsnorm_kernel(const __half *__restrict__ x,
                                     const float *__restrict__ w,
                                     const __half *__restrict__ gate,
                                     __half *__restrict__ out, int n, float eps)
{
    __shared__ float scratch[kBlock];
    const int row = blockIdx.x;
    const __half *xr = x + static_cast<size_t>(row) * n;
    const __half *gr = gate + static_cast<size_t>(row) * n;
    __half *orr = out + static_cast<size_t>(row) * n;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        sum += v * v;
    }
    const float rstd = rsqrtf(block_sum(sum, scratch) / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(xr[i]) * rstd * w[i];
        orr[i] = __float2half(v * sigmoid_f(__half2float(gr[i])));
    }
}

// ── SiTU-GLU ────────────────────────────────────────────────────────────────
__global__ void situ_kernel(const __half *__restrict__ gate,
                            const __half *__restrict__ up,
                            __half *__restrict__ out, int64_t n, float beta,
                            float lin_beta)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    const float a = beta * tanhf(g / beta) * sigmoid_f(g);
    if (lin_beta > 0.0f) u = lin_beta * tanhf(u / lin_beta);
    out[i] = __float2half(a * u);
}

// ── router ──────────────────────────────────────────────────────────────────
__global__ void router_scores_kernel(const __half *__restrict__ x,
                                     const __half *__restrict__ w,
                                     const float *__restrict__ bias, int H,
                                     float *__restrict__ sel,
                                     float *__restrict__ raw)
{
    __shared__ float scratch[kBlock];
    const int e = blockIdx.x;
    const __half *wr = w + static_cast<size_t>(e) * H;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        dot += __half2float(x[i]) * __half2float(wr[i]);
    const float d = block_sum(dot, scratch);
    if (threadIdx.x == 0) {
        const float s = sigmoid_f(d);
        raw[e] = s;
        sel[e] = s + bias[e];
    }
}

// Deterministic selection: one thread walks all experts; ties resolve to the
// lower index (the neutral greedy convention).
__global__ void router_topk_kernel(const float *__restrict__ sel,
                                   const float *__restrict__ raw, int E, int K,
                                   float scale, int renorm,
                                   int32_t *__restrict__ idx,
                                   float *__restrict__ wts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t best[kMaxTopK];
    float bestv[kMaxTopK];
    int n = 0;
    for (int e = 0; e < E; ++e) {
        const float v = sel[e];
        int pos = n;
        while (pos > 0 && bestv[pos - 1] < v) --pos;
        if (pos >= K) continue;
        const int last = (n < K) ? n : K - 1;
        for (int j = last; j > pos; --j) {
            bestv[j] = bestv[j - 1];
            best[j] = best[j - 1];
        }
        bestv[pos] = v;
        best[pos] = e;
        if (n < K) ++n;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += raw[best[i]];
    const float denom = (renorm && n > 1) ? (sum + 1e-20f) : 1.0f;
    for (int i = 0; i < n; ++i) {
        idx[i] = best[i];
        wts[i] = raw[best[i]] / denom * scale;
    }
}

// ── AttnRes depth mixer ─────────────────────────────────────────────────────
__global__ void depth_mix_kernel(const __half *__restrict__ sources, int n_src,
                                 const __half *__restrict__ norm_w,
                                 const __half *__restrict__ proj_w,
                                 __half *__restrict__ out, int hidden, float eps)
{
    __shared__ float scratch[kBlock];
    __shared__ float logits[16];
    for (int s = 0; s < n_src; ++s) {
        const __half *p = sources + static_cast<size_t>(s) * hidden;
        float sq = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __half2float(p[i]);
            sq += v * v;
        }
        const float rstd = rsqrtf(block_sum(sq, scratch) / hidden + eps);
        float dot = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            const float v = __half2float(p[i]) * rstd * __half2float(norm_w[i]);
            dot += v * __half2float(proj_w[i]);
        }
        const float total = block_sum(dot, scratch);
        if (threadIdx.x == 0) logits[s] = total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = logits[0];
        for (int s = 1; s < n_src; ++s) mx = fmaxf(mx, logits[s]);
        float sum = 0.0f;
        for (int s = 0; s < n_src; ++s) {
            logits[s] = __expf(logits[s] - mx);
            sum += logits[s];
        }
        for (int s = 0; s < n_src; ++s) logits[s] /= sum;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_src; ++s)
            acc += logits[s] *
                   __half2float(sources[static_cast<size_t>(s) * hidden + i]);
        out[i] = __float2half(acc);
    }
}

// ── KDA conv (seq==1, depthwise, kernel k, silu) ────────────────────────────
__global__ void kda_conv_kernel(const __half *__restrict__ x,
                                const float *__restrict__ w,
                                float *__restrict__ hist,
                                __half *__restrict__ out, int C, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.0f;
    float *h = hist + static_cast<size_t>(c) * (k - 1);
    for (int j = 0; j < k - 1; ++j) acc += w[c * k + j] * h[j];
    acc += w[c * k + (k - 1)] * __half2float(x[c]);
    out[c] = __float2half(silu_f(acc));
    // shift history: h[j] = h[j+1], last = x
    for (int j = 0; j + 1 < k - 1; ++j) h[j] = h[j + 1];
    if (k > 1) h[k - 2] = __half2float(x[c]);
}

// ── KDA recurrence ──────────────────────────────────────────────────────────
__global__ void kda_recur_kernel(const __half *__restrict__ q,
                                 const __half *__restrict__ k,
                                 const __half *__restrict__ v,
                                 const __half *__restrict__ g,
                                 const __half *__restrict__ beta,
                                 const float *__restrict__ a_log,
                                 const float *__restrict__ dt_bias,
                                 float *__restrict__ state,
                                 __half *__restrict__ out, int H, int KL,
                                 int KD, int VD, float lower_bound)
{
    extern __shared__ float sh[];
    float *sq = sh;              // [KD]
    float *sk = sh + KD;         // [KD]
    float *salpha = sh + 2 * KD; // [KD]
    float *red = sh + 3 * KD;    // [blockDim.x]
    const int h = blockIdx.x;
    // GQA-style repeat: query/key head h maps to value head h % H.
    const int hv = h % H;
    const __half *br = beta + hv;
    const float beta_raw = __half2float(*br);
    const float b = sigmoid_f(beta_raw);
    const float scale = rsqrtf(static_cast<float>(KD));

    float qsum = 0.0f, ksum = 0.0f;
    for (int d = threadIdx.x; d < KD; d += blockDim.x) {
        const float qv = __half2float(q[(size_t)h * KD + d]);
        const float kv = __half2float(k[(size_t)h * KD + d]);
        sq[d] = qv;
        sk[d] = kv;
        qsum += qv * qv;
        ksum += kv * kv;
        const float a = __half2float(g[(size_t)hv * KD + d]) +
                        dt_bias[(size_t)hv * KD + d];
        // lower-bound gate: lower_bound * sigmoid(exp(A_log[d]) * (g + dt_bias))
        const float gl = lower_bound * sigmoid_f(__expf(a_log[d]) * a);
        salpha[d] = __expf(gl);
    }
    const float qn = block_sum(qsum, red);
    const float kn = block_sum(ksum, red);
    const float qscale = rsqrtf(qn + 1e-6f) * scale;
    const float kscale = rsqrtf(kn + 1e-6f);
    for (int d = threadIdx.x; d < KD; d += blockDim.x) {
        sq[d] *= qscale;
        sk[d] *= kscale;
    }
    __syncthreads();

    float *st = state + (size_t)hv * KD * VD;
    for (int vd = threadIdx.x; vd < VD; vd += blockDim.x) {
        float *col = st + vd;
        float pred = 0.0f;
        for (int d = 0; d < KD; ++d) {
            const size_t idx = static_cast<size_t>(d) * VD;
            col[idx] *= salpha[d];
            pred += sk[d] * col[idx];
        }
        const float delta = (__half2float(v[(size_t)hv * VD + vd]) - pred) * b;
        float o = 0.0f;
        for (int d = 0; d < KD; ++d) {
            const size_t idx = static_cast<size_t>(d) * VD;
            col[idx] += sk[d] * delta;
            o += sq[d] * col[idx];
        }
        out[(size_t)hv * VD + vd] = __float2half(o);
    }
    (void)KL;
}

// ── absorbed MLA ────────────────────────────────────────────────────────────
// q_abs[h][k] = sum_n W_k[h][n][k] * q_nope[h][n]
__global__ void mla_absorb_q_kernel(const __half *__restrict__ q_nope,
                                    const uint8_t *__restrict__ wk_packed,
                                    const uint8_t *__restrict__ wk_scales,
                                    float wk_gs, __half *__restrict__ q_abs,
                                    int heads, int nope, int latent,
                                    int rows_per_head, int row_offset,
                                    int q_head_stride, int row_packed,
                                    int row_scales)
{
    const int h = blockIdx.x;
    const __half *xn = q_nope + (size_t)h * q_head_stride;
    for (int k = threadIdx.x; k < latent; k += blockDim.x) {
        float sum = 0.0f;
        for (int n = 0; n < nope; ++n) {
            const size_t row = (size_t)(h * rows_per_head + row_offset + n);
            const uint8_t byte = wk_packed[row * row_packed + (k >> 1)];
            const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            const uint8_t sc = wk_scales[row * row_scales + (k >> 4)];
            sum += __half2float(xn[n]) *
                   gpu::nvfp4::E2M1Value(nib) *
                   gpu::nvfp4::E4M3ScaleToFloat(sc);
        }
        q_abs[(size_t)h * latent + k] = __float2half(sum * wk_gs);
    }
}

__global__ void mla_attn_kernel(const __half *__restrict__ latent, int n_pos,
                                const __half *__restrict__ q_abs,
                                const __half *__restrict__ q_rot,
                                const uint8_t *__restrict__ wv_packed,
                                const uint8_t *__restrict__ wv_scales,
                                float wv_gs, __half *__restrict__ out,
                                int heads, int L, int rope, int V,
                                int rows_per_head, int row_offset,
                                int q_rot_stride, float scale, int row_packed,
                                int row_scales, const int32_t *__restrict__ sel,
                                int n_sel)
{
    extern __shared__ float sh[];
    float *ctx = sh;              // [L]
    float *sqa = sh + L;          // [L]
    float *red = sh + 2 * L;      // [blockDim.x]
    const int h = blockIdx.x;
    const __half *src = q_abs + (size_t)h * L;
    for (int k = threadIdx.x; k < L; k += blockDim.x)
        sqa[k] = __half2float(src[k]);
    for (int k = threadIdx.x; k < L; k += blockDim.x) ctx[k] = 0.0f;
    __syncthreads();

    const int rot_off = L;  // k_rot lives after the latent in the cache row
    const int steps = (sel != nullptr && n_sel > 0) ? n_sel : n_pos;
    float m = -INFINITY;
    float l = 0.0f;
    for (int i = 0; i < steps; ++i) {
        const int t = (sel != nullptr) ? sel[i] : i;
        const __half *row = latent + (size_t)t * (L + rope);
        float dot = 0.0f;
        for (int k = threadIdx.x; k < L; k += blockDim.x)
            dot += sqa[k] * __half2float(row[k]);
        if (threadIdx.x < rope)
            dot += __half2float(q_rot[(size_t)h * q_rot_stride + threadIdx.x]) *
                   __half2float(row[rot_off + threadIdx.x]);
        const float s = block_sum(dot, red) * scale;
        if (s > m) {
            const float corr = __expf(m - s);
            l *= corr;
            for (int k = threadIdx.x; k < L; k += blockDim.x) ctx[k] *= corr;
            m = s;
            __syncthreads();
        }
        const float p = __expf(s - m);
        l += p;
        for (int k = threadIdx.x; k < L; k += blockDim.x)
            ctx[k] += p * __half2float(row[k]);
        __syncthreads();
    }
    const float inv = 1.0f / l;
    for (int k = threadIdx.x; k < L; k += blockDim.x) ctx[k] *= inv;
    __syncthreads();

    // out[h][m] = sum_k W_v[h][m][k] * ctx[k]
    for (int vd = threadIdx.x; vd < V; vd += blockDim.x) {
        const size_t row = (size_t)(h * rows_per_head + row_offset + vd);
        float acc = 0.0f;
        for (int k = 0; k < L; ++k) {
            const uint8_t byte = wv_packed[row * row_packed + (k >> 1)];
            const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            const uint8_t sc = wv_scales[row * row_scales + (k >> 4)];
            acc += gpu::nvfp4::E2M1Value(nib) *
                   gpu::nvfp4::E4M3ScaleToFloat(sc) * ctx[k];
        }
        out[(size_t)h * V + vd] = __float2half(acc * wv_gs);
    }
}

// ── Lightning Indexer (Gated DSA) ───────────────────────────────────────────
// k = LayerNorm(wk @ x): one block, index_dim rows of GEMV into shared FP32,
// then a two-pass FP32 mean/variance, normalise, apply weight + bias.
__global__ void indexer_key_kernel(const __half *__restrict__ x,
                                   const __half *__restrict__ wk,
                                   const __half *__restrict__ knorm_w,
                                   const __half *__restrict__ knorm_b,
                                   __half *__restrict__ k_out, int kv_len,
                                   int index_dim, float eps)
{
    extern __shared__ float sh[];
    float *raw = sh;              // [index_dim]
    float *red = sh + index_dim;  // [blockDim.x]
    for (int d = 0; d < index_dim; ++d) {
        const __half *wr = wk + static_cast<size_t>(d) * kv_len;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < kv_len; i += blockDim.x)
            acc += __half2float(wr[i]) * __half2float(x[i]);
        const float total = block_sum(acc, red);
        if (threadIdx.x == 0) raw[d] = total;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) sum += raw[d];
    const float mean = block_sum(sum, red) / static_cast<float>(index_dim);
    float sq = 0.0f;
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x) {
        const float dv = raw[d] - mean;
        sq += dv * dv;
    }
    const float var = block_sum(sq, red) / static_cast<float>(index_dim);
    const float rstd = rsqrtf(var + eps);
    for (int d = threadIdx.x; d < index_dim; d += blockDim.x)
        k_out[d] = __float2half((raw[d] - mean) * rstd *
                                    __half2float(knorm_w[d]) +
                                __half2float(knorm_b[d]));
}

// q[h][:] = (wq_b @ qa) * q_scale, w[h] = (weights_proj @ x) * w_scale.
// Blocks 0..index_heads-1 own one head's rows; the extra block owns the head
// weights. Both run the same fixed-geometry reduction as the key kernel.
__global__ void indexer_query_kernel(const __half *__restrict__ qa,
                                     const __half *__restrict__ x,
                                     const __half *__restrict__ wq_b,
                                     const __half *__restrict__ wproj,
                                     __half *__restrict__ q_out,
                                     __half *__restrict__ w_out, int q_lora,
                                     int kv_len, int index_heads,
                                     int index_dim, float q_scale,
                                     float w_scale)
{
    __shared__ float red[kBlock];
    if (blockIdx.x < index_heads) {
        const int h = blockIdx.x;
        for (int d = 0; d < index_dim; ++d) {
            const size_t row = static_cast<size_t>(h) * index_dim + d;
            const __half *wr = wq_b + row * q_lora;
            float acc = 0.0f;
            for (int j = threadIdx.x; j < q_lora; j += blockDim.x)
                acc += __half2float(wr[j]) * __half2float(qa[j]);
            const float total = block_sum(acc, red);
            if (threadIdx.x == 0)
                q_out[row] = __float2half(total * q_scale);
        }
    } else {
        for (int h = 0; h < index_heads; ++h) {
            const __half *wr = wproj + static_cast<size_t>(h) * kv_len;
            float acc = 0.0f;
            for (int i = threadIdx.x; i < kv_len; i += blockDim.x)
                acc += __half2float(wr[i]) * __half2float(x[i]);
            const float total = block_sum(acc, red);
            if (threadIdx.x == 0) w_out[h] = __float2half(total * w_scale);
        }
    }
}

// score[s] = sum_h w[h] * max(0, q[h] . k[s]): one block per cached position,
// FP32 fixed-geometry reductions, no atomics.
__global__ void indexer_scores_kernel(const __half *__restrict__ q,
                                      const __half *__restrict__ w,
                                      const __half *__restrict__ k_cache,
                                      int n_pos, int index_heads, int index_dim,
                                      float *__restrict__ score)
{
    __shared__ float scratch[kBlock];
    const int s = blockIdx.x;
    if (s >= n_pos) return;
    const __half *kr = k_cache + static_cast<size_t>(s) * index_dim;
    float acc = 0.0f;
    for (int h = 0; h < index_heads; ++h) {
        const __half *qr = q + static_cast<size_t>(h) * index_dim;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < index_dim; d += blockDim.x)
            dot += __half2float(qr[d]) * __half2float(kr[d]);
        const float total = block_sum(dot, scratch);
        acc += __half2float(w[h]) * fmaxf(total, 0.0f);
    }
    if (threadIdx.x == 0) score[s] = acc;
}

// Order-preserving float -> uint32 mapping (the radix key).
__device__ __forceinline__ unsigned int monotone_key(float f)
{
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exact top-k by bitwise radix refinement over one warp: the monotone-uint32
// key keeps float order, and each of the 32 bits is resolved by an exact
// integer count of the keys >= the tentative prefix (strided, then a fixed
// butterfly reduction — integer sums are associative, so the result is
// order-independent). The surviving prefix is the k-th largest key, exactly.
// The emission walk (thread 0) then reproduces "the top-k set, boundary ties to
// the lowest positions" by construction: strictly greater first, then the tie
// group from the lowest position up, both in ascending position order.
__global__ void indexer_topk_kernel(const float *__restrict__ score, int n_pos,
                                    int k, int32_t *__restrict__ out_idx,
                                    int *__restrict__ out_n)
{
    const int lane = threadIdx.x & 31;
    const int want = (k < n_pos) ? k : n_pos;
    unsigned int prefix = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned int cand = prefix | (1u << bit);
        int cnt = 0;
        for (int i = lane; i < n_pos; i += 32)
            if (monotone_key(score[i]) >= cand) ++cnt;
        for (int off = 16; off > 0; off >>= 1)
            cnt += __shfl_xor_sync(0xFFFFFFFFu, cnt, off);
        if (cnt >= want) prefix = cand;
    }
    if (threadIdx.x != 0) return;
    const unsigned int thr = prefix;
    int n = 0;
    for (int i = 0; i < n_pos && n < want; ++i)
        if (monotone_key(score[i]) > thr) out_idx[n++] = i;
    for (int i = 0; i < n_pos && n < want; ++i)
        if (monotone_key(score[i]) == thr) out_idx[n++] = i;
    *out_n = n;
}

// ── elementwise glue ────────────────────────────────────────────────────────
__global__ void sigmoid_mul_kernel(__half *__restrict__ x,
                                   const __half *__restrict__ gate, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) * sigmoid_f(__half2float(gate[i])));
}

__global__ void axpy_kernel(__half *__restrict__ acc,
                            const __half *__restrict__ add, float scale,
                            int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    acc[i] = __float2half(__half2float(acc[i]) +
                          scale * __half2float(add[i]));
}

__global__ void dequant_row_kernel(const uint8_t *__restrict__ packed,
                                   const uint8_t *__restrict__ scales,
                                   float global_scale, __half *__restrict__ out,
                                   int cols)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const uint8_t byte = packed[c >> 1];
    const int nib = (c & 1) ? (byte >> 4) : (byte & 0xF);
    const uint8_t sc = scales[c >> 4];
    out[c] = __float2half(gpu::nvfp4::E2M1Value(nib) *
                          gpu::nvfp4::E4M3ScaleToFloat(sc) * global_scale);
}

int Grid(int64_t n, int block) { return static_cast<int>((n + block - 1) / block); }

}  // namespace

cudaError_t RunKimiK3RmsNorm(cudaStream_t stream, const __half *x,
                             const __half *w, __half *out, int rows, int n,
                             float eps)
{
    rmsnorm_kernel<<<rows, kBlock, 0, stream>>>(x, w, out, n, eps);
    return cudaGetLastError();
}

cudaError_t RunKimiK3GatedRmsNorm(cudaStream_t stream, const __half *x,
                                  const float *w, const __half *gate,
                                  __half *out, int rows, int n, float eps)
{
    gated_rmsnorm_kernel<<<rows, kBlock, 0, stream>>>(x, w, gate, out, n, eps);
    return cudaGetLastError();
}

cudaError_t RunKimiK3Situ(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int64_t n, float beta,
                          float linear_beta)
{
    situ_kernel<<<Grid(n, kBlock), kBlock, 0, stream>>>(gate, up, out, n, beta,
                                                        linear_beta);
    return cudaGetLastError();
}

cudaError_t RunKimiK3RouterScores(cudaStream_t stream, const __half *x,
                                  const __half *w, const float *bias, int H,
                                  int E, float *sel, float *raw)
{
    router_scores_kernel<<<E, kBlock, 0, stream>>>(x, w, bias, H, sel, raw);
    return cudaGetLastError();
}

cudaError_t RunKimiK3RouterTopK(cudaStream_t stream, const float *sel,
                                const float *raw, int E, int K, float scale,
                                int renorm, int32_t *idx, float *wts)
{
    if (K > kMaxTopK) return cudaErrorInvalidValue;
    router_topk_kernel<<<1, 32, 0, stream>>>(sel, raw, E, K, scale, renorm, idx,
                                             wts);
    return cudaGetLastError();
}

cudaError_t RunKimiK3DepthMix(cudaStream_t stream, const __half *sources,
                              int n_src, const __half *norm_w,
                              const __half *proj_w, __half *out, int hidden,
                              float eps)
{
    if (n_src <= 0 || n_src > 16) return cudaErrorInvalidValue;
    depth_mix_kernel<<<1, kBlock, 0, stream>>>(sources, n_src, norm_w, proj_w,
                                               out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunKimiK3KdaConv(cudaStream_t stream, const __half *x,
                             const float *w, float *hist, __half *out, int C,
                             int k)
{
    kda_conv_kernel<<<Grid(C, kBlock), kBlock, 0, stream>>>(x, w, hist, out, C,
                                                            k);
    return cudaGetLastError();
}

cudaError_t RunKimiK3KdaRecur(cudaStream_t stream, const __half *q,
                              const __half *k, const __half *v,
                              const __half *g, const __half *beta,
                              const float *a_log, const float *dt_bias,
                              float *state, __half *out, int H, int KL,
                              int KD, int VD, float lower_bound)
{
    const size_t smem = static_cast<size_t>(3 * KD + kBlock) * sizeof(float);
    cudaFuncSetAttribute(kda_recur_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    kda_recur_kernel<<<H, kBlock, smem, stream>>>(
        q, k, v, g, beta, a_log, dt_bias, state, out, H, KL, KD, VD,
        lower_bound);
    return cudaGetLastError();
}

cudaError_t RunKimiK3MlaAbsorbQ(cudaStream_t stream, const __half *q_nope,
                                const uint8_t *wk_packed,
                                const uint8_t *wk_scales, float wk_gs,
                                __half *q_abs, int heads, int nope, int latent,
                                int rows_per_head, int row_offset,
                                int q_head_stride, int row_packed,
                                int row_scales)
{
    mla_absorb_q_kernel<<<heads, kBlock, 0, stream>>>(
        q_nope, wk_packed, wk_scales, wk_gs, q_abs, heads, nope, latent,
        rows_per_head, row_offset, q_head_stride, row_packed, row_scales);
    return cudaGetLastError();
}

cudaError_t RunKimiK3MlaAttention(cudaStream_t stream, const __half *latent,
                                  int n_pos, const __half *q_abs,
                                  const __half *q_rot,
                                  const uint8_t *wv_packed,
                                  const uint8_t *wv_scales, float wv_gs,
                                  __half *out, int heads, int latent_dim,
                                  int rope, int v_head, int rows_per_head,
                                  int row_offset, int q_rot_stride, float scale,
                                  int row_packed, int row_scales,
                                  const int32_t *sel, int n_sel)
{
    if (n_pos <= 0) return cudaSuccess;
    if (sel != nullptr && (n_sel <= 0 || n_sel > n_pos))
        return cudaErrorInvalidValue;
    const size_t smem =
        static_cast<size_t>(2 * latent_dim + kBlock) * sizeof(float);
    cudaFuncSetAttribute(mla_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
    mla_attn_kernel<<<heads, kBlock, smem, stream>>>(
        latent, n_pos, q_abs, q_rot, wv_packed, wv_scales, wv_gs, out, heads,
        latent_dim, rope, v_head, rows_per_head, row_offset, q_rot_stride,
        scale, row_packed, row_scales, sel, n_sel);
    return cudaGetLastError();
}

cudaError_t RunKimiK3IndexerKey(cudaStream_t stream, const __half *x,
                                const __half *wk, const __half *knorm_w,
                                const __half *knorm_b, __half *k_out,
                                int kv_len, int index_dim, float eps)
{
    if (kv_len <= 0 || index_dim <= 0) return cudaErrorInvalidValue;
    const size_t smem =
        static_cast<size_t>(index_dim + kBlock) * sizeof(float);
    indexer_key_kernel<<<1, kBlock, smem, stream>>>(x, wk, knorm_w, knorm_b,
                                                    k_out, kv_len, index_dim,
                                                    eps);
    return cudaGetLastError();
}

cudaError_t RunKimiK3IndexerQuery(cudaStream_t stream, const __half *qa,
                                  const __half *x, const __half *wq_b,
                                  const __half *wproj, __half *q_out,
                                  __half *w_out, int q_lora, int kv_len,
                                  int index_heads, int index_dim,
                                  float q_scale, float w_scale)
{
    if (q_lora <= 0 || kv_len <= 0 || index_heads <= 0 || index_dim <= 0)
        return cudaErrorInvalidValue;
    indexer_query_kernel<<<index_heads + 1, kBlock, 0, stream>>>(
        qa, x, wq_b, wproj, q_out, w_out, q_lora, kv_len, index_heads,
        index_dim, q_scale, w_scale);
    return cudaGetLastError();
}

cudaError_t RunKimiK3IndexerScores(cudaStream_t stream, const __half *q,
                                   const __half *w, const __half *k_cache,
                                   int n_pos, int index_heads, int index_dim,
                                   float *score)
{
    if (n_pos <= 0) return cudaSuccess;
    if (index_heads <= 0 || index_dim <= 0) return cudaErrorInvalidValue;
    indexer_scores_kernel<<<n_pos, kBlock, 0, stream>>>(
        q, w, k_cache, n_pos, index_heads, index_dim, score);
    return cudaGetLastError();
}

cudaError_t RunKimiK3IndexerTopK(cudaStream_t stream, const float *score,
                                 int n_pos, int k, int32_t *out_idx,
                                 int *out_n)
{
    if (n_pos <= 0 || k <= 0) return cudaErrorInvalidValue;
    indexer_topk_kernel<<<1, 32, 0, stream>>>(score, n_pos, k, out_idx, out_n);
    return cudaGetLastError();
}

cudaError_t RunKimiK3SigmoidMul(cudaStream_t stream, __half *x,
                                const __half *gate, int64_t n)
{
    sigmoid_mul_kernel<<<Grid(n, kBlock), kBlock, 0, stream>>>(x, gate, n);
    return cudaGetLastError();
}

cudaError_t RunKimiK3Axpy(cudaStream_t stream, __half *acc, const __half *add,
                          float scale, int64_t n)
{
    axpy_kernel<<<Grid(n, kBlock), kBlock, 0, stream>>>(acc, add, scale, n);
    return cudaGetLastError();
}

cudaError_t RunKimiK3DequantRow(cudaStream_t stream, const uint8_t *packed,
                                const uint8_t *scales, float global_scale,
                                __half *out, int cols)
{
    dequant_row_kernel<<<Grid(cols, kBlock), kBlock, 0, stream>>>(
        packed, scales, global_scale, out, cols);
    return cudaGetLastError();
}

}  // namespace lsxcommon::kimi_k3
