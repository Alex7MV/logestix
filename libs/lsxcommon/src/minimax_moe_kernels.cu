#include "lsxcommon/minimax_moe_kernels.h"

#include "nvfp4_codec.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace lsxcommon::minimax {
namespace {

constexpr int kThreads = 256;

// Precise (libm) transcendentals: the numeric goldens are compared at
// FastMathMode::kOff, so the fast intrinsics (__expf/__cosf) stay out of
// anything that feeds final_logits.
__device__ __forceinline__ float SigmoidF(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// Order-preserving int encoding so a float max can use atomicMax.
__device__ __forceinline__ int FloatToOrderedInt(float v)
{
    int i = __float_as_int(v);
    return (i >= 0) ? i : (0x7fffffff + (i & 0x7fffffff) + 1);
}
__device__ __forceinline__ float OrderedIntToFloat(int i)
{
    const int mask = (i >> 31) & 0x7fffffff;
    i ^= mask;
    return __int_as_float(i);
}

__device__ __forceinline__ float BlockReduceSum(float v, float *red)
{
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// ── Router ──────────────────────────────────────────────────────────────────

__global__ void minimax_router_kernel(const float *__restrict__ w,
                                      const float *__restrict__ bias,
                                      const __half *__restrict__ x, int E,
                                      int K, int H,
                                      MinimaxRoutedSlot *__restrict__ out)
{
    extern __shared__ float smem[];
    float *red = smem;                 // [blockDim.x]
    float *score = red + blockDim.x;   // [E]
    float *choice = score + E;         // [E]
    __half *xs = reinterpret_cast<__half *>(choice + E);  // [H]

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const __half *xt = x + static_cast<int64_t>(t) * H;
    for (int k = tid; k < H; k += blockDim.x) xs[k] = xt[k];
    __syncthreads();

    for (int e = 0; e < E; ++e) {
        const float *we = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = tid; k < H; k += blockDim.x)
            acc += we[k] * __half2float(xs[k]);
        const float s = BlockReduceSum(acc, red);
        if (tid == 0) {
            const float p = SigmoidF(s);
            score[e] = p;
            choice[e] = p + bias[e];
        }
    }
    __syncthreads();

    if (tid == 0) {
        // Top-K by (score + bias), lower expert id wins ties. Selected ids are
        // then stored ascending so the caller's accumulation order is fixed.
        bool taken[512];
        int sel[32];
        for (int i = 0; i < E && i < 512; ++i) taken[i] = false;
        for (int k = 0; k < K && k < 32; ++k) {
            int best = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < E; ++e) {
                if (e < 512 && taken[e]) continue;
                if (choice[e] > best_v) {
                    best_v = choice[e];
                    best = e;
                }
            }
            if (best < 0) break;
            taken[best] = true;
            sel[k] = best;
        }
        // Insertion sort ascending by expert id (K <= 32).
        for (int i = 1; i < K; ++i) {
            const int v = sel[i];
            int j = i - 1;
            while (j >= 0 && sel[j] > v) {
                sel[j + 1] = sel[j];
                --j;
            }
            sel[j + 1] = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += score[sel[k]];
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        MinimaxRoutedSlot *dst = out + static_cast<int64_t>(t) * K;
        for (int k = 0; k < K; ++k) {
            dst[k].eid = sel[k];
            dst[k].coeff = score[sel[k]] * inv;
        }
    }
}

// ── SwiGLU-OAI ──────────────────────────────────────────────────────────────

__global__ void minimax_swigluoai_kernel(const __half *__restrict__ gate,
                                         const __half *__restrict__ up,
                                         __half *__restrict__ out, int64_t n,
                                         float alpha, float limit)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(__half2float(gate[i]), limit);
    const float u = fminf(fmaxf(__half2float(up[i]), -limit), limit);
    const float v = (u + 1.0f) * g * SigmoidF(alpha * g);
    out[i] = __float2half(v);
}

// ── Norms ───────────────────────────────────────────────────────────────────

// One block per row; blockDim.x = min(hidden, 1024) padded to a power of two.
__global__ void minimax_norm_kernel(const __half *__restrict__ x,
                                    const __half *__restrict__ w,
                                    __half *__restrict__ out, int hidden,
                                    float eps)
{
    extern __shared__ float red[];
    const int64_t row = blockIdx.x;
    const __half *xr = x + row * hidden;
    __half *orow = out + row * hidden;
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        acc += v * v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / static_cast<float>(hidden) + eps);
    __syncthreads();
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        const float wf = __half2float(w[i]);
        orow[i] = __float2half(v * inv * (1.0f + wf));
    }
}

// One block per (row, head); blockDim.x = head_dim padded to a power of two.
__global__ void minimax_head_norm_kernel(const __half *__restrict__ x,
                                         const __half *__restrict__ w,
                                         __half *__restrict__ out, int heads,
                                         int head_dim, float eps)
{
    extern __shared__ float red[];
    const int64_t idx = blockIdx.x;   // row*heads + head
    const int64_t base = idx * head_dim;
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = __half2float(x[base + i]);
        acc += v * v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / static_cast<float>(head_dim) + eps);
    __syncthreads();
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = __half2float(x[base + i]);
        const float wf = __half2float(w[i]);
        out[base + i] = __float2half(v * inv * (1.0f + wf));
    }
}

// ── Partial rotary (HF rotate_half over the first rotary_dim channels) ──────

__global__ void minimax_partial_rope_kernel(__half *__restrict__ x, int heads,
                                            int head_dim, int rotary_dim,
                                            float theta, int pos)
{
    const int half = rotary_dim / 2;
    const int64_t row = blockIdx.x;      // (token, head)
    const int j = threadIdx.x;
    if (j >= half) return;
    const int64_t base = row * head_dim;
    const float inv_freq =
        1.0f / powf(theta, static_cast<float>(j) / static_cast<float>(half));
    const float ang = static_cast<float>(pos) * inv_freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const float x1 = __half2float(x[base + j]);
    const float x2 = __half2float(x[base + j + half]);
    x[base + j] = __float2half(x1 * c - x2 * s);
    x[base + j + half] = __float2half(x1 * s + x2 * c);
}

__global__ void minimax_embed_nvfp4_kernel(
    const uint8_t *__restrict__ packed, const uint8_t *__restrict__ scales,
    float global_scale, const int32_t *__restrict__ ids,
    __half *__restrict__ out, int hidden)
{
    const int t = blockIdx.y;
    const int64_t row = ids[t];
    const uint8_t *prow = packed + (row * hidden) / 2;
    const uint8_t *srow = scales + row * (hidden / 16);
    __half *dst = out + static_cast<int64_t>(t) * hidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += blockDim.x * gridDim.x) {
        const uint8_t byte = prow[i >> 1];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        const float sc = gpu::nvfp4::E4M3ScaleToFloat(srow[i >> 4]);
        dst[i] = __float2half(gpu::nvfp4::E2M1Value(nib) * sc * global_scale);
    }
}

__global__ void minimax_embed_kernel(const __half *__restrict__ table,
                                     const int32_t *__restrict__ ids,
                                     __half *__restrict__ out, int hidden)
{
    const int t = blockIdx.y;
    const int id = ids[t];
    const __half *src = table + static_cast<int64_t>(id) * hidden;
    __half *dst = out + static_cast<int64_t>(t) * hidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
         i += blockDim.x * gridDim.x)
        dst[i] = src[i];
}

// ── Indexer block selection ────────────────────────────────────────────────

__global__ void minimax_indexer_select_kernel(
    const __half *__restrict__ idx_q, const __half *__restrict__ idx_k,
    int kv_len, int pos, int index_head_dim, int block_size, int topk_blocks,
    int local_blocks, int num_blocks, uint8_t *__restrict__ mask)
{
    extern __shared__ int smax[];   // [num_blocks] ordered-int maxima
    const int tid = threadIdx.x;
    const int head = static_cast<int>(blockIdx.x);
    const __half *q = idx_q + static_cast<int64_t>(head) * index_head_dim;

    for (int b = tid; b < num_blocks; b += blockDim.x)
        smax[b] = FloatToOrderedInt(-INFINITY);
    __syncthreads();

    for (int k0 = 0; k0 < kv_len; k0 += blockDim.x) {
        const int k = k0 + tid;
        // Causal: keys past the query position never reach a block max, so a
        // caller that hands over the whole cache cannot leak the future.
        if (k < kv_len && k <= pos) {
            const __half *kk = idx_k + static_cast<int64_t>(k) * index_head_dim;
            float acc = 0.0f;
            for (int d = 0; d < index_head_dim; ++d)
                acc += __half2float(q[d]) * __half2float(kk[d]);
            const int ord = FloatToOrderedInt(acc);
            atomicMax(&smax[k / block_size], ord);
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int q_block = pos / block_size;
        for (int i = 0; i < local_blocks; ++i) {
            int b = q_block - i;
            if (b < 0) b = 0;
            if (b < num_blocks) smax[b] = FloatToOrderedInt(INFINITY);
        }
        const int topk = topk_blocks < num_blocks ? topk_blocks : num_blocks;
        uint8_t *m = mask + static_cast<int64_t>(head) * num_blocks;
        for (int b = 0; b < num_blocks; ++b) m[b] = 0;
        // Threshold = the topk-th largest block score; ties keep lower block
        // ids (deterministic), so we walk descending and mark the first topk.
        float prev = INFINITY;
        for (int s = 0; s < topk; ++s) {
            int best = -1;
            int best_v = -2147483647 - 1;
            for (int b = 0; b < num_blocks; ++b) {
                if (m[b]) continue;
                if (smax[b] > best_v) {
                    best_v = smax[b];
                    best = b;
                }
            }
            if (best < 0) break;
            // A block whose max is -inf never wins a slot.
            if (OrderedIntToFloat(best_v) == -INFINITY) break;
            m[best] = 1;
            prev = OrderedIntToFloat(best_v);
        }
        (void)prev;
    }
}

// ── Masked decode attention (two passes; used past the block budget) ────────

// scores[h][k] = q[h] . K[pos k] * scale   (-inf outside the kept blocks)
__global__ void minimax_attn_scores_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k_base,
    int kv_len, int heads, int kv_heads, int head_dim, float scale,
    int64_t pos_stride, int64_t head_stride, const uint8_t *__restrict__ mask,
    int num_blocks, int block_size, int group_heads, int64_t score_stride,
    float *__restrict__ scores)
{
    const int h = blockIdx.x;
    const int kvh = h / (heads / kv_heads);
    const int idx_head = h / group_heads;
    const __half *qh = q + static_cast<int64_t>(h) * head_dim;
    const __half *kb = k_base + kvh * head_stride;
    const uint8_t *m = mask ? mask + static_cast<int64_t>(idx_head) * num_blocks
                            : nullptr;
    for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < kv_len;
         k += blockDim.x * gridDim.y) {
        float s = 0.0f;
        if (!m || m[k / block_size]) {
            const __half *kp = kb + static_cast<int64_t>(k) * pos_stride;
            for (int d = 0; d < head_dim; ++d)
                s += __half2float(qh[d]) * __half2float(kp[d]);
            s *= scale;
        } else {
            s = -INFINITY;
        }
        scores[h * score_stride + k] = s;
    }
}

// out[h][d] = sum_k softmax(scores[h][:]) * V[pos k][h][d]
__global__ void minimax_attn_out_kernel(const float *__restrict__ scores,
                                        const __half *__restrict__ v_base,
                                        int kv_len, int heads, int kv_heads,
                                        int head_dim, int64_t pos_stride,
                                        int64_t head_stride,
                                        int64_t score_stride,
                                        __half *__restrict__ out)
{
    extern __shared__ float red[];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int kvh = h / (heads / kv_heads);
    const float *sc = scores + static_cast<int64_t>(h) * score_stride;

    float m = -INFINITY;
    for (int k = tid; k < kv_len; k += blockDim.x) m = fmaxf(m, sc[k]);
    red[tid] = m;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float l = 0.0f;
    for (int k = tid; k < kv_len; k += blockDim.x)
        l += expf(sc[k] - m);
    red[tid] = l;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    l = red[0];
    __syncthreads();

    const __half *vb = v_base + kvh * head_stride + tid;
    __half *o = out + static_cast<int64_t>(h) * head_dim + tid;
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int k = 0; k < kv_len; ++k) {
            const float p = expf(sc[k] - m) / l;
            acc += p * __half2float(vb[static_cast<int64_t>(k) * pos_stride]);
        }
        *o = __float2half(acc);
    }
}

__global__ void minimax_routed_accumulate_kernel(float *__restrict__ acc,
                                                 const __half *__restrict__ y,
                                                 float coeff, int64_t n)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += coeff * __half2float(y[i]);
}

__global__ void minimax_moe_combine_kernel(const float *__restrict__ acc,
                                           const __half *__restrict__ shared,
                                           __half *__restrict__ out, int64_t n,
                                           float routed_scale)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v =
        __half2float(shared[i]) + routed_scale * acc[i];
    out[i] = __float2half(v);
}

} // namespace

cudaError_t RunMinimaxRouter(cudaStream_t stream, const float *router_w,
                             const float *router_bias, const __half *x, int seq,
                             int E, int K, int H, MinimaxRoutedSlot *d_slots)
{
    if (seq <= 0 || E <= 0 || K <= 0 || K > E || H <= 0) return cudaErrorInvalidValue;
    if (E > 512 || K > 32) return cudaErrorInvalidValue;
    const size_t shmem = (kThreads + 2 * static_cast<size_t>(E)) * sizeof(float) +
                         static_cast<size_t>(H) * sizeof(__half);
    minimax_router_kernel<<<seq, kThreads, shmem, stream>>>(
        router_w, router_bias, x, E, K, H, d_slots);
    return cudaGetLastError();
}

cudaError_t RunMinimaxSwiGLUOai(cudaStream_t stream, const __half *gate,
                                const __half *up, __half *out, int64_t n,
                                float alpha, float limit)
{
    if (n <= 0) return cudaSuccess;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    minimax_swigluoai_kernel<<<blocks, kThreads, 0, stream>>>(gate, up, out, n,
                                                              alpha, limit);
    return cudaGetLastError();
}

cudaError_t RunMinimaxNorm(cudaStream_t stream, const __half *x,
                           const __half *w, __half *out, int rows, int hidden,
                           float eps)
{
    if (rows <= 0 || hidden <= 0) return cudaSuccess;
    int threads = 1;
    while (threads < hidden && threads < 1024) threads <<= 1;
    const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
    minimax_norm_kernel<<<rows, threads, shmem, stream>>>(x, w, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunMinimaxHeadNorm(cudaStream_t stream, const __half *x,
                               const __half *w, __half *out, int rows,
                               int heads, int head_dim, float eps)
{
    if (rows <= 0 || heads <= 0 || head_dim <= 0) return cudaSuccess;
    int threads = 1;
    while (threads < head_dim && threads < 1024) threads <<= 1;
    const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
    minimax_head_norm_kernel<<<rows * heads, threads, shmem, stream>>>(
        x, w, out, heads, head_dim, eps);
    return cudaGetLastError();
}

cudaError_t RunMinimaxPartialRope(cudaStream_t stream, __half *q, __half *k,
                                  int seq, int q_heads, int k_heads,
                                  int head_dim, int rotary_dim, float theta,
                                  int pos_base)
{
    if (seq <= 0 || rotary_dim <= 0 || (rotary_dim & 1) != 0) return cudaErrorInvalidValue;
    if (rotary_dim > head_dim) return cudaErrorInvalidValue;
    const int half = rotary_dim / 2;
    for (int t = 0; t < seq; ++t) {
        const int pos = pos_base + t;
        minimax_partial_rope_kernel<<<q_heads, half, 0, stream>>>(
            q + static_cast<int64_t>(t) * q_heads * head_dim, q_heads, head_dim,
            rotary_dim, theta, pos);
        if (k && k_heads > 0)
            minimax_partial_rope_kernel<<<k_heads, half, 0, stream>>>(
                k + static_cast<int64_t>(t) * k_heads * head_dim, k_heads,
                head_dim, rotary_dim, theta, pos);
    }
    return cudaGetLastError();
}

cudaError_t RunMinimaxEmbed(cudaStream_t stream, const __half *table,
                            const int32_t *ids, __half *out, int seq, int hidden)
{
    if (seq <= 0 || hidden <= 0) return cudaSuccess;
    const int bx = (hidden + 255) / 256;
    dim3 grid(bx, seq);
    minimax_embed_kernel<<<grid, 256, 0, stream>>>(table, ids, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunMinimaxEmbedNvfp4(cudaStream_t stream, const uint8_t *packed,
                                 const uint8_t *scales, float global_scale,
                                 const int32_t *ids, __half *out, int seq,
                                 int hidden)
{
    if (seq <= 0 || hidden <= 0 || (hidden % 16) != 0)
        return cudaErrorInvalidValue;
    const int bx = (hidden + 255) / 256;
    dim3 grid(bx, seq);
    minimax_embed_nvfp4_kernel<<<grid, 256, 0, stream>>>(
        packed, scales, global_scale, ids, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunMinimaxIndexerSelect(cudaStream_t stream, const __half *idx_q,
                                    const __half *idx_k, int kv_len, int pos,
                                    int index_heads, int index_head_dim,
                                    int block_size, int topk_blocks,
                                    int local_blocks, uint8_t *d_mask)
{
    if (kv_len <= 0 || index_heads <= 0 || block_size <= 0) return cudaErrorInvalidValue;
    const int num_blocks = (kv_len + block_size - 1) / block_size;
    const size_t shmem = static_cast<size_t>(num_blocks) * sizeof(int);
    minimax_indexer_select_kernel<<<index_heads, 128, shmem, stream>>>(
        idx_q, idx_k, kv_len, pos, index_head_dim, block_size, topk_blocks,
        local_blocks, num_blocks, d_mask);
    return cudaGetLastError();
}

cudaError_t RunMinimaxMaskedAttn(cudaStream_t stream, const __half *q,
                                 const __half *k_base, const __half *v_base,
                                 int kv_len, int heads, int kv_heads,
                                 int head_dim, float scale, int64_t pos_stride,
                                 int64_t head_stride, const uint8_t *block_mask,
                                 int num_blocks, int block_size,
                                 int group_heads, int64_t score_stride,
                                 float *d_scores, __half *out)
{
    if (kv_len <= 0 || heads <= 0 || head_dim <= 0) return cudaErrorInvalidValue;
    const int threads = 128;
    const int yblocks = (kv_len + threads - 1) / threads;
    dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(yblocks));
    minimax_attn_scores_kernel<<<grid, threads, 0, stream>>>(
        q, k_base, kv_len, heads, kv_heads, head_dim, scale, pos_stride,
        head_stride, block_mask, num_blocks, block_size, group_heads,
        score_stride, d_scores);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    int othreads = 1;
    while (othreads < head_dim && othreads < 1024) othreads <<= 1;
    const size_t shmem = static_cast<size_t>(othreads) * sizeof(float);
    minimax_attn_out_kernel<<<heads, othreads, shmem, stream>>>(
        d_scores, v_base, kv_len, heads, kv_heads, head_dim, pos_stride,
        head_stride, score_stride, out);
    return cudaGetLastError();
}

cudaError_t RunMinimaxRoutedAccumulate(cudaStream_t stream, float *acc,
                                       const __half *y, float coeff, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    minimax_routed_accumulate_kernel<<<blocks, kThreads, 0, stream>>>(acc, y,
                                                                     coeff, n);
    return cudaGetLastError();
}

cudaError_t RunMinimaxMoECombine(cudaStream_t stream, const float *acc,
                                 const __half *shared, __half *out, int rows,
                                 int hidden, float routed_scale)
{
    if (rows <= 0 || hidden <= 0) return cudaSuccess;
    const int64_t n = static_cast<int64_t>(rows) * hidden;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    minimax_moe_combine_kernel<<<blocks, kThreads, 0, stream>>>(
        acc, shared, out, n, routed_scale);
    return cudaGetLastError();
}

} // namespace lsxcommon::minimax
