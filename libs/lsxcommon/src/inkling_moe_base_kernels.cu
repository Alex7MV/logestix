// Inkling (base) numeric kernels (Model Ingestion Protocol stages 5/6).
//
// Shared-memory sandbox + deterministic reductions: every reduction below is
// a fixed-order tree over shared memory with thread-0-only publication, and no
// kernel uses atomicAdd. Nothing here feeds `final_logits` through an
// order-dependent accumulation, so the bit-exact cycle check of stage 8 holds.
//
// Numeric contract per kernel is stated in inkling_moe_base_kernels.h; the
// host references that pin them are tests/test_inkling_moe_base_kernels.cu.
// The deltas vs the inkling2 sibling (a different architecture): scale
// 1/head_dim, NO sinks/gates/clips, static conv WITHOUT normalizer, joint
// routed+shared router renorm, plain SwiGLU.
#include "lsxcommon/inkling_moe_base_kernels.h"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_misc.cuh"

#include <algorithm>
#include <cmath>

namespace lsxcommon::inkling::base {
namespace {

constexpr int kThreads = 256;
// Upper bound on a cached key block that attention keeps in shared memory.
constexpr int kMaxKvShmem = 8192;

// Deterministic block-wide sum (fixed tree order, thread 0 publishes).
__device__ __forceinline__ float BlockSum(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] =
                __fadd_rn(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float BlockMax(float v, float *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x],
                                         scratch[threadIdx.x + s]);
        __syncthreads();
    }
    return scratch[0];
}

// ── 1. per-head weighted RMSNorm (q/k AFTER the short conv) ───────────────
__global__ void head_norm_kernel(const float *__restrict__ x,
                                 const __half *__restrict__ w,
                                 float *__restrict__ out, int dim, float eps)
{
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float *xr = x + static_cast<int64_t>(row) * dim;
    float *orow = out + static_cast<int64_t>(row) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
    ss = BlockSum(ss, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += blockDim.x)
        orow[i] = __fmul_rn(__fmul_rn(xr[i], inv), __half2float(w[i]));
}

// ── 2. relative-position bias ─────────────────────────────────────────────
// One block per head; each thread owns a key position and computes its own
// d_rel-length dot, so the whole gather is one pass with no scratch.

// ── 3. KV append + attention (NO sinks) ───────────────────────────────────
__global__ void kv_append_kernel(const __half *__restrict__ k_new,
                                 const __half *__restrict__ v_new,
                                 __half *__restrict__ d_k,
                                 __half *__restrict__ d_v, int head_dim,
                                 int64_t pos_stride, int64_t head_stride,
                                 int pos)
{
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= head_dim) return;
    const int64_t dst = static_cast<int64_t>(pos) * pos_stride +
                        static_cast<int64_t>(h) * head_stride + i;
    const int64_t src = static_cast<int64_t>(h) * head_dim + i;
    d_k[dst] = k_new[src];
    d_v[dst] = v_new[src];
}

__global__ void attention_kernel(const __half *__restrict__ q,
                                 const __half *__restrict__ d_k,
                                 const __half *__restrict__ d_v,
                                 const float *__restrict__ bias,
                                 __half *__restrict__ out, int heads,
                                 int kv_heads, int head_dim,
                                 int64_t pos_stride, int64_t head_stride,
                                 int pos, int window, float scale, int kv_len)
{
    extern __shared__ float smem[];
    float *logit = smem;                        // [kv_len]
    float *scratch = logit + kv_len;            // [blockDim.x]
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kvh = h / group;
    const __half *qh = q + static_cast<int64_t>(h) * head_dim;
    const int64_t vbase = static_cast<int64_t>(kvh) * head_stride;

    float mx = -INFINITY;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        // Sliding window: only the last `window` positions are visible.
        const bool visible = (window == 0) || (j > pos - window);
        float val = -INFINITY;
        if (visible) {
            const __half *kj = d_k + static_cast<int64_t>(j) * pos_stride + vbase;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i)
                dot = __fadd_rn(dot, __fmul_rn(__half2float(qh[i]),
                                               __half2float(kj[i])));
            val = __fmul_rn(dot, scale);
            if (bias)
                val = __fadd_rn(val, bias[static_cast<int64_t>(h) * kv_len + j]);
        }
        logit[j] = val;
        mx = fmaxf(mx, val);
    }
    mx = BlockMax(mx, scratch);

    float sum = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        const float e = (logit[j] == -INFINITY) ? 0.0f : expf(logit[j] - mx);
        logit[j] = e;
        sum += e;
    }
    sum = BlockSum(sum, scratch);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    __half *oh = out + static_cast<int64_t>(h) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            const float p = logit[j] * inv;
            if (p == 0.0f) continue;
            acc = __fadd_rn(
                acc, __fmul_rn(p,
                               __half2float(d_v[static_cast<int64_t>(j) * pos_stride +
                                                vbase + i])));
        }
        oh[i] = __float2half(acc);
    }
}

// ── 4. static short convolution: conv(x)+x, NO normalizer ─────────────────
// One thread per channel: it owns the whole [k] ring for that channel, so no
// cross-thread synchronisation is needed for the shift.
__global__ void short_conv_kernel(const __half *__restrict__ x_t,
                                  float *__restrict__ state,
                                  const float *__restrict__ w,
                                  __half *__restrict__ out, int channels, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int j = 0; j + 1 < k; ++j)
        state[static_cast<int64_t>(j) * channels + c] =
            state[static_cast<int64_t>(j + 1) * channels + c];
    const float cur = __half2float(x_t[c]);
    state[static_cast<int64_t>(k - 1) * channels + c] = cur;

    // conv1d taps: w[c][j] multiplies state[j], and state[k-1] is the CURRENT
    // token. The residual add is the reference's `hidden_states + residual`
    // with NO 1/sqrt(k+1) factor (that is inkling2's, not this family's).
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const float wv = w[static_cast<int64_t>(c) * k + j];
        const float sv = state[static_cast<int64_t>(j) * channels + c];
        acc = __fadd_rn(acc, __fmul_rn(wv, sv));
    }
    out[c] = __float2half(__fadd_rn(acc, cur));
}

__global__ void short_conv_f32_kernel(const float *__restrict__ x_t,
                                      float *__restrict__ state,
                                      const float *__restrict__ w,
                                      float *__restrict__ out, int channels,
                                      int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (int j = 0; j + 1 < k; ++j)
        state[static_cast<int64_t>(j) * channels + c] =
            state[static_cast<int64_t>(j + 1) * channels + c];
    const float cur = x_t[c];
    state[static_cast<int64_t>(k - 1) * channels + c] = cur;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const float wv = w[static_cast<int64_t>(c) * k + j];
        const float sv = state[static_cast<int64_t>(j) * channels + c];
        acc = __fadd_rn(acc, __fmul_rn(wv, sv));
    }
    out[c] = __fadd_rn(acc, cur);
}

// ── 5. router: joint routed+shared renorm ─────────────────────────────────
__global__ void router_kernel(const float *__restrict__ w,
                              const float *__restrict__ bias,
                              const __half *__restrict__ x, int E, int S, int K,
                              int H, float route_scale, float global_scale,
                              RoutedSlot *__restrict__ out,
                              int32_t *__restrict__ out_ids,
                              float *__restrict__ gammas)
{
    extern __shared__ float smem[];
    float *logit = smem;                    // [E+S] raw router logits
    float *select_score = logit + E + S;    // [E] sigmoid+bias
    float *scratch = select_score + E;      // [blockDim.x]
    __half *xs = reinterpret_cast<__half *>(scratch + blockDim.x);  // [H]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    for (int k = tid; k < H; k += blockDim.x) xs[k] = x[k];
    __syncthreads();

    // One warp per row: 32 lanes stride the hidden dim, shfl-reduce.
    for (int e = warp; e < E + S; e += warps) {
        const float *we = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = lane; k < H; k += 32)
            acc = __fadd_rn(acc, __fmul_rn(we[k], __half2float(xs[k])));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            logit[e] = acc;
            if (e < E) {
                const float s = 1.0f / (1.0f + expf(-acc));
                select_score[e] = s + (bias ? bias[e] : 0.0f);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        bool taken[512];
        int sel[32];
        for (int e = 0; e < E && e < 512; ++e) taken[e] = false;
        for (int k = 0; k < K && k < 32; ++k) {
            int best = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < E; ++e) {
                if (e < 512 && taken[e]) continue;
                // Ties resolve to the lower id; a NaN selection score is never
                // chosen.
                if (select_score[e] > best_v) {
                    best_v = select_score[e];
                    best = e;
                }
            }
            if (best < 0) {
                best = 0;
                while (best < E && taken[best]) ++best;
                if (best >= E) break;
            }
            taken[best] = true;
            sel[k] = best;
        }
        for (int i = 1; i < K; ++i) {
            const int v = sel[i];
            int j = i - 1;
            while (j >= 0 && sel[j] > v) {
                sel[j + 1] = sel[j];
                --j;
            }
            sel[j + 1] = v;
        }
        // Joint renorm: softmax over logsigmoid of the K selected routed +
        // S shared logits (the reference's logsumexp over the concatenated
        // topk_logits). The selection bias never enters the weight.
        float ls[64];
        float mx = -INFINITY;
        for (int k = 0; k < K; ++k) {
            ls[k] = -logf(1.0f + expf(-logit[sel[k]]));
            mx = fmaxf(mx, ls[k]);
        }
        for (int s = 0; s < S; ++s) {
            ls[K + s] = -logf(1.0f + expf(-logit[E + s]));
            mx = fmaxf(mx, ls[K + s]);
        }
        float sum = 0.0f;
        for (int i = 0; i < K + S; ++i) {
            ls[i] = expf(ls[i] - mx);
            sum += ls[i];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const float scale = __fmul_rn(route_scale, global_scale);
        for (int k = 0; k < K; ++k) {
            out[k].eid = sel[k];
            out[k].coeff = __fmul_rn(__fmul_rn(ls[k], inv), scale);
            if (out_ids) out_ids[k] = sel[k];
        }
        for (int s = 0; s < S; ++s)
            gammas[s] = __fmul_rn(__fmul_rn(ls[K + s], inv), scale);
    }
}

// ── 6. plain SwiGLU (no softcaps) ─────────────────────────────────────────
__global__ void swiglu_kernel(const __half *__restrict__ gate,
                              const __half *__restrict__ up,
                              __half *__restrict__ out, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2half(__fmul_rn(__fmul_rn(g, sig), u));
}

// ── 7. accumulation and plumbing ──────────────────────────────────────────
__global__ void expert_accumulate_kernel(float *__restrict__ acc,
                                         const __half *__restrict__ y,
                                         const RoutedSlot *slots, int count,
                                         int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float v = 0.0f;
    for (int k = 0; k < count; ++k)
        v = __fadd_rn(v, __fmul_rn(slots[k].coeff,
                                   __half2float(y[static_cast<int64_t>(k) * rows + i])));
    acc[i] = v;
}

__global__ void expert_accumulate_f32_kernel(float *__restrict__ acc,
                                             const float *__restrict__ y,
                                             const RoutedSlot *slots, int count,
                                             int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float v = 0.0f;
    for (int k = 0; k < count; ++k)
        v = __fadd_rn(v, __fmul_rn(slots[k].coeff,
                                   y[static_cast<int64_t>(k) * rows + i]));
    acc[i] = v;
}

__global__ void shared_accumulate_kernel(float *__restrict__ acc,
                                         const __half *__restrict__ y,
                                         float gamma, int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    acc[i] = __fadd_rn(acc[i], __fmul_rn(gamma, __half2float(y[i])));
}

__global__ void shared_accumulate_f32_kernel(float *__restrict__ acc,
                                             const float *__restrict__ y,
                                             float gamma, int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    acc[i] = __fadd_rn(acc[i], __fmul_rn(gamma, y[i]));
}

__global__ void scale_f32_kernel(float *__restrict__ x, int64_t n, float scale)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    x[i] = __fmul_rn(x[i], scale);
}

__global__ void norm_f32_kernel(const float *__restrict__ x,
                                const __half *__restrict__ w,
                                float *__restrict__ out, int hidden, float eps)
{
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float *xr = x + static_cast<int64_t>(row) * hidden;
    float *orow = out + static_cast<int64_t>(row) * hidden;
    float ss = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
        ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
    ss = BlockSum(ss, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += blockDim.x)
        orow[i] = __fmul_rn(__fmul_rn(xr[i], inv), __half2float(w[i]));
}

__global__ void matvec_f16_kernel(const __half *__restrict__ w,
                                  const __half *__restrict__ x,
                                  float *__restrict__ y, int cols)
{
    extern __shared__ float scratch[];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const __half *wr = w + static_cast<int64_t>(r) * cols;
    float acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x)
        acc = __fadd_rn(acc, __fmul_rn(__half2float(wr[c]),
                                       __half2float(x[c])));
    acc = BlockSum(acc, scratch);
    if (tid == 0) y[r] = acc;
}

inline int Blocks(int64_t n, int threads = kThreads)
{
    return static_cast<int>((n + threads - 1) / threads);
}

}  // namespace

cudaError_t RunHeadNorm(cudaStream_t stream, const float *x, const __half *w,
                        float *out, int rows, int dim, float eps)
{
    if (!x || !w || !out || rows <= 0 || dim <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    head_norm_kernel<<<rows, kThreads, shmem, stream>>>(x, w, out, dim, eps);
    return cudaGetLastError();
}

cudaError_t RunRelBias(cudaStream_t stream, const float *rel_states,
                       const float *proj, float *bias, int heads, int d_rel,
                       int rel_extent, int q_pos, int kv_len)
{
    if (!rel_states || !proj || !bias) return cudaErrorInvalidValue;
    if (heads <= 0 || d_rel <= 0 || rel_extent <= 0 || kv_len <= 0)
        return cudaErrorInvalidValue;
    lsxcommon::kernels::RelBiasKernel<><<<heads, kThreads, 0, stream>>>(
        rel_states, proj, bias, d_rel, rel_extent, q_pos, kv_len);
    return cudaGetLastError();
}

cudaError_t RunLogScale(cudaStream_t stream, float *q, int64_t nq, float *rel,
                        int64_t nrel, float tau)
{
    if (!q || nq <= 0) return cudaErrorInvalidValue;
    const int64_t n = std::max<int64_t>(nq, nrel);
    lsxcommon::kernels::LogScaleKernel<><<<Blocks(n), kThreads, 0, stream>>>(q, nq, rel, nrel, tau);
    return cudaGetLastError();
}

cudaError_t RunKvAppend(cudaStream_t stream, const __half *k_new,
                        const __half *v_new, __half *d_k, __half *d_v,
                        const KvCacheGeometry &geom, int pos)
{
    if (!k_new || !v_new || !d_k || !d_v) return cudaErrorInvalidValue;
    if (pos < 0 || pos >= geom.capacity) return cudaErrorInvalidValue;
    const int threads = 128;
    if (geom.head_dim > threads) return cudaErrorInvalidValue;
    kv_append_kernel<<<static_cast<int>(geom.kv_heads), threads, 0, stream>>>(
        k_new, v_new, d_k, d_v, static_cast<int>(geom.head_dim),
        geom.pos_stride, geom.head_stride, pos);
    return cudaGetLastError();
}

cudaError_t RunAttention(cudaStream_t stream, const __half *q,
                         const __half *d_k, const __half *d_v,
                         const KvCacheGeometry &geom, int heads, int pos,
                         const float *bias, int window, float scale,
                         __half *out)
{
    if (!q || !d_k || !d_v || !out) return cudaErrorInvalidValue;
    if (pos < 0 || pos >= geom.capacity) return cudaErrorInvalidValue;
    if (heads <= 0 || geom.kv_heads <= 0 || heads % geom.kv_heads != 0)
        return cudaErrorInvalidValue;
    if (geom.head_dim <= 0 || geom.head_dim > 1024) return cudaErrorInvalidValue;
    const int kv_len = pos + 1;
    if (kv_len > kMaxKvShmem) return cudaErrorInvalidValue;
    const size_t shmem =
        (static_cast<size_t>(kv_len) + kThreads) * sizeof(float);
    attention_kernel<<<heads, kThreads, shmem, stream>>>(
        q, d_k, d_v, bias, out, heads, static_cast<int>(geom.kv_heads),
        static_cast<int>(geom.head_dim), geom.pos_stride, geom.head_stride,
        pos, window, scale, kv_len);
    return cudaGetLastError();
}

cudaError_t RunShortConv(cudaStream_t stream, const __half *x_t, float *state,
                         const float *w, __half *out, int channels, int k)
{
    if (!x_t || !state || !w || !out) return cudaErrorInvalidValue;
    if (channels <= 0 || k <= 0) return cudaErrorInvalidValue;
    short_conv_kernel<<<Blocks(channels), kThreads, 0, stream>>>(
        x_t, state, w, out, channels, k);
    return cudaGetLastError();
}

cudaError_t RunShortConvF32(cudaStream_t stream, const float *x_t, float *state,
                            const float *w, float *out, int channels, int k)
{
    if (!x_t || !state || !w || !out) return cudaErrorInvalidValue;
    if (channels <= 0 || k <= 0) return cudaErrorInvalidValue;
    short_conv_f32_kernel<<<Blocks(channels), kThreads, 0, stream>>>(
        x_t, state, w, out, channels, k);
    return cudaGetLastError();
}

cudaError_t RunRouter(cudaStream_t stream, const float *router_w,
                      const float *router_bias, const __half *x, int E, int S,
                      int K, int H, float route_scale, float global_scale,
                      RoutedSlot *d_slots, int32_t *d_ids, float *d_gammas)
{
    if (!router_w || !x || !d_slots || !d_gammas) return cudaErrorInvalidValue;
    if (E <= 0 || E > 512 || S < 0 || S > 32 || K <= 0 || K > 32 || K > E ||
        H <= 0 || K + S > 64)
        return cudaErrorInvalidValue;
    const size_t shmem = (static_cast<size_t>(E + S) + E) * sizeof(float) +
                         static_cast<size_t>(kThreads) * sizeof(float) +
                         static_cast<size_t>(H) * sizeof(__half);
    if (shmem > 48 * 1024) return cudaErrorInvalidValue;
    router_kernel<<<1, kThreads, shmem, stream>>>(
        router_w, router_bias, x, E, S, K, H, route_scale, global_scale,
        d_slots, d_ids, d_gammas);
    return cudaGetLastError();
}

cudaError_t RunSwiGLU(cudaStream_t stream, const __half *gate,
                      const __half *up, __half *out, int64_t n)
{
    if (!gate || !up || !out) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    swiglu_kernel<<<Blocks(n), kThreads, 0, stream>>>(gate, up, out, n);
    return cudaGetLastError();
}

cudaError_t RunExpertAccumulate(cudaStream_t stream, float *acc,
                                const __half *y, const RoutedSlot *slots,
                                int count, int rows)
{
    if (!acc || !y || !slots || count <= 0 || rows <= 0)
        return cudaErrorInvalidValue;
    expert_accumulate_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        acc, y, slots, count, rows);
    return cudaGetLastError();
}

cudaError_t RunExpertAccumulateF32(cudaStream_t stream, float *acc,
                                   const float *y, const RoutedSlot *slots,
                                   int count, int rows)
{
    if (!acc || !y || !slots || count <= 0 || rows <= 0)
        return cudaErrorInvalidValue;
    expert_accumulate_f32_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        acc, y, slots, count, rows);
    return cudaGetLastError();
}

cudaError_t RunSharedAccumulate(cudaStream_t stream, float *acc,
                                const __half *y, float gamma, int rows)
{
    if (!acc || !y || rows <= 0) return cudaErrorInvalidValue;
    shared_accumulate_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        acc, y, gamma, rows);
    return cudaGetLastError();
}

cudaError_t RunSharedAccumulateF32(cudaStream_t stream, float *acc,
                                   const float *y, float gamma, int rows)
{
    if (!acc || !y || rows <= 0) return cudaErrorInvalidValue;
    shared_accumulate_f32_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        acc, y, gamma, rows);
    return cudaGetLastError();
}

cudaError_t RunEmbedGather(cudaStream_t stream, const __half *table,
                           const int32_t *d_id, __half *out, int hidden)
{
    if (table == nullptr || d_id == nullptr || out == nullptr || hidden <= 0)
        return cudaErrorInvalidValue;
    lsxcommon::kernels::EmbedGatherRowKernel<><<<1, 512, 0, stream>>>(table, d_id, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunNormF32In(cudaStream_t stream, const float *x, const __half *w,
                         __half *out, int rows, int hidden, float eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    lsxcommon::kernels::NormF32InKernel<><<<rows, kThreads, shmem, stream>>>(x, w, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunNormF32(cudaStream_t stream, const float *x, const __half *w,
                       float *out, int rows, int hidden, float eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    norm_f32_kernel<<<rows, kThreads, shmem, stream>>>(x, w, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunHalfToF32(cudaStream_t stream, const __half *src, float *dst,
                         int64_t n)
{
    if (!src || !dst) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F16ToF32Kernel<><<<Blocks(n), kThreads, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}

cudaError_t RunCastToHalf(cudaStream_t stream, const float *acc, __half *out,
                          int64_t n)
{
    if (!acc || !out) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F32ToF16Kernel<><<<Blocks(n), kThreads, 0, stream>>>(acc, out, n);
    return cudaGetLastError();
}

cudaError_t RunScaleToHalf(cudaStream_t stream, const float *src, __half *dst,
                           int64_t n, float scale)
{
    if (!src || !dst) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::ScaleToHalfKernel<><<<Blocks(n), kThreads, 0, stream>>>(src, dst, n, scale);
    return cudaGetLastError();
}

cudaError_t RunAddF32(cudaStream_t stream, float *acc, const float *add,
                      int64_t n)
{
    if (!acc || !add) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::AddF32Kernel<><<<Blocks(n), kThreads, 0, stream>>>(acc, add, n);
    return cudaGetLastError();
}

cudaError_t RunMatVecF16(cudaStream_t stream, const __half *w, const __half *x,
                         float *y, int rows, int cols)
{
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    matvec_f16_kernel<<<rows, kThreads, shmem, stream>>>(w, x, y, cols);
    return cudaGetLastError();
}

cudaError_t RunScaleF32(cudaStream_t stream, float *x, int64_t n, float scale)
{
    if (!x) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    scale_f32_kernel<<<Blocks(n), kThreads, 0, stream>>>(x, n, scale);
    return cudaGetLastError();
}

}  // namespace lsxcommon::inkling::base