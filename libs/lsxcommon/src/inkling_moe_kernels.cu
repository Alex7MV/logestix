// Inkling-2-Small numeric kernels (Model Ingestion Protocol stages 5/6).
//
// Shared-memory sandbox + deterministic reductions: every reduction below is a
// fixed-order tree over shared memory with thread-0-only publication, and no
// kernel uses atomicAdd. Nothing here feeds `final_logits` through an
// order-dependent accumulation, so the bit-exact cycle check of stage 8 holds.
//
// Numeric contract per kernel is stated in inkling_moe_kernels.h; the host
// references that pin them are tests/test_inkling_moe_kernels.cu.
#include "lsxcommon/inkling_moe_kernels.h"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_misc.cuh"

#include "nvfp4_codec.h"

#include <algorithm>
#include <cmath>

namespace lsxcommon::inkling {
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

// ── 1. AttnRes depth mixture ──────────────────────────────────────────────
// One block per query. The source rows are normalised on the fly (the norm is
// recomputed once per source, which is cheaper than spilling a
// [n_src, hidden] FP32 copy for n_src <= 9).
__global__ void inkling_depth_mix_kernel(const float *__restrict__ sources,
                                         int n_src,
                                         const float *__restrict__ queries,
                                         float *__restrict__ out, int hidden,
                                         float eps)
{
    extern __shared__ float smem[];
    float *q = smem;                     // [hidden]
    float *scratch = q + hidden;          // [blockDim.x]
    float *logits = scratch + blockDim.x; // [n_src]
    float *invs = logits + n_src;         // [n_src]
    float *weights = invs + n_src;        // [n_src]

    const int tid = threadIdx.x;
    const int qIdx = blockIdx.x;
    for (int i = tid; i < hidden; i += blockDim.x)
        q[i] = queries[static_cast<int64_t>(qIdx) * hidden + i];
    __syncthreads();

    const float qscale = rsqrtf(static_cast<float>(hidden));
    for (int j = 0; j < n_src; ++j) {
        const float *s = sources + static_cast<int64_t>(j) * hidden;
        float ss = 0.0f;
        for (int i = tid; i < hidden; i += blockDim.x)
            ss = __fadd_rn(ss, __fmul_rn(s[i], s[i]));
        ss = BlockSum(ss, scratch);
        const float inv = rsqrtf(ss / static_cast<float>(hidden) + eps);
        if (tid == 0) invs[j] = inv;
        float dot = 0.0f;
        for (int i = tid; i < hidden; i += blockDim.x)
            dot = __fadd_rn(dot, __fmul_rn(q[i], __fmul_rn(s[i], inv)));
        dot = BlockSum(dot, scratch);
        if (tid == 0) logits[j] = __fmul_rn(dot, qscale);
        __syncthreads();
    }

    if (tid == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < n_src; ++j) mx = fmaxf(mx, logits[j]);
        float sum = 0.0f;
        for (int j = 0; j < n_src; ++j) {
            weights[j] = expf(logits[j] - mx);
            sum += weights[j];
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int j = 0; j < n_src; ++j) weights[j] *= inv;
    }
    __syncthreads();

    float *o = out + static_cast<int64_t>(qIdx) * hidden;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_src; ++j)
            acc = __fadd_rn(
                acc, __fmul_rn(weights[j],
                               __fmul_rn(sources[static_cast<int64_t>(j) * hidden + i],
                                         invs[j])));
        o[i] = acc;
    }
}

// ── 2. L2 clip ─────────────────────────────────────────────────────────────
// One block per head row; the scaled vector is staged in shared memory so the
// norm and the scale-down see exactly the same values.
__global__ void inkling_l2_clip_kernel(float *__restrict__ x,
                                       const float *__restrict__ scale,
                                       int dim, float threshold)
{
    extern __shared__ float smem[];
    float *v = smem;
    float *scratch = v + dim;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float *xr = x + static_cast<int64_t>(row) * dim;
    const float *sr = scale ? scale + static_cast<int64_t>(row) * dim : nullptr;

    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float val = sr ? __fmul_rn(xr[i], sr[i]) : xr[i];
        v[i] = val;
        ss = __fadd_rn(ss, __fmul_rn(val, val));
    }
    ss = BlockSum(ss, scratch);
    const float norm = sqrtf(ss);
    const float factor = fminf(threshold / (norm + 1e-6f), 1.0f);
    for (int i = tid; i < dim; i += blockDim.x)
        xr[i] = __fmul_rn(v[i], factor);
}

// ── 3. unweighted RMSNorm (in place, FP32) ────────────────────────────────
__global__ void inkling_unweighted_norm_kernel(float *__restrict__ x, int dim,
                                               float eps)
{
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float *xr = x + static_cast<int64_t>(row) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        ss = __fadd_rn(ss, __fmul_rn(xr[i], xr[i]));
    ss = BlockSum(ss, scratch);
    const float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += blockDim.x) xr[i] = __fmul_rn(xr[i], inv);
}

// ── 4. relative-position bias ─────────────────────────────────────────────
// One block per head; each thread owns a key position and computes its own
// d_rel-length dot, so the whole gather is one pass with no scratch.

// ── 5. attention with learned sinks ───────────────────────────────────────
__global__ void inkling_kv_append_kernel(const __half *__restrict__ k_new,
                                         const __half *__restrict__ v_new,
                                         __half *__restrict__ d_k,
                                         __half *__restrict__ d_v,
                                         int kv_heads, int head_dim,
                                         int64_t pos_stride,
                                         int64_t head_stride, int pos)
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

__global__ void inkling_attention_kernel(const __half *__restrict__ q,
                                         const __half *__restrict__ d_k,
                                         const __half *__restrict__ d_v,
                                         const float *__restrict__ sink,
                                         const float *__restrict__ bias,
                                         __half *__restrict__ out, int heads,
                                         int kv_heads, int head_dim,
                                         int64_t pos_stride,
                                         int64_t head_stride, int pos,
                                         int window, float scale, int kv_len)
{
    extern __shared__ float smem[];
    float *logit = smem;                       // [kv_len]
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
            if (bias) val = __fadd_rn(val, bias[static_cast<int64_t>(h) * kv_len + j]);
        }
        logit[j] = val;
        mx = fmaxf(mx, val);
    }
    mx = BlockMax(mx, scratch);
    if (sink) mx = fmaxf(mx, sink[h]);

    float sum = 0.0f;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        const float e = (logit[j] == -INFINITY) ? 0.0f : expf(logit[j] - mx);
        logit[j] = e;
        sum += e;
    }
    sum = BlockSum(sum, scratch);
    if (sink) sum += expf(sink[h] - mx);
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

__global__ void inkling_gate_mul_kernel(const __half *__restrict__ x,
                                        const __half *__restrict__ gate,
                                        __half *__restrict__ out, int64_t n)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    out[i] = __float2half(__fmul_rn(__half2float(x[i]), 1.0f / (1.0f + expf(-g))));
}


__global__ void inkling_dynamic_taps_kernel(const float *__restrict__ static_w,
                                            const float *__restrict__ proj_out,
                                            float *__restrict__ taps,
                                            int channels, int head_dim, int k)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    const int head = ch / head_dim;
    for (int lag = 0; lag < k; ++lag) {
        const float base = static_w[static_cast<int64_t>(lag) * channels + ch];
        const float dyn = proj_out[static_cast<int64_t>(head) * k + lag];
        taps[static_cast<int64_t>(ch) * k + lag] = __fadd_rn(base, dyn);
    }
}

// ── 6. short convolution ──────────────────────────────────────────────────
// One thread per channel: it owns the whole [k] ring for that channel, so no
// cross-thread synchronisation is needed for the shift.
__global__ void inkling_short_conv_kernel(const __half *__restrict__ x_t,
                                          float *__restrict__ state,
                                          const float *__restrict__ w,
                                          const float *__restrict__ dyn_taps,
                                          __half *__restrict__ out, int channels,
                                          int k, float inv_scale)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // Push the current token into the newest slot, shifting the ring.
    for (int j = 0; j + 1 < k; ++j)
        state[static_cast<int64_t>(j) * channels + c] =
            state[static_cast<int64_t>(j + 1) * channels + c];
    const float cur = __half2float(x_t[c]);
    state[static_cast<int64_t>(k - 1) * channels + c] = cur;

    float acc = 0.0f;
    if (dyn_taps == nullptr) {
        // conv1d taps: w[c][j] multiplies state[j], and state[k-1] is the
        // CURRENT token.
        for (int j = 0; j < k; ++j) {
            const float wv = w[static_cast<int64_t>(c) * k + j];
            const float sv = state[static_cast<int64_t>(j) * channels + c];
            acc = __fadd_rn(acc, __fmul_rn(wv, sv));
        }
    } else {
        // lag taps: lag 0 (dyn[c][0]) is the CURRENT token, so lag l reads
        // state[k-1-l].
        for (int lag = 0; lag < k; ++lag) {
            const float wv = dyn_taps[static_cast<int64_t>(c) * k + lag];
            const float sv =
                state[static_cast<int64_t>(k - 1 - lag) * channels + c];
            acc = __fadd_rn(acc, __fmul_rn(wv, sv));
        }
    }
    out[c] = __float2half(__fmul_rn(__fadd_rn(acc, cur), inv_scale));
}

// ── 7. softcapped SwiGLU ──────────────────────────────────────────────────
__global__ void inkling_softcap_swiglu_kernel(const __half *__restrict__ gate,
                                              const __half *__restrict__ up,
                                              __half *__restrict__ out, int64_t n,
                                              float gate_cap, float up_cap)
{
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float lin = tanhf(g / gate_cap) * gate_cap;
    const float sig = 1.0f / (1.0f + expf(-g));
    const float cap_u = tanhf(u / up_cap) * up_cap;
    out[i] = __float2half(__fmul_rn(__fmul_rn(lin, sig), cap_u));
}

// ── 8. router ─────────────────────────────────────────────────────────────
__global__ void inkling_router_kernel(const float *__restrict__ w,
                                      const float *__restrict__ bias,
                                      const __half *__restrict__ x, int E, int K,
                                      int H, float route_scale,
                                      InklingRoutedSlot *__restrict__ out,
                                      int32_t *__restrict__ out_ids)
{
    extern __shared__ float smem[];
    float *score = smem;                                 // [E]
    float *select_score = score + E;                     // [E]
    float *scratch = select_score + E;                   // [blockDim.x]
    __half *xs = reinterpret_cast<__half *>(scratch + blockDim.x);  // [H]

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    for (int k = tid; k < H; k += blockDim.x) xs[k] = x[k];
    __syncthreads();

    // One warp per expert: 32 lanes stride the hidden dim, shfl-reduce.
    for (int e = warp; e < E; e += warps) {
        const float *we = w + static_cast<int64_t>(e) * H;
        float acc = 0.0f;
        for (int k = lane; k < H; k += 32)
            acc = __fadd_rn(acc, __fmul_rn(we[k], __half2float(xs[k])));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            const float s = 1.0f / (1.0f + expf(-acc));
            score[e] = s;
            select_score[e] = s + (bias ? bias[e] : 0.0f);
        }
    }
    __syncthreads();

    if (tid == 0) {
        bool taken[512];
        int sel[32];
        for (int e = 0; e < E && e < 512; ++e) taken[e] = false;
        for (int k = 0; k < K && k < 32; ++k) sel[k] = k;
        for (int k = 0; k < K && k < 32; ++k) {
            int best = -1;
            float best_v = -INFINITY;
            for (int e = 0; e < E; ++e) {
                if (e < 512 && taken[e]) continue;
                // Ties resolve to the lower id; a NaN selection score is never
                // chosen, so it can only fall back to an untouched expert (in
                // bounds), which surfaces as text rather than an illegal access.
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
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += isnan(score[sel[k]]) ? 0.0f : score[sel[k]];
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k = 0; k < K; ++k) {
            const float sc = isnan(score[sel[k]]) ? 0.0f : score[sel[k]];
            out[k].eid = sel[k];
            // The selected SCORES renormalise to 1, then scale by route_scale
            // (the selection bias never enters the weight).
            out[k].coeff = __fmul_rn(__fmul_rn(sc, inv), route_scale);
            if (out_ids) out_ids[k] = sel[k];
        }
    }
}

// ── 9. routed accumulation and residual plumbing ──────────────────────────
__global__ void inkling_expert_accumulate_kernel(float *__restrict__ acc,
                                                 const __half *__restrict__ y,
                                                 const InklingRoutedSlot *slots,
                                                 int count, int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float v = 0.0f;
    for (int k = 0; k < count; ++k)
        v = __fadd_rn(v, __fmul_rn(slots[k].coeff,
                                   __half2float(y[static_cast<int64_t>(k) * rows + i])));
    acc[i] = v;
}





__global__ void inkling_logit_softcap_kernel(const float *__restrict__ logits,
                                             float *__restrict__ out, int n,
                                             float cap)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = tanhf(logits[i] / cap) * cap;
}


// ── 9. plumbing ───────────────────────────────────────────────────────────
__global__ void inkling_matvec_f16_h_kernel(const __half *__restrict__ w,
                                            const __half *__restrict__ x,
                                            __half *__restrict__ y, int rows,
                                            int cols)
{
    extern __shared__ float scratch[];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const __half *wr = w + static_cast<int64_t>(r) * cols;
    float acc = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x)
        acc = __fadd_rn(acc, __fmul_rn(__half2float(wr[c]), __half2float(x[c])));
    acc = BlockSum(acc, scratch);
    if (tid == 0) y[r] = __float2half(acc);
}



__global__ void inkling_matvec_f16_kernel(const __half *__restrict__ w,
                                          const __half *__restrict__ x,
                                          float *__restrict__ y, int rows,
                                          int cols)
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

__global__ void inkling_embed_row_kernel(const uint8_t *__restrict__ packed,
                                         const uint8_t *__restrict__ scales,
                                         float global_scale,
                                         const int32_t *__restrict__ d_id,
                                         __half *__restrict__ out, int hidden)
{
    const int64_t row = d_id[0];
    const uint8_t *p = packed + row * (hidden / 2);
    const uint8_t *s = scales + row * (hidden / 16);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const uint8_t byte = p[i >> 1];
        const uint8_t nib = (i & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0xF);
        const float scl =
            gpu::nvfp4::E4M3ScaleToFloat(s[i >> 4]) * global_scale;
        out[i] = __float2half(__fmul_rn(gpu::nvfp4::E2M1Value(nib), scl));
    }
}

inline int Blocks(int64_t n, int threads = kThreads)
{
    return static_cast<int>((n + threads - 1) / threads);
}

// ── 7b. FP8 expert GEMV (the release's MXFP8 plane) ───────────────────────
__device__ __forceinline__ float Ue8m0ToFloat(uint8_t b)
{
    return exp2f(static_cast<float>(static_cast<int>(b) - 127));
}

// E4M3 *weights* are SIGNED, while `E4M3ScaleToFloat` decodes the unsigned block
// scale type and masks bit 7 by design. Reusing it for weights silently drops
// every negative weight's sign, which turns a signed dot product into a sum of
// magnitudes — measured 43x too large on a 4608-wide gate row. Take the
// magnitude from the canonical codec (never a local table) and apply the sign.
__device__ __forceinline__ float E4m3WeightToFloat(uint8_t b)
{
    const float mag =
        gpu::nvfp4::E4M3ScaleToFloat(static_cast<uint8_t>(b & 0x7Fu));
    return (b & 0x80u) ? -mag : mag;
}

// One warp per output row: 32 lanes stride K, then a shfl reduction. Grid y picks
// the expert, so the eight routed slots run in parallel like the NVFP4 batch.
__global__ void inkling_fp8_expert_gemm_kernel(
    const __half *__restrict__ x, InklingFp8ExpertPtrs ptrs, int x_stride, int K,
    int N, __half *__restrict__ y, bool acc)
{
    const int e = blockIdx.y;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= N) return;
    const uint8_t *w = ptrs.packed[e] + static_cast<int64_t>(row) * K;
    const uint8_t *s = ptrs.scales[e] + static_cast<int64_t>(row) * (K / 16);
    const __half *xe = x + static_cast<int64_t>(e) * x_stride;
    float a = 0.0f;
    for (int k = lane; k < K; k += 32)
        a = __fadd_rn(a,
                      __fmul_rn(__fmul_rn(__half2float(xe[k]),
                                          E4m3WeightToFloat(w[k])),
                                Ue8m0ToFloat(s[k >> 4])));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a += __shfl_down_sync(0xffffffffu, a, off);
    if (lane == 0) {
        __half *d = y + static_cast<int64_t>(e) * N + row;
        *d = acc ? __float2half(__half2float(*d) + a) : __float2half(a);
    }
}

}  // namespace

cudaError_t RunInklingFp8ExpertGemm(cudaStream_t stream, const __half *x,
                                    const InklingFp8ExpertPtrs &ptrs, int count,
                                    int x_stride, __half *y, int K, int N,
                                    bool acc)
{
    if (x == nullptr || y == nullptr) return cudaErrorInvalidValue;
    if (count <= 0 || count > 8 || K <= 0 || N <= 0) return cudaErrorInvalidValue;
    if ((K & 15) != 0) return cudaErrorInvalidValue;
    for (int e = 0; e < count; ++e)
        if (ptrs.packed[e] == nullptr || ptrs.scales[e] == nullptr)
            return cudaErrorInvalidValue;
    const dim3 grid(N, count);
    inkling_fp8_expert_gemm_kernel<<<grid, 32, 0, stream>>>(x, ptrs, x_stride, K,
                                                            N, y, acc);
    return cudaGetLastError();
}

cudaError_t RunInklingDepthMix(cudaStream_t stream, const float *sources,
                               int n_src, const float *queries, int num_queries,
                               float *out, int hidden, float eps)
{
    if (!sources || !queries || !out) return cudaErrorInvalidValue;
    if (n_src <= 0 || num_queries <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const size_t shmem = (static_cast<size_t>(hidden) + kThreads +
                          3 * static_cast<size_t>(n_src)) *
                         sizeof(float);
    if (shmem > 48 * 1024) return cudaErrorInvalidValue;
    inkling_depth_mix_kernel<<<num_queries, kThreads, shmem, stream>>>(
        sources, n_src, queries, out, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunInklingL2Clip(cudaStream_t stream, float *x, const float *scale,
                             int rows, int dim, float threshold)
{
    if (!x || rows <= 0 || dim <= 0) return cudaErrorInvalidValue;
    const size_t shmem = (static_cast<size_t>(dim) + kThreads) * sizeof(float);
    if (shmem > 48 * 1024) return cudaErrorInvalidValue;
    inkling_l2_clip_kernel<<<rows, kThreads, shmem, stream>>>(x, scale, dim,
                                                              threshold);
    return cudaGetLastError();
}

cudaError_t RunInklingUnweightedNorm(cudaStream_t stream, float *x, int rows,
                                     int dim, float eps)
{
    if (!x || rows <= 0 || dim <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    inkling_unweighted_norm_kernel<<<rows, kThreads, shmem, stream>>>(x, dim, eps);
    return cudaGetLastError();
}

cudaError_t RunInklingRelBias(cudaStream_t stream, const float *rel_states,
                              const float *proj, float *bias, int heads,
                              int d_rel, int rel_extent, int q_pos, int kv_len)
{
    if (!rel_states || !proj || !bias) return cudaErrorInvalidValue;
    if (heads <= 0 || d_rel <= 0 || rel_extent <= 0 || kv_len <= 0)
        return cudaErrorInvalidValue;
    lsxcommon::kernels::RelBiasKernel<><<<heads, kThreads, 0, stream>>>(
        rel_states, proj, bias, d_rel, rel_extent, q_pos, kv_len);
    return cudaGetLastError();
}

cudaError_t RunInklingKvAppend(cudaStream_t stream, const __half *k_new,
                               const __half *v_new, __half *d_k, __half *d_v,
                               const KvCacheGeometry &geom, int pos)
{
    if (!k_new || !v_new || !d_k || !d_v) return cudaErrorInvalidValue;
    if (pos < 0 || pos >= geom.capacity) return cudaErrorInvalidValue;
    const int threads = 128;
    if (geom.head_dim > threads) return cudaErrorInvalidValue;
    inkling_kv_append_kernel<<<static_cast<int>(geom.kv_heads), threads, 0,
                               stream>>>(
        k_new, v_new, d_k, d_v, static_cast<int>(geom.kv_heads),
        static_cast<int>(geom.head_dim), geom.pos_stride, geom.head_stride, pos);
    return cudaGetLastError();
}

cudaError_t RunInklingAttention(cudaStream_t stream, const __half *q,
                                const __half *d_k, const __half *d_v,
                                const KvCacheGeometry &geom, int heads, int pos,
                                const float *sink, const float *bias, int window,
                                float scale, __half *out)
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
    inkling_attention_kernel<<<heads, kThreads, shmem, stream>>>(
        q, d_k, d_v, sink, bias, out, heads, static_cast<int>(geom.kv_heads),
        static_cast<int>(geom.head_dim), geom.pos_stride, geom.head_stride, pos,
        window, scale, kv_len);
    return cudaGetLastError();
}

cudaError_t RunInklingGateMul(cudaStream_t stream, const __half *x,
                              const __half *gate, __half *out, int64_t n)
{
    if (!x || !gate || !out) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    inkling_gate_mul_kernel<<<Blocks(n), kThreads, 0, stream>>>(x, gate, out, n);
    return cudaGetLastError();
}

cudaError_t RunInklingShortConv(cudaStream_t stream, const __half *x_t,
                                float *state, const float *w,
                                const float *dyn_taps, __half *out, int channels,
                                int k, float inv_scale)
{
    if (!x_t || !state || !out) return cudaErrorInvalidValue;
    if (channels <= 0 || k <= 0) return cudaErrorInvalidValue;
    if (dyn_taps == nullptr && w == nullptr) return cudaErrorInvalidValue;
    inkling_short_conv_kernel<<<Blocks(channels), kThreads, 0, stream>>>(
        x_t, state, w, dyn_taps, out, channels, k, inv_scale);
    return cudaGetLastError();
}

cudaError_t RunInklingLogScale(cudaStream_t stream, float *q, float *rel,
                              int64_t nq, int64_t nrel, float tau)
{
    if (!q || nq <= 0) return cudaErrorInvalidValue;
    const int64_t n = std::max<int64_t>(nq, nrel);
    lsxcommon::kernels::LogScaleKernel<><<<Blocks(n), kThreads, 0, stream>>>(q, nq, rel, nrel,
                                                                tau);
    return cudaGetLastError();
}

cudaError_t RunInklingDynamicTaps(cudaStream_t stream, const float *static_w,
                                 const float *proj_out, float *taps,
                                 int channels, int head_dim, int k)
{
    if (!static_w || !proj_out || !taps) return cudaErrorInvalidValue;
    if (channels <= 0 || head_dim <= 0 || k <= 0) return cudaErrorInvalidValue;
    inkling_dynamic_taps_kernel<<<Blocks(channels), kThreads, 0, stream>>>(
        static_w, proj_out, taps, channels, head_dim, k);
    return cudaGetLastError();
}

cudaError_t RunInklingSoftcapSwiGLU(cudaStream_t stream, const __half *gate,
                                    const __half *up, __half *out, int64_t n,
                                    float gate_cap, float up_cap)
{
    if (!gate || !up || !out) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    inkling_softcap_swiglu_kernel<<<Blocks(n), kThreads, 0, stream>>>(
        gate, up, out, n, gate_cap, up_cap);
    return cudaGetLastError();
}

cudaError_t RunInklingRouter(cudaStream_t stream, const float *router_w,
                             const float *router_bias, const __half *x, int E,
                             int K, int H, float route_scale,
                             InklingRoutedSlot *d_slots, int32_t *d_ids)
{
    if (!router_w || !x || !d_slots) return cudaErrorInvalidValue;
    if (E <= 0 || E > 512 || K <= 0 || K > 32 || K > E || H <= 0)
        return cudaErrorInvalidValue;
    const size_t shmem = 2 * static_cast<size_t>(E) * sizeof(float) +
                         static_cast<size_t>(kThreads) * sizeof(float) +
                         static_cast<size_t>(H) * sizeof(__half);
    if (shmem > 48 * 1024) return cudaErrorInvalidValue;
    inkling_router_kernel<<<1, kThreads, shmem, stream>>>(
        router_w, router_bias, x, E, K, H, route_scale, d_slots, d_ids);
    return cudaGetLastError();
}

cudaError_t RunInklingExpertAccumulate(cudaStream_t stream, float *acc,
                                       const __half *y,
                                       const InklingRoutedSlot *slots, int count,
                                       int rows)
{
    if (!acc || !y || !slots || count <= 0 || rows <= 0)
        return cudaErrorInvalidValue;
    inkling_expert_accumulate_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        acc, y, slots, count, rows);
    return cudaGetLastError();
}

cudaError_t RunInklingCastToHalf(cudaStream_t stream, const float *acc,
                                 __half *out, int64_t n)
{
    if (!acc || !out) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F32ToF16Kernel<><<<Blocks(n), kThreads, 0, stream>>>(acc, out, n);
    return cudaGetLastError();
}

cudaError_t RunInklingAddF32(cudaStream_t stream, float *acc, const float *add,
                             int64_t n)
{
    if (!acc || !add) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::AddF32Kernel<><<<Blocks(n), kThreads, 0, stream>>>(acc, add, n);
    return cudaGetLastError();
}

cudaError_t RunInklingHalfToF32(cudaStream_t stream, const __half *src,
                                float *dst, int64_t n)
{
    if (!src || !dst) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::F16ToF32Kernel<><<<Blocks(n), kThreads, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}

cudaError_t RunInklingScaleToHalf(cudaStream_t stream, const float *src,
                                  __half *dst, int64_t n, float scale)
{
    if (!src || !dst) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    lsxcommon::kernels::ScaleToHalfKernel<><<<Blocks(n), kThreads, 0, stream>>>(src, dst, n,
                                                                    scale);
    return cudaGetLastError();
}

cudaError_t RunInklingLogitSoftcap(cudaStream_t stream, const float *logits,
                                   float *out, int n, float cap)
{
    if (!logits || !out || n <= 0 || !(cap > 0.0f)) return cudaErrorInvalidValue;
    inkling_logit_softcap_kernel<<<Blocks(n), kThreads, 0, stream>>>(logits, out,
                                                                    n, cap);
    return cudaGetLastError();
}

cudaError_t RunInklingMatVecF16ToHalf(cudaStream_t stream, const __half *w,
                                      const __half *x, __half *y, int rows,
                                      int cols)
{
    if (w == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0)
        return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    inkling_matvec_f16_h_kernel<<<rows, kThreads, shmem, stream>>>(w, x, y, rows,
                                                                   cols);
    return cudaGetLastError();
}

cudaError_t RunInklingEmbedGather(cudaStream_t stream, const __half *table,
                                  const int32_t *d_id, __half *out, int hidden)
{
    if (table == nullptr || d_id == nullptr || out == nullptr || hidden <= 0)
        return cudaErrorInvalidValue;
    lsxcommon::kernels::EmbedGatherRowKernel<><<<1, 512, 0, stream>>>(table, d_id, out, hidden);
    return cudaGetLastError();
}

cudaError_t RunInklingNormF32In(cudaStream_t stream, const float *x,
                                const __half *w, __half *out, int rows,
                                int hidden, float eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    lsxcommon::kernels::NormF32InKernel<><<<rows, kThreads, shmem, stream>>>(x, w, out,
                                                                  hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunInklingMatVecF16(cudaStream_t stream, const __half *w,
                                const __half *x, float *y, int rows, int cols)
{
    if (!w || !x || !y || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const size_t shmem = static_cast<size_t>(kThreads) * sizeof(float);
    inkling_matvec_f16_kernel<<<rows, kThreads, shmem, stream>>>(w, x, y, rows,
                                                                 cols);
    return cudaGetLastError();
}

cudaError_t RunInklingEmbedRow(cudaStream_t stream, const uint8_t *packed,
                               const uint8_t *scales, float global_scale,
                               const int32_t *d_id, __half *out, int hidden)
{
    if (!packed || !scales || !d_id || !out) return cudaErrorInvalidValue;
    if (hidden <= 0 || (hidden & 15) != 0) return cudaErrorInvalidValue;
    inkling_embed_row_kernel<<<1, 512, 0, stream>>>(packed, scales, global_scale,
                                                    d_id, out, hidden);
    return cudaGetLastError();
}

}  // namespace lsxcommon::inkling
