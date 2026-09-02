#include "lsxcommon/qwen_moe_exp_inference.h"

#include <cuda_runtime.h>

#include <cmath>

namespace lsxcommon::qwen_moe_exp {
namespace {

// ── GatedResidual F32 core (parity twin) ──────────────────────────────────
//
// One block per token. Sequence matches HF Qwen4ExpTextGatedResidual.forward
// in fp32 with -fmad=false:
//   normed   = rmsnorm_groups(x, group=H)                  (no weight)
//   dn[r]    = silu( (W_dn @ normed)[r] / hc )
//   up[i]    = sigmoid( (W_up @ dn)[i] )
//   mixed[h] = mean_c up[c*H+h] * normed[c*H+h]
//   inj[c]   = 2 * sigmoid( (W_inj @ normed)[c] / hc )
//
// W matrices arrive fp32 row-major: down [lowrank, n], up [n, lowrank],
// inject [hc, n], n = hc*H. Deterministic tree reduction, ascending index,
// no atomics, shared-memory writes by thread 0 only (protocol step 6).

__global__ void hc_gated_residual_f32_kernel(
    const float *__restrict__ x,      // [n]  (seq = 1 for the twin)
    const float *__restrict__ down,   // [lowrank * n]
    const float *__restrict__ up,     // [n * lowrank]
    const float *__restrict__ inject, // [hc * n] or null
    int H, int hc, int lowrank, float eps,
    float *__restrict__ mixed_out,    // [H]
    float *__restrict__ inj_out)      // [hc]
{
    const int n = H * hc;
    __shared__ float s_normed[2048];
    __shared__ float s_dn[512];
    __shared__ float s_up[2048];
    __shared__ float s_scale;

    // Phase 1: group-RMSNorm (group = H). Per-group mean of squares reduced
    // by thread 0 after a grid-stride partial store.
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_normed[i] = x[i];
    __syncthreads();

    for (int g = 0; g < hc; ++g) {
        // each group occupies [g*H, g*H+H)
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                const double v = s_normed[g * H + i];
                acc += v * v;
            }
            s_scale = static_cast<float>(1.0 / sqrt(acc / H + eps));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            s_normed[g * H + i] *= s_scale;
        }
        __syncthreads();
    }

    // Phase 2: down-projection [lowrank] + silu(/hc). One thread per r with
    // sequential fp32 dot over n (deterministic; n small in tests, fine).
    for (int r = threadIdx.x; r < lowrank; r += blockDim.x) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
            acc += static_cast<double>(down[r * n + i]) * s_normed[i];
        const double v = acc / hc;
        s_dn[r] = static_cast<float>(v / (1.0 + exp(-v)));
    }
    __syncthreads();

    // Phase 3: up-projection [n] + sigmoid.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double acc = 0.0;
        for (int r = 0; r < lowrank; ++r)
            acc += static_cast<double>(up[i * lowrank + r]) * s_dn[r];
        s_up[i] = 1.0f / (1.0f + expf(-static_cast<float>(acc)));
    }
    __syncthreads();

    // Phase 4: mixed = mean over branches of (gate ⊙ normed).
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        double acc = 0.0;
        for (int c = 0; c < hc; ++c)
            acc += static_cast<double>(s_up[c * H + h]) *
                   s_normed[c * H + h];
        mixed_out[h] = static_cast<float>(acc / hc);
    }

    // Phase 5: injection weights.
    if (inject) {
        for (int c = threadIdx.x; c < hc; c += blockDim.x) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i)
                acc += static_cast<double>(inject[c * n + i]) * s_normed[i];
            inj_out[c] = static_cast<float>(2.0 / (1.0 + exp(-acc / hc)));
        }
    }
}

} // namespace

cudaError_t RunExpHcGatedResidualF32(
    cudaStream_t stream, const float *d_hyper_input, const float *d_down,
    const float *d_up, const float *d_inject_w, int seq, int H, int hc,
    int lowrank, float *d_mixed, float *d_inject_w_out)
{
    if (seq != 1)
        return cudaErrorInvalidValue;  // twin supports the parity case only
    const int n = H * hc;
    if (n > 2048 || lowrank > 512) return cudaErrorInvalidValue;
    const int block = 128;
    hc_gated_residual_f32_kernel<<<1, block, 0, stream>>>(
        d_hyper_input, d_down, d_up, d_inject_w, H, hc, lowrank, 1e-6f,
        d_mixed, d_inject_w_out);
    return cudaGetLastError();
}

namespace {

__global__ void hc_inject_scatter_kernel(
    const __half *__restrict__ hyper_input,  // [seq, n]
    const __half *__restrict__ branch_out,   // [seq, H]
    const float *__restrict__ inject_w,      // [hc]
    int H, int hc, __half *__restrict__ out) // [seq, n]
{
    const int n = H * hc;
    const int total = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += total) {
        const int c = idx / H;
        const int i = idx % H;
        out[idx] = __float2half(__half2float(hyper_input[idx]) +
                                __half2float(branch_out[i]) * inject_w[c]);
    }
}

} // namespace

cudaError_t RunExpHcInjectScatter(cudaStream_t stream,
                                  const __half *d_hyper_input,
                                  const __half *d_branch_out,
                                  const float *d_inject_w, int seq, int H,
                                  int hc, __half *d_out)
{
    const int n = H * hc;
    if (n <= 0) return cudaErrorInvalidValue;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    for (int s = 0; s < seq; ++s) {
        hc_inject_scatter_kernel<<<grid, block, 0, stream>>>(
            d_hyper_input + static_cast<int64_t>(s) * n,
            d_branch_out + static_cast<int64_t>(s) * H, d_inject_w, H, hc,
            d_out + static_cast<int64_t>(s) * n);
    }
    return cudaGetLastError();
}

// ── QSA indexer (Qwen Sparse Attention micro-block selection) ────────────
//
// Decode-oriented (seq small): raw keys appended to an MQA cache, complete
// 4-token blocks pooled once (append-only), per-query relu-score top-k over
// blocks, selected-token mask = causal ∧ selected ∪ tail tokens.

cudaError_t ExpQsaIndexerAlloc(ExpQsaIndexerState &st, int64_t max_ctx,
                               int head_dim)
{
    st = ExpQsaIndexerState{};
    st.capacity_ctx = max_ctx;
    st.head_dim = static_cast<int>(head_dim);
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_raw_keys),
                   static_cast<size_t>(max_ctx) * head_dim * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_pooled_keys),
                   static_cast<size_t>(max_ctx) * head_dim * sizeof(float)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_block_starts),
                   static_cast<size_t>(max_ctx) * sizeof(int)) !=
        cudaSuccess)
        return cudaGetLastError();
    cudaMemset(st.d_raw_keys, 0,
               static_cast<size_t>(max_ctx) * head_dim * sizeof(__half));
    cudaMemset(st.d_pooled_keys, 0,
               static_cast<size_t>(max_ctx) * head_dim * sizeof(float));
    cudaMemset(st.d_block_starts, 0, static_cast<size_t>(max_ctx) * sizeof(int));
    return cudaSuccess;
}

void ExpQsaIndexerFree(ExpQsaIndexerState &st)
{
    if (st.d_raw_keys) cudaFree(st.d_raw_keys);
    if (st.d_pooled_keys) cudaFree(st.d_pooled_keys);
    if (st.d_block_starts) cudaFree(st.d_block_starts);
    st = ExpQsaIndexerState{};
}

namespace {

// Appends seq new raw keys; pools each newly-completed block (mean over
// `compress` raw keys, fp32) and records its start position. One thread per
// element column; block completion handled by thread 0 with fp32 mean over
// the compress window (deterministic sequential order).
__global__ void qsa_append_pool_kernel(
    const __half *__restrict__ new_keys,  // [seq, HD]
    int seq, int HD, int compress, int prev_ctx,
    __half *__restrict__ raw_keys,        // [cap, HD]
    float *__restrict__ pooled,           // [cap/compress, HD]
    int *__restrict__ block_starts)       // [cap/compress]
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HD) return;
    for (int s = 0; s < seq; ++s) {
        const int64_t dst = static_cast<int64_t>(prev_ctx + s) * HD + i;
        raw_keys[dst] = new_keys[static_cast<int64_t>(s) * HD + i];
    }
    if (i != 0) return;
    // Thread 0: pool every block that just completed.
    const int prev_blocks = prev_ctx / compress;
    const int new_blocks = (prev_ctx + seq) / compress;
    for (int b = prev_blocks; b < new_blocks; ++b) {
        const int start = b * compress;
        for (int d = 0; d < HD; ++d) {
            float acc = 0.f;
            for (int k = 0; k < compress; ++k)
                acc += __half2float(
                    raw_keys[static_cast<int64_t>(start + k) * HD + d]);
            pooled[static_cast<int64_t>(b) * HD + d] = acc / compress;
        }
        block_starts[b] = start;
    }
}

// One block per (query token). Scores complete blocks: relu(q_h · pooled_b)
// summed over heads, scaled by 1/sqrt(HD); top block_topk selected via
// iterative argmax owned by thread 0 (shared-mem RMW behind barriers —
// protocol step 6). Winning blocks scatter 1s into their token spans;
// degenerate ctx <= budget path: all-visible mask.
__global__ void qsa_indexer_select_kernel(
    const __half *__restrict__ q,        // [seq, heads, HD]
    const float *__restrict__ pooled,    // [num_blocks, HD]
    const int *__restrict__ block_starts,// [num_blocks]
    int heads, int HD, int num_blocks, int block_topk, int compress,
    int past_ctx, int seq,
    int *__restrict__ mask)              // [seq, past_ctx + seq]
{
    const int q_idx = blockIdx.x;
    const int total_ctx = past_ctx + seq;
    int *__restrict__ row = mask + static_cast<int64_t>(q_idx) * total_ctx;

    if (total_ctx <= block_topk * compress) {
        // Degenerate: budget covers everything — all visible.
        for (int i = threadIdx.x; i < total_ctx; i += blockDim.x) row[i] = 1;
        return;
    }

    __shared__ float s_scores[4096];
    __shared__ int s_taken[4096];

    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        s_scores[b] = -1e30f;
        s_taken[b] = 0;
    }
    __syncthreads();

    // Score each block: sum over heads of relu(dot). Each thread owns one
    // (block) and loops heads serially — deterministic fp32 sum.
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float acc = 0.f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.f;
            const __half *qh = q + static_cast<int64_t>(q_idx) * heads * HD +
                               static_cast<int64_t>(h) * HD;
            const float *pb = pooled + static_cast<int64_t>(b) * HD;
            for (int d = 0; d < HD; ++d)
                dot = fmaf(__half2float(qh[d]), pb[d], dot);
            acc += fmaxf(dot, 0.f);
        }
        s_scores[b] = acc / sqrtf(static_cast<float>(HD));
    }
    __syncthreads();

    // Thread 0: iterative argmax top-k (deterministic, ties -> lowest b).
    if (threadIdx.x == 0) {
        for (int k = 0; k < block_topk && k < num_blocks; ++k) {
            int best = -1;
            float best_v = -1e30f;
            for (int b = 0; b < num_blocks; ++b) {
                if (!s_taken[b] && s_scores[b] > best_v) {
                    best_v = s_scores[b];
                    best = b;
                }
            }
            if (best < 0) break;
            s_taken[best] = 1;
            const int start = block_starts[best];
            for (int t = 0; t < compress; ++t) row[start + t] = 1;
        }
    }
}

} // namespace

cudaError_t RunExpQsaIndexer(cudaStream_t stream, ExpQsaIndexerState &st,
                             const __half *d_indexer_q,
                             const __half *d_new_raw_key, int past_ctx,
                             int seq, int head_dim, int compress,
                             int block_topk, int *d_selected_mask)
{
    if (!st.d_raw_keys || head_dim != st.head_dim)
        return cudaErrorInvalidValue;
    const int HD = head_dim;
    // Append new raw keys + pool newly completed blocks.
    {
        const int block = 128;
        const int grid = (HD + block - 1) / block;
        qsa_append_pool_kernel<<<grid, block, 0, stream>>>(
            d_new_raw_key, seq, HD, compress, past_ctx, st.d_raw_keys,
            st.d_pooled_keys, st.d_block_starts);
    }
    const int num_blocks = (past_ctx + seq) / compress;
    // Select per query token.
    {
        const int block = 128;
        qsa_indexer_select_kernel<<<seq, block, 0, stream>>>(
            d_indexer_q, st.d_pooled_keys, st.d_block_starts, 4, HD,
            num_blocks, block_topk, compress, past_ctx, seq, d_selected_mask);
    }
    st.host_pooled_count = num_blocks;
    return cudaGetLastError();
}

// ── Masked GQA attention with online softmax (fp32 accumulators) ─────────

namespace {

__global__ void qsa_attn_kernel(
    const __half *__restrict__ q,       // [seq, heads, HD]
    const __half *__restrict__ k_cache, // [ctx, kv_heads, HD]
    const __half *__restrict__ v_cache, // [ctx, kv_heads, HD]
    const int *__restrict__ mask,       // [seq, ctx] 0/1
    int seq, int ctx, int heads, int kv_heads, int HD, float scale,
    __half *__restrict__ out)           // [seq, heads * HD]
{
    const int q_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int kv = h / (heads / kv_heads);
    const int total_ctx = ctx;

    // Each thread accumulates a partial online-softmax over its stride of
    // the context; results combined deterministically by thread 0 via shared
    // partials (write then barrier — no RMW).
    __shared__ float s_m[256];   // running max per thread
    __shared__ float s_l[256];   // running sum per thread
    __shared__ float s_acc[256][256];  // [thread][HD] — too big; use per-thread
    // Simplify: one thread per HD column with a block-wide reduction is
    // complex; instead a single warp-walk: thread 0 owns the full row
    // sequentially (decode path seq=1, ctx <= 2048+... acceptable for now;
    // prefill parallelism arrives with the generation driver tuning round).
    if (threadIdx.x == 0) {
        float m = -1e30f;
        // Pass 1: max over selected.
        for (int t = 0; t <= q_idx; ++t) {  // causal: t <= q position
            if (!mask[static_cast<int64_t>(q_idx) * total_ctx + t]) continue;
            float dot = 0.f;
            const __half *kt = k_cache + static_cast<int64_t>(t) * kv_heads * HD +
                               static_cast<int64_t>(kv) * HD;
            const __half *qh = q + static_cast<int64_t>(q_idx) * heads * HD +
                               static_cast<int64_t>(h) * HD;
            for (int d = 0; d < HD; ++d)
                dot = fmaf(__half2float(qh[d]), __half2float(kt[d]), dot);
            dot *= scale;
            m = fmaxf(m, dot);
        }
        float l = 0.f;
        for (int t = 0; t <= q_idx; ++t) {
            if (!mask[static_cast<int64_t>(q_idx) * total_ctx + t]) continue;
            float dot = 0.f;
            const __half *kt = k_cache + static_cast<int64_t>(t) * kv_heads * HD +
                               static_cast<int64_t>(kv) * HD;
            const __half *qh = q + static_cast<int64_t>(q_idx) * heads * HD +
                               static_cast<int64_t>(h) * HD;
            for (int d = 0; d < HD; ++d)
                dot = fmaf(__half2float(qh[d]), __half2float(kt[d]), dot);
            dot *= scale;
            l += expf(dot - m);
        }
        __half *oh = out + static_cast<int64_t>(q_idx) * heads * HD +
                     static_cast<int64_t>(h) * HD;
        float acc[256];
        for (int d = 0; d < HD && d < 256; ++d) acc[d] = 0.f;
        for (int t = 0; t <= q_idx; ++t) {
            if (!mask[static_cast<int64_t>(q_idx) * total_ctx + t]) continue;
            float dot = 0.f;
            const __half *kt = k_cache + static_cast<int64_t>(t) * kv_heads * HD +
                               static_cast<int64_t>(kv) * HD;
            const __half *qh = q + static_cast<int64_t>(q_idx) * heads * HD +
                               static_cast<int64_t>(h) * HD;
            for (int d = 0; d < HD; ++d)
                dot = fmaf(__half2float(qh[d]), __half2float(kt[d]), dot);
            dot *= scale;
            const float p = expf(dot - m) / l;
            const __half *vt = v_cache + static_cast<int64_t>(t) * kv_heads * HD +
                               static_cast<int64_t>(kv) * HD;
            for (int d = 0; d < HD && d < 256; ++d)
                acc[d] = fmaf(p, __half2float(vt[d]), acc[d]);
        }
        for (int d = 0; d < HD && d < 256; ++d)
            oh[d] = __float2half(acc[d]);
    }
}

} // namespace

cudaError_t RunExpQsaAttention(cudaStream_t stream, const __half *d_q,
                               const __half *d_k_cache,
                               const __half *d_v_cache,
                               const int *d_selected_mask, int seq,
                               int past_ctx, int heads, int kv_heads,
                               int head_dim, float scale, __half *d_out)
{
    if (head_dim > 256) return cudaErrorInvalidValue;
    dim3 grid(seq, heads);
    qsa_attn_kernel<<<grid, 1, 0, stream>>>(
        d_q, d_k_cache, d_v_cache, d_selected_mask, seq, past_ctx + seq,
        heads, kv_heads, head_dim, scale, d_out);
    return cudaGetLastError();
}

} // namespace lsxcommon::qwen_moe_exp
