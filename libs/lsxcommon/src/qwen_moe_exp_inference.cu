#include "lsxcommon/qwen_moe_exp_inference.h"
#include "lsx_kernels_misc.cuh"
#include "lsx_kernels_common.cuh"

#include "lsxcommon/core.h"

#include "qwen_moe_exp_constants.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/qwen_moe_inference_pipeline.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/cuda_event_wait.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_exp_load.h"  // GenContextResidentBytes declaration
#include "lsxcommon/shared_experts.h"

#include <vector>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include <cmath>

namespace lsxcommon::qwen_moe_exp {
namespace {

#if defined(_MSC_VER)
#define LSX_EXP_NOINLINE __declspec(noinline)
#else
#define LSX_EXP_NOINLINE __attribute__((noinline, noclone))
#endif

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
// W matrices arrive as WeightT (float row-major for the raw twin, __half for
// the dequant-cache path — fp16->double upcast is lossless, so both paths
// produce bit-identical results). Deterministic tree reduction, ascending
// index, no atomics, shared-memory writes by thread 0 only (protocol step 6).

template <typename T>
__device__ __forceinline__ float WToF(T v) { return v; }
template <>
__device__ __forceinline__ float WToF<__half>(__half v) { return __half2float(v); }

template <typename WeightT>
__global__ void hc_gated_residual_f32_kernel(
    const float *__restrict__ x,      // [n]  (seq = 1 for the twin)
    const WeightT *__restrict__ down, // [lowrank * n]
    const WeightT *up,                // [n * lowrank] (g_up may alias it)
    const WeightT *__restrict__ inject, // [hc * n] or null
    int H, int hc, int lowrank, float eps,
    float *__restrict__ mixed_out,    // [H]
    float *__restrict__ inj_out,      // [hc]
    float *g_up,                      // [n] global scratch (aliases up)
    const uint8_t *__restrict__ norm_w,  // BF16 (1+w), [n]
    __half *__restrict__ mixed_h)     // [H] half copy of mixed_out
{
    const int n = H * hc;
    extern __shared__ float hc_smem[];
    float *s_normed = hc_smem;            // [n]
    float *s_dn = s_normed + n;           // [lowrank] (smem tail)
    float *s_up = g_up;                   // [n] global scratch (< 48KB smem)
    __shared__ float s_scale;

    // Phase 1: group-RMSNorm (group = H). Per-group mean of squares reduced
    // by thread 0 after a grid-stride partial store.
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_normed[i] = x[i];
    __syncthreads();

    for (int g = 0; g < hc; ++g) {
        // each group occupies [g*H, g*H+H)
        if (threadIdx.x == 0) {
            float acc = 0.f;
            for (int i = 0; i < H; ++i) {
                const float v = s_normed[g * H + i];
                acc = fmaf(v, v, acc);
            }
            s_scale = rsqrtf(acc / H + eps);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            float v = s_normed[g * H + i] * s_scale;
            if (norm_w) {
                const int idx = g * H + i;
                const uint16_t b16 = static_cast<uint16_t>(norm_w[idx * 2]) |
                                     (static_cast<uint16_t>(norm_w[idx * 2 + 1])
                                      << 8);
                float wf = 1.f;
                if (b16) {
                    const uint32_t u = static_cast<uint32_t>(b16) << 16;
                    const float f = __uint_as_float(u);
                    wf = 1.f + f;
                }
                v *= wf;
            }
            s_normed[g * H + i] = v;
        }
        __syncthreads();
    }

    // Phase 2: down-projection [lowrank] + silu(/hc). Warp-per-row: lane l
    // accumulates elements i = l, l+32, ... (strided, deterministic); ordered
    // shfl-down tree reduce (bit-deterministic for a fixed warp layout).
    const int warps = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int r = warp; r < lowrank; r += warps) {
        float part = 0.f;
        for (int i = lane; i < n; i += 32)
            part = fmaf(WToF(down[r * n + i]), s_normed[i], part);
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, off);
        if (lane == 0) {
            const float v = part / hc;
            s_dn[r] = v / (1.f + expf(-v));
        }
    }
    __syncthreads();

    // Phase 3: up-projection [n] + sigmoid. One thread per element; the
    // lowrank-dim dot runs sequentially per thread (deterministic).
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < lowrank; ++r)
            acc = fmaf(WToF(up[i * lowrank + r]), s_dn[r], acc);
        s_up[i] = 1.0f / (1.0f + expf(-acc));
    }
    __syncthreads();

    // Phase 4: mixed = mean over branches of (gate ⊙ normed).
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int c = 0; c < hc; ++c)
            acc = fmaf(s_up[c * H + h], s_normed[c * H + h], acc);
        mixed_out[h] = acc / hc;
    }

    if (mixed_h) {
        for (int h = threadIdx.x; h < H; h += blockDim.x)
            mixed_h[h] = __float2half(mixed_out[h]);
    }
    // Phase 5: injection weights. Warp-per-c (<= hc warps active).
    if (inject) {
        for (int c = warp; c < hc; c += warps) {
            float part = 0.f;
            for (int i = lane; i < n; i += 32)
                part = fmaf(WToF(inject[c * n + i]), s_normed[i], part);
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0)
                inj_out[c] = 2.0f / (1.0f + expf(-part / hc));
        }
    }
}

// ── Multi-block fast path (decode): the single-block kernel is bandwidth-
// bound to one SM (~9GB/s effective). Splitting the phases into separate
// launches engages all SMs. Bit-determinism preserved: same sequential
// accumulation orders, thread-0 norm reduction, fixed-shape shfl trees.

// grid-stride group-RMSNorm with (1+w): reads x[n] (already upcast), writes
// out[n]. One group of H normalized by block g (thread 0 sequential — the
// bit-exact reduction order). The row is cooperatively staged in shared memory
// first: thread 0's 2560-iteration dependent sum then runs at shared latency
// instead of stalling on global loads (measured 119 -> ~20 us/call), and the
// elementwise normalize/weight/write phase is spread over the block. Sum order
// and per-element arithmetic are unchanged, so the output is bit-identical.
__global__ void hc_norm_kernel(const float *__restrict__ x, int H, int hc,
                               float eps, const uint8_t *__restrict__ norm_w,
                               float *__restrict__ out)
{
    extern __shared__ float sh_x[];
    __shared__ float sh_scale;
    for (int g = blockIdx.x; g < hc; g += gridDim.x) {
        const float *xg = x + static_cast<int64_t>(g) * H;
        for (int i = threadIdx.x; i < H; i += blockDim.x) sh_x[i] = xg[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            float acc = 0.f;
            for (int i = 0; i < H; ++i) {
                const float v = sh_x[i];
                acc = fmaf(v, v, acc);
            }
            sh_scale = rsqrtf(acc / H + eps);
        }
        __syncthreads();
        const float scale = sh_scale;
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            float v = sh_x[i] * scale;
            if (norm_w) {
                const int idx = g * H + i;
                const uint16_t b16 =
                    static_cast<uint16_t>(norm_w[idx * 2]) |
                    (static_cast<uint16_t>(norm_w[idx * 2 + 1]) << 8);
                if (b16) {
                    const uint32_t u = static_cast<uint32_t>(b16) << 16;
                    v *= 1.f + __uint_as_float(u);
                }
            }
            out[g * H + i] = v;
        }
        __syncthreads();
    }
}

// warp-per-row down GEMV: y[lowrank] = W_dn @ x (sequential lane-strided).
__global__ void hc_down_gemv_kernel(const __half *__restrict__ down,
                                    const float *__restrict__ x, int n,
                                    int lowrank, int hc, float *dn)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int r = blockIdx.x * warps + warp; r < lowrank;
         r += gridDim.x * warps) {
        float part = 0.f;
        for (int i = lane; i < n; i += 32)
            part = fmaf(__half2float(down[r * n + i]), x[i], part);
        for (int off = 16; off > 0; off >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, off);
        if (lane == 0) {
            const float v = part / hc;
            dn[r] = v / (1.f + expf(-v));
        }
    }
}

// elementwise up GEMV + sigmoid: up_row[i], accumulate over lowrank.
// Warp-per-output-row with a lane-strided accumulation and a fixed shfl tree
// (deterministic, same reduction class as hc_down_gemv_kernel): the former
// thread-per-row form read up[i*lowrank + r] with a 640-byte lane stride, so
// every warp touched 32 separate cache lines per step (~15 GB/s effective).
__global__ void hc_up_gemv_kernel(const __half *__restrict__ up,
                                  const float *__restrict__ dn, int n,
                                  int lowrank, float *upout)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int i = blockIdx.x * warps + warp; i < n; i += gridDim.x * warps) {
        float acc = 0.f;
        for (int r = lane; r < lowrank; r += 32)
            acc = fmaf(__half2float(up[i * lowrank + r]), dn[r], acc);
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) upout[i] = 1.0f / (1.0f + expf(-acc));
    }
}

// epilogue: mixed[h] = mean_c upout[c*H+h]*normed[c*H+h] (+ mixed_h half copy)
// and inj[c] = 2*sigmoid(W_inj[c,:] . normed / hc) when inject present.
__global__ void hc_epilogue_kernel(const float *__restrict__ upout,
                                   const float *__restrict__ normed,
                                   const __half *__restrict__ inject, int H,
                                   int hc, int n, float *mixed_out,
                                   __half *mixed_h, float *inj_out)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < H;
         h += gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int c = 0; c < hc; ++c)
            acc = fmaf(upout[c * H + h], normed[c * H + h], acc);
        mixed_out[h] = acc / hc;
        if (mixed_h) mixed_h[h] = __float2half(mixed_out[h]);
    }
    if (inject && inj_out) {
        for (int c = warp; c < hc; c += warps) {
            float part = 0.f;
            for (int i = lane; i < n; i += 32)
                part = fmaf(__half2float(inject[c * n + i]), normed[i], part);
            for (int off = 16; off > 0; off >>= 1)
                part += __shfl_down_sync(0xffffffffu, part, off);
            if (lane == 0)
                inj_out[c] = 2.0f / (1.0f + expf(-part / hc));
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
    const size_t smem_bytes = (static_cast<size_t>(n) + lowrank) * sizeof(float);
    if (smem_bytes > 48 * 1024) return cudaErrorInvalidValue;
    const int block = 128;
    hc_gated_residual_f32_kernel<<<1, block, smem_bytes, stream>>>(
        d_hyper_input, d_down, d_up, d_inject_w, H, hc, lowrank, 1e-6f,
        d_mixed, d_inject_w_out, const_cast<float *>(d_up), nullptr,
        nullptr);
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

// ── PLE n-gram inject (decode, single token) ─────────────────────────────
//
// Decomposed composition (parity-first): one block, thread 0 owns the
// sequential fp32 op sequence; NVFP4 weights are dequantized to shared
// scratch via the E2M1/E4M3 tables, then plain fp32 GEMVs run. The parallel
// GEMV variant lands with the generation-driver tuning round; this path
// pins the exact op order the parity suite verifies.

namespace {

} // namespace

void launch_half_to_float(const __half *src, float *dst, int total,
                          cudaStream_t stream)
{
    const int block = 256;
    const int grid = (total + block - 1) / block;
    lsxcommon::kernels::F16ToF32Kernel<<<grid, block, 0, stream>>>(src, dst, total);
}

namespace {

// E2M1 value table shared by the dequant paths.
__device__ __forceinline__ float e2m1_value(uint8_t nib)
{
    static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f,
                                    -4.f, -6.f};
    return kE2M1[nib & 0xF];
}

// Dequantizes one NVFP4 row [cols] into out (fp32), block scale = E4M3 byte
// decoded as 2^((b&0x3F)-1) with sign at 0x80 (converter's packed layout).
struct Nvfp4RowView {
    const uint8_t *d_packed;
    const uint8_t *d_scales;
    float global_scale;
    int cols;
};

__device__ __forceinline__ void dequant_row(Nvfp4RowView w, int row,
                                            float *__restrict__ out)
{
    const uint8_t *packed = w.d_packed + static_cast<int64_t>(row) * (w.cols >> 1);
    const uint8_t *scales = w.d_scales +
                            static_cast<int64_t>(row) * (w.cols >> 4);
    for (int c = 0; c < w.cols; ++c) {
        const uint8_t byte = packed[c >> 1];
        const uint8_t nib = (c & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0xF);
        const float v = e2m1_value(nib);
        if (v == 0.f) {
            out[c] = 0.f;
            continue;
        }
        const uint8_t sb = scales[c >> 4];
        // E4M3FN block-scale byte (converter layout): e4 = sb >> 3, m = sb & 0x7;
        // scale = 2^(e4-7)*(1+m/8) for e4 >= 1, and the e4 == 0 SUBNORMAL
        // m*2^-9. Bit 7 is the reserved sign bit of the unsigned UE4M3 hardware
        // scale and is masked. Matches nvfp4_codec.h::E4M3ScaleToFloat.
        const uint32_t e4 = (sb >> 3) & 0x0Fu;
        const float scale =
            e4 == 0
                ? static_cast<float>(sb & 0x7u) * 0.001953125f
                : __int_as_float(static_cast<int>(((e4 + 120u) << 23) | ((sb & 0x7u) << 20)));
        out[c] = v * scale * w.global_scale;
    }
}

__device__ __forceinline__ float bf16_one_plus(const uint8_t *bits, int idx)
{
    const uint8_t lo = bits[idx * 2];
    const uint8_t hi = bits[idx * 2 + 1];
    const uint16_t b16 = static_cast<uint16_t>(lo) |
                         (static_cast<uint16_t>(hi) << 8);
    if (!b16) return 1.f;
    const uint32_t u = static_cast<uint32_t>(b16) << 16;
    return 1.f + __uint_as_float(u);
}

// Multi-block projection phases (key/value GEMV, one thread per output row).
// Same sequential ascending-c double accumulation as the parity twin, so each
// row is bit-identical to the former single-block loop; splitting across
// blocks lifts the double-throughput ceiling off a single SM (the key GEMV is
// 26.2 M DP FMAs — the dominant decode cost when confined to one block).
__global__ void ple_row_proj_kernel(Nvfp4RowView w,
                                    const float *__restrict__ d_emb_row,
                                    int rows, float *__restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const uint8_t *packed =
        w.d_packed + static_cast<int64_t>(r) * (w.cols >> 1);
    const uint8_t *scales =
        w.d_scales + static_cast<int64_t>(r) * (w.cols >> 4);
    double acc = 0.0;
    for (int c = 0; c < w.cols; ++c) {
        const uint8_t byte = packed[c >> 1];
        const uint8_t nib = (c & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0xF);
        const float v = e2m1_value(nib);
        float dv = 0.f;
        if (v != 0.f) {
            const uint8_t sb = scales[c >> 4];
            const uint32_t e4 = (sb >> 3) & 0x0Fu;
            const float scale =
                e4 == 0
                    ? static_cast<float>(sb & 0x7u) * 0.001953125f
                    : __int_as_float(static_cast<int>(((e4 + 120u) << 23) | ((sb & 0x7u) << 20)));
            dv = v * scale * w.global_scale;
        }
        acc += static_cast<double>(dv) * d_emb_row[c];
    }
    out[r] = static_cast<float>(acc);
}

// One block (1024 threads) per token. Parallel decomposition of the
// bit-exact op sequence (decode path): the projection GEMVs run as separate
// multi-block kernels (ple_row_proj_kernel above) so the DP-bound work is not
// capped by a single SM; norm reductions stay on thread 0, elementwise
// scales/scatters parallel across threads with __syncthreads() barriers
// between phases (protocol step 6: no multi-writer RMW races).
__global__ void ple_inject_kernel(
    Nvfp4RowView key_proj,     // [hc*H, E]
    Nvfp4RowView value_proj,   // [H, E]
    const uint8_t *__restrict__ d_norm_key,   // BF16 bits [hc*H]
    const uint8_t *__restrict__ d_norm_query, // BF16 bits [hc*H]
    const uint8_t *__restrict__ d_norm_conv,  // BF16 bits [hc*H]
    const float *__restrict__ d_conv_w,       // [hc*H, k] fp32
    const float *__restrict__ d_emb_row,      // [heads*head_dim] fp32
    __half *__restrict__ d_hidden,            // [hc*H] in/out
    float *__restrict__ d_window,             // [state_len, hc*H]
    float *__restrict__ d_scratch,            // [4*n + H + E] global workspace
    int state_len, int H, int hc, int k_conv, int dilation, float eps)
{
    (void)key_proj;    // projections run as multi-block ple_row_proj_kernel
    (void)value_proj;
    (void)d_emb_row;
    const int n = H * hc;
    float *s_key = d_scratch;          // [n]
    float *s_query = s_key + n;        // [n]
    float *s_gated = s_query + n;      // [n]
    float *s_normed = s_gated + n;     // [n]
    float *s_value = s_normed + n;     // [H]
    float *s_row = s_value + H;        // [E] dequant scratch (legacy layout)
    (void)s_row;
    __shared__ float sh_scale[8];      // per-group norm scale
    __shared__ float sh_sg[8];         // sigmoid(gate) per stream

    // 1. key = key_proj @ emb — precomputed by ple_row_proj_kernel (multi-
    //    block, bit-identical per-row accumulation) into s_key.
    __syncthreads();

    // 1b. group-RMS(key), then (1 + norm_key) — thread-0 reduction (the
    //     ascending-i sum order is part of the bit-exact contract), parallel
    //     elementwise scale.
    for (int g = 0; g < hc; ++g) {
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                const float v = s_key[g * H + i];
                acc += v * v;
            }
            sh_scale[g] = static_cast<float>(1.0 / sqrt(acc / H + eps));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            s_key[g * H + i] = s_key[g * H + i] * sh_scale[g] *
                               bf16_one_plus(d_norm_key, g * H + i);
        __syncthreads();
    }

    // 2. query = group-RMS(hidden), then (1 + norm_query_w).
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s_query[i] = __half2float(d_hidden[i]);
    __syncthreads();
    for (int g = 0; g < hc; ++g) {
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                const float v = s_query[g * H + i];
                acc += v * v;
            }
            sh_scale[g] = static_cast<float>(1.0 / sqrt(acc / H + eps));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            s_query[g * H + i] = s_query[g * H + i] * sh_scale[g] *
                                 bf16_one_plus(d_norm_query, g * H + i);
        __syncthreads();
    }

    // 3. gate per stream: s = sum_i key_i * query_i / sqrt(H) (thread 0 —
    //    hc*H double MACs, tiny).
    const double sqrtH = sqrt(static_cast<double>(H));
    if (threadIdx.x == 0) {
        for (int c = 0; c < hc && c < 8; ++c) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i)
                acc += static_cast<double>(s_key[c * H + i]) *
                       s_query[c * H + i];
            const double s = acc / sqrtH;
            const double a = fmax(1e-6, fabs(s));
            const double gate = sqrt(a) * (s < 0 ? -1.0 : 1.0);
            sh_sg[c] = static_cast<float>(1.0 / (1.0 + exp(-gate)));
        }
    }
    __syncthreads();

    // 4. value = value_proj @ emb — precomputed by the value projection
    //    launch into s_value (bit-identical per-row accumulation).
    __syncthreads();

    // 5. gated = sigmoid(gate) * value, broadcast to streams.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int c = i / H;
        s_gated[i] = sh_sg[c] * s_value[i % H];
    }
    __syncthreads();

    // 6. normed = group-RMS(gated), then (1 + norm_conv_w).
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s_normed[i] = s_gated[i];
    __syncthreads();
    for (int g = 0; g < hc; ++g) {
        if (threadIdx.x == 0) {
            double acc = 0.0;
            for (int i = 0; i < H; ++i) {
                const float v = s_normed[g * H + i];
                acc += v * v;
            }
            sh_scale[g] = static_cast<float>(1.0 / sqrt(acc / H + eps));
        }
        __syncthreads();
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            s_normed[g * H + i] = s_normed[g * H + i] * sh_scale[g] *
                                  bf16_one_plus(d_norm_conv, g * H + i);
        __syncthreads();
    }

    // 7. HF conv1d semantics (decode, seq=1): x_pad = [prev(state_len), current].
    //    Tap k of the K-tap kernel reads x_pad[state_len - (K-1-k)*dilation]:
    //    w0 -> the oldest reachable row (skipped when negative), w{K-1} -> the
    //    CURRENT row. Taps are computed BEFORE the current row is appended.
    //    One thread per (c,i); the k-loop stays sequential per element.
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const int c = idx / H;
        const int i = idx % H;
        float conv_acc = 0.f;
        for (int k = 0; k < k_conv; ++k) {
            const int off = state_len - (k_conv - 1 - k) * dilation;
            if (off == state_len) {
                conv_acc = fmaf(d_conv_w[idx * k_conv + k], s_normed[idx],
                                conv_acc);
            } else if (off >= 0 && off < state_len) {
                conv_acc = fmaf(d_conv_w[idx * k_conv + k],
                                d_window[static_cast<int64_t>(off) * n + idx],
                                conv_acc);
            }
        }
        const float out_v = s_gated[idx] + conv_acc / (1.0f + expf(-conv_acc));
        d_hidden[idx] = __float2half(__half2float(d_hidden[idx]) + out_v);
    }

    // 8. Shift the window: drop oldest, append the current normed row.
    for (int s = 0; s < state_len - 1; ++s) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            d_window[static_cast<int64_t>(s) * n + i] =
                d_window[static_cast<int64_t>(s + 1) * n + i];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        d_window[static_cast<int64_t>(state_len - 1) * n + i] = s_normed[i];

}

} // namespace

cudaError_t ExpPleStateAlloc(ExpPleState &st, int H, int hc, int state_len)
{
    st = ExpPleState{};
    st.state_len = state_len;
    st.H = H;
    st.hc = hc;
    const size_t bytes =
        static_cast<size_t>(state_len) * H * hc * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_window), bytes) !=
        cudaSuccess)
        return cudaGetLastError();
    // Kernel workspace: key[3*H] wait — key[n] + query[n] + gated[n] + normed[n]
    // + value[H] + row[E] live in global memory (shared-memory opt-in limit
    // on sm_120 is 99KB/block — far below the 5*hc*H-float footprint).
    const size_t scratch_floats =
        4u * static_cast<size_t>(H) * hc + H + 2560 + 32;
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_scratch),
                   scratch_floats * sizeof(float)) != cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(reinterpret_cast<void **>(&st.d_ctx),
                   static_cast<size_t>(state_len) * sizeof(int32_t)) !=
        cudaSuccess)
        return cudaGetLastError();
    cudaMemset(st.d_window, 0, bytes);
    cudaMemset(st.d_scratch, 0, scratch_floats * sizeof(float));
    cudaMemset(st.d_ctx, 0, static_cast<size_t>(state_len) * sizeof(int32_t));
    return cudaSuccess;
}

void ExpPleStateFree(ExpPleState &st)
{
    if (st.d_window) cudaFree(st.d_window);
    if (st.d_scratch) cudaFree(st.d_scratch);
    if (st.d_ctx) cudaFree(st.d_ctx);
    st = ExpPleState{};
}

static void LaunchPleInjectKernel(const Nvfp4RowView &kv,
                                  const Nvfp4RowView &vv,
                                  const uint8_t *nk, const uint8_t *nq,
                                  const uint8_t *nc, const float *cw,
                                  const float *emb, __half *hid,
                                  float *win, float *scratch, int state_len,
                                  int H, int hc, int k_conv, int dilation,
                                  float eps, cudaStream_t stream)
{
    // cudaLaunchKernel (not <<<>>>): the triple-chevron path relies on
    // __cudaPushCallConfiguration, which silently no-ops under some LTO
    // cloning splits of this multi-exit wrapper.
    Nvfp4RowView kv_a = kv, vv_a = vv;
    const uint8_t *nk_a = nk, *nq_a = nq, *nc_a = nc;
    const float *cw_a = cw, *emb_a = emb;
    __half *hid_a = hid;
    float *win_a = win, *scratch_a = scratch;
    int state_len_a = state_len, H_a = H, hc_a = hc, k_conv_a = k_conv,
        dilation_a = dilation;
    float eps_a = eps;

    // Projection phases first (multi-block): key -> s_key (scratch[0]),
    // value -> s_value (scratch[4*n]). Disjoint outputs; same stream orders
    // them before ple_inject_kernel consumes them.
    const int n = H * hc;
    const int proj_block = 256;
    ple_row_proj_kernel<<<(n + proj_block - 1) / proj_block, proj_block, 0,
                          stream>>>(kv, emb, n, scratch);
    ple_row_proj_kernel<<<(H + proj_block - 1) / proj_block, proj_block, 0,
                          stream>>>(vv, emb, H, scratch + 4 * n);

    void *args[] = {&kv_a, &vv_a, &nk_a, &nq_a, &nc_a, &cw_a, &emb_a,
                    &hid_a, &win_a, &scratch_a, &state_len_a, &H_a, &hc_a,
                    &k_conv_a, &dilation_a, &eps_a};
    cudaLaunchKernel(reinterpret_cast<const void *>(&ple_inject_kernel),
                     dim3(1), dim3(1024), args, 0, stream);
}

// Persistent fp32 conversion scratch for the PLE conv weights (avoids a
// cudaMalloc/cudaFree pair per decode step inside RunExpPleInject).
struct PLEConvScratch {
    float *d_w = nullptr;
    size_t bytes = 0;
    ~PLEConvScratch()
    {
        cudaFreeIgnoreShutdown(d_w);
    }
};

PLEConvScratch &PLEConvScratchInstance()
{
    // Per-host-thread: two model instances live in one process (the engine
    // allows several models; the residency lock is per-model), so a
    // process-global scratch would let one decode's Ensure/free race another
    // thread's in-flight kernels. Same contract as GreedyScratch (sampling.cu).
    static thread_local PLEConvScratch s;
    return s;
}

__global__ void bf16_bits_to_float_kernel(const uint8_t *__restrict__ bits,
                                          float *__restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t b16 = static_cast<uint16_t>(bits[i * 2]) |
                         (static_cast<uint16_t>(bits[i * 2 + 1]) << 8);
    const uint32_t u = static_cast<uint32_t>(b16) << 16;
    out[i] = __uint_as_float(u);
}

LSX_EXP_NOINLINE cudaError_t RunExpPleInject(cudaStream_t stream, ExpPleState &st,
                            const lsx_nvfp4::LsxNvfp4Weight &key_proj,
                            const lsx_nvfp4::LsxNvfp4Weight &value_proj,
                            const uint8_t *d_norm_key,
                            const uint8_t *d_norm_query,
                            const uint8_t *d_norm_conv,
                            const uint8_t *d_ple_conv_bf16,
                            const int32_t *d_token, const float *d_emb_row,
                            int heads, int head_dim, float eps,
                            __half *d_hidden)
{
    (void)heads;
    (void)head_dim;
    (void)d_token;
    const int H = st.H;
    const int hc = st.hc;
    const int n = H * hc;
    // The conv weights arrive as raw BF16 bits [n, k]; convert to fp32 once
    // into a persistent scratch (per-step cudaMalloc/cudaFree of this buffer
    // cost measurable host time in the decode loop).
    float *d_conv_w = nullptr;
    {
        PLEConvScratch &cs = PLEConvScratchInstance();
        const size_t need = static_cast<size_t>(n) * 4 * sizeof(float);
        if (cs.bytes < need) {
            if (cs.d_w) cudaFree(cs.d_w);
            if (cudaMalloc(reinterpret_cast<void **>(&cs.d_w), need) !=
                cudaSuccess) {
                cs.d_w = nullptr;
                cs.bytes = 0;
                return cudaGetLastError();
            }
            cs.bytes = need;
        }
        d_conv_w = cs.d_w;
        const int total = n * 4;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        bf16_bits_to_float_kernel<<<grid, block, 0, stream>>>(
            d_ple_conv_bf16, d_conv_w, total);
        cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) return ce;
    }
    LaunchPleInjectKernel(
        Nvfp4RowView{key_proj.d_packed, key_proj.d_scales,
                     key_proj.global_scale, key_proj.cols},
        Nvfp4RowView{value_proj.d_packed, value_proj.d_scales,
                     value_proj.global_scale, value_proj.cols},
        d_norm_key, d_norm_query, d_norm_conv,
        d_conv_w, d_emb_row, d_hidden, st.d_window, st.d_scratch,
        st.state_len, H, hc, 4, 3, eps, stream);
    return cudaGetLastError();
}

cudaError_t RunExpRouterTop10(cudaStream_t stream, const uint8_t *d_router_bf16,
                              const __half *d_x, int seq, int H, int E, int K,
                              moe::MoERoutedSlot *d_slots)
{
    return moe::RunMoERouterTopK(
        stream, reinterpret_cast<const __nv_bfloat16 *>(d_router_bf16), d_x,
        seq, E, K, H, d_slots);
}

// ── NVFP4 production wrappers: dequant hc weights to device scratch, then
// run the identical F32 sequence. ──────────────────────────────────────────

namespace {

// Scratch manager: per-process dequant workspace for the hc weights.
struct HcScratch {
    __half *h_down = nullptr, *h_up = nullptr, *h_inject = nullptr;
    float *d_down = nullptr, *d_up = nullptr, *d_inject = nullptr;
    float *d_x32 = nullptr;   // fp32 input upcast
    float *d_upout = nullptr; // fp32 up-gate output [n]
    float *d_normed = nullptr; // fp32 normed [n] (multi-block fast path)
    float *d_dn = nullptr;     // fp32 down output [lowrank]
    float *d_mixer_mixed = nullptr; // mixer epilogue scratch [H]
    float *d_mixer_inj = nullptr;   // mixer inject scratch [hc] (unused out)
    size_t down_bytes = 0, up_bytes = 0, x_bytes = 0, upout_bytes = 0,
           inject_bytes = 0, normed_bytes = 0, dn_bytes = 0, mixer_bytes = 0;
    ~HcScratch()
    {
        // Tolerates cudaErrorCudartUnloading at process exit (see
        // cudaFreeIgnoreShutdown).
        cudaFreeIgnoreShutdown(h_down);
        cudaFreeIgnoreShutdown(h_up);
        cudaFreeIgnoreShutdown(h_inject);
        cudaFreeIgnoreShutdown(d_down);
        cudaFreeIgnoreShutdown(d_up);
        cudaFreeIgnoreShutdown(d_inject);
        cudaFreeIgnoreShutdown(d_x32);
        cudaFreeIgnoreShutdown(d_upout);
        cudaFreeIgnoreShutdown(d_normed);
        cudaFreeIgnoreShutdown(d_dn);
        cudaFreeIgnoreShutdown(d_mixer_mixed);
        // d_mixer_inj aliases d_mixer_mixed + offset (no separate allocation);
        // freeing it would pass an interior pointer to cudaFree.
    }
};

HcScratch &HcScratchInstance() {
    // Per-host-thread (see PLEConvScratchInstance): a process-global scratch
    // would be shared by concurrent same-model instances, so one thread's
    // EnsureScratchScratchOnly could cudaFree a buffer another thread's decode
    // loop is still reading.
    static thread_local HcScratch s;
    return s;
}

// Grow-only scratch allocators. A reallocation frees the previous buffers,
// which a prior call's kernels on `stream` may still be consuming; the caller
// passes its generation stream and the reclaim path syncs it before the free
// (steady state never reallocates, so the hot path is unaffected). Clearing the
// alias before the realloc keeps a failed cudaMalloc from leaving a dangling
// interior pointer.
cudaError_t EnsureScratch(const lsx_nvfp4::LsxNvfp4Weight &down,
                          const lsx_nvfp4::LsxNvfp4Weight &up, int x32_floats,
                          cudaStream_t stream)
{
    HcScratch &s = HcScratchInstance();
    const auto reclaim = [stream] {
        if (stream) (void)cudaStreamSynchronize(stream);
    };
    const size_t db = static_cast<size_t>(down.rows) * down.cols;
    const size_t ub = static_cast<size_t>(up.rows) * up.cols;
    if (s.down_bytes < db * sizeof(__half)) {
        reclaim();
        if (s.h_down) cudaFree(s.h_down);
        if (cudaMalloc(&s.h_down, db * sizeof(__half)) != cudaSuccess)
            return cudaGetLastError();
        if (cudaMalloc(&s.d_down, db * sizeof(float)) != cudaSuccess)
            return cudaGetLastError();
        s.down_bytes = db * sizeof(__half);
    }
    if (s.up_bytes < ub * sizeof(__half)) {
        reclaim();
        if (s.h_up) cudaFree(s.h_up);
        if (cudaMalloc(&s.h_up, ub * sizeof(__half)) != cudaSuccess)
            return cudaGetLastError();
        if (cudaMalloc(&s.d_up, ub * sizeof(float)) != cudaSuccess)
            return cudaGetLastError();
        s.up_bytes = ub * sizeof(__half);
    }
    const size_t need_x = static_cast<size_t>(x32_floats) * sizeof(float);
    if (s.x_bytes < need_x) {
        reclaim();
        if (s.d_x32) cudaFree(s.d_x32);
        if (cudaMalloc(&s.d_x32, need_x) != cudaSuccess)
            return cudaGetLastError();
        s.x_bytes = need_x;
    }
    if (s.upout_bytes < need_x) {
        reclaim();
        if (s.d_upout) cudaFree(s.d_upout);
        if (cudaMalloc(&s.d_upout, need_x) != cudaSuccess)
            return cudaGetLastError();
        s.upout_bytes = need_x;
    }
    return cudaSuccess;
}

cudaError_t EnsureScratchScratchOnly(int x32_floats, int lowrank,
                                     cudaStream_t stream)
{
    HcScratch &s = HcScratchInstance();
    const auto reclaim = [stream] {
        if (stream) (void)cudaStreamSynchronize(stream);
    };
    const size_t need_x = static_cast<size_t>(x32_floats) * sizeof(float);
    if (s.x_bytes < need_x) {
        reclaim();
        if (s.d_x32) cudaFree(s.d_x32);
        if (cudaMalloc(&s.d_x32, need_x) != cudaSuccess)
            return cudaGetLastError();
        s.x_bytes = need_x;
    }
    if (s.upout_bytes < need_x) {
        reclaim();
        if (s.d_upout) cudaFree(s.d_upout);
        if (cudaMalloc(&s.d_upout, need_x) != cudaSuccess)
            return cudaGetLastError();
        s.upout_bytes = need_x;
    }
    if (s.normed_bytes < need_x) {
        reclaim();
        if (s.d_normed) cudaFree(s.d_normed);
        if (cudaMalloc(&s.d_normed, need_x) != cudaSuccess)
            return cudaGetLastError();
        s.normed_bytes = need_x;
    }
    const size_t need_dn = static_cast<size_t>(lowrank) * sizeof(float);
    if (s.dn_bytes < need_dn) {
        reclaim();
        if (s.d_dn) cudaFree(s.d_dn);
        if (cudaMalloc(&s.d_dn, need_dn) != cudaSuccess)
            return cudaGetLastError();
        s.dn_bytes = need_dn;
    }
    // mixed[H] then inj[hc] in one allocation; x32_floats == H*hc >= H covers
    // the mixed part and kHcCount floats size the inject tail. Both are plain
    // float buffers (scalar access in hc_epilogue_kernel), so no alignment
    // padding is needed between them. The `hc` passed by the mixer entry is
    // checked against frozen::kHcCount there, so the two stay equal.
    const size_t need_mixer =
        (static_cast<size_t>(x32_floats) + frozen::kHcCount) * sizeof(float);
    if (s.mixer_bytes < need_mixer) {
        reclaim();
        // d_mixer_inj aliases d_mixer_mixed + offset: free only the real
        // allocation, and clear the alias so a failed realloc below cannot
        // leave a dangling interior pointer for the destructor to free.
        if (s.d_mixer_mixed) cudaFree(s.d_mixer_mixed);
        s.d_mixer_mixed = nullptr;
        s.d_mixer_inj = nullptr;
        if (cudaMalloc(&s.d_mixer_mixed, need_mixer) != cudaSuccess)
            return cudaGetLastError();
        s.d_mixer_inj = s.d_mixer_mixed + x32_floats;
        s.mixer_bytes = need_mixer;
    }
    return cudaSuccess;
}

cudaError_t EnsureInjectScratch(const lsx_nvfp4::LsxNvfp4Weight &inject_w,
                                cudaStream_t stream)
{
    HcScratch &s = HcScratchInstance();
    const size_t ib =
        static_cast<size_t>(inject_w.rows) * inject_w.cols * sizeof(__half);
    if (s.inject_bytes < ib) {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (s.h_inject) cudaFree(s.h_inject);
        if (cudaMalloc(&s.h_inject, ib) != cudaSuccess)
            return cudaGetLastError();
        if (cudaMalloc(&s.d_inject, ib * 2) != cudaSuccess)
            return cudaGetLastError();
        s.inject_bytes = ib;
    }
    return cudaSuccess;
}

} // namespace

void ExpHcDequantCache::Free()
{
    if (down) { cudaFree(down); down = nullptr; }
    if (up) { cudaFree(up); up = nullptr; }
    if (inject) { cudaFree(inject); inject = nullptr; }
    down_elems = up_elems = inject_elems = 0;
}

cudaError_t BuildExpHcDequantCache(cudaStream_t stream,
                                   const lsx_nvfp4::LsxNvfp4Weight &down,
                                   const lsx_nvfp4::LsxNvfp4Weight &up,
                                   const lsx_nvfp4::LsxNvfp4Weight &inject_w,
                                   ExpHcDequantCache &out)
{
    out.Free();
    const int64_t de = static_cast<int64_t>(down.rows) * down.cols;
    const int64_t ue = static_cast<int64_t>(up.rows) * up.cols;
    const bool have_inject = inject_w.d_packed && inject_w.rows > 0 &&
                             inject_w.cols > 0;
    const int64_t ie =
        have_inject ? static_cast<int64_t>(inject_w.rows) * inject_w.cols : 0;
    if (de <= 0 || ue <= 0) return cudaErrorInvalidValue;
    if (cudaMalloc(&out.down, static_cast<size_t>(de) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&out.up, static_cast<size_t>(ue) * sizeof(__half)) !=
        cudaSuccess) {
        cudaError_t le = cudaGetLastError();
        out.Free();
        return le;
    }
    if (have_inject) {
        if (cudaMalloc(&out.inject,
                       static_cast<size_t>(ie) * sizeof(__half)) !=
            cudaSuccess) {
            cudaError_t le = cudaGetLastError();
            out.Free();
            return le;
        }
    }
    // Same dequant kernel as the legacy per-call path -> bit-identical fp16
    // values; the hc kernel consumes them losslessly (fp16->double).
    if (gpu::dequant_shared_nvfp4(stream, down.d_packed, down.packed_bytes,
                                  down.d_scales, down.scale_bytes,
                                  down.global_scale, out.down, de) !=
        cudaSuccess) {
        cudaError_t le = cudaGetLastError();
        out.Free();
        return le;
    }
    if (gpu::dequant_shared_nvfp4(stream, up.d_packed, up.packed_bytes,
                                  up.d_scales, up.scale_bytes,
                                  up.global_scale, out.up, ue) !=
        cudaSuccess) {
        cudaError_t le = cudaGetLastError();
        out.Free();
        return le;
    }
    if (have_inject) {
        if (gpu::dequant_shared_nvfp4(
                stream, inject_w.d_packed, inject_w.packed_bytes,
                inject_w.d_scales, inject_w.scale_bytes,
                inject_w.global_scale, out.inject, ie) != cudaSuccess) {
            cudaError_t le = cudaGetLastError();
            out.Free();
            return le;
        }
    }
    out.down_elems = de;
    out.up_elems = ue;
    out.inject_elems = ie;
    return cudaSuccess;
}

__global__ void half_to_float_rows_kernel(const __half *__restrict__ src,
                                          float *__restrict__ dst, int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __half2float(src[i]);
}

cudaError_t RunExpHcGatedResidual(
    cudaStream_t stream, const __half *d_hyper_input,
    const lsx_nvfp4::LsxNvfp4Weight &down, const lsx_nvfp4::LsxNvfp4Weight &up,
    const lsx_nvfp4::LsxNvfp4Weight &inject_w, const uint8_t *d_hc_norm,
    int seq, int H, int hc, int lowrank, float *d_mixed, float *d_inject_out,
    __half *d_mixed_h, const uint8_t *d_hc_norm_override)
{
    if (seq != 1) return cudaErrorInvalidValue;
    const uint8_t *norm_w =
        d_hc_norm_override ? d_hc_norm_override : d_hc_norm;
    const int n = H * hc;
    // Shared footprint: 2n + lowrank floats; guard the Blackwell smem cap.
    const size_t smem_floats = 2 * static_cast<size_t>(n) + lowrank;
    if (smem_floats * sizeof(float) > 227 * 1024)
        return cudaErrorInvalidValue;
    HcScratch &s = HcScratchInstance();
    cudaError_t e = EnsureScratch(down, up, n, stream);
    if (e != cudaSuccess) return e;
    e = EnsureInjectScratch(inject_w, stream);
    if (e != cudaSuccess) return e;
    if (gpu::dequant_shared_nvfp4(stream, down.d_packed, down.packed_bytes,
                                  down.d_scales, down.scale_bytes,
                                  down.global_scale, s.h_down,
                                  static_cast<int64_t>(down.rows) * down.cols) !=
        cudaSuccess) {
        return cudaGetLastError();
    }
    if (gpu::dequant_shared_nvfp4(stream, up.d_packed, up.packed_bytes,
                                  up.d_scales, up.scale_bytes,
                                  up.global_scale, s.h_up,
                                  static_cast<int64_t>(up.rows) * up.cols) !=
        cudaSuccess)
        return cudaGetLastError();
    if (inject_w.d_packed && inject_w.rows > 0) {
        if (gpu::dequant_shared_nvfp4(
                stream, inject_w.d_packed, inject_w.packed_bytes,
                inject_w.d_scales, inject_w.scale_bytes,
                inject_w.global_scale, s.h_inject,
                static_cast<int64_t>(inject_w.rows) * inject_w.cols) !=
            cudaSuccess)
            return cudaGetLastError();
        half_to_float_rows_kernel<<<(inject_w.rows * inject_w.cols + 255) /
                                        256,
                                    256, 0, stream>>>(
            s.h_inject, s.d_inject, inject_w.rows * inject_w.cols);
    }
    half_to_float_rows_kernel<<<(down.rows * down.cols + 255) / 256, 256, 0,
                                stream>>>(s.h_down, s.d_down,
                                          down.rows * down.cols);
    half_to_float_rows_kernel<<<(up.rows * up.cols + 255) / 256, 256, 0,
                                stream>>>(s.h_up, s.d_up, up.rows * up.cols);
    // Upcast the fp16 hyper input to the fp32 scratch buffer.
    if (static_cast<size_t>(n) * sizeof(float) > s.x_bytes) {
        return cudaErrorInvalidValue;
    }
    half_to_float_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_hyper_input, s.d_x32, n);
    const int block = 1024;
    const size_t smem_bytes = (static_cast<size_t>(n) + lowrank) * sizeof(float);
    hc_gated_residual_f32_kernel<<<1, block, smem_bytes, stream>>>(
        s.d_x32, s.d_down, s.d_up, s.d_inject, H, hc, lowrank, 1e-6f,
        d_mixed, d_inject_out, s.d_upout, norm_w, d_mixed_h);
    return cudaGetLastError();
}

cudaError_t RunExpHcGatedResidualCached(
    cudaStream_t stream, const __half *d_hyper_input,
    const ExpHcDequantCache &w, const uint8_t *d_hc_norm, int seq, int H,
    int hc, int lowrank, float *d_mixed, float *d_inject_out,
    __half *d_mixed_h)
{
    if (seq != 1) return cudaErrorInvalidValue;
    const int n = H * hc;
    if (!w.down || !w.up || w.down_elems < static_cast<int64_t>(lowrank) * n ||
        w.up_elems < static_cast<int64_t>(n) * lowrank)
        return cudaErrorInvalidValue;
    // Multi-block fast path: separate launches per phase engage all SMs
    // (the single-block kernel is bandwidth-bound to one SM). Bit-exact
    // reduction orders are preserved in each kernel.
    HcScratch &s = HcScratchInstance();
    cudaError_t e = EnsureScratchScratchOnly(n, lowrank, stream);
    if (e != cudaSuccess) return e;
    half_to_float_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        d_hyper_input, s.d_x32, n);
    hc_norm_kernel<<<hc, 128, static_cast<size_t>(H) * sizeof(float),
                     stream>>>(s.d_x32, H, hc, 1e-6f, d_hc_norm, s.d_normed);
    hc_down_gemv_kernel<<<(lowrank + 7) / 8, 256, 0, stream>>>(
        w.down, s.d_normed, n, lowrank, hc, s.d_dn);
    // Warp-per-row: grid covers n rows with 8 warps per 256-thread block.
    hc_up_gemv_kernel<<<(n + 7) / 8, 256, 0, stream>>>(
        w.up, s.d_dn, n, lowrank, s.d_upout);
    hc_epilogue_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        s.d_upout, s.d_normed, w.inject, H, hc, n, d_mixed, d_mixed_h,
        d_inject_out);
    return cudaGetLastError();
}

cudaError_t RunExpHcMixer(cudaStream_t stream, const __half *d_hidden,
                          const lsx_nvfp4::LsxNvfp4Weight &down,
                          const lsx_nvfp4::LsxNvfp4Weight &up,
                          const uint8_t *d_mixer_norm, int seq, int H, int hc,
                          int lowrank, __half *d_out, __half *d_normed_scratch)
{
    (void)d_normed_scratch;
    if (seq != 1) return cudaErrorInvalidValue;
    float *d_mixed = nullptr, *d_inj = nullptr;
    if (cudaMalloc(&d_mixed, sizeof(float) * H) != cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&d_inj, sizeof(float) * hc) != cudaSuccess) {
        cudaFree(d_mixed);
        return cudaGetLastError();
    }
    cudaError_t e = RunExpHcGatedResidual(
        stream, d_hidden, down, up,
        lsx_nvfp4::LsxNvfp4Weight{}, nullptr, seq, H, hc, lowrank, d_mixed, d_inj,
        nullptr, d_mixer_norm);
    if (e == cudaSuccess) {
        // Float->half conversion for the mixed output.
        lsxcommon::kernels::F32ToF16Kernel<<<(H + 255) / 256, 256, 0, stream>>>(d_mixed,
                                                                  d_out, H);
        e = cudaGetLastError();
    }
    cudaFree(d_mixed);
    cudaFree(d_inj);
    return e;
}

cudaError_t RunExpHcMixerCached(cudaStream_t stream, const __half *d_hidden,
                                const ExpHcDequantCache &w,
                                const uint8_t *d_mixer_norm, int seq, int H,
                                int hc, int lowrank, __half *d_out)
{
    if (seq != 1) return cudaErrorInvalidValue;
    // EnsureScratchScratchOnly sizes the mixer's inject tail from the frozen
    // passport (frozen::kHcCount), so a runtime `hc` that disagrees with it
    // would run against a mismatched scratch. hc_count is cross-checked
    // against the passport at model load; assert the same contract here so the
    // entry is safe on its own.
    if (hc != frozen::kHcCount) return cudaErrorInvalidValue;
    // Mixer output rides persistent scratch (per-call cudaMalloc/cudaFree
    // showed up as measurable host stalls in the decode loop); no inject.
    HcScratch &s = HcScratchInstance();
    cudaError_t e = EnsureScratchScratchOnly(H * hc, lowrank, stream);
    if (e != cudaSuccess) return e;
    float *d_mixed = s.d_mixer_mixed, *d_inj = s.d_mixer_inj;
    if (!d_mixed || !d_inj) return cudaErrorInvalidValue;
    e = RunExpHcGatedResidualCached(stream, d_hidden, w, d_mixer_norm, seq, H,
                                    hc, lowrank, d_mixed, d_inj, nullptr);
    if (e == cudaSuccess) {
        lsxcommon::kernels::F32ToF16Kernel<<<(H + 255) / 256, 256, 0, stream>>>(d_mixed,
                                                                  d_out, H);
        e = cudaGetLastError();
    }
    return e;
}

// ── Generation driver (Task 8) ────────────────────────────────────────────
//
// Incremental single-stream decode for the 48-layer qwen4_exp stack:
//   embed (persistent fp16 table) -> repeat x4 -> per layer
//     [PLE(1)] -> hc -> GDN|QSA -> hc-scatter -> hc -> MoE(top-10) ->
//     hc-scatter -> global mixer -> final norm -> lm_head.
// KV caches + GDN recurrent states are reused from the qwen primitives via
// shallow views (lsxcommon::attn::LayerWeights mirrors); the qwen4_exp-specific
// hc/PLE/QSA stages run before/after the reused attention cores.

namespace {

// Device-side token record/feed (etalon qwen_record_token_kernel).
__global__ void exp_record_token_kernel(int32_t *__restrict__ hist,
                                        int idx, const int32_t *__restrict__ v)
{
    hist[idx] = *v;
}

__global__ void exp_feed_token_kernel(int32_t *__restrict__ slot,
                                      const int32_t *__restrict__ v)
{
    *slot = *v;
}

// hidden [n] (fp16) -> repeat x hc streams (fp32 identity replication).

// hyper stream mean: out[i] = mean_c hyper[c*H+i]; folds the hc mixer
// epilogue when the caller has already applied the input-mix gate (the
// global mixer with use_combine=false is exactly GatedResidual's mixed
// branch).
__global__ void exp_mean_streams_kernel(const __half *__restrict__ hyper,
                                        int H, int hc, __half *__restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float acc = 0.f;
    for (int c = 0; c < hc; ++c) acc += __half2float(hyper[c * H + i]);
    out[i] = __float2half(acc / hc);
}

// Process-lifetime device scratch owned by the generation driver: the fp16
// embed table and the max_seq-scaled arena (fp32 logits dominate it). Both are
// deliberately shared across model instances ("single session per process",
// etalon simplification), which is why they are namespace-scope rather than
// per-instance state — but that also means they survive a model Unload, so the
// expert-cache sizing has to know they exist: it sizes from free VRAM, and a
// second Load in one process would otherwise reserve the same context twice and
// starve the cache (measured: 5350 -> 2202 slots in the perf harness's r3 leg).
struct ExpGenScratch {
    __half *d_embed_h = nullptr;
    int64_t embed_elems = 0;
    void *arena = nullptr;
    size_t arena_cap = 0;
};

ExpGenScratch &ExpGenScratchInstance()
{
    static ExpGenScratch s;
    return s;
}

} // namespace

std::size_t GenContextResidentBytes()
{
    const ExpGenScratch &s = ExpGenScratchInstance();
    // The arena is held at its grown capacity and only ever grows by the delta
    // (it is freed before the larger cudaMalloc), and the embed table is exact
    // for the frozen geometry, so both are fully reusable.
    return s.arena_cap + static_cast<std::size_t>(s.embed_elems) * sizeof(__half);
}

cudaError_t RunExpGeneration(
    cudaStream_t stream, const qwen_moe_exp::ExpPinnedWeights &pw,
    qwen_moe_exp::ExpExpertStore &store, gpu::ExpertDeviceCache &cache,
    qwen_moe_exp::ExpPleState &ple,
    std::vector<qwen_moe_exp::ExpQsaIndexerState> &qsa,
    const std::vector<qwen_moe_exp::ExpPleShardView> &ple_shards,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    qwen::GenerationResult &result, lsxcommon::sample::ISampler &sampler,
    qwen::KvCache *kv,
    const lsxcommon::GenerationStream *gs)
{
    using namespace lsxcommon::qwen_moe_exp;
    if (!kv) return cudaErrorInvalidValue;
    const qwen::QwenTopology &top = pw.top;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads, HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    const int V = top.vocab_size;
    const int E = pw.exp_top.num_experts;
    const int K = pw.exp_top.num_experts_per_tok;
    const int I = pw.exp_top.moe_intermediate_size;
    const int SI = pw.exp_top.shared_expert_intermediate_size;
    const int hc = pw.exp_top.hc_count;
    const int HCH = H * hc;
    const int rotary_dim =
        static_cast<int>(HD * top.partial_rotary_factor);
    result = qwen::GenerationResult{};

    if (H <= 0 || V <= 0 || prompt.empty() || max_new <= 0 || eos_id < 0 ||
        E <= 0 || K <= 0 || I <= 0 || SI <= 0 || !pw.embed.d_packed ||
        !pw.d_head_f16) {
        result.error_msg = "invalid qwen4_exp generation inputs";
        return cudaErrorInvalidValue;
    }
    const int max_seq = static_cast<int>(prompt.size()) + max_new;

    {
        const attn::KvGeometry kv_geom = qwen::MakeKvGeometry(top);
        cudaError_t ce = kv->allocated()
                             ? kv->EnsureCapacity(kv_geom, kv->kv_rows + max_new,
                                                  stream)
                             : kv->Allocate(kv_geom, max_seq, stream);
        if (ce != cudaSuccess) {
            result.error_msg = "kv cache alloc failed";
            return ce;
        }
    }

    // Persistent fp16 embed table (dequant once per process).
    ExpGenScratch &scratch = ExpGenScratchInstance();
    __half *&d_embed_h = scratch.d_embed_h;
    int64_t &embed_elems = scratch.embed_elems;
    if (!d_embed_h || embed_elems < static_cast<int64_t>(V) * H) {
        if (d_embed_h) cudaFree(d_embed_h);
        const int64_t bytes = static_cast<int64_t>(V) * H * sizeof(__half);
        if (cudaMalloc(&d_embed_h, static_cast<size_t>(bytes)) !=
            cudaSuccess)
            return cudaGetLastError();
        embed_elems = static_cast<int64_t>(V) * H;
        if (gpu::dequant_shared_nvfp4(stream, pw.embed.d_packed,
                                      pw.embed.packed_bytes,
                                      pw.embed.d_scales, pw.embed.scale_bytes,
                                      pw.embed.global_scale, d_embed_h,
                                      embed_elems) != cudaSuccess)
            return cudaGetLastError();
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return cudaGetLastError();
    }

    // Arena (process-lifetime — single session per process, etalon
    // simplification; see ExpGenScratch for the sizing consequence).
    void *&arena = scratch.arena;
    size_t &arena_cap = scratch.arena_cap;
    size_t half_elems = 0;
    half_elems += (size_t)max_seq * H;             // d_act (fp16 stream)
    half_elems += (size_t)max_seq * HCH;           // d_hyper
    half_elems += (size_t)max_seq * HCH;           // d_hyper2
    half_elems += (size_t)max_seq * H;             // d_mixed
    half_elems += (size_t)max_seq * NH * HD * 2;   // d_qc
    half_elems += (size_t)max_seq * NH * HD;       // d_q
    half_elems += (size_t)max_seq * NH * HD;       // d_qgate
    half_elems += (size_t)max_seq * NKV * HD;      // d_k
    half_elems += (size_t)max_seq * NKV * HD;      // d_v
    half_elems += (size_t)max_seq * NH * HD;       // d_attn
    half_elems += (size_t)max_seq * rotary_dim;    // d_cos
    half_elems += (size_t)max_seq * rotary_dim;    // d_sin
    half_elems += (size_t)max_seq * S;             // d_qkv
    half_elems += (size_t)max_seq * VD;            // d_z
    half_elems += (size_t)max_seq * nv * 2;        // d_ab
    half_elems += (size_t)max_seq * S;             // d_conv
    half_elems += (size_t)max_seq * VD;            // d_core
    half_elems += (size_t)max_seq * VD;            // d_gated
    half_elems += (size_t)max_seq * HCH;           // d_normed_scratch
    half_elems += (size_t)K * 2 * I;               // d_ygu_batch
    half_elems += (size_t)K * I;                   // d_h_batch
    half_elems += (size_t)K * H;                   // d_yp_batch
    half_elems += (size_t)2 * I;                   // d_yg (fused w13: gate|up)
    half_elems += (size_t)I;                       // d_yu
    half_elems += (size_t)H;                       // d_y2
    half_elems += (size_t)SI;                      // d_si (shared scratch)
    half_elems += (size_t)H;                       // d_sh
    size_t float_elems = (size_t)max_seq * nv * 2; // d_gf/betaf
    float_elems += (size_t)max_seq * H;            // d_moe_acc
    float_elems += (size_t)max_seq * H;            // d_mixed_f (hc mixed out)
    float_elems += (size_t)max_seq * 16;           // hc inject weights
    float_elems += (size_t)max_seq * 2560;         // PLE gathered emb rows
    float_elems += 1024;                           // gate/argmax scratch
    float_elems += (size_t)max_seq * V;            // fp32 logits
    float_elems += (size_t)max_seq * V;            // fp32 logits
    // Batched TC routed-expert chain scratch (streaming decode): pre-built
    // device slot table + one-shot layer activation quant + per-slot swiglu
    // quant outputs + w13 split-K partials. Mirrors RunQwenMoeGeneration.
    const size_t xq_bytes = ((static_cast<size_t>(H) / 2 + 63) / 64) * 64;
    const size_t xqs_bytes =
        (((static_cast<size_t>(H) + 15) / 16) + 63) / 64 * 64;
    const size_t qpack_b_bytes = ((static_cast<size_t>(K) * I / 2) + 63) / 64 * 64;
    const size_t qscales_b_bytes =
        ((static_cast<size_t>(K) * ((I + 15) / 16)) + 63) / 64 * 64;
    const size_t gemv_tbl_bytes =
        ((sizeof(moe::MoEGvSlotDev) * static_cast<size_t>(K)) + 63) /
        64 * 64;
    const int tc_splitk = lsx_tc::LsxTcBatchSplitK(H, 2 * I);
    const size_t tc_splitk_bytes =
        tc_splitk > 1
            ? ((static_cast<size_t>(tc_splitk) * K * 2 * I * 4 + 63) / 64) * 64
            : 64;
    // Hoisted per-layer SwiGLU prequant: K*I FP32 products + 2*K alpha slots.
    const size_t sw_batch_bytes =
        ((static_cast<size_t>(K) * I + 2 * static_cast<size_t>(K)) *
             sizeof(float) +
         63) / 64 * 64;
    const size_t extra_bytes = xq_bytes + xqs_bytes + qpack_b_bytes +
                               qscales_b_bytes + gemv_tbl_bytes +
                               tc_splitk_bytes + sw_batch_bytes;
    const size_t half_bytes =
        ((half_elems * sizeof(__half)) + 255) & ~static_cast<size_t>(255);
    const size_t float_bytes =
        (float_elems * sizeof(float) + 255) & ~static_cast<size_t>(255);
    const size_t need = half_bytes + float_bytes + extra_bytes + 4096;
    if (arena_cap < need) {
        if (arena) cudaFree(arena);
        if (cudaMalloc(&arena, need) != cudaSuccess) {
            arena = nullptr;
            arena_cap = 0;
            result.error_msg = "arena alloc failed";
            return cudaGetLastError();
        }
        arena_cap = need;
    }
    void *base = arena;
    __half *ph = reinterpret_cast<__half *>(base);
    auto adv = [&](__half *&p, size_t elems) {
        __half *r = p;
        p += ((elems + 7) / 8) * 8;
        return r;
    };
    __half *d_act = adv(ph, (size_t)max_seq * H);
    __half *d_hyper = adv(ph, (size_t)max_seq * HCH);
    __half *d_hyper2 = adv(ph, (size_t)max_seq * HCH);
    __half *d_mixed = adv(ph, (size_t)max_seq * H);
    __half *d_qc = adv(ph, (size_t)max_seq * NH * HD * 2);
    __half *d_q = adv(ph, (size_t)max_seq * NH * HD);
    __half *d_qgate = adv(ph, (size_t)max_seq * NH * HD);
    __half *d_k = adv(ph, (size_t)max_seq * NKV * HD);
    __half *d_v = adv(ph, (size_t)max_seq * NKV * HD);
    __half *d_attn = adv(ph, (size_t)max_seq * NH * HD);
    __half *d_cos = adv(ph, (size_t)max_seq * rotary_dim);
    __half *d_sin = adv(ph, (size_t)max_seq * rotary_dim);
    __half *d_qkv = adv(ph, (size_t)max_seq * S);
    __half *d_z = adv(ph, (size_t)max_seq * VD);
    __half *d_ab = adv(ph, (size_t)max_seq * nv * 2);
    __half *d_conv = adv(ph, (size_t)max_seq * S);
    __half *d_core = adv(ph, (size_t)max_seq * VD);
    __half *d_gated = adv(ph, (size_t)max_seq * VD);
    // d_logits now fp32 (head GEMM accumulates in fp32); carve from the float arena.

    __half *d_normed_scr = adv(ph, (size_t)max_seq * HCH);
    __half *d_ygu_batch = adv(ph, (size_t)K * 2 * I);
    __half *d_h_batch = adv(ph, (size_t)K * I);
    __half *d_yp_batch = adv(ph, (size_t)K * H);
    __half *d_yg = adv(ph, (size_t)2 * I);
    __half *d_yu = adv(ph, (size_t)I);
    __half *d_y2 = adv(ph, (size_t)H);
    __half *d_si = adv(ph, (size_t)SI);
    __half *d_sh = adv(ph, (size_t)H);
    float *d_gf = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(base) + half_bytes);
    float *d_betaf = d_gf + (size_t)max_seq * nv;
    float *d_moe_acc = d_betaf + (size_t)max_seq * nv;
    float *d_mixed_f = d_moe_acc + (size_t)max_seq * H;
    float *d_inj_f = d_mixed_f + (size_t)max_seq * H;
    float *d_gate_val = d_inj_f + (size_t)max_seq * 16;
    float *d_maxv_scratch = d_gate_val + 16;
    // PLE gathered embedding rows: [heads*head_dim] = 2560 floats. Must NOT
    // alias d_gate_val/d_maxv_scratch (they are written by MoE shared-combine
    // and the sampler between the gather and the PLE kernel at layer 1).
    float *d_emb_rows = d_maxv_scratch + 16;
    float *d_logits_f32 = d_emb_rows + (size_t)V;
    uint8_t *extra = reinterpret_cast<uint8_t *>(base) + half_bytes + float_bytes;
    uint8_t *d_xq = extra;
    uint8_t *d_xqs = d_xq + xq_bytes;
    uint8_t *d_qpack_b = d_xqs + xqs_bytes;
    uint8_t *d_qscales_b = d_qpack_b + qpack_b_bytes;
    moe::MoEGvSlotDev *d_gemv_tbl =
        reinterpret_cast<moe::MoEGvSlotDev *>(d_qscales_b +
                                                       qscales_b_bytes);
    float *d_tc_splitk = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(d_gemv_tbl) + gemv_tbl_bytes);
    float *d_sw_batch = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(d_tc_splitk) + tc_splitk_bytes);
    float *d_alpha_batch = d_sw_batch + static_cast<size_t>(K) * I;

    int32_t *d_ids = nullptr, *d_pos = nullptr, *d_argmax = nullptr,
            *d_tokens = nullptr;
    moe::MoERoutedSlot *d_slots = nullptr;
    cudaError_t e = cudaMalloc(&d_ids, sizeof(int32_t) * 4);
    if (e == cudaSuccess) e = cudaMalloc(&d_pos, sizeof(int32_t) * 4);
    if (e == cudaSuccess) e = cudaMalloc(&d_argmax, sizeof(int32_t));
    if (e == cudaSuccess)
        e = cudaMalloc(&d_tokens, sizeof(int32_t) * (size_t)max_new);
    if (e == cudaSuccess)
        e = cudaMalloc(&d_slots, sizeof(moe::MoERoutedSlot) * (size_t)K);
    if (e != cudaSuccess) {
        result.error_msg = "device scalar alloc failed";
        cudaFree(d_ids); cudaFree(d_pos); cudaFree(d_argmax);
        cudaFree(d_tokens); cudaFree(d_slots);
        return e;
    }

    // Host mirrors for hc inject weights + expert slot staging.
    std::vector<float> h_inj(hc);
    int32_t *h_ids = nullptr;
    bool have_pinned = cudaMallocHost(&h_ids, sizeof(int32_t) * 4) ==
                       cudaSuccess;
    // Persistent pinned PLE gather buffer: a pageable std::vector + the
    // cudaStreamSynchronize that kept it alive forced a full pipeline flush
    // every token. Pinned + persistent keeps the H2D truly stream-ordered.
    float *h_emb_rows = nullptr;
    constexpr int kEmbRowsCap = 32 * 256;  // ExpPleMeta::kMaxHeads * head_dim
    cudaMallocHost(reinterpret_cast<void **>(&h_emb_rows),
                   sizeof(float) * kEmbRowsCap);
    cudaEvent_t tok_ready_ev = nullptr;
    cudaEventCreate(&tok_ready_ev);

    // Copy channel for cache-miss expert uploads: H2D staging overlaps the
    // routed-expert compute chain instead of serializing with it (etalon:
    // RunQwenMoeGeneration). Events fence each upload; ev_front guards slot
    // reuse against still-running prior compute.
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ev_uploads[16] = {};
    cudaEvent_t ev_front = nullptr;
    bool have_copy =
        lsxcommon::lsx_nvfp4::LsxDualStreamStaging() &&
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) ==
            cudaSuccess &&
        cudaEventCreateWithFlags(&ev_front, cudaEventDisableTiming) ==
            cudaSuccess;
    for (int k = 0; have_copy && k < 16; ++k)
        if (cudaEventCreateWithFlags(&ev_uploads[k],
                                     cudaEventDisableTiming) != cudaSuccess)
        {
            have_copy = false;
            break;
        }
    if (!have_copy) {
        // Graceful degradation: staging falls back to the compute stream.
        if (copy_stream) cudaStreamDestroy(copy_stream);
        copy_stream = nullptr;
        if (ev_front) cudaEventDestroy(ev_front);
        ev_front = nullptr;
        for (int k = 0; k < 16; ++k) {
            if (ev_uploads[k]) cudaEventDestroy(ev_uploads[k]);
            ev_uploads[k] = nullptr;
        }
    }

    // Staged-overlap depth ceiling for the routed TC chain: the K slots run in
    // up to this many ascending groups, so group c+1's cache-miss H2D staging
    // runs on the copy stream while group c's TC work executes. The depth is
    // per layer, from its miss count (see the chain call): a chunk's compute can
    // only hide another chunk's copy, so a layer with fewer than two misses
    // gains nothing and pays the extra launch. Fixed, not a knob — the numbers
    // are in docs/open-tasks/2026-11-04-qwen38-expert-staging.md.
    constexpr int kMoeStageChunks = 2;

    int full_idx = 0, lin_idx = 0, qsa_idx = 0;
    // GPU-timeline timing (etalon: qwen_moe driver t_start/t_stop events).
    cudaEvent_t t_start_ev = nullptr, t_ttft_ev = nullptr, t_stop_ev = nullptr;
    cudaEventCreate(&t_start_ev);
    cudaEventCreate(&t_ttft_ev);
    cudaEventCreate(&t_stop_ev);
    if (t_start_ev) cudaEventRecord(t_start_ev, stream);
    // t_ttft_ev is recorded only when the first token's forward completes, so a
    // request cancelled before any token leaves it unrecorded. Calling
    // cudaEventElapsedTime on an unrecorded event sets a STICKY
    // cudaErrorInvalidResourceHandle that the next request's first checked
    // launch then reports as its own failure (an aborted run poisoning the
    // following one). Track the record explicitly and skip the read when it
    // never happened.
    bool ttft_recorded = false;
    std::vector<int32_t> token_history;  // PLE n-gram context (grows)
    const int prompt_steps = static_cast<int>(prompt.size());

    const int total_steps = prompt_steps + max_new - 1;

    int32_t cur = prompt[0];
    int generated = 0;
    const int threads = 256;
    const double eps = top.rms_norm_eps;

    // Reset the persistent PLE conv window and QSA indexer state: the states
    // live in model-owned buffers that survive across generation calls (the
    // selfcheck runs several cycles on the same model object) — stale
    // context would leak into the next run and break run-to-run parity.
    cudaMemsetAsync(ple.d_window, 0,
                    static_cast<size_t>(ple.state_len) * H * hc * sizeof(float),
                    stream);
    cudaMemsetAsync(ple.d_ctx, 0,
                    static_cast<size_t>(ple.state_len) * sizeof(int32_t),
                    stream);
    for (auto &st : qsa) {
        if (!st.d_raw_keys) continue;
        const size_t hd = static_cast<size_t>(st.head_dim > 0 ? st.head_dim
                                                              : 128);
        cudaMemsetAsync(st.d_raw_keys, 0,
                        static_cast<size_t>(st.capacity_ctx) * hd *
                            sizeof(__half),
                        stream);
        cudaMemsetAsync(st.d_pooled_keys, 0,
                        static_cast<size_t>(st.capacity_ctx) * hd *
                            sizeof(float),
                        stream);
        cudaMemsetAsync(st.d_block_starts, 0,
                        static_cast<size_t>(st.capacity_ctx) * sizeof(int),
                        stream);
        st.host_pooled_count = 0;
    }

    for (int step = 0; step < total_steps; ++step) {
        // Abort contract (shared lifecycle contract, ModelBase): every
        // generation loop polls the flag per step, so a cancel stops the
        // request instead of running it to max_gen_tokens. This driver used
        // to emit tokens but never poll, so `req.abort_flag` was accepted and
        // silently ignored - the one family whose abort path was dead while
        // the lifecycle guard stayed green, because the guard accepted the
        // `abort_flag = ` assignment as proof. The bulk drain below then
        // published a full-length answer for a cancelled request.
        if (gs && gs->Aborted()) break;
        const int kv_base = kv->kv_rows;
        full_idx = 0;
        lin_idx = 0;
        if (h_ids) {
            h_ids[0] = cur;
            h_ids[1] = kv_base;
            cudaMemcpyAsync(d_ids, h_ids, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_pos, h_ids + 1, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos, &kv_base, sizeof(int32_t), cudaMemcpyHostToDevice);
        }
        e = qwen::RunAttnEmbed(stream, d_embed_h, d_ids, d_act, 1, H);
        if (e != cudaSuccess) { result.error_msg = "embed"; break; }
        // hidden -> 4 streams.
        lsxcommon::kernels::RepeatHcKernel<><<<(H + threads - 1) / threads, threads, 0, stream>>>(
            d_act, H, hc, d_hyper);

        // PLE n-gram context: previous tokens (eos-padded), host mirror.
        token_history.push_back(cur);
        if (token_history.size() > 64) token_history.erase(token_history.begin());
        if (!ple_shards.empty() && pw.ple_meta.num_heads > 0) {
            // HF Qwen4ExpTextNGramEmbedding.forward decode: shifted tokens
            // [0..ngram_size-1] over the token history (eos-padded context),
            // hashed with splitmix64 multipliers, per-head prime mod.
            const auto &meta = pw.ple_meta;
            const int ctx_len = meta.context_len;              // 2
            const int64_t eos = lsxcommon::qwen_moe_exp::frozen::kEosSecondary;
            auto tok_at = [&](int back) {
                const int64_t total = static_cast<int64_t>(token_history.size());
                const int64_t idx = total - 1 - back;
                return idx >= 0 ? token_history[static_cast<size_t>(idx)] : eos;
            };
            const int32_t t0 = tok_at(0), t1 = tok_at(1), t2 = tok_at(2);
            const int heads_per = meta.num_heads / ctx_len;    // 8
            // head_dim 160; each head contributes head_dim floats; gather
            // heads*head_dim floats total into d_emb_rows.
            const int emb_count = meta.num_heads * meta.head_dim;
            std::vector<float> emb_rows_fallback;
            float *emb_rows = h_emb_rows;
            if (!emb_rows || emb_count > kEmbRowsCap) {
                emb_rows_fallback.resize(emb_count);
                emb_rows = emb_rows_fallback.data();
            }
            for (int h = 0; h < meta.num_heads; ++h) {
                int64_t mixed;
                if (h < heads_per) {
                    // bigram heads: shifted[0]*m0 xor shifted[1]*m1
                    mixed = (static_cast<int64_t>(t0) * meta.multipliers[0]) ^
                            (static_cast<int64_t>(t1) * meta.multipliers[1]);
                } else {
                    // trigram heads
                    mixed = (static_cast<int64_t>(t0) * meta.multipliers[0]) ^
                            (static_cast<int64_t>(t1) * meta.multipliers[1]) ^
                            (static_cast<int64_t>(t2) * meta.multipliers[2]);
                }
                const int64_t idx = (mixed % meta.head_vocab_sizes[h] +
                                     meta.head_vocab_sizes[h]) %
                                        meta.head_vocab_sizes[h];
                const int64_t flat = meta.head_offsets[h] + idx;
                // shard s holds contiguous rows of the padded vocab:
                // rows_per_shard = padded_vocab / split_parts
                const int rows_per = static_cast<int>(
                    ple_shards.empty() ? 0 : 1);
                (void)rows_per;
                // Locate the shard containing this flat row (round-robin by
                // the converter's shard partitioning: shard = flat % parts
                // would be wrong; the converter stored shards sequentially —
                // row r of shard s corresponds to flat = s*rows_per + r).
                // HF splits the padded vocab column-wise across
                // split_ngram_parts shards: shard s holds rows [s*rows_per,
                // (s+1)*rows_per).
                const int64_t padded = meta.padded_vocab;
                const int64_t rows_per_shard = padded / 128;
                const int shard = static_cast<int>(flat / rows_per_shard);
                const int64_t row = flat % rows_per_shard;
                if (shard < static_cast<int>(ple_shards.size())) {
                    const uint8_t *base = ple_shards[static_cast<size_t>(shard)]
                                              .data;
                    // Each shard file embeds one BF16 [rows, head_dim] tensor
                    // inside the Arrow container; the view data points at the
                    // record payload.
                    const int64_t row_bytes =
                        static_cast<int64_t>(meta.head_dim) * 2;
                    const uint8_t *src = base + row * row_bytes;
                    // BF16 -> float into the tail of the gather buffer.
                    float *dst = emb_rows +
                                 static_cast<int64_t>(h) * meta.head_dim;
                    for (int d = 0; d < meta.head_dim; ++d) {
                        const uint16_t b16 = static_cast<uint16_t>(src[d * 2]) |
                                             (static_cast<uint16_t>(src[d * 2 + 1])
                                              << 8);
                        uint32_t u = static_cast<uint32_t>(b16) << 16;
                        float f;
                        memcpy(&f, &u, 4);
                        dst[d] = f;
                    }
                }
            }
            // The model stream is created with cudaStreamNonBlocking: this
            // H2D MUST be enqueued on the model stream (a legacy-stream copy
            // does not order against it — the PLE kernel at layer 1 would
            // otherwise read the un-gathered arena). From the persistent
            // pinned buffer no completion sync is needed; the fallback path
            // (pageable) keeps the old flush.
            cudaMemcpyAsync(d_emb_rows, emb_rows,
                            emb_count * sizeof(float),
                            cudaMemcpyHostToDevice, stream);
            if (emb_rows == emb_rows_fallback.data())
                cudaStreamSynchronize(stream);
        }

        for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
            const qwen_moe_exp::ExpLayerWeights &lw = pw.layers[l];
            // ── PLE (layer 1) ──
            if (lw.is_ple_layer) {
                e = RunExpPleInject(stream, ple, lw.ple_key, lw.ple_value,
                                    lw.d_ple_norm_key, lw.d_ple_norm_query,
                                    lw.d_ple_norm_conv, lw.d_ple_conv_bf16,
                                    d_ids, d_emb_rows, 16, 160, (float)eps,
                                    d_hyper);
                if (e != cudaSuccess) { result.error_msg = "ple"; break; }
            }
            // ── attn hyper-connection: mixed + inject weights ──
            if (lw.attn_hc_cache) {
                e = RunExpHcGatedResidualCached(
                    stream, d_hyper, *lw.attn_hc_cache, lw.d_attn_hc_norm, 1,
                    H, hc, pw.exp_top.hc_lowrank, d_mixed_f, d_inj_f,
                    d_mixed);
            } else {
                e = RunExpHcGatedResidual(stream, d_hyper, lw.attn_hc_down,
                                          lw.attn_hc_up, lw.attn_hc_inject,
                                          lw.d_attn_hc_norm, 1, H, hc,
                                          pw.exp_top.hc_lowrank, d_mixed_f,
                                          d_inj_f, d_mixed);
            }
            if (e != cudaSuccess) { result.error_msg = "hc-attn"; break; }
            if (lw.attn.is_linear) {
                e = qwen::RunAttnFusedLinearAttnKernel(
                    stream, lw.attn, qwen::MakeAttnGeometry(top), 1, d_mixed, d_act,
                    kv->d_qkv + (int64_t)lin_idx * kv->max_total * S,
                    d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                    kv_base,
                    kv->d_col + (int64_t)lin_idx * nv * vd * kd,
                    pw.exp_top.output_gate_type == "sigmoid");
                if (e != cudaSuccess)
                    result.error_msg = "gdn l" + std::to_string(l) + " lin" +
                                       std::to_string(lin_idx) + " kb" +
                                       std::to_string(kv_base) + " mt" +
                                       std::to_string(kv->max_total) + ": " +
                                       cudaGetErrorString(e);
                ++lin_idx;
            } else {
                // QSA layer: projection path via the reused qwen attention
                // core with full causal mask for the fixture/selfcheck
                // horizon (indexer budget covers ctx <= 2048 — degenerate
                // fast path in RunExpQsaIndexer).
                e = qwen::RunAttnFullAttention(
                    stream, lw.attn, qwen::MakeAttnGeometry(top), 1, d_mixed, d_act, d_qc, d_q,
                    d_qgate, kv->d_kv_k +
                                 (int64_t)full_idx * kv->max_total * NKV * HD,
                    kv->d_kv_v +
                        (int64_t)full_idx * kv->max_total * NKV * HD,
                    d_attn, d_cos, d_sin, d_pos, kv_base);
                if (e != cudaSuccess)
                    result.error_msg = "qsa l" + std::to_string(l) + ": " +
                                       cudaGetErrorString(e);
                ++full_idx;
                ++qsa_idx;
            }
            if (e != cudaSuccess) break;
            // hyper + branch*w scatter (w from device d_inj_f).
            e = RunExpHcInjectScatter(stream, d_hyper, d_act, d_inj_f, 1, H,
                                      hc, d_hyper2);
            if (e != cudaSuccess) { result.error_msg = "hc-scatter1"; break; }
            // ── mlp hyper-connection ──
            if (lw.mlp_hc_cache) {
                e = RunExpHcGatedResidualCached(
                    stream, d_hyper2, *lw.mlp_hc_cache, lw.d_mlp_hc_norm, 1,
                    H, hc, pw.exp_top.hc_lowrank, d_mixed_f, d_inj_f,
                    d_mixed);
            } else {
                e = RunExpHcGatedResidual(stream, d_hyper2, lw.mlp_hc_down,
                                          lw.mlp_hc_up, lw.mlp_hc_inject,
                                          lw.d_mlp_hc_norm, 1, H, hc,
                                          pw.exp_top.hc_lowrank, d_mixed_f,
                                          d_inj_f, d_mixed);
            }
            if (e != cudaSuccess) { result.error_msg = "hc-mlp"; break; }
            // ── MoE: router -> routed experts -> shared combine ──
            e = RunExpRouterTop10(stream, lw.d_router, d_mixed, 1, H, E, K,
                                  d_slots);
            if (e != cudaSuccess) { result.error_msg = "router"; break; }
            // Host readback of the selected slots. Busy-poll the completion
            // event instead of the blocking cudaEventSynchronize: at 48
            // handshakes per token the host wake latency is pure GPU idle
            // time (qwen_moe driver rule). Residency staging uses the
            // fused-store fast path (2 contiguous blobs per miss).
            moe::MoERoutedSlot h_slots[16];
            int dev_slots[16] = {0};
            bool missed[16] = {};
            cudaMemcpyAsync(h_slots, d_slots, sizeof(moe::MoERoutedSlot) * K,
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready_ev, stream);
            {
                const cudaError_t q =
                    lsxcommon::cuda::WaitEventChecked(tok_ready_ev);
                if (q != cudaSuccess) {
                    result.error_msg = "router handshake failed";
                    return q;
                }
            }
            // Cache-miss staging rides the copy engine: nsys over the
            // 2048-token protocol showed the H2D uploads fully serialized in
            // front of the layer's kernels, because residency for all K slots
            // was staged on the compute stream before any FFN compute.
            // ev_front fences slot reuse (a recycled slot may still have a
            // reader queued on `stream`); ev_uploads[k] fences the individual
            // upload so the consumer — the chunked TC chain, or the fused
            // fallback below — waits only for the slots it is about to read.
            const bool stage_on_copy = copy_stream != nullptr;
            if (stage_on_copy) {
                cudaError_t fe = cudaEventRecord(ev_front, stream);
                if (fe == cudaSuccess)
                    fe = cudaStreamWaitEvent(copy_stream, ev_front, 0);
                if (fe != cudaSuccess) {
                    result.error_msg = "expert stage fence";
                    e = fe;
                    break;
                }
            }
            int n_miss = 0;  // this layer's cache misses, drives the stage depth
            for (int k = 0; k < K && k < 16; ++k) {
                if (h_slots[k].eid < 0 || h_slots[k].eid >= E) {
                    result.error_msg = "router returned out-of-range expert";
                    e = cudaErrorInvalidValue;
                    break;
                }
                std::string ensure_err;
                missed[k] = false;
                const int slot = EnsureQwen4ExpExpertResident(
                    store, l, h_slots[k].eid, cache, stream, ensure_err,
                    stage_on_copy ? copy_stream : nullptr,
                    stage_on_copy ? &missed[k] : nullptr);
                if (slot < 0) {
                    result.error_msg = "ensure l" + std::to_string(l) +
                                       " k" + std::to_string(k) + " slot" +
                                       std::to_string(slot) + ": " +
                                       ensure_err;
                    e = cudaGetLastError();
                    if (e == cudaSuccess) e = cudaErrorUnknown;
                    break;
                }
                dev_slots[k] = slot;
                if (stage_on_copy && missed[k]) {
                    ++n_miss;
                    const cudaError_t ce =
                        cudaEventRecord(ev_uploads[k], copy_stream);
                    if (ce != cudaSuccess) {
                        result.error_msg = "expert upload event l" +
                                           std::to_string(l) + " k" +
                                           std::to_string(k);
                        e = ce;
                        break;
                    }
                }
            }
            if (e != cudaSuccess) break;
            // Non-chunked consumers (the fused per-slot fallback) need every
            // upload resident before their first read.
            auto fence_uploads = [&]() -> cudaError_t {
                if (!stage_on_copy) return cudaSuccess;
                for (int k = 0; k < K && k < 16; ++k) {
                    if (!missed[k]) continue;
                    const cudaError_t w =
                        cudaStreamWaitEvent(stream, ev_uploads[k], 0);
                    if (w != cudaSuccess) return w;
                }
                return cudaSuccess;
            };
            std::string ffn_err;
            // Routed accumulation runs acc += coeff * y2 per expert; the fp32
            // accumulator must be zeroed per layer (the arena is persistent).
            cudaMemsetAsync(d_moe_acc, 0, sizeof(float) * H, stream);
            const moe::MoERoutedTcScratch tc_scratch{
                d_qpack_b, d_qscales_b, d_sw_batch, d_alpha_batch};
            const uint8_t *shared_xq = nullptr, *shared_xqs = nullptr;
            static const bool no_ffn_sbatch =
                std::getenv("LSX_EXP_FFN_NOSBATCH") != nullptr;
            const bool tc_batch_ok =
                qwen::QwenTcFastMath() && (I % 16) == 0 && !no_ffn_sbatch;
            if (tc_batch_ok) {
                // Batched TC routed chain (streaming handshake): the host
                // already holds dev_slot + segment scales, so upload the slot
                // table directly and replace ~4 launches per expert with 4
                // batch kernels + one layer activation quant.
                moe::MoEGvSlotDev tbl[16];
                for (int k = 0; k < K; ++k) {
                    const auto &rec =
                        store.records[static_cast<size_t>(l)]
                                     [static_cast<size_t>(h_slots[k].eid)];
                    tbl[k].dev_slot = dev_slots[k];
                    tbl[k].gs1 = rec.gs1;
                    tbl[k].gs3 = rec.gs3;
                    tbl[k].gs2 = rec.gs2;
                    tbl[k].coeff = h_slots[k].coeff;
                }
                const auto &rec0 = store.records[static_cast<size_t>(l)][0];
                const int64_t w13_b =
                    static_cast<int64_t>(rec0.w1_bytes + rec0.w3_bytes);
                const int64_t s13_b =
                    static_cast<int64_t>(rec0.s1_bytes + rec0.s3_bytes);
                cudaMemcpyAsync(d_gemv_tbl, tbl,
                                sizeof(moe::MoEGvSlotDev) * K,
                                cudaMemcpyHostToDevice, stream);
                e = lsx_tc::LsxTcQuantizeActsNvfp4(stream, d_mixed, 1, H, d_xq,
                                                   d_xqs);
                if (e == cudaSuccess) {
                    shared_xq = d_xq;
                    shared_xqs = d_xqs;
                    e = moe::RunMoERoutedTcBatchStreamed(
                        stream, stage_on_copy ? copy_stream : nullptr,
                        stage_on_copy ? ev_uploads : nullptr,
                        stage_on_copy ? missed : nullptr, cache, d_gemv_tbl,
                        d_xq, d_xqs, K, H, I, w13_b, s13_b, d_moe_acc,
                        d_ygu_batch, d_h_batch, d_yp_batch, d_qpack_b,
                        d_qscales_b, tc_splitk > 1 ? d_tc_splitk : nullptr,
                        // Depth follows this layer's miss count: a chunk's
                        // compute can only hide another chunk's copy, so with
                        // fewer than two misses there is nothing to hide and
                        // the extra batch launch is pure cost (lifecycle trail
                        // 2026-11-04-qwen38-flash-next-lifecycle.md, §6).
                        n_miss >= 2 ? kMoeStageChunks : 1);
                    if (e != cudaSuccess) {
                        result.error_msg =
                            "tc-batch experts l" + std::to_string(l);
                        break;
                    }
                } else {
                    e = fence_uploads();
                    if (e == cudaSuccess)
                        e = moe::RunMoERoutedFFN(
                            stream, store, cache, l, h_slots, K, d_mixed, H, I,
                            d_moe_acc, d_yg, d_yu, d_y2, ffn_err, nullptr,
                            nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, d_ygu_batch, d_h_batch, d_yp_batch,
                            d_gemv_tbl, nullptr, &tc_scratch);
                    if (e != cudaSuccess) {
                        result.error_msg =
                            "experts l" + std::to_string(l) + ": " + ffn_err;
                        break;
                    }
                }
            } else {
                e = fence_uploads();
                if (e != cudaSuccess) {
                    result.error_msg = "expert upload fence l" +
                                       std::to_string(l);
                    break;
                }
                e = moe::RunMoERoutedFFN(
                    stream, store, cache, l, h_slots, K, d_mixed, H, I,
                    d_moe_acc, d_yg, d_yu, d_y2, ffn_err, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, d_ygu_batch,
                    d_h_batch, d_yp_batch, d_gemv_tbl, nullptr, &tc_scratch);
                if (e != cudaSuccess) {
                    result.error_msg = "experts l" + std::to_string(l) +
                                       " kb" + std::to_string(kv_base) + " mt" +
                                       std::to_string(kv->max_total) + " lin" +
                                       std::to_string(lin_idx) + ": " + ffn_err;
                    break;
                }
            }
            e = moe::RunMoESharedCombine(
                stream, lw.shared_w1, lw.shared_w2, lw.shared_w3,
                lw.d_shared_gate, d_mixed, 1, H, SI, d_moe_acc,
                d_si, d_sh, d_gate_val, d_act, shared_xq, shared_xqs);
            if (e != cudaSuccess) { result.error_msg = "shared"; break; }
            // scatter MoE output back into the streams.
            e = RunExpHcInjectScatter(stream, d_hyper2, d_act, d_inj_f, 1, H,
                                      hc, d_hyper);
            if (e != cudaSuccess) { result.error_msg = "hc-scatter2"; break; }
        }
        if (e != cudaSuccess) break;

        // ── global mixer (input-mix only) + final norm + head ──
        if (pw.mixer_hc_cache) {
            e = RunExpHcMixerCached(stream, d_hyper, *pw.mixer_hc_cache,
                                    pw.d_mixer_norm, 1, H, hc,
                                    pw.exp_top.hc_lowrank, d_act);
        } else {
            e = RunExpHcMixer(stream, d_hyper, pw.mixer_down, pw.mixer_up,
                              pw.d_mixer_norm, 1, H, hc,
                              pw.exp_top.hc_lowrank, d_act, d_normed_scr);
        }
        if (e != cudaSuccess) { result.error_msg = "mixer"; break; }
        e = RunExpHeadGemm(stream, pw.d_head_f16, d_act, d_logits_f32, H, V);
        if (e != cudaSuccess) { result.error_msg = "head"; break; }

        kv->kv_rows += 1;
        if (step >= prompt_steps - 1) {
            e = sampler.Sample(stream, d_logits_f32, V,
                               lsxcommon::sample::LogitType::kF32, d_argmax,
                               d_maxv_scratch);
            if (e != cudaSuccess) { result.error_msg = "argmax"; break; }
            exp_record_token_kernel<<<1, 1, 0, stream>>>(d_tokens, generated,
                                                         d_argmax);
            // TTFT: the first generated token's forward is complete.
            if (generated == 0 && t_ttft_ev) {
                cudaEventRecord(t_ttft_ev, stream);
                ttft_recorded = true;
            }
            if (step + 1 < total_steps)
                exp_feed_token_kernel<<<1, 1, 0, stream>>>(d_ids, d_argmax);
            // The PLE n-gram context lives on the host (mmap gather): pull
            // the sampled token back so the NEXT step's n-gram contains the
            // real generated history. Without this, `cur` stays pinned to
            // the last prompt token, the n-gram freezes, and the hidden
            // state collapses into repetition loops (root-caused vs the
            // HF greedy reference: gen0 parity cos=0.91, gen1 cos=-0.04).
            int32_t h_tok = -1;
            {
                cudaMemcpyAsync(&h_tok, d_argmax, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready_ev, stream);
                {
                    // The status must not be dropped: a sticky launch failure
                    // from the decode step makes this return an error, and
                    // swallowing it fed a -1 token back into the model.
                    const cudaError_t q =
                        lsxcommon::cuda::WaitEventChecked(tok_ready_ev);
                    if (q != cudaSuccess)
                    {
                        result.error_msg = "token readback: " +
                                           std::string(cudaGetErrorString(q));
                        break;
                    }
                }
                token_history.push_back(h_tok);
                if (token_history.size() > 64)
                    token_history.erase(token_history.begin());
                cur = h_tok;
            }
            if (std::getenv("LSX_TRACE_TOKENS") && h_ids) {
                cudaEventRecord(tok_ready_ev, stream);
                cudaEventSynchronize(tok_ready_ev);
                // Full-vocab logits top-8: compares against the HF greedy
                // reference (hf_greedy8.py) step by step.
                static std::vector<float> hlog;
                hlog.resize(static_cast<size_t>(V));
                cudaMemcpy(hlog.data(), d_logits_f32, sizeof(float) * V,
                           cudaMemcpyDeviceToHost);
                int idx[8] = {0};
                float val[8] = {-1e30f, -1e30f, -1e30f, -1e30f,
                                -1e30f, -1e30f, -1e30f, -1e30f};
                for (int i = 0; i < V; ++i) {
                    for (int k = 0; k < 8; ++k) {
                        if (hlog[static_cast<size_t>(i)] > val[k]) {
                            for (int m = 7; m > k; --m) {
                                val[m] = val[m - 1];
                                idx[m] = idx[m - 1];
                            }
                            val[k] = hlog[static_cast<size_t>(i)];
                            idx[k] = i;
                            break;
                        }
                    }
                }
                std::fprintf(stderr,
                             "[trace] gen=%d top: %d:%.3f %d:%.3f %d:%.3f"
                             " %d:%.3f %d:%.3f\n",
                             generated, idx[0], val[0], idx[1], val[1],
                             idx[2], val[2], idx[3], val[3], idx[4],
                             val[4]);
            }
            // Streaming contract (OPEN_TASKS §E1c): publish the token now, not
            // in a bulk drain after the whole budget has been generated. This
            // readback already existed for the PLE n-gram history, so it costs
            // nothing extra; without it a `--serve` consumer saw nothing until
            // the request finished, and a flag raised from the token callback
            // could not stop the decode (it was only read after generation had
            // already run to max_gen_tokens). EOS is not published, matching the
            // contract the bulk drain had.
            if (gs && h_tok != eos_id) {
                gs->Emit(static_cast<std::uint32_t>(h_tok),
                         static_cast<std::uint32_t>(generated));
            }
            const bool eos_hit = h_tok == eos_id;
            ++generated;
            // EOS terminates generation here rather than after the fact: the
            // tokens after it could never reach the consumer (the drain stopped
            // at the first EOS), so decoding them was pure waste. Greedy
            // decoding makes this an observation, not a behaviour change - the
            // published token list is unchanged, because every token before the
            // EOS depends only on the tokens before it.
            if (eos_hit) {
                result.eos_terminated = true;
                break;
            }
        } else {
            cur = prompt[step + 1];
            if (h_ids)
                cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        }
    }

    // Assemble the published token list from the recorded history and publish
    // the last step's logits row. The tokens were already streamed above.
    if (generated > 0) {
        std::vector<int32_t> hgen(static_cast<size_t>(generated));
        cudaMemcpy(hgen.data(), d_tokens, sizeof(int32_t) * (size_t)generated,
                   cudaMemcpyDeviceToHost);
        result.tokens.assign(hgen.begin(), hgen.end());
        // A no-op on the early-stop path (EOS is the last recorded token); kept
        // for the paths that still reach the budget without an EOS, and because
        // it is the single definition of the truncation contract.
        result.eos_terminated = result.eos_terminated ||
                                moe::MoETruncateTokensAtEos(result.tokens, eos_id);
        // Publish the last step's fp32 logits row for parity harnesses.
        std::vector<float> hlast(static_cast<size_t>(V));
        cudaStreamSynchronize(stream);
        if (cudaMemcpy(hlast.data(), d_logits_f32, sizeof(float) * V,
                       cudaMemcpyDeviceToHost) == cudaSuccess)
            result.final_logits = std::move(hlast);
    }
    if (t_stop_ev) {
        cudaEventRecord(t_stop_ev, stream);
        cudaEventSynchronize(t_stop_ev);
        float ms_total = 0.f;
        if (cudaEventElapsedTime(&ms_total, t_start_ev, t_stop_ev) ==
            cudaSuccess)
            result.total_ms = ms_total;
        float ms_ttft = 0.f;
        if (t_ttft_ev && ttft_recorded &&
            cudaEventElapsedTime(&ms_ttft, t_start_ev, t_ttft_ev) ==
                cudaSuccess)
            result.ttft_ms = ms_ttft;
    }
    if (t_start_ev) cudaEventDestroy(t_start_ev);
    if (t_ttft_ev) cudaEventDestroy(t_ttft_ev);
    if (t_stop_ev) cudaEventDestroy(t_stop_ev);
    if (h_ids) cudaFreeHost(h_ids);
    if (h_emb_rows) cudaFreeHost(h_emb_rows);
    if (tok_ready_ev) cudaEventDestroy(tok_ready_ev);
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (ev_front) cudaEventDestroy(ev_front);
    for (int k = 0; k < 16; ++k)
        if (ev_uploads[k]) cudaEventDestroy(ev_uploads[k]);
    cudaFree(d_ids);
    cudaFree(d_pos);
    cudaFree(d_argmax);
    cudaFree(d_tokens);
    cudaFree(d_slots);
    if (e != cudaSuccess) {
        if (result.error_msg.empty()) result.error_msg = "generation failed";
        return e;
    }
    // Residency diagnostics. Decode wall time for this family is dominated by
    // staging evicted experts from the pinned host arena (each miss is a
    // ~2.6 MiB H2D inside the per-layer router handshake), so the hit rate IS
    // the throughput explanation: a delta in t/s with no hit-rate record cannot
    // be attributed to a code change or to a different continuation. Printed
    // for every completed request, like the `[v41-cache]` line in the DeepSeek
    // driver. Precedent: the GLM audit measured 46.6% hit -> 13.0 t/s against
    // 57.4% -> 14.6 t/s on one binary, same protocol.
    {
        const std::int64_t routes = cache.hits + cache.misses;
        if (routes > 0 && generated > 0) {
            const double hit_rate =
                static_cast<double>(cache.hits) / static_cast<double>(routes);
            log::info(
                "[qwen38-cache] hits=" + std::to_string(cache.hits) +
                " misses=" + std::to_string(cache.misses) + " hit_rate=" +
                std::to_string(hit_rate) + " misses_per_token=" +
                std::to_string(static_cast<double>(cache.misses) /
                               static_cast<double>(generated)) +
                " slots=" + std::to_string(cache.slots) + " experts=" +
                std::to_string(cache.total_experts));
        }
    }
    result.resident_bytes = pw.resident_bytes;
    result.ok = true;
    return cudaSuccess;
}

// ── LM head GEMM (deterministic FP16-in / FP32-out GEMV) ──────────────────
// The lm_head is kept in BF16-derived FP16: NVFP4 logit noise exceeds the
// top-1/top-2 margin of this model. One warp per output row; fp32 accumulate
// with an ordered shfl reduce — bit-deterministic across runs (the cuBLAS
// default algo may use split-K atomics, breaking the selfcheck parity
// memcmp contract).
namespace {

__global__ void exp_head_gemv_kernel(const __half *__restrict__ head,
                                     const __half *__restrict__ hidden,
                                     float *__restrict__ logits, int H, int V)
{
    const int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (v >= V) return;
    const int lane = threadIdx.x & 31;
    const __half *row = head + static_cast<int64_t>(v) * H;
    float part = 0.f;
    for (int h = lane; h < H; h += 32)
        part += __half2float(row[h]) * __half2float(hidden[h]);
    for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffffu, part, off);
    if (lane == 0) logits[v] = part;
}

} // namespace

cudaError_t RunExpHeadGemm(cudaStream_t stream, const __half *d_head_f16,
                           const __half *d_hidden, float *d_logits, int H,
                           int V)
{
    const int warps_per_block = 8;
    const int blocks = (V + warps_per_block - 1) / warps_per_block;
    exp_head_gemv_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        d_head_f16, d_hidden, d_logits, H, V);
    return cudaGetLastError();
}

} // namespace lsxcommon::qwen_moe_exp
