// Xing4.0-29B-A4B kernels (stages 5-6). Mirrors modeling_xing4_0.py:
// Xing4_0UnweightedRMSNorm / Xing4_0HyperConnection / Xing4_0TopkRouter.
//
// Numerics at kOff: exp/rsqrt resolve to the model-neutral glibc replicas
// (ExpfGlibc / RsqrtfGlibc) so the block matches the BF16 host reference, and
// every divide is IEEE __fdiv_rn (never __frcp_rn, whose reciprocal-then-
// multiply reassociation would drift from the host on the gate values that pick
// the expert set). The only reductions (the mHC sum-of-squares and the router
// dot products) are deterministic and atomic-free: they feed final_logits.

#include "lsxcommon/xing_moe_kernels.h"

#include "xing_moe_device_math.h"

#include <cuda_bf16.h>

#include <cmath>

namespace lsxcommon::xing {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxHcMult = 8;
constexpr int kMaxMix = (2 + kMaxHcMult) * kMaxHcMult;   // 80
constexpr int kMaxExperts = 256;
// Shared layout inside the mHC kernel: [0, hc_mult*hidden) streams,
// then kThreads reduction slots, then kMaxHcMult pre gates and kMaxMix comb
// slots. Offsets are compile-time so nothing aliases.
constexpr int kCombBase = kMaxHcMult;

__device__ __forceinline__ float XingSigmoid(float x)
{
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, device::XingExpfGlibc(-x)));
}

__global__ void xing_unweighted_rmsnorm4_kernel(
    const float *__restrict__ x, float *__restrict__ out, int hc_mult,
    int hidden, float eps)
{
    extern __shared__ float red[];
    const int n = hc_mult * hidden;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const float v = x[j];
        sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float inv =
        device::XingRsqrtfGlibc(__fadd_rn(__fdiv_rn(red[0], static_cast<float>(n)), eps));
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        out[j] = __fmul_rn(x[j], inv);
}

// Phase 1 of the depth mixer runs one block per hc_fn row and is load-latency
// bound, so it gets the widest block an SM offers; phase 2 is a single block
// over `hidden` and does not care.
constexpr int kProjThreads = 1024;

// Mirrors kAttnSplits in xing_moe_inference.h (the session sizes the scratch).
constexpr int kMaxAttnSplits = 16;

// mHC, phase 1 of 2: the hc_fn projection, one block per row (mix = 24 at
// hc_mult=4).
//
// The whole depth mixer used to run as `<<<1, 256>>>` - one block is one SM, so
// the 745 KB of streams + weights were read through a single SM's cache slice:
// 61 us standalone, 82 us in situ, 6.6 ms of every decode step across the 40
// resident layers plus the MTP layer. Spread over mix blocks the same bytes cost
// a few us.
//
// Each block re-derives the RMSNorm factor (dot(normed, w) == norm_factor *
// dot(raw, w)), so no normalised copy has to be staged in global memory between
// the two kernels: every block reads the streams once and both reductions (the
// sum of squares and the dot product) share that pass.
__global__ void xing_hyper_conn_proj_kernel(
    const float *__restrict__ streams, const __nv_bfloat16 *__restrict__ hc_fn,
    int hc_mult, int hidden, float eps, float *__restrict__ rows)
{
    __shared__ float warp_acc[2][kProjThreads / 32];
    const int k = blockIdx.x;
    const int n = hc_mult * hidden;
    const __nv_bfloat16 *row = hc_fn + static_cast<size_t>(k) * n;
    float ss = 0.0f, dot = 0.0f;
    // Unrolled 4x: the loop is latency-bound, not bandwidth-bound (1.4 MB of
    // mostly L2-resident traffic over 24 blocks), so what matters is how many
    // loads are in flight per thread.
#pragma unroll 4
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        const float v = streams[j];
        ss = __fadd_rn(ss, __fmul_rn(v, v));
        dot = __fadd_rn(dot, __fmul_rn(v, __bfloat162float(row[j])));
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        ss = __fadd_rn(ss, __shfl_down_sync(0xffffffffu, ss, off));
        dot = __fadd_rn(dot, __shfl_down_sync(0xffffffffu, dot, off));
    }
    if ((threadIdx.x & 31) == 0)
    {
        warp_acc[0][threadIdx.x >> 5] = ss;
        warp_acc[1][threadIdx.x >> 5] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int nwarps = blockDim.x >> 5;
        float s = 0.0f, d = 0.0f;
        for (int w = 0; w < nwarps; ++w)
        {
            s = __fadd_rn(s, warp_acc[0][w]);
            d = __fadd_rn(d, warp_acc[1][w]);
        }
        const float inv = device::XingRsqrtfGlibc(
            __fadd_rn(__fdiv_rn(s, static_cast<float>(n)), eps));
        rows[k] = __fmul_rn(d, inv);
    }
}

// mHC, phase 2 of 2: the gates, the Sinkhorn mixing matrix and the collapse.
// Still one block, but the expensive part (24 x 14336 multiply-adds) moved to
// phase 1; what is left is hc_mult sigmoids, hc_mult^2 Sinkhorn cells on thread 0
// and a 57 KB collapse spread over the block.
// mHC, phase 2 of 2: the gates, the Sinkhorn mixing matrix and the collapse.
//
// Still one block (the tile is hc_mult x hc_mult), but the arithmetic is spread
// over it: ExpfGlibc is a bit-exact glibc replica written in DOUBLE precision
// and a consumer Blackwell SM retires FP64 at 1/64 of FP32, so 20 exps + 4
// sigmoids run serially on thread 0 took ~36 us - more than the whole projection
// phase. One cell per thread (and one row/column reciprocal per thread) is
// numerically IDENTICAL: exp and the max are element-wise, and the Sinkhorn's
// per-cell updates were already independent. What is left on thread 0 is the
// 24-element row copy.
//
// Measured: 61 us (single-block original) -> 3.1 us projection + ~2 us gates.
__global__ void xing_hyper_conn_gates_kernel(
    const float *__restrict__ rows_in, const __nv_bfloat16 *__restrict__ hc_base,
    const float *__restrict__ hc_scale, int hc_mult, int hidden, float eps,
    int sinkhorn_iters, float clamp_min, float clamp_max, float post_gain,
    float *__restrict__ post, float *__restrict__ comb,
    float *__restrict__ pre_gate_out)
{
    __shared__ float pre_gate[kMaxHcMult];
    __shared__ float comb_buf[kMaxMix];
    __shared__ float rows[kMaxMix];
    __shared__ float tile_max;
    const int hw = hc_mult * hc_mult;
    const int tid = threadIdx.x;
    for (int k = tid; k < (2 + hc_mult) * hc_mult; k += blockDim.x)
        rows[k] = rows_in[k];
    if (tid == 0) tile_max = -INFINITY;
    __syncthreads();
    // gates: one thread per stream
    for (int d = tid; d < hc_mult; d += blockDim.x)
    {
        pre_gate[d] = XingSigmoid(__fadd_rn(
            __fmul_rn(rows[d], hc_scale[0]),
            __bfloat162float(hc_base[d])));
        post[d] = __fmul_rn(
            post_gain,
            XingSigmoid(__fadd_rn(__fmul_rn(rows[hc_mult + d], hc_scale[1]),
                                 __bfloat162float(hc_base[hc_mult + d]))));
    }
    // comb logits, one thread per cell, then the max as a block reduction
    const int off = 2 * hc_mult;
    float local = -INFINITY;
    for (int i = tid; i < hw; i += blockDim.x)
    {
        const float lg = fminf(
            fmaxf(__fadd_rn(__fmul_rn(rows[off + i], hc_scale[2]),
                           __bfloat162float(hc_base[off + i])),
                  clamp_min),
            clamp_max);
        comb_buf[kCombBase + i] = lg;
        local = fmaxf(local, lg);
    }
    for (int stride = 16; stride > 0; stride >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, stride));
    if ((tid & 31) == 0) atomicMax(reinterpret_cast<int *>(&tile_max),
                                   __float_as_int(local));
    __syncthreads();
    if (hw > 32)
        for (int i = tid; i < hw; i += blockDim.x)
            comb_buf[kCombBase + i] =
                device::XingExpfGlibc(comb_buf[kCombBase + i] - tile_max);
    // Sinkhorn. hc_mult^2 cells fit in one warp at the frozen geometry (16 of
    // 32 lanes), so warp 0 keeps the whole tile in REGISTERS and the 20
    // iterations need no shared-memory round trip and no __syncthreads at all -
    // that alone was most of the block's 12.6 us. The arithmetic per cell is
    // unchanged: a row sum still accumulates c = 0..hc_mult-1 in that order (the
    // four values arrive by shuffle in lane order) and a column sum r = 0.. in
    // that order, so this is bit-identical to the serial version, not a
    // tolerance-class change. Wider hc_mult falls back to the shared-memory
    // path below.
    if (hw <= 32)
    {
        const int lane = tid & 31;
        const int r = lane / hc_mult;
        const int c = lane - r * hc_mult;
        const bool live = lane < hw;
        float cell = live ? device::XingExpfGlibc(
                                __fsub_rn(comb_buf[kCombBase + lane], tile_max))
                          : 0.0f;
        for (int it = 0; it < sinkhorn_iters; ++it)
        {
            float s = 0.0f;
            for (int j = 0; j < hc_mult; ++j)
                s += __shfl_sync(0xffffffffu, cell, r * hc_mult + j);
            const float inv = __fdiv_rn(1.0f, __fadd_rn(s, eps));
            cell = __fmul_rn(cell, inv);
            s = 0.0f;
            for (int j = 0; j < hc_mult; ++j)
                s += __shfl_sync(0xffffffffu, cell, j * hc_mult + c);
            const float inv2 = __fdiv_rn(1.0f, __fadd_rn(s, eps));
            cell = __fmul_rn(cell, inv2);
        }
        if (live) comb[lane] = cell;
    }
    else
    {
        // Sinkhorn: thread r owns row r (then column r). The per-cell arithmetic
        // is unchanged; only who executes it moved.
        for (int it = 0; it < sinkhorn_iters; ++it)
        {
            __syncthreads();
            for (int r = tid; r < hc_mult; r += blockDim.x)
            {
                float s = 0.0f;
                for (int c = 0; c < hc_mult; ++c)
                    s += comb_buf[kCombBase + r * hc_mult + c];
                s = __fdiv_rn(1.0f, __fadd_rn(s, eps));
                for (int c = 0; c < hc_mult; ++c)
                    comb_buf[kCombBase + r * hc_mult + c] *= s;
            }
            __syncthreads();
            for (int c = tid; c < hc_mult; c += blockDim.x)
            {
                float s = 0.0f;
                for (int r = 0; r < hc_mult; ++r)
                    s += comb_buf[kCombBase + r * hc_mult + c];
                s = __fdiv_rn(1.0f, __fadd_rn(s, eps));
                for (int r = 0; r < hc_mult; ++r)
                    comb_buf[kCombBase + r * hc_mult + c] *= s;
            }
        }
        if (tid == 0)
            for (int i = 0; i < hw; ++i) comb[i] = comb_buf[kCombBase + i];
    }
    __syncthreads();

    // pre_gate goes to the shared scratch for the collapse kernel: hc_mult
    // floats, and the alternative is recomputing four sigmoids per block.
    for (int d = tid; d < hc_mult; d += blockDim.x)
        pre_gate_out[d] = pre_gate[d];
}

// mHC, phase 3: the collapse, on a wide block.
//
// Multiplies the RAW streams, not the normalised ones: the unweighted RMSNorm
// only feeds hc_fn, while upstream collapses with `pre * hidden_streams`
// (Xing4_0HyperConnection.forward). Reading a normalised copy here would apply
// the factor twice - caught by the host-parity test.
__global__ void xing_hyper_conn_collapse_kernel(const float *__restrict__ streams,
                                                const float *__restrict__ gates,
                                                int hc_mult, int hidden,
                                                float *__restrict__ collapsed)
{
    float g[8];
    for (int d = 0; d < hc_mult; ++d) g[d] = gates[d];
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    {
        float acc = 0.0f;
        for (int d = 0; d < hc_mult; ++d)
            acc = __fadd_rn(acc, __fmul_rn(g[d], streams[d * hidden + c]));
        collapsed[c] = acc;
    }
}

__global__ void xing_mix_kernel(const float *__restrict__ streams,
                                const float *__restrict__ out,
                                const float *__restrict__ post,
                                const float *__restrict__ comb, int hc_mult,
                                int hidden, float *__restrict__ hidden_new)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden) return;
    // Read the whole column into registers BEFORE the first write: the decode
    // loop mixes in place, and writing stream d while later iterations still
    // read stream d corrupts the residual (column isolation is not enough).
    float in[kMaxHcMult];
    for (int k = 0; k < hc_mult; ++k) in[k] = streams[k * hidden + c];
    const float o = out[c];
    for (int d = 0; d < hc_mult; ++d)
    {
        float acc = __fmul_rn(post[d], o);
        for (int k = 0; k < hc_mult; ++k)
            acc = __fadd_rn(acc, __fmul_rn(comb[d * hc_mult + k], in[k]));
        hidden_new[d * hidden + c] = acc;
    }
}

// Sigmoid noaux_tc router. One WARP PER EXPERT for the [num_experts, hidden]
// projection, and a block-parallel top-k instead of a serial scan by thread 0.
//
// Before: <<<1, 256>>> with one thread per expert (64 of 256 busy, each reading
// a whole 3584-wide row alone) and then 4 x 64 dependent shared-memory reads on
// thread 0 - 129 us per launch, 38 launches per token (~4.9 ms, the second
// largest serial cost after the mHC block). Budget:
// test_xing_moe_kernels "Xing router stays inside its latency budget".
//
// Selection semantics match the serial scan it replaces: the bias only steers the
// CHOICE, the weight stays the raw sigmoid score, and ties go to the LOWEST
// expert index.
// Router phase 1: the projection, one block per expert.
//
// The whole router ran as `<<<1, 256>>>` - one block is one SM, so all
// E x hidden router weights (64 x 3584 = 458 KiB at the frozen geometry) were
// read through a single SM's cache slice, and 38 calls per token made this the
// largest single kernel in a decode step. One block per expert is bit-identical
// (same 32-lane strided k-order, same shuffle reduction) and turns 458 KiB
// through one SM into 7 KiB through 64.
__global__ __launch_bounds__(32) void xing_router_proj_kernel(
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ x, int num_experts, int hidden,
    float *__restrict__ scores)
{
    constexpr int kWarpSz = 32;
    const int e = blockIdx.x;
    if (e >= num_experts) return;
    const int lane = threadIdx.x;
    const __nv_bfloat16 *row = weight + static_cast<size_t>(e) * hidden;
    float acc = 0.0f;
    for (int i = lane; i < hidden; i += kWarpSz)
        acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(row[i]),
                                       __bfloat162float(x[i])));
    for (int off = kWarpSz / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[e] = acc;
}

// Router phase 2: sigmoid, top-k and the weight renorm. One block, but the
// expensive part (the projection) moved to phase 1; what is left is E sigmoids
// (one thread each - see the note in phase 1's caller about ExpfGlibc being a
// double-precision replica), top_k rounds of a warp argmax, and 16-lane writes.
__global__ void xing_router_topk_kernel(
    const float *__restrict__ bias, const float *__restrict__ scores_in,
    int num_experts, int top_k, bool norm_topk_prob, float routed_scale,
    float *__restrict__ out_weights, int32_t *__restrict__ out_experts)
{
    constexpr int kWarpSz = 32;
    constexpr int kWarps = kThreads / kWarpSz;
    __shared__ float score[kMaxExperts];
    __shared__ bool taken[kMaxExperts];
    __shared__ float w[16];
    __shared__ int ids[16];
    __shared__ float cand_score[kWarps];
    __shared__ int cand_idx[kWarps];

    const int lane = threadIdx.x & (kWarpSz - 1);
    const int warp = threadIdx.x >> 5;

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        score[e] = scores_in[e];
    __syncthreads();
    // The sigmoid is one thread per expert, NOT inside the loop above:
    // XingSigmoid is a bit-exact glibc replica in DOUBLE precision and a
    // consumer Blackwell SM retires FP64 at 1/64 of FP32, so E/kWarps = 8
    // back-to-back double-precision calls per warp cost ~13 us - measurable as
    // the part of this kernel that does NOT scale with `hidden` (13.2 us at
    // hidden=224 vs 32.7 us at 3584). Same function on the same value, so the
    // scores stay bit-identical.
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        score[e] = XingSigmoid(score[e]);
        taken[e] = false;
    }
    __syncthreads();

    // top-k: warp-local argmax, then kWarps candidates, repeated top_k times
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        const int e = warp * kWarpSz + lane;
        float best = -INFINITY;
        int best_idx = -1;
        if (e < num_experts && !taken[e])
        {
            best = __fadd_rn(score[e], bias[e]);
            best_idx = e;
        }
        for (int off = kWarpSz / 2; off > 0; off >>= 1)
        {
            const float other = __shfl_down_sync(0xffffffffu, best, off);
            const int other_idx = __shfl_down_sync(0xffffffffu, best_idx, off);
            if (other > best || (other == best && other_idx < best_idx &&
                                 best_idx >= 0))
            {
                best = other;
                best_idx = other_idx;
            }
        }
        if (lane == 0)
        {
            cand_score[warp] = best;
            cand_idx[warp] = best_idx;
        }
        __syncthreads();
        // warp 0 picks the winner among the kWarps candidates
        if (threadIdx.x < kWarpSz)
        {
            float bs = (threadIdx.x < kWarps) ? cand_score[threadIdx.x] : -INFINITY;
            int bi = (threadIdx.x < kWarps) ? cand_idx[threadIdx.x] : -1;
            for (int off = kWarpSz / 2; off > 0; off >>= 1)
            {
                const float other = __shfl_down_sync(0xffffffffu, bs, off);
                const int other_idx = __shfl_down_sync(0xffffffffu, bi, off);
                if (other > bs || (other == bs && other_idx < bi && bi >= 0))
                {
                    bs = other;
                    bi = other_idx;
                }
            }
            if (threadIdx.x == 0)
            {
                cand_idx[0] = bi;    // broadcast through shared memory
                cand_score[0] = bs;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int best_e = cand_idx[0];
            taken[best_e] = true;
            ids[k] = best_e;
            w[k] = score[best_e];
            wsum = __fadd_rn(wsum, score[best_e]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        const float denom =
            norm_topk_prob ? __fdiv_rn(1.0f, __fadd_rn(wsum, 1e-20f)) : 1.0f;
        for (int k = 0; k < top_k; ++k)
        {
            out_weights[k] = __fmul_rn(__fmul_rn(w[k], denom), routed_scale);
            out_experts[k] = ids[k];
        }
        // Zero-fill the unused lanes so the caller's 16-wide buffers are fully
        // defined even though this family selects 4.
        for (int i = top_k; i < 16; ++i)
        {
            out_weights[i] = 0.0f;
            out_experts[i] = -1;
        }
    }
}

// Causal MHA decode for one layer: one block per head.
//
// All three phases use the whole block now. The softmax and the value
// accumulation used to be serial over t <= pos (thread 0 for the softmax, one
// thread per head dim for the values), so at pos = 2047 a layer cost ~814 us -
// the entire context-dependent half of a 31 ms token on the real 29B checkpoint,
// measured as us/token = 14.3 ms + 16.5 ms per 1000 cached positions. Budget:
// test_xing_moe_kernels "Xing attention stays inside its latency budget at a 2k
// context".
// `out16` receives __float2half of the same value: the decode step's next kernel
// (o_proj) consumes FP16 and used to run a whole elementwise ToHalf launch over
// the attention output to get it - 40 per token at an empty-launch cost of
// 1.84 us. Bit-identical, because the rounding happens at the same value.
// CAPTURE_SAFE: with the fixed-geometry launch (below) this kernel is launched at
// every position, so it must exclude itself where the split path owns the answer.
// The condition is exactly the one the host used to pick the path with
// (`used <= 1`), and it is uniform across the block.
// Score dot product q . k_row, accumulating in ASCENDING i order so the result is
// bit-identical to the scalar loop below it.
//
// The float4 form exists because the scalar loop was LATENCY bound, not sector
// bound: it issued one 4-byte load per iteration over 192 iterations with a serial
// FP accumulation, so a thread had one row in flight. Measured on this card, the
// attention KV read sustains 744-817 GB/s where a plain float4 grid-stride read of
// the same 335 MB reaches 1687 GB/s (mem.stream_read_ceiling_gbps) - a 2x gap that
// vectorising the load addresses directly. Both forms walk i ascending and use the
// same __fadd_rn/__fmul_rn, so they agree to the bit.
//
// Falls back to scalar when qk is not a multiple of 4. Row starts are always 16-B
// aligned: a row offset is a multiple of head_dim in BOTH layouts, and cudaMalloc
// returns 256-B aligned storage.
__device__ __forceinline__ float XingDotRow(const float *__restrict__ qt,
                                             const float *__restrict__ kt, int qk,
                                             bool vec4)
{
    float acc = 0.0f;
    if (vec4)
    {
        const float4 *q4 = reinterpret_cast<const float4 *>(qt);
        const float4 *k4 = reinterpret_cast<const float4 *>(kt);
        const int qk4 = qk >> 2;
        for (int j = 0; j < qk4; ++j)
        {
            const float4 a = q4[j];
            const float4 b = k4[j];
            acc = __fadd_rn(acc, __fmul_rn(a.x, b.x));
            acc = __fadd_rn(acc, __fmul_rn(a.y, b.y));
            acc = __fadd_rn(acc, __fmul_rn(a.z, b.z));
            acc = __fadd_rn(acc, __fmul_rn(a.w, b.w));
        }
    }
    else
    {
        for (int i = 0; i < qk; ++i)
            acc = __fadd_rn(__fmul_rn(qt[i], kt[i]), acc);
    }
    return acc;
}

__global__ void xing_attn_kernel(const float *__restrict__ q,
                                 const float *__restrict__ kc,
                                 const float *__restrict__ vc,
                                 const int *__restrict__ d_scalars,
                                 KvCacheGeometry kgeom, KvCacheGeometry vgeom,
                                 int heads, int qk, int vh, float scale,
                                 float *__restrict__ scores,
                                 float *__restrict__ out,
                                 __half *__restrict__ out16, int capacity)
{
    extern __shared__ float sm[];
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    __shared__ float partial[2][128];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int pos = d_scalars[0];
    const int n_pos = pos + 1;
    if (n_pos > nthreads) return;  // split-K owns it; see CAPTURE_SAFE
    float *sc = scores + static_cast<size_t>(h) * capacity;
    const float *qt = q + static_cast<size_t>(h) * qk;

    // phase 1: scores, parallel over positions with a 2-way unroll so each
    // thread keeps two K rows in flight.
    {
        int t = tid;
        for (; t + nthreads < n_pos; t += 2 * nthreads)
        {
            const float *kt0 = kc + kgeom.Offset(0, h, t);
            // nthreads positions ahead, so the hop is nthreads*pos_stride - not
            // nthreads*heads*qk, which only holds for pos-major. Unreachable today
            // (the guard above caps n_pos at nthreads, so t+nthreads >= n_pos and
            // this loop never iterates), but leaving the pos-major stride in a
            // migrated file is a trap for whoever lifts that cap.
            const float *kt1 =
                kt0 + static_cast<size_t>(nthreads) * kgeom.pos_stride;
            float a0 = 0.0f, a1 = 0.0f;
            for (int i = 0; i < qk; ++i)
            {
                a0 = __fadd_rn(a0, __fmul_rn(qt[i], kt0[i]));
                a1 = __fadd_rn(a1, __fmul_rn(qt[i], kt1[i]));
            }
            sc[t] = __fmul_rn(a0, scale);
            sc[t + nthreads] = __fmul_rn(a1, scale);
        }
        const bool vec4 = (qk & 3) == 0;
        for (; t < n_pos; t += nthreads)
        {
            const float *kt = kc + kgeom.Offset(0, h, t);
            sc[t] = __fmul_rn(XingDotRow(qt, kt, qk, vec4), scale);
        }
    }

    // Block reduction over the score row: shuffle inside a warp, then one slot
    // per warp. Replaces the 2048-iteration serial scan on thread 0.
    float m = -INFINITY;
    for (int t = tid; t < n_pos; t += nthreads) m = fmaxf(m, sc[t]);
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((tid & 31) == 0) warp_max[tid >> 5] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < (nthreads >> 5); ++w) m = fmaxf(m, warp_max[w]);

    float sum = 0.0f;
    for (int t = tid; t < n_pos; t += nthreads)
    {
        const float e = device::XingExpfGlibc(__fsub_rn(sc[t], m));
        sm[t] = e;
        sum = __fadd_rn(sum, e);
    }
    for (int off = 16; off > 0; off >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
    if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int w = 0; w < (nthreads >> 5); ++w) sum = __fadd_rn(sum, warp_sum[w]);
    const float inv = __fdiv_rn(1.0f, sum);
    for (int t = tid; t < n_pos; t += nthreads) sm[t] = __fmul_rn(sm[t], inv);
    __syncthreads();

    // phase 3: weighted sum over positions. Each head dim is accumulated by
    // nthreads / vh threads at once (two groups at vh=128, blockDim=256), each
    // striding the positions; the partials are summed through shared memory.
    const int groups = nthreads / vh;
    const int group = tid / vh;
    const int d = tid - group * vh;
    float acc = 0.0f;
    for (int t = group; t < n_pos; t += groups)
        acc = __fadd_rn(
            acc,
            __fmul_rn(sm[t], vc[vgeom.Offset(0, h, t, d)]));
    if (groups > 1 && group > 0) partial[group - 1][d] = acc;
    __syncthreads();
    if (tid < vh)
    {
        float total = acc;
        for (int g = 1; g < groups; ++g) total = __fadd_rn(total, partial[g - 1][d]);
        out[static_cast<size_t>(h) * vh + d] = total;
        if (out16) out16[static_cast<size_t>(h) * vh + d] = __float2half(total);
    }
}
// Split-K phase 1: one block per (head, chunk of positions). Same per-chunk
// arithmetic as the single-block kernel - score, block softmax over the chunk,
// weighted V sum - but the chunk's max and sum are published instead of
// normalised in place, because only the combine kernel knows the global max.
// `splits` is the CHUNK-COUNT DIVISOR, and stays 8 in both launch modes: the
// per-position chunk is recomputed here from the device position with the formula
// the host used, so the chunk boundaries - the reduction order, and therefore the
// bits - are exactly what the host-chosen launch produced. The published slots
// are still strided by gridDim.y, which is `used` on the eager path (fewer chunks
// than the divisor at small contexts) and the full `splits` on the capture-safe
// one, and each mode's combine is told the same number.
__global__ void xing_attn_split_kernel(const float *__restrict__ q,
                                       const float *__restrict__ kc,
                                       const float *__restrict__ vc,
                                       const int *__restrict__ d_scalars,
                                       KvCacheGeometry kgeom, KvCacheGeometry vgeom,
                                       int heads, int qk, int vh, float scale,
                                       int capacity, int splits, int max_chunk,
                                       float *d_m, float *d_l, float *d_part)
{
    extern __shared__ float sm[];
    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    __shared__ float partial[2][128];
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    // gridDim.y == the number of chunk slots that exist for this launch; the
    // divisor `splits` and the slot count differ whenever a chunk would be empty.
    const int pos = d_scalars[0];
    const int n_pos = pos + 1;
    // Single-block path owns it. Uniform across the block.
    if (n_pos <= nthreads) return;
    const int chunk = ((n_pos + splits - 1) / splits + nthreads - 1) /
                      nthreads * nthreads;
    const int begin = s * chunk;
    const int end = min(begin + chunk, n_pos);
    const int count = end - begin;
    // A chunk that starts past the context has nothing to contribute. It still has
    // to publish: the combine sums all `splits` entries, and an unwritten slot
    // would be read as whatever the previous position left there. (m=-inf, l=0)
    // makes its weight exp(-inf - gmax) = 0, so it adds nothing to either sum.
    if (begin >= n_pos)
    {
        if (tid == 0)
        {
            d_m[static_cast<size_t>(h) * gridDim.y + s] = -INFINITY;
            d_l[static_cast<size_t>(h) * gridDim.y + s] = 0.0f;
        }
        return;
    }
    float *sc = sm;
    const float *qt = q + static_cast<size_t>(h) * qk;

    const bool vec4 = (qk & 3) == 0;
    for (int t = tid; t < count; t += nthreads)
    {
        const float *kt = kc + kgeom.Offset(0, h, begin + t);
        sc[t] = __fmul_rn(XingDotRow(qt, kt, qk, vec4), scale);
    }
    float m = -INFINITY;
    for (int t = tid; t < count; t += nthreads) m = fmaxf(m, sc[t]);
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((tid & 31) == 0) warp_max[tid >> 5] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < (nthreads >> 5); ++w) m = fmaxf(m, warp_max[w]);

    float sum = 0.0f;
    for (int t = tid; t < count; t += nthreads)
    {
        const float e = device::XingExpfGlibc(__fsub_rn(sc[t], m));
        sc[t] = e;
        sum = __fadd_rn(sum, e);
    }
    for (int off = 16; off > 0; off >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
    if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int w = 0; w < (nthreads >> 5); ++w) sum = __fadd_rn(sum, warp_sum[w]);
    if (tid == 0)
    {
        d_m[static_cast<size_t>(h) * gridDim.y + s] = m;
        d_l[static_cast<size_t>(h) * gridDim.y + s] = sum;
    }

    // Unnormalised weighted sum: the combine applies exp(m_s - m_global)/l.
    const int groups = nthreads / vh;
    const int group = tid / vh;
    const int d = tid - group * vh;
    float acc = 0.0f;
    for (int t = group; t < count; t += groups)
        acc = __fadd_rn(
            acc, __fmul_rn(sc[t],
                           vc[vgeom.Offset(0, h, begin + t, d)]));
    if (groups > 1 && group > 0) partial[group - 1][d] = acc;
    __syncthreads();
    if (tid < vh)
    {
        float total = acc;
        for (int g = 1; g < groups; ++g) total = __fadd_rn(total, partial[g - 1][d]);
        d_part[(static_cast<size_t>(h) * gridDim.y + s) * vh + d] = total;
    }
}

// Split-K phase 2: rescale each chunk by exp(m_s - m_global), sum the chunks and
// divide by the global sum. One block per head, one thread per head dim.
// Split-K phase 2: rescale each chunk by exp(m_s - m_global), sum the chunks and
// divide by the global sum. One block per head, one thread per head dim.
//
// The kAttnSplits rescale factors are computed ONCE into shared by the first
// kAttnSplits threads instead of once per head dim: they are per-chunk scalars,
// and ExpfGlibc is a double-precision replica on a part that retires FP64 at
// 1/64, so 16 chunks x 128 dims of threads meant 16 serial exps per thread and
// the combine cost 6.8 us - 0.28 ms of a decode step across the layers.
__global__ void xing_attn_combine_kernel(const float *__restrict__ d_m,
                                         const float *__restrict__ d_l,
                                         const float *__restrict__ d_part,
                                         const int *__restrict__ d_scalars,
                                         int heads, int vh, int splits,
                                         float *__restrict__ out,
                                         __half *__restrict__ out16)
{
    __shared__ float w[kMaxAttnSplits];
    __shared__ float gmax;
    // The single-block kernel wrote `out` at this position; combining stale
    // partials over it would corrupt it. Uniform across the block.
    if (d_scalars[0] + 1 <= kThreads) return;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float *m = d_m + static_cast<size_t>(h) * splits;
    const float *l = d_l + static_cast<size_t>(h) * splits;
    const float *part = d_part + static_cast<size_t>(h) * splits * vh;
    float local = -INFINITY;
    for (int s = tid; s < splits; s += blockDim.x) local = fmaxf(local, m[s]);
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, off));
    if (tid == 0) gmax = local;
    __syncthreads();
    const float gm = gmax;
    for (int s = tid; s < splits; s += blockDim.x)
        w[s] = device::XingExpfGlibc(__fsub_rn(m[s], gm));
    __syncthreads();
    if (tid < vh)
    {
        float denom = 0.0f, total = 0.0f;
        for (int s = 0; s < splits; ++s)
        {
            denom = __fadd_rn(denom, __fmul_rn(l[s], w[s]));
            total = __fadd_rn(
                total,
                __fmul_rn(part[static_cast<size_t>(s) * vh + tid], w[s]));
        }
        const float v = __fdiv_rn(total, denom);
        out[static_cast<size_t>(h) * vh + tid] = v;
        if (out16) out16[static_cast<size_t>(h) * vh + tid] = __float2half(v);
    }
}
}  // namespace

cudaError_t LaunchXingAttention(const float *q, const float *kc,
                                const float *vc, const int *d_scalars,
                                KvCacheGeometry kgeom, KvCacheGeometry vgeom,
                                int heads, int qk, int vh, float scale,
                                float *scores, float *out, int n_pos,
                                int splits, float *d_m, float *d_l,
                                float *d_part, __half *out16,
                                bool capture_safe, cudaStream_t stream)
{
    const int capacity = static_cast<int>(kgeom.capacity);
    if (!q || !kc || !vc || !scores || !out || !d_scalars || heads <= 0 ||
        qk <= 0 || vh <= 0 || n_pos <= 0 || capacity < n_pos || splits <= 0 ||
        kgeom.head_dim != qk || vgeom.head_dim != vh)
        return cudaErrorInvalidValue;
    if (splits > kMaxAttnSplits) return cudaErrorInvalidValue;
    // One block per position chunk on top of the one per head: at pos=2047 the
    // plain form filled 32 of 170 SMs, and the whole context-dependent half of a
    // step ran there. Chunks are multiples of the block size so the score phase
    // keeps its coalesced (and its shared-memory staging contiguous) layout.
    const int chunk = ((n_pos + splits - 1) / splits + kThreads - 1) /
                      kThreads * kThreads;
    const int used = (n_pos + chunk - 1) / chunk;
    if (!capture_safe)
    {
        // Eager path: pick the path on the host, launch only what is needed.
        if (used <= 1)
        {
            xing_attn_kernel<<<heads, kThreads, n_pos * sizeof(float),
                               stream>>>(q, kc, vc, d_scalars, kgeom, vgeom, heads,
                                         qk, vh, scale, scores, out, out16,
                                         capacity);
            return cudaGetLastError();
        }
        if (!d_m || !d_l || !d_part) return cudaErrorInvalidValue;
        xing_attn_split_kernel<<<dim3(heads, used), kThreads,
                                 chunk * sizeof(float), stream>>>(
            q, kc, vc, d_scalars, kgeom, vgeom, heads, qk, vh, scale, capacity,
            splits, chunk,
            d_m, d_l, d_part);
        xing_attn_combine_kernel<<<heads, vh, 0, stream>>>(
            d_m, d_l, d_part, d_scalars, heads, vh, used, out, out16);
        return cudaGetLastError();
    }
    // Capture-safe path: a graph bakes the launch configuration in, so it must not
    // depend on the position. Both paths are launched at their maximum geometry and
    // exclude each other on the device position (see CAPTURE_SAFE). Which one does
    // the work is the same decision the host made, so the result is bit-identical -
    // the extra launches cost ~1.8 us each and only on the graph path.
    if (!d_m || !d_l || !d_part) return cudaErrorInvalidValue;
    const int max_chunk = ((capacity + splits - 1) / splits + kThreads - 1) /
                          kThreads * kThreads;
    xing_attn_kernel<<<heads, kThreads, kThreads * sizeof(float), stream>>>(
        q, kc, vc, d_scalars, kgeom, vgeom, heads, qk, vh, scale, scores, out,
        out16, capacity);
    xing_attn_split_kernel<<<dim3(heads, splits), kThreads,
                             max_chunk * sizeof(float), stream>>>(
        q, kc, vc, d_scalars, kgeom, vgeom, heads, qk, vh, scale, capacity, splits,
        max_chunk,
        d_m, d_l, d_part);
    xing_attn_combine_kernel<<<heads, vh, 0, stream>>>(
        d_m, d_l, d_part, d_scalars, heads, vh, splits, out, out16);
    return cudaGetLastError();
}

cudaError_t LaunchXingUnweightedRmsNorm4(const float *x, float *out,
                                          int hc_mult, int hidden, float eps,
                                          cudaStream_t stream)
{
    if (hc_mult <= 0 || hc_mult > kMaxHcMult || hidden <= 0)
        return cudaErrorInvalidValue;
    const size_t smem = static_cast<size_t>(kThreads) * sizeof(float);
    xing_unweighted_rmsnorm4_kernel<<<1, kThreads, smem, stream>>>(x, out,
                                                                   hc_mult, hidden,
                                                                   eps);
    return cudaGetLastError();
}

cudaError_t LaunchXingHyperConnection(
    const float *streams, const __nv_bfloat16 *hc_fn,
    const __nv_bfloat16 *hc_base, const float *hc_scale, int hc_mult,
    int hidden, float eps, int sinkhorn_iters, float clamp_min, float clamp_max,
    float post_gain, float *collapsed, float *post, float *comb, float *rows,
    cudaStream_t stream)
{
    if (!streams || !hc_fn || !hc_base || !hc_scale || !collapsed || !post ||
        !comb || !rows || hc_mult <= 0 || hc_mult > kMaxHcMult || hidden <= 0)
        return cudaErrorInvalidValue;
    const int mix = (2 + hc_mult) * hc_mult;
    if (mix > kMaxMix) return cudaErrorInvalidValue;
    // Phase 1 across mix blocks (24 SMs at hc_mult=4), phase 2 on one. Neither
    // needs dynamic shared memory any more, so the 56 KiB opt-in a single block
    // used to require is gone with it.
    // Phase 1 gets 1024 threads: 24 blocks x 256 was only 8 warps per SM to hide
    // the load latency of a 14 KiB-stride dot product.
    xing_hyper_conn_proj_kernel<<<mix, kProjThreads, 0, stream>>>(
        streams, hc_fn, hc_mult, hidden, eps, rows);
    // Phase 2 is split again: the gates/Sinkhorn block is tiny, but its collapse
    // was a 57 KB read that ONE block had to issue, and that is latency-bound
    // (256 threads = 8 warps of loads in flight). Running the gates at 1024
    // threads instead measured 15.4 us against 12.6 - __launch_bounds__'s 64
    // register cap made the FP64 path spill, which cost more than the warps won.
    // So the collapse gets its own wide block with no FP64 in it at all.
    xing_hyper_conn_gates_kernel<<<1, kThreads, 0, stream>>>(
        rows, hc_base, hc_scale, hc_mult, hidden, eps, sinkhorn_iters, clamp_min,
        clamp_max, post_gain, post, comb, rows + kMaxMix);
    xing_hyper_conn_collapse_kernel<<<(hidden + kProjThreads - 1) / kProjThreads,
                                      kProjThreads, 0, stream>>>(
        streams, rows + kMaxMix, hc_mult, hidden, collapsed);
    return cudaGetLastError();
}
cudaError_t LaunchXingMix(const float *streams, const float *out,
                          const float *post, const float *comb, int hc_mult,
                          int hidden, float *hidden_new, cudaStream_t stream)
{
    if (hc_mult <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const int blocks = (hidden + kThreads - 1) / kThreads;
    xing_mix_kernel<<<blocks, kThreads, 0, stream>>>(streams, out, post, comb,
                                                     hc_mult, hidden, hidden_new);
    return cudaGetLastError();
}

cudaError_t LaunchXingRouter(const __nv_bfloat16 *weight, const float *bias,
                             const __nv_bfloat16 *x, int num_experts, int hidden,
                             int top_k, bool norm_topk_prob,
                             float routed_scale, float *out_weights,
                             int32_t *out_experts, float *scores,
                             cudaStream_t stream)
{
    if (num_experts <= 0 || num_experts > kMaxExperts || hidden <= 0 ||
        top_k <= 0 || top_k > 16)
        return cudaErrorInvalidValue;
    // A top_k larger than the expert count has no k-th candidate: the selection
    // loop would emit expert id -1 and the caller's resolve step would reject
    // the whole layer. Clamp instead, and zero the lanes we do not fill.
    const int k = std::min(top_k, num_experts);
    if (!scores) return cudaErrorInvalidValue;
    xing_router_proj_kernel<<<num_experts, 32, 0, stream>>>(
        weight, x, num_experts, hidden, scores);
    xing_router_topk_kernel<<<1, kThreads, 0, stream>>>(
        bias, scores, num_experts, k, norm_topk_prob, routed_scale, out_weights,
        out_experts);
    return cudaGetLastError();
}

}  // namespace lsxcommon::xing