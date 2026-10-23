#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/log.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/stream_scratch.h"
#include "nvfp4_codec.h"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
#include "lsx_kernels_quant.cuh"
#include "nvfp4_scale.h"
#include "qwen_reduce.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon {
namespace qwen {

namespace {

__device__ __forceinline__ float silu_dev(float x)
{
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_dev(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_dev(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

// RMSNorm (offset 1+w): unified kernels::RmsNormTreeKernel
// <__half, Offset1PlusW> (lsx_kernels_norm.cuh).

// ── RMSNormGated: out = rmsnorm(x) * head * act(z) ────────────────────────
// qwen3.5 uses silu(z); qwen4_exp sets output_gate_type="sigmoid". The kernel
// is parameterized by `use_sigmoid_gate` so both families share one launch.
__global__ void rmsnorm_gated_kernel(const __half *__restrict__ x, const __half *__restrict__ z,
                                     const __half *whead, __half *out,
                                     int rows, int dim, double eps,
                                     bool use_sigmoid_gate = false)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __half *xr = x + (int64_t)row * dim;
    const __half *zr = z + (int64_t)row * dim;
    extern __shared__ float sarray[];
    float *squares = sarray;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = __half2float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / (float)dim + (float)eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float wf = __half2float(whead[j]);
        float g = __half2float(zr[j]);
        const float act = use_sigmoid_gate ? sigmoid_dev(g) : silu_dev(g);
        out[(int64_t)row * dim + j] = __float2half(__half2float(xr[j]) * inv * wf * act);
    }
}

// ── RoPE cos/sin (partial rotary, dims = rotary_dim) ───────────────────────
__global__ void rope_cos_sin_kernel(const int32_t *__restrict__ pos, __half *__restrict__ cos_buf,
                                    __half *sin_buf, int seq, int rotary_dim,
                                    double theta)
{
    // Layout: cos_buf/sin_buf hold [seq][rotary_dim/2]; entry (t,d) =
    // cos/sin of the pair at rotary index 2d.
    const int64_t total = (int64_t)seq * (rotary_dim / 2);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        const int t = (int)(idx / (rotary_dim / 2));
        const int d = (int)(idx % (rotary_dim / 2));
        double inv = 1.0 / std::pow(theta, (double)(2 * d) / (double)rotary_dim);
        double ang = (double)pos[t] * inv;
        cos_buf[idx] = __float2half((float)std::cos(ang));
        sin_buf[idx] = __float2half((float)std::sin(ang));
    }
}

__global__ void partial_rope_kernel(const __half *__restrict__ q, const __half *__restrict__ k,
                                    const __half *cos_buf, const __half *sin_buf,
                                    __half *q_out, __half *k_out,
                                    int seq, int q_heads, int k_heads,
                                    int head_dim, int rotary_dim)
{
    // HF Qwen3_5 rotate_half pairing: for j in [0, rot/2) the pair is
    // (j, j + rot/2); cos/sin index j. cos/sin layout: [seq, rot/2].
    const int64_t half = rotary_dim / 2;
    // q rotation
    {
        const int64_t total = (int64_t)seq * q_heads * half;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
            const int j = (int)(idx % half);
            const int h = (int)((idx / half) % q_heads);
            const int t = (int)(idx / (half * q_heads));
            const float c = __half2float(cos_buf[t * half + j]);
            const float s = __half2float(sin_buf[t * half + j]);
            const int64_t row = ((int64_t)t * q_heads + h) * head_dim;
            const float x1 = __half2float(q[row + j]);
            const float x2 = __half2float(q[row + j + half]);
            q_out[row + j]        = __float2half(x1 * c - x2 * s);
            q_out[row + j + half] = __float2half(x2 * c + x1 * s);
        }
    }
    // q pass-through
    {
        const int64_t npt = (int64_t)seq * q_heads * (head_dim - rotary_dim);
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < npt; idx += (int64_t)blockDim.x * gridDim.x) {
            const int pk = (int)(idx % (head_dim - rotary_dim));
            const int rh = (int)(idx / (head_dim - rotary_dim));
            const int64_t base = (int64_t)rh * head_dim + rotary_dim;
            q_out[base + pk] = q[base + pk];
        }
    }
    // k rotation
    {
        const int64_t total = (int64_t)seq * k_heads * half;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
            const int j = (int)(idx % half);
            const int h = (int)((idx / half) % k_heads);
            const int t = (int)(idx / (half * k_heads));
            const float c = __half2float(cos_buf[t * half + j]);
            const float s = __half2float(sin_buf[t * half + j]);
            const int64_t row = ((int64_t)t * k_heads + h) * head_dim;
            const float x1 = __half2float(k[row + j]);
            const float x2 = __half2float(k[row + j + half]);
            k_out[row + j]        = __float2half(x1 * c - x2 * s);
            k_out[row + j + half] = __float2half(x2 * c + x1 * s);
        }
    }
    // k pass-through
    {
        const int64_t npt = (int64_t)seq * k_heads * (head_dim - rotary_dim);
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < npt; idx += (int64_t)blockDim.x * gridDim.x) {
            const int pk = (int)(idx % (head_dim - rotary_dim));
            const int rh = (int)(idx / (head_dim - rotary_dim));
            const int64_t base = (int64_t)rh * head_dim + rotary_dim;
            k_out[base + pk] = k[base + pk];
        }
    }
}

// Embed gather: unified kernels::EmbedGatherKernel (lsx_kernels_math.cuh).

// ── GatedDeltaNet causal conv1d (depthwise, kernel per channel) ─────────────
// mixed: [seq, conv_dim]; w: [conv_dim, kernel]; out: [seq, conv_dim].
// causal: out[t][c] = sum_{k=0..kernel-1} mixed[kv_base+t-k][c] * w[c][k]
//         (skip rows < 0). kv_base offsets the history read for incremental
//         decode; kv_base == 0 keeps the transient full-recompute semantics.
__global__ void qwen_conv1d_kernel(const __half *__restrict__ mixed, const __half *__restrict__ w,
                                   __half *out, int seq, int conv_dim,
                                   int kernel, int kv_base)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int tt = kv_base + t - k;
            if (tt < 0) continue;
            // HF causal conv1d: padding=kernel-1 both sides + right trim ==
            // F.conv1d cross-correlation == kernel reversal vs. this loop's
            // index, hence (kernel-1-k).
            acc += __half2float(w[c * kernel + (kernel - 1 - k)]) *
                   __half2float(mixed[(int64_t)tt * conv_dim + c]);
        }
        out[(int64_t)t * conv_dim + c] = __float2half(silu_dev(acc));
    }
}

// ── Full attention softmax core (per block per (t, head)) ──────────────────
// Thread 0 builds the logit row into shared memory; after the barrier every
// thread treats slogits as READ-ONLY (any read-modify-write there is a race:
// it corrupted softmax weights -> 0/0 -> NaN, which poisoned the residual
// stream and made generation non-deterministic).
__global__ void full_attn_softmax_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    __half *__restrict__ out, int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int ctx = kv_base + t;
    const int nscores = ctx + 1;
    const int threads = blockDim.x;

    const __half *qrow = q + ((int64_t)t * heads + h) * head_dim;
    const __half *kbase = k + (int64_t)kvh * head_dim;
    const __half *vbase = v + (int64_t)kvh * head_dim;

    extern __shared__ float slogits[];      // dynamic: [nscores]
    __shared__ float sreduce[512];          // separate block-reduction scratch

    // 1. Scores, chunk-per-thread (parallel dots).
    for (int j = threadIdx.x; j < nscores; j += threads) {
        const __half *krow = kbase + (int64_t)j * kv_heads * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += __half2float(qrow[d]) * __half2float(krow[d]);
        slogits[j] = dot * scale;
    }
    __syncthreads();

    // 2. Global max via warp-shuffle block reduction (bit-identical to the
    //    strided tree: fmaxf is associative/commutative).
    float m = -1e30f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        m = fmaxf(m, slogits[j]);
    m = qwen_block_max_shfl(m);
    __syncthreads();

    // 3. Softmax denominator via block reduction (one exp per score).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nscores; j += threads)
        sum += __expf(slogits[j] - m);
    sreduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / (sreduce[0] + 1e-10f);
    __syncthreads();

    // 4. Per-output-dim value pass: thread d sums all scores with the SHARED
    //    exp(s-m)*inv. Stride-d loop (idles when threads > head_dim).
    for (int d = threadIdx.x; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int j = 0; j < nscores; ++j) {
            const __half *vrow = vbase + (int64_t)j * kv_heads * head_dim;
            acc += __expf(slogits[j] - m) * inv * __half2float(vrow[d]);
        }
        out[((int64_t)t * heads + h) * head_dim + d] = __float2half(acc);
    }
}

} // namespace

// ── Public numeric kernels ──────────────────────────────────────────────────

// Block-wide chunked-parallel causal attention (GQA). Serves as the numeric
// entry for the standalone softmax-kernel launch (the high-level
// RunAttnFullAttention calls the same kernel with projections/RoPE/gate).
cudaError_t RunAttnFullAttnKernel(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale, int kv_base)
{
    if (!q || !k || !v || !out || seq <= 0 || heads <= 0 || head_dim <= 0 ||
        kv_heads <= 0 || kv_base < 0)
        return cudaErrorInvalidValue;
    // The softmax denominator uses a `threads/2` tree over blockDim =
    // min(head_dim, 512) and the GQA map is `h / (heads / kv_heads)`. Both
    // require a power-of-two head_dim and an exact division (the split-KV
    // entry already checks this; audit X7).
    if ((head_dim & (head_dim - 1)) != 0 || kv_heads > heads ||
        heads % kv_heads != 0)
        return cudaErrorInvalidValue;
    const dim3 grid(seq, heads);
    const int threads = std::min(head_dim, 512);
    // dynamic shared = slogits[nscores]; the block REDUCTION scratch is a
    // static __shared__ (sreduce[512]) and needs no dynamic smem.
    const size_t smem = (static_cast<size_t>(kv_base) + seq + 1) * sizeof(float);
    full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
        q, k, v, out, seq, heads, head_dim, kv_heads, scale, kv_base);
    return cudaGetLastError();
}

// ── Split-KV decode attention (seq==1, long context) ────────────────────────
// Two deterministic kernels (flash-decode shape):
//   1. partials: grid (heads, nchunks) x head_dim threads; each block owns a
//      fixed 512-row KV chunk, computes its scores into shared memory
//      (per-row dot identical to the legacy kernel), block-reduces the local
//      max m_c and denominator l_c, and emits the LOCAL-max-weighted value
//      partial pacc_c[d] (FP32). The extra block axis restores memory-level
//      parallelism the legacy one-block-per-head layout loses at long
//      contexts (measured: 0.546 ms/layer at pos 2048 vs ~0 at pos 64).
//   2. combine: grid heads x head_dim threads; global max is fmaxf-order-free,
//      the denominator and the chunk partial sum walk chunks in ASCENDING
//      order (fixed sequence -> run-to-run bit-identical).
// Numerics differ from the legacy kernel only by the chunk-rescale grouping
// (same contract class as the TC-batch amendment: tolerance-envelope parity
// vs legacy, binding run-to-run determinism).
// q . k_row over `hd` halves, in ASCENDING d with a PLAIN `+=`.
//
// Ascending d with a plain `+=` keeps the per-element order identical to the scalar
// form, which is the part that is guaranteed.
//
// NOT guaranteed by a flag: this object library is compiled WITHOUT -fmad=false
// (libs/lsxcommon/CMakeLists.txt, "qwen dense on" - FMA contraction is ENABLED, and
// the file's own policy note says "DO NOT rely on contraction in any new kernel"). So
// bit-identity rests on the scalar and vector forms contracting the same way under the
// current compiler, not on a flag forbidding otherwise. That is EMPIRICAL: it is what
// test_qwen_moe_{selfcheck,fsi,code,det}_selfcheck verify on real weights, and they are
// the arbiter - do not relax them on the strength of this comment. A change of
// optimisation level or toolkit could break parity here, and the gates would catch it.
//
// The load: TWO 4-byte __half2 loads per iteration replace four 2-byte loads - the
// scalar loop held a single K row in flight per thread, which is what Xing's float4
// change fixed for +6.5% end to end. Scalar fallback for hd % 8 != 0. The guard
// is % 8, not % 4, because the loop consumes k2[j] AND k2[j+1] with j stepping by 2:
// an odd hd/4 would read one element past the row.
//
// SPLIT KERNEL ONLY, deliberately. Vectorising the non-split kernel as well BREAKS
// parity (test_qwen_moe_{fsi,code,det}_selfcheck all fail on Qwen3.5-35B-A3B), even
// though this same helper is bit-identical in the split kernel - so the two kernels
// are not interchangeable and the short-context one is left alone. Since SelfCheck
// prompts are short enough that splits collapses to one chunk, that non-split kernel
// is what the golden gates actually exercise, which is how the difference was found.
// The split kernel is the long-context path this change is for.
__device__ __forceinline__ float AttnHalfDotRow(const __half *__restrict__ qrow,
                                                const __half *__restrict__ krow,
                                                int hd)
{
    float dot = 0.0f;
    if ((hd & 7) == 0)
    {
        // q is vectorised too, matching gemma. It is shared by every thread in the
        // block so it is already hot in L1, which is why this is worth little - but
        // four 2-byte loads per iteration become two, and the two helpers no longer
        // disagree about what "vectorised" means.
        const __half2 *k2 = reinterpret_cast<const __half2 *>(krow);
        const __half2 *q2 = reinterpret_cast<const __half2 *>(qrow);
        // k2/q2 hold hd/2 half2 elements and each iteration consumes two of
        // them, so the bound is hd/2 (the old hd/4 covered only half the row).
        for (int j = 0; j < (hd >> 1); j += 2)
        {
            const float2 f0 = __half22float2(k2[j]);
            const float2 f1 = __half22float2(k2[j + 1]);
            const float2 g0 = __half22float2(q2[j]);
            const float2 g1 = __half22float2(q2[j + 1]);
            dot += g0.x * f0.x;
            dot += g0.y * f0.y;
            dot += g1.x * f1.x;
            dot += g1.y * f1.y;
        }
        return dot;
    }
    for (int d = 0; d < hd; ++d) dot += __half2float(qrow[d]) * __half2float(krow[d]);
    return dot;
}

__global__ void split_kv_attn_partials_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, float *__restrict__ d_part,
    float *__restrict__ d_st, int heads, int head_dim, int kv_heads,
    float scale, int kv_base, int nchunks, int chunk)
{
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int nscores = kv_base + 1;
    const int begin = c * chunk;
    const int end = min(begin + chunk, nscores);
    const int rows = end - begin;
    const int threads = blockDim.x;

    const __half *qrow = q + (int64_t)h * head_dim;
    const __half *kbase_p = k + (int64_t)kvh * head_dim;
    const __half *vbase_p = v + (int64_t)kvh * head_dim;

    extern __shared__ float slogits[];      // dynamic: [rows]
    __shared__ float sreduce[512];          // separate block-reduction scratch

    // 1. Scores, chunk-per-thread (same per-row dot as the legacy kernel).
    for (int j = threadIdx.x; j < rows; j += threads) {
        const __half *krow =
            kbase_p + (int64_t)(begin + j) * kv_heads * head_dim;
        const float dot = AttnHalfDotRow(qrow, krow, head_dim);
        slogits[j] = dot * scale;
    }
    __syncthreads();

    // 2. Chunk max via warp-shuffle block reduction (order-free).
    float m = -1e30f;
    for (int j = threadIdx.x; j < rows; j += threads)
        m = fmaxf(m, slogits[j]);
    m = qwen_block_max_shfl(m);
    __syncthreads();

    // 3. Chunk denominator (one exp per score).
    float sum = 0.0f;
    for (int j = threadIdx.x; j < rows; j += threads)
        sum += __expf(slogits[j] - m);
    sreduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sreduce[threadIdx.x] += sreduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float l_c = sreduce[0];
    __syncthreads();

    // 4. Local-max-weighted value partial: one thread per output dim,
    //    ascending j inside the chunk.
    {
        const int d = threadIdx.x;          // blockDim == head_dim (entry)
        float acc = 0.0f;
        for (int j = 0; j < rows; ++j) {
            const __half *vrow =
                vbase_p + (int64_t)(begin + j) * kv_heads * head_dim;
            acc += __expf(slogits[j] - m) * __half2float(vrow[d]);
        }
        d_part[((int64_t)h * nchunks + c) * head_dim + d] = acc;
    }
    if (threadIdx.x == 0) {
        d_st[(int64_t)(h * nchunks + c) * 2 + 0] = m;
        d_st[(int64_t)(h * nchunks + c) * 2 + 1] = l_c;
    }
}

__global__ void split_kv_attn_combine_kernel(
    const float *__restrict__ d_part, const float *__restrict__ d_st,
    __half *__restrict__ out, int heads, int head_dim, int nchunks)
{
    const int h = blockIdx.x;
    const int d = threadIdx.x;              // blockDim == head_dim (entry)
    const int64_t base = (int64_t)h * nchunks;
    __shared__ float sreduce[512];

    // Global max: fmaxf is order-free (deterministic).
    float m = -1e30f;
    for (int c = d; c < nchunks; c += blockDim.x)
        m = fmaxf(m, d_st[(base + c) * 2 + 0]);
    m = qwen_block_max_shfl(m);
    __syncthreads();

    // Denominator over rescaled chunk sums (fixed chunk-strided order).
    float den = 0.0f;
    for (int c = d; c < nchunks; c += blockDim.x)
        den += d_st[(base + c) * 2 + 1] * __expf(d_st[(base + c) * 2 + 0] - m);
    sreduce[d] = den;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) sreduce[d] += sreduce[d + stride];
        __syncthreads();
    }
    const float inv = 1.0f / (sreduce[0] + 1e-10f);
    __syncthreads();

    // Weighted value: ascending chunk order (deterministic).
    float acc = 0.0f;
    for (int c = 0; c < nchunks; ++c)
        acc += __expf(d_st[(base + c) * 2 + 0] - m) *
               d_part[(base + c) * head_dim + d];
    out[(int64_t)h * head_dim + d] = __float2half(acc * inv);
}

cudaError_t RunAttnFullAttnKernelSplitKV(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale, int kv_base)
{
    if (!q || !k || !v || !out || seq != 1 || heads <= 0 || head_dim <= 0 ||
        head_dim > 512 || kv_heads <= 0 || kv_base < 0)
        return cudaErrorInvalidValue;
    // Block reductions (sreduce tree in both kernels) and the one-thread-per-
    // dim value pass assume a power-of-two blockDim (== head_dim); a non-pow2
    // block silently corrupts the softmax denominator. GQA mapping
    // kvh = h / (heads / kv_heads) requires kv_heads to divide heads exactly
    // (kv_heads > heads would divide by zero in-kernel).
    if ((head_dim & (head_dim - 1)) != 0 || kv_heads > heads ||
        heads % kv_heads != 0)
        return cudaErrorInvalidValue;
    const int nscores = kv_base + 1;
    constexpr int kChunk = 512;
    const int nchunks = (nscores + kChunk - 1) / kChunk;

    struct PartialsTag;
    struct StatsTag;
    void *part_raw = nullptr, *st_raw = nullptr;
    cudaError_t e = lsxcommon::cuda::StreamScratch<PartialsTag>::Get(
        stream, (size_t)heads * nchunks * head_dim * sizeof(float),
        &part_raw);
    if (e != cudaSuccess) return e;
    e = lsxcommon::cuda::StreamScratch<StatsTag>::Get(
        stream, (size_t)heads * nchunks * 2 * sizeof(float), &st_raw);
    if (e != cudaSuccess) return e;
    float *d_part = static_cast<float *>(part_raw);
    float *d_st = static_cast<float *>(st_raw);

    {
        const dim3 grid(heads, nchunks);
        const size_t smem =
            (size_t)std::min(kChunk, nscores) * sizeof(float);
        split_kv_attn_partials_kernel<<<grid, head_dim, smem, stream>>>(
            q, k, v, d_part, d_st, heads, head_dim, kv_heads, scale, kv_base,
            nchunks, kChunk);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) {
        split_kv_attn_combine_kernel<<<heads, head_dim, 0, stream>>>(
            d_part, d_st, out, heads, head_dim, nchunks);
        e = cudaGetLastError();
    }
    return e;
}

cudaError_t RunAttnRMSNorm(cudaStream_t stream, const __half *x, const __half *w,
                           __half *out, int rows, int hidden, double eps)
{
    if (!x || !w || !out || rows <= 0 || hidden <= 0) return cudaErrorInvalidValue;
    const int threads = std::min(hidden, 256);
    ::lsxcommon::kernels::RmsNormTreeKernel<__half, ::lsxcommon::kernels::NormWeight::Offset1PlusW><<<rows, threads, threads * sizeof(float) + std::min(hidden * int(sizeof(__half)), 48 * 1024), stream>>>(
        x, w, out, rows, hidden, eps);
    return cudaGetLastError();
}

cudaError_t RunAttnRopeCosSin(cudaStream_t stream, const int32_t *pos, int seq,
                              int rotary_dim, double theta, __half *cos_buf,
                              __half *sin_buf)
{
    const int threads = 256;
    const int64_t total = (int64_t)seq * (rotary_dim / 2);
    const int blocks = (int)((total + threads - 1) / threads);
    rope_cos_sin_kernel<<<blocks, threads, 0, stream>>>(
        pos, cos_buf, sin_buf, seq, rotary_dim, theta);
    return cudaGetLastError();
}

cudaError_t RunAttnPartialRope(cudaStream_t stream, const __half *q,
                               const __half *k, const __half *cos_buf,
                               const __half *sin_buf, __half *q_out,
                               __half *k_out, int seq, int q_heads,
                               int k_heads, int head_dim, int rotary_dim)
{
    const int threads = 256;
    const int64_t work = std::max(
        (int64_t)seq * std::max(q_heads, k_heads) * (rotary_dim / 2),
        (int64_t)seq * std::max(q_heads, k_heads) * (head_dim - rotary_dim));
    const int blocks = (int)((work + threads - 1) / threads);
    partial_rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos_buf, sin_buf, q_out, k_out, seq, q_heads, k_heads,
        head_dim, rotary_dim);
    return cudaGetLastError();
}

cudaError_t RunAttnPartialRope(cudaStream_t stream, const __half *q,
                               const __half *k, const __half *cos_buf,
                               const __half *sin_buf, __half *q_out,
                               __half *k_out, int seq, int heads,
                               int head_dim, int rotary_dim)
{
    return RunAttnPartialRope(stream, q, k, cos_buf, sin_buf, q_out, k_out,
                              seq, heads, heads, head_dim, rotary_dim);
}

cudaError_t RunAttnEmbed(cudaStream_t stream, const __half *embed,
                         const int32_t *ids, __half *out, int seq, int hidden)
{
    const int threads = 256;
    const int blocks = (int)(((int64_t)seq * hidden + threads - 1) / threads);
    lsxcommon::kernels::EmbedGatherKernel<__half><<<blocks, threads, 0, stream>>>(embed, ids, out, seq, hidden);
    return cudaGetLastError();
}

// ── Inference kernels (NVFP4 decode + dense pipeline) ───────────────────────

namespace {

// Shared NVFP4 decode primitives (nvfp4_codec.h). Aliased into this anonymous
// namespace so existing call sites stay unchanged.
using gpu::nvfp4::E2M1Value;
using gpu::nvfp4::E4M3ScaleToFloat;

// Fused NVFP4 dense GEMM: y[t][n] = sum_k x[t*K+k] * W[n*K+k], where W is the
// [N, K] row-major weight with per-16 E4M3 block scales + global_scale.
// x: [seq, K] FP16; y: [seq, N] FP16 (FP32 accumulate).

// Output gate: y = x * sigmoid(gate)  (per-feature).
__global__ void qwen_out_gate_kernel(const __half *x, const __half *gate,
                                     __half *y, int seq, int N)
{
    const int64_t total = (int64_t)seq * N;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    y[idx] = __float2half(__half2float(x[idx]) * sigmoid_dev(__half2float(gate[idx])));
}

// SwiGLU activation: y[i] = silu(a[i]) * b[i].
__global__ void qwen_swiglu_kernel(const __half *a, const __half *b,
                                   __half *y, int64_t n)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = __float2half(silu_dev(__half2float(a[idx])) * __half2float(b[idx]));
}

// Producer-fused swiglu + NVFP4 activation quant: one thread per 16-group.
// Emits the fp16 output AND its quantized form so the following down-GEMM
// skips the standalone ActQuant pass. Quantization is byte-identical to
// ActQuantKernel: values are rounded to half BEFORE encoding (same
// round-trip), same canonical FloatToE4m3 / tie-up chain.
__global__ void qwen_swiglu_quant_kernel(const __half *__restrict__ a,
                                         const __half *__restrict__ b,
                                         __half *__restrict__ y,
                                         uint8_t *__restrict__ packed,
                                         uint8_t *__restrict__ scales,
                                         int64_t n)
{
    const int64_t base =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (base >= n) return;
    const int len = static_cast<int>(
        min(static_cast<long long>(16), static_cast<long long>(n - base)));

    float vals[16];
    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const __half hv = __float2half(silu_dev(__half2float(a[base + i])) *
                                       __half2float(b[base + i]));
        y[base + i] = hv;
        vals[i] = __half2float(hv);
        absmax = fmaxf(absmax, fabsf(vals[i]));
    }

    const uint8_t sb =
        lsxcommon::nvfp4::FloatToE4m3(absmax * (1.0f / 6.0f));
    scales[base / 16] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;

    for (int i = 0; i < len; i += 2) {
        const uint8_t lo =
            lsxcommon::kernels::QuantizeE2m1TieUp(vals[i] * inv);
        const uint8_t hi = (i + 1 < len)
                               ? lsxcommon::kernels::QuantizeE2m1TieUp(
                                     vals[i + 1] * inv)
                               : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

// Residual add: unified kernels::AddInPlaceKernel (lsx_kernels_math.cuh).

cudaError_t LaunchNorm1d(const __half *x, const __half *w, __half *y,
                         int rows, int hidden, double eps, cudaStream_t stream)
{
    const int threads = std::min(hidden, 256);
    ::lsxcommon::kernels::RmsNormTreeKernel<__half, ::lsxcommon::kernels::NormWeight::Offset1PlusW><<<rows, threads, threads * sizeof(float) + std::min(hidden * int(sizeof(__half)), 48 * 1024), stream>>>(
        x, w, y, rows, hidden, eps);
    return cudaGetLastError();
}

// Splits q_proj output [seq, NH*HD*2] into query (first half) and gate (second
// half), reshaping query to [seq, NH, HD], and applies per-head RMSNorm.
// d_k_norm unused (key norm done separately); primal q_norm applied here.
__global__ void qwen_split_query_kernel(
    const __half *__restrict__ d_qc, __half *__restrict__ d_gate, __half *__restrict__ d_q,
    const __half *__restrict__ d_k_norm, int seq, int NH, int HD, const __half *__restrict__ d_q_norm,
    const __half *__restrict__ d_k_norm2, float eps)
{
    const int64_t idx = (int64_t)blockIdx.x;      // t*NH + h
    const int h = (int)(idx % NH);
    const int t = (int)(idx / NH);
    const int d = threadIdx.x;
    if (d >= HD) return;

    const int64_t qc_base = idx * 2 * HD;
    const __half qv = d_qc[qc_base + d];
    const __half gv = d_qc[qc_base + HD + d];

    d_q[idx * HD + d] = qv;
    d_gate[idx * HD + d] = gv;

    // RMSNorm over this head's HD dims (offset form).
    extern __shared__ float ssum[];
    const __half *qrow = d_q + idx * HD;
    float v = __half2float(qrow[d]);
    ssum[d] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (d < s) ssum[d] += ssum[d + s];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] / (float)HD + eps);
    float wf = __half2float(d_q_norm[d]);
    d_q[idx * HD + d] = __float2half(v * inv * (1.0f + wf));
    (void)d_k_norm; (void)d_k_norm2;
}

// Per-head RMSNorm offset form on [seq, NH, HD] (in place).
__global__ void qwen_head_norm_kernel(const __half *__restrict__ x, const __half *__restrict__ w,
                                      __half *y, int seq, int NH, int HD,
                                      float eps)
{
    const int64_t idx = (int64_t)blockIdx.x;      // t*NH + h
    const int d = threadIdx.x;
    if (d >= HD) return;
    extern __shared__ float ssum[];
    const __half *row = x + idx * HD;
    float v = __half2float(row[d]);
    ssum[d] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (d < s) ssum[d] += ssum[d + s];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] / (float)HD + eps);
    float wf = __half2float(w[d]);
    y[idx * HD + d] = __float2half(v * inv * (1.0f + wf));
}

// Converts a/b (half) projections into g/beta (float).
//   a_in -> a[h]; b_in -> b[h]; g = -exp(A_log[h])*softplus(a+dt[h]),
//   beta = sigmoid(b). h = e % nv.
__global__ void qwen_prep_gating_kernel(
    const __half *__restrict__ a_in, const __half *__restrict__ b_in, const float *__restrict__ A_log,
    const float *__restrict__ dt_bias, float *__restrict__ g, float *__restrict__ beta, int64_t total, int nv)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int h = (int)(idx % nv);
    float a = __half2float(a_in[idx]);
    float b = __half2float(b_in[idx]);
    beta[idx] = sigmoid_dev(b);
    g[idx] = -__expf(A_log[h]) * softplus_dev(a + dt_bias[h]);
}

// L2-normalize one (t, head) row of kd elements in place within a strided
// buffer (row stride = `stride`, head offset = `head_off`), then multiply by
// extra_scale (1/sqrt(kd) for q, 1 for k). Mirrors HF l2norm(eps=1e-6).
// Grid: (seq, nk); one thread per element. blockDim is a power of 2 >= kd.
__global__ void qwen_l2norm_rows_kernel(__half *__restrict__ base, int seq, int nk, int kd,
                                        int stride, int head_off,
                                        float extra_scale)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int l = threadIdx.x;
    __shared__ float vals[256];
    __shared__ float red[256];
    __shared__ float inv;
    __half *row = base + (int64_t)t * stride + head_off + (int64_t)h * kd;
    float v = 0.0f;
    if (l < kd) v = __half2float(row[l]);
    vals[l] = v;
    red[l] = v * v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        __syncthreads();
    }
    if (l == 0) inv = rsqrtf(red[0] + 1e-6f) * extra_scale;
    __syncthreads();
    if (l < kd) row[l] = __float2half(vals[l] * inv);
}

// Gated delta rule, block per (t, v_head), thread per value-dim d.
// q and k arrive already l2-normalised (q pre-scaled by 1/sqrt(kd)) in the
// strided conv buffer (row stride = `stride`); v is read from the same buffer.
// Columns d are independent -> one thread owns the whole column S[0..kd, d].
// Ordering mirrors HF torch_recurrent_gated_delta_rule:
//   S=0; for tt in 0..t: S*=exp(g); kv=sum_l S[l]*ktt[l]; Δ=(v-kv)β;
//        S[l]+=ktt[l]*Δ;  out=sum_l S[l]*q[l].
// A non-null `col_seed` (per-linear-layer slice of the persistent state)
// initialises col to the previous call's accumulator (incremental decode) and
// writes the final state back from the last row's block (t == seq-1).
__global__ void qwen_gated_delta_kernel(
    const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, __half *__restrict__ out,
    int seq, int nk, int kd, int nv, int vd, int r, int stride,
    float *__restrict__ col_seed)
{
    const int t = blockIdx.x;
    const int vh = blockIdx.y;
    const int kh = vh / r;         // key head shared by r value heads
    const int d = threadIdx.x;     // value-dim column [0, vd)
    if (d >= vd) return;

    // State layout is transposed [kd][nv*vd]: consecutive threads touch
    // consecutive floats -> coalesced state streaming (the old [col][kd]
    // layout strided by kd per thread and was bandwidth-bound at ~600 GB/s).
    const int vflat = vh * vd + d;
    const int vtotal = nv * vd;

    // Registers hold this thread's column [l in 0..kd).
    float col[256];
    for (int l = 0; l < kd; ++l)
        col[l] = col_seed ? col_seed[static_cast<int64_t>(l) * vtotal + vflat]
                          : 0.0f;

    const __half *qrow = q + (int64_t)t * stride + (int64_t)kh * kd;

    for (int tt = 0; tt <= t; ++tt) {
        const float decay = __expf(g[tt * nv + vh]);
        const float betaf = beta[tt * nv + vh];
        const __half *ktt = k + (int64_t)tt * stride + (int64_t)kh * kd;
        const __half *vtt = v + (int64_t)tt * stride + (int64_t)vh * vd;

        for (int l = 0; l < kd; ++l) col[l] *= decay;
        float kv = 0.0f;
        for (int l = 0; l < kd; ++l) kv += col[l] * __half2float(ktt[l]);
        const float delta = (__half2float(vtt[d]) - kv) * betaf;
        for (int l = 0; l < kd; ++l) col[l] += __half2float(ktt[l]) * delta;
    }

    if (col_seed && t == seq - 1) {
        for (int l = 0; l < kd; ++l)
            col_seed[static_cast<int64_t>(l) * vtotal + vflat] = col[l];
    }

    float acc = 0.0f;
    for (int l = 0; l < kd; ++l) acc += col[l] * __half2float(qrow[l]);
    out[((int64_t)t * nv + vh) * vd + d] = __float2half(acc);
}

// ── Fused decoder (seq=1) full-attention segments ─────────────────────────
// Stage 1: per-head q/k/v+gate (inline NVFP4 dequant to registers) + per-head
// RMSNorm(q)/(k) + inline RoPE (rotate-half pairing from PRECOMPUTED half
// tables, bit-identical values to rope_cos_sin_kernel at seq==1) + append
// k/v to persistent KV row kv_base. The old per-thread double pow/cos/sin ran
// NH*HD transcendental evaluations per layer per token; the table variant
// needs rotary_dim/2 once per token and matches the split path's numerics.
__global__ void fused_rope_tables_kernel(__half *__restrict__ cos_buf,
                                         __half *__restrict__ sin_buf,
                                         int rotary_dim, double theta, int pos)
{
    const int64_t total = rotary_dim / 2;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)blockDim.x * gridDim.x) {
        const double inv =
            1.0 / std::pow(theta, (double)(2 * idx) / (double)rotary_dim);
        const double ang = (double)pos * inv;
        cos_buf[idx] = __float2half((float)std::cos(ang));
        sin_buf[idx] = __float2half((float)std::sin(ang));
    }
}


// ── Fused linear-attention front stage A: raw qkv NVFP4 GEMM into the
// persistent d_qkv history. One block per token row t; blockDim = pow2(kd).
// Own launch: the conv stage (stage B) reads history rows written by other
// tokens of the same call, which requires kernel-launch ordering.
__global__ void fused_linear_qkv_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const lsx_nvfp4::LsxNvfp4Weight qkv,              // rows=H, cols=S
    int seq, int nk, int kd, int nv, int vd, int H, int kv_base,
    __half * __restrict__ d_qkv)            // persistent raw history
{
    const int t = blockIdx.x;
    const int S = nk * kd * 2 + nv * vd;
    const int N = S;
    const int K = H;
    const int nth = blockDim.x;

    for (int c = threadIdx.x; c < N; c += nth) {
        float acc = 0.0f;
        const int64_t cbase = (int64_t)c * K;
        for (int k = 0; k < K; ++k) {
            const int64_t e = cbase + k;
            const uint8_t byte = qkv.d_packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl = E4M3ScaleToFloat(qkv.d_scales[e >> 4]) * qkv.global_scale;
            // Tile-GEMM parity: dequantized weight is rounded to half before
            // the product (mirrors qwen_nvfp4_gemm_tile_kernel sW staging).
            const __half wv = __float2half(E2M1Value(nib) * scl);
            acc += __half2float(x[(int64_t)t * K + k]) * __half2float(wv);
        }
        d_qkv[((int64_t)kv_base + t) * S + c] = __float2half(acc);
    }
}

// ── Fused linear-attention front stage B: causal conv1d+silu + l2norm(q,k)
// for blocks [0, seq); z + a + b NVFP4 GEMMs + prep_gating for blocks
// [seq, 2*seq). The two halves are data-independent (conv reads d_qkv from
// stage A; z/a/b read d_normed directly), so they share one launch.
__global__ void fused_linear_conv_zab_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const lsx_nvfp4::LsxNvfp4Weight z,
    const lsx_nvfp4::LsxNvfp4Weight a, const lsx_nvfp4::LsxNvfp4Weight b,
    const float * __restrict__ A_log, const float * __restrict__ dt_bias,
    const __half * __restrict__ conv_w,     // [S, kernel]
    int seq, int nk, int kd, int nv, int vd, int H, int kernel, int kv_base,
    const __half * __restrict__ d_qkv,      // raw history from stage A
    __half * __restrict__ d_conv,           // [seq, S]
    __half * __restrict__ d_z, __half * __restrict__ d_ab,
    float * __restrict__ d_g, float * __restrict__ d_beta)
{
    const int VD = nv * vd;
    if (blockIdx.x < (unsigned)seq) {
        // ---- conv1d + silu + l2norm(q,k) for token blockIdx.x ----
        const int t = blockIdx.x;
        const int S = nk * kd * 2 + nv * vd;
        const int KD = nk * kd;
        for (int c = threadIdx.x; c < S; c += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < kernel; ++k) {
                const int tt = kv_base + t - k;
                if (tt < 0) continue;
                const float wv = __half2float(conv_w[(int64_t)c * kernel + (kernel - 1 - k)]);
                const float mv = __half2float(d_qkv[(int64_t)tt * S + c]);
                acc += wv * mv;
            }
            d_conv[(int64_t)t * S + c] = __float2half(silu_dev(acc));
        }
        __syncthreads();

        __shared__ float svals[256], sred[256], sinv;
        const float q_scale = 1.0f / sqrtf((float)kd);
        for (int h = 0; h < nk; ++h) {
            for (int slice = 0; slice < 2; ++slice) {
                const int head_off = (slice == 0) ? 0 : KD;
                const float extra = (slice == 0) ? q_scale : 1.0f;
                const int lane = threadIdx.x;
                const int c = head_off + h * kd + lane;
                float v = 0.0f;
                if (lane < kd) v = __half2float(d_conv[(int64_t)t * S + c]);
                svals[lane] = v; sred[lane] = v * v;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (lane < s) sred[lane] += sred[lane + s];
                    __syncthreads();
                }
                if (lane == 0) sinv = rsqrtf(sred[0] + 1e-6f) * extra;
                __syncthreads();
                if (lane < kd) d_conv[(int64_t)t * S + c] = __float2half(svals[lane] * sinv);
                __syncthreads();
            }
        }
    } else {
        // ---- z + a + b projections + prep gating for token blockIdx.x-seq ----
        const int t = blockIdx.x - (unsigned)seq;
        const int nth = blockDim.x;
        const __half *xrow = x + (int64_t)t * H;

        for (int c = threadIdx.x; c < VD; c += nth) {
            float acc = 0.0f;
            for (int k = 0; k < H; ++k) {
                const int64_t e = (int64_t)c * H + k;
                const uint8_t byte = z.d_packed[e >> 1];
                const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scl = E4M3ScaleToFloat(z.d_scales[e >> 4]) * z.global_scale;
                const __half wv = __float2half(E2M1Value(nib) * scl);
                acc += __half2float(xrow[k]) * __half2float(wv);
            }
            d_z[(int64_t)t * VD + c] = __float2half(acc);
        }

        __half *a_buf = d_ab;
        __half *b_buf = d_ab + (int64_t)seq * nv;
        for (int c = threadIdx.x; c < nv; c += nth) {
            float acca = 0.0f, accb = 0.0f;
            for (int k = 0; k < H; ++k) {
                const int64_t ea = (int64_t)c * H + k;
                const uint8_t ba = a.d_packed[ea >> 1];
                const uint8_t na = (ea & 1) ? (ba >> 4) : (ba & 0x0F);
                const float sa = E4M3ScaleToFloat(a.d_scales[ea >> 4]) * a.global_scale;
                acca += __half2float(xrow[k]) *
                        __half2float(__float2half(E2M1Value(na) * sa));
                const int64_t eb = (int64_t)c * H + k;
                const uint8_t bb = b.d_packed[eb >> 1];
                const uint8_t nb = (eb & 1) ? (bb >> 4) : (bb & 0x0F);
                const float sb = E4M3ScaleToFloat(b.d_scales[eb >> 4]) * b.global_scale;
                accb += __half2float(xrow[k]) *
                        __half2float(__float2half(E2M1Value(nb) * sb));
            }
            a_buf[(int64_t)t * nv + c] = __float2half(acca);
            b_buf[(int64_t)t * nv + c] = __float2half(accb);
        }
        __syncthreads();

        for (int c = threadIdx.x; c < nv; c += nth) {
            const float a_v = __half2float(a_buf[(int64_t)t * nv + c]);
            const float b_v = __half2float(b_buf[(int64_t)t * nv + c]);
            d_beta[(int64_t)t * nv + c] = sigmoid_dev(b_v);
            d_g[(int64_t)t * nv + c] = -__expf(A_log[c]) * softplus_dev(a_v + dt_bias[c]);
        }
    }
}


} // namespace

// MoE variant: matches kernels::SwigluKernel<Precise,false> bit-for-bit
// (sig = 1/(1+exp(-g)); out = sig*(g*u)) before the NVFP4 encode.
__global__ void qwen_swiglu_precise_quant_kernel(
    const __half *__restrict__ a, const __half *__restrict__ b,
    __half *__restrict__ y, uint8_t *__restrict__ packed,
    uint8_t *__restrict__ scales, int64_t n)
{
    const int64_t base =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (base >= n) return;
    const int len = static_cast<int>(
        min(static_cast<long long>(16), static_cast<long long>(n - base)));

    float vals[16];
    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float g = __half2float(a[base + i]);
        const float u = __half2float(b[base + i]);
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        const __half hv =
            __float2half(__fmul_rn(sig, __fmul_rn(g, u)));
        y[base + i] = hv;
        vals[i] = __half2float(hv);
        absmax = fmaxf(absmax, fabsf(vals[i]));
    }

    const uint8_t sb =
        lsxcommon::nvfp4::FloatToE4m3(absmax * (1.0f / 6.0f));
    scales[base / 16] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;

    for (int i = 0; i < len; i += 2) {
        const uint8_t lo =
            lsxcommon::kernels::QuantizeE2m1TieUp(vals[i] * inv);
        const uint8_t hi = (i + 1 < len)
                               ? lsxcommon::kernels::QuantizeE2m1TieUp(
                                     vals[i + 1] * inv)
                               : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

// ── Overflow-proof swiglu + NVFP4 activation quantization ──────────────────
// The fused kernel above stores the SwiGLU product to FP16 before quantizing:
//     const __half hv = __float2half(sig * g * u);
// FP16 tops out at 65504, so any model with massive activations overflows it
// to +inf, and the inf then poisons the whole block -- absmax becomes inf,
// FloatToE4m3(inf) saturates the block scale, and every nibble quantizes from
// an infinite value. Mistral-Small-4-119B layer 32 expert 67 reaches
// |silu(g)*u| = 2.5e5 from perfectly ordinary FP16 projections (max 444 and
// 568), which measured ~52x low in the layer accumulator on the tensor-core
// path while the scalar path was exact.
//
// Fix: keep the product in FP32, prescale it by a power of two so both the
// FP16 store and the E4M3 block scale are representable, and hand the
// reciprocal to the caller to fold into its FP32 accumulator. Two launches
// instead of one, and no host synchronization -- alpha is only known on the
// device, while the TC GEMM entry takes its scale as a host float.
//
// Bit-exact no-op when the activation already fits (alpha == 1): the FP32
// product is the same value the fused kernel computed before rounding it.

// Smallest power of two bringing the product within both spans: FP16 (65504)
// and the E4M3 block scale (448 * 6, the E2M1 grid's maximum).
__device__ __forceinline__ float SwigluPrescaleAlpha(float absmax)
{
    int k = 0;
    if (absmax > 0.0f) {
        float t = 448.0f * 6.0f;
        while (t < absmax && k < 60) { t *= 2.0f; ++k; }
    }
    return ldexpf(1.0f, k);
}

// Stage 1: the FP32 SwiGLU product (no FP16 store, so no overflow) plus a
// global absmax reduction.
__global__ void swiglu_f32_absmax_kernel(
    const __half *__restrict__ a, const __half *__restrict__ b,
    float *__restrict__ sw_f32, float *__restrict__ absmax_io, int64_t n)
{
    const int64_t base =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    if (base >= n) return;
    const int len = static_cast<int>(
        min(static_cast<long long>(16), static_cast<long long>(n - base)));
    for (int i = 0; i < len; ++i) {
        const float g = __half2float(a[base + i]);
        const float u = __half2float(b[base + i]);
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        const float sw = __fmul_rn(sig, __fmul_rn(g, u));
        sw_f32[base + i] = sw;
        // absmax is non-negative, so the int-domain atomicMax is the float
        // ordering (IEEE-754 non-negatives compare like their bit patterns).
        atomicMax(reinterpret_cast<int *>(absmax_io), __float_as_int(fabsf(sw)));
    }
}

// Stage 2: prescale, store FP16, then the canonical per-16 codec unchanged.
__global__ void swiglu_quant_prescaled_kernel(
    const float *__restrict__ sw_f32, __half *__restrict__ y,
    uint8_t *__restrict__ packed, uint8_t *__restrict__ scales,
    float *__restrict__ absmax_io, int64_t n)
{
    const int64_t base =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
    // Every block reads slot 0 (the absmax) for the whole launch, so alpha is
    // published in a separate slot: overwriting slot 0 here would race with
    // blocks that have not started yet.
    const float alpha = SwigluPrescaleAlpha(*absmax_io);
    const float inv_alpha = 1.0f / alpha;
    if (blockIdx.x == 0 && threadIdx.x == 0) absmax_io[1] = alpha;
    if (base >= n) return;
    const int len = static_cast<int>(
        min(static_cast<long long>(16), static_cast<long long>(n - base)));

    float vals[16];
    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float v = sw_f32[base + i] * inv_alpha;
        vals[i] = v;
        y[base + i] = __float2half(v);
        absmax = fmaxf(absmax, fabsf(v));
    }
    const uint8_t sb = lsxcommon::nvfp4::FloatToE4m3(absmax * (1.0f / 6.0f));
    scales[base / 16] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;
    for (int i = 0; i < len; i += 2) {
        const uint8_t lo =
            lsxcommon::kernels::QuantizeE2m1TieUp(vals[i] * inv);
        const uint8_t hi = (i + 1 < len)
                               ? lsxcommon::kernels::QuantizeE2m1TieUp(
                                     vals[i + 1] * inv)
                               : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

cudaError_t RunAttnSwigluPreScaledQuantInternal(
    cudaStream_t stream, const __half *a, const __half *b, float *sw_f32,
    __half *y, uint8_t *packed, uint8_t *scales, int64_t n, float *d_alpha)
{
    if (!a || !b || !sw_f32 || !y || !packed || !scales || !d_alpha || n <= 0)
        return cudaErrorInvalidValue;
    const int threads = 256;
    const int64_t groups = (n + 15) / 16;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);
    // Slot 0 accumulates across launches via atomicMax; the absmax domain is
    // non-negative, so zero is the correct identity.
    cudaError_t e = cudaMemsetAsync(d_alpha, 0, sizeof(float), stream);
    if (e != cudaSuccess) return e;
    swiglu_f32_absmax_kernel<<<blocks, threads, 0, stream>>>(a, b, sw_f32,
                                                             d_alpha, n);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    swiglu_quant_prescaled_kernel<<<blocks, threads, 0, stream>>>(
        sw_f32, y, packed, scales, d_alpha, n);
    return cudaGetLastError();
}

cudaError_t RunAttnSwigluPreciseQuantInternal(cudaStream_t stream,
                                             const __half *a, const __half *b,
                                             __half *y, uint8_t *packed,
                                             uint8_t *scales, int64_t n)
{
    if (!a || !b || !y || !packed || !scales || n <= 0)
        return cudaErrorInvalidValue;
    const int threads = 256;
    const int64_t groups = (n + 15) / 16;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);
    qwen_swiglu_precise_quant_kernel<<<blocks, threads, 0, stream>>>(
        a, b, y, packed, scales, n);
    return cudaGetLastError();
}

cudaError_t RunAttnSwigluQuant(cudaStream_t stream, const __half *a,
                               const __half *b, __half *y, uint8_t *packed,
                               uint8_t *scales, int64_t n)
{
    if (!a || !b || !y || !packed || !scales || n <= 0)
        return cudaErrorInvalidValue;
    const int threads = 256;
    const int64_t groups = (n + 15) / 16;
    const int blocks = static_cast<int>((groups + threads - 1) / threads);
    qwen_swiglu_quant_kernel<<<blocks, threads, 0, stream>>>(a, b, y, packed,
                                                             scales, n);
    return cudaGetLastError();
}

// ── Fused linear-attention (GatedDeltaNet) front-end entry ──────────────────
// Fused F1 + F2 replace the split front of RunAttnLinearAttention; the
// gated-delta recurrence, RMSNormGated and out_proj tail are unchanged.

// Decode fast path (audit F-05, measured): the block-per-token design below
// collapses to a single 128-thread block at seq==1 and costs ~30 ms per layer
// in the qkv projection alone. These variants assign ONE WARP per output
// column (grid scales with n_out), lanes striding K -- coalesced nibble reads
// plus a shfl reduction. Accumulation order differs from the scalar path;
// Qwen numerics are tolerance-class, and run-to-run determinism is kept.
__global__ void warpdot_nvfp4_kernel(
    const __half *__restrict__ x,           // [1, H] (single token row)
    const uint8_t *__restrict__ packed,
    const uint8_t *__restrict__ scales,
    float global_scale, int H, int n_out,
    __half *__restrict__ out)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 8 + warp;
    if (c >= n_out) return;
    const int64_t cbase = static_cast<int64_t>(c) * H;
    float acc = 0.0f;
    for (int k = lane; k < H; k += 32) {
        const int64_t e = cbase + k;
        const uint8_t byte = packed[e >> 1];
        const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
        const float scl = E4M3ScaleToFloat(scales[e >> 4]) * global_scale;
        acc += __half2float(x[k]) *
               __half2float(__float2half(E2M1Value(nib) * scl));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[c] = __float2half(acc);
}

// conv1d+silu+l2norm(q,k) only (the z/a/b dots moved to warpdot launches).
__global__ void fused_linear_conv_norm_kernel(
    const __half *__restrict__ d_qkv,
    const __half *__restrict__ conv_w,
    int seq, int nk, int kd, int nv, int vd, int kernel, int kv_base,
    __half *__restrict__ d_conv)
{
    const int t = blockIdx.x;
    const int S = nk * kd * 2 + nv * vd;
    const int KD = nk * kd;
    for (int c = threadIdx.x; c < S; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int tt = kv_base + t - k;
            if (tt < 0) continue;
            const float wv =
                __half2float(conv_w[(int64_t)c * kernel + (kernel - 1 - k)]);
            const float mv = __half2float(d_qkv[(int64_t)tt * S + c]);
            acc += wv * mv;
        }
        d_conv[(int64_t)t * S + c] = __float2half(silu_dev(acc));
    }
    __syncthreads();

    __shared__ float svals[256], sred[256], sinv;
    const float q_scale = 1.0f / sqrtf(static_cast<float>(kd));
    for (int h = 0; h < nk; ++h) {
        for (int slice = 0; slice < 2; ++slice) {
            const int head_off = (slice == 0) ? 0 : KD;
            const float extra = (slice == 0) ? q_scale : 1.0f;
            const int lane = threadIdx.x;
            const int c = head_off + h * kd + lane;
            float v = 0.0f;
            if (lane < kd) v = __half2float(d_conv[(int64_t)t * S + c]);
            svals[lane] = v; sred[lane] = v * v;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (lane < s) sred[lane] += sred[lane + s];
                __syncthreads();
            }
            if (lane == 0) sinv = rsqrtf(sred[0] + 1e-6f) * extra;
            __syncthreads();
            if (lane < kd)
                d_conv[(int64_t)t * S + c] = __float2half(svals[lane] * sinv);
            __syncthreads();
        }
    }
}

// Decode split of fused_linear_conv_norm_kernel: the monolithic variant runs
// the whole S-column conv and all 2*nk head norms in ONE 256-thread block.
// Column-parallel conv + block-per-head l2norm keep the exact same
// accumulation order (tree reduce) while filling the SM array.
__global__ void conv_silu_cols_kernel(const __half *__restrict__ d_qkv,
                                      const __half *__restrict__ conv_w,
                                      int S, int kernel, int kv_base,
                                      __half *__restrict__ d_conv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= S) return;
    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        const int tt = kv_base - k;   // decode: t == 0
        if (tt < 0) continue;
        const float wv =
            __half2float(conv_w[(int64_t)c * kernel + (kernel - 1 - k)]);
        const float mv = __half2float(d_qkv[(int64_t)tt * S + c]);
        acc += wv * mv;
    }
    d_conv[c] = __float2half(silu_dev(acc));
}

__global__ void l2norm_qk_heads_kernel(__half *__restrict__ d_conv, int nk,
                                       int kd, int KD, float q_scale)
{
    const bool is_q = blockIdx.x < (unsigned)nk;
    const int h = is_q ? blockIdx.x : blockIdx.x - nk;
    const int off = (is_q ? 0 : KD) + h * kd;
    const int lane = threadIdx.x;
    __shared__ float sred[256];
    float v = 0.0f;
    if (lane < kd) v = __half2float(d_conv[off + lane]);
    sred[lane] = v * v;
    __syncthreads();
    for (int sz = blockDim.x / 2; sz > 0; sz >>= 1) {
        if (lane < sz) sred[lane] += sred[lane + sz];
        __syncthreads();
    }
    if (lane == 0)
        sred[0] = rsqrtf(sred[0] + 1e-6f) * (is_q ? q_scale : 1.0f);
    __syncthreads();
    if (lane < kd) d_conv[off + lane] = __float2half(v * sred[0]);
}

// gating prep (beta / g) from the precomputed a|b buffer, single-token layout.
__global__ void gating_prep_kernel(const __half *__restrict__ a_buf,
                                   const __half *__restrict__ b_buf,
                                   const float *__restrict__ A_log,
                                   const float *__restrict__ dt_bias,
                                   float *__restrict__ d_g,
                                   float *__restrict__ d_beta, int nv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nv) return;
    const float a_v = __half2float(a_buf[c]);
    const float b_v = __half2float(b_buf[c]);
    d_beta[c] = sigmoid_dev(b_v);
    d_g[c] = -__expf(A_log[c]) * softplus_dev(a_v + dt_bias[c]);
}

// Fused F1 + F2 replace the split front of RunAttnLinearAttention; the
// gated-delta recurrence, RMSNormGated and out_proj tail are unchanged.
cudaError_t RunAttnFusedLinearAttnKernel(
    cudaStream_t stream,
    const attn::LayerWeights &lw, const attn::AttnGeometry &geo, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base /* = 0 */, float *d_gd_state /* = nullptr */,
    bool use_sigmoid_gate /* = false */)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    // A persistent GatedDeltaNet state is only safe for a single-token chunk:
    // for seq > 1 there is no grid barrier between the col_seed read and the
    // t == seq-1 write, so a block could read the state another block already
    // updated (audit X8). Every current pipeline uses seq == 1 here; keep the
    // contract explicit instead of relying on that.
    if (d_gd_state != nullptr && seq > 1) return cudaErrorInvalidValue;
    const int nk = geo.linear_num_key_heads;
    const int kd = geo.linear_key_head_dim;
    const int nv = geo.linear_num_value_heads;
    const int vd = geo.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    if (nk <= 0 || kd <= 0 || nv <= 0 || vd <= 0) return cudaErrorInvalidValue;
    if (kd > 256) return cudaErrorInvalidValue;  // shared arrays sized [256]
    if (vd > 256) return cudaErrorInvalidValue;  // col[256] in delta kernel
    if (lw.qkv.cols <= 0) return cudaErrorInvalidValue;
    cudaError_t e;

    if (seq == 1) {
        // Decode fast path (audit F-05): warp-per-output dots + split
        // conv/norm and gating stages. Grid scales with output count instead
        // of collapsing to one 128-thread block.
        const int H = lw.qkv.cols;          // input dim (K)
        const int S = KD * 2 + VD;
        const bool tc_attn = QwenTcFastMath();
        {
            __half *qkv_row =
                d_qkv + static_cast<int64_t>(kv_base) * S;
            if (tc_attn) {
                // Tensor-core W4A4 route: ONE fused GEMM for
                // qkv|z|a|b (same input row), per-segment global scales
                // applied by the kernel epilogue. Cuts 3 ActQuant passes
                // and 3 launches per layer per decode step.
                const uint8_t *fp = nullptr, *fs = nullptr;
                const float *fw = nullptr;
                const int *fb = nullptr;
                int fn = 0;
                const int VDl = nv * vd;
                e = lsx_tc::LsxTcBuildFusedGemmWeights(
                    stream, lw.qkv.d_packed, lw.qkv.d_scales, S,
                    lw.z.d_packed, lw.z.d_scales, VDl,
                    lw.a.d_packed, lw.a.d_scales, nv,
                    lw.b.d_packed, lw.b.d_scales, nv,
                    H / 2, (H + 15) / 16, lw.qkv.global_scale,
                    lw.z.global_scale, lw.a.global_scale,
                    lw.b.global_scale, &fp, &fs, &fw, &fb, &fn);
                if (e == cudaSuccess) {
                    // Persistent per-weight-set output scratch: the fused
                    // GEMM shape (fn) is stable for the process lifetime,
                    // so the per-layer cudaMallocAsync/FreeAsync pair is
                    // pure driver overhead at decode (~60 calls/step).
                    // Stream-ordered pool would reuse one block anyway.
                    __half *fy = lsx_tc::LsxTcFusedOutputScratch(fp, fn);
                    if (!fy) return cudaErrorMemoryAllocation;
                    e = lsx_tc::RunLsxTcNvfp4Gemm(stream, d_normed, fp, fs,
                                               1.0f, fy, 1, H, fn, false,
                                               fw, fb);
                    if (e == cudaSuccess) {
                        // Split the fused output: qkv stays contiguous;
                        // z | a | b land in their own buffers.
                        e = cudaMemcpyAsync(qkv_row, fy, sizeof(__half) * S,
                                            cudaMemcpyDeviceToDevice,
                                            stream);
                        if (e == cudaSuccess)
                            e = cudaMemcpyAsync(
                                d_z, fy + S, sizeof(__half) * VDl,
                                cudaMemcpyDeviceToDevice, stream);
                        if (e == cudaSuccess)
                            e = cudaMemcpyAsync(
                                d_ab, fy + S + VDl, sizeof(__half) * 2 * nv,
                                cudaMemcpyDeviceToDevice, stream);
                    }
                    if (e != cudaSuccess) return e;
                } else {
                    const int grid = (S + 7) / 8;
                    warpdot_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                        d_normed, lw.qkv.d_packed, lw.qkv.d_scales,
                        lw.qkv.global_scale, H, S, qkv_row);
                    if ((e = cudaGetLastError()) != cudaSuccess) return e;
                }
            } else {
                const int grid = (S + 7) / 8;
                warpdot_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                    d_normed, lw.qkv.d_packed, lw.qkv.d_scales,
                    lw.qkv.global_scale, H, S, qkv_row);
                if ((e = cudaGetLastError()) != cudaSuccess) return e;
            }
        }
        // conv1d+silu+l2norm(q,k): column/head-parallel decode split
        // (bit-exact tree order; fills the SM array at seq==1).
        conv_silu_cols_kernel<<<(S + 255) / 256, 256, 0, stream>>>(
            d_qkv, lw.d_conv1d, S, lw.conv_kernel, kv_base, d_conv);
        e = cudaGetLastError(); if (e) return e;
        l2norm_qk_heads_kernel<<<2 * nk, 256, 0, stream>>>(
            d_conv, nk, kd, KD, 1.0f / sqrtf(static_cast<float>(kd)));
        e = cudaGetLastError(); if (e) return e;
        // z / a / b projections: already produced by the fused qkv|z|a|b
        // GEMM above when tc_attn; scalar fallback below keeps parity.
        {
            const __half *xrow = d_normed;   // t == 0
            __half *a_buf = d_ab;
            __half *b_buf = d_ab + static_cast<int64_t>(seq) * nv;
            if (!tc_attn) {
                int grid = (VD + 7) / 8;
                warpdot_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                    xrow, lw.z.d_packed, lw.z.d_scales,
                    lw.z.global_scale, H, VD, d_z);
                if ((e = cudaGetLastError()) != cudaSuccess) return e;
                grid = (nv + 7) / 8;
                warpdot_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                    xrow, lw.a.d_packed, lw.a.d_scales,
                    lw.a.global_scale, H, nv, a_buf);
                if ((e = cudaGetLastError()) != cudaSuccess) return e;
                warpdot_nvfp4_kernel<<<grid, 256, 0, stream>>>(
                    xrow, lw.b.d_packed, lw.b.d_scales,
                    lw.b.global_scale, H, nv, b_buf);
                if ((e = cudaGetLastError()) != cudaSuccess) return e;
            }
            gating_prep_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(
                d_ab, d_ab + static_cast<int64_t>(seq) * nv,
                lw.d_A_log, lw.d_dt_bias, d_g, d_beta, nv);
            if ((e = cudaGetLastError()) != cudaSuccess) return e;
        }
    } else {
    // Stage A: raw qkv GEMM into the persistent history (own launch — stage B
    // conv reads rows written by other tokens of this call).
    int nthreads = 1;
    while (nthreads < kd) nthreads <<= 1;
    fused_linear_qkv_kernel<<<seq, nthreads, 0, stream>>>(
        d_normed, lw.qkv, seq, nk, kd, nv, vd, lw.qkv.cols, kv_base, d_qkv);
    e = cudaGetLastError(); if (e) return e;

    // Stage B: conv1d+silu+l2norm(q,k) ∥ z+a+b+prep_gating (one launch).
    fused_linear_conv_zab_kernel<<<2 * seq, nthreads, 0, stream>>>(
        d_normed, lw.z, lw.a, lw.b, lw.d_A_log, lw.d_dt_bias, lw.d_conv1d,
        seq, nk, kd, nv, vd, lw.z.cols, lw.conv_kernel, kv_base,
        d_qkv, d_conv, d_z, d_ab, d_g, d_beta);
    e = cudaGetLastError(); if (e) return e;
    }

    // Tail (unchanged, mirrors RunAttnLinearAttention stages 4-6).
    {
        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const int r = nv / nk;
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, S, d_gd_state);
        e = cudaGetLastError(); if (e) return e;
    }
    {
        const int threads = std::min(vd, 256);
        rmsnorm_gated_kernel<<<seq * nv, threads, vd * sizeof(float), stream>>>(
            d_core, d_z, lw.d_attn_norm, d_gated, seq * nv, vd,
            geo.rms_norm_eps, use_sigmoid_gate);
        e = cudaGetLastError(); if (e) return e;
    }
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_gated, lw.out.d_packed, lw.out.d_scales,
                            lw.out.global_scale, d_out, seq, lw.out.cols,
                            lw.out.rows);
}

cudaError_t RunAttnMLP(cudaStream_t stream,
                       const attn::LayerWeights &lw, int seq,
                       const __half *d_in, __half *d_out, __half *d_scratch)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int H = lw.mlp_gate.cols;       // input dim
    const int I = lw.mlp_gate.rows;       // intermediate
    if (H <= 0 || I <= 0) return cudaErrorInvalidValue;

    __half *d_gate = d_scratch;                 // [seq,I]
    __half *d_up   = d_scratch + (int64_t)seq * I;

    cudaError_t e;
    if (!lw.mlp_gate.d_packed) return cudaErrorInvalidValue;
    const int64_t n_act = (int64_t)seq * I;
    const int threads = 256;
    const int blocks = (int)((n_act + threads - 1) / threads);
    if (lw.mlp_gate_up.d_packed) {
        // B1 fast path: one GEMM over the fused [2I,K] weight, then swiglu
        // across the column halves. Scratch holds [seq, 2I] == old gate+up.
        e = LaunchNvfp4Gemm(lw.mlp_gate_up, d_in, seq, d_gate, stream);
        if (e) return e;
        qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(
            d_gate, d_up, d_gate, n_act);
        e = cudaGetLastError(); if (e) return e;
    } else {
        e = LaunchNvfp4Gemm(lw.mlp_gate, d_in, seq, d_gate, stream); if (e) return e;
        e = LaunchNvfp4Gemm(lw.mlp_up,   d_in, seq, d_up, stream);   if (e) return e;
        qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_gate, n_act);
        e = cudaGetLastError(); if (e) return e;
    }
    e = LaunchNvfp4Gemm(lw.mlp_down, d_gate, seq, d_out, stream);
    return e;
}

// Fused MLP layer. Takes d_in as the ALREADY post-RMSNormed input stream and
// accumulates the down-projection INTO d_resid (the residual stream) via the
// GEMM's acc=true path, absorbing the separate qwen_add residual. This mirrors
// RunAttnMLP except the final down-GEMM writes resid (d_resid) rather than a
// separate output buffer that the caller then adds. `top` is reserved for
// future use (geometry is read from `lw`).
cudaError_t RunAttnFusedMLP(cudaStream_t stream, const attn::LayerWeights &lw,
                            const attn::AttnGeometry &geo, int seq,
                            const __half *d_in, __half *d_resid, __half *d_scratch)
{
    (void)geo;
    if (!lw.loaded || !d_in || !d_resid || !d_scratch || seq <= 0)
        return cudaErrorInvalidValue;
    const int H = lw.mlp_gate.cols;   // input hidden dim
    const int I = lw.mlp_gate.rows;   // intermediate
    if (H <= 0 || I <= 0) return cudaErrorInvalidValue;
    if (!lw.mlp_gate.d_packed || !lw.mlp_up.d_packed || !lw.mlp_down.d_packed)
        return cudaErrorInvalidValue;

    __half *d_gate = d_scratch;                // [seq,I]
    __half *d_up   = d_scratch + (int64_t)seq * I;

    cudaError_t e;
    const int64_t n_act = (int64_t)seq * I;
    const int threads = 256;
    const int blocks = (int)((n_act + threads - 1) / threads);
    if (lw.mlp_gate_up.d_packed) {
        // B1 fast path: single fused-weight GEMM into [seq, 2I].
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_in, lw.mlp_gate_up.d_packed,
                             lw.mlp_gate_up.d_scales,
                             lw.mlp_gate_up.global_scale, d_gate, seq, H, 2 * I,
                             /*acc=*/false);
        if (e) return e;

        // Producer-fused quant: swiglu emits the NVFP4 form of d_gate so
        // the down GEMM skips its standalone ActQuant pass (TC only).
        // 27B decode profile: when the down projection will actually route
        // through the warp-per-row GEMV (K%64==0, deep-K large-N policy
        // active), skipping PreQuant lets the plain GEMV handle both GEMMs
        // and wins. For shallower shapes (e.g. qwen3.5 down K=512 N=2048)
        // the PreQuant TC branch stays — the sliced GEMV loses to TC there
        // (qwen3.5 139 -> 130 tok/s A/B, bisected 2026-09-13).
        const bool down_warprow =
            QwenGemmLargeActive() && seq == 1 && (I % 64) == 0 &&
            H >= 2048 && I >= 4096;
        // The prequant kernel consumes d_gate and overwrites it with the
        // activated value. If the TC GEMM then fails, re-running SwiGLU on
        // d_gate would apply silu twice (audit X12), so only enter this branch
        // when the TC path is known to exist and return the GEMM error rather
        // than falling through.
        if (!down_warprow && QwenTcFastMath() && (I % 16) == 0 &&
            lsx_tc::LsxTcRuntimeSupported()) {
            const size_t qb = static_cast<size_t>(seq) * I / 2;
            const size_t sb2 = static_cast<size_t>(seq) * ((I + 15) / 16);
            uint8_t *ap = nullptr, *as = nullptr;
            cudaError_t e2 = cudaMallocAsync(&ap, qb ? qb : 1, stream);
            if (e2 == cudaSuccess)
                e2 = cudaMallocAsync(&as, sb2 ? sb2 : 1, stream);
            if (e2 != cudaSuccess) return e2;
            qwen_swiglu_quant_kernel<<<blocks, threads, 0, stream>>>(
                d_gate, d_up, d_gate, ap, as, n_act);
            e2 = cudaGetLastError();
            if (e2 == cudaSuccess)
                e2 = lsx_tc::RunLsxTcGemmPreQuant(
                    stream, ap, as, lw.mlp_down.d_packed,
                    lw.mlp_down.d_scales, lw.mlp_down.global_scale, d_resid,
                    seq, I, H, /*acc=*/true);
            cudaFreeAsync(ap, stream);
            cudaFreeAsync(as, stream);
            return e2;
        }
        qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(
            d_gate, d_up, d_gate, n_act);
        e = cudaGetLastError();
        if (e) return e;
    } else {
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_in, lw.mlp_gate.d_packed, lw.mlp_gate.d_scales,
                             lw.mlp_gate.global_scale, d_gate, seq, H, I,
                             /*acc=*/false);
        if (e) return e;
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_in, lw.mlp_up.d_packed, lw.mlp_up.d_scales,
                             lw.mlp_up.global_scale, d_up, seq, H, I,
                             /*acc=*/false);
        if (e) return e;
        qwen_swiglu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, d_gate, n_act);
        e = cudaGetLastError();
        if (e) return e;
    }
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_gate, lw.mlp_down.d_packed,
                            lw.mlp_down.d_scales, lw.mlp_down.global_scale,
                            d_resid, seq, I, H, /*acc=*/true);
}

// ── Full attention layer. ───────────────────────────────────────────────────
// scratch (halves), sized by caller (see generation/arena):
//   d_qc  : [seq, NH*HD*2]  raw q_proj output (query + gate)
//   d_q   : [seq, NH*HD]    query (post-norm, post-rope)
//   d_gate: [seq, NH*HD]    output gate
//   d_k   : [seq, NKV*HD]   key (post-norm, post-rope)
//   d_v   : [seq, NKV*HD]   value
//   d_attn: [seq, NH*HD]    attention output (pre-o_proj)
//   d_cos : [seq, rotary_dim]
//   d_sin : [seq, rotary_dim]
cudaError_t RunAttnFullAttention(cudaStream_t stream,
                                 const attn::LayerWeights &lw,
                                 const attn::AttnGeometry &geo, int seq,
                                 const __half *d_normed, __half *d_out,
                                 __half *d_qc, __half *d_q, __half *d_gate,
                                 __half *d_k, __half *d_v, __half *d_attn,
                                 __half *d_cos, __half *d_sin,
                                 int32_t *d_pos, int kv_base)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int NH = geo.num_attention_heads;
    const int HD = geo.head_dim;
    const int NKV = geo.num_key_value_heads;
    const int H = geo.hidden_size;
    const int rotary_dim = (int)(HD * geo.partial_rotary_factor);
    if (NH <= 0 || HD <= 0 || NKV <= 0 || H <= 0) return cudaErrorInvalidValue;

    // kv rows are appended at [kv_base..kv_base+seq) when kv_base > 0
    // (persistent cache) or [0..seq) when kv_base == 0 (transient arena).
    __half *k_row0 = d_k + (int64_t)kv_base * NKV * HD;
    __half *v_row0 = d_v + (int64_t)kv_base * NKV * HD;

    cudaError_t e;
    // 1. q_proj: [seq, H] -> [seq, NH*HD*2].
    e = LaunchNvfp4Gemm(lw.q, d_normed, seq, d_qc, stream); if (e) return e;
    // 2. k,v projections.
    e = LaunchNvfp4Gemm(lw.k, d_normed, seq, k_row0, stream);  if (e) return e;
    e = LaunchNvfp4Gemm(lw.v, d_normed, seq, v_row0, stream);  if (e) return e;

    // 3. Split query + gate, reshape to [seq, NH, HD], per-head RMSNorm(q).
    {
        const int threads = std::min(HD, 512);
        qwen_split_query_kernel<<<seq * NH, threads, HD * sizeof(float),
                                  stream>>>(
            d_qc, d_gate, d_q, nullptr, seq, NH, HD, lw.d_q_norm,
            lw.d_k_norm, (float)geo.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 4. per-head RMSNorm(k) into k_row0 itself.
    {
        const int threads = std::min(HD, 512);
        qwen_head_norm_kernel<<<seq * NKV, threads, HD * sizeof(float),
                                  stream>>>(
            k_row0, lw.d_k_norm, k_row0, seq, NKV, HD, (float)geo.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 5. RoPE on q,k (first rotary_dim dims). k has NKV heads, q has NH heads.
    e = RunAttnRopeCosSin(stream, d_pos, seq, rotary_dim, geo.rope_theta,
                          d_cos, d_sin); if (e) return e;
    e = RunAttnPartialRope(stream, d_q, k_row0, d_cos, d_sin, d_q, k_row0,
                           seq, NH, NKV, HD, rotary_dim); if (e) return e;

    // 6. Scaled dot-product attention (GQA, causal) -> d_attn [seq, NH, HD].
    {
        const float scale = 1.0f / sqrtf((float)HD);
        const dim3 grid(seq, NH);
        const int threads = std::min(HD, 512);
        // Decode with a grown context: the legacy single-block kernel is
        // memory-latency bound (one block per head) and its cost grows
        // linearly with kv_base; the split-KV path keeps it flat. Short
        // contexts keep the legacy kernel (bit-identical established
        // behavior, incl. every parity/selfcheck fixture).
        if (seq == 1 && kv_base + 1 > 512) {
            e = RunAttnFullAttnKernelSplitKV(stream, d_q, d_k, d_v, d_attn,
                                             seq, NH, HD, NKV, scale,
                                             kv_base);
            if (e) return e;
        } else {
            // shared holds up to (kv_base+seq+1) logits; cap at threads to
            // bound memory.
            const size_t smem =
                (static_cast<size_t>(kv_base) + static_cast<size_t>(seq) + 1) *
                sizeof(float);
            full_attn_softmax_kernel<<<grid, threads, smem, stream>>>(
                d_q, d_k, d_v, d_attn, seq, NH, HD, NKV, scale, kv_base);
            e = cudaGetLastError(); if (e) return e;
        }
    }

    // 7. Output gate: attn * sigmoid(gate).
    {
        const int64_t n = (int64_t)seq * NH * HD;
        const int threads = 256;
        qwen_out_gate_kernel<<<(int)((n + threads - 1) / threads), threads,
                               0, stream>>>(d_attn, d_gate, d_attn, seq, NH * HD);
        e = cudaGetLastError(); if (e) return e;
    }

    // 8. o_proj: [seq, NH*HD] -> [seq, H].
    e = LaunchNvfp4Gemm(lw.o, d_attn, seq, d_out, stream);
    return e;
}

// ── Linear attention (GatedDeltaNet) layer ─────────────────────────────────
// scratch (halves unless noted):
//   d_qkv : [seq, KD*2+VD]  in_proj_qkv output
//   d_z   : [seq, VD]       in_proj_z output (RMSNormGated gate)
//   d_g   : [seq, NVR] float decay (-exp(A)*softplus(a+dt))  [reused as fp32]
//   d_beta: [seq, NVR] float beta = sigmoid(b)
//   d_conv: [seq, KD*2+VD]  conv1d output (= qkv post-conv)
//   d_core: [seq, NVR*VD]   gated-delta output (pre-RMSNormGated)
//   d_gated:[seq, NVR*VD]   RMSNormGated output
// NVR = linear_num_value_heads.
cudaError_t RunAttnLinearAttention(cudaStream_t stream,
                                   const attn::LayerWeights &lw,
                                   const attn::AttnGeometry &geo, int seq,
                                   const __half *d_normed, __half *d_out,
                                   __half *d_qkv, __half *d_z, __half *d_ab,
                                   float *d_g, float *d_beta,
                                   __half *d_conv, __half *d_core,
                                   __half *d_gated, int kv_base,
                                   float *d_gd_state)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    // A persistent GatedDeltaNet state is only safe for a single-token chunk:
    // for seq > 1 there is no grid barrier between the col_seed read and the
    // t == seq-1 write, so a block could read the state another block already
    // updated (audit X8). Every current pipeline uses seq == 1 here; keep the
    // contract explicit instead of relying on that.
    if (d_gd_state != nullptr && seq > 1) return cudaErrorInvalidValue;
    const int nk = geo.linear_num_key_heads;
    const int kd = geo.linear_key_head_dim;
    const int nv = geo.linear_num_value_heads;
    const int vd = geo.linear_value_head_dim;
    const int KD = nk * kd;
    const int VD = nv * vd;
    const int S = KD * 2 + VD;
    if (nk <= 0 || kd <= 0 || nv <= 0 || vd <= 0) return cudaErrorInvalidValue;

    cudaError_t e;
    // 1. qkv, z projections via NVFP4 GEMM. For incremental decode `d_qkv`
    //    points at the layer's persistent cache base (rows [kv_base..kv_base+seq)
    //    of the cache); the conv1d below then reads the full history.
    e = LaunchNvfp4Gemm(lw.qkv, d_normed, seq,
                        d_qkv + (int64_t)kv_base * S, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.z,   d_normed, seq, d_z,   stream);   if (e) return e;

    // 2. conv1d over S channels (reads history rows [kv_base-kernel+1..kv_base+seq)).
    if (lw.d_conv1d && lw.conv_dim == S) {
        const int threads = 128;
        qwen_conv1d_kernel<<<(S + threads - 1) / threads, threads, 0, stream>>>(
            d_qkv, lw.d_conv1d, d_conv, seq, S, lw.conv_kernel, kv_base);
        e = cudaGetLastError(); if (e) return e;
    } else {
        cudaMemcpyAsync(d_conv, d_qkv + (int64_t)kv_base * S,
                        (int64_t)seq * S * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // 3. a,b projections into d_ab [seq*nv*2].
    __half *a_buf = d_ab;
    __half *b_buf = d_ab + (int64_t)seq * nv;
    e = LaunchNvfp4Gemm(lw.a, d_normed, seq, a_buf, stream); if (e) return e;
    e = LaunchNvfp4Gemm(lw.b, d_normed, seq, b_buf, stream); if (e) return e;
    {
        const int threads = 256;
        const int64_t total = (int64_t)seq * nv;
        qwen_prep_gating_kernel<<<(int)((total + threads - 1) / threads), threads,
                                  0, stream>>>(
            a_buf, b_buf, lw.d_A_log, lw.d_dt_bias, d_g, d_beta, total, nv);
        e = cudaGetLastError(); if (e) return e;
    }

    // 4. l2-normalize q,k rows in place (HF: l2norm eps=1e-6, q scaled by
    //    1/sqrt(kd)), then run the gated delta rule. q/k/v are slices of the
    //    strided conv buffer [seq, S], so the recurrence must use stride S.
    {
        if (kd > 256) return cudaErrorInvalidValue;
        if (vd > 256) return cudaErrorInvalidValue;  // col[256] in delta kernel
        int nthreads = 1;
        while (nthreads < kd) nthreads <<= 1;
        const float q_scale = 1.0f / sqrtf((float)kd);
        const dim3 norm_grid(seq, nk);
        qwen_l2norm_rows_kernel<<<norm_grid, nthreads, 0, stream>>>(
            d_conv, seq, nk, kd, S, 0, q_scale);
        e = cudaGetLastError(); if (e) return e;
        qwen_l2norm_rows_kernel<<<norm_grid, nthreads, 0, stream>>>(
            d_conv, seq, nk, kd, S, KD, 1.0f);
        e = cudaGetLastError(); if (e) return e;

        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const int r = nv / nk;
        // one block per (t, v_head)
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, S, d_gd_state);
        e = cudaGetLastError(); if (e) return e;
    }

    // 5. RMSNormGated(core, z, norm) -> d_gated.
    {
        const int threads = std::min(vd, 256);
        rmsnorm_gated_kernel<<<seq * nv, threads, vd * sizeof(float), stream>>>(
            d_core, d_z, lw.d_attn_norm, d_gated, seq * nv, vd,
            geo.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }

    // 6. out_proj: [seq, VD] -> [seq, H].
    e = LaunchNvfp4Gemm(lw.out, d_gated, seq, d_out, stream);
    return e;
}

} // namespace qwen
} // namespace lsxcommon
