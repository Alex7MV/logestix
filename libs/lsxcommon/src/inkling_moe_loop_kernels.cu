// Inkling-2-Mini loop kernels: the MessageGate controller and the cross-pass
// message retrieval. The rest of the inkling2 arithmetic is shared with Small
// (geometry-parameterized entry points in inkling_moe_kernels.h).
//
// MessageGate (the reference's Inkling2LoopMessageGate):
//   g_in = concat(unweighted_rmsnorm(h), unweighted_rmsnorm(msg))  [2*hidden]
//   g    = down @ g_in                       (down FP16, FP32 accumulate)
//   out  = (1 - tanh(forget @ g)) * h + tanh(gain @ g) * msg
// forget/gain are FP32 [hidden, rank]; the blend is FP32. Deterministic: one
// block per output row, fixed-order reduction, no atomicAdd.
#include "lsxcommon/inkling_moe_loop_kernels.h"
#include "lsxcommon/inkling_moe_kernels.h"

#include <cmath>

namespace lsxcommon::inkling {
namespace {

// One block computes the whole gate vector g[rank]: each thread accumulates a
// strided subset of the 2*hidden inputs for every rank row, then a shared
// reduction sums the partials in a FIXED order (thread 0 only RMW behind
// __syncthreads — the shared-memory sandbox rule).
__global__ void message_gate_down_kernel(const float *h, const float *msg,
                                         const __half *down, float *g,
                                         int hidden, int rank, float eps)
{
    extern __shared__ float smem[];
    float *norm_h = smem;              // [hidden]
    float *norm_m = smem + hidden;     // [hidden]
    float *partial = smem + 2 * hidden;  // [blockDim.x] per-row scratch

    const int t = threadIdx.x;
    const int n = blockDim.x;

    // Unweighted RMSNorm of h and msg (FP32), computed cooperatively.
    float ss_h = 0.0f, ss_m = 0.0f;
    for (int i = t; i < hidden; i += n) {
        ss_h += h[i] * h[i];
        ss_m += msg[i] * msg[i];
    }
    partial[t] = ss_h;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += partial[i];
        partial[0] = rsqrtf(s / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const float inv_h = partial[0];
    __syncthreads();
    partial[t] = ss_m;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += partial[i];
        partial[0] = rsqrtf(s / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const float inv_m = partial[0];
    __syncthreads();
    for (int i = t; i < hidden; i += n) {
        norm_h[i] = h[i] * inv_h;
        norm_m[i] = msg[i] * inv_m;
    }
    __syncthreads();

    // g[r] = sum_c down[r][c] * concat[c], FP32 accumulate over FP16 weights.
    for (int r = 0; r < rank; ++r) {
        float acc = 0.0f;
        const __half *row = down + static_cast<int64_t>(r) * (2 * hidden);
        for (int c = t; c < 2 * hidden; c += n) {
            const float x = c < hidden ? norm_h[c] : norm_m[c - hidden];
            acc += __half2float(row[c]) * x;
        }
        partial[t] = acc;
        __syncthreads();
        if (t == 0) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) s += partial[i];
            g[r] = s;
        }
        __syncthreads();
    }
}

// out[i] = (1 - tanh(forget[i] . g)) * h[i] + tanh(gain[i] . g) * msg[i].
// One thread per output channel; the rank-dot is a fixed sequential sum.
__global__ void message_gate_blend_kernel(const float *h, const float *msg,
                                          const float *g, const float *forget,
                                          const float *gain, float *out,
                                          int hidden, int rank)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    float f = 0.0f, gn = 0.0f;
    const float *fr = forget + static_cast<int64_t>(i) * rank;
    const float *gr = gain + static_cast<int64_t>(i) * rank;
    for (int r = 0; r < rank; ++r) {
        f += fr[r] * g[r];
        gn += gr[r] * g[r];
    }
    out[i] = (1.0f - tanhf(f)) * h[i] + tanhf(gn) * msg[i];
}

}  // namespace

cudaError_t RunInklingLoopMessageGate(cudaStream_t stream, const float *h,
                                      const float *msg, const __half *down,
                                      const float *forget, const float *gain,
                                      float *out, int hidden, int rank,
                                      float eps, float *concat_scratch,
                                      float *rank_scratch)
{
    if (h == nullptr || msg == nullptr || down == nullptr ||
        forget == nullptr || gain == nullptr || out == nullptr ||
        rank_scratch == nullptr || hidden <= 0 || rank <= 0)
        return cudaErrorInvalidValue;
    (void)concat_scratch;  // the norms live in the down kernel's shared memory
    const int threads = 128;
    const size_t smem =
        static_cast<size_t>(2 * hidden + threads) * sizeof(float);
    message_gate_down_kernel<<<1, threads, smem, stream>>>(h, msg, down,
                                                         rank_scratch, hidden,
                                                         rank, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    const int blocks = (hidden + 127) / 128;
    message_gate_blend_kernel<<<blocks, 128, 0, stream>>>(
        h, msg, rank_scratch, forget, gain, out, hidden, rank);
    return cudaGetLastError();
}

cudaError_t RunInklingLoopMessageMix(cudaStream_t stream, const float *sources,
                                     int n_src, const float *queries,
                                     float *out, int hidden, float eps)
{
    // The message retrieval is the AttnRes depth-mix arithmetic with four
    // queries: unweighted RMSNorm on every source row, softmax over sources
    // with scale hidden**-0.5, normalized sources on both sides.
    return RunInklingDepthMix(stream, sources, n_src, queries, 4, out, hidden,
                              eps);
}

}  // namespace lsxcommon::inkling