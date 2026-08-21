#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace sample {

namespace {

// Fixed device-side partial buffers for the two-phase greedy argmax.
// Phase 1: up to kMaxPartials blocks each produce a (value, index) partial;
// Phase 2: one block combines them in ascending block order. The tie-break
// ("smaller index wins") composes correctly with strict-> partial maxima and
// keeps results bit-identical to the old single-block kernel.
constexpr int kMaxPartials = 512;
__device__ float g_greedy_val[kMaxPartials];
__device__ int32_t g_greedy_idx[kMaxPartials];

// Phase-1 kernel: one partial (value, index) per block into the fixed
// device buffers (indexed by blockIdx.x — no atomics, fully deterministic).
template <typename T>
__global__ void kGreedyPartial(const T* __restrict__ logits, int vocab,
                               int items_per_thread) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    const int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x
                          * items_per_thread;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items_per_thread);

    float best = -1e30f;
    int32_t best_i = -1;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = static_cast<float>(logits[i]);
        if (v > best) { best = v; best_i = static_cast<int32_t>(i); }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        g_greedy_val[blockIdx.x] = s_best[0];
        g_greedy_idx[blockIdx.x] = s_idx[0];
    }
}

__global__ void kGreedyFinalize(int num_partials, int32_t* out_tok,
                                float* out_maxv) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    float best = -1e30f;
    int32_t best_i = -1;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        const float v = g_greedy_val[i];
        const int32_t vi = g_greedy_idx[i];
        if (v > best || (v == best && vi < best_i)) { best = v; best_i = vi; }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_tok = s_idx[0];
        *out_maxv = s_best[0];
    }
}

// GLM's fused argmax semantics: each FP32 logit is rounded to FP16 (RNE) before
// comparing — a bit-exact mirror of the host baseline's FP16-stored lm_head row.
// maxv stores the rounded value, exactly as GlmArgmaxKernel does.
__global__ void kGreedyPartialRounded16(const float* __restrict__ logits,
                                        int vocab, int items_per_thread) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    const int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x
                          * items_per_thread;
    const int64_t end = min(static_cast<int64_t>(vocab),
                            begin + static_cast<int64_t>(blockDim.x) * items_per_thread);

    float best = -1e30f;
    int32_t best_i = -1;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const float v = __half2float(__float2half(logits[i]));
        if (v > best) { best = v; best_i = static_cast<int32_t>(i); }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        g_greedy_val[blockIdx.x] = s_best[0];
        g_greedy_idx[blockIdx.x] = s_idx[0];
    }
}

}  // namespace

cudaError_t GreedySampler::Sample(cudaStream_t stream, const void* d_logits,
                                  int vocab, LogitType dtype,
                                  int32_t* d_out_token, float* d_out_maxv) {
    if (!d_logits || !d_out_token || !d_out_maxv || vocab <= 0)
        return cudaErrorInvalidValue;
    constexpr int kThreads = 256;
    // Two-phase multi-block argmax: phase 1 spreads the vocab scan across
    // ~vocab/(256*32) blocks (vs one block on one SM before), phase 2 combines
    // the partials. Same tie-break semantics as the single-block kernel.
    constexpr int kItemsPerThread = 32;
    const int blocks = static_cast<int>(
        (static_cast<int64_t>(vocab) + kThreads * kItemsPerThread - 1)
        / (kThreads * kItemsPerThread));
    if (blocks > kMaxPartials) return cudaErrorInvalidValue;

    if (dtype == LogitType::kF32) {
        kGreedyPartial<float><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, kItemsPerThread);
    } else if (dtype == LogitType::kF16) {
        kGreedyPartial<__half><<<blocks, kThreads, 0, stream>>>(
            static_cast<const __half*>(d_logits), vocab, kItemsPerThread);
    } else if (dtype == LogitType::kF32Rounded16) {
        // GLM's fused argmax semantics: each FP32 logit is rounded to FP16
        // (RNE) before comparing — a bit-exact mirror of the host baseline's
        // FP16-stored lm_head row. maxv stores the rounded value, exactly as
        // GlmArgmaxKernel does.
        kGreedyPartialRounded16<<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, kItemsPerThread);
    } else {
        return cudaErrorInvalidValue;
    }
    kGreedyFinalize<<<1, kThreads, 0, stream>>>(blocks, d_out_token, d_out_maxv);
    return cudaGetLastError();
}

ISampler& DefaultSampler() {
    static GreedySampler g;
    return g;
}

}  // namespace sample
}  // namespace lsxcommon