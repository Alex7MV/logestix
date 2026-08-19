#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace sample {

namespace {

template <typename T>
__global__ void kGreedyArgmax(const T* __restrict__ logits, int vocab,
                              int32_t* out_tok, float* out_maxv) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    float best = -1e30f;
    int32_t best_i = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = static_cast<float>(logits[i]);
        if (v > best) { best = v; best_i = i; }
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
__global__ void kGreedyArgmaxF16Rounded(const float* __restrict__ logits,
                                        int vocab, int32_t* out_tok,
                                        float* out_maxv) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    float best = -1e30f;
    int32_t best_i = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = __half2float(__float2half(logits[i]));
        if (v > best) { best = v; best_i = i; }
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

}  // namespace

cudaError_t GreedySampler::Sample(cudaStream_t stream, const void* d_logits,
                                  int vocab, LogitType dtype,
                                  int32_t* d_out_token, float* d_out_maxv) {
    if (!d_logits || !d_out_token || !d_out_maxv || vocab <= 0)
        return cudaErrorInvalidValue;
    constexpr int kThreads = 256;
    if (dtype == LogitType::kF32) {
        kGreedyArgmax<float><<<1, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, d_out_token, d_out_maxv);
    } else if (dtype == LogitType::kF16) {
        kGreedyArgmax<__half><<<1, kThreads, 0, stream>>>(
            static_cast<const __half*>(d_logits), vocab, d_out_token, d_out_maxv);
    } else if (dtype == LogitType::kF32Rounded16) {
        kGreedyArgmaxF16Rounded<<<1, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, d_out_token, d_out_maxv);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

ISampler& DefaultSampler() {
    static GreedySampler g;
    return g;
}

}  // namespace sample
}  // namespace lsxcommon