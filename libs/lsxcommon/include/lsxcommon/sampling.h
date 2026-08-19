#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Isolated sampling seam (spec item 13). HOST + CUDA safe: no absl containers,
// no device-side state. Engines keep their fused argmax kernels; GreedySampler
// is the default wrapper where a TopK/TopP sampler later reads the full row.
namespace lsxcommon {
namespace sample {

enum class LogitType {
    kF32 = 0,   // float logits rows (GLM d_logits_head, DeepSeek d_logits)
    kF16 = 1,   // __half logits rows (Qwen d_logits)
};

class ISampler {
public:
    virtual ~ISampler() = default;

    // Selects one token id from a [vocab] logits row on `stream`.
    //   d_logits    device pointer to the row (FP32 or FP16 per @p dtype).
    //   d_out_token device int32 slot receiving the selected id.
    //   d_out_maxv  device float slot receiving the raw max logit.
    // Returns cudaSuccess or a CUDA error code; never throws.
    virtual cudaError_t Sample(cudaStream_t stream, const void* d_logits,
                               int vocab, LogitType dtype,
                               int32_t* d_out_token, float* d_out_maxv) = 0;
};

// Greedy argmax sampler. Matches the engines' fused-argmax semantics: on ties
// the FIRST (lowest index) maximum wins. Single block, <= 256 threads.
class GreedySampler : public ISampler {
public:
    cudaError_t Sample(cudaStream_t stream, const void* d_logits, int vocab,
                       LogitType dtype, int32_t* d_out_token,
                       float* d_out_maxv) override;
};

// Function-local static default instance (stable address for default args).
ISampler& DefaultSampler();

}  // namespace sample
}  // namespace lsxcommon