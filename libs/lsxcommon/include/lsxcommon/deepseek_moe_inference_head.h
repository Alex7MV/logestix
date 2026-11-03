#pragma once

// Shared lm_head entry point of the DeepSeek-V4 family (V4-Flash, V4.1-Flash
// and V4-Pro-0813 all run the same FP16 logits GEMV + optional fused argmax).
//
// The kernel lives in deepseek_moe_inference_head.cu; the declaration is a
// header because three pipelines forward-declared it by hand and the
// V4-Pro copies were pure namespace duplicates (dedup phase 2, task 2).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace gpu {

// d_next_token == nullptr skips the fused argmax entirely (a sampler may run
// its own reduction over d_logits right after).
cudaError_t run_lm_head(cudaStream_t stream, const __half *d_head_weight,
                        const __half *d_hidden, int vocab_size, int hidden_dim,
                        float *d_logits, int32_t *d_next_token,
                        float *d_next_maxv);

} // namespace gpu
} // namespace lsxcommon
