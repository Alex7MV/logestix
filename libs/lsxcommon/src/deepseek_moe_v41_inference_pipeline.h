// deepseek_moe_v41_inference_pipeline.h — DeepSeek-V4.1-Flash token-by-token
// inference engine entry (implemented in deepseek_moe_v41_inference_pipeline.cu).
//
// Single declaration shared by the model wrapper and the pipeline test so the
// signature can never drift out of sync between them (a stale local
// forward declaration previously omitted the GenerationStream parameter and
// broke the test link).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"
#include "lsxcommon/tokenizer.h"

#include "deepseek_moe_v41_load_internal.h"

namespace lsxcommon {
namespace gpu {

gpu::InferenceResult RunDeepSeekV41Inference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, ::lsxcommon::ArrowTokenizerDb *tok_db,
    const deepseek_v41::gpu::PinnedWeights &pw,
    deepseek_v41::gpu::ExpertDb &expert_db,
    lsxcommon::sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos = false,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace gpu
} // namespace lsxcommon
