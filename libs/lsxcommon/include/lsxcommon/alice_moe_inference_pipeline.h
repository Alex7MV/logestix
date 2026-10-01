#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/alice_moe_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::alice {

struct AliceGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int32_t first_token = -1;
    std::vector<float> final_logits;
    bool eos_terminated = false;
    double ttft_ms = 0.0;   // prefill duration (time to the first token)
    double total_ms = 0.0;  // prefill + decode
};

// Correctness-first greedy generation: each step recomputes the whole sequence
// token-by-token (fresh KV cache + KDA state per step, full O(T^2) attention).
// Routed experts stream from the host store on demand.
cudaError_t RunAliceGeneration(
    cudaStream_t stream, const AlicePinnedWeights &pw, AliceExpertHost &experts,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    AliceGenerationResult &result,
    lsxcommon::sample::ISampler &sampler = lsxcommon::sample::DefaultSampler(),
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace lsxcommon::alice
