#pragma once

// Kimi-K3 generation driver: one forward per token (token-by-token prefill and
// decode, correctness first), the AttnRes block bookkeeping, the streamed
// latent/shared plane staged per layer, and the routed-expert store.
//
// The session state is the latent KV cache (gated NoPE MLA; the value is
// absorbed into the output projection, so only the 576-wide latent per
// position is cached) plus the KDA recurrent state and conv history. Nothing is
// published outside the caller's residency lock.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/kimi_moe_k3_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::kimi_k3 {

struct KimiK3GenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    std::vector<float> final_logits;
    int32_t first_token = -1;
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
};

// Maximum context this driver allocates for (prompt + generated). Overflow is a
// clean error, never a silent wrap.
inline constexpr int kDefaultMaxContext = 4096;

cudaError_t RunKimiK3Generation(cudaStream_t stream, KimiK3PinnedWeights &pw,
                                KimiK3ExpertDb &experts,
                                const std::vector<int32_t> &prompt, int max_new,
                                int eos_id, int max_context,
                                sample::ISampler &sampler,
                                KimiK3GenerationResult &result,
                                GenerationStream *gs);

}  // namespace lsxcommon::kimi_k3
