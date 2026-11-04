#pragma once

// Kimi-K3 generation driver: one forward per token (token-by-token prefill and
// decode, correctness first), the AttnRes block bookkeeping, the streamed
// latent/shared plane staged per layer, and the routed-expert store.
//
// The session state is the latent KV cache (gated NoPE MLA; the value is
// absorbed into the output projection, so only the 576-wide latent per
// position is cached) plus the KDA recurrent state and conv history. On the
// Kimi-K3.1-Pro variant every full-attention layer also keeps its indexer key
// cache (one 128-wide FP16 row per position) and, above `index_topk`, selects
// the visible keys before the MLA attention runs; at or below it the selection
// is bypassed entirely and the driver is the Kimi-K3 driver bit for bit.
// Nothing is published outside the caller's residency lock.

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
    // Gated DSA observables (Kimi-K3.1-Pro only): how many token x layer steps
    // ran a key selection and how many keys those steps selected in total. Both
    // stay 0 on a Kimi-K3 tree and on every prompt at or below `index_topk`,
    // which is the identity path -- so a silently skipped branch is visible
    // from the result alone.
    int64_t dsa_steps = 0;
    int64_t dsa_keys_selected = 0;
    // The LAST engaged step's raw material, copied back once at the end (the
    // device buffers still hold it, so a step costs nothing): the score of
    // every causal position and the exact selection the attention consumed.
    // This is the only place the selection is observable outside the kernels,
    // and it is what a test can check against its own top-k.
    int dsa_last_n_pos = 0;
    std::vector<float> dsa_last_score;         // [dsa_last_n_pos]
    std::vector<int32_t> dsa_last_selection;   // [min(index_topk, n_pos)]
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
