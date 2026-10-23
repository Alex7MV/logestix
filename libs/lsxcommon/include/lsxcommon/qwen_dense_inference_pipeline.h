#pragma once
// Qwen-3.x generation driver. Model-owned: RunQwenGeneration takes
// QwenPinnedWeights, so it stays family-named even though the attention/MLP
// kernels and the KV session it uses are the neutral layer
// (lsx_attn_inference.h: attn::KvCache, attn::GenerationResult).
//
// Named "dense" to satisfy the repo file-name scheme (test_code_naming_guard),
// not because it is dense-specific: the hybrid linear+full-attention KvCache
// is also used by the MoE families, which include this header.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/lsx_attn_inference.h"        // kernel entry points
#include "lsxcommon/qwen_inference_load.h"   // QwenPinnedWeights
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace qwen {

// Neutral session structs (lsx_attn_inference.h): attn::KvCache owns the
// hybrid KV state, attn::GenerationResult the driver's token/logit output.
// Family aliases keep the qwen-family call sites readable without defining
// either type here.
using KvCache = attn::KvCache;
using GenerationResult = attn::GenerationResult;

// Full greedy multi-token generation over the frozen Qwen3.5-9B hybrid loop:
// per decoder layer, RMSNorm input -> (linear or full attention) -> residual ->
// RMSNorm -> MLP -> residual; final norm -> lm_head -> greedy argmax.
// A non-null `kv` selects the incremental path (spec item 15): each step runs
// exactly one token at the persistent kv_base and the recurrent GatedDeltaNet
// state is carried across calls; `nullptr` keeps today's full-recompute path.
cudaError_t RunQwenGeneration(
    cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    int eos_id,
    GenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    KvCache *kv = nullptr,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace qwen
} // namespace lsxcommon
