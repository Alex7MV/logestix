#pragma once
// Qwen-MoE generation pipeline. Model-owned: it takes QwenMoePinnedWeights
// and the qwen KV cache, so it stays family-named even though the kernels it
// calls are a neutral layer (see lsx_moe_inference.h, the neutral
// routed-MoE kernel layer).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>  // __half, used by QwenMoeGenScratch::d_embed
#include <cuda_runtime.h>

#include "lsxcommon/qwen_inference.h"         // QwenGenerationResult
#include "lsxcommon/qwen_inference_load.h"    // QwenKvCache
#include "lsxcommon/qwen_moe_expert_cache.h"  // QwenMoeExpertDeviceCache
#include "lsxcommon/qwen_moe_load.h"          // QwenMoePinnedWeights
#include "lsxcommon/lsx_moe_inference.h"     // kernel entry points
#include "lsxcommon/sampling.h"               // ISampler
#include "lsxcommon/token_stream.h"           // GenerationStream

namespace lsxcommon {
namespace qwen_moe {

// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct QwenMoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch);
cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes);
void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch);

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr); mirrors RunQwenGeneration's per-token
// handshake and final-logits publication.
cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    lsxcommon::moe::MoEExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::QwenKvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace qwen_moe
} // namespace lsxcommon