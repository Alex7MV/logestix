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

#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"         // GenerationResult
#include "lsxcommon/qwen_inference_load.h"    // KvCache
#include "lsxcommon/qwen_moe_expert_cache.h"  // QwenMoeExpertDeviceCache
#include "lsxcommon/qwen_moe_load.h"
#include "lsxcommon/lsx_moe_inference_pipeline.h"          // QwenMoePinnedWeights
#include "lsxcommon/lsx_moe_inference.h"     // kernel entry points
#include "lsxcommon/sampling.h"               // ISampler
#include "lsxcommon/token_stream.h"           // GenerationStream

namespace lsxcommon {
namespace qwen_moe {

// Dedup Ф6: the scratch struct and the driver are the neutral
// lsxcommon::moe template (lsx_moe_inference_pipeline.h); public names stay.
using QwenMoeGenScratch = lsxcommon::moe::MoeGenScratch;

inline cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                        const QwenMoePinnedWeights &pw,
                                        QwenMoeGenScratch &scratch)
{
    return lsxcommon::moe::EnsureMoeScratch(stream, pw, scratch);
}
inline cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes)
{
    return lsxcommon::moe::ReserveMoeArena(scratch, bytes);
}
inline void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch)
{
    lsxcommon::moe::FreeMoeGenScratch(scratch);
}

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr); mirrors RunQwenGeneration's per-token
// handshake and final-logits publication.
inline cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    lsxcommon::moe::MoEExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::GenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::KvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr)
{
    return lsxcommon::moe::RunMoeGeneration<qwen_moe::QwenMoeTopology, true>(
        stream, pw, store, cache, scratch, prompt, max_new, eos_id, result,
        sampler, kv, gs);
}

} // namespace qwen_moe
} // namespace lsxcommon