#pragma once
// Ornith-1.5-35B-A3B generation pipeline. Model-owned: it takes
// OrnithMoePinnedWeights and the qwen KV cache, so it stays family-named even
// though the routed-MoE kernels it calls are the neutral lsx_moe_inference
// layer (see lsx_moe_inference.h).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>  // __half, used by OrnithMoeGenScratch::d_embed
#include <cuda_runtime.h>

#include "lsxcommon/lsx_moe_inference.h"    // MoEExpertStore, kernel entry points
#include "lsxcommon/ornith_moe_load.h"
#include "lsxcommon/lsx_moe_inference_pipeline.h"      // OrnithMoePinnedWeights
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"       // GenerationResult
#include "lsxcommon/qwen_inference_load.h"  // KvCache
#include "lsxcommon/sampling.h"             // ISampler
#include "lsxcommon/token_stream.h"         // GenerationStream

namespace lsxcommon {
namespace ornith {

// Dedup Ф6: the scratch struct and the driver are the neutral
// lsxcommon::moe template (lsx_moe_inference_pipeline.h); public names stay.
using OrnithMoeGenScratch = lsxcommon::moe::MoeGenScratch;

inline cudaError_t EnsureOrnithMoeScratch(cudaStream_t stream,
                                          const OrnithMoePinnedWeights &pw,
                                          OrnithMoeGenScratch &scratch)
{
    return lsxcommon::moe::EnsureMoeScratch(stream, pw, scratch);
}
inline cudaError_t ReserveOrnithMoeArena(OrnithMoeGenScratch &scratch, size_t bytes)
{
    return lsxcommon::moe::ReserveMoeArena(scratch, bytes);
}
inline void FreeOrnithMoeGenScratch(OrnithMoeGenScratch &scratch)
{
    lsxcommon::moe::FreeMoeGenScratch(scratch);
}

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr). AbortStream=false: ornith never had
// the qwen diagnostic flag; the abort break itself is identical.
inline cudaError_t RunOrnithMoeGeneration(
    cudaStream_t stream, const OrnithMoePinnedWeights &pw,
    moe::MoEExpertStore &store, OrnithMoeExpertDeviceCache &cache,
    OrnithMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::GenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::KvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr)
{
    return lsxcommon::moe::RunMoeGeneration<ornith::OrnithMoeTopology, false>(
        stream, pw, store, cache, scratch, prompt, max_new, eos_id, result,
        sampler, kv, gs);
}

} // namespace ornith
} // namespace lsxcommon
