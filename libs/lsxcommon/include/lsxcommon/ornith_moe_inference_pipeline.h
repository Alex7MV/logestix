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
#include "lsxcommon/ornith_moe_load.h"      // OrnithMoePinnedWeights
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"       // GenerationResult
#include "lsxcommon/qwen_inference_load.h"  // KvCache
#include "lsxcommon/sampling.h"             // ISampler
#include "lsxcommon/token_stream.h"         // GenerationStream

namespace lsxcommon {
namespace ornith {

// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct OrnithMoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

cudaError_t EnsureOrnithMoeScratch(cudaStream_t stream,
                                 const OrnithMoePinnedWeights &pw,
                                 OrnithMoeGenScratch &scratch);
cudaError_t ReserveOrnithMoeArena(OrnithMoeGenScratch &scratch, size_t bytes);
void FreeOrnithMoeGenScratch(OrnithMoeGenScratch &scratch);

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr); mirrors RunQwenGeneration's per-token
// handshake and final-logits publication.
cudaError_t RunOrnithMoeGeneration(
    cudaStream_t stream, const OrnithMoePinnedWeights &pw,
    moe::MoEExpertStore &store, OrnithMoeExpertDeviceCache &cache,
    OrnithMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::GenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::KvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace ornith
} // namespace lsxcommon
