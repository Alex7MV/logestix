// Neutral templated MoE generation driver shared by qwen3.5/3.6 and ornith
// (dedup phase 6, Task 7). The ornith pipeline was a 99.4 % rename-copy of
// the qwen pipeline; the only semantic deltas were the abort_stream
// diagnostic flag (qwen-only, compiled out for ornith via AbortStream=false)
// and the fast-math toggle alias (QwenTcFastMath == LsxTcFastMath).
//
// Family headers keep their public names as aliases/inline delegates.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "lsxcommon/lsx_moe_load.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace moe {

// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct MoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

template <class MoeTopology>
cudaError_t EnsureMoeScratch(cudaStream_t stream,
                             const MoePinnedWeights<MoeTopology> &pw,
                             MoeGenScratch &scratch);
cudaError_t ReserveMoeArena(MoeGenScratch &scratch, size_t bytes);
void FreeMoeGenScratch(MoeGenScratch &scratch);

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr). AbortStream is the family policy:
// qwen true (keeps the diagnostic flag), ornith false (never had it).
template <class MoeTopology, bool AbortStream>
cudaError_t RunMoeGeneration(
    cudaStream_t stream, const MoePinnedWeights<MoeTopology> &pw,
    lsxcommon::moe::MoEExpertStore &store, gpu::ExpertDeviceCache &cache,
    MoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::GenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::KvCache *kv,
    const lsxcommon::GenerationStream *gs = nullptr);

} // namespace moe
} // namespace lsxcommon