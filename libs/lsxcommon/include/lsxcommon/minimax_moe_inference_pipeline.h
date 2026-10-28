#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/minimax_moe_kernels.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/minimax_moe_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::minimax {

// Persistent incremental session: pos-major KV cache for all 60 layers plus
// the per-layer indexer key history (only sparse layers write it). Session
// state is created and published INSIDE RunUnderResidencyLock (see the model).
struct MinimaxSession {
    __half *d_k = nullptr;      // [layers][capacity][kv_heads][head_dim]
    __half *d_v = nullptr;      // same shape
    __half *d_idx_k = nullptr;  // [layers][capacity][index_head_dim]
    KvCacheGeometry geom;       // pos-major; the layout this engine builds
    int64_t capacity = 0;
    int index_head_dim = 0;     // indexer key row width (frozen 128)
    int64_t kv_rows = 0;        // positions cached so far
    int32_t last_tok = -1;      // last generated token (session continuation)
    std::vector<int32_t> gen_tail;   // last call's generated tokens

    bool allocated() const { return d_k != nullptr; }
    std::int64_t Bytes() const;
    void Free();

    // Grows (never shrinks) the cache to `capacity`; a fresh session allocates.
    cudaError_t EnsureCapacity(const MinimaxTopology &top, int64_t capacity,
                               cudaStream_t stream);
    void Reset();
};

// Per-session device scratch (reused across Infer calls; sized for the max
// sequence of the call and grown on demand).
struct MinimaxScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    int64_t half_elems = 0;
    int64_t float_elems = 0;
    int mask_bytes = 0;
    int64_t seq_capacity = 0;
    int64_t score_floats = 0;
    int64_t logits_elems = 0;
    int32_t *d_ids = nullptr;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    MinimaxRoutedSlot *d_slots = nullptr;
    MinimaxRoutedSlot *h_slots = nullptr;
    float *d_logits = nullptr;   // [vocab] F32 (head is FP16, accumulated F32)
    void Free();
};

struct MinimaxGenResult {
    std::vector<int32_t> tokens;
    std::vector<float> final_logits;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
    int kv_slots_used = 0;
    float first_maxv = 0.0f;
    std::uint64_t resident_bytes = 0;
    std::string error_msg;
};

// End-to-end generation: prompt prefill (token by token) + decode, streaming
// one Emit per decoded token and polling gs->Aborted() between steps.
gpu::InferenceResult RunMinimaxInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const MinimaxPinnedWeights &pw,
    MinimaxExpertStore &experts, gpu::ExpertDeviceCache &cache,
    MinimaxSession &session, MinimaxScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs);

} // namespace lsxcommon::minimax
