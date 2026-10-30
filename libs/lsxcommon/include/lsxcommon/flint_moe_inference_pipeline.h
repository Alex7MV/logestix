#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/flint_moe_kernels.h"
#include "lsxcommon/flint_moe_load.h"
#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::flint {

// Persistent incremental session: pos-major KV cache for all 49 layers (the
// sliding layers are full-length too; a 513-row ring is a later memory
// optimisation, not part of ingestion). Session state is created and published
// INSIDE RunUnderResidencyLock (see the model TU).
struct FlintSession {
    __half *d_k = nullptr;      // [layers][capacity][kv_heads][head_dim]
    __half *d_v = nullptr;      // same shape
    KvCacheGeometry geom;       // pos-major; the layout this engine builds
    int64_t capacity = 0;
    int64_t kv_rows = 0;        // positions cached so far
    int32_t last_tok = -1;      // last generated token (session continuation)
    std::vector<int32_t> gen_tail;   // last call's generated tokens

    bool allocated() const { return d_k != nullptr; }
    std::int64_t Bytes() const;
    void Free();

    // Grows (never shrinks) the cache to `capacity`; a fresh session allocates.
    cudaError_t EnsureCapacity(const FlintTopology &top, int64_t capacity,
                               cudaStream_t stream);
    void Reset();
};

// Per-session device scratch (reused across Infer calls; grown on demand).
struct FlintScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    int64_t logits_elems = 0;
    int32_t *d_ids = nullptr;
    int32_t *d_top_ids = nullptr;   // [K] router selection, device-resident
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    FlintRoutedSlot *d_slots = nullptr;
    FlintRoutedSlot *h_slots = nullptr;
    float *d_logits = nullptr;   // [vocab] F32 (head is FP16, accumulated F32)
    void Free();
};

// End-to-end generation: token-by-token prefill + decode, streaming one Emit
// per decoded token and polling gs->Aborted() between steps. Any of the
// checkpoint's stop ids terminates the turn.
gpu::InferenceResult RunFlintInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    const FlintPinnedWeights &pw, FlintSession &session, FlintScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs);

} // namespace lsxcommon::flint
