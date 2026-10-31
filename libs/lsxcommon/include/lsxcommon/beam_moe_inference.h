#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/beam_moe_kernels.h"
#include "lsxcommon/beam_moe_load.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::beam {

// Persistent incremental session: one pos-major KV block for every layer (the
// sliding layers keep full history too; a 512-row ring is a later memory
// optimisation, not part of ingestion). Created and published INSIDE
// RunUnderResidencyLock (see the model TU).
struct BeamSession {
    __half *d_kv_k = nullptr;   // [layers][capacity][kv_heads][head_dim]
    __half *d_kv_v = nullptr;
    KvCacheGeometry geom;
    int64_t capacity = 0;
    int64_t kv_rows = 0;        // positions cached so far
    int32_t last_tok = -1;      // last generated token (session continuation)
    std::vector<int32_t> gen_tail;  // last call's generated tokens

    bool allocated() const { return d_kv_k != nullptr; }
    std::int64_t Bytes() const;
    void Free();
    void Reset();

    cudaError_t EnsureCapacity(const BeamTopology &top, int64_t capacity,
                               cudaStream_t stream);
};

// Per-session device scratch (reused across Infer calls; grown on demand).
struct BeamScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    float *d_logits = nullptr;
    int64_t logits_elems = 0;
    int32_t *d_ids = nullptr;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    BeamRoutedSlot *d_slots = nullptr;
    BeamRoutedSlot *h_slots = nullptr;
    gpu::ExpertDeviceCache cache;   // routed-expert slot LRU
    void Free();
};

// End-to-end generation: token-by-token prefill + decode, streaming one Emit
// per decoded token and polling gs->Aborted() between steps.
gpu::InferenceResult RunBeamInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    BeamPinnedWeights &pw, BeamSession &session, BeamScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs);

} // namespace lsxcommon::beam
