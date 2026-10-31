#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/step_moe_kernels.h"
#include "lsxcommon/step_moe_load.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::step {

// Persistent incremental session: pos-major KV cache for all 92 layers.
// Session state is created and published INSIDE RunUnderResidencyLock.
struct StepSession {
    __half *d_k = nullptr;  // [layers][capacity][kv_heads][head_dim]
    __half *d_v = nullptr;
    KvCacheGeometry geom;
    int64_t capacity = 0;
    int64_t kv_rows = 0;
    int32_t last_tok = -1;
    std::vector<int32_t> gen_tail;

    bool allocated() const { return d_k != nullptr; }
    std::int64_t Bytes() const;
    void Free();
    void Reset();
    cudaError_t EnsureCapacity(const StepTopology &top, int64_t capacity,
                               cudaStream_t stream);
};

// Per-session device scratch (reused across Infer calls; grown on demand).
struct StepScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    int64_t logits_elems = 0;
    int32_t *d_ids = nullptr;
    int32_t *d_top_ids = nullptr;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    StepRoutedSlot *d_slots = nullptr;
    StepRoutedSlot *h_slots = nullptr;
    float *d_logits = nullptr;
    void Free();
};

// End-to-end generation: token-by-token prefill + decode, streaming one Emit
// per decoded token and polling gs->Aborted() between steps.
gpu::InferenceResult RunStepInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, StepPinnedWeights &pw, StepSession &session,
    StepScratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs);

} // namespace lsxcommon::step
