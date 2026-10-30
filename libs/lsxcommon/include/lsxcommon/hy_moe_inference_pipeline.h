#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/hy_moe_kernels.h"
#include "lsxcommon/hy_moe_load.h"
#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::hy {

// Persistent incremental session: pos-major KV cache for all 48 layers.
// Session state is created and published INSIDE RunUnderResidencyLock (see the
// model TU).
struct HySession {
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
    cudaError_t EnsureCapacity(const HyTopology &top, int64_t capacity,
                               cudaStream_t stream);
    void Reset();
};

// Per-session device scratch (reused across Infer calls; grown on demand).
struct HyScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    int64_t half_elems = 0;
    int64_t float_elems = 0;
    int64_t logits_elems = 0;
    int32_t *d_ids = nullptr;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    HyRoutedSlot *d_slots = nullptr;
    HyRoutedSlot *h_slots = nullptr;
    float *d_logits = nullptr;   // [vocab] F32 (head is FP16, accumulated F32)
    void Free();
};

struct HyGenResult {
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
gpu::InferenceResult RunHyInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const HyPinnedWeights &pw, HySession &session,
    HyScratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs);

} // namespace lsxcommon::hy
