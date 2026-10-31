#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/ling_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::ling {

// Persistent incremental session. MLA layers keep a pos-major KV cache
// addressed through KvCacheGeometry; KDA layers keep the recurrent state and
// the short-conv history. Both are created and published inside the residency
// lock (see the model TU).
struct LingSession {
    __half *d_kv = nullptr;      // [mla_layers][capacity][heads][192+128]
    KvCacheGeometry geom;        // pos-major, layers == mla layer count
    float *d_kda_state = nullptr;  // [kda_layers][heads][head_dim][head_dim]
    __half *d_conv_hist = nullptr; // [kda_layers][conv-1][3*KD]
    int64_t capacity = 0;
    int64_t kv_rows = 0;
    int32_t last_tok = -1;
    std::vector<int32_t> gen_tail;

    bool allocated() const { return d_kv != nullptr; }
    std::int64_t Bytes() const;
    void Free();
    void Reset(const LingIndex &top, cudaStream_t stream);
    cudaError_t EnsureCapacity(const LingIndex &top, int64_t capacity,
                               cudaStream_t stream);
};

struct LingScratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    float *d_logits = nullptr;
    int64_t logits_elems = 0;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    moe::MoERoutedSlot *d_slots = nullptr;
    moe::MoERoutedSlot *h_slots = nullptr;
    float *d_scores = nullptr;
    int64_t scores_elems = 0;
    float *d_moe_acc = nullptr;
    gpu::ExpertDeviceCache cache;
    bool cache_ready = false;
    void Free();
};

// Allocates the device expert slot cache from free VRAM (2 GiB reserve;
// LSX_LING_EXPERT_CACHE_SLOTS overrides). Idempotent.
bool LingPrepareExpertCache(LingPinnedWeights &pw, LingScratch &scratch,
                            std::string &error);

// Token-by-token prefill + decode, one GenerationStream::Emit per decoded
// token, abort polled between steps.
gpu::InferenceResult RunLingInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    LingPinnedWeights &pw, LingSession &session, LingScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs);

} // namespace lsxcommon::ling
