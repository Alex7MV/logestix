#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_expert_cache.h"

namespace lsxcommon::qwen_moe {

// Per-layer device weights: reused dense attention block + MoE extras
// (router kept BF16 on device; shared expert NVFP4 trio + FP16 scalar gate).
struct QwenMoeLayerWeights {
    qwen::QwenLayerWeights attn;             // norms + attention (no dense MLP)
    uint8_t *d_router = nullptr;             // raw BF16 bits [E, H]
    int64_t router_bytes = 0;
    qwen::QwenNvfp4Weight shared_w1;         // gate_proj  [SI, H]
    qwen::QwenNvfp4Weight shared_w2;         // down_proj  [H, SI]
    qwen::QwenNvfp4Weight shared_w3;         // up_proj    [SI, H]
    __half *d_shared_gate = nullptr;         // [H] fp16
    bool moe_loaded = false;
};

struct QwenMoePinnedWeights {
    qwen::QwenTopology top;                  // dense view for kernel reuse
    QwenMoeTopology moe_top;                 // full topology incl. MoE fields
    qwen::QwenNvfp4Weight embed;
    qwen::QwenNvfp4Weight head;
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<QwenMoeLayerWeights> layers;
    std::size_t resident_bytes = 0;

    void Free();
};

// Loads dense parts (globals, attention, norms, router, shared expert) from
// <model_path>/logestix/ (legacy logestix-qwen35moe/ still read). Routed experts are NOT resident; they go
// through QwenMoeExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
bool LoadQwenMoeDenseWeights(const std::string &model_path,
                             QwenMoePinnedWeights &pw, cudaStream_t stream,
                             std::string &error);

void FreeQwenMoePinnedWeights(QwenMoePinnedWeights &pw);

// Host-resident routed-expert payloads (pinned slabs), laid out per layer.
// The generation context stages slices of these into the LRU device cache.
struct QwenMoeExpertRecord {
    // Byte ranges inside the owning layer slab (packed w1|w2|w3 first, then
    // E4M3 scales s1|s2|s3).
    int64_t w1_off = 0, w1_bytes = 0;
    int64_t w2_off = 0, w2_bytes = 0;
    int64_t w3_off = 0, w3_bytes = 0;
    int64_t s1_off = 0, s1_bytes = 0;
    int64_t s2_off = 0, s2_bytes = 0;
    int64_t s3_off = 0, s3_bytes = 0;
    float gs1 = 1.f, gs2 = 1.f, gs3 = 1.f;
};

struct QwenMoeExpertStore {
    // [num_layers] pinned slabs; slab l holds num_experts records back to back.
    std::vector<uint8_t *> h_slabs;
    std::vector<size_t> slab_bytes;
    std::vector<std::vector<QwenMoeExpertRecord>> records;  // [layers][expert]
    int num_layers = 0;
    int num_experts = 0;

    void Free();
};

// Loads every experts/layer_<l>/exp_<e>.arrow under
// <model_path>/logestix/ into pinned host memory. Strict errors.
bool LoadQwenMoeExpertStore(const std::string &model_path,
                            QwenMoeExpertStore &store, std::string &error);

// Allocates the LRU device cache sized to the largest expert payload.
bool AllocateQwenMoeExpertCache(QwenMoeExpertStore &store, int slots,
                                QwenMoeExpertDeviceCache &cache,
                                std::string &error);
void FreeQwenMoeExpertCache(QwenMoeExpertDeviceCache &cache);

// Ensures expert `eid` of `layer` is resident in the device cache; stages its
// payload H2D from the pinned slab on miss. Returns slot index, -1 on error
// (error text goes to `error` when non-null). When `h2d_stream` is provided,
// miss staging is enqueued on that (copy) stream instead of `stream`; set
// *missed so the caller can fence the upload before consuming the slot.
int EnsureQwenMoeExpertResident(QwenMoeExpertStore &store, int layer, int eid,
                                QwenMoeExpertDeviceCache &cache,
                                cudaStream_t stream, std::string &error,
                                cudaStream_t h2d_stream = nullptr,
                                bool *missed = nullptr);

} // namespace lsxcommon::qwen_moe
