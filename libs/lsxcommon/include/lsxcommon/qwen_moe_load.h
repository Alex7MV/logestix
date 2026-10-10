#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/lsx_moe_inference.h"  // moe::MoEExpertStore, lsxcommon::moe::EnsureMoEExpertResident

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
// through lsxcommon::moe::MoEExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
bool LoadQwenMoeDenseWeights(const std::string &model_path,
                             QwenMoePinnedWeights &pw, cudaStream_t stream,
                             std::string &error);

void FreeQwenMoePinnedWeights(QwenMoePinnedWeights &pw);

// Loads every experts/layer_<l>/exp_<e>.arrow under
// <model_path>/logestix/ into pinned host memory. Strict errors.
bool LoadMoEExpertStore(const std::string &model_path,
                            lsxcommon::moe::MoEExpertStore &store, std::string &error);

// Allocates the LRU device cache sized to the largest expert payload.
bool AllocateMoEExpertCache(moe::MoEExpertStore &store, int slots,
                                gpu::ExpertDeviceCache &cache,
                                std::string &error);
void FreeMoEExpertCache(gpu::ExpertDeviceCache &cache);

// Ensures expert `eid` of `layer` is resident in the device cache; stages its
// payload H2D from the pinned slab on miss. Returns slot index, -1 on error
// (error text goes to `error` when non-null). When `h2d_stream` is provided,
// miss staging is enqueued on that (copy) stream instead of `stream`; set
// *missed so the caller can fence the upload before consuming the slot.
// *evicted_slot receives the recycled slot index on eviction (-1 otherwise):
// the caller must fence in-flight readers of that slot (event wait) before
// the staged upload overwrites it.

} // namespace lsxcommon::qwen_moe
