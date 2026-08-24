#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_convert.h"

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
// <model_path>/logestix-qwen35moe/. Routed experts are NOT resident; they go
// through QwenMoeExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
bool LoadQwenMoeDenseWeights(const std::string &model_path,
                             QwenMoePinnedWeights &pw, cudaStream_t stream,
                             std::string &error);

void FreeQwenMoePinnedWeights(QwenMoePinnedWeights &pw);

} // namespace lsxcommon::qwen_moe
