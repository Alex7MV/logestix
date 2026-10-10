#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/ornith_moe_convert.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_moe_inference.h"  // MoEExpertStore, LoadMoEExpertStore

namespace lsxcommon::ornith {

// Unified expert device cache (shared neutral LRU; local alias keeps the
// ornith call sites stable).
using OrnithMoeExpertDeviceCache = gpu::ExpertDeviceCache;

// Per-layer device weights: reused dense attention block + MoE extras
// (router kept BF16 on device; shared expert NVFP4 trio + FP16 scalar gate).
struct OrnithMoeLayerWeights {
    lsxcommon::attn::LayerWeights attn;             // norms + attention (no dense MLP)
    uint8_t *d_router = nullptr;             // raw BF16 bits [E, H]
    int64_t router_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight shared_w1;         // gate_proj  [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_w2;         // down_proj  [H, SI]
    lsx_nvfp4::LsxNvfp4Weight shared_w3;         // up_proj    [SI, H]
    __half *d_shared_gate = nullptr;         // [H] fp16
    bool moe_loaded = false;
};

struct OrnithMoePinnedWeights {
    qwen::QwenTopology top;                  // dense view for kernel reuse
    OrnithMoeTopology moe_top;                 // full topology incl. MoE fields
    lsx_nvfp4::LsxNvfp4Weight embed;
    lsx_nvfp4::LsxNvfp4Weight head;
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<OrnithMoeLayerWeights> layers;
    std::size_t resident_bytes = 0;

    void Free();
};

// Loads dense parts (globals, attention, norms, router, shared expert) from
// <model_path>/logestix/ (legacy logestix-qwen35moe/ still read). Routed experts are NOT resident; they go
// through moe::MoEExpertStore separately. Strict: returns false and fills
// `error` on any missing/malformed input.
bool LoadOrnithMoeDenseWeights(const std::string &model_path,
                             OrnithMoePinnedWeights &pw, cudaStream_t stream,
                             std::string &error);

void FreeOrnithMoePinnedWeights(OrnithMoePinnedWeights &pw);



} // namespace lsxcommon::ornith
