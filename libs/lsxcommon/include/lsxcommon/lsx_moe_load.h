// Neutral templated MoE dense loader shared by qwen3.5/3.6 and ornith
// (dedup phase 6, Task 6). The ornith load.cpp was a 100 % rename-copy of
// qwen_moe_load.cpp (proven by normalized diff); the bodies move here
// verbatim, templated on the family's weights/topology types.
//
// Family headers keep their public names as aliases/instantiations:
//   qwen_moe_load.h   -> QwenMoePinnedWeights = MoePinnedWeights<QwenMoeTopology>
//   ornith_moe_load.h -> OrnithMoePinnedWeights = MoePinnedWeights<OrnithMoeTopology>
// The topology structs are field-identical (proven by diff); the template
// keeps them separate so each family's passport stays its own.
#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_dense_convert.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace moe {

// Per-layer device weights: reused dense attention block + MoE extras
// (router kept BF16 on device; shared expert NVFP4 trio + FP16 scalar gate).
struct MoeLayerWeights {
    lsxcommon::attn::LayerWeights attn;             // norms + attention (no dense MLP)
    uint8_t *d_router = nullptr;             // raw BF16 bits [E, H]
    int64_t router_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight shared_w1;         // gate_proj  [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_w2;         // down_proj  [H, SI]
    lsx_nvfp4::LsxNvfp4Weight shared_w3;         // up_proj    [SI, H]
    __half *d_shared_gate = nullptr;         // [H] fp16
    bool moe_loaded = false;
};

template <class MoeTopology>
struct MoePinnedWeights {
    qwen::QwenTopology top;                  // dense view for kernel reuse
    MoeTopology moe_top;                     // full topology incl. MoE fields
    lsx_nvfp4::LsxNvfp4Weight embed;
    lsx_nvfp4::LsxNvfp4Weight head;
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<MoeLayerWeights> layers;
    std::size_t resident_bytes = 0;

    void Free();
};

void MoeLayerWeightsFree(MoeLayerWeights &lw);

// Loads dense parts (globals, attention, norms, router, shared expert) from
// <model_path>/logestix/ (legacy logestix-qwen35moe/ still read). Routed
// experts are NOT resident; they go through lsxcommon::moe::MoEExpertStore
// separately. Strict: returns false and fills `error` on any missing or
// malformed input.
template <class MoeTopology>
bool LoadMoeDenseWeights(const std::string &model_path,
                         MoePinnedWeights<MoeTopology> &pw,
                         cudaStream_t stream, std::string &error);

template <class MoeTopology>
void FreeMoePinnedWeights(MoePinnedWeights<MoeTopology> &pw)
{
    pw.Free();
}

} // namespace moe
} // namespace lsxcommon