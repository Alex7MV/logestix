#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/granite_dense_switch_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace granite_switch {

// One adapter's LoRA pair for one projection. A is [rank, in], B is [out, rank].
// B is stored pre-scaled by alpha/rank at composition time, so no runtime
// scaling is applied.
struct GraniteSwitchLora {
    __half *d_A = nullptr;
    __half *d_B = nullptr;
    int rank = 0;
    int in = 0;
    int out = 0;
};

struct GraniteSwitchProjection {
    lsx_nvfp4::LsxNvfp4Weight base;
    // [adapter][slice]; a slice is one fused output region (qkv has 3,
    // gate_up has 2, o/down have 1). rank 0 means "adapter does not target
    // this slice".
    std::vector<std::vector<GraniteSwitchLora>> lora;
};

struct GraniteSwitchLayerWeights {
    bool loaded = false;
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    GraniteSwitchProjection qkv;      // [Q|K|V] fused
    GraniteSwitchProjection o;
    GraniteSwitchProjection gate_up;  // [gate|up] fused
    GraniteSwitchProjection down;

    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct GraniteSwitchPinnedWeights {
    GraniteSwitchTopology top;
    lsx_nvfp4::LsxNvfp4Weight embed; // tied embedding matrix
    lsx_nvfp4::LsxNvfp4Weight head;  // non-owning view of embed, scale / logits
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<GraniteSwitchLayerWeights> layers;
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Loads the converted model from output_root (either <root>/logestix/pinned or
// <root>/pinned) into device memory on `stream`.
bool LoadGraniteSwitchPinnedWeights(const std::string &output_root,
                                    GraniteSwitchPinnedWeights &pw,
                                    cudaStream_t stream);

} // namespace granite_switch
} // namespace lsxcommon
