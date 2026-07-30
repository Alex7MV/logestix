#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace lsxcommon {
namespace gpu {

constexpr int kHiddenDim = 7168;
constexpr int kNumLayers = 43;
constexpr int kNumRouted = 256;
constexpr int kTopK = 8;
constexpr int kIntermediateDim = 2048;

struct LayerWeights {
    __half *d_ffn_norm = nullptr;
    __half *d_gate_weight = nullptr;
    uint8_t *d_shared_w1 = nullptr;
    uint8_t *d_shared_w2 = nullptr;
    uint8_t *d_shared_w3 = nullptr;
    uint8_t *d_shared_w1_scales = nullptr;
    uint8_t *d_shared_w2_scales = nullptr;
    uint8_t *d_shared_w3_scales = nullptr;
    float shared_w1_global_scale = 1.0f;
    float shared_w2_global_scale = 1.0f;
    float shared_w3_global_scale = 1.0f;
    int64_t shared_w1_bytes = 0;
    int64_t shared_w2_bytes = 0;
    int64_t shared_w3_bytes = 0;

    void Free();
};

struct PinnedWeights {
    __half *d_embed = nullptr;
    __half *d_head = nullptr;
    __half *d_final_norm = nullptr;
    std::vector<LayerWeights> layers;
    int vocab_size = 0;

    void Free();
};

cudaError_t load_pinned_weights(const std::string &model_path,
                                 PinnedWeights &pw,
                                 cudaStream_t stream);

} // namespace gpu
} // namespace lsxcommon
