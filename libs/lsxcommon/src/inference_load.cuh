#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>

namespace lsxcommon {
namespace gpu {

constexpr int kHiddenDim = 4096;
constexpr int kNumLayers = 43;
constexpr int kNumRouted = 256;
constexpr int kTopK = 8;
constexpr int kIntermediateDim = 2048;

struct ExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
};

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    std::string error_msg;
};

ModelTopology load_topology(const std::string &model_path);

struct ExpertDb {
    std::vector<ExpertEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::string error_msg;
};

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer);

inline int moe_route(int token_idx, int layer_idx, int experts_per_layer)
{
    unsigned h = static_cast<unsigned>(token_idx) * 2654435761U
               + static_cast<unsigned>(layer_idx) * 314159265U;
    h ^= h >> 16; h *= 0x85ebca6bU;
    h ^= h >> 13; h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return static_cast<int>(h % static_cast<unsigned>(experts_per_layer));
}

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
