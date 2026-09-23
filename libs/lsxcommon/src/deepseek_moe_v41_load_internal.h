// deepseek_moe_v41_load_internal.h — DeepSeek-V4.1-Flash device weight set.
// Autonomous contour: no DeepSeek-V4-Flash headers. Dense weights arrive as
// F8_E4M3 + per-32x32 UE8M0 scales and are dequantized to FP16 on upload;
// routed experts stay NVFP4 and are served by the neutral lsx_nvfp4 layer via
// the shared ExpertStore LRU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/expert_store.h"

#include "deepseek_moe_v41_constants.h"
#include "deepseek_moe_v41_load.h"

namespace lsxcommon {
namespace deepseek_v41 {
namespace gpu {

// One routed expert (NVFP4): decoded from the shared 9-column ExpertSchema by
// ExpertStore::DecodeEntry. Field order is the aggregate-init contract.
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

using ExpertDb = lsxcommon::gpu::expert_store::ExpertStore<ExpertEntry>;

// ── Per-layer dense device weights (FP16 unless noted F32) ─────────────────
struct LayerWeights {
    LayerMeta meta{};

    __half *d_attn_norm = nullptr;
    __half *d_ffn_norm = nullptr;
    __half *d_q_norm = nullptr;
    __half *d_kv_norm = nullptr;
    // Dense FP8 weights are stored as raw F8_E4M3 (1 byte/elem) + per-32x32
    // UE8M0 scales so the GEMV dequantizes on the fly (halves HBM traffic and
    // frees VRAM for the expert cache).
    uint8_t *d_wq_a = nullptr;
    uint8_t *d_wq_a_scale = nullptr;
    uint8_t *d_wq_b = nullptr;
    uint8_t *d_wq_b_scale = nullptr;
    uint8_t *d_wkv = nullptr;
    uint8_t *d_wkv_scale = nullptr;
    uint8_t *d_wo_a = nullptr;
    uint8_t *d_wo_a_scale = nullptr;
    uint8_t *d_wo_b = nullptr;
    uint8_t *d_wo_b_scale = nullptr;
    __half *d_gate_weight = nullptr;   // router BF16 -> FP16
    float *d_gate_bias = nullptr;      // [n_routed] F32
    float *d_gate_bias_vl = nullptr;   // [n_routed] F32
    __half *d_shared_w1 = nullptr;
    __half *d_shared_w2 = nullptr;
    __half *d_shared_w3 = nullptr;

    // Compressor (kv_source layers only).
    __half *d_comp_wkv = nullptr;
    __half *d_comp_wgate = nullptr;
    __half *d_comp_norm = nullptr;

    // Indexer (index_source layers only).
    uint8_t *d_idx_wq_b = nullptr;
    uint8_t *d_idx_wq_b_scale = nullptr;
    __half *d_idx_wk = nullptr;
    __half *d_idx_k_norm = nullptr;
    __half *d_idx_weights_proj = nullptr;

    // Attention sink + hyper-connections (F32).
    float *d_attn_sink = nullptr;
    float *d_hc_attn_fn = nullptr;
    float *d_hc_attn_base = nullptr;
    float *d_hc_attn_scale = nullptr;
    float *d_hc_ffn_fn = nullptr;
    float *d_hc_ffn_base = nullptr;
    float *d_hc_ffn_scale = nullptr;

    // Shapes for GEMM (rows, cols).
    int wq_a_rows = 0, wq_a_cols = 0;
    int wq_b_rows = 0, wq_b_cols = 0;
    int wkv_rows = 0, wkv_cols = 0;
    int wo_a_rows = 0, wo_a_cols = 0;
    int wo_b_rows = 0, wo_b_cols = 0;
    int gate_rows = 0, gate_cols = 0;

    std::vector<void *> owned;   // device allocations to free

    void Free();
};

struct PinnedWeights {
    __half *d_embed = nullptr;
    __half *d_head = nullptr;
    __half *d_final_norm = nullptr;
    std::vector<LayerWeights> layers;
    std::vector<void *> owned;   // global device allocations to free
    int vocab_size = 0;
    int hidden = 0;
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Uploads the dense/global device set for @p model_path. Strict: any missing
// or malformed required tensor returns cudaErrorInvalidValue.
cudaError_t load_pinned_weights(const std::string &model_path,
                                PinnedWeights &pw, cudaStream_t stream);

// ── Expert store (host mmap LRU) ───────────────────────────────────────────
ExpertDb load_expert_index(const std::string &model_path, int num_layers,
                           int experts_per_layer);
void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream);
void free_expert_index(ExpertDb &db);

} // namespace gpu
} // namespace deepseek_v41
} // namespace lsxcommon
