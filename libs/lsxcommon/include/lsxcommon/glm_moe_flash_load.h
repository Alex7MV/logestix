#pragma once

// GLM-5.3-Flash loader types. Per-layer weights are a KDA/DSA union:
// KDA layers keep high-precision FP16 projections (converter policy), DSA
// layers carry NVFP4 MLA projections + BF16 raw indexer/kv_b weights.
// Routed experts reuse the generic pinned-slab store (QwenMoeExpertStore).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference_load.h"  // QwenNvfp4Weight
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/qwen_moe_load.h"  // QwenMoeExpertStore
#include "lsxcommon/glm_moe_flash_convert.h"  // FlashTopology
#include "lsxcommon/schema.h"  // TensorRecord

#ifndef LSXCOMMON_GLM_MOE_FLASH_LOAD_H
#define LSXCOMMON_GLM_MOE_FLASH_LOAD_H

namespace lsxcommon {
namespace glm_moe_flash {

// Per-layer device weights (KDA/DSA union).
struct FlashLayerWeights {
    bool is_kda = true;
    bool loaded = false;
    bool is_moe = false;

    // Shared per-layer: input/post-attention norms FP16 [H].
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    // Hyper-connection weights per layer (raw bits on device).
    uint16_t *d_hc_attn_fn = nullptr;   // BF16 bits [24, hc*H]
    float *d_hc_attn_base = nullptr;    // F32 [24]
    float *d_hc_attn_scale = nullptr;   // F32 [3]
    uint16_t *d_hc_ffn_fn = nullptr;    // BF16 bits [24, hc*H]
    float *d_hc_ffn_base = nullptr;     // F32 [24]
    float *d_hc_ffn_scale = nullptr;    // F32 [3]
    int64_t hc_fn_rows = 0;
    int64_t hc_fn_cols = 0;

    // KDA (linear_attention) — FP16 weights, F32 scalars.
    __half *d_kda_q = nullptr;      // [Hq=8192, H]
    __half *d_kda_k = nullptr;      // [8192, H]
    __half *d_kda_v = nullptr;      // [8192, H]
    __half *d_kda_o = nullptr;      // [H, 8192]
    __half *d_kda_q_conv = nullptr; // [8192, 1, 4] -> flat [8192*4]
    __half *d_kda_k_conv = nullptr;
    __half *d_kda_v_conv = nullptr;
    float *d_A_log = nullptr;       // [64]
    float *d_dt_bias = nullptr;     // [8192]
    __half *d_b_proj = nullptr;     // [64, H]
    __half *d_f_a = nullptr;        // [128, H]
    __half *d_f_b = nullptr;        // [8192, 128]
    __half *d_g_a = nullptr;        // [128, H]
    __half *d_g_b = nullptr;        // [8192, 128]
    __half *d_o_norm = nullptr;     // [128]

    // DSA (deepseek_sparse_attention) — NVFP4 + FP16 norms + BF16 raw.
    qwen::QwenNvfp4Weight q_a;            // [q_lora, H]
    qwen::QwenNvfp4Weight q_b;            // [heads*qk, q_lora]
    qwen::QwenNvfp4Weight kv_a;           // [kv_lora + qk, H]
    qwen::QwenNvfp4Weight o_big;          // [H, heads*qk]
    __half *d_q_a_norm = nullptr;   // [q_lora]
    __half *d_kv_a_norm = nullptr;  // [kv_lora]
    uint16_t *d_kv_b = nullptr;     // BF16 bits [heads*(qk+qk), kv_lora]
    // Indexer raw BF16 payloads (stored as opaque records, consumed by the
    // DSA kernel path).
    std::vector<TensorRecord> indexer_raw;

    // MoE: router gate BF16 bits [E, H] + bias F32 [E]; shared NVFP4 trio.
    uint8_t *d_router = nullptr;
    int64_t router_bytes = 0;
    float *d_router_bias = nullptr;
    qwen::QwenNvfp4Weight shared_w1;  // gate_proj [SI, H]
    qwen::QwenNvfp4Weight shared_w2;  // down_proj [H, SI]
    qwen::QwenNvfp4Weight shared_w3;  // up_proj [SI, H]

    // Dense MLP (layers 0..2): NVFP4 gate/up/down.
    qwen::QwenNvfp4Weight mlp_gate;   // [dense_I, H]
    qwen::QwenNvfp4Weight mlp_up;     // [dense_I, H]
    qwen::QwenNvfp4Weight mlp_down;   // [H, dense_I]

    void Free();
};

// Device-resident globals + per-layer weights.
struct FlashPinnedWeights {
    FlashTopology top;
    // embed / lm_head kept as raw BF16 bits (vocab x hidden).
    uint8_t *d_embed_bf16 = nullptr;
    uint8_t *d_head_bf16 = nullptr;
    int64_t embed_bytes = 0;
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<FlashLayerWeights> layers;
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Loads dense parts (globals, per-layer raw/NVFP4 tensors) from
// <model_path>/logestix/ into device memory on `stream`. Strict: returns
// false and fills `error` on any missing/malformed input.
bool LoadGlmFlashDenseWeights(const std::string &model_path,
                              FlashPinnedWeights &pw, cudaStream_t stream,
                              std::string &error);

// Loads every experts/layer_NNN/exp_EEE.arrow under <model_path>/logestix/
// into pinned host memory. Strict errors.
bool LoadGlmFlashExpertStore(const std::string &model_path,
                             qwen_moe::QwenMoeExpertStore &store,
                             std::string &error);

} // namespace glm_moe_flash
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_MOE_FLASH_LOAD_H
