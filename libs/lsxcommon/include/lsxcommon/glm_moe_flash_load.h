#pragma once

// GLM-5.3-Flash loader types. Per-layer weights are a KDA/DSA union:
// KDA layers carry NVFP4 q/k/v/o projections (the four [4096,8192] pairs
// dominated the device graph; the recurrence scalars - A_log, dt_bias,
// conv1d, b/f/g projections, o_norm - stay FP16/F32), DSA layers carry
// NVFP4 MLA projections + BF16 raw indexer/kv_b weights. Routed experts
// reuse the generic pinned-slab store (moe::MoEExpertStore).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference_load.h"  // lsx_nvfp4::LsxNvfp4Weight
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/qwen_moe_load.h"  // moe::MoEExpertStore
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

    // Hyper-connection weights per layer (fn upconverted to FP32 on load
    // for the shared deepseek-style run_hc_pre/run_hc_post path).
    float *d_hc_attn_fn = nullptr;      // F32 [24, hc*H]
    float *d_hc_attn_base = nullptr;    // F32 [24]
    float *d_hc_attn_scale = nullptr;   // F32 [3]
    float *d_hc_ffn_fn = nullptr;       // F32 [24, hc*H]
    float *d_hc_ffn_base = nullptr;     // F32 [24]
    float *d_hc_ffn_scale = nullptr;    // F32 [3]
    int64_t hc_fn_rows = 0;
    int64_t hc_fn_cols = 0;

    // KDA (linear_attention) — q/k/v/o are NVFP4 (the four [4096,8192]
    // projections dominated the device-dense footprint; conv/scalar/gating
    // tensors stay FP16/F32 — numerics of the recurrence path unchanged).
    lsx_nvfp4::LsxNvfp4Weight kda_q;    // [8192, H]
    lsx_nvfp4::LsxNvfp4Weight kda_k;    // [8192, H]
    lsx_nvfp4::LsxNvfp4Weight kda_v;    // [8192, H]
    lsx_nvfp4::LsxNvfp4Weight kda_o;    // [H, 8192]
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
    lsx_nvfp4::LsxNvfp4Weight q_a;            // [q_lora, H]
    lsx_nvfp4::LsxNvfp4Weight q_b;            // [heads*qk, q_lora]
    lsx_nvfp4::LsxNvfp4Weight kv_a;           // [kv_lora + qk, H]
    lsx_nvfp4::LsxNvfp4Weight o_big;          // [H, heads*qk]
    __half *d_q_a_norm = nullptr;   // [q_lora]
    __half *d_kv_a_norm = nullptr;  // [kv_lora]
    uint16_t *d_kv_b = nullptr;     // BF16 bits [heads*(qk+qk), kv_lora]
    // DSA indexer, BF16 bits on device. The indexer is its own lightweight
    // attention (transformers `Glm5NextTextIndexer`): scores compressed
    // k-pools, so only the top `index_topk` tokens reach the main MLA.
    uint16_t *d_idx_wq_b = nullptr;    // [idx_heads*idx_head_dim, q_lora]
    uint16_t *d_idx_wk = nullptr;      // [idx_head_dim, H]
    uint16_t *d_idx_knorm_w = nullptr; // [idx_head_dim]  (LayerNorm, eps 1e-6)
    uint16_t *d_idx_knorm_b = nullptr; // [idx_head_dim]
    uint16_t *d_idx_wproj = nullptr;   // [idx_heads, H]
    uint16_t *d_idx_gate = nullptr;    // [idx_head_dim, H]  (per-channel pool gate)
    uint16_t *d_idx_ape = nullptr;     // [idx_kpool, idx_head_dim]

    // MoE: router gate BF16 bits [E, H] + bias F32 [E]; shared NVFP4 trio.
    uint8_t *d_router = nullptr;
    int64_t router_bytes = 0;
    float *d_router_bias = nullptr;
    lsx_nvfp4::LsxNvfp4Weight shared_w1;  // gate_proj [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_w2;  // down_proj [H, SI]
    lsx_nvfp4::LsxNvfp4Weight shared_w3;  // up_proj [SI, H]

    // Dense MLP (layers 0..2): NVFP4 gate/up/down.
    lsx_nvfp4::LsxNvfp4Weight mlp_gate;   // [dense_I, H]
    lsx_nvfp4::LsxNvfp4Weight mlp_up;     // [dense_I, H]
    lsx_nvfp4::LsxNvfp4Weight mlp_down;   // [H, dense_I]

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

// Layer range of the routed-expert store. The resident graph routes layers
// [kMoeFirstLayer, kLayers) only: the MTP nextn layer (kMtpLayer == kLayers)
// is stored-not-applied, so its 288 expert files stay on disk but are NOT
// pinned (~4 GB of host RAM held for nothing at 14.2 MB per expert).
// @p max_layer_seen is the highest experts/layer_NNN directory present.
// Returns false with @p why filled when the tree names a layer beyond the MTP
// slot: a malformed tree must fail loudly rather than silently drop experts.
bool FlashExpertLayerRange(int max_layer_seen, int &first, int &last,
                           std::string &why);

// Loads every experts/layer_NNN/exp_EEE.arrow under <model_path>/logestix/
// into pinned host memory (layers [kMoeFirstLayer, kLayers) - see
// FlashExpertLayerRange). Strict errors.
bool LoadGlmFlashExpertStore(const std::string &model_path,
                             lsxcommon::moe::MoEExpertStore &store,
                             std::string &error);

} // namespace glm_moe_flash
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_MOE_FLASH_LOAD_H
