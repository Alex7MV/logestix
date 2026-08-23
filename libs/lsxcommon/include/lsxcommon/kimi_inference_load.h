#pragma once

#include "lsxcommon/schema.h"
#include "lsxcommon/expert_store.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {
namespace kimi {

// Kimi-K2.7-Code geometry (config.json text_config, verified against the
// vault checkpoint). Compile-time locked per MODEL INGEST INVARIANT; config
// values are cross-checked at Load() and any mismatch is a hard error.
constexpr int kKimiHidden = 7168;
constexpr int kKimiNumLayers = 61;
constexpr int kKimiNumHeads = 64;
constexpr int kKimiKvLoraRank = 512;   // compressed KV latent
constexpr int kKimiQLoraRank = 1536;   // q_lora_rank
constexpr int kKimiQkNope = 128;       // per-head non-rope q/k dim
constexpr int kKimiQkRope = 64;        // shared rope dim
constexpr int kKimiHeadV = 128;        // v_head_dim
constexpr int kKimiMoeIntermediate = 2048;
constexpr int kKimiDenseIntermediate = 18432;
constexpr int kKimiNumRouted = 384;
constexpr int kKimiNumShared = 1;
constexpr int kKimiTopK = 8;
constexpr int kKimiVocab = 163840;
constexpr int kKimiFirstKDenseReplace = 1;
constexpr float kKimiRoutedScalingFactor = 2.827f;
constexpr float kKimiRmsEps = 1e-5f;
constexpr float kKimiRopeTheta = 50000.0f;
constexpr float kKimiYarnFactor = 64.0f;
constexpr int kKimiYarnOriginalMaxPos = 4096;

// Derived MLA projection row counts (rows of the checkpoint weight matrices).
constexpr int kKimiQaRows = kKimiQLoraRank;                               // 1536
constexpr int kKimiQbRows = kKimiNumHeads * (kKimiQkNope + kKimiQkRope);  // 12288
constexpr int kKimiKvaRows = kKimiKvLoraRank + kKimiQkRope;               // 576
constexpr int kKimiKvbRows = kKimiNumHeads * (kKimiQkNope + kKimiHeadV);  // 16384
constexpr int kKimiORows = kKimiHidden;                                   // 7168

// Per-layer pinned weights for one Kimi layer, uploaded as FP16 (host-side
// BF16→FP16 via the pinned staging ring). Layer 0 fills the mlp_* fields,
// MoE layers (1..60) fill the router fields. Pointers stay null when the
// layer file lacks the tensor.
struct KimiLayerWeights {
    // Layer norms
    __half *d_input_norm = nullptr;
    __half *d_post_attn_norm = nullptr;

    // MLA attention (raw BF16 → FP16)
    __half *d_q_a_proj = nullptr;
    __half *d_q_a_norm = nullptr;
    __half *d_q_b_proj = nullptr;
    __half *d_kv_a_proj = nullptr;
    __half *d_kv_a_norm = nullptr;
    __half *d_kv_b_proj = nullptr;
    __half *d_o_proj = nullptr;
    int64_t q_a_rows = 0, q_a_cols = 0;
    int64_t q_b_rows = 0, q_b_cols = 0;
    int64_t kv_a_rows = 0, kv_a_cols = 0;
    int64_t kv_b_rows = 0, kv_b_cols = 0;
    int64_t o_rows = 0, o_cols = 0;

    // Router (MoE layers 1..60): mlp.gate.weight + e_score_correction_bias
    __half *d_router_gate = nullptr;
    float *h_gate_bias = nullptr;
    int gate_bias_n = 0;

    // Dense MLP (layer 0 only)
    __half *d_mlp_gate = nullptr;
    __half *d_mlp_down = nullptr;
    __half *d_mlp_up = nullptr;

    // Non-null when the layer was uploaded arena-backed: every d_* pointer
    // above is a view into this single allocation and Free() releases the
    // arena instead of the individual pointers.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct KimiPinnedWeights {
    __half *d_embed = nullptr;
    __half *d_head = nullptr;
    __half *d_final_norm = nullptr;
    int vocab_size = 0;
    int hidden = 0;
    std::vector<KimiLayerWeights> layers;   // indexed by layer 0..60

    // Shared experts: one NVFP4-packed shared expert per MoE layer from
    // pinned/shared_experts.arrow. Zero-copy mmap views, host-side only.
    std::vector<PackedExpertView> shared_host;
    std::shared_ptr<void> shared_keepalive;

    // Peak device-resident weight set uploaded by LoadKimiPinnedWeights.
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Reads <model_path>/logestix/pinned/{global,layer_<n>}.arrow and uploads
// every pinned tensor to device memory as FP16. Missing layer files are a
// hard error (uniform Kimi grid: layer 0 dense, layers 1..60 MoE).
cudaError_t LoadKimiPinnedWeights(const std::string &model_path,
                                  KimiPinnedWeights &pw,
                                  cudaStream_t stream);

// ── On-demand routed-expert index ──────────────────────────────────────────
// Memory-mapped view of one routed expert: NVFP4 packed E2M1 nibble payloads
// + per-block E4M3 scales + decoded global scales, zero-copy into the Arrow
// IPC buffer. Dequant invariant matches GlmExpertEntry:
//   w[i] = E2M1(nibble_i) * 2^(((scales[i>>4]>>3)&0xF) - 7) * global_scale
struct KimiExpertEntry {
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

using KimiExpertDb = lsxcommon::gpu::expert_store::ExpertStore<KimiExpertEntry>;

// Builds the flat file-path index of every routed expert
// (<model_path>/logestix/experts/layer_<l>/exp_<e>.arrow with fallback to
// <model_path>/experts/...) and warms the LRU up to max_open entries.
KimiExpertDb load_kimi_expert_index(const std::string &model_path,
                                    int num_layers, int experts_per_layer);

// Ensures the mmap for (layer, expert) is LRU-resident, synchronizing the
// stream before any eviction (in-flight DMA reads must finish first).
void ensure_kimi_expert_resident(KimiExpertDb &db, int layer, int expert,
                                 cudaStream_t stream);

// Releases all expert mmaps.
void free_kimi_expert_index(KimiExpertDb &db);

} // namespace kimi
} // namespace lsxcommon
