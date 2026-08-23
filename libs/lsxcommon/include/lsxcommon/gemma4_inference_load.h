#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/gemma4_convert.h"

namespace lsxcommon {
namespace gemma {

// ── Geometry constants (real checkpoint; tests bypass via
//    LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY in GemmaBaseModel::ValidateGeometry) ──
constexpr int kGemmaHidden = 2816;
constexpr int kGemmaNumLayers = 30;
constexpr int kGemmaNumHeads = 16;
constexpr int kGemmaKVHeads = 8;            // sliding-attention layers
constexpr int kGemmaHeadDim = 256;
constexpr int kGemmaGlobalKVHeads = 2;      // full-attention layers
constexpr int kGemmaGlobalHeadDim = 512;
constexpr int kGemmaNumExperts = 128;
constexpr int kGemmaTopK = 8;

// Per-layer pinned weights. All FP16 tensors are 256-byte-aligned views of one
// arena allocation (GLM-style); F32 tensors get their own allocations.
struct alignas(16) GemmaLayerWeights {
    __nv_bfloat16 *d_input_norm = nullptr;
    __nv_bfloat16 *d_post_norm = nullptr;       // post_attention_layernorm
    __nv_bfloat16 *d_pre_ffn = nullptr;         // pre_feedforward_layernorm   (dense branch)
    __nv_bfloat16 *d_post_ffn = nullptr;        // post_feedforward_layernorm  (dense branch)
    __nv_bfloat16 *d_pre_ffn2 = nullptr;        // pre_feedforward_layernorm_2 (MoE branch)
    __nv_bfloat16 *d_post_ffn2 = nullptr;       // post_feedforward_layernorm_2(MoE branch)

    float *d_layer_scalar = nullptr;     // F32 [1]
    float *d_router_scale = nullptr;     // F32-uploaded BF16 [hidden]
    float *d_per_expert_scale = nullptr; // F32 [num_experts]

    // Attention projections (FP16). Sliding layers use q/k/v with
    // kv_heads=8/head_dim=256 rows; full-attention layers carry the global
    // geometry heads=16/global_head_dim=512, kv_heads=2.
    __nv_bfloat16 *d_q = nullptr;
    __nv_bfloat16 *d_k = nullptr;
    __nv_bfloat16 *d_v = nullptr;
    __nv_bfloat16 *d_o = nullptr;
    __nv_bfloat16 *d_q_norm = nullptr;          // per-head RMSNorm weights
    __nv_bfloat16 *d_k_norm = nullptr;
    int64_t q_rows = 0, o_cols = 0, kv_rows = 0;

    // Dense MLP branch (FP16).
    __nv_bfloat16 *d_mlp_gate = nullptr;
    __nv_bfloat16 *d_mlp_up = nullptr;
    __nv_bfloat16 *d_mlp_down = nullptr;

    // Router logits projection (FP16) [num_experts, hidden].
    __nv_bfloat16 *d_router_proj = nullptr;

    // Non-null when uploaded arena-backed: every d_* half pointer above is a
    // view into this single allocation and Free() releases it as a whole.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct GemmaPinnedWeights {
    __nv_bfloat16 *d_embed = nullptr;           // tied: also serves as the LM head
    __nv_bfloat16 *d_final_norm = nullptr;
    int vocab_size = 0;
    int hidden = 0;
    std::vector<GemmaLayerWeights> layers;

    // Topology from <root>/logestix/model.index.json (present for converted
    // models). error_msg set when the index is missing/malformed.
    GemmaTopology top;

    // Peak device-resident weight set (VRAM only; host data excluded).
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Persistent per-session KV cache: per-layer k/v rows sized by that layer's
// attention geometry (sliding: kv_heads*head_dim; full: gkv*g_head_dim).
struct GemmaKvCache {
    std::vector<__nv_bfloat16 *> dk;            // per layer [max_total][kv_rows]
    std::vector<__nv_bfloat16 *> dv;
    int max_total = 0;
    std::uint64_t cache_bytes = 0;

    bool allocated() const { return !dk.empty(); }
    cudaError_t Allocate(const GemmaPinnedWeights &pw, int max_total);
    void Free();
};

// Reads <model_path>/logestix/pinned/{global,layer_N}.arrow and uploads every
// tensor as FP16 (device-side BF16->FP16 through the pinned staging ring).
// Strict: missing files / malformed tensors are hard errors. Sizes
// GemmaPinnedWeights.layers from layer_*.arrow files present on disk.
cudaError_t LoadGemmaPinnedWeights(const std::string &model_path,
                                   GemmaPinnedWeights &pw,
                                   cudaStream_t stream);

// ── On-demand routed-expert index ───────────────────────────────────────────
// Zero-copy view into the mmap'd Arrow IPC buffer. Layout follows the shared
// ExpertSchema decode: w1 = gate half, w3 = up half, w2 = down. A null host_w1
// marks the entry missing/inactive (deterministic skip).
struct GemmaExpertEntry {
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

using GemmaExpertDb = lsxcommon::gpu::expert_store::ExpertStore<GemmaExpertEntry>;

GemmaExpertDb load_gemma_expert_index(const std::string &model_path,
                                      int num_layers, int experts_per_layer);

void ensure_gemma_expert_resident(GemmaExpertDb &db, int layer, int expert,
                                  cudaStream_t stream);
void free_gemma_expert_index(GemmaExpertDb &db);

} // namespace gemma
} // namespace lsxcommon

