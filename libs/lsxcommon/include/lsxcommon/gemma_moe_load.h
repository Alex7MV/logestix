#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

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
    __nv_bfloat16 *d_pre_ffn = nullptr;         // pre_feedforward_layernorm
    __nv_bfloat16 *d_post_ffn = nullptr;        // post_feedforward_layernorm
    __nv_bfloat16 *d_post_ffn1 = nullptr;       // post_feedforward_layernorm_1 (MLP out)
    __nv_bfloat16 *d_pre_ffn2 = nullptr;        // pre_feedforward_layernorm_2 (MoE)
    __nv_bfloat16 *d_post_ffn2 = nullptr;       // post_feedforward_layernorm_2 (MoE out)

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

    // NVFP4 dense MLP (converted pinned schema carries the packed/scales/
    // global_scale triple). Non-null d_packed when present; the BF16
    // d_mlp_* fields above stay null then. `mlp_nvfp4` selects the GEMV
    // chain in ForwardLayer.
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    bool mlp_nvfp4 = false;

    // NVFP4 attention projections (same record triple). Full-attention
    // layers tie V to K: attn_v.d_packed is null there (v := raw k).
    lsx_nvfp4::LsxNvfp4Weight attn_q, attn_k, attn_v, attn_o;
    bool attn_nvfp4 = false;

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
    // NVFP4 tied embed/lm_head (global_nvfp4.arrow). When present the BF16
    // d_embed stays null; head_nvfp4 aliases embed_nvfp4's buffers (tied
    // embeddings — Free() releases them once via embed_nvfp4 only).
    lsx_nvfp4::LsxNvfp4Weight embed_nvfp4;
    lsx_nvfp4::LsxNvfp4Weight head_nvfp4;
    std::vector<GemmaLayerWeights> layers;

    // Host mirror of per-layer layer_scalar (constant weight; read once at
    // upload instead of a full stream sync per layer per forward).
    std::vector<float> h_layer_scalar;

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

// Resident-slot budget for the routed-expert device cache (qwen 17348a1
// analogy): keep every expert device-resident, bounded by free VRAM with a
// fixed headroom reserve; explicit env override (LSX_GEMMA_EXPERT_CACHE_SLOTS,
// passed pre-parsed as @p explicit_slots) takes precedence and is clamped to
// the pool size. Returns the slot count (0 when not even one expert fits).
inline int ComputeGemmaResidentSlots(std::size_t free_bytes,
                                     std::size_t per_expert_bytes,
                                     int total_experts,
                                     int explicit_slots = -1)
{
    if (total_experts <= 0) return 0;
    if (explicit_slots > 0) return explicit_slots < total_experts
                                   ? explicit_slots : total_experts;
    // Unified launch-strategy policy (resource_governor.h) — single reserve
    // constant for all models, maximizes expert-cache VRAM utilization.
    return lifecycle::ExpertCacheSlots(free_bytes, per_expert_bytes,
                                       total_experts,
                                       lsxcommon::perf::CurrentKnobs()
                                           .expert_slots_pct);
}

GemmaExpertDb load_gemma_expert_index(const std::string &model_path,
                                      int num_layers, int experts_per_layer);

void ensure_gemma_expert_resident(GemmaExpertDb &db, int layer, int expert,
                                  cudaStream_t stream);
void free_gemma_expert_index(GemmaExpertDb &db);

} // namespace gemma
} // namespace lsxcommon

