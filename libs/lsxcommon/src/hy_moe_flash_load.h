// hy_moe_flash_load.h — Hy4-flash device weight set (Stage 3).
//
// Autonomous contour: no other family's headers. Dense projections are NVFP4
// (`lsx_nvfp4`, the only GEMM path); norms are FP16 (BF16 source is exactly
// representable); the iHC scalars, the router and the learned sinks keep F32
// because the reference computes their products in fp32 and a rounding there
// changes the routing / the softmax. Routed experts stay NVFP4 and are served by
// the neutral ExpertStore LRU: 41 layers x 512 experts is 83 GB, so the store is
// mmap-backed and bounded, never materialized.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/index_layout.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/vram_estimate.h"

#include "hy_moe_flash_constants.h"
#include "lsxcommon/hy_moe_flash_convert.h"   // LayerShape
#include "lsxcommon/lsx_kv_cache.h"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

// ── Routed expert ──────────────────────────────────────────────────────────
// One NVFP4 expert, decoded from the shared 9-column ExpertSchema by
// ExpertStore::DecodeEntry. The field order IS the aggregate-init contract
// (w1, w2, w3, s1, s2, s3, gs1, gs2, gs3, byte counts) and the names are what
// the store reads. Repo expert layout: w1 = gate, w2 = down, w3 = up; this
// family's gate and up are the two halves of the source `gate_up_proj`, and
// `down` maps the 1536-wide expert width back into the 1536-wide latent.
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

static_assert(std::is_trivially_copyable<ExpertEntry>::value,
              "ExpertStore entries must stay POD views");
static_assert(sizeof(ExpertEntry) <= 128,
              "ExpertStore entries must stay <= 128 bytes");

using ExpertDb = lsxcommon::gpu::expert_store::ExpertStore<ExpertEntry>;

// ── Per-layer weights ──────────────────────────────────────────────────────

// iHC (identity Hyper-Connections) parameters, fp32 like the reference.
//   fn    [2*hc_mult, hc_mult*hidden]  row-major mixes for pre/post
//   base  [2*hc_mult]
//   scale [2]
struct IhcParams {
    float *d_fn = nullptr;
    float *d_base = nullptr;
    float *d_scale = nullptr;
};

// Indexer (DSA layers only). The two 128x128 projections and the three scalars
// are F32 in the checkpoint; keeping them F32 costs 128 KB per layer.
struct IndexerWeights {
    lsx_nvfp4::LsxNvfp4Weight q;      // [n_heads*head_dim, q_lora] NVFP4
    lsx_nvfp4::LsxNvfp4Weight tok;    // [head_dim, hidden]      NVFP4
    lsx_nvfp4::LsxNvfp4Weight hw;     // [n_heads, hidden]       NVFP4
    float *d_key_value = nullptr;     // [head_dim, head_dim]    F32
    float *d_key_gate = nullptr;      // [head_dim, head_dim]    F32
    __half *d_token_key_norm = nullptr;  // [head_dim]
    float *d_key_norm = nullptr;      // [head_dim]
    float *d_ape = nullptr;           // [pool_ratio, head_dim]
    bool loaded = false;
};

struct AttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q_a;    // [q_lora, hidden]
    lsx_nvfp4::LsxNvfp4Weight q_b;    // [heads*qk_head, q_lora]
    lsx_nvfp4::LsxNvfp4Weight kv_a;   // [kv_lora + qk_rope, hidden]
    // kv_b is stored SPLIT (see the converter): nope rows [heads*qk_nope,
    // kv_lora] and value rows [heads*v_head, kv_lora], each contiguous per head.
    lsx_nvfp4::LsxNvfp4Weight kv_b;   // [heads*qk_nope, kv_lora]
    lsx_nvfp4::LsxNvfp4Weight kv_b_value;  // [heads*v_head, kv_lora]
    lsx_nvfp4::LsxNvfp4Weight o;      // [hidden, heads*v_head]
    lsx_nvfp4::LsxNvfp4Weight gate;   // [heads*v_head, hidden]  (sigma gate)
    __half *d_q_norm = nullptr;       // [q_lora]
    __half *d_kv_norm = nullptr;      // [kv_lora]
    __half *d_post_proj_norm = nullptr;  // [hidden] sandwich norm
    float *d_sink = nullptr;          // [heads] learnable softmax logit
    IndexerWeights indexer;           // DSA layers
    int heads = 0;
    bool loaded = false;
};

struct LayerWeights {
    LayerShape shape;
    AttnWeights attn;
    IhcParams hc_attn;
    IhcParams hc_mlp;
    __half *d_input_norm = nullptr;   // [hidden]
    __half *d_post_attn_norm = nullptr;  // [hidden]
    // Dense FFN (layer 0 only).
    lsx_nvfp4::LsxNvfp4Weight d_gate, d_up, d_down;
    __half *d_post_fc2_norm = nullptr;
    // MoE (layers 1..41).
    __half *d_router = nullptr;       // [num_routed, hidden] BF16->FP16 (exact)
    float *d_router_bias = nullptr;   // [num_routed] F32 (selection only)
    lsx_nvfp4::LsxNvfp4Weight s_gate, s_up, s_down;  // shared expert
    lsx_nvfp4::LsxNvfp4Weight r_down; // [latent, hidden]
    lsx_nvfp4::LsxNvfp4Weight r_up;   // [hidden, latent]
    __half *d_combine_norm = nullptr; // [hidden]
    int64_t router_bytes = 0;
    bool loaded = false;
};

// iHC head (final 4-stream collapse).
struct HeadWeights {
    float *d_fn = nullptr;      // [hc_mult, hc_mult*hidden]
    float *d_base = nullptr;    // [hc_mult]
    float *d_scale = nullptr;   // [1]
};

struct PinnedWeights {
    int num_layers = 0;
    int experts_per_layer = 0;
    int vocab_size = 0;
    // The embedding table and the fp32 head matrix: both live on the device at
    // source precision (FP16 holds BF16 exactly), never quantized.
    __half *d_embed = nullptr;
    __half *d_lm_head = nullptr;
    __half *d_final_norm = nullptr;
    HeadWeights head;
    std::vector<LayerWeights> layers;
    std::uint64_t resident_bytes = 0;

    void Free();
};

void FreeHy4PinnedWeights(PinnedWeights &pw);

// ── Topology (model.index.json) ────────────────────────────────────────────
struct Topology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    int64_t total_pinned_tensors = 0;
    int hidden_size = 0;
    int vocab_size = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int latent_size = 0;
    int shared_intermediate_size = 0;
    int dense_intermediate_size = 0;
    int num_attention_heads = 0;
    int swa_num_attention_heads = 0;
    int q_lora_rank = 0;
    int kv_lora_rank = 0;
    int qk_nope_head_dim = 0;
    int qk_rope_head_dim = 0;
    int v_head_dim = 0;
    int sliding_window = 0;
    int hc_mult = 0;
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    int64_t deferred_engram = 0;
    int64_t deferred_vision = 0;
    int64_t deferred_dfly = 0;
    std::vector<LayerShape> layers;
    VramProfile vram;
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
    std::string model;
    std::string error_msg;
};

// Reads <model_path>/logestix/model.index.json. Fills error_msg on failure.
Topology LoadTopology(const std::string &model_path);

// Frozen-passport cross-check. A mismatch is a hard Load failure: an artifact
// written from a different architecture must not decode as if it were ours.
bool ValidateGeometry(const Topology &topo, std::string *error);

// The fold `gpu::detect_model_type` applies to a card name ("Hy4-flash" ->
// "hy4-flash").
std::string NormalizeCardName(std::string_view card);

// True when this tree's index names @p model_type (or names no model at all).
bool MatchesModelType(const Topology &topo, std::string_view model_type);

inline bool IsSupportedFormat(const Topology &topo)
{
    return IndexLayoutSupported(topo.layout);
}

// ── Load / expert store ────────────────────────────────────────────────────
bool LoadHy4Weights(const std::string &model_path, const Topology &topo,
                    PinnedWeights &pw, cudaStream_t stream, std::string &error);

// Builds the bounded host expert store (mmap views + LRU).
ExpertDb LoadExpertIndex(const std::string &model_path, int num_layers,
                         int experts_per_layer, int max_open);
void EnsureExpertResident(ExpertDb &db, int layer, int expert,
                          cudaStream_t stream);
void FreeExpertIndex(ExpertDb &db);

// KV cache geometry of the persistent session: one MLA latent per position per
// layer (kv_lora + qk_rope wide, one "head").
KvCacheGeometry KvGeometry(int64_t capacity);

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
