#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/minimax_moe_convert.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::minimax {

// Per-layer attention weights (pinned). Projections are NVFP4; every norm is
// FP16 with the Gemma offset form (scale = 1 + w), including the four
// per-head norms (q, k, index_q, index_k).
struct MinimaxAttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;      // [NH*HD,H] / [NKV*HD,H] / [H,NH*HD]
    lsx_nvfp4::LsxNvfp4Weight index_q, index_k;  // sparse layers only
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    __half *d_q_norm = nullptr;         // [HD]
    __half *d_k_norm = nullptr;         // [HD]
    __half *d_index_q_norm = nullptr;   // [index_head_dim]
    __half *d_index_k_norm = nullptr;   // [index_head_dim]
    __half *d_arena = nullptr;          // norm payload (one allocation)
    size_t arena_bytes = 0;
    bool loaded = false;
};

struct MinimaxLayerWeights {
    MinimaxAttnWeights attn;
    bool is_moe = false;
    // Dense SwiGLU-OAI MLP (layers < first_moe_layer).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    // MoE block (layers >= first_moe_layer).
    float *d_router = nullptr;       // [E,H] F32 (high precision)
    std::int64_t router_bytes = 0;
    float *d_router_bias = nullptr;  // [E] F32 e_score_correction_bias
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;
    bool loaded = false;
};

struct MinimaxPinnedWeights {
    MinimaxTopology top;
    // FP16 embed table and head matrix (see minimax_moe_convert.cpp: the head's
    // row error multiplies every logit, so it must not be 4-bit).
    __half *d_embed = nullptr;   // [V, H]
    __half *d_head = nullptr;    // [V, H]
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    __half *d_final_norm = nullptr;
    std::vector<MinimaxLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
    void Free();
};

void FreeMinimaxPinnedWeights(MinimaxPinnedWeights &pw);

// Uploads the pinned graph (globals + per-layer attention/dense-MLP/router/
// shared experts) from <model_path>/logestix. Returns false with `error` set.
bool LoadMinimaxDenseWeights(const std::string &model_path,
                             MinimaxPinnedWeights &pw, cudaStream_t stream,
                             std::string &error);

// ── Routed-expert host store ────────────────────────────────────────────────
//
// The routed-expert set (57 x 128 experts, ~232 GiB in NVFP4) is never pinned:
// it lives behind the neutral mmap ExpertStore (R1/R2 in the ingestion
// protocol), mapped on demand under LSX_MINIMAX_EXPERT_MAX_OPEN. Layers
// [0, first_moe_layer) have no expert files and are skipped by the store.

// Per-expert GEOMETRY (immutable metadata, never a pointer into host memory).
struct MinimaxExpertRecord {
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    int64_t s1_bytes = 0, s2_bytes = 0, s3_bytes = 0;
    float gs1 = 1.0f, gs2 = 1.0f, gs3 = 1.0f;
};

// View into one expert's mmap, in the field order the shared
// ExpertStore<TEntry>::DecodeEntry() aggregate-fills. Never owns bytes.
struct MinimaxExpertEntry {
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

using MinimaxExpertDb = gpu::expert_store::ExpertStore<MinimaxExpertEntry>;

struct MinimaxExpertStore {
    MinimaxExpertDb db;                                  // neutral mmap views
    std::vector<std::vector<MinimaxExpertRecord>> records;  // [layer][expert]
    int num_layers = 0;
    int num_experts = 0;
    int first_moe_layer = 0;
    bool loaded = false;
    void Free();
};

bool LoadMinimaxExpertStore(const std::string &model_path,
                            MinimaxExpertStore &out, std::string &error);

// Maps expert (layer, eid) into host memory (LRU-bounded), staging it into a
// device slot on a cache miss. Returns the slot, -1 on error (error filled).
int EnsureMinimaxExpertResident(MinimaxExpertStore &store,
                                gpu::ExpertDeviceCache &cache, int layer,
                                int eid, cudaStream_t stream,
                                std::string &error, bool *missed = nullptr,
                                int *evicted_slot = nullptr);

// Allocates the device slot cache from the store geometry + free VRAM.
bool AllocateMinimaxExpertCache(MinimaxExpertStore &experts,
                                gpu::ExpertDeviceCache &cache, int slots,
                                int top_k, std::string &error);
void FreeMinimaxExpertCache(gpu::ExpertDeviceCache &cache);

// Reads model.index.json -> MinimaxTopology (format/geometry gates).
MinimaxTopology LoadMinimaxTopology(const std::string &model_path);

// Geometry of the persistent KV cache (pos-major; the measured default).
KvCacheGeometry MinimaxKvGeometry(const MinimaxTopology &top, int64_t capacity);

} // namespace lsxcommon::minimax
