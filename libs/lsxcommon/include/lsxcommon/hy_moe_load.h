#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/hy_moe_convert.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::hy {

// Per-layer attention weights (pinned). Projections are NVFP4; every norm is
// FP16 with the PLAIN weight form (scale = w, not Gemma's 1+w), including the
// two per-head norms.
struct HyAttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;  // [NH*HD,H] / [NKV*HD,H] / [H,NH*HD]
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    __half *d_q_norm = nullptr;         // [HD]
    __half *d_k_norm = nullptr;         // [HD]
    __half *d_arena = nullptr;          // norm payload (one allocation)
    size_t arena_bytes = 0;
    bool loaded = false;
};

// Per-expert byte geometry inside a layer slab (immutable metadata).
struct HyExpertRecord {
    int64_t w1_off = 0, w1_bytes = 0;  // gate rows  [I,H]
    int64_t w2_off = 0, w2_bytes = 0;  // down rows  [H,I]
    int64_t w3_off = 0, w3_bytes = 0;  // up rows    [I,H]
    int64_t s1_off = 0, s1_bytes = 0;
    int64_t s2_off = 0, s2_bytes = 0;
    int64_t s3_off = 0, s3_bytes = 0;
    float gs1 = 1.0f, gs2 = 1.0f, gs3 = 1.0f;
};

struct HyLayerWeights {
    HyAttnWeights attn;
    bool is_moe = false;
    // Dense SwiGLU MLP (layers < first_moe_layer).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    // MoE block (layers >= first_moe_layer).
    float *d_router = nullptr;       // [E,H] F32 (high precision)
    std::int64_t router_bytes = 0;
    float *d_router_bias = nullptr;  // [E] F32 expert bias
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;
    // ALL routed experts resident in one device slab per MoE layer. The set is
    // ~16 GiB at NVFP4 (47 x 128 x 3 x 768 x 2048), which fits a 32 GiB card,
    // so this family trades the expert cache for a plain slab index (Xing
    // precedent: no LRU, routing computes offsets).
    uint8_t *expert_slab = nullptr;
    size_t slab_bytes = 0;
    std::vector<HyExpertRecord> experts;      // [E]
    // Host pointer tables for the neutral batched expert GEMV (top-k slots are
    // resolved on the host from the router's top-k readback).
    std::vector<const uint8_t *> packed_gate, scales_gate;
    std::vector<const uint8_t *> packed_down, scales_down;
    std::vector<const uint8_t *> packed_up, scales_up;
    std::vector<float> gs_gate, gs_down, gs_up;
    // Device-resident [3E] LsxExpertPtrRow table ([gate E][down E][up E]) for
    // the ids-streaming expert GEMV, so decode needs no router D2H.
    lsx_nvfp4::LsxExpertPtrRow *d_row_table = nullptr;
    bool loaded = false;
};

struct HyPinnedWeights {
    HyTopology top;
    // FP16 embed table and head matrix (the head's row error multiplies every
    // logit, so it is never 4-bit — phi-4 / gemma-4 / MiniMax rule).
    __half *d_embed = nullptr;   // [V, H]
    __half *d_head = nullptr;    // [V, H]
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    __half *d_final_norm = nullptr;
    std::vector<HyLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
    void Free();
};

void FreeHyPinnedWeights(HyPinnedWeights &pw);

// Uploads the whole converted model (globals + per-layer attention/dense-MLP/
// router/shared experts + every routed expert) from <model_path>/logestix.
// Returns false with `error` set (including a loud failure when the expert set
// does not fit the free VRAM — this family never silently streams).
//
// The passport variant lets a sibling (Hy-SimulMT) reuse this loader with ITS
// frozen geometry instead of borrowing Hy-Memory's namespace.
//
// There is no `format_tag` parameter any more. It existed to compare a per-model
// tag that the index used to carry, and that comparison was the ONLY thing that
// told the parent's tree from the sibling's — which is why removing the tag
// (design §2.4.3) silently made `LoadHyDenseWeights` accept a Hy-SimulMT tree.
// The two trees have the same tower, so no geometry check can separate them
// either. The identity check belongs where the identity is recorded:
// `artifact.fb` `model_type`, enforced once in ModelBase::BeginLoad.
bool LoadHyDenseWeights(const std::string &model_path,
                        const HyPassportGeometry &expect, HyPinnedWeights &pw,
                        cudaStream_t stream, std::string &error);

// Hy-Memory's frozen passport (the default for the parent model).
bool LoadHyDenseWeights(const std::string &model_path, HyPinnedWeights &pw,
                        cudaStream_t stream, std::string &error);

// Reads model.index.json -> HyTopology (format/geometry gates).
HyTopology LoadHyTopology(const std::string &model_path);

// Geometry of the persistent KV cache (pos-major; the measured default).
KvCacheGeometry HyKvGeometry(const HyTopology &top, int64_t capacity);

} // namespace lsxcommon::hy
