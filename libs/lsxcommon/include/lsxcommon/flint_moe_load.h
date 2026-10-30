#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/flint_moe_convert.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::flint {

// Per-layer attention weights (pinned). Projections are NVFP4; every norm and
// the learned sink are FP16 (plain weight form, not Gemma's 1+w).
struct FlintAttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o;  // [NH*HD,H] / [NKV*HD,H] / [H,NH*HD]
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    __half *d_q_norm = nullptr;         // [HD]
    __half *d_k_norm = nullptr;         // [HD]
    __half *d_sink = nullptr;           // [NH] learned softmax logit
    __half *d_arena = nullptr;          // norm/sink payload (one allocation)
    size_t arena_bytes = 0;
    bool loaded = false;
};

// Per-expert byte geometry inside a layer slab (immutable metadata).
struct FlintExpertRecord {
    int64_t w1_off = 0, w1_bytes = 0;  // gate rows  [I,H]
    int64_t w2_off = 0, w2_bytes = 0;  // down rows  [H,I]
    int64_t w3_off = 0, w3_bytes = 0;  // up rows    [I,H]
    int64_t s1_off = 0, s1_bytes = 0;
    int64_t s2_off = 0, s2_bytes = 0;
    int64_t s3_off = 0, s3_bytes = 0;
    float gs1 = 1.0f, gs2 = 1.0f, gs3 = 1.0f;
};

struct FlintLayerWeights {
    FlintAttnWeights attn;
    bool is_moe = false;
    // Dense SwiGLU MLP (mlp_only_layers).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    // MoE block.
    float *d_router = nullptr;       // [E,H] F32 (high precision, never 4-bit)
    std::int64_t router_bytes = 0;
    // Ungated shared expert (plain SwiGLU, added to the routed sum).
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;
    // ALL routed experts resident in one device slab per MoE layer (~21 GiB
    // total at NVFP4, which fits a 32 GiB card): no expert cache, no host
    // store, routing resolves offsets.
    uint8_t *expert_slab = nullptr;
    size_t slab_bytes = 0;
    std::vector<FlintExpertRecord> experts;   // [E]
    // Host pointer tables for the neutral batched expert GEMV.
    std::vector<const uint8_t *> packed_gate, scales_gate;
    std::vector<const uint8_t *> packed_down, scales_down;
    std::vector<const uint8_t *> packed_up, scales_up;
    std::vector<float> gs_gate, gs_down, gs_up;
    // Device-resident [3E] LsxExpertPtrRow table ([gate E][down E][up E]).
    lsx_nvfp4::LsxExpertPtrRow *d_row_table = nullptr;
    bool loaded = false;
};

struct FlintPinnedWeights {
    FlintTopology top;
    // FP16 embed table and head matrix (the head's row error multiplies every
    // logit, and the vendor serves the head in FP32).
    __half *d_embed = nullptr;   // [V, H]
    __half *d_head = nullptr;    // [V, H]
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    __half *d_final_norm = nullptr;
    std::vector<FlintLayerWeights> layers;
    std::uint64_t resident_bytes = 0;
    void Free();
};

void FreeFlintPinnedWeights(FlintPinnedWeights &pw);

// Uploads the whole converted model (globals + per-layer attention/dense-MLP/
// router/shared experts + every routed expert) from <model_path>/logestix.
// Returns false with `error` set.
bool LoadFlintDenseWeights(const std::string &model_path, FlintPinnedWeights &pw,
                           cudaStream_t stream, std::string &error);

// Reads model.index.json -> FlintTopology (format/geometry gates).
FlintTopology LoadFlintTopology(const std::string &model_path);

// Geometry of the persistent KV cache (pos-major; the measured default).
KvCacheGeometry FlintKvGeometry(const FlintTopology &top, int64_t capacity);

} // namespace lsxcommon::flint
