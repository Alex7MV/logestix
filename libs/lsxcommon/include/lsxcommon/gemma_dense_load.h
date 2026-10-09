#pragma once

// Gemma-4-31B-it (dense) pinned-weight loader — Model Ingestion Protocol
// step 3.
//
// Reads <model_path>/logestix/pinned/{global.arrow,global_nvfp4.arrow,
// layer_N.arrow} and uploads every tensor: the eight linear projections of a
// layer as NVFP4 triples (packed nibbles + E4M3 block scales + an F32 global
// scale), the four norms + q_norm/k_norm as FP16, and layer_scalar as F32.
// Norms and scalars are NEVER quantized.
//
// Activations and weights are FP16 (FP32 accumulation); the per-layer scalar
// keeps an F32 device copy plus a host mirror so the hot loop never reads it
// back.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/gemma_dense_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"

namespace lsxcommon {
namespace gemma_dense {

struct GemmaDenseLayerWeights {
    // FP16 norms (raw high precision — never quantized).
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;    // post_attention_layernorm
    __half *d_pre_ffn = nullptr;      // pre_feedforward_layernorm
    __half *d_post_ffn = nullptr;     // post_feedforward_layernorm
    __half *d_q_norm = nullptr;
    __half *d_k_norm = nullptr;
    float *d_layer_scalar = nullptr;  // F32 [1]

    // NVFP4 projections. On a full-attention layer attn_v.d_packed is null:
    // upstream ties V to K (attention_k_eq_v), so V is the RAW k_proj output
    // and v_norm is applied to it (NOT the k-normed, rotated key).
    lsx_nvfp4::LsxNvfp4Weight attn_q, attn_k, attn_v, attn_o;
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;

    // Non-null when the norms came from one arena allocation; Free() releases
    // it as a whole.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    bool HasV() const { return attn_v.d_packed != nullptr; }

    void Free();
};

struct GemmaDensePinnedWeights {
    // Tied embed / lm_head: ONE NVFP4 table, sampled through the embed gather
    // and again through the logits kernel.
    lsx_nvfp4::LsxNvfp4Weight embed;
    __half *d_final_norm = nullptr;
    // Arena holding the final norm (released by Free() as a whole).
    void *final_norm_arena = nullptr;
    size_t final_norm_arena_bytes = 0;
    int vocab_size = 0;
    int hidden = 0;
    std::vector<GemmaDenseLayerWeights> layers;

    // Host mirror of layer_scalar (constant weight; read once at upload
    // instead of a device readback per layer per forward).
    std::vector<float> h_layer_scalar;

    GemmaDenseTopology top;  // from <root>/logestix/model.index.json
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Persistent per-session KV cache. The two layer kinds store DIFFERENT
// geometries, so the cache carries one KvCacheGeometry per layer (pos-major:
// [pos][head][dim]); every address in the runtime goes through Offset().
struct GemmaDenseKvCache {
    std::vector<__half *> dk;
    std::vector<__half *> dv;
    std::vector<KvCacheGeometry> geo;
    int max_total = 0;
    std::uint64_t cache_bytes = 0;

    bool allocated() const { return !dk.empty(); }
    cudaError_t Allocate(const GemmaDensePinnedWeights &pw, int max_total);
    void Free();
};

// Reads the pinned layout and uploads every tensor. Strict: a missing file or
// a malformed tensor is a hard error. Sizes
// GemmaDensePinnedWeights::layers from the layer_*.arrow files present.
cudaError_t LoadGemmaDensePinnedWeights(const std::string &model_path,
                                        GemmaDensePinnedWeights &pw,
                                        cudaStream_t stream);

} // namespace gemma_dense
} // namespace lsxcommon
