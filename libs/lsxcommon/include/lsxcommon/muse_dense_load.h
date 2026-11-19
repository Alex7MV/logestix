#pragma once

// Muse-Glimmer-30B pinned-weight loader — Model Ingestion Protocol step 3.
//
// Reads <model_path>/logestix/pinned/{global.arrow,layer_N.arrow} and uploads
// every tensor: the eight linear projections of a layer as NVFP4 triples
// (packed nibbles + E4M3 block scales + an F32 global scale), the four
// centered norms + the final norm as FP16, and the UNTIED embed / lm_head
// tables as FP16. Norms and the embedding tables are NEVER quantized.
//
// Activations are FP16 with FP32 accumulation; the BF16 -> FP16 conversion
// happens at upload (fastload::CvKind::kBf16ToHalf), the pinned records keep
// the source dtype.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"

namespace lsxcommon {
namespace muse_dense {

struct MuseDenseLayerWeights {
    // FP16 centered norms (raw high precision — never quantized).
    __half *d_input_norm = nullptr;     // eps rms_norm_eps
    __half *d_post_attn_norm = nullptr; // eps post_norm_eps
    __half *d_pre_ffn = nullptr;        // eps rms_norm_eps
    __half *d_post_ffn = nullptr;       // eps post_norm_eps

    // NVFP4 projections. ALL EIGHT exist on EVERY layer: the attention gate
    // (attn_gate) is present on sliding AND full layers, and V is a real
    // projection (no attention_k_eq_v tying in this family).
    lsx_nvfp4::LsxNvfp4Weight attn_q, attn_k, attn_v, attn_o, attn_gate;
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;

    // Non-null when the norms came from one arena allocation; Free() releases
    // it as a whole.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct MuseDensePinnedWeights {
    // FP16 globals: the embedding table (sampled through the gather + the
    // weightless embed norm) and the UNTIED lm_head (exact greedy logits).
    __half *d_embed = nullptr;
    __half *d_lm_head = nullptr;
    __half *d_final_norm = nullptr;
    // Arena holding the three globals (released by Free() as a whole).
    void *global_arena = nullptr;
    size_t global_arena_bytes = 0;
    int vocab_size = 0;
    int hidden = 0;
    std::vector<MuseDenseLayerWeights> layers;

    MuseDenseTopology top;  // from <root>/logestix/model.index.json
    std::uint64_t resident_bytes = 0;

    void Free();
};

// Persistent per-session KV cache. Every layer stores the SAME geometry
// (2 KV heads x 128); the sliding/full difference is the attention window,
// not the cache shape. Pos-major ([pos][head][dim]); every address in the
// runtime goes through KvCacheGeometry::Offset().
struct MuseDenseKvCache {
    std::vector<__half *> dk;
    std::vector<__half *> dv;
    std::vector<KvCacheGeometry> geo;
    int max_total = 0;
    std::uint64_t cache_bytes = 0;

    bool allocated() const { return !dk.empty(); }
    cudaError_t Allocate(const MuseDensePinnedWeights &pw, int max_total);
    void Free();
};

// Reads the pinned layout and uploads every tensor. Strict: a missing file or
// a malformed tensor is a hard error. Sizes MuseDensePinnedWeights::layers
// from the layer_*.arrow files present.
cudaError_t LoadMuseDensePinnedWeights(const std::string &model_path,
                                       MuseDensePinnedWeights &pw,
                                       cudaStream_t stream);

} // namespace muse_dense
} // namespace lsxcommon