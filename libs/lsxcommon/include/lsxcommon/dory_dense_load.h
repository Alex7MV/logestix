#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace dory {

// Per-mixer device weights of a Dory checkpoint. Every physical layer is ONE
// mixer: either attention (qkv/g/o) or mlp (gate_up/down). The nGPT
// per-dimension vectors are pre-scaled on the HOST at load time, so the decode
// never re-derives a factor:
//   sqk2[i]     = (sqk[i]     * sqrt(head_dim)/std)^2      [q-size]
//   sgate2[i]   = (s_gate[i]  * sqrt(sqrt(hidden))/std)^2   [q-size]
//   suv_gate[i] =  suv_gate[i] * sqrt(hidden)               [intermediate]
//   suv_up[i]   =  suv_up[i]   * sqrt(hidden)               [intermediate]
//   alpha[i]    = |attn_alpha[i] (or mlp_alpha[i]) * alpha_init/std|
// They are F32 because the reference accumulates them in FP32 and FP16 would
// quantize the squared nGPT scales (values in the hundreds).
struct DoryLayerWeights {
    bool is_attention = false;
    bool loaded = false;

    // Attention mixer (NVFP4). qkv is the fused [q|k|v] record.
    lsx_nvfp4::LsxNvfp4Weight qkv;
    lsx_nvfp4::LsxNvfp4Weight g;
    lsx_nvfp4::LsxNvfp4Weight o;
    lsx_nvfp4::LsxNvfp4Weight qkv_res;
    lsx_nvfp4::LsxNvfp4Weight g_res;
    lsx_nvfp4::LsxNvfp4Weight o_res;

    // MLP mixer (NVFP4). gate_up is the fused [gate|up] record.
    lsx_nvfp4::LsxNvfp4Weight gate_up;
    lsx_nvfp4::LsxNvfp4Weight down;
    lsx_nvfp4::LsxNvfp4Weight gate_up_res;
    lsx_nvfp4::LsxNvfp4Weight down_res;

    float *d_sqk2 = nullptr;
    float *d_sgate2 = nullptr;
    float *d_suv_gate = nullptr;
    float *d_suv_up = nullptr;
    float *d_alpha = nullptr;

    void Free();
};

struct DoryPinnedWeights {
    DoryTopology top;

    // Edges stay FP16 (embed + lm head). The LM head's per-vocabulary logit
    // scale is F32 and pre-divided by initializer_range.
    __half *d_embed_f16 = nullptr;
    int embed_rows = 0, embed_cols = 0;
    __half *d_head_f16 = nullptr;
    int head_rows = 0, head_cols = 0;
    float *d_logit_scale = nullptr;
    int logit_scale_size = 0;

    std::vector<DoryLayerWeights> layers;
    // slot_of_layer[l] = KV cache slot of layer l (loop 0), or -1 for an mlp
    // layer. A recurrent attention layer occupies slots [s, s+loops).
    std::vector<int> slot_of_layer;
    int kv_slots = 0;

    std::uint64_t resident_bytes = 0;

    void Free();
};

// Loads all Dory weights from output_root (either <root>/logestix/pinned or
// <root>/pinned) into device memory on `stream`. Returns false on failure.
bool LoadDoryPinnedWeights(const std::string &output_root,
                           DoryPinnedWeights &pw, cudaStream_t stream);

} // namespace dory
} // namespace lsxcommon
