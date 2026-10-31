#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/beam_moe_convert.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/schema.h"

namespace lsxcommon::beam {

// Per-layer attention weights (pinned). Projections are NVFP4; the four
// sandwich norms, the per-head q/k norms and the per-head gate bias are FP16
// (plain weight form, not Gemma's 1+w).
struct BeamAttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o, gate;  // [NH*HD,H] / [H,NH*HD]
    __half *d_input_norm = nullptr;
    __half *d_post_attn_norm = nullptr;
    __half *d_pre_ffn_norm = nullptr;
    __half *d_post_ffn_norm = nullptr;
    __half *d_q_norm = nullptr;   // [HD]
    __half *d_k_norm = nullptr;   // [HD]
    __half *d_gate_bias = nullptr; // [NH]
    __half *d_arena = nullptr;    // norm/qk-norm/gate-bias payload
    size_t arena_bytes = 0;
    bool loaded = false;
};

struct BeamLayerWeights {
    BeamAttnWeights attn;
    bool is_moe = false;
    // Dense SwiGLU MLP (mlp_layer_types == "dense").
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    // MoE block router: F32, never quantized (it moves the top-k selection).
    float *d_router = nullptr;       // [E,H]
    float *d_router_bias = nullptr;  // [E] e_score_correction_bias
    int64_t router_bytes = 0;
    bool loaded = false;
};

struct BeamPinnedWeights {
    BeamTopology top;
    // FP16 embed table and head matrix (the head's row error multiplies every
    // logit).
    __half *d_embed = nullptr;   // [V, H]
    __half *d_head = nullptr;    // [V, H]
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    __half *d_final_norm = nullptr;
    std::vector<BeamLayerWeights> layers;
    // Routed experts (49 layers x 256) streamed through the neutral mmap store:
    // 277 GB at NVFP4 does not fit the card and is never pinned.
    moe::MoEExpertRuntime expert_store;
    std::uint64_t resident_bytes = 0;

    void Free();
};

void FreeBeamPinnedWeights(BeamPinnedWeights &pw);

// Uploads the dense graph (globals + per-layer attention/dense-MLP/router) and
// opens the routed-expert host store from <model_path>/logestix. Returns false
// with `error` set.
bool LoadBeamWeights(const std::string &model_path, BeamPinnedWeights &pw,
                     cudaStream_t stream, std::string &error);

// Reads model.index.json -> BeamTopology (format/geometry gates).
BeamTopology LoadBeamTopology(const std::string &model_path);

// Geometry of the persistent KV cache (pos-major; the measured default).
KvCacheGeometry BeamKvGeometry(const BeamTopology &top, int64_t capacity);

} // namespace lsxcommon::beam
