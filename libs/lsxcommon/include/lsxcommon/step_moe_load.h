#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/step_moe_convert.h"

namespace lsxcommon::step {

// Per-layer attention weights (pinned). Projections are NVFP4; every norm is
// F32 (the checkpoint's `norm_dtype: float32`), including the two per-head QK
// norms. The per-head sigmoid gate projection (g_proj) is NVFP4 like the rest.
struct StepAttnWeights {
    lsx_nvfp4::LsxNvfp4Weight q, k, v, o, g;  // [NH*HD,H]/[NKV*HD,H]/[H,NH*HD]/[NH,H]
    float *d_q_norm = nullptr;   // [HD]
    float *d_k_norm = nullptr;   // [HD]
    bool loaded = false;
};

struct StepLayerWeights {
    StepAttnWeights attn;
    float *d_input_norm = nullptr;  // [H] F32
    float *d_post_norm = nullptr;   // [H] F32
    bool is_moe = false;
    // Dense SwiGLU MLP (layers < first_moe_layer).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate, mlp_up, mlp_down;
    // MoE block (layers >= first_moe_layer).
    __nv_bfloat16 *d_router = nullptr;  // [E,H] BF16 (the reference multiplies in F32)
    float *d_router_bias = nullptr;     // [E] F32 selection-only bias
    lsx_nvfp4::LsxNvfp4Weight shared_gate, shared_up, shared_down;
    std::int64_t router_bytes = 0;
    bool loaded = false;
};

struct StepPinnedWeights {
    StepTopology top;
    // BF16 embed table and head matrix (the head's row error multiplies every
    // logit, so it is never 4-bit — phi-4 / gemma-4 / MiniMax rule).
    __nv_bfloat16 *d_embed = nullptr;  // [V, H]
    __nv_bfloat16 *d_head = nullptr;   // [V, H]
    float *d_final_norm = nullptr;     // [H]
    int64_t embed_elems = 0;
    int64_t head_elems = 0;
    std::vector<StepLayerWeights> layers;
    // Routed experts: 30976 x ~10.6 MB NVFP4 do not fit the card, so the
    // neutral mmap ExpertStore streams them through a device slot LRU.
    moe::MoEExpertRuntime expert_store;
    gpu::ExpertDeviceCache expert_cache;
    std::int64_t expert_cache_bytes = 0;
    std::uint64_t resident_bytes = 0;
    void Free();
};

void FreeStepPinnedWeights(StepPinnedWeights &pw);

// Uploads the dense graph (globals + attention/dense-MLP/router/shared expert
// per layer) from <model_path>/logestix, then opens the routed-expert store and
// allocates the device slot cache.
bool LoadStepDenseWeights(const std::string &model_path, StepPinnedWeights &pw,
                          cudaStream_t stream, std::string &error);

// Reads model.index.json -> StepTopology (format/geometry gates).
StepTopology LoadStepTopology(const std::string &model_path);

// Geometry of the persistent KV cache (pos-major; the measured default).
KvCacheGeometry StepKvGeometry(const StepTopology &top, int64_t capacity);

} // namespace lsxcommon::step
