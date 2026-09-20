#pragma once

// GigaChat3.5 attention decode launchers (Plan 2b).
//
// MLA (full-attention layers): q_a/q_b and kv_a/kv_b latent projections, the
// interleaved YaRN RoPE over the 64-wide rope subspace, scaled softmax over a
// per-layer KV cache, sigmoid gating and o_proj. GatedDeltaNet lives in the
// companion GDN launcher; both are composed by the layer pipeline.
//
// All activations are F32; the NVFP4 GEMV boundary converts to/from FP16.

#include <cstdint>

#include <cuda_runtime.h>

#include "lsxcommon/gigachat_moe_load.h"

namespace lsxcommon::gigachat {

// Caller-selected MLA geometry (frozen for the real model; tests use small
// shapes). The projected weight shapes in GcLayerWeights must agree.
struct GcMlaDims {
    int hidden = 0;
    int heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int rope = 0;
    int v_head = 0;
};

// Per-launcher scratch, allocated once and reused across tokens. Pointers are
// private to the implementation.
struct GcMlaScratch {
    void *d_base = nullptr;
    int64_t bytes = 0;
    // Carved buffers (implementation detail).
    void *x_h = nullptr;
    void *q_a_h = nullptr;
    void *q_a_f = nullptr;
    void *q_a_n = nullptr;
    void *q_a_s = nullptr;
    void *q_h = nullptr;
    void *q_f = nullptr;
    void *kv_h = nullptr;
    void *kv_f = nullptr;
    void *kpass_n = nullptr;
    void *kpass_s = nullptr;
    void *kv_b_h = nullptr;
    void *kv_b_f = nullptr;
    void *k_rot = nullptr;
    void *gate_h = nullptr;
    void *attn_f = nullptr;
    void *attn_g_h = nullptr;
    void *out_h = nullptr;
    void *scores = nullptr;

    bool Alloc(const GcMlaDims &d, int capacity);
    void Free();
};

// Interleaved YaRN RoPE over the rope subspace of q [heads, qk] (at offset
// `nope`) and the single k vector [rope]; adjacent pairs, tables row `pos`.
// Exposed for the bit-exact rope test.
cudaError_t RunGigaChatRopeQk(float *q, float *k, int heads, int qk, int nope,
                              int rope, const float *cos_t, const float *sin_t,
                              cudaStream_t stream);

// One-token MLA decode. `x` is the pre-attention hidden state [hidden] F32.
// `rope_cos`/`rope_sin` are the YaRN tables [seqlen, rope/2] F32; row `pos` is
// used. `k_cache`/`v_cache` are [capacity, heads, qk]/[capacity, heads, vh]
// F32 row-major; entry `pos` is written before attention. `out` is [hidden] F32.
cudaError_t RunGigaChatMlaDecode(const float *x, const GcLayerWeights &lw,
                                 const GcMlaDims &d, float norm_eps,
                                 float attn_scale, const float *rope_cos,
                                 const float *rope_sin, int pos, int capacity,
                                 float *k_cache, float *v_cache,
                                 GcMlaScratch &sc, float *out,
                                 cudaStream_t stream);

} // namespace lsxcommon::gigachat
