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

#include "lsxcommon/gigachat_moe_37b_inference_indexer.h"
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
    void *x_bf = nullptr;   // __nv_bfloat16 [hidden], the indexer's hidden input

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
//
// DSA hook: when `dsa` is set the decode first runs the layer's lightning
// indexer on the alpha_q-scaled q latent (same stream, before the score pass)
// and then masks the softmax to the selected keys. `dsa->sel`/`n_sel` may be
// left null/0 when `run_indexer` is false (a pre-computed selection), which is
// what the unit tests use.
struct GcDsaHook {
    GcDsaIndexerWeights *weights = nullptr;
    GcDsaIndexerDims dims;
    GcDsaIndexerScratch *scratch = nullptr;
    float *k_cache = nullptr;   // [capacity, index_head_dim] F32, this layer
    int32_t *sel = nullptr;     // selection out, ascending
    int *count = nullptr;       // selection count out (device)
    int capacity = 0;
    bool run_indexer = true;
    // Filled by the decode: min(index_topk, pos + 1), i.e. the number of keys
    // the softmax may attend to. Computed on the host from the geometry, never
    // read back from the device.
    int n_sel = 0;
};

// Per-variant attention options. Defaults are GigaChat3.5's behaviour: plain
// (`1+w`-free) verification of the zero-centred norm, no DSA selection.
struct GcAttnVariant {
    float norm_w_offset = 1.0f;  // 3.5 zero-centred `1+w`, 4.0 `w`
    bool norm_gate_from_scaled = false;  // 4.0 gates on the scaled state
    const int32_t *dsa_sel = nullptr;
    int dsa_n_sel = 0;
    GcDsaHook *dsa = nullptr;
};

// `dsa_sel`/`dsa_n_sel` are the DSA lightning-indexer selection for this token
// (nullptr = attend to the whole causal prefix, which is what every layer of a
// non-DSA model does). The selector returns exactly min(topk, pos+1) ascending
// indices, so the caller computes `dsa_n_sel` on the host and never reads the
// device count back; masked positions are set to -inf before the softmax, which
// is bit-identical to the unmasked path when nothing is masked.
cudaError_t RunGigaChatMlaDecode(const float *x, const GcLayerWeights &lw,
                                 const GcMlaDims &d, float norm_eps,
                                 float attn_scale, const float *rope_cos,
                                 const float *rope_sin, int pos, int capacity,
                                 float *k_cache, float *v_cache,
                                 GcMlaScratch &sc, float *out,
                                 cudaStream_t stream,
                                 const GcAttnVariant &variant = GcAttnVariant{});

// ── GatedDeltaNet decode ────────────────────────────────────────────────────

struct GcGdnDims {
    int hidden = 0;
    int k_heads = 0;
    int v_heads = 0;
    int hk = 0;      // key head dim
    int hv = 0;      // value head dim
    int conv_k = 0;  // depthwise conv kernel
};

struct GcGdnScratch {
    void *d_base = nullptr;
    int64_t bytes = 0;
    void *x_h = nullptr;
    void *x_bf16 = nullptr;
    void *qkvz_h = nullptr;
    void *qkvz_f = nullptr;
    void *ba_f = nullptr;
    void *mixed = nullptr;
    void *z = nullptr;
    void *beta = nullptr;
    void *g = nullptr;
    void *conv_out = nullptr;
    void *q = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *o = nullptr;
    void *onorm = nullptr;
    void *onorm_h = nullptr;
    void *out_h = nullptr;
    bool Alloc(const GcGdnDims &d);
    void Free();
};

// One-token GatedDeltaNet decode. `conv_state` is [conv_dim, conv_k-1] F32 and
// `rec_state` is [v_heads, hk, hv] F32; both are updated in place and must be
// zero-initialized at sequence start. `out` is [hidden] F32.
// `w_offset` is the GDN output-norm variant: 1.0 for GigaChat3.5's
// zero-centred scale (`1+w`), 0.0 for GigaChat4's plain `w`.
cudaError_t RunGigaChatGdnDecode(const float *x, const GcLayerWeights &lw,
                                 const GcGdnDims &d, float norm_eps,
                                 float o_norm_scale, float *conv_state,
                                 float *rec_state, GcGdnScratch &sc, float *out,
                                 cudaStream_t stream,
                                 float w_offset = 1.0f);

} // namespace lsxcommon::gigachat
