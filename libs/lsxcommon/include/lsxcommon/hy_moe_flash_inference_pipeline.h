// hy_moe_flash_inference_pipeline.h — Hy4-flash token-by-token decode engine.
//
// One declaration shared by the model wrapper and the pipeline test, so the
// signature cannot drift.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

#include "hy_moe_flash_load.h"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

// Incremental session: the MLA latent cache (kv_lora + rope per position, one
// copy for all heads), the IndexPool pooled-key cache of the sparse layers, and
// the generated tail. Session state is created and published INSIDE
// RunUnderResidencyLock (see the model TU).
struct Session {
    __half *d_latent = nullptr;      // [layers][capacity][kv_lora + rope]
    KvCacheGeometry geom;
    int64_t capacity = 0;
    int64_t kv_rows = 0;             // positions cached so far
    // Per sparse layer: one pooled key per COMPLETE group, and the token keys of
    // the group currently being filled (the pooling input).
    std::vector<__half *> pooled;      // [capacity/ratio][index_dim]
    std::vector<__half *> group_keys;  // [ratio][index_dim]
    std::vector<int> group_filled;     // how many keys the current group holds
    int32_t last_tok = -1;
    std::vector<int32_t> gen_tail;

    bool allocated() const { return d_latent != nullptr; }
    std::int64_t Bytes() const;
    void Free();
    cudaError_t EnsureCapacity(int64_t capacity, cudaStream_t stream);
    void Reset();
};

// Per-call device scratch, reused across Infer calls (grown on demand).
struct Scratch {
    void *arena = nullptr;
    size_t arena_bytes = 0;
    float *streams = nullptr;    // [hc_mult * hidden] fp32
    float *agg = nullptr;        // [hidden] fp32
    float *pre = nullptr;        // [hc_mult]
    float *post = nullptr;       // [hc_mult]
    float *mix = nullptr;        // [2*hc_mult]
    float *rsqrt = nullptr;      // [1]
    __half *x = nullptr;         // [hidden]
    __half *q = nullptr;         // [max_heads * qk_head]
    __half *qr = nullptr;        // [q_lora]
    __half *kv = nullptr;        // [kv_lora + rope]
    __half *latent = nullptr;    // [kv_lora + rope]
    float *qk = nullptr;         // [max_heads * kv_lora]
    float *scores = nullptr;     // [max_heads * max_visible]
    float *acc = nullptr;        // [max_heads * kv_lora]
    __half *o = nullptr;         // [max_heads * v_head]
    __half *gate = nullptr;      // [max_heads * v_head]
    __half *attn_out = nullptr;  // [hidden]
    __half *ffn_out = nullptr;   // [hidden]
    float *moe_acc = nullptr;    // [latent]
    __half *lat = nullptr;       // [latent]
    __half *gu = nullptr;        // [2 * moe_intermediate]
    __half *act = nullptr;       // [moe_intermediate]
    __half *s_gu = nullptr;      // [2 * shared_intermediate]
    __half *s_act = nullptr;     // [shared_intermediate]
    // The dense FFN (layer 0) is 13824 wide, not 1536: it needs its OWN buffers.
    // Sizing the dense SwiGLU into the MoE scratch is a 27k-element write into a
    // 3k-element allocation, and the driver faults later, far from the cause.
    __half *d_gu = nullptr;      // [2 * dense_intermediate]
    __half *d_act = nullptr;     // [dense_intermediate]
    int32_t *pos = nullptr;      // [max_visible]
    float *idx_q = nullptr;      // [index_n_heads * index_head_dim] fp32
    __half *idx_key = nullptr;   // [index_head_dim] (this token's key)
    float *idx_scores = nullptr;       // [max groups]
    unsigned char *idx_used = nullptr; // [max groups]
    float *idx_wh = nullptr;     // [index_n_heads] head weights
    float *router_scores = nullptr;  // [num_routed]
    int32_t *router_idx = nullptr;   // [top_k]
    float *router_w = nullptr;       // [top_k]
    float *logits = nullptr;         // [vocab] fp32
    int32_t *d_tok = nullptr;
    int32_t *d_argmax = nullptr;
    float *d_maxv = nullptr;
    // Expert staging / device cache.
    uint8_t *d_slots = nullptr;
    size_t slot_bytes = 0;
    int slot_count = 0;
    void Free();
};

// End-to-end generation: token-by-token prefill + decode, streaming one Emit per
// decoded token and polling gs->Aborted() between steps.
::lsxcommon::gpu::InferenceResult RunHy4Inference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    const PinnedWeights &pw, ExpertDb &expert_db, Session &session,
    Scratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs);

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
