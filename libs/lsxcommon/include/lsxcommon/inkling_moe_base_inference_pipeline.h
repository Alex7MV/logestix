#pragma once

// Inkling (base) generation pipeline (Model Ingestion Protocol stage 5).
//
// Residency classes:
//   * pinned   — embed/unembed FP16 tables, the NVFP4 attention projections,
//                the dense-MLP prefix, the 2 shared experts per MoE layer, the
//                router and every norm/side tensor. Device-resident.
//   * streamed — the 256 routed experts per MoE layer (~486 GiB NVFP4 on the
//                real geometry). One decode step stages the 6 chosen experts
//                of the current layer into a reusable slab; the neutral
//                ExpertStore owns the mmap views.
//
// The prompt is decoded token by token: the reference's conv state machine is
// single-position, so one code path serves the prompt and the generated tail.
// The checkpoint's 1M-token context is bounded here by kMaxContext.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_base_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::inkling::base {

// Engine-side context bound (a VRAM decision, not a model limit).
inline constexpr int kMaxContext = 8192;

struct Session {
    // One KV block per layer; the two layer classes have different kv widths
    // (sliding 16 heads / full 8), so the geometry is PER LAYER. Both are
    // pos-major; LayerStride differs with kv_heads.
    std::vector<__half *> d_k;
    std::vector<__half *> d_v;
    std::vector<KvCacheGeometry> geom;
    int64_t capacity = 0;
    int kv_rows = 0;
    int last_tok = -1;
    std::vector<int32_t> gen_tail;

    std::int64_t Bytes() const;
    void Free();
    void Reset();
    cudaError_t EnsureCapacity(const Topology &top, int64_t want,
                               cudaStream_t stream);
};

struct Scratch {
    void *arena = nullptr;
    int64_t arena_bytes = 0;
    // Named pointers INTO the arena (never freed individually).
    __half *x_h = nullptr;       // residual stream, fp16 [hidden]
    float *x_f = nullptr;        // fp32 copy of the residual stream
    __half *hh = nullptr;        // [hidden] normalized projection input
    __half *q = nullptr;         // [heads*head_dim]
    __half *kv_key = nullptr;    // [max_kv_dim]
    __half *kv_val = nullptr;
    float *qf = nullptr;         // [heads*head_dim] fp32 q for tau/log-scale
    float *relf = nullptr;       // [heads*d_rel]
    __half *attn_o = nullptr;    // [heads*head_dim]
    __half *o_out = nullptr;     // [hidden]
    __half *conv_out = nullptr;  // [max(hidden, max_kv_dim)]
    float *conv_f = nullptr;     // [hidden]
    float *conv_f32 = nullptr;   // [hidden] FP32 conv output (residual path)
    float *moe_f32 = nullptr;    // [hidden] FP32 MLP output before the conv
    float *residual = nullptr;   // [hidden] fp32
    float *rel_bias = nullptr;   // [heads * kMaxContext]
    __half *moe_gate = nullptr;  // [topk * I]
    __half *moe_up = nullptr;
    __half *moe_act = nullptr;
    __half *moe_down = nullptr;  // [topk * hidden]
    float *moe_down_f32 = nullptr;  // [topk * hidden] FP32 down output
    float *moe_acc = nullptr;    // [hidden] fp32 routed+shared accumulator
    __half *sh_gate = nullptr;   // [I] shared-expert scratch (one at a time)
    __half *sh_up = nullptr;
    __half *sh_act = nullptr;
    __half *sh_down = nullptr;   // [hidden]
    float *sh_down_f32 = nullptr;  // [hidden] FP32 shared down output
    float *logits = nullptr;     // [vocab]
    uint8_t *expert_stage = nullptr;
    int64_t expert_stage_bytes = 0;
    int32_t *d_ids = nullptr;    // [topk]
    void *d_slots = nullptr;     // [topk] RoutedSlot (kernels header type)
    float *d_gammas = nullptr;   // [shared] shared-expert weights
    int32_t *d_token = nullptr;
    float *d_maxv = nullptr;

    void Free();
    bool Allocate(const Topology &top, cudaStream_t stream);
};

struct GenerationResult {
    bool ok = false;
    std::vector<int32_t> tokens;
    std::vector<float> final_logits;
    int32_t first_token = -1;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
    std::string error_msg;
};

// Full-attention log-scale factor tau(pos) = 1 + alpha*log(max((pos+1)/floor,1)).
float LogScaleTau(const Topology &top, int pos);

cudaError_t RunGeneration(cudaStream_t stream, PinnedWeights &pw,
                          ExpertDb &experts, Session &session, Scratch &scratch,
                          const std::vector<int32_t> &input_ids,
                          int max_gen_tokens, const std::vector<int32_t> &eos_ids,
                          int unpadded_vocab, sample::ISampler &sampler,
                          GenerationResult &out, const GenerationStream *gs);

// Parity diagnostic: one token per position from 0, writing the residual
// stream after every layer ([num_layers+1][hidden], row 0 = post-embed_norm)
// plus the final logits row. Used by the layer-by-layer reference comparison.
cudaError_t RunForwardTrace(cudaStream_t stream, PinnedWeights &pw,
                            ExpertDb &experts, Session &session, Scratch &scratch,
                            const std::vector<int32_t> &tokens,
                            int unpadded_vocab, std::vector<float> &out_states,
                            std::vector<float> *out_logits);

}  // namespace lsxcommon::inkling::base