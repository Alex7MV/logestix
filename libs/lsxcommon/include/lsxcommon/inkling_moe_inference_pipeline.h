#pragma once

// Inkling-2-Small generation pipeline.
//
// Residency classes, same discipline as every other family here:
//   * pinned   — embed/unembed, the attention projections, the router, every
//                norm and every FP32 side tensor (~4.7 GiB of NVFP4 + ~350 MiB
//                of FP32 routers and clips). Device-resident for the session.
//   * streamed — the routed experts. 594B parameters cannot live on a 32 GiB
//                card, so one decode step stages the 8 chosen experts of the
//                current layer into a reusable buffer (the neutral
//                `ExpertStore` owns the mmap, this file owns the staging copy).
//
// The prompt is decoded token by token rather than in a prefill pass: the
// reference's conv/AttnRes state machine is single-position, so one code path
// serves both the prompt and the generated tail, and the 1M-token context of
// the checkpoint is bounded here by `kMaxContext`.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/inkling_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::inkling {

// Engine-side context bound: the KV block is allocated full-context for every
// layer, so this is a VRAM decision (34 MiB per 2048 positions), not a model
// limit.
inline constexpr int kMaxContext = 8192;

struct InklingSession {
    __half *d_k = nullptr;
    __half *d_v = nullptr;
    KvCacheGeometry geom;
    int64_t capacity = 0;
    int kv_rows = 0;              // positions already written
    int last_tok = -1;            // the token the cache ends on
    std::vector<int32_t> gen_tail;

    std::int64_t Bytes() const;
    void Free();
    void Reset();
    cudaError_t EnsureCapacity(const InklingTopology &top, int64_t want,
                               cudaStream_t stream);
};

struct InklingScratch {
    void *arena = nullptr;
    int64_t arena_bytes = 0;
    // Named pointers INTO the arena (never freed individually).
    __half *x_h = nullptr;      // residual stream, fp16
    float *x_f = nullptr;       // fp32 copy of the residual stream
    float *mix = nullptr;       // [3*hidden] AttnRes outputs
    __half *hh = nullptr;       // [hidden] normalized projection input
    __half *qh = nullptr;
    __half *kh = nullptr;
    __half *vh = nullptr;
    __half *q = nullptr;        // [heads*head_dim]
    __half *kv_key = nullptr;   // [kv_heads*head_dim]
    __half *kv_val = nullptr;
    float *qf = nullptr;
    float *kf = nullptr;
    float *vf = nullptr;
    float *relf = nullptr;      // [heads*d_rel]
    float *sink = nullptr;      // [heads]
    __half *attn_gate = nullptr;// [heads*head_dim]
    __half *attn_o = nullptr;   // [heads*head_dim]
    __half *o_out = nullptr;    // [hidden]
    __half *conv_out = nullptr; // [max(hidden, kv_dim)]
    float *conv_f = nullptr;    // [hidden] fp32 view of a conv output
    float *residual = nullptr;  // [hidden] fp32
    float *rel_bias = nullptr;  // [heads * capacity]
    float *taps_k = nullptr;    // [kv_dim * k]
    float *taps_v = nullptr;
    float *proj_k = nullptr;    // [kv_heads * k]
    float *proj_v = nullptr;
    float *stack = nullptr;     // [kAttnResMaxSources][hidden] fp32
    __half *moe_gate = nullptr; // [top_k * moe_intermediate]
    __half *moe_up = nullptr;
    __half *moe_act = nullptr;
    __half *moe_down = nullptr; // [top_k * hidden]
    float *moe_acc = nullptr;   // [hidden]
    float *logits = nullptr;    // [vocab] fp32
    // expert staging: [top_k][6 planes] laid out back to back
    uint8_t *expert_stage = nullptr;
    int64_t expert_stage_bytes = 0;
    int32_t *d_ids = nullptr;
    InklingRoutedSlot *d_slots = nullptr;
    int32_t *d_token = nullptr;
    float *d_maxv = nullptr;

    void Free();
    bool Allocate(const InklingTopology &top, int64_t capacity,
                  cudaStream_t stream);
};

struct InklingGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int first_token = -1;
    std::vector<float> final_logits;
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
};

// Decodes `input_ids` and then up to `max_gen_tokens` tokens. `forbidden_id`
// (when >= 0) is masked out of the first step's logits, which is how the FSI /
// CODE prompt sets pin a frame the model must not answer with.
// `pw` is mutable: the four per-layer short-convolution states are recurrent
// sequence state and are reset at the start of every request.
cudaError_t RunInklingGeneration(cudaStream_t stream,
                                 InklingPinnedWeights &pw,
                                 InklingExpertDb &experts,
                                 InklingSession &session, InklingScratch &scratch,
                                 const std::vector<int32_t> &input_ids,
                                 int max_gen_tokens,
                                 const std::vector<int32_t> &eos_ids,
                                 float logit_softcap, int unpadded_vocab,
                                 sample::ISampler &sampler,
                                 InklingGenerationResult &out,
                                 const GenerationStream *gs);

// ── parity diagnostic ──────────────────────────────────────────────────────
// Runs ONE token through the stack and copies the residual stream after each
// layer into `out_states`: (num_layers + 1) rows of `hidden` FP32 values, where
// row 0 is the post-`embed_norm` embedding and row l+1 follows layer l.
//
// This exists for the parity investigation recorded in
// docs/open-tasks/2026-11-14-inkling-2-small-ingest.md (the real-weight
// SelfCheck is red): without per-layer states the only observable is the decoded
// text, and the defect could be anywhere in 48 layers. The caller compares the
// rows against a host reference computed from the same converted artifact, so a
// divergence names the layer. Not used by the decode path.
// `tokens` are run at positions 0..n-1 in order and the states are dumped at the
// LAST position, so the four conv rings and the KV cache are populated exactly
// as they are during a real decode. With n == 1 this is the position-0 trace.
cudaError_t RunInklingForwardTrace(cudaStream_t stream,
                                   InklingPinnedWeights &pw,
                                   InklingExpertDb &experts,
                                   InklingSession &session, InklingScratch &scratch,
                                   const std::vector<int32_t> &tokens,
                                   float logit_softcap, int unpadded_vocab,
                                   std::vector<float> &out_states);

// The engine's position-dependent log-scaling factor (1.0 on the sliding
// layers, where it is never applied).
float InklingLogScaleTau(const InklingTopology &top, int pos);

}  // namespace lsxcommon::inkling
