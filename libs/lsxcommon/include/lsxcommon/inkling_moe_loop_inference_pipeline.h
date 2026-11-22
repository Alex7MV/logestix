#pragma once

// Inkling-2-Mini generation pipeline.
//
// Residency classes, same discipline as every other family here:
//   * pinned   — embed/unembed (FP16), the attention projections, the router,
//                every norm and FP32 side tensor, the AttnRes banks, the
//                merged message queries and the loop controllers.
//   * streamed — the routed experts (neutral ExpertStore mmap views; the tree
//                is small enough that the LRU effectively pins everything).
//
// THE LOOP: layers [loop_start, loop_end) run `loop_passes` times per token.
// Pass 0 runs inline; every later pass restarts from the pre-loop hidden +
// block snapshots and receives per-layer messages retrieved from the PREVIOUS
// pass's block stack + final hidden. Each pass owns its OWN KV and conv-state
// slots: `top.CacheLayers()` = num_layers + (max_passes-1)*loop_layers slots
// (sized for the artifact's ceiling, so a varloop tree can change its pass
// count without re-allocating), addressed through KvCacheGeometry (pos-major)
// and the session's conv ring. `top.LayerApplications()` is the ACTIVE count.
//
// The prompt is decoded token by token (the reference's conv/AttnRes state
// machine is single-position), bounded by kMaxContext.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/inkling_moe_loop_kernels.h"
#include "lsxcommon/inkling_moe_loop_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon::inkling {

// Engine-side context bound: the KV block is allocated full-context for every
// cache slot, so this is a VRAM decision, not a model limit.
inline constexpr int kLoopMaxContext = 8192;

struct InklingLoopSession {
    __half *d_k = nullptr;
    __half *d_v = nullptr;
    KvCacheGeometry geom;         // layers == top.CacheLayers()
    int64_t capacity = 0;
    int kv_rows = 0;              // positions already written
    int last_tok = -1;            // the token the cache ends on
    std::vector<int32_t> gen_tail;
    // Per-cache-slot conv rings: [cache_layers][2*k*kv_dim + 2*k*hidden] F32.
    // Slot order matches CacheSlot(layer, pass). PER-SEQUENCE state: Reset()
    // zeroes every slot, or a second request inherits the first one's history.
    float *conv_state = nullptr;
    int64_t conv_slot_floats = 0; // floats per slot
    int64_t conv_state_bytes = 0;

    std::int64_t Bytes() const;
    void Free();
    void Reset(cudaStream_t stream);
    cudaError_t EnsureCapacity(const InklingLoopTopology &top, int64_t want,
                               cudaStream_t stream);
    // Conv ring base of one cache slot.
    float *ConvSlot(int slot) const
    {
        return conv_state + static_cast<int64_t>(slot) * conv_slot_floats;
    }
};

struct InklingLoopScratch {
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
    float *rel_bias = nullptr;  // [heads * kLoopMaxContext]
    float *taps_k = nullptr;    // [kv_dim * k] x2 (k and v)
    float *taps_v = nullptr;
    float *proj_k = nullptr;    // [kv_heads * k] x2
    float *proj_v = nullptr;
    float *stack = nullptr;     // [kAttnResMaxSources][hidden] fp32
    __half *moe_gate = nullptr; // [top_k * moe_intermediate]
    __half *moe_up = nullptr;
    __half *moe_act = nullptr;
    __half *moe_down = nullptr; // [top_k * hidden]
    float *moe_acc = nullptr;   // [hidden]
    float *logits = nullptr;    // [vocab] fp32
    // ── loop machinery ────────────────────────────────────────────────────
    // Pre-loop snapshots: the hidden state and block stack at loop entry,
    // restored at the start of every extra pass.
    float *pre_loop_hidden = nullptr;   // [hidden]
    float *pre_loop_stack = nullptr;    // [kAttnResMaxSources][hidden]
    int pre_loop_sources = 0;           // host mirror of the snapshot depth
    // The message sources of the previous pass, packed as ONE contiguous
    // [prev_sources][hidden] array for the retrieval kernel: its block states
    // (skipping the first one it appended, which merely re-states the pre-loop
    // hidden) followed by its final hidden state.
    float *prev_stack = nullptr;        // [kAttnResMaxSources][hidden]
    float *prev_hidden = nullptr;       // [hidden] the same final hidden, alone
    int prev_sources = 0;
    // Retrieved messages [4][hidden] and the MessageGate intermediates.
    float *msg = nullptr;               // [4*hidden]
    float *gate_h = nullptr;            // [2*hidden] normalized concat
    float *gate_rank = nullptr;         // [gate_rank]
    float *gate_out = nullptr;          // [hidden] gated stream
    // expert staging: [top_k][6 planes] laid out back to back
    uint8_t *expert_stage = nullptr;
    int64_t expert_stage_bytes = 0;
    // MXFP8 expert layers read E4M3 bytes directly (1 byte per element).
    uint8_t *fp8_stage = nullptr;
    int64_t fp8_stage_bytes = 0;
    int32_t *d_ids = nullptr;
    // The shared inkling2 router/accumulate kernels consume InklingRoutedSlot
    // (eid, coeff) directly.
    InklingRoutedSlot *d_slots = nullptr;
    int32_t *d_token = nullptr;
    float *d_maxv = nullptr;

    void Free();
    bool Allocate(const InklingLoopTopology &top, int64_t capacity,
                  cudaStream_t stream);
};

struct InklingLoopGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int first_token = -1;
    std::vector<float> final_logits;
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
};

// Decodes `input_ids` and then up to `max_gen_tokens` tokens. The session's
// conv rings and KV bookkeeping are per-sequence state and are reset at the
// start of every request.
cudaError_t RunInklingLoopGeneration(cudaStream_t stream,
                                     InklingLoopPinnedWeights &pw,
                                     InklingLoopExpertDb &experts,
                                     InklingLoopSession &session,
                                     InklingLoopScratch &scratch,
                                     const std::vector<int32_t> &input_ids,
                                     int max_gen_tokens,
                                     const std::vector<int32_t> &eos_ids,
                                     float logit_softcap, int unpadded_vocab,
                                     sample::ISampler &sampler,
                                     InklingLoopGenerationResult &out,
                                     const GenerationStream *gs);

// ── parity diagnostic ──────────────────────────────────────────────────────
// Runs ONE token through the stack and copies the residual stream after each
// layer APPLICATION into `out_states`: (LayerApplications() + 1) rows of
// `hidden` FP32 values, row 0 = post-embed_norm, then the applications in
// execution order (pre-loop layers, loop pass 0, loop pass 1, ...,
// post-loop layers). LayerApplications() follows the ACTIVE pass count; the
// KV/conv slab (`CacheLayers()`) is sized for the artifact's maximum.
cudaError_t RunInklingLoopForwardTrace(cudaStream_t stream,
                                       InklingLoopPinnedWeights &pw,
                                       InklingLoopExpertDb &experts,
                                       InklingLoopSession &session,
                                       InklingLoopScratch &scratch,
                                       const std::vector<int32_t> &tokens,
                                       float logit_softcap, int unpadded_vocab,
                                       std::vector<float> &out_states,
                                       std::vector<float> *out_logits);

}  // namespace lsxcommon::inkling