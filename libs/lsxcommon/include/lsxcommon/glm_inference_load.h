#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace glm {

// GLM-5.2 geometry (from config.json, verified against the real model).
constexpr int kGlmHidden = 6144;
constexpr int kGlmNumLayers = 79;
constexpr int kGlmNumHeads = 64;
constexpr int kGlmHeadDim = 192;         // qk_nope(192) + qk_rope(64) = 256 per kv head
constexpr int kGlmQkNope = 192;
constexpr int kGlmQkRope = 64;
constexpr int kGlmHeadV = 256;           // v_head
constexpr int kGlmQLoraRank = 2048;
constexpr int kGlmKVLoraRank = 512;
constexpr int kGlmMoeIntermediate = 2048;
constexpr int kGlmNumRouted = 256;
constexpr int kGlmNumShared = 1;
constexpr int kGlmIndexNHeads = 32;
constexpr int kGlmIndexHeadDim = 128;
constexpr float kGlmRmsEps = 1e-5f;

// Per-layer pinned weights for one GLM-5.2 layer. All weights are stored as
// raw BF16 in the logestix layout and uploaded as FP16 here (host-side
// BF16→FP16 conversion, mirroring the DeepSeek loader). Dense layers (0-2)
// fill the mlp_* fields, MoE layers fill the router fields; indexer layers
// additionally fill idx_*; the MTP block (layer 78) fills eh_*/enorm/hnorm/
// shared_head_norm. Pointers stay null when the layer file lacks the tensor.
struct GlmLayerWeights {
    // Layer norms
    __half *d_input_norm = nullptr;         // input_layernorm.weight
    __half *d_post_attn_norm = nullptr;     // post_attention_layernorm.weight

    // MLA attention (raw BF16 → FP16)
    __half *d_q_a_proj = nullptr;           // self_attn.q_a_proj.weight
    __half *d_q_a_norm = nullptr;           // self_attn.q_a_layernorm.weight
    __half *d_q_b_proj = nullptr;           // self_attn.q_b_proj.weight
    __half *d_kv_a_proj = nullptr;          // self_attn.kv_a_proj_with_mqa.weight
    __half *d_kv_a_norm = nullptr;          // self_attn.kv_a_layernorm.weight
    __half *d_kv_b_proj = nullptr;          // self_attn.kv_b_proj.weight
    __half *d_o_proj = nullptr;             // self_attn.o_proj.weight
    int64_t q_a_rows = 0, q_a_cols = 0;
    int64_t q_b_rows = 0, q_b_cols = 0;
    int64_t kv_a_rows = 0, kv_a_cols = 0;
    int64_t kv_b_rows = 0, kv_b_cols = 0;
    int64_t o_rows = 0, o_cols = 0;

    // Router (MoE layers 3-78): mlp.gate.weight (BF16) + e_score_correction_bias (F32)
    __half *d_router_gate = nullptr;        // mlp.gate.weight
    float  *h_gate_bias = nullptr;          // mlp.gate.e_score_correction_bias (host pinned)
    int     gate_bias_n = 0;

    // Dense MLP (layers 0-2): mlp.{gate,down,up}_proj.weight
    __half *d_mlp_gate = nullptr;
    __half *d_mlp_down = nullptr;
    __half *d_mlp_up = nullptr;

    // Indexer (indexer layers)
    __half *d_idx_wq_b = nullptr;           // self_attn.indexer.wq_b.weight
    __half *d_idx_wk = nullptr;             // self_attn.indexer.wk.weight
    __half *d_idx_weights_proj = nullptr;   // self_attn.indexer.weights_proj.weight
    __half *d_idx_k_norm = nullptr;         // self_attn.indexer.k_norm.weight
    float  *d_idx_k_norm_bias = nullptr;    // self_attn.indexer.k_norm.bias (F32)

    // MTP block (layer 78)
    __half *d_eh_proj = nullptr;            // eh_proj.weight
    __half *d_enorm = nullptr;              // enorm.weight
    __half *d_hnorm = nullptr;              // hnorm.weight
    __half *d_shared_head_norm = nullptr;   // shared_head.norm.weight

    void Free();
};

struct GlmPinnedWeights {
    __half *d_embed = nullptr;              // model.embed_tokens.weight [vocab, hidden]
    __half *d_head = nullptr;               // lm_head.weight [vocab, hidden]
    __half *d_final_norm = nullptr;         // model.norm.weight [hidden]
    int vocab_size = 0;
    int hidden = 0;
    std::vector<GlmLayerWeights> layers;    // indexed by layer 0..num_layers-1

    void Free();
};

// Reads <model_path>/logestix/pinned/{global,layer_<n>}.arrow for a
// GLM-5.2 converted model and uploads every pinned tensor to device memory as
// FP16 (host-side BF16→FP16 conversion). Per-layer files that don't exist are
// skipped (heterogeneous GLM layer grid). Sizes GlmPinnedWeights.layers from
// the extended topology (glm::load_topology). Returns cudaErrorInvalidValue on
// a missing pinned dir / unreadable global.arrow.
//
// When layer_filter is non-null, only the layer files whose index appears in
// the list are uploaded (global tensors are always uploaded). This lets a
// partial forward pass (e.g. the dense layers 0-2) avoid uploading the full
// 32 GiB pinned set.
cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream,
                                 const std::vector<int> *layer_filter);

cudaError_t LoadGlmPinnedWeights(const std::string &model_path,
                                 GlmPinnedWeights &pw,
                                 cudaStream_t stream);

// ── IndexShare attention state (TDD Cycle 5, layers 3-77) ──────────────────
// Persistent, pre-allocated "virtual staging" block that carries the central
// memory-mapped shared indexer mask arrays across every call of
// RunGlmIndexShareAttention:
//   kv      [slots][kGlmHidden]       FP16 cached token values (KV cache)
//   q_key   [slots][kGlmIndexHeadDim] FP16 cached winner-head queries (wq_b)
//   key     [slots][kGlmIndexHeadDim] FP16 normalized projected keys (wk)
//   score   [slots][kGlmIndexNHeads]  FP32 last-refresh per-head scores
//   slot    [slots]                   winning (argmax) head index per slot
// The mask arrays (q_key/key/score/slot) are calculated, block-aligned and
// refreshed ONLY on the structural modulo check (layer_idx % 4 == 0). All
// mask arrays start on a 16-byte boundary; the struct is one single cudaMalloc
// allocation (>= sizeof(GlmIndexShareState), ~12.7 MiB at kGlmIndexMaxSlots
// 1024), so the KV cache lives directly inside the pre-allocated staging.
constexpr int kGlmIndexMaxSlots = 1024;

struct alignas(16) GlmIndexShareState {
    __half kv[kGlmIndexMaxSlots][kGlmHidden];
    __half q_key[kGlmIndexMaxSlots][kGlmIndexHeadDim];
    __half key[kGlmIndexMaxSlots][kGlmIndexHeadDim];
    float  score[kGlmIndexMaxSlots][kGlmIndexNHeads];
    int    slot[kGlmIndexMaxSlots];
    int    n_used        = 0;   // occupied slot count (device-maintained)
    int    turn_id       = -1;  // last appended turn
    int    refresh_layer = -1;  // last layer that refreshed the masks
};

static_assert(offsetof(GlmIndexShareState, kv) % 16 == 0 &&
                  offsetof(GlmIndexShareState, q_key) % 16 == 0 &&
                  offsetof(GlmIndexShareState, key) % 16 == 0 &&
                  offsetof(GlmIndexShareState, score) % 16 == 0 &&
                  offsetof(GlmIndexShareState, slot) % 16 == 0,
              "IndexShare mask arrays must stay 16-byte block-aligned");

// Asynchronously resets a GlmIndexShareState to the untouched state
// (n_used=0, turn_id=-1, refresh_layer=-1) on <stream>. Must be called once
// per allocation before the first RunGlmIndexShareAttention, exactly mirroring
// LoadGlmPinnedWeights for the pinned-weight sets. The whole struct (including
// the kv/q_key/key/score/slot arrays) is cudaMemsetAsync'ed to zero first, so
// re-entry never observes stale mask content from a previous run or a state
// buffer that was reused without Free().
cudaError_t GlmIndexShareReset(GlmIndexShareState *d_state, cudaStream_t stream);

// IndexShare attention for ONE layer (layer_idx in [0,78] — relaxed for the
// full 79-layer generation loop; layers 0-2 and the layer-78 MTP epilogue
// still attend over the shared KV staging, exactly like the intermediate MoE
// layers) entirely on the caller's explicit async stream (no host-side
// synchronization fence is ever issued by this pass).
//
// The rigid structural modulo check (layer_idx % 4 == 0) drives the refresh
// of the central shared indexer mask arrays from this layer's indexer weights
// (indexer.wq_b with input rank kGlmQLoraRank, indexer.wk, indexer.weights_proj
// and indexer.k_norm weight + bias):
//   score[h] = dot(kv[slot], weights_proj[h])     -> winning head slot[τ]
//   key[r]   = RMSNorm(dot(kv[slot], wk[r])) + k_norm_bias[r]
//   q_key[r] = dot(kv[slot][0:kGlmQLoraRank], wq_b[head{τ}][r])
// For every intermediate layer (layer_idx % 4 != 0) the execution thread
// blocks completely bypass that matrix sorting: they pull the cached sparse
// token memory pointers straight from the pre-allocated virtual staging
// buffers (d_state) and never read the indexer matrices. A layer that passes
// the modulo check but carries no indexer weights (null pointers) degrades to
// reuse as well.
//
// KV registration happens once per turn (first call with a new <turn>): the
// seq hidden vectors of that turn are appended to the staging KV cache. Turn
// tokens are then attended over every occupied slot with the frozen cached
// masks; Multi-turn decoding therefore keeps old slots alive across turns.
//
// Capacity contract: the staging KV cache is fixed at kGlmIndexMaxSlots slots.
// RunGlmGeneration validates prompt+generated against the capacity up front;
// a caller that bypasses it is still memory-safe because the append kernel
// consumes an over-capacity turn without writing and the attention kernel
// clamps its occupancy to kGlmIndexMaxSlots. The host-side validation makes
// the rejection deterministic and explicit.
//
// Attention per token (deterministic, elementwise): sim(s) = dot(q_key[τ],
// key[s]) * kGlmIndexAttnScale, softmax over the occupied slots, then
// out = x + Σ_s p(s)·kv[s] (residual added). The per-block key/query cache is
// staged in CUDA shared memory; KV values stay in the staging buffer.
//
//   layer_idx: in [0, 78]; the mod-4 group head recomputes the shared masks.
//   turn:      monotonically increasing token-block id; seq must stay constant
//              within a turn.
//   lw:        per-layer weights; only the indexer fields are consumed.
//   seq:       tokens in the current turn's block.
//   d_hidden:  [seq, hidden] FP16 input hidden states.
//   d_out:     [seq, hidden] FP16 result (residual-added); distinct from
//              d_hidden. Guarded output tails must be pre-filled by the caller.
//   d_state:   persistent shared indexer mask arrays (see GlmIndexShareState).
cudaError_t RunGlmIndexShareAttention(
    cudaStream_t stream, int layer_idx, int turn,
    const GlmLayerWeights &lw, int seq,
    const __half *d_hidden, __half *d_out, GlmIndexShareState *d_state);

} // namespace glm
} // namespace lsxcommon
