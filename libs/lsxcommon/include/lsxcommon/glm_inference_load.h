#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/schema.h"
#include "lsxcommon/expert_store.h"

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

    // Shared experts: one NVFP4-packed PackedExpert per MoE layer (3-77) from
    // pinned/shared_experts.arrow (w1=gate, w2=down, w3=up — same layout as a
    // routed expert). Kept HOST-side only: the generation pass stages each
    // layer's shared expert into the reusable per-expert device staging
    // buffers on demand, so the ~1.4 GiB shared set never has to live in VRAM
    // alongside the ~31 GiB pinned set. Index = layer; empty for dense layers.
    std::vector<PackedExpert> shared_host;

    // Peak device-resident weight set uploaded by LoadGlmPinnedWeights (embed
    // + head + final_norm + per-layer tensors), in bytes. Host-side data
    // (shared_host, h_gate_bias) is excluded — only VRAM counts.
    std::uint64_t resident_bytes = 0;

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

// ── On-demand routed-expert index (layers 3-78) ─────────────────────────────
// Host-side memory-mapped view of one routed expert: NVFP4 packed E2M1 nibble
// payloads + per-128-block E4M3 scales + decoded E8M0 global modifiers, all
// pointing straight into the Arrow IPC buffer of the mmap'd expert file
// (zero-copy — no per-expert heap copy). Sized like the DeepSeek ExpertEntry
// the lsxcommon loader uses for its own on-demand routed-expert path.
//
// Dequant invariant (identical to GlmExpertDeviceBlock):
//   w[i] = E2M1(nibble_i) * 2^(((scales[i>>4]>>3)&0xF) - 7) * global_scale
// A null host_w1 marks the entry missing/inactive (deterministic skip).
struct GlmExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
};

// Flat on-disk routed-expert registry for GLM-5.2. Entries are indexed by
// (layer * experts_per_layer + expert). Files are opened lazily (mmap) and
// cached through a bounded LRU so a full 19456-expert model never needs more
// than `max_open` CRT file descriptors at once (Windows cap is 8192) and host
// memory stays proportional to the resident set, mirroring the DeepSeek
// loader (ExpertDb in deepseek_inference_load.cuh). The generation loop calls
// ensure_glm_expert_resident per routed slot before uploading the selected
// expert's packed payload to device staging.
//
// THREAD-SAFETY: this struct is NOT internally synchronized. It is safe only
// because GlmV5Model::Infer() constructs, uses and destroys its own GlmExpertDb
// entirely within a single call (never shared across calls or threads) and
// LsxOrchestrator::ServeTaskMosaic is documented single-consumer. If a future
// change makes an IModel instance serve concurrent Infer() calls from
// multiple threads (or hoists a GlmExpertDb into persistent per-model state —
// see the Load()/Unload() design note on GlmV5Model), ensure_glm_expert_resident
// and load_glm_expert_index/free_glm_expert_index will need an external lock:
// concurrent LRU touch/evict mutates `lru`/`entries`/`mmaps` without any
// synchronization.
using GlmExpertDb = lsxcommon::gpu::expert_store::ExpertStore<GlmExpertEntry>;

// Builds the flat file-path index of every expert of a converted GLM-5.2
// model (<model_path>/logestix/experts/layer_<l>/exp_<e>.arrow, falling back to
// <model_path>/experts/... like the pinned loader) and warms the LRU cache up
// to max_open entries. Parsing is deferred until ensure_glm_expert_resident
// touches an index, so the full 19456-expert model costs no memory up front.
GlmExpertDb load_glm_expert_index(const std::string &model_path,
                                  int num_layers, int experts_per_layer);

// Ensures the mmap for (layer, expert) is resident in the LRU cache, evicting
// the least-recently-used entries as needed. No-op when already resident.
//
// `stream` is synchronized (once, only when an eviction is actually about to
// happen) before any mmap is unmapped: RunGlmExpertsStackOnDemand issues
// cudaMemcpyAsync() calls that read directly from an entry's mmap'd host
// memory (host_w1/host_w2/... — the whole point of the zero-copy design), and
// cudaMemcpyAsync() returning to the host does not mean the DMA engine has
// finished reading the source buffer yet. Without this barrier, evicting
// (munmap-ing) an entry whose async H2D copy is still in flight is a
// use-after-free read of unmapped host memory from the GPU's DMA engine.
void ensure_glm_expert_resident(GlmExpertDb &db, int layer, int expert,
                                cudaStream_t stream);

// Releases all expert mmaps (frees CRT file descriptors).
void free_glm_expert_index(GlmExpertDb &db);

} // namespace glm
} // namespace lsxcommon
