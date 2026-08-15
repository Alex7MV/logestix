#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/glm_inference_load.h"

namespace lsxcommon {
namespace glm {

// Forward pass over the heterogeneous Dense MLP layers 0-2 on ONE explicit
// async stream. MoE routing, expert loading and indexer ("IndexShare")
// attention logic are fully branched out for these layers: activations feed
// the layer norms, then pure FP16 cuBLAS gate_proj -> up_proj -> swiglu ->
// down_proj, residual-added per layer. hidden_size is hard-locked to
// kGlmHidden (6144); intermediate is kGlmMoeIntermediate (2048).
//
//   d_act:    [seq, hidden] FP16 input
//   d_alt:    [seq, hidden] FP16 ping-pong scratch
//   d_result: [seq, hidden] FP16 output (must be distinct from d_act/d_alt)
//   d_scratch: >= seq*(hidden + 2*intermediate) __half scratch
cudaError_t RunGlmDenseMLPStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, int seq,
    __half *d_act, __half *d_alt, __half *d_result, __half *d_scratch);

// Token embedding for the dense/MLP entry path: out[t][h] =
// embed[ids[t]][h] for a [seq] ids block into a [seq, hidden] FP16 buffer.
// Streamed on the caller's explicit async stream; the ONLY synchronization a
// caller performs is whatever it needs to publish the output.
cudaError_t RunGlmEmbed(cudaStream_t stream, const __half *d_embed,
                        const int32_t *d_ids, __half *d_out, int seq,
                        int hidden);

// Top-k expert slot cardinality for every MoE layer of GLM-5.2.
constexpr int kGlmMoETopK = 8;

// NVFP4 block granularity of the GLM expert weights: one E4M3 power-of-two
// block scale descriptor covers kGlmNvfp4Block consecutive elements of the
// packed E2M1 nibble stream. The per-tensor E8M0 global modifier is stored
// already decoded as a float (see GlmExpertDeviceBlock).
constexpr int kGlmNvfp4Block = 16;

// Element count of a single expert weight matrix: w1/w3 are [moe_intermediate,
// hidden], w2 is [hidden, moe_intermediate] — same 6144*2048 element count.
constexpr int64_t kGlmExpertMatEl =
    static_cast<int64_t>(kGlmMoeIntermediate) * kGlmHidden;

// One NVFP4-packed expert weight set resident in persistent device (VRAM)
// allocations. The packed payloads hold E2M1 nibbles (2 elements per byte, lo
// nibble first); each per-block scales array holds one E4M3 power-of-two
// descriptor per kGlmNvfp4Block elements; the global_scale floats are the
// already-decoded E8M0 tensor modifiers. Dequant invariant:
//   w[i] = E2M1(nibble_i) * 2^(((scales[i>>4]>>3)&0xF) - 7) * global_scale
// A descriptor with any null/zero payload pointer marks the expert inactive:
// routing slots that select it contribute nothing (deterministic skip).
struct GlmExpertDeviceBlock {
    const uint8_t *d_w1_packed = nullptr;
    const uint8_t *d_w1_scales = nullptr;
    float          w1_global_scale = 1.0f;
    int64_t        w1_packed_bytes = 0;
    int64_t        w1_scale_bytes  = 0;
    const uint8_t *d_w2_packed = nullptr;
    const uint8_t *d_w2_scales = nullptr;
    float          w2_global_scale = 1.0f;
    int64_t        w2_packed_bytes = 0;
    int64_t        w2_scale_bytes  = 0;
    const uint8_t *d_w3_packed = nullptr;
    const uint8_t *d_w3_scales = nullptr;
    float          w3_global_scale = 1.0f;
    int64_t        w3_packed_bytes = 0;
    int64_t        w3_scale_bytes  = 0;
};

// One routed-expert slot of the output routing grid. Exactly 8 bytes, so a
// token's 8 slots form an 8-byte-boundary-aligned record that the top-k
// kernel writes with single 64-bit stores (see glm_inference_load.cu).
//   eid:   selected expert index, [0, kGlmNumRouted)
//   coeff: unquantized softmax scaling coefficient of that expert
struct alignas(8) GlmRoutedSlot {
    int32_t eid;
    float   coeff;
};

// MoE router for layers 3-77: token -> top_k expert distribution grid.
// Accepts the RAW input activations of the token block (hidden_size =
// kGlmHidden = 6144, kGlmNumRouted = 256 total experts) and runs entirely on
// the caller's explicit async stream:
//
//   1. RMSNorm(d_hidden, lw.d_post_attn_norm) -> d_normed            [seq, hidden]
//   2. logits = d_normed @ lw.d_router_gate^T -> d_logits FP32       [seq, 256]
//   3. warp-per-token softmax + top_k (__shfl_down_sync selection) ->
//      d_routing [seq, kGlmMoETopK], 8-byte-aligned 64-bit grid writes.
//
// Selection key per expert is logit + bias (bias = optional per-expert
// correction, null when unbiased); the output coeff is the plain unquantized
// softmax weight of the selected expert. Ties resolve deterministically toward
// the smallest expert index, so the 8 winners are always distinct.
//
//   d_hidden:    [seq, kGlmHidden] FP16 raw layer input (residual, pre-norm)
//   d_gate_bias: [kGlmNumRouted] FP32 selection bias, or nullptr
//   d_normed:    [seq, kGlmHidden] FP16 scratch (RMSNorm out)
//   d_logits:    [seq, kGlmNumRouted] FP32 scratch (gate projections)
//   d_routing:   [seq, kGlmMoETopK] output grid (8-byte aligned)
cudaError_t RunGlmMoERouter(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmLayerWeights &lw, int seq,
    const __half *d_hidden, const float *d_gate_bias,
    __half *d_normed, float *d_logits, GlmRoutedSlot *d_routing);

// Real GLM-5.2 router (GlmMoeDsaTopkRouter) for the faithful generation pass.
// Identical entry contract to RunGlmMoERouter (post-attn RMSNorm -> FP64
// logits -> d_routing grid), but the top-k selection follows the real model:
// scores = sigmoid(logits), selection key = score + e_score_correction_bias
// (bias never leaks into the weights), then norm_topk_prob (÷(Σ+1e-20)) and
// routed_scaling_factor (×2.5). Grouping is a no-op (n_group = topk_group = 1).
cudaError_t RunGlmRealRouter(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmLayerWeights &lw, int seq,
    const __half *d_hidden, const float *d_gate_bias,
    __half *d_normed, float *d_logits, GlmRoutedSlot *d_routing);

// MoE expert stack for layers 3-77: NVFP4 block dequantization + SwiGLU
// projection, evaluating the locked GLM-5.2 path
//
//   out[t] += Σ_r coeff_r · w2( silu( w1(x_t) ) * w3(x_t) )
//
// with hidden_size locked to kGlmHidden (6144), intermediate to
// kGlmMoeIntermediate (2048). Weights are decoded inline from the packed E2M1
// nibble / per-16 E4M3 block-scale / E8M0 global-modifier layout straight
// into FP16 scratch and piped through cuBLAS FP16 GEMMs (CUBLAS_COMPUTE_32F,
// matching the dense-MLP stack). Every kernel/enqueue runs on the caller's
// explicit async stream; the ONLY synchronization is a single localized
// cudaStreamSynchronize used to stage the tiny routing grid back to a pinned
// host buffer so per-slot expert launches can be issued from the host. No
// cudaDeviceSynchronize is ever invoked.
//
//   d_routing: [seq, kGlmMoETopK] 64-bit slot grid from RunGlmMoERouter.
//   d_act:     [seq, kGlmHidden] FP16 input (RMS-normed activations).
//   h_experts: [n_experts] host descriptor registry; inactive blocks are
//              deterministically skipped when a slot selects them.
//   d_out:     [seq, kGlmHidden] FP16 result (accumulated, NOT residual-added).
//   d_scratch: >= kGlmExpertMatEl + kGlmHidden + 2*kGlmMoeIntermediate halves.
//   h_routing_staging: caller-provided pinned host buffer of at least
//              seq*kGlmMoETopK GlmRoutedSlot entries, reused across calls to
//              avoid a cudaMallocHost/cudaFreeHost pair per invocation, or
//              nullptr to have this function allocate and free it internally.
//              The buffer is only touched during the single staging barrier.
//
// Throughput contract: this is a parity-first, host-driven implementation. One
// kernel launch + one cudaStreamSynchronize (the routing-grid staging barrier)
// is issued per routed slot, plus the per-expert dequant/GEMM/accumulate
// sequence; decoding throughput is therefore bounded by launch latency, not by
// device compute. Callers needing throughput should batch routing slots or
// hoist per-expert state across turns; correctness (bit-exact parity) is not
// affected by these costs.
//
// Per-slot decode chain shared by RunGlmExpertsStack and
// RunGlmExpertsStackOnDemand:
//   inter = silu(w1^T @ x_t) * (w3^T @ x_t)
//   out_t += coeff * (w2^T @ inter)
// <eb> is one NVFP4 descriptor (row-major dequantized to FP16, then FP64
// GEMMs); inactive / foreign-format blocks contribute nothing. Runs entirely
// on <stream> with no synchronization. d_scratch follows the RunGlmExpertsStack
// layout: kGlmExpertMatEl + kGlmHidden + 2*kGlmMoeIntermediate halves.
cudaError_t RunGlmExpertSlot(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmExpertDeviceBlock &eb,
    const __half *x_t, __half *out_t, __half *d_scratch, float coeff);

cudaError_t RunGlmExpertsStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_out, __half *d_scratch,
    GlmRoutedSlot *h_routing_staging = nullptr);

// On-demand variant of RunGlmExpertsStack for layers 3-78: identical
// NVFP4-dequant / SwiGLU / FP64-GEMM / accumulate sequence and the same single
// localized routing-grid staging barrier, but the routed experts are read from
// a memory-mapped on-disk index (GlmExpertDb) instead of a pre-uploaded device
// descriptor table. For every occupied (token, expert) slot this function
//   1. calls ensure_glm_expert_resident(db, layer, eid) — mmap LRU touch,
//   2. copies the expert's packed E2M1 payloads + E4M3 scales from the host
//      mmap into the caller-provided device staging buffers,
//   3. runs the identical per-expert decode/GEMM chain against those staging
//      pointers (reusing one temporary descriptor, overwritten per slot).
// Inactive/missing experts contribute nothing (deterministic skip), matching
// the device-table contract. This is the VRAM-budgeted path the generation
// loop uses: a full 385 GiB expert set is never resident at once.
//
//   db:            flat host mmap index (load_glm_expert_index).
//   layer:         current MoE layer (3..78) — selects the db slice.
//   n_experts:     must be kGlmNumRouted (the routing grid's eid domain).
//   d_pack_stage:  device staging >= w1+w2+w3 packed payload bytes of one
//                  expert (kGlmExpertMatEl/2 * 3, ~19 MiB).
//   d_scale_stage: device staging >= w1+w2+w3 scale bytes of one expert
//                  (kGlmExpertMatEl/kGlmNvfp4Block * 3).
cudaError_t RunGlmExpertsStackOnDemand(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    GlmExpertDb &db, int layer, int n_experts,
    __half *d_out, __half *d_scratch,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    GlmRoutedSlot *h_routing_staging = nullptr);

// ── TDD Cycle 9: MoE router + NVFP4 expert stacks (layers 3-77) ─────────────
struct GlmExpertTables;  // fwd decl (defined in the Cycle-6 section below)

// ── TDD Cycle 10: unified arena — IndexShare KV + Layer-78 MTP spaces ───────
// The greedy-argmax staging slot the loop writes. Public (defined here) so the
// arena can carry both its device region and its pinned host copy as typed
// pointers while preserving the single-contiguous-block invariant.
struct GlmArgmaxSlot { int best; float maxv; };

// Single contiguous VRAM arena that holds every activation workspace of the
// MoE MLP forward for layers 3-77 — including one virtual activation slice per
// routed expert (per token row) — inside ONE cudaMalloc allocation plus ONE
// pinned host routing-staging buffer. The inference loop allocates the arena
// once and drives all 75 layers through it, so mid-flight runtime heap
// fragmentation is structurally outlawed.
//
// TDD Cycle 10 extends the SAME single device allocation to additionally carry
// the complete IndexShare K-V cache slots (GlmIndexShareState, visited by the
// mod-4 attention passes) and the Layer-78 MTP execution spaces (attention
// output, dense/MoE scratch, eh_proj/enorm norm buffers, shared-head normalized
// hidden and the [vocab] logits row). AllocateGeneration() therefore lays the
// whole 79-layer loop footprint — KV staging + MoE + attention + MTP + logits
// — into one contiguous device block, so a generation run performs exactly one
// device cudaMalloc and one host cudaMalloc overall.
struct GlmMoEArena {
    int seq       = 0;
    int hidden    = 0;
    int n_experts = 0;
    int vocab     = 0;
    size_t      bytes     = 0;   // total bytes of the single device allocation
    void        *d_base    = nullptr; // single contiguous device allocation
    __half      *d_normed  = nullptr; // [seq, hidden]            RMSNorm out
    float       *d_logits  = nullptr; // [seq, kGlmNumRouted]     router logits
    GlmRoutedSlot *d_routing = nullptr; // [seq, kGlmMoETopK]     routing grid
    __half      *d_slices  = nullptr; // [seq][n_experts][hidden] virtual expert slices
    __half      *d_moe     = nullptr; // [seq, hidden]            Σ coeff·slice
    __half      *d_scratch = nullptr; // expert dequant/inter/gate workspace
    float       *d_gate_bias = nullptr; // [kGlmNumRouted] router-bias staging
    GlmRoutedSlot *h_routing_staging = nullptr; // pinned host lookup grid

    // ── Cycle 10: full-loop footprint (all carved from the same d_base) ──────
    GlmIndexShareState *d_state = nullptr; // IndexShare K-V cache + shared masks
    __half *d_attn  = nullptr; // [seq, hidden]    attention output
    __half *d_gate  = nullptr; // [seq, inter]     dense/MoE gate scratch
    __half *d_up    = nullptr; // [seq, inter]     dense/MoE up scratch
    __half *d_cur   = nullptr; // [seq, hidden]    ping-pong activation A
    __half *d_alt   = nullptr; // [seq, hidden]    ping-pong activation B
    __half *d_hidrow = nullptr; // [hidden]        row-0 collapse buffer
    __half *d_mtp   = nullptr; // [hidden]         Layer-78 eh_proj output
    __half *d_hmtp  = nullptr; // [hidden]         Layer-78 enorm-normed
    __half *d_h78   = nullptr; // [hidden]         hmtp + moe
    __half *d_hn    = nullptr; // [hidden]         shared_head.norm output
    float  *d_logits_head = nullptr; // [vocab]     raw logits row
    GlmArgmaxSlot *d_argmax = nullptr;  // [1] greedy argmax (device)
    int32_t *d_ids = nullptr;  // [seq]             token id staging
    GlmArgmaxSlot *h_argmax = nullptr;  // pinned host argmax slot

    // Allocates the MoE-only workspace with a single cudaMalloc (device) + a
    // single cudaMallocHost (staging). Re-allocation without Free() is
    // forbidden: the caller must Free() before reusing with a larger seq.
    cudaError_t Allocate(int seq_, int hidden_, int n_experts_);
    // Allocates the complete 79-layer loop footprint — MoE + IndexShare KV +
    // Layer-78 MTP + logits — into ONE contiguous device allocation plus one
    // combined pinned host staging block (routing grid + argmax slot).
    cudaError_t AllocateGeneration(int seq_, int hidden_, int n_experts_,
                                   int vocab_);
    void Free();
};

// Per-token virtual-slice expert writer for layers 3-77. Drives the routing
// grid on <stream>: for every occupied (token, expert) slot that selects an
// active NVFP4 block it decodes w1/w3, evaluates the w2( silu(w1·x) * w3·x )
// projection and writes the per-expert contribution into the (token, expert)
// virtual activation slice; inactive/foreign slots contribute nothing
// (deterministic zero). A final kernel folds the co-scaled selected slices
// into `d_moe`, accumulating
//
//   moe[t][h] = Σ_r coeff_r · slices[(t·n_experts + eid_r)][h]
//
// entirely on the caller's async stream (the only synchronization is the
// single localized staging barrier that pulls the tiny routing grid to host,
// exactly like RunGlmExpertsStack).
//
//   d_routing:  [seq, kGlmMoETopK] routing grid from RunGlmMoERouter.
//   d_act:      [seq, hidden] RMS-normed activations (router output).
//   h_experts:  [n_experts] host descriptor registry; inactive blocks skip.
//   d_slices:   [seq][n_experts][hidden] contiguous virtual activation grid.
//   d_out:      [seq, hidden] FP16 accumulated result (NOT residual-added).
//   d_scratch:  >= kGlmExpertMatEl + hidden + 2*intermediate halves.
//   h_routing_staging: pinned host buffer of >= seq*kGlmMoETopK slots (may be
//                nullptr to have it allocated internally, but the arena owns
//                one that a caller reuses to avoid per-call MallocHost pairs).
cudaError_t RunGlmMoESlices(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_slices, __half *d_out, __half *d_scratch,
    GlmRoutedSlot *h_routing_staging);

// MoE MLP stack for the heterogeneous layers [first_layer, last_layer) of
// GLM-5.2 (the MoE range is 3..77): per layer,
//
//   RunGlmMoERouter   (post-attn RMSNorm -> top_k=8 routing grid)
//   RunGlmMoESlices   (NVFP4 expert stack -> per-expert virtual slices -> d_moe)
//   residual          out = d_moe + layer input (ping-pong)
//
// All work runs on the caller's explicit async stream through the caller-owned
// GlmMoEArena (single contiguous allocation); no cudaDeviceSynchronize is ever
// issued. hidden/intermediate geometry is locked to kGlmHidden/kGlmMoeInter.
// A layer missing its router gate/post-attn norm degrades to identity
// (activations pass through unchanged), mirroring the heterogeneous grid.
//
//   d_act:    [seq, hidden] FP16 input  (output of the dense stack layers 0-2)
//   d_alt:    [seq, hidden] FP16 ping-pong scratch (distinct from d_act)
//   d_result: [seq, hidden] FP16 final layer output (distinct from d_act)
//   arena:    must be Allocate()'d with n_experts >= kGlmNumRouted, hidden
//             == kGlmHidden, seq == the block length.
cudaError_t RunGlmMoEStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, const GlmExpertTables &experts,
    int first_layer, int last_layer, int seq,
    __half *d_act, __half *d_alt, __half *d_result,
    GlmMoEArena &arena);

// ── TDD Cycle 6: full 79-layer generation pass ─────────────────────────────
// Routed-expert registry for the generation pass: a per-layer table of
// NVFP4-packed GlmExpertDeviceBlock descriptors resident in persistent VRAM.
// `tables` is indexed by layer (num_layers entries, sized from the extended
// topology); a layer without a loaded expert set holds a full kGlmNumRouted
// table of all-inactive blocks (null payload pointers), so routing slots that
// select them contribute nothing — the same deterministic skip contract
// RunGlmExpertsStack enforces on inactive descriptors.
struct GlmExpertTables {
    std::vector<std::vector<GlmExpertDeviceBlock>> tables;

    void Free();
};

// Loads the routed-expert tables for every layer in `layer_filter` of a
// converted GLM-5.2 model (<model_path>/logestix/experts/layer_<l>/exp_<e>.
// arrow, `max_experts` experts per layer; falls back to
// <model_path>/experts/... like the pinned loader), uploading the packed E2M1
// nibble payloads, the per-block E4M3 scales and the decoded E8M0 global
// modifiers on <stream>. A missing or unreadable expert file leaves its table
// slot inactive (null payload pointers), which routing deterministically
// skips. `layer_filter` may be null (all layers).
cudaError_t LoadGlmExpertTables(const std::string &model_path,
                                GlmExpertTables &experts,
                                cudaStream_t stream,
                                const std::vector<int> *layer_filter,
                                int max_experts);

// Result of one greedy multi-token generation run.
struct GlmGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;  // greedy-argmax token ids, EOS-terminated
    float first_maxv = 0.0f;      // first generated token's argmax logit
    int  kv_slots_used = 0;       // decoder IndexShare slots after the run
    // TDD Cycle 10: raw [vocab] logits row produced by the final layer-78 MTP
    // forward (the row whose argmax would be the next hypothetical token).
    // Published for the SKWOS regression harness, which compares it
    // element-for-element against the independent CPU FP64 reference. Empty
    // when the run fails before its first decode forward.
    std::vector<float> final_logits;
};

// Full greedy multi-token generation over the frozen 79-layer GLM-5.2 loop on
// ONE explicit async stream (a caller-owned cublasHandle + cudaStream_t):
//
//   layers 0-2   Dense MLP (post-attn RMSNorm -> gate/up -> swiglu -> down)
//   layers 3-77  IndexShare attention + MoE router + NVFP4 expert stack
//   layer 78     MTP epilogue: eh_proj -> enorm -> router -> experts ->
//                residual -> shared_head.norm -> lm_head^T (raw logits row)
//
// Every layer runs IndexShare attention over the shared decoder KV staging
// (GlmIndexShareState, allocated + reset per call); only group heads
// (layer_idx % 4 == 0) carrying indexer weights refresh the shared masks.
// A layer whose pinned/MoE weights are absent degrades its branch to identity
// (attention still runs), mirroring the heterogeneous GLM grid. The generation
// loop: embed the prompt, forward all 79 layers (turn 0), then greedily argmax
// the layer-78 logits row, embed the winner and forward again per step
// (turn = step + 1) until max_new tokens or EOS. The first generated token's
// argmax logit is reported in `first_maxv`, and the decoder KV slot count after
// the run in `kv_slots_used`. When EOS is generated, the EOS token itself is
// NOT appended to the KV cache: `kv_slots_used` then equals
// prompt.size() + (tokens_until_eos) (i.e. one slot per generated token that
// was actually forwarded through the decoder).
//
// Capacity + input validation (memory safety): the decoder KV staging is fixed
// at kGlmIndexMaxSlots = 1024 slots (see GlmIndexShareState). This function
// returns cudaErrorInvalidValue (and leaves `result.ok` false) when
// prompt.size() + max_new > kGlmIndexMaxSlots, so the pre-allocated staging can
// never be written past its bounds, and when any prompt id falls outside
// [0, pw.vocab_size) (the embed table would be read out of bounds). Device-side
// guards in the KV append/attention kernels additionally clamp overflows, but
// the host validation makes the rejection deterministic and explicit.
//
// Throughput contract: parity-first, host-driven. One call allocates the
// pinned routing staging buffer for the whole run (hoisted out of
// RunGlmExpertsStack), then per generated token issues the full 79-layer
// forward with per-layer stream barriers; throughput is launch-latency-bound,
// not device-bound. Correctness is the goal; see RunGlmExpertsStack.
cudaError_t RunGlmGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, const GlmExpertTables &experts,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result);

// VRAM-budgeted generation pass over the same frozen 79-layer GLM-5.2 loop,
// driven by a memory-mapped expert index (GlmExpertDb) instead of a fully
// resident device table set. Routed experts are staged into two reusable
// per-call device buffers (one expert at a time) by RunGlmExpertsStackOnDemand;
// the loop allocates those staging buffers once and frees them on exit, so a
// full 385 GiB expert model never exceeds the pinned-weights footprint plus
// ~2 × one-expert staging. Semantics, capacity validation and the
// GlmGenerationResult contract are identical to the device-table overload.
cudaError_t RunGlmGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, GlmExpertDb &expert_db,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result);

// ── Faithful GLM-5.2 generation (real forward pass) ─────────────────────────
// Greedy multi-token generation over the REAL GlmMoeDsaForCausalLM forward:
//   layers 0-77 only (the layer-78 MTP block is explicitly NOT part of the
//   main model — transformers ignores model.layers.78.* on load),
//   per layer: input_layernorm -> MLA attention (q_a/q_b, kv_a/kv_b lora
//   expansion, interleaved RoPE θ=8e6, 64×256 heads, causal, o_proj) ->
//   residual -> post_attention_layernorm -> MLP -> residual,
//   MLP: dense (0-2) silu SwiGLU; MoE (3-77) sigmoid router top-8 +
//   norm_topk×2.5 routed experts + shared expert, routed+shared residual,
//   head: logits = lm_head(model.norm(layer77_out)) on the last token.
//
// Per-token causal decoding: each prompt token is forwarded through all 78
// layers in order, writing its MLA K/V into a per-layer expanded cache at
// slot = its position; decode steps follow the same loop at position P+step.
// The DSA indexer is mathematically a no-op within the decoder's KV capacity
// (index_topk=2048 >= capacity 1024, so every key is always selected) and is
// therefore not executed; attention is full causal. Runs entirely on the
// caller's async stream with a single sync per decode step for the argmax
// readback. VRAM-budgeted like the RunGlmGeneration(GlmExpertDb) overload:
// routed experts are mmap'd on demand; the shared experts + pinned weights
// are resident.
cudaError_t RunGlmRealGeneration(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw, GlmExpertDb &expert_db,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result);

} // namespace glm
} // namespace lsxcommon
