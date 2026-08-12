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
constexpr int kGlmNvfp4Block = 128;

// Element count of a single expert weight matrix: w1/w3 are [moe_intermediate,
// hidden], w2 is [hidden, moe_intermediate] — same 6144*2048 element count.
constexpr int64_t kGlmExpertMatEl =
    static_cast<int64_t>(kGlmMoeIntermediate) * kGlmHidden;

// One NVFP4-packed expert weight set resident in persistent device (VRAM)
// allocations. The packed payloads hold E2M1 nibbles (2 elements per byte, lo
// nibble first); each per-block scales array holds one E4M3 power-of-two
// descriptor per kGlmNvfp4Block elements; the global_scale floats are the
// already-decoded E8M0 tensor modifiers. Dequant invariant:
//   w[i] = E2M1(nibble_i) * 2^(((scales[i>>7]>>3)&0xF) - 7) * global_scale
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

// MoE expert stack for layers 3-77: NVFP4 block dequantization + SwiGLU
// projection, evaluating the locked GLM-5.2 path
//
//   out[t] += Σ_r coeff_r · w2( silu( w1(x_t) ) * w3(x_t) )
//
// with hidden_size locked to kGlmHidden (6144), intermediate to
// kGlmMoeIntermediate (2048). Weights are decoded inline from the packed E2M1
// nibble / per-128 E4M3 block-scale / E8M0 global-modifier layout straight
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
cudaError_t RunGlmExpertsStack(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmRoutedSlot *d_routing, int seq,
    const __half *d_act,
    const GlmExpertDeviceBlock *h_experts, int n_experts,
    __half *d_out, __half *d_scratch,
    GlmRoutedSlot *h_routing_staging = nullptr);

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

} // namespace glm
} // namespace lsxcommon
