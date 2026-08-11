#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

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

// Top-k expert slot cardinality for every MoE layer of GLM-5.2.
constexpr int kGlmMoETopK = 8;

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

} // namespace glm
} // namespace lsxcommon