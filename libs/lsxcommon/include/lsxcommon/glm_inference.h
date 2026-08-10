#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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

} // namespace glm
} // namespace lsxcommon