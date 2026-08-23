#pragma once

#include "lsxcommon/kimi_inference_load.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon {
namespace kimi {

// Dequantizes compressed-tensors INT4 group-32 weights to BF16 bits.
// d_packed: int32 words, 8 nibbles each (low nibble = element first),
//   row-major [rows, cols/8].
// d_scales_bf16: BF16 bits [rows, cols/32], one scale per group of 32 along
//   the input (cols) dimension.
// d_out_bf16: BF16 bits [rows, cols]. Stream-async; no synchronization.
cudaError_t KimiInt4DequantToBf16(const int *d_packed,
                                  const uint16_t *d_scales_bf16,
                                  int rows, int cols,
                                  uint16_t *d_out_bf16,
                                  cudaStream_t stream);

// Kimi noaux_tc router: sigmoid scoring + e_score_correction_bias selection,
// top-k with lower-id tie-break, norm_topk_prob renormalization and
// routed-scaling. d_scores/d_bias: FP32 [n_routed]. Outputs: d_out_ids int
// [top_k], d_out_weights FP32 [top_k]. Stream-async.
//
// The host variant (KimiRouterTopKHost) computes the identical math on CPU
// for tests/fallbacks; both share the tie-break rule so ids match exactly.
void KimiRouterTopKHost(const float *scores, const float *bias, int n_routed,
                        int top_k, int *out_ids, float *out_weights);

cudaError_t KimiRouterTopK(const float *d_scores, const float *d_bias,
                           int n_routed, int top_k,
                           int *d_out_ids, float *d_out_weights,
                           cudaStream_t stream);

// Routed-expert NVFP4 FFN for ONE token: dequantizes the expert's three
// projections from the mmap'd entry, applies gate/up GEMMs + SwiGLU + down
// GEMM and accumulates `expert_weight * y` into d_out (caller pre-zeroes or
// seeds d_out). d_staging must hold >= RunKimiExpertFfnStagingBytes bytes.
// Staging layout is internal; sizes are derived from kKimi* geometry.
cudaError_t RunKimiExpertFfn(cublasHandle_t cublas_h, cudaStream_t stream,
                             const KimiExpertEntry &expert,
                             float expert_weight, const __half *d_x,
                             __half *d_out, void *d_staging,
                             size_t staging_bytes);

// Upper bound of the staging requirement of RunKimiExpertFfn.
size_t KimiExpertFfnStagingBytes();

} // namespace kimi
} // namespace lsxcommon
