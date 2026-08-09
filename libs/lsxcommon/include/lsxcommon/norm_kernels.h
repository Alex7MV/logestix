#pragma once

#include <cuda_runtime.h>
#include <cstdint>

//==============================================================================
// High‑performance RMSNorm kernel suite — Blackwell (CC 12.0)
//
// rmsnorm_float
//   Row‑wise RMS normalisation:  output[r][i] = input[r][i] * gamma[i] / RMS
//   where  RMS = sqrt( mean( input[r]^2 ) + epsilon ).
//
//   hidden_dim  must be >= 1  (should be a multiple of 4 for vectorised paths).
//   batch_size  number of independent rows to process.
//
//   Returns cudaError_t (zero on success).
//==============================================================================

namespace lsxcommon {
namespace gpu {

cudaError_t rmsnorm_float(
    cudaStream_t      stream,
    int               batch_size,
    int               hidden_dim,
    const float*      d_input,
    const float*      d_gamma,
    float             epsilon,
    float*            d_output);

} // namespace gpu
} // namespace lsxcommon
