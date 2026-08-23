#pragma once

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

} // namespace kimi
} // namespace lsxcommon
