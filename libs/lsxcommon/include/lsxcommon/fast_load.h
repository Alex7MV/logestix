#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lsxcommon {
namespace fastload {

// In-place device conversion of raw BF16 bits to FP16 bits. d_buf holds
// numel BF16 values on entry and their FP16 images on exit. Rounding is
// nearest-even via __float2half — bit-identical to the legacy host scalar
// path (bits<<16 reinterpreted as FP32). d_buf must be 16-byte aligned.
cudaError_t Bf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                              cudaStream_t stream);

} // namespace fastload
} // namespace lsxcommon
