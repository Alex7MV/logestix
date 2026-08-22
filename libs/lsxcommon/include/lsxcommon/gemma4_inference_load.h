#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace gemma {

// In-place device BF16-bit -> FP16-bit conversion, nearest-even, bit-identical
// to the host scalar path (pattern: GlmBf16ToHalfInPlace). d_buf must be
// 16-byte aligned.
cudaError_t GemmaBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                   cudaStream_t stream);

} // namespace gemma
} // namespace lsxcommon
