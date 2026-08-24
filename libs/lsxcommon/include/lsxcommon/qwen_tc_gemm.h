// qwen_tc_gemm.h - Blackwell tensor-core NVFP4 W4A4 fast path (audit Phase 3,
// F-01). Compiled EXCLUSIVELY for compute_120a/sm_120a + compute_121a/sm_121a:
// the kind::mxf4nvf4 PTX qualifier does not assemble on plain sm_120 targets
// (PTX ISA 9.3). All entry points are stream-ordered and allocate nothing.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon {
namespace qwen {
namespace tc {

// Quantize FP16 activations to NVFP4 (per-16 E4M3 power-of-2 scale +
// E2M1 nibble pairs), matching the repo's canonical weight-side codec
// (schema.h float_to_e4m3_p2 / nvfp4_codec.h decode). Output layouts:
// d_packed [seq, K/2] row-major nibble pairs (low nibble = even k),
// d_scales [seq, K/16] row-major, one byte per 16-element block.
// Buffers must be preallocated by the caller (device memory).
cudaError_t QuantizeActsNvfp4(cudaStream_t stream, const __half *x, int seq,
                              int K, uint8_t *d_packed, uint8_t *d_scales);

} // namespace tc
} // namespace qwen
} // namespace lsxcommon
