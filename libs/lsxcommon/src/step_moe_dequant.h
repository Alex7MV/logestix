// step_moe_dequant.h — Step-5-Preview converter-side FP8 block dequant.
//
// The routed experts ship as FUSED F8_E4M3 stacks [E, N, K] with a
// [E, ceil(N/128), ceil(K/128)] F32 scale grid (`weight_scale_inv`), and the
// reference multiplies (not divides) by that scale. The converter dequantizes
// the stack to BF16 on the GPU and then runs the sanctioned
// gpu::quantize_tensor_gpu path per expert — no family-owned NVFP4 codec.
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace lsxcommon::step {

// Dequantizes `experts` stacked [rows, cols] F8_E4M3 matrices, each with its
// own [ceil(rows/128), ceil(cols/128)] F32 scale tile, into packed BF16.
// scale index layout: scales[e][row/128][col/128].
cudaError_t LaunchStepFp8StackedDequantToBf16(const uint8_t *w,
                                              const float *scales, int experts,
                                              int rows, int cols,
                                              uint16_t *out,
                                              cudaStream_t stream);

// Host wrapper: consumes host FP8 + scale buffers, returns host BF16 bytes.
// Returns false on any CUDA failure.
bool StepFp8StackedDequantToBf16Host(const uint8_t *w, const float *scales,
                                     int experts, int rows, int cols,
                                     std::vector<uint8_t> &out_bf16);

} // namespace lsxcommon::step
