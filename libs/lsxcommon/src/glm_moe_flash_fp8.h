#pragma once

// Host-reference FP8(blockwise 128x128 scale_inv) -> NVFP4 requant.
// Bit-compatible with the GPU converter path (safetensors_gpu.cu
// quantize_tensor_gpu): identical E2M1 nearest tie flavor, pow2-E4M3
// per-16 block scales (ceil, FloatToE4m3P2), and the
// global_scale = global_max_abs / 6 convention
// (dequant = E2M1 * e4m3_p2 * global_scale).
//
// Host-only; never include from .cu/.cuh TUs.
#include <cstdint>
#include <vector>

#include "nvfp4_scale.h"

namespace lsxcommon::glm_moe_flash {

struct FlashNvfp4Blob {
    std::vector<uint8_t> packed;  // E2M1 nibbles, 2 per byte, row-major
    std::vector<uint8_t> scales;  // pow2-E4M3 bytes, one per 16-elem block
    float global_scale = 1.0f;
    int rows = 0, cols = 0;
};

// Dequantizes fp8 [rows, cols] (F8_E4M3 bytes) with per-128x128 F32
// scale_inv tiles then requantizes into NVFP4 using the exact GPU-path
// convention. scale_inv layout: [ceil(rows/128), ceil(cols/128)] row-major.
FlashNvfp4Blob RequantFp8ToNvfp4Host(const void *fp8_data,
                                     const float *scale_inv, int rows,
                                     int cols);

// F8_E4M3 decode (cuda_fp8.hpp semantics: 0x7F/0xFF -> NaN, e==0
// subnormal) — host twin of lsxcommon::kernels::F8E4m3ToFloat.
float FlashFp8E4m3Decode(uint8_t bits);

} // namespace lsxcommon::glm_moe_flash
