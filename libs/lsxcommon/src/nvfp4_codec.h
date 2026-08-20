// nvfp4_codec.h — shared device-only NVFP4 decode primitives.
// Reachable by all three CUDA object libraries (lsxcommon_cu /
// lsxcommon_glm_cu / lsxcommon_qwen_cu). Device symbols only
// (__constant__ / __device__): no host linkage, so each NVCC TU gets its own
// const-bank copy — no ODR risk across the object-lib split.
//
// Bit-exactness: E4M3ScaleToFloat uses `e4==0 -> 0.0`, else (e4+120)<<23.
// All three prior sites (shared_experts, GLM, Qwen, benchmark) produced the
// same result for valid power-of-two block scales; see plan note #1.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace gpu {
namespace nvfp4 {

// E2M1 decode table (indexed by 4-bit nibble, sign bit 3).
__constant__ float kE2M1Table[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 power-of-2 block scale (only exponent bits used). e4==0 -> 0.0;
// else 2^(e4-7) reconstructed bit-wise as (e4+120)<<23.
__device__ __forceinline__ float E4M3ScaleToFloat(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    return __uint_as_float((e4 + 120u) << 23);
}

} // namespace nvfp4
} // namespace gpu
} // namespace lsxcommon