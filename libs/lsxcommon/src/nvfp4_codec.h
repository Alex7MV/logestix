// nvfp4_codec.h — shared __host__ __device__ NVFP4 decode primitives
// (E2M1 value LUT + pow2-E4M3 scale decode). Reachable by all CUDA object
// libraries (lsxcommon_cu / lsxcommon_glm_cu / lsxcommon_qwen_cu /
// lsxcommon_gemma_cu) and usable from host code in CUDA TUs; the mandatory
// <cuda_runtime.h> makes it unfit for non-CUDA TUs — those use the
// CUDA-free codecs in nvfp4_scale.h directly; the scale decode here simply
// delegates to it, so both entry points share one textual source.
//
// ODR / constant-bank note: the E2M1 decode LUT is intentionally an inline
// __device__ FUNCTION with immediate values instead of a __constant__ array
// defined in this header. A header-defined __constant__ array is compiled
// into EVERY including TU; when several object libraries are linked into one
// binary the device linker merges same-name symbols and a TU can end up
// reading another (stale/uninitialized) copy — observed on RTX 5090 /
// CUDA 13.3 as garbage decode values in the Gemma TU while other models
// stayed correct. Immediates have no global storage, so no cross-TU aliasing
// is possible.
//
// Bit-exactness: E4M3ScaleToFloat uses `e4==0 -> 0.0`, else (e4+120)<<23.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "nvfp4_scale.h"

namespace lsxcommon {
namespace gpu {
namespace nvfp4 {

// E2M1 decode (indexed by 4-bit nibble, sign bit 3):
//   magnitude 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign bit negates.
__host__ __device__ __forceinline__ float E2M1Value(int nib)
{
    const int mag = nib & 7;
    float v;
    switch (mag) {
    case 1: v = 0.5f; break;
    case 2: v = 1.0f; break;
    case 3: v = 1.5f; break;
    case 4: v = 2.0f; break;
    case 5: v = 3.0f; break;
    case 6: v = 4.0f; break;
    case 7: v = 6.0f; break;
    default: v = 0.0f; break;
    }
    return (nib & 8) ? -v : v;
}

// E4M3 power-of-2 block scale (only exponent bits used). e4==0 -> 0.0;
// else 2^(e4-7). Delegates to the canonical CUDA-free decode in
// nvfp4_scale.h — single textual source, no sync-by-comment contract.
__host__ __device__ __forceinline__ float E4M3ScaleToFloat(uint8_t e4m3)
{
    return nvfp4::E4m3P2ToFloat(e4m3);
}

} // namespace nvfp4
} // namespace gpu
} // namespace lsxcommon
