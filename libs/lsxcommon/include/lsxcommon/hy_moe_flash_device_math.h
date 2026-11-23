// hy_moe_flash_device_math.h — Hy4-flash device constants and inline helpers.
// Header must stay free of abseil and of host-only containers: it is included
// from .cu translation units.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include "nvfp4_codec.h"

namespace lsxcommon {
namespace hy4 {
namespace gpu {

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float Sigmoid(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// The reference SwiGLU of the routed experts clamps BEFORE the activation:
//   gate = gate.clamp(max=limit)          (no lower bound)
//   up   = up.clamp(-limit, limit)
//   out  = silu(gate) * up
__device__ __forceinline__ float SwigluClamp(float gate, float up, float limit)
{
    const float g = gate > limit ? limit : gate;
    const float u = up > limit ? limit : (up < -limit ? -limit : up);
    return Silu(g) * u;
}

// Dequantizes ONE NVFP4 element: packed byte `b`, element index within the
// 16-wide scale block `lane16`. Mirrors nvfp4_codec.h (nibble low = even k).
__device__ __forceinline__ float Nvfp4Element(uint8_t b, int lane16,
                                             uint8_t scale_byte, float gs)
{
    const int nib = (lane16 & 1) ? (b >> 4) : (b & 0xF);
    return ::lsxcommon::gpu::nvfp4::E2M1Value(nib) *
           ::lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(scale_byte) * gs;
}

} // namespace gpu
} // namespace hy4
} // namespace lsxcommon
