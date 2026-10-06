// lsx_kernels_quant.cuh — unified quantization device kernels (Phase A).
//
// Single textual source for the NVFP4/F8 elementwise codecs previously
// duplicated across shared_experts.cu / gemma_moe_inference.cu / kimi_moe_inference.cu
// / glm_moe_load.cu / benchmark_gpu.cu / safetensors_gpu.cu /
// deepseek_moe_inference_attn.cu.
//
// Byte-equality contracts preserved:
//   - dequant: canonical nvfp4_codec decode; OutT cast per former site
//     (__float2half for FP16 sites, __float2bfloat16_rn for BF16 sites).
//   - F8_E4M3: cuda_fp8.hpp semantics (0x7F/0xFF -> NaN, e==0 subnormal).
//   - E2M1 encode exists in TWO tie flavors — they were genuinely different
//     in the codebase and callers depend on their own flavor:
//       QuantizeE2m1TieUp    (qwen fast-path act quant, threshold chain)
//       QuantizeE2m1Nearest  (offline conversion, nearest loop, ties lower)
#pragma once

#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cuda_runtime.h>
#include "nvfp4_codec.h"
#include "lsx_kernels_common.cuh"

namespace lsxcommon {
namespace kernels {

// ── F8 decoders ─────────────────────────────────────────────────────────────

__host__ __device__ __forceinline__ float F8E4m3ToFloat(uint8_t b)
{
    if (!b || b == 0x80) return 0.0f;
    if ((b & 0x7F) == 0x7F) {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_v;
        memcpy(&nan_v, &nan_bits, sizeof(nan_v));
        return nan_v;
    }
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0x0F;
    const int m = b & 0x07;
    if (e == 0) {
        const float v = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
        return s ? -v : v;
    }
    const uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                       | (static_cast<uint32_t>(e + 127 - 7) << 23)
                       | (static_cast<uint32_t>(m) << 20);
    float v;
    memcpy(&v, &f32, sizeof(v));
    return v;
}

__host__ __device__ __forceinline__ float E8m0ToFloat(uint8_t b)
{
    if (!b) return 0.0f;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

// Clamped flavor used by the routed-expert FFN paths (shared_experts.cu):
// saturated scale 0xFF yields FLT_MAX (not +Inf) and 0 yields 0. Kept as a
// separate documented flavor — callers rely on finite clamping.
__host__ __device__ __forceinline__ float E8m0ClampedToFloat(uint8_t b)
{
    if (b == 0) return 0.0f;
    if (b == 0xFF) return FLT_MAX;
    const uint32_t bits = static_cast<uint32_t>(b) << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

// ── NVFP4 elementwise dequant ───────────────────────────────────────────────

// CastOut lives in lsx_kernels_common.cuh.

__device__ __forceinline__ float Nvfp4DecodeScale(uint8_t scale_e4m3,
                                                  float global_scale)
{
    return gpu::nvfp4::E4M3ScaleToFloat(scale_e4m3) * global_scale;
}

template <typename OutT>
__global__ void Nvfp4DequantKernel(const uint8_t* __restrict__ packed,
                                   const uint8_t* __restrict__ scales,
                                   float global_scale, OutT* __restrict__ out,
                                   int64_t n)
{
    const int64_t i =
        blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const uint8_t byte = packed[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    out[i] = detail::CastOut<OutT>(
        gpu::nvfp4::E2M1Value(nib) * Nvfp4DecodeScale(scales[i >> 4], global_scale));
}

// NOTE: a historical frexp-based pow2 encoder (clamped e4<=15, doubling the
// scale on exact powers of two) was removed -- use the canonical
// lsxcommon/nvfp4_scale.h FloatToE4m3 everywhere.

// ── E2M1 encode (two documented tie flavors) ────────────────────────────────

__host__ __device__ __forceinline__ uint8_t QuantizeE2m1TieUp(float v)
{
    // Nearest-level with ties resolved UPWARD (strict `<` thresholds):
    // exact midpoints map to the higher-magnitude level.
    const bool neg = v < 0.0f;
    const float a = fabsf(v);
    uint8_t mag;
    if (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4;
    else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6;
    else mag = 7;
    return neg ? static_cast<uint8_t>(mag | 8) : mag;
}

__host__ __device__ __forceinline__ uint8_t QuantizeE2m1Nearest(float val)
{
    // Nearest-level loop from safetensors_gpu.cu dev_quantize_nvfp4:
    // strict `d < best_diff` keeps the LOWER level on exact midpoint ties.
    if (val == 0.0f) return 0;
    const int sign = val < 0 ? 1 : 0;
    const float abs_val = fabsf(val);
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float best_diff = fabsf(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        const float d = fabsf(abs_val - levels[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

} // namespace kernels
} // namespace lsxcommon
