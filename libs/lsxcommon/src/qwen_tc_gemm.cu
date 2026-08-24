// qwen_tc_gemm.cu - Blackwell tensor-core NVFP4 W4A4 fast path (audit Phase 3,
// F-01). See qwen_tc_gemm.h for the compilation-target constraint.
//
// Task 2: activation quantization. One thread per 16-element block computes
// the block absmax, encodes a power-of-two E4M3 scale (same math as
// lsxcommon::float_to_e4m3_p2 / nvfp4_codec.h E4M3ScaleToFloat: value
// = 2^(e4-7), e4==0 -> 0), and packs 16 E2M1 nibbles (low nibble = even k).
#include "lsxcommon/qwen_tc_gemm.h"

#include <cmath>
#include <cstdint>

#include "nvfp4_codec.h"

namespace lsxcommon {
namespace qwen {
namespace tc {

namespace {

// Power-of-two E4M3 encode matching schema.cpp float_to_e4m3_p2 and the
// device decoder in nvfp4_codec.h: byte = (e4+7)<<3 with value 2^(e4-7).
// Ceil semantics: smallest power of two >= s.
__device__ __forceinline__ uint8_t Pow2E4m3Encode(float s)
{
    if (!(s > 0.0f)) return 0; // zero / NaN guard: zero scale zeroes the block
    int exp;
    (void)frexpf(s, &exp); // s = m * 2^exp, m in [0.5, 1)
    // ceil-log2(s): log2(s) = exp + log2(m) with log2(m) in [-1, 0),
    // so ceil == exp except when m == 0.5 exactly (log2 == exp-1 -> an
    // exact power of two, already minimal).
    const int e4 = exp + 7;
    if (e4 < 1) return static_cast<uint8_t>(1 << 3);      // smallest positive
    if (e4 > 15) return static_cast<uint8_t>(15 << 3);    // saturate at 2^8
    return static_cast<uint8_t>(e4 << 3);
}

__device__ __forceinline__ uint8_t QuantizeE2m1(float v)
{
    // Nearest-level E2M1 with sign bit 3; levels match kE2M1Table.
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

__global__ void ActQuantKernel(const __half *__restrict__ x, int64_t n,
                               uint8_t *__restrict__ packed,
                               uint8_t *__restrict__ scales)
{
    const int64_t blk = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const int64_t base = blk * 16;
    if (base >= n) return;
    const int len = static_cast<int>(min(static_cast<long long>(16),
                                         n - base));

    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float v = fabsf(__half2float(x[base + i]));
        absmax = fmaxf(absmax, v);
    }

    // scale = 2^ceil(log2(absmax/6)); quantized magnitude in [~0.75, 6].
    const uint8_t sb = Pow2E4m3Encode(absmax * (1.0f / 6.0f));
    scales[blk] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;

    for (int i = 0; i < len; i += 2) {
        const uint8_t lo = QuantizeE2m1(__half2float(x[base + i]) * inv);
        const uint8_t hi = (i + 1 < len)
                               ? QuantizeE2m1(__half2float(x[base + i + 1]) *
                                              inv)
                               : 0;
        packed[(base + i) / 2] = static_cast<uint8_t>((hi << 4) | lo);
    }
}

} // namespace

cudaError_t QuantizeActsNvfp4(cudaStream_t stream, const __half *x, int seq,
                              int K, uint8_t *d_packed, uint8_t *d_scales)
{
    if (!x || !d_packed || !d_scales || seq <= 0 || K <= 0 || (K & 15))
        return cudaErrorInvalidValue;
    const int64_t n = static_cast<int64_t>(seq) * K;
    const int64_t blocks16 = (n + 15) / 16;
    const int threads = 256;
    const int64_t grid = (blocks16 + threads - 1) / threads;
    if (grid > 0x7FFFFFFFLL) return cudaErrorInvalidValue;
    ActQuantKernel<<<static_cast<int>(grid), threads, 0, stream>>>(
        x, n, d_packed, d_scales);
    return cudaGetLastError();
}

} // namespace tc
} // namespace qwen
} // namespace lsxcommon
