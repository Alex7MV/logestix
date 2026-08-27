// Shared device math for the weight-streaming NVFP4 GEMV chains (qwen_moe
// batched chain + gemma routed chain, rounds 19b/21). One source of truth
// for the byte/nibble/scale-block contract so both suites exercise the same
// code: byte j of a uint4 chunk holds elements 2j (low nibble) / 2j+1 (high
// nibble); elements 0..15 of each 32-element chunk decode with scale block
// sb, elements 16..31 with sb+1.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "nvfp4_codec.h"

namespace lsxcommon {
namespace nvfp4gemv {

// Dot product of one packed NVFP4 row against an FP16 activation row.
// `kb` is the K-slice base element (multiple of 32), `chunks` counts
// 32-element units owned by this lane stratum; caller reduces lanes via
// shfl afterwards. Deterministic: fixed loop partitioning, FP32 accumulate.
__device__ __forceinline__ float GemvDotRow(
    const __half *__restrict__ x, const uint8_t *__restrict__ wrow,
    const uint8_t *__restrict__ srow, int kb, int chunks, int lane)
{
    float acc = 0.0f;
    const uint4 *wv = reinterpret_cast<const uint4 *>(wrow + (kb >> 1));
    const bool x_vec =
        (reinterpret_cast<uintptr_t>(x) & 3u) == 0;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 pw = wv[c];
        const uint8_t *pb = reinterpret_cast<const uint8_t *>(&pw);
        const int sb = (kb >> 4) + 2 * c;
        const float sc0 = gpu::nvfp4::E4M3ScaleToFloat(srow[sb]);
        const float sc1 = gpu::nvfp4::E4M3ScaleToFloat(srow[sb + 1]);        const int xb = kb + c * 32;
        const __half2 *xh =
            x_vec ? reinterpret_cast<const __half2 *>(x + xb) : nullptr;
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float scl = (b < 8) ? sc0 : sc1;
            float xv0, xv1;
            if (x_vec) {
                const __half2 pair = xh[b];
                xv0 = __half2float(pair.x);
                xv1 = __half2float(pair.y);
            } else {
                xv0 = __half2float(x[xb + 2 * b]);
                xv1 = __half2float(x[xb + 2 * b + 1]);
            }
            acc += xv0 * gpu::nvfp4::E2M1Value(pb[b] & 0xF) * scl;
            acc += xv1 * gpu::nvfp4::E2M1Value(pb[b] >> 4) * scl;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;   // tree totals are meaningful on every lane
}

} // namespace nvfp4gemv
} // namespace lsxcommon
