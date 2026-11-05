// nvfp4_codec.h — shared device-only NVFP4 decode primitives.
// Reachable by all CUDA object libraries (lsxcommon_cu / lsxcommon_glm_cu /
// lsxcommon_qwen_cu / lsxcommon_gemma_cu).
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
// Bit-exactness: E4M3ScaleToFloat decodes the complete unsigned E4M3 block
// scale via nvfp4_scale.h::E4m3ToFloat — normals 2^(e-7)*(1+m/8) AND the
// e4==0 subnormals m*2^-9, matching cuda_fp8/torch.float8_e4m3fn over the
// valid scale domain 0x00..0x7F (bit 7 is the reserved sign bit of the
// unsigned hardware scale type and is masked).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nvfp4_scale.h"

namespace lsxcommon {
namespace gpu {
namespace nvfp4 {

// Defined below; E2M1PairValue's non-family fallback uses it.
__host__ __device__ __forceinline__ float E2M1Value(int nib);

// E2M1 PAIR decode: both nibbles of one packed byte in a single instruction.
// `cvt.rn.bf16x2.e2m1x2` (SASS `F2FP.BF16.E2M1.UNPACK_B`) exists only on
// family-specific targets, so the branch is compiled against
// __CUDA_ARCH_FAMILY_SPECIFIC__ (sm_120f/sm_121f, CUDA 13.5). The nibble->float
// table above stays the reference AND the fallback: every E2M1 level is exactly
// representable in BF16, so the two are bit-identical, never "close" — pinned
// exhaustively over all 256 bytes by tests/test_lsx_kernels.cu
// "E2M1PairValue: both nibbles of every byte match the table".
// Returned in (low nibble, high nibble) order, matching E2M1Value(b & 0xF) /
// E2M1Value(b >> 4) as the GEMV consumes them.
__host__ __device__ __forceinline__ float2 E2M1PairValue(uint8_t packed)
{
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200
    // The instruction consumes a b8 holding one e2m1x2 operand; the low nibble
    // of the byte is element 0 (the packing order pinned by the exhaustive
    // table case below), and it produces bf16x2 with element 0 in the low half.
    unsigned out = 0;
    const unsigned short src = static_cast<unsigned short>(packed);
    asm("{ .reg .b8 v, z;                        \n"
        "  mov.b16 {v, z}, %1;                    \n"
        "  cvt.rn.bf16x2.e2m1x2 %0, v; }          \n"
        : "=r"(out)
        : "h"(src));
    const unsigned short lo = static_cast<unsigned short>(out & 0xFFFFu);
    const unsigned short hi = static_cast<unsigned short>(out >> 16);
    return make_float2(__bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&lo)),
                       __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&hi)));
#else
    return make_float2(E2M1Value(packed & 0xF), E2M1Value(packed >> 4));
#endif
}

// E2M1 decode (indexed by 4-bit nibble, sign bit 3):
//   magnitude 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign bit negates.
// Branchless on purpose (do NOT reintroduce a switch/table): the previous
// `switch (mag)` compiled to a data-dependent indirect branch whose cost
// scaled with the weight nibble distribution. Measured on Granite-4.2-30B
// decode (RTX 5090, 2026-09-22): real NVFP4 weights 655 GB/s vs 933 for a
// synthetic uniform buffer on the SAME kernel -- a same-codec, data-dependent
// 30% loss that in-situ made the whole decode GEMV latency-bound. This form
// lifts both to ~1595 GB/s (near the ~1618 GB/s raw read ceiling) and is
// bit-identical: the 8 magnitudes are stored as 2*value nibble codes
// {0,1,2,3,4,6,8,12} in 0xC8643210, and code*0.5f is exact in fp32.
// Guarded by tests/test_lsx_kernels.cu "E2M1Value exhaustive table and
// lo-nibble packing order" (all 16 nibbles + the -0.0f sign bit).
__host__ __device__ __forceinline__ float E2M1Value(int nib)
{
    const unsigned code = (0xC8643210u >> ((nib & 7u) * 4u)) & 0xFu;
    const float v = static_cast<float>(code) * 0.5f;
    return (nib & 8) ? -v : v;
}

// Full unsigned-E4M3 block scale (mantissa-aware). Delegates to the single
// canonical codec in nvfp4_scale.h: the COMPLETE E4M3FN value table — e4 >= 1
// gives 2^(e-7)*(1+m/8), e4 == 0 gives the SUBNORMAL m*2^-9 (0x00 -> 0), and
// 0x7F stays the reserved NaN. Never re-implement this table locally; a
// hand-copied host reference is how a family silently diverges from the
// runtime. Hardware parity (sm_120a mma.sync ... block_scale .ue4m3 decodes
// 0x01..0x07 as m*2^-9, not 0) is pinned by
// tests/test_blackwell_tc_mma.cu "sm_120a nvfp4 mma decodes subnormal UE4M3
// scale bytes as m*2^-9".
__host__ __device__ __forceinline__ float E4M3ScaleToFloat(uint8_t e4m3)
{
    return ::lsxcommon::nvfp4::E4m3ToFloat(e4m3);
}

} // namespace nvfp4
} // namespace gpu
} // namespace lsxcommon
