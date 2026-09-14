// deepseek_moe_v41_dequant.h — DeepSeek-V4.1-Flash load-time dense dequant.
// Dense FP8 weights (F8_E4M3) carry per-32x32 UE8M0 block scales; the loader
// materializes them as FP16 device caches (correctness-first decision, no FP8
// GEMM path in milestone 1). FP16 is passed as uint16_t so host TUs need no
// cuda_fp16.h.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace deepseek_v41 {

// src: n BF16 elements (2 bytes each) -> dst: n __half (bit pattern via RNE).
cudaError_t LaunchBf16ToHalf(const uint8_t *src, uint16_t *dst, int64_t n,
                             cudaStream_t stream);

// Dequantizes a [rows, cols] F8_E4M3 matrix with [ceil(rows/32), ceil(cols/32)]
// UE8M0 block scales into FP16: out[i,j] = e4m3(w[i,j]) * 2^(s - 127).
cudaError_t LaunchFp8BlockDequantToHalf(const uint8_t *w,
                                        const uint8_t *scales, uint16_t *out,
                                        int rows, int cols,
                                        cudaStream_t stream);

// In-place FP4 activation quantize-dequant (E2M1, per `block`-element block) on
// FP16 x [rows, n]. e8m0_scale=true -> pow2 scale (indexer); false -> E4M3
// (compressed KV). Mirrors kernel.py:fp4_act_quant.
cudaError_t LaunchFp4ActQuantInplace(uint16_t *x, int rows, int n, int block,
                                     bool e8m0_scale, cudaStream_t stream);

// In-place FP8 (E4M3) activation quantize-dequant, per `block`-element block,
// pow2 scale (window KV; kernel.py:act_quant with scale_fmt="ue8m0").
cudaError_t LaunchFp8ActQuantInplace(uint16_t *x, int rows, int n, int block,
                                     cudaStream_t stream);

} // namespace deepseek_v41
} // namespace lsxcommon
