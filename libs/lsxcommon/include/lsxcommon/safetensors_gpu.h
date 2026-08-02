#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon {
namespace gpu {

// Quantizes `raw_data` (dtype) to NVFP4 E2M1 nibbles + per-16 E4M3 scales +
// a global float scale. If the source weights are already quantized
// (F8_E4M3 / I8), the paired F8_E8M0 per-block scale tensor is passed via
// `e8m0_scales`/`e8m0_scale_shape`/`tensor_shape` and applied during the
// float conversion so the true magnitude is preserved (dequant value is
//   raw_value * e8m0_block_scale).
// Pass e8m0_scales == nullptr when the source is unquantized (BF16/F32/…).
bool quantize_tensor_gpu(const uint8_t *raw_data, int64_t numel,
                         const std::string &dtype,
                         const uint8_t *e8m0_scales,
                         const std::vector<int64_t> &tensor_shape,
                         const std::vector<int64_t> &e8m0_scale_shape,
                         std::vector<uint8_t> &packed,
                         std::vector<uint8_t> &scales_e4m3,
                         float &global_scale);

}
} // namespace lsxcommon
