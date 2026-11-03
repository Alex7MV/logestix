// Host-reference FP8 -> NVFP4 requant implementation. Mirrors
// quantize_tensor_gpu (safetensors_gpu.cu) step by step so parity tests
// pin the converter against the GPU path.
#include "glm_moe_flash_fp8.h"

#include <cmath>
#include <cstring>

namespace lsxcommon::glm_moe_flash {

namespace {

// Host twin of kernels::QuantizeE2m1Nearest (lsx_kernels_quant.cuh):
// nearest level, strict '<' keeps the LOWER level on midpoint ties.
uint8_t QuantizeE2m1NearestHost(float val)
{
    if (val == 0.0f) return 0;
    const int sign = val < 0 ? 1 : 0;
    const float abs_val = std::fabs(val);
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float best_diff = std::fabs(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        const float d = std::fabs(abs_val - levels[i]);
        if (d < best_diff) {
            best_diff = d;
            best = i;
        }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

// Host twin of kernels::F8E4m3ToFloat (cuda_fp8.hpp semantics).
float F8E4m3ToFloatHost(uint8_t b)
{
    if (b == 0 || b == 0x80) return 0.0f;
    if ((b & 0x7F) == 0x7F) return std::nanf("");
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0x0F;
    const int m = b & 0x07;
    if (e == 0) {
        const float v = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
        return s ? -v : v;
    }
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

} // namespace

float FlashFp8E4m3Decode(uint8_t bits) { return F8E4m3ToFloatHost(bits); }

FlashNvfp4Blob RequantFp8ToNvfp4Host(const void *fp8_data,
                                     const float *scale_inv, int rows,
                                     int cols)
{
    FlashNvfp4Blob out;
    out.rows = rows;
    out.cols = cols;

    const int64_t numel = static_cast<int64_t>(rows) * cols;
    const int nblocks16 =
        static_cast<int>((numel + 15) / 16);  // per-16 blocks, row-major
    const int scale_cols = (cols + 127) / 128;

    // GPU path (convert_kernel + block_maxabs): dequant to F32, then
    // per-16-element block max over the flattened row-major array.
    // Note: the GPU block_maxabs is over the FLAT 16-block, which crosses
    // row boundaries when cols % 16 != 0 — the real checkpoint matrices
    // all have cols % 128 == 0, so flat == per-row there. This host twin
    // matches the GPU flat semantics exactly.
    std::vector<float> block_max(nblocks16, 0.0f);
    float global_max_abs = 0.0f;
    const uint8_t *src = static_cast<const uint8_t *>(fp8_data);
    for (int64_t i = 0; i < numel; ++i) {
        const int r = static_cast<int>(i / cols);
        const int c = static_cast<int>(i % cols);
        const float v = F8E4m3ToFloatHost(src[i])
                      * scale_inv[static_cast<int64_t>(r / 128) * scale_cols
                                  + c / 128];
        const int blk = static_cast<int>(i / 16);
        const float av = std::fabs(v);
        if (av > block_max[blk]) block_max[blk] = av;
        if (av > global_max_abs) global_max_abs = av;
    }

    // GPU convention: global max maps onto the E2M1 top level 6.0.
    if (!std::isfinite(global_max_abs) || global_max_abs == 0.0f)
        global_max_abs = 1.0f / 128.0f;
    out.global_scale = global_max_abs / 6.0f;

    out.packed.assign(static_cast<size_t>((numel + 1) / 2), 0);
    out.scales.assign(static_cast<size_t>(nblocks16), 0);

    // Per-block scale byte: s = block_max / global_max (1/128 fallback for
    // empty blocks — GPU convention), encoded as pow2-E4M3 (ceil).
    for (int blk = 0; blk < nblocks16; ++blk) {
        const float max_val = block_max[blk];
        const float s =
            (max_val == 0.0f) ? (1.0f / 128.0f) : (max_val / global_max_abs);
        out.scales[blk] = lsxcommon::nvfp4::FloatToE4m3(s);
    }

    // Quantize + pack: q = E2M1(x / (e4m3_p2(byte) * global_scale)).
    for (int64_t i = 0; i < numel; ++i) {
        const int r = static_cast<int>(i / cols);
        const int c = static_cast<int>(i % cols);
        const float v = F8E4m3ToFloatHost(src[i])
                      * scale_inv[static_cast<int64_t>(r / 128) * scale_cols
                                  + c / 128];
        const int blk = static_cast<int>(i / 16);
        const float s =
            lsxcommon::nvfp4::E4m3ToFloat(out.scales[blk])
            * out.global_scale;
        const uint8_t nib = QuantizeE2m1NearestHost(v / s);
        if (i % 2 == 0) out.packed[i / 2] = nib;
        else out.packed[i / 2] |= static_cast<uint8_t>(nib << 4);
    }
    return out;
}

} // namespace lsxcommon::glm_moe_flash
