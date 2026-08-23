#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kimi_inference_kernels.h"

namespace {

uint16_t FloatToBf16(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000U) == 0x7f800000U)
        return static_cast<uint16_t>(u >> 16);
    uint32_t rounding_bias = 0x7fffU + ((u >> 16) & 1U);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

} // namespace

using lsxcommon::kimi::KimiInt4DequantToBf16;

TEST_CASE("Kimi INT4 dequant matches CPU reference", "[kimi][int4]")
{
    const int rows = 4;
    const int cols = 256;
    const int groups = cols / 32;
    const int words_per_row = cols / 8;

    std::vector<int> packed(static_cast<size_t>(rows) * words_per_row, 0);
    std::vector<uint16_t> scales(static_cast<size_t>(rows) * groups);
    std::vector<uint16_t> ref(static_cast<size_t>(rows) * cols);

    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) {
            float s = 0.25f + 0.5f * static_cast<float>((r * 31 + g) % 7);
            scales[static_cast<size_t>(r) * groups + g] = FloatToBf16(s);
        }
        for (int c = 0; c < cols; ++c) {
            int nib = (r * 131 + c * 17) % 15;
            if (nib > 7) nib -= 16;
            float s;
            {
                uint32_t bits = static_cast<uint32_t>(
                    scales[static_cast<size_t>(r) * groups + c / 32]) << 16;
                std::memcpy(&s, &bits, sizeof(s));
            }
            ref[static_cast<size_t>(r) * cols + c] =
                FloatToBf16(static_cast<float>(nib) * s);
            packed[static_cast<size_t>(r) * words_per_row + c / 8] |=
                ((nib & 0xF) << (4 * (c % 8)));
        }
    }

    int *d_packed = nullptr;
    uint16_t *d_scales = nullptr;
    uint16_t *d_out = nullptr;
    size_t packed_bytes = packed.size() * sizeof(int);
    size_t scale_bytes = scales.size() * sizeof(uint16_t);
    size_t out_bytes = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
    REQUIRE(cudaMalloc(&d_packed, packed_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scales, scale_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, out_bytes) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_packed, packed.data(), packed_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_scales, scales.data(), scale_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(KimiInt4DequantToBf16(d_packed, d_scales, rows, cols, d_out,
                                  stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<uint16_t> out(rows * cols);
    REQUIRE(cudaMemcpy(out.data(), d_out, out_bytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < out.size(); ++i)
        REQUIRE(out[i] == ref[i]);

    cudaStreamDestroy(stream);
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_out);
}

TEST_CASE("Kimi INT4 dequant sign extends negative nibbles", "[kimi][int4]")
{
    const int rows = 1;
    const int cols = 64;
    const int words_per_row = cols / 8;

    std::vector<int> packed(words_per_row, 0x99999999); // nibble 9 -> -7
    std::vector<uint16_t> scales(cols / 32, FloatToBf16(0.5f));
    uint16_t expected = FloatToBf16(-3.5f);

    int *d_packed = nullptr;
    uint16_t *d_scales = nullptr;
    uint16_t *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_packed, packed.size() * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scales, scales.size() * sizeof(uint16_t)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d_out,
                       static_cast<size_t>(rows) * cols * sizeof(uint16_t)) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(int),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_scales, scales.data(),
                       scales.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(KimiInt4DequantToBf16(d_packed, d_scales, rows, cols, d_out,
                                  stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<uint16_t> out(static_cast<size_t>(rows) * cols);
    REQUIRE(cudaMemcpy(out.data(), d_out,
                       out.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (uint16_t v : out)
        REQUIRE(v == expected);

    cudaStreamDestroy(stream);
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_out);
}
