// tests/test_deepseek_moe_v41_dequant.cu — DeepSeek-V4.1-Flash load-time dense
// dequant: BF16->FP16 (RNE) and F8_E4M3 x UE8M0 per-32x32 -> FP16, checked
// against an independent host reference plus hand-pinned values.
#include <catch2/catch_test_macros.hpp>

#include "deepseek_moe_v41_dequant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

float HostE4m3(uint8_t b)
{
    const int sign = (b >> 7) & 1;
    const int exp = (b >> 3) & 0xF;
    const int mant = b & 0x7;
    float v;
    if (exp == 0)
        v = static_cast<float>(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    else
        v = (1.0f + static_cast<float>(mant) / 8.0f) *
            std::ldexp(1.0f, exp - 7);
    return sign ? -v : v;
}

float HostE8m0(uint8_t b)
{
    return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

uint16_t HalfBits(float f)
{
    __half h = __float2half_rn(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

} // namespace

TEST_CASE("deepseek-v41 bf16->fp16 matches (host, RNE)",
          "[deepseek-v41][dequant]")
{
    const int n = 1024;
    std::vector<uint16_t> src(n);
    for (int i = 0; i < n; ++i) src[i] = static_cast<uint16_t>((i * 257) & 0xFFFF);

    uint8_t *d_src = nullptr;
    uint16_t *d_dst = nullptr;
    REQUIRE(cudaMalloc(&d_src, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_dst, n * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_src, src.data(), n * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    REQUIRE(lsxcommon::deepseek_v41::LaunchBf16ToHalf(
                d_src, d_dst, n, nullptr) == cudaSuccess);

    std::vector<uint16_t> got(n);
    REQUIRE(cudaMemcpy(got.data(), d_dst, n * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (int i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        CHECK(got[i] == HalfBits(f));
    }
    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST_CASE("deepseek-v41 fp8 32x32 dequant matches host reference",
          "[deepseek-v41][dequant]")
{
    const int rows = 64, cols = 64;
    std::vector<uint8_t> w(rows * cols);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = static_cast<uint8_t>((i * 37 + 11) & 0x7F);  // positive E4M3
    const uint8_t s[4] = {127, 128, 126, 125};               // 2x2 scale tiles

    uint8_t *d_w = nullptr, *d_s = nullptr;
    uint16_t *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_w, w.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, rows * cols * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(d_s, s, 4, cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::deepseek_v41::LaunchFp8BlockDequantToHalf(
                d_w, d_s, d_out, rows, cols, nullptr) == cudaSuccess);

    std::vector<uint16_t> got(rows * cols);
    REQUIRE(cudaMemcpy(got.data(), d_out, rows * cols * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    const int scols = cols / 32;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t i = static_cast<size_t>(r) * cols + c;
            const float expected =
                HostE4m3(w[i]) * HostE8m0(s[(r / 32) * scols + (c / 32)]);
            CHECK(got[i] == HalfBits(expected));
        }
    }

    // Hand-pinned: 0x38 is E4M3 1.0; with scale 2^0 gives 1.0.
    CHECK(HostE4m3(0x38) == 1.0f);
    CHECK(HostE8m0(127) == 1.0f);

    cudaFree(d_w);
    cudaFree(d_s);
    cudaFree(d_out);
}
