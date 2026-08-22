// Тест Gemma-4 загрузчика: бит-точность GPU BF16→FP16 in-place конверсии.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lsxcommon/gemma4_inference_load.h"

TEST_CASE("GemmaBf16ToHalfInPlace bit-exact vs host", "[gemma][loader]") {
    // BF16 bit patterns: zeros, normals, subnormals, inf, nan, sign.
    const std::vector<uint16_t> bits = {
        0x0000, 0x8000, 0x3F80, 0xBF80, 0x4049, 0xC049,
        0x0080, 0x007F, 0x7F80, 0xFF80, 0x7FC0, 0x3C00, 0x42C8,
    };
    const int n = static_cast<int>(bits.size());

    uint16_t *d_buf = nullptr;
    REQUIRE(cudaMalloc(&d_buf, n * sizeof(uint16_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_buf, bits.data(), n * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    auto rc = lsxcommon::gemma::GemmaBf16ToHalfInPlace(
        reinterpret_cast<uint16_t *>(d_buf), n, nullptr);
    REQUIRE(rc == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);

    std::vector<uint16_t> got(n);
    REQUIRE(cudaMemcpy(got.data(), d_buf, n * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int i = 0; i < n; ++i) {
        // Host reference: nearest-even via float round-trip — the same math
        // GlmBf16ToHalfInPlace is held to.
        uint32_t wide = static_cast<uint32_t>(bits[i]) << 16;
        float f;
        std::memcpy(&f, &wide, 4);
        __half h = __float2half(f);
        uint16_t want;
        std::memcpy(&want, &h, 2);
        INFO("bf16 0x" << std::hex << bits[i] << " got 0x" << got[i]);
        CHECK(got[i] == want);
    }
    REQUIRE(cudaFree(d_buf) == cudaSuccess);
}

TEST_CASE("GemmaBf16ToHalfInPlace rejects null/misaligned input",
          "[gemma][loader]") {
    CHECK(lsxcommon::gemma::GemmaBf16ToHalfInPlace(nullptr, 8, nullptr) ==
          cudaErrorInvalidValue);

    uint16_t *p = nullptr;
    REQUIRE(cudaMalloc(&p, 64 * sizeof(uint16_t)) == cudaSuccess);
    // 2-byte-aligned but not 16-byte-aligned buffer must be rejected.
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    uintptr_t mis = (addr % 16 == 0) ? addr + 2 : addr;
    CHECK(lsxcommon::gemma::GemmaBf16ToHalfInPlace(
              reinterpret_cast<uint16_t *>(mis), 8, nullptr) ==
          cudaErrorInvalidValue);
    REQUIRE(cudaFree(p) == cudaSuccess);
}
