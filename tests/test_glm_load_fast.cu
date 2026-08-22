#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <lsxcommon/glm_inference_load.h>

namespace {

uint16_t RefBf16ToHalf(uint16_t bits)
{
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    __half h = __float2half(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

std::vector<uint16_t> TestPatterns()
{
    return {
        0x0000,
        0x8000,
        0x0001,
        0x03FF,
        0x0400,
        0x3555,
        0x3556,
        0x4248,
        0xC048,
        0x7F80,
        0xFF80,
        0x7FC0,
    };
}

std::vector<uint16_t> RandomFinite(size_t n, uint32_t seed)
{
    std::vector<uint16_t> v(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        uint16_t bits = static_cast<uint16_t>(s >> 13);
        if ((bits & 0x7F80u) == 0x7F80u) bits &= 0x7F7Fu;
        v[i] = bits;
    }
    return v;
}

std::vector<uint16_t> RunOnDevice(const std::vector<uint16_t> &host_bits,
                                  cudaStream_t stream)
{
    const int64_t n = static_cast<int64_t>(host_bits.size());
    uint16_t *d = nullptr;
    REQUIRE(cudaMalloc(&d, host_bits.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d, host_bits.data(), host_bits.size() * 2,
                            cudaMemcpyHostToDevice, stream) == cudaSuccess);
    cudaError_t e = lsxcommon::glm::GlmBf16ToHalfInPlace(d, n, stream);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<uint16_t> out(host_bits.size());
    REQUIRE(cudaMemcpy(out.data(), d, host_bits.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d);
    return out;
}

void CheckBitExact(const std::vector<uint16_t> &host_bits)
{
    auto ref = host_bits;
    for (auto &b : ref) b = RefBf16ToHalf(b);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto got = RunOnDevice(host_bits, stream);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < ref.size(); ++i)
        INFO("index " << i << " bf16=0x" << std::hex << host_bits[i]);
    for (size_t i = 0; i < ref.size(); ++i)
        REQUIRE(got[i] == ref[i]);
}

} // namespace

TEST_CASE("Bf16ToHalfInPlace bit-exact vs host reference", "[glm][loadfast]")
{
    CheckBitExact(TestPatterns());
    CheckBitExact(RandomFinite(100003, 1));
}

TEST_CASE("Bf16ToHalfInPlace ragged tails stay bit-exact", "[glm][loadfast]")
{
    for (int64_t n : {1, 7, 9, 4097}) {
        CheckBitExact(RandomFinite(static_cast<size_t>(n), 42));
    }
}
