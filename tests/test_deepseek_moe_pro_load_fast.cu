#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <lsxcommon/fast_load.h>

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
        0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3555,
        0x3556, 0x4248, 0xC048, 0x7F80, 0xFF80, 0x7FC0,
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
    cudaError_t e = lsxcommon::fastload::Bf16ToHalfInPlace(d, n, stream);
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

TEST_CASE("fastload Bf16ToHalfInPlace bit-exact vs host reference",
          "[deepseek][pro][loadfast]")
{
    CheckBitExact(TestPatterns());
    CheckBitExact(RandomFinite(100003, 1));
}

TEST_CASE("fastload Bf16ToHalfInPlace ragged tails stay bit-exact",
          "[deepseek][pro][loadfast]")
{
    for (int64_t n : {1, 7, 9, 4097}) {
        CheckBitExact(RandomFinite(static_cast<size_t>(n), 42));
    }
}

#include "fast_load_internal.h"

using lsxcommon::fastload::PinnedStagingRing;

namespace {

std::vector<uint16_t> RefConvertAll(const std::vector<uint16_t> &bits)
{
    std::vector<uint16_t> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) out[i] = RefBf16ToHalf(bits[i]);
    return out;
}

bool DeviceBitsEqual(cudaStream_t stream, const __half *d,
                     const std::vector<uint16_t> &expect)
{
    std::vector<uint16_t> got(expect.size());
    cudaMemcpyAsync(got.data(), d, expect.size() * 2,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return got == expect;
}

} // namespace

TEST_CASE("Staging ring chunked BF16 upload is bit-exact",
          "[deepseek][pro][loadfast]")
{
    constexpr int64_t kNumel = 20001;
    constexpr size_t kChunk = 4096;
    PinnedStagingRing ring(kChunk);
    REQUIRE(ring.ok());

    auto bits = RandomFinite(static_cast<size_t>(kNumel), 7);

    __half *d = nullptr;
    REQUIRE(cudaMalloc(&d, kNumel * 2) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    std::uint64_t acc = 0;
    REQUIRE(ring.UploadBf16AsFp16(reinterpret_cast<const uint8_t *>(bits.data()),
                                  kNumel, d, stream, acc) == cudaSuccess);
    CHECK(acc == static_cast<std::uint64_t>(kNumel) * 2);
    REQUIRE(DeviceBitsEqual(stream, d, RefConvertAll(bits)));

    cudaStreamDestroy(stream);
    cudaFree(d);
}

TEST_CASE("Staging ring buffer reuse never corrupts in-flight chunks",
          "[deepseek][pro][loadfast]")
{
    constexpr int64_t kNumel = 8192 + 3;
    constexpr size_t kChunk = 1024;
    PinnedStagingRing ring(kChunk);
    REQUIRE(ring.ok());

    std::vector<std::vector<uint16_t>> tensors;
    std::vector<std::vector<uint16_t>> refs;
    std::vector<__half *> devs;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    for (int t = 0; t < 4; ++t) {
        tensors.push_back(
            RandomFinite(static_cast<size_t>(kNumel), 100 + t));
        refs.push_back(RefConvertAll(tensors.back()));
        __half *d = nullptr;
        REQUIRE(cudaMalloc(&d, kNumel * 2) == cudaSuccess);
        devs.push_back(d);
    }

    std::uint64_t acc = 0;
    for (int t = 0; t < 4; ++t) {
        REQUIRE(ring.UploadBf16AsFp16(
                    reinterpret_cast<const uint8_t *>(tensors[t].data()),
                    kNumel, devs[t], stream, acc) == cudaSuccess);
    }
    for (int t = 0; t < 4; ++t)
        REQUIRE(DeviceBitsEqual(stream, devs[t], refs[t]));

    cudaStreamDestroy(stream);
    for (__half *d : devs) cudaFree(d);
}

TEST_CASE("Staging ring raw upload matches source", "[deepseek][pro][loadfast]")
{
    constexpr size_t kBytes = 5000;
    PinnedStagingRing ring(2048);
    REQUIRE(ring.ok());

    std::vector<uint8_t> src(kBytes);
    for (size_t i = 0; i < kBytes; ++i) src[i] = static_cast<uint8_t>(i * 31);

    uint8_t *d = nullptr;
    REQUIRE(cudaMalloc(&d, kBytes) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    std::uint64_t acc = 0;
    REQUIRE(ring.UploadRaw(src.data(), d, kBytes, stream, acc) == cudaSuccess);
    CHECK(acc == kBytes);
    std::vector<uint16_t> expect(kBytes / 2);
    std::memcpy(expect.data(), src.data(), expect.size() * 2);
    REQUIRE(DeviceBitsEqual(stream,
                            reinterpret_cast<const __half *>(d), expect));

    cudaStreamDestroy(stream);
    cudaFree(d);
}
