// Per-stream grow-only scratch (audit X11): the split-KV attention and router
// paths used to cudaMallocAsync/cudaFreeAsync on every layer/token. This helper
// keeps one device buffer per (stream, tag), grows it on demand, and never
// hands the same buffer to two streams.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/stream_scratch.h"

namespace {

struct TestTagA;
struct TestTagB;

bool GpuAvailable()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

} // namespace

TEST_CASE("stream scratch is stable per stream and grows in place",
          "[stream-scratch][fast]")
{
    if (!GpuAvailable()) std::exit(4);
    using lsxcommon::cuda::StreamScratch;
    StreamScratch<TestTagA>::Reset();

    cudaStream_t s0 = nullptr, s1 = nullptr;
    REQUIRE(cudaStreamCreate(&s0) == cudaSuccess);
    REQUIRE(cudaStreamCreate(&s1) == cudaSuccess);

    void *a1 = nullptr, *a2 = nullptr, *b1 = nullptr;
    REQUIRE(StreamScratch<TestTagA>::Get(s0, 256, &a1) == cudaSuccess);
    REQUIRE(StreamScratch<TestTagA>::Get(s0, 64, &a2) == cudaSuccess);
    CHECK(a1 == a2);                    // reuse while it fits
    REQUIRE(StreamScratch<TestTagA>::Get(s1, 64, &b1) == cudaSuccess);
    CHECK(b1 != a1);                    // streams never share a region

    // Grow on the same stream: a fresh pointer, still valid for writes.
    void *a3 = nullptr;
    REQUIRE(StreamScratch<TestTagA>::Get(s0, 1024, &a3) == cudaSuccess);
    CHECK(a3 != a1);
    const uint32_t pattern = 0xA5A5A5A5u;
    REQUIRE(cudaMemcpyAsync(a3, &pattern, sizeof(pattern),
                            cudaMemcpyHostToDevice, s0) == cudaSuccess);
    uint32_t back = 0;
    REQUIRE(cudaMemcpyAsync(&back, a3, sizeof(back),
                            cudaMemcpyDeviceToHost, s0) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s0) == cudaSuccess);
    CHECK(back == pattern);

    // A different tag on the same stream is a different region.
    void *other = nullptr;
    REQUIRE(StreamScratch<TestTagB>::Get(s0, 64, &other) == cudaSuccess);
    CHECK(other != a3);

    StreamScratch<TestTagA>::Reset();
    StreamScratch<TestTagB>::Reset();
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
}

TEST_CASE("stream scratch accepts a zero-byte request",
          "[stream-scratch][fast]")
{
    if (!GpuAvailable()) std::exit(4);
    using lsxcommon::cuda::StreamScratch;
    StreamScratch<TestTagA>::Reset();
    void *p = nullptr;
    cudaStream_t s = nullptr;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    CHECK(StreamScratch<TestTagA>::Get(s, 0, &p) == cudaSuccess);
    CHECK(p != nullptr);
    StreamScratch<TestTagA>::Reset();
    cudaStreamDestroy(s);
}
