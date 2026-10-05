// Streaming-read ceiling: how fast can this card read a buffer at all?
//
// This exists to answer ONE question honestly, and it is deliberately the cheapest
// experiment in the repo: before rewriting attention to stage K/V tiles in shared
// memory, is there anything to win? On Xing-4.0 the decode attention sustains
// 744-817 GB/s reading its KV cache, while the dense decode GEMVs on the same card
// in the same harness reach 1202 GB/s (xing.gemv.dense_decode_gbps). That 1.5x gap
// is the entire headroom a tiling rewrite is chasing - and it is only real if a
// PLAIN coalesced read of the same bytes goes faster than attention already does.
//
// So: read the same number of bytes with the canonical access pattern (float4,
// consecutive threads at consecutive addresses, grid-stride) and report the ratio
// against attention's measured rate. Then the decision is arithmetic:
//
//   ceiling >> attention  -> tiling has a target; attempt it
//   ceiling ~= attention  -> attention is already at the memory limit for this
//                            footprint; a tiling rewrite cannot help, close it
//
// Model-neutral on purpose: every decoder family with a long-context KV read will
// want this number, and a per-family copy would drift.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "test_kernel_budget.h"

namespace
{
bool HasDevice()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// float4 = 16 B per thread = 512 B per warp: the canonical coalesced maximum.
__global__ void StreamReadCeiling(const float4 *__restrict__ src, size_t n4,
                                  float *__restrict__ sink, int grid)
{
    float acc = 0.0f;
    const size_t stride = static_cast<size_t>(grid) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n4; i += stride)
    {
        const float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    // One predicated store per block: without a sink the compiler is entitled to
    // delete the whole loop, and the "ceiling" would be the launch overhead.
    if (acc == 1234.5678f) sink[blockIdx.x] = acc;
}

struct Reading
{
    double gbps;
    double us;
};

Reading Measure(const float *d_buf, size_t bytes, float *d_sink, int iters,
                cudaStream_t stream)
{
    const size_t n4 = bytes / sizeof(float4);
    const int block = 256;
    // Enough blocks to fill the array many times over, so the grid-stride loop is
    // the access pattern rather than the block count.
    int grid = 0;
    REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &grid, reinterpret_cast<const void *>(StreamReadCeiling), block,
                0) == cudaSuccess);
    int sms = 0;
    REQUIRE(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) ==
            cudaSuccess);
    grid = std::max(1, grid * sms);
    cudaEvent_t a, b;
    REQUIRE(cudaEventCreate(&a) == cudaSuccess);
    REQUIRE(cudaEventCreate(&b) == cudaSuccess);
    for (int i = 0; i < 2; ++i)
        StreamReadCeiling<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4 *>(const_cast<float *>(d_buf)), n4,
            d_sink, grid);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(cudaEventRecord(a, stream) == cudaSuccess);
    for (int i = 0; i < iters; ++i)
        StreamReadCeiling<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float4 *>(const_cast<float *>(d_buf)), n4,
            d_sink, grid);
    REQUIRE(cudaEventRecord(b, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, a, b) == cudaSuccess);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    Reading r;
    r.us = static_cast<double>(ms) * 1000.0 / iters;
    r.gbps = static_cast<double>(bytes) * iters / (static_cast<double>(ms) * 1e-3) /
             1e9;
    return r;
}
}  // namespace

TEST_CASE("a plain coalesced read of the KV cache size is the ceiling",
          "[perf][bandwidth]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Sizes taken from a real Xing-4.0 KV cache: 8191 positions is the long-context
    // case where attention is slowest, 2048 the canonical benchmark context, 1024
    // the one that still fits L2 and where the old note measured 1364 GB/s.
    struct Case {
        const char *name;
        size_t bytes;
    };
    const std::vector<Case> cases = {
        {"ctx 1024 (L2-resident)", 1024ull * 32 * (192 + 128) * 4},
        {"ctx 2048 (canonical)", 2048ull * 32 * (192 + 128) * 4},
        {"ctx 8191 (long context)", 8191ull * 32 * (192 + 128) * 4},
    };

    float *d_sink = nullptr;
    REQUIRE(cudaMalloc(&d_sink, 4096 * sizeof(float)) == cudaSuccess);
    std::string table;
    double ceiling_long = 0.0;
    for (const auto &c : cases)
    {
        float *d_buf = nullptr;
        REQUIRE(cudaMalloc(&d_buf, c.bytes) == cudaSuccess);
        // Content is irrelevant to bandwidth but must not be a NaN pattern that
        // could be optimised away; cudaMemset gives a uniform, finite buffer.
        REQUIRE(cudaMemset(d_buf, 0x3f, c.bytes) == cudaSuccess);
        const auto r = Measure(d_buf, c.bytes, d_sink, 20, stream);
        table += std::string(c.name) + ": " + std::to_string(static_cast<int>(r.gbps)) +
                 " GB/s (" + std::to_string(static_cast<int>(r.us)) + " us)\n";
        if (std::string(c.name).rfind("ctx 8191", 0) == 0) ceiling_long = r.gbps;
        cudaFree(d_buf);
    }
    cudaFree(d_sink);
    cudaStreamDestroy(stream);

    // The two small cases are CACHE bandwidth, not DRAM: the same buffer is re-read
    // every iteration and fits in this card's L2, so they come out at 5-6 TB/s and
    // say nothing about a DRAM-bound read. Only the 335 MB case is the ceiling a
    // long-context KV read is measured against, which is why `ceiling_long` is the
    // only figure the budget and the decision use.
    WARN("streaming-read ceiling, float4 grid-stride "
         "(small cases are L2-resident, not DRAM):\n"
         << table);
    INFO("long-context ceiling = " << ceiling_long
         << " GB/s; Xing attention measured 744-817 GB/s on the same bytes");

    REQUIRE(ceiling_long > 0.0);
#if defined(NDEBUG)
    // A floor on the MEASUREMENT, not a claim about attention: if a plain
    // coalesced read of 335 MB cannot sustain this, the probe itself is broken
    // (clock throttling, a device other than the one the budgets were taken on,
    // an occupancy collapse) and any ratio computed from it would be fiction.
    // The value is a conservative fraction of what the dense decode GEMVs already
    // achieve on this card (xing.gemv.dense_decode_gbps observed 1202 GB/s).
    kernel_budget::Require("mem.stream_read_ceiling_gbps", ceiling_long,
                           kernel_budget::Kind::kMin,
                           "float4 grid-stride read of a 335 MB buffer; the "
                           "reference point for any KV-read bandwidth claim");
#else
    WARN("bandwidth budget is a Release-only gate");
#endif
}
