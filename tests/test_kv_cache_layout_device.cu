// Device-side agreement for KvCacheGeometry.
//
// constexpr (not __host__ __device__) already makes host and device share ONE
// function, so this is close to a tautology - and that is the point: it fails the
// day someone drops constexpr, or adds a second definition behind a macro. Cheap
// insurance against exactly the host/device drift the single-function form exists
// to make impossible.
#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/lsx_kv_cache.h"

namespace
{
bool HasDevice()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

__global__ void KvOffsetProbe(lsxcommon::KvCacheGeometry g, int64_t layer,
                              int64_t head, int64_t pos, int64_t dim,
                              int64_t *out)
{
    *out = g.Offset(layer, head, pos, dim);
}
}  // namespace

TEST_CASE("device index agrees with host index", "[kv_layout][fast]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    for (auto layout : {lsxcommon::KvCacheLayout::kHeadMajor,
                        lsxcommon::KvCacheLayout::kPosMajor})
    {
        const auto g = lsxcommon::KvCacheGeometry::Make(layout, 64, 32, 192, 40);
        int64_t *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_out, sizeof(int64_t)) == cudaSuccess);
        struct Probe
        {
            int64_t layer, head, pos, dim;
        };
        const Probe probes[] = {{0, 0, 0, 0},     {3, 7, 11, 191},
                                {39, 31, 63, 5}, {39, 31, 63, 191}};
        for (const auto &p : probes)
        {
            KvOffsetProbe<<<1, 1>>>(g, p.layer, p.head, p.pos, p.dim, d_out);
            REQUIRE(cudaGetLastError() == cudaSuccess);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
            int64_t got = 0;
            REQUIRE(cudaMemcpy(&got, d_out, sizeof(int64_t),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            CHECK(got == g.Offset(p.layer, p.head, p.pos, p.dim));
        }
        cudaFree(d_out);
    }
}
