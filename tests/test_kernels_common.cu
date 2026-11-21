// Neutral elementwise cast kernels (dedup phase 3, Task 3).
//
// One grid-stride int64_t implementation per direction replaces the
// per-family one-shot/grid-stride copies. The casts are elementwise with no
// RMW: every index is written exactly once with the same value regardless
// of grid shape, so a capped grid must cover n > blocks*threads. Rounding
// is pinned: __float2half (RNE), __float2bfloat16_rn, and the bf16->f16
// composition __float2half(__bfloat162float(x)).
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsx_kernels_common.cuh"

#include <cstdint>
#include <vector>

namespace {

float SrcVal(int64_t i)
{
    return 0.001f * static_cast<float>((i * 7919) % 100003) - 50.0f;
}

int Blocks(int64_t n)
{
    const int64_t b = (n + 255) / 256;
    return b > 1024 ? 1024 : static_cast<int>(b); // cap: forces grid-stride
}

TEST_CASE("elementwise f32<->f16 casts cover every index once",
          "[kernels][convert]")
{
    for (int64_t n : {1, 255, 1024, 100000}) {
        INFO("n=" << n);
        std::vector<float> src(n), back(n);
        std::vector<__half> h(n);
        for (int64_t i = 0; i < n; ++i) src[i] = SrcVal(i);
        float *d_src = nullptr, *d_back = nullptr;
        __half *d_h = nullptr;
        REQUIRE(cudaMalloc(&d_src, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_h, n * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_back, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_src, src.data(), n * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        lsxcommon::kernels::F32ToF16Kernel<><<<Blocks(n), 256>>>(d_src, d_h, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        lsxcommon::kernels::F16ToF32Kernel<><<<Blocks(n), 256>>>(d_h, d_back, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(h.data(), d_h, n * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(back.data(), d_back, n * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_src);
        cudaFree(d_h);
        cudaFree(d_back);
        for (int64_t i = 0; i < n; ++i) {
            const __half ref = __float2half(src[i]);
            REQUIRE(__half2float(h[i]) == __half2float(ref));
            REQUIRE(back[i] == __half2float(ref));
        }
    }
}

TEST_CASE("elementwise f32->bf16 uses round-to-nearest-even",
          "[kernels][convert]")
{
    for (int64_t n : {1, 255, 1024, 100000}) {
        INFO("n=" << n);
        std::vector<float> src(n);
        std::vector<__nv_bfloat16> out(n);
        for (int64_t i = 0; i < n; ++i) src[i] = SrcVal(i);
        float *d_src = nullptr;
        __nv_bfloat16 *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_src, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_out, n * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_src, src.data(), n * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        lsxcommon::kernels::F32ToBf16Kernel<><<<Blocks(n), 256>>>(d_src, d_out, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(out.data(), d_out, n * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_src);
        cudaFree(d_out);
        for (int64_t i = 0; i < n; ++i) {
            const __nv_bfloat16 ref = __float2bfloat16_rn(src[i]);
            REQUIRE(__bfloat162float(out[i]) == __bfloat162float(ref));
        }
    }
}

TEST_CASE("elementwise bf16->f16 matches the two-step composition",
          "[kernels][convert]")
{
    for (int64_t n : {1, 255, 1024, 100000}) {
        INFO("n=" << n);
        std::vector<__nv_bfloat16> src(n);
        std::vector<__half> out(n);
        for (int64_t i = 0; i < n; ++i)
            src[i] = __float2bfloat16_rn(SrcVal(i));
        __nv_bfloat16 *d_src = nullptr;
        __half *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_src, n * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_out, n * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_src, src.data(), n * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        lsxcommon::kernels::Bf16ToF16Kernel<><<<Blocks(n), 256>>>(d_src, d_out, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(out.data(), d_out, n * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_src);
        cudaFree(d_out);
        for (int64_t i = 0; i < n; ++i) {
            const __half ref = __float2half(__bfloat162float(src[i]));
            REQUIRE(__half2float(out[i]) == __half2float(ref));
        }
    }
}

TEST_CASE("elementwise bf16->f32 and f16->bf16 cover every index once",
          "[kernels][convert]")
{
    for (int64_t n : {1, 255, 1024, 100000}) {
        INFO("n=" << n);
        std::vector<__nv_bfloat16> bsrc(n);
        std::vector<float> bout(n);
        std::vector<__half> hsrc(n);
        std::vector<__nv_bfloat16> hout(n);
        for (int64_t i = 0; i < n; ++i) {
            bsrc[i] = __float2bfloat16_rn(SrcVal(i));
            hsrc[i] = __float2half(SrcVal(i));
        }
        __nv_bfloat16 *d_bsrc = nullptr, *d_hout = nullptr;
        float *d_bout = nullptr;
        __half *d_hsrc = nullptr;
        REQUIRE(cudaMalloc(&d_bsrc, n * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_bout, n * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_hsrc, n * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_hout, n * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_bsrc, bsrc.data(), n * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_hsrc, hsrc.data(), n * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        lsxcommon::kernels::Bf16ToF32Kernel<><<<Blocks(n), 256>>>(d_bsrc, d_bout, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        lsxcommon::kernels::F16ToBf16Kernel<><<<Blocks(n), 256>>>(d_hsrc, d_hout, n);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(bout.data(), d_bout, n * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(hout.data(), d_hout, n * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(d_bsrc);
        cudaFree(d_bout);
        cudaFree(d_hsrc);
        cudaFree(d_hout);
        for (int64_t i = 0; i < n; ++i) {
            REQUIRE(bout[i] == __bfloat162float(bsrc[i]));
            const __nv_bfloat16 ref = __float2bfloat16_rn(__half2float(hsrc[i]));
            REQUIRE(__bfloat162float(hout[i]) == __bfloat162float(ref));
        }
    }
}

} // namespace