// TDD — Blackwell audit micro-benchmark: raw NVFP4 GEMM kernel throughput on
// Qwen3.5-9B-representative shapes. Reporting test (no hard gates): separates
// kernel-efficiency bottlenecks from host-overhead bottlenecks by measuring
// the production entry directly with CUDA events.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/log.h"

namespace {

struct GemmShape { int K, N; const char *label; };

} // namespace

TEST_CASE("NVFP4 GEMM micro-benchmark (seq=1 decode shapes)",
          "[qwen][gemm][perf][micro]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    int mem_clock_khz = 0, bus_width_bits = 0;
    REQUIRE(cudaDeviceGetAttribute(&mem_clock_khz,
                                   cudaDevAttrMemoryClockRate, dev) ==
            cudaSuccess);
    REQUIRE(cudaDeviceGetAttribute(&bus_width_bits,
                                   cudaDevAttrGlobalMemoryBusWidth, dev) ==
            cudaSuccess);
    {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "[MICRO] device=%s sms=%d bw_theory_gb_s=%.0f", prop.name,
                 prop.multiProcessorCount,
                 2.0 * mem_clock_khz * (bus_width_bits / 8.0) / 1e6);
        lsxcommon::log::info(msg);
    }

    const GemmShape shapes[] = {
        {4096, 4096, "attn_qkv_o"},
        {4096, 9728, "mlp_gate_up"},
        {9728, 4096, "mlp_down"},
        {1536, 4096, "head_proj"},
    };

    cudaStream_t st = nullptr;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    cudaEvent_t t0, t1;
    REQUIRE(cudaEventCreate(&t0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&t1) == cudaSuccess);

    for (const auto &sh : shapes) {
        const int K = sh.K, N = sh.N;
        std::vector<__half> hx((size_t)K);
        std::vector<uint8_t> hp((size_t)((int64_t)K * N / 2)),
            hs((size_t)((int64_t)K * N / 16));
        for (int i = 0; i < K; ++i)
            hx[i] = __float2half(float((i % 13) - 6) * 0.25f);
        for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i * 29) & 0xFF);
        for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128 + (i % 48));

        __half *dx, *dy; uint8_t *dp, *ds;
        REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dp, hp.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&ds, hs.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, (size_t)N * 2) == cudaSuccess);
        cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

        // Warmup.
        for (int w = 0; w < 5; ++w)
            REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                        st, dx, dp, ds, 0.125f, dy, 1, K, N) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

        constexpr int kIters = 50;
        REQUIRE(cudaEventRecord(t0, st) == cudaSuccess);
        for (int it = 0; it < kIters; ++it)
            REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                        st, dx, dp, ds, 0.125f, dy, 1, K, N) == cudaSuccess);
        REQUIRE(cudaEventRecord(t1, st) == cudaSuccess);
        REQUIRE(cudaEventSynchronize(t1) == cudaSuccess);
        float ms = -1.0f;
        REQUIRE(cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess);
        const float us_per_call = ms * 1000.0f / kIters;

        // Weight bytes touched per call: packed + scales.
        const double mbytes = double(hp.size() + hs.size()) / (1024.0 * 1024.0);
        const double eff_gbs = mbytes / (us_per_call * 1e-3) / 1000.0;
        {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "[MICRO][%s] K=%d N=%d %.1f us/call, weight_mb=%.1f, "
                     "effective_bw=%.0f GB/s",
                     sh.label, K, N, us_per_call, mbytes, eff_gbs);
            lsxcommon::log::info(msg);
        }

        cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(st);
}
