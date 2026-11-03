// TDD — Blackwell audit micro-benchmark: raw NVFP4 GEMM kernel throughput on
// Qwen3.5-9B-representative shapes. Reporting test (no hard gates): separates
// kernel-efficiency bottlenecks from host-overhead bottlenecks by measuring
// the production entry directly with CUDA events.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_tc_gemm.h"
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
        char msg[512];
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
        // Round-18 MoE expert decode shapes (qwen35moe geometry).
        {2048, 512, "moe_w13_half"},
        {512, 2048, "moe_w2"},
        // LM head decode shape (qwen35moe vocab): DRAM-bound candidate.
        {2048, 248320, "lm_head"},
    };

    cudaStream_t st = nullptr;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    cudaEvent_t t0, t1;
    REQUIRE(cudaEventCreate(&t0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&t1) == cudaSuccess);

    auto bench_call =
        [&](auto &&call_fn) -> float {
        for (int w = 0; w < 5; ++w)
            REQUIRE(call_fn() == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
        constexpr int kIters = 50;
        REQUIRE(cudaEventRecord(t0, st) == cudaSuccess);
        for (int it = 0; it < kIters; ++it)
            REQUIRE(call_fn() == cudaSuccess);
        REQUIRE(cudaEventRecord(t1, st) == cudaSuccess);
        REQUIRE(cudaEventSynchronize(t1) == cudaSuccess);
        float ms = -1.0f;
        REQUIRE(cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess);
        return ms * 1000.0f / kIters;
    };

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

        // Scalar/tile dispatch baseline (fast-math off).
        lsxcommon::qwen::SetQwenTcFastMath(false);
        const float us_scalar = bench_call([&] {
            return lsxcommon::lsx_nvfp4::RunLsxNvfp4Gemm(
                st, dx, dp, ds, 0.125f, dy, 1, K, N);
        });
        // Weight-streaming GEMV.
        const float us_gemv = bench_call([&] {
            return lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                st, dx, dp, ds, 0.125f, dy, K, N);
        });
        // TC W4A4 mxf4nvf4 (current fast-math production baseline at seq==1).
        const float us_tc = bench_call([&] {
            return lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                st, dx, dp, ds, 0.125f, dy, 1, K, N, false, nullptr, nullptr);
        });

        // Weight bytes touched per call: packed + scales.
        const double mbytes = double(hp.size() + hs.size()) / (1024.0 * 1024.0);
        const double gemv_bw = mbytes / (us_gemv * 1e-3);   // MiB/s
        {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "[MICRO][%s] K=%d N=%d tile=%.1f us/call, tc=%.1f, "
                     "gemv=%.1f us/call (tile/x%.2f), weight_mb=%.1f, "
                     "gemv_bw=%.0f MiB/s",
                     sh.label, K, N, us_scalar, us_tc, us_gemv,
                     us_gemv > 0.0f ? us_scalar / us_gemv : 0.0f,
                     mbytes, gemv_bw);
            lsxcommon::log::info(msg);
        }

        cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(st);
}
