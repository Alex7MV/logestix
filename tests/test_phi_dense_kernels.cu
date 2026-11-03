// Phi dense kernel-level parity tests (Model Ingestion Protocol step 5).
// Host-only CPU references in FP64/FP32 against the exposed launchers:
//   RunPhiRmsNormKernel / RunPhiRopeKernel / RunPhiAttentionKernel.
#include <catch2/catch_all.hpp>

#include "test_kernel_budget.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <lsxcommon/phi_dense_inference.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::phi;
namespace lsxg = lsxcommon::lsx_nvfp4;

namespace {

__half H(float v) { return __float2half(v); }

float lcg(int64_t &s)
{
    s = (s * 6364136223846793005LL + 1442695040888963407LL);
    return static_cast<float>((s >> 40) & 0xffff) / 65536.0f - 0.5f;
}

std::vector<__half> ToHalf(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

} // namespace

TEST_CASE("phi launchers delegate to the neutral dense layer",
          "[phi][kernels]")
{
    // The five parity cases for these kernels live in test_dense_kernels.cu;
    // this keeps the family delegates (RunPhi*) on the tested path.
    const int rows = 2, hidden = 64;
    std::vector<float> x(rows * hidden, 0.5f), w(hidden, 1.0f);
    auto hx = ToHalf(x), hw = ToHalf(w);
    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, hw.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunPhiRmsNormKernel(nullptr, dout, dx, dw, rows, hidden,
                                1e-5f) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> out(rows * hidden);
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < hidden; ++j)
            CHECK(std::fabs(__half2float(out[r * hidden + j]) - 1.0f) < 2e-2f);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

TEST_CASE("phi seq1 GEMV decode-shape aggregate bandwidth budget",
          "[phi][kernels][nvfp4][budget]")
{
#if defined(NDEBUG)
    
        lsxg::SetLsxTcFastMath(true);
        lsxg::SetLsxSeq1Gemv(true);

    struct Shape { int K, N; const char *name; };
    // Every GEMM the phi decode step issues, one entry per dispatch class.
    // `qkv` is the fused [NH*HD + 2*NKV*HD, K] projection: N=7680 puts it on
    // the sliced (N < 8192) route while `o`/`down` sit below and `gate_up`/
    // `head` above, so leaving it out measured 4 of the 5 shapes and none of
    // the ones the step spends its time in (22 MB/token, 17% of the weights).
    const Shape shapes[] = {
        {5120, 7680, "qkv"},      {5120, 5120, "q_o"},
        {5120, 35840, "gate_up"}, {17920, 5120, "down"},
        {5120, 100352, "head"},
    };
    constexpr int kIters = 30;
    constexpr size_t kFlushBytes = 128u << 20;  // > 5090 L2 (96 MiB)
    uint8_t *flush = nullptr;
    REQUIRE(cudaMalloc(&flush, kFlushBytes) == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    double total_bytes = 0.0, total_us = 0.0;
    std::string report;
    for (const auto &c : shapes) {
        const size_t pbytes = static_cast<size_t>(c.K) * c.N / 2;
        const size_t sbytes = static_cast<size_t>(c.K) * c.N / 16;
        std::vector<__half> hx(static_cast<size_t>(c.K));
        std::vector<uint8_t> hp(pbytes), hs(sbytes);
        for (int i = 0; i < c.K; ++i)
            hx[static_cast<size_t>(i)] = __float2half(float((i % 13) - 6) * 0.25f);
        for (size_t i = 0; i < pbytes; ++i)
            hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
        for (size_t i = 0; i < sbytes; ++i)
            hs[i] = static_cast<uint8_t>(128 + (i % 48));
        __half *dx = nullptr, *dy = nullptr;
        uint8_t *dp = nullptr, *ds = nullptr;
        REQUIRE(cudaMalloc(&dx, static_cast<size_t>(c.K) * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dp, pbytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&ds, sbytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, static_cast<size_t>(c.N) * 2) == cudaSuccess);
        cudaMemcpy(dx, hx.data(), static_cast<size_t>(c.K) * 2,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dp, hp.data(), pbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), sbytes, cudaMemcpyHostToDevice);
        REQUIRE(lsxg::RunLsxNvfp4GemmSeq1GemmV(nullptr, dx, dp, ds, 0.125f, dy, c.K,
                                         c.N, false) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        double shape_us = 0.0;
        for (int i = 0; i < kIters; ++i) {
            REQUIRE(cudaMemsetAsync(flush, 0, kFlushBytes) == cudaSuccess);
            REQUIRE(cudaEventRecord(e0) == cudaSuccess);
            REQUIRE(lsxg::RunLsxNvfp4GemmSeq1GemmV(nullptr, dx, dp, ds, 0.125f, dy,
                                             c.K, c.N, false) == cudaSuccess);
            REQUIRE(cudaEventRecord(e1) == cudaSuccess);
            REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
            float ms = 0.0f;
            REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
            shape_us += static_cast<double>(ms) * 1000.0;
        }
        const double gbps =
            static_cast<double>(pbytes + sbytes) * kIters /
            (shape_us * 1e-6) / 1e9;
        report += std::string(c.name) + " (" + std::to_string(c.K) + "x" +
                  std::to_string(c.N) + ") = " + std::to_string(gbps) +
                  " GB/s\n";
        total_us += shape_us;
        total_bytes += static_cast<double>(pbytes + sbytes) * kIters;
        cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
    }
    const double agg = total_bytes / (total_us * 1e-6) / 1e9;
    INFO(report << "aggregate = " << agg << " GB/s");
    // Measured 2026-09-22 (RTX 5090): 812 (one-warp-per-row everywhere)
    // -> 873 (occupancy floor) -> 918 (128-bit activation loads + half22float2
    // in the slice/warp-row kernels) -> **1327** with the branchless
    // E2M1Value codec. That last step is the big one: the old `switch (mag)`
    // compiled to a data-dependent indirect branch, so the SAME kernel ran at
    // 655 GB/s on real NVFP4 weights vs 933 on a synthetic uniform buffer; the
    // branchless form lifted both (this harness, memset L2 flush: gate_up
    // 906 -> 1336, down 858 -> 1306, head 934 -> 1369, q_o 576 -> 764). Raw
    // streaming ceiling on this part = ~1618 GB/s. Budget sits just below the
    // current figure so a silent regression to the branchy decode,
    // low-occupancy or scalar-x launch fails here.
    //
    // 2026-10-24: the shape set gained the real `qkv` shape (5120x7680), which
    // the four-shape set never measured, so the aggregate is not comparable to
    // the 1327 above: the five-shape aggregate reads 1258-1267 and the worst
    // shape is q_o at 722 (sliced route, N < 8192). Same day, the E4M3FN
    // block-scale decode was folded to a single return (bit-identical) and
    // q_o moved 722 -> 758, which is also what the end-to-end phi decode
    // showed (+0.8%). The row was recorded from the five-shape run.
    kernel_budget::Require("phi.dense_gemm.bandwidth", agg,
                           kernel_budget::Kind::kMin,
                           "aggregate dense-decode NVFP4 GEMV, GB/s");
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(flush);
        lsxg::SetLsxTcFastMath(false);
        lsxg::SetLsxSeq1Gemv(false);
#endif
}
