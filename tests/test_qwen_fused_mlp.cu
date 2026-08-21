// Qwen fused MLP: accumulate-mode NVFP4 GEMM + full MLP parity vs split.
//
// Direction A: the MLP residual-add is absorbed into the down GEMM via an
// `accumulate` mode on RunQwenNvfp4Gemm (y = y + sum). This unit verifies the
// acc mode, and (Task 2) the full fused MLP vs split parity for seq=1 / seq>1.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"

namespace {

void FillNvfp4(lsxcommon::qwen::QwenNvfp4Weight &w, int K, int N,
               uint8_t seed_pack, uint8_t seed_scale)
{
    w.rows = K; w.cols = N;
    w.packed_bytes = (int64_t)K * N / 2;
    w.scale_bytes  = (int64_t)K * N / 16;
    w.global_scale = 0.5f;
    cudaMalloc(&w.d_packed, (size_t)w.packed_bytes);
    cudaMalloc(&w.d_scales, (size_t)w.scale_bytes);
    std::vector<uint8_t> pk((size_t)w.packed_bytes), sc((size_t)w.scale_bytes);
    for (int64_t i = 0; i < w.packed_bytes; ++i) pk[(size_t)i] = (uint8_t)((i * 7 + seed_pack) & 0xFF);
    for (int64_t i = 0; i < w.scale_bytes; ++i)  sc[(size_t)i]  = (uint8_t)(128 + ((int)(i + seed_scale) % 32));
    cudaMemcpy((void*)w.d_packed, pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)w.d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
}

}  // namespace

TEST_CASE("RunQwenNvfp4Gemm accumulate adds into the output buffer", "[qwen][gemm][acc]")
{
    constexpr int seq = 2, K = 64, N = 128;
    std::vector<__half> hx(seq * K), hsrc(seq * N);
    std::vector<uint8_t> hp((size_t)K*N/2), hs((size_t)K*N/16);
    for (int i = 0; i < seq*K; ++i) hx[i] = __float2half((float)((i%7)-3));
    for (int i = 0; i < (int)hp.size(); ++i) hp[i] = (uint8_t)((i*13)&0xFF);
    for (int i = 0; i < (int)hs.size(); ++i) hs[i] = (uint8_t)(128+(i%32));
    for (int i = 0; i < seq*N; ++i) hsrc[i] = __float2half(0.5f + 0.01f*(float)(i%9));
    const float gs = 0.5f;

    __half *dx, *dout, *dacc; uint8_t *dp, *ds;
    cudaMalloc(&dx, seq*K*2);
    cudaMalloc(&dout, seq*N*2); cudaMalloc(&dacc, seq*N*2);
    cudaMalloc(&dp, hp.size()); cudaMalloc(&ds, hs.size());
    cudaMemcpy(dx, hx.data(), seq*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
        nullptr, dx, dp, ds, gs, dout, seq, K, N) == cudaSuccess);
    cudaMemcpy(dacc, hsrc.data(), seq*N*2, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
        nullptr, dx, dp, ds, gs, dacc, seq, K, N, /*acc=*/true) == cudaSuccess);

    std::vector<__half> ho(seq*N), ha(seq*N);
    cudaMemcpy(ho.data(), dout, seq*N*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), dacc, seq*N*2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < seq*N; ++i) {
        const float sum = __half2float(hsrc[i]) + __half2float(ho[i]);
        CHECK(std::abs(__half2float(ha[i]) - __half2float(__float2half(sum))) < 1e-3f);
    }
    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dout); cudaFree(dacc);
}