// Qwen fused MLP: accumulate-mode NVFP4 GEMM + full MLP parity vs split.
//
// Direction A: the MLP residual-add is absorbed into the down GEMM via an
// `accumulate` mode on RunQwenNvfp4Gemm (y = y + sum). This unit verifies the
// acc mode, and (Task 2) the full fused MLP vs split parity for seq=1 / seq>1.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "lsxcommon/lsx_inference_load.h"
#include "lsxcommon/qwen_inference.h"

namespace attn = lsxcommon::attn;
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/schema.h"

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

// Host reference for the shared NVFP4 decode: the canonical E4M3FN block
// scale (lsxcommon::e4m3_to_float, single source nvfp4_scale.h — it used to
// be a local power-of-2 replica) and the E2M1 nibble
// table. Weight packed layout is [N-output, K-input] row-major: element
// e = n*K + k lives at byte e>>1 (low nibble if even, high if odd); the E4M3
// scale is per-16 elements (e>>4).
const float kHostE2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Canonical block-scale decode (complete E4M3FN, single source nvfp4_scale.h).
float HostE4M3Scale(uint8_t u)
{
    return lsxcommon::e4m3_to_float(u);
}

double HostDeqW(const std::vector<uint8_t> &pk, const std::vector<uint8_t> &sc,
                float gs, int64_t e)
{
    const uint8_t b = pk[(size_t)(e >> 1)];
    const uint8_t nib = (e & 1) ? (b >> 4) : (b & 0x0Fu);
    return (double)kHostE2M1[nib] * (double)HostE4M3Scale(sc[(size_t)(e >> 4)]) * (double)gs;
}

double HostSilu(double x) { return x / (1.0 + std::exp(-x)); }

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

TEST_CASE("RunQwenFusedMLP full-MLP parity vs host split", "[qwen][mlp][fused]")
{
    constexpr int H = 16, I = 32;
    lsxcommon::attn::LayerWeights lw;
    lw.loaded = true;
    // MLP weights: gate/up are [output=I, input-H] (gemm K=H reduction),
    // down is [output=H, input=I] (gemm K=I reduction).
    FillNvfp4(lw.mlp_gate, I, H, 11, 1);
    FillNvfp4(lw.mlp_up,   I, H, 3,  2);
    FillNvfp4(lw.mlp_down, H, I, 7,  3);

    // Host copies of the same packed/scales (mirrors FillNvfp4's deterministic
    // generation) for the deq reference.
    auto host_bytes = [](int64_t bytes, uint8_t seed_pack, uint8_t seed_scale) {
        std::vector<uint8_t> pk((size_t)bytes), sc((size_t)(bytes / 8));
        for (int64_t i = 0; i < bytes; ++i) pk[(size_t)i] = (uint8_t)((i * 7 + seed_pack) & 0xFF);
        for (int64_t i = 0; i < bytes / 8; ++i) sc[(size_t)i] = (uint8_t)(128 + ((int)(i + seed_scale) % 32));
        return std::make_pair(pk, sc);
    };
    auto g = host_bytes(lw.mlp_gate.packed_bytes, 11, 1);
    auto u = host_bytes(lw.mlp_up.packed_bytes,   23, 2);
    auto d = host_bytes(lw.mlp_down.packed_bytes, 7,  3);

    lsxcommon::qwen::QwenTopology top{};
    top.hidden_size = H;

    for (int seq : {1, 2}) {
        // post-RMSNormed input stream [seq, H] (deterministic, small).
        std::vector<__half> hx((size_t)seq * H), hres((size_t)seq * H);
        for (int t = 0; t < seq; ++t)
            for (int k = 0; k < H; ++k)
                hx[(size_t)t * H + k] =
                    __float2half(0.05f * (float)(((t * 3 + k * 5) % 7) - 3));
        // residual seed into which the down-GEMM accumulates (FP16).
        for (int t = 0; t < seq; ++t)
            for (int h = 0; h < H; ++h)
                hres[(size_t)t * H + h] =
                    __float2half(0.1f * (float)(((t + h * 2) % 5) - 2));

        __half *dx, *dresid, *dscratch;
        cudaMalloc(&dx,      (size_t)seq * H * 2);
        cudaMalloc(&dresid,  (size_t)seq * H * 2);
        cudaMalloc(&dscratch, (size_t)2 * seq * I * 2);
        cudaMemcpy(dx,      hx.data(),   (size_t)seq * H * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dresid,  hres.data(), (size_t)seq * H * 2, cudaMemcpyHostToDevice);

        REQUIRE(lsxcommon::qwen::RunQwenFusedMLP(
            nullptr, lw, top, seq, dx, dresid, dscratch) == cudaSuccess);
        cudaDeviceSynchronize();

        std::vector<__half> hout((size_t)seq * H);
        cudaMemcpy(hout.data(), dresid, (size_t)seq * H * 2, cudaMemcpyDeviceToHost);

        // Host split reference (FP64): gate=x@deq(gate), up=x@deq(up),
        // act=silu(gate)*up, out=act@deq(down); expected = resid + out.
        for (int t = 0; t < seq; ++t) {
            std::vector<double> gate((size_t)I), up((size_t)I), act((size_t)I);
            for (int i = 0; i < I; ++i) {
                double gg = 0.0, uu = 0.0;
                for (int k = 0; k < H; ++k) {
                    const double x = __half2float(hx[(size_t)t * H + k]);
                    gg += x * HostDeqW(g.first, g.second, lw.mlp_gate.global_scale,
                                       (int64_t)i * H + k);
                    uu += x * HostDeqW(u.first, u.second, lw.mlp_up.global_scale,
                                       (int64_t)i * H + k);
                }
                gate[(size_t)i] = gg;
                up[(size_t)i]   = uu;
                act[(size_t)i]  = HostSilu(gg) * uu;
            }
            for (int h = 0; h < H; ++h) {
                double o = 0.0;
                for (int i = 0; i < I; ++i)
                    o += act[(size_t)i] * HostDeqW(
                             d.first, d.second, lw.mlp_down.global_scale,
                             (int64_t)h * I + i);
                const double ref =
                    (double)__half2float(hres[(size_t)t * H + h]) + o;
                const double got = (double)__half2float(hout[(size_t)t * H + h]);
                CAPTURE(seq, t, h, ref, got);
                CHECK(std::fabs(got - ref) <= 1e-2 * (1.0 + std::fabs(ref)));
            }
        }

        cudaFree(dx); cudaFree(dresid); cudaFree(dscratch);
    }
    lw.mlp_gate.Free(); lw.mlp_up.Free(); lw.mlp_down.Free();
}