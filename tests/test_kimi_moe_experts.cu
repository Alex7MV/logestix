#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_load.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

using namespace lsxcommon::kimi;

constexpr int I = kKimiMoeIntermediate;
constexpr int Hd = kKimiHidden;

float Frand(int seed, int i)
{
    uint32_t v = static_cast<uint32_t>(seed * 2654435761u + i * 40503u);
    v ^= v >> 13;
    v *= 1274126177u;
    v ^= v >> 16;
    return (static_cast<float>(v % 20001) - 10000.0f) / 10000.0f;
}

// Canonical E4M3 block-scale decode (matches nvfp4_codec::E4M3ScaleToFloat).
float HostE4M3(uint8_t b)
{
    const uint8_t e4 = (b >> 3) & 0x0F;
    if (e4 == 0) return 0.0f;
    return std::ldexp(1.0f, static_cast<int>(e4) - 7);
}

// Encodes a power-of-two scale 2^e as an E4M3 block-scale byte.
uint8_t EncodeScalePow2(int e)
{
    return static_cast<uint8_t>(((e + 7) & 0xF) << 3);
}

// Builds one full-geometry projection: packed E2M1 nibbles + E4M3 scales.
struct ProjData {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    float global_scale;
};

ProjData MakeProj(int seed, int64_t rows, int64_t cols)
{
    ProjData p;
    const int64_t numel = rows * cols;
    p.packed.assign(numel / 2, 0);
    p.scales.assign(numel / 16, 0);
    p.global_scale = 0.75f;
    for (int64_t i = 0; i < numel; ++i) {
        // Nibble pattern biased to small magnitudes (valid E2M1 codes 0..15).
        const int nib = static_cast<int>(Frand(seed, i) * 8.0f) & 0xF;
        p.packed[i >> 1] |= static_cast<uint8_t>(nib << ((i & 1) ? 4 : 0));
        if ((i & 15) == 0)
            p.scales[i >> 4] =
                EncodeScalePow2(-3 + (static_cast<int>(Frand(seed + 1, i)) & 3));
        if ((i & 1) == 0)
            p.packed[i >> 1] = static_cast<uint8_t>(p.packed[i >> 1]);
    }
    // Fix packing: rebuild cleanly lo-nibble-first.
    std::fill(p.packed.begin(), p.packed.end(), 0);
    for (int64_t i = 0; i < numel; ++i) {
        const int nib = static_cast<int>(Frand(seed, i) * 8.0f) & 0xF;
        p.packed[i >> 1] |= static_cast<uint8_t>(nib << ((i & 1) ? 4 : 0));
    }
    return p;
}

} // namespace

TEST_CASE("Kimi routed-expert FFN matches CPU reference", "[kimi][experts]")
{
    // Full-geometry projections (real Kimi MoE shapes).
    ProjData w1 = MakeProj(11, I, Hd);   // gate
    ProjData w3 = MakeProj(12, I, Hd);   // up
    ProjData w2 = MakeProj(13, Hd, I);   // down

    KimiExpertEntry entry{};
    entry.host_w1 = w1.packed.data();
    entry.host_w3 = w3.packed.data();
    entry.host_w2 = w2.packed.data();
    entry.host_w1_scales = w1.scales.data();
    entry.host_w3_scales = w3.scales.data();
    entry.host_w2_scales = w2.scales.data();
    entry.w1_global_scale = w1.global_scale;
    entry.w3_global_scale = w3.global_scale;
    entry.w2_global_scale = w2.global_scale;
    entry.w1_bytes = static_cast<int64_t>(w1.packed.size());
    entry.w3_bytes = static_cast<int64_t>(w3.packed.size());
    entry.w2_bytes = static_cast<int64_t>(w2.packed.size());
    entry.w1_scales_bytes = static_cast<int64_t>(w1.scales.size());
    entry.w3_scales_bytes = static_cast<int64_t>(w3.scales.size());
    entry.w2_scales_bytes = static_cast<int64_t>(w2.scales.size());

    // Input row.
    std::vector<float> xf(Hd);
    for (int i = 0; i < Hd; ++i) xf[i] = 0.5f * Frand(7, i);
    __nv_bfloat16 *d_x = nullptr;
    {
        std::vector<__nv_bfloat16> tmp(Hd);
        for (int i = 0; i < Hd; ++i) tmp[i] = __float2bfloat16_rn(xf[i]);
        REQUIRE(cudaMalloc(&d_x, Hd * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_x, tmp.data(), Hd * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }
    __nv_bfloat16 *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, Hd * 2) == cudaSuccess);
    REQUIRE(cudaMemset(d_out, 0, Hd * 2) == cudaSuccess);

    const size_t staging_bytes = KimiExpertFfnStagingBytes();
    void *d_staging = nullptr;
    REQUIRE(cudaMalloc(&d_staging, staging_bytes) == cudaSuccess);

    cublasHandle_t cublas;
    REQUIRE(cublasCreate(&cublas) == CUBLAS_STATUS_SUCCESS);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = RunKimiExpertFfn(cublas, stream, entry, /*weight=*/1.0f,
                                       d_x, d_out, d_staging, staging_bytes);
    REQUIRE(err == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__nv_bfloat16> out_h(Hd);
    REQUIRE(cudaMemcpy(out_h.data(), d_out, Hd * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // ── CPU reference ────────────────────────────────────────────────────
    auto decode = [&](const ProjData &p, std::vector<double> &out) {
        out.resize(p.packed.size() * 2);
        static const float tbl[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        for (size_t i = 0; i < out.size(); ++i) {
            uint8_t byte = p.packed[i >> 1];
            int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            out[i] = tbl[nib] * HostE4M3(p.scales[i >> 4]) * p.global_scale;
        }
    };
    std::vector<double> dw1, dw3, dw2;
    decode(w1, dw1);
    decode(w3, dw3);
    decode(w2, dw2);

    std::vector<double> h_act(I);
    for (int k = 0; k < I; ++k) {
        double g = 0, u = 0;
        for (int j = 0; j < Hd; ++j) {
            double xv = __bfloat162float(__float2bfloat16_rn(xf[j]));
            g += dw1[static_cast<size_t>(k) * Hd + j] * xv;
            u += dw3[static_cast<size_t>(k) * Hd + j] * xv;
        }
        h_act[k] = g / (1.0 + std::exp(-g)) * u;
    }
    double max_abs = 0.0, ref_scale = 0.0;
    for (int o = 0; o < Hd; ++o) {
        double acc = 0.0;
        for (int k = 0; k < I; ++k)
            acc += h_act[k] * dw2[static_cast<size_t>(o) * I + k];
        double got = __bfloat162float(out_h[o]);
        ref_scale = std::max(ref_scale, std::abs(acc));
        max_abs = std::max(max_abs, std::abs(got - acc));
    }
    INFO("max abs error: " << max_abs << " ref scale: " << ref_scale);
    CHECK(max_abs <= 0.01 * ref_scale + 1.0);

    cudaDeviceSynchronize();
    cublasDestroy(cublas);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_staging);
    cudaStreamDestroy(stream);
}
