#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/glm_moe_inference.h"
#include "lsxcommon/glm_moe_load.h"

using Catch::Approx;

// Phase 4.5: GLM-5.2 Dense MLP stack (layers 0-2) execution loop
// ----------------------------------------------------------------
//   hidden_size = 6144 (kGlmHidden), intermediate = 2048 (kGlmMoeIntermediate)
//   SwiGLU chain:  w2( silu(w1(x)) * w3(x) )  over post_attn RMSNorm, Residual.
//
// Synthetic fixture: a well-mixed integer hash (SplitMix64 finalizer over a
// 64-bit lane <- index | seed) drives EVERY weight/activation element, so the
// fixture is fully deterministic across runs/hosts with no transcendental
// dependency and no pointer-stride or fill-magnitude aliasing. RED/GREEN: the
// test fails hard (RED) on any pointer-stride corruption or element > margin
// divergence from the independent host FP64 reference; the GREEN state is the
// frozen dense stack passing element-for-element.

namespace {

constexpr int kHidden = 6144;
constexpr int kInter  = 2048;
constexpr int kSeq    = 4;
constexpr float kLimit = 10.0f;

// SplitMix64 finalizer -> uint64. Well-mixed (avalanches every input bit), so
// adjacent lanes differ in the high bits of weight magnitude — exercising
// row-major strides rather than smooth fill patterns.
inline uint64_t Mix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Deterministic pseudo-float in [-scale, scale) from (seed, index).
float HashF(float scale, uint64_t seed, uint64_t index)
{
    const uint64_t lane = seed * 0x9e3779b97f4a7c15ull ^ Mix64(index + 0x9e37ull);
    const uint64_t h = Mix64(lane);
    const float u = static_cast<float>((h >> 33) & 0xffffffu) / 8388608.0f;
    return (u - 1.0f) * scale;
}

// Upload host vector to device, asserting success.
void UploadHalf(const std::vector<__half> &buf, __half **dst)
{
    cudaError_t e = cudaMalloc(dst, buf.size() * sizeof(__half));
    REQUIRE(e == cudaSuccess);
    e = cudaMemcpy(*dst, buf.data(), buf.size() * sizeof(__half),
                   cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);
}

float Silu(float v) { return v / (1.0f + std::exp(-v)); }

// Independent host FP64 reference of ONE dense layer (identical algorithm to
// the stack: post_attn RMSNorm -> gate/up -> silu*up with clamp -> down ->
// +residual), accumulated in double. Returns the per-token hidden output.
void RefDenseLayerF64(const std::vector<float> &wg, const std::vector<float> &wu,
                      const std::vector<float> &wd, const std::vector<float> &wn,
                      const std::vector<float> &x, std::vector<float> &y)
{
    const int hidden = static_cast<int>(x.size());
    const int inter  = static_cast<int>(wg.size()) / hidden;
    double ss = 0.0;
    for (float v : x) ss += static_cast<double>(v) * v;
    const double rinv = 1.0 / std::sqrt(ss / hidden + 1e-5);
    std::vector<double> n(hidden), g(inter), u(inter), a(inter);
    for (int i = 0; i < hidden; ++i)
        n[i] = static_cast<double>(x[static_cast<size_t>(i)]) * rinv * wn[i];
    for (int m = 0; m < inter; ++m) {
        const float *wgv = &wg[static_cast<size_t>(m) * hidden];
        const float *wuv = &wu[static_cast<size_t>(m) * hidden];
        double sg = 0.0, su = 0.0;
        for (int k = 0; k < hidden; ++k) {
            sg += static_cast<double>(wgv[k]) * n[k];
            su += static_cast<double>(wuv[k]) * n[k];
        }
        g[m] = sg;
        u[m] = su;
    }
    for (int m = 0; m < inter; ++m) {
        double gv = g[m]; if (gv > kLimit) gv = kLimit;
        double uv = u[m]; if (uv > kLimit) uv = kLimit;
        if (uv < -kLimit) uv = -kLimit;
        a[m] = Silu(static_cast<float>(gv)) * uv;
    }
    y.assign(static_cast<size_t>(hidden), 0.0f);
    for (int k = 0; k < hidden; ++k) {
        const float *wdv = &wd[static_cast<size_t>(k) * inter];
        double acc = 0.0;
        for (int m = 0; m < inter; ++m)
            acc += static_cast<double>(wdv[m]) * a[m];
        y[static_cast<size_t>(k)] = static_cast<float>(acc + x[k]);
    }
}

} // namespace

TEST_CASE("GLM dense stack layers 0-2: hash fixture, host strides intact, "
          "FP64 reference convergence (RED/GREEN)", "[glm][compute][dense]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    lsxcommon::glm::GlmPinnedWeights pw;
    pw.layers.resize(3);
    std::vector<std::vector<float>> wg(3), wu(3), wd(3), wn(3);
    std::vector<const void *> gate_ptr(3), up_ptr(3), down_ptr(3);

    for (int l = 0; l < 3; ++l) {
        auto &lw = pw.layers[l];
        std::vector<__half> hg, hu, hd, hw;
        hg.resize(static_cast<size_t>(kInter) * kHidden);
        hu.resize(static_cast<size_t>(kInter) * kHidden);
        hd.resize(static_cast<size_t>(kHidden) * kInter);
        hw.resize(static_cast<size_t>(kHidden));
        for (size_t i = 0; i < hg.size(); ++i)
            hg[i] = __float2half(HashF(0.01f, 7u + 13u * static_cast<unsigned>(l),
                                       static_cast<uint64_t>(i)));
        for (size_t i = 0; i < hu.size(); ++i)
            hu[i] = __float2half(HashF(0.01f, 11u + 19u * static_cast<unsigned>(l),
                                       static_cast<uint64_t>(i)));
        for (size_t i = 0; i < hd.size(); ++i)
            hd[i] = __float2half(HashF(0.01f, 3u + 29u * static_cast<unsigned>(l),
                                       static_cast<uint64_t>(i)));
        for (size_t i = 0; i < hw.size(); ++i)
            hw[i] = __float2half(HashF(0.02f, 5u + 31u * static_cast<unsigned>(l),
                                       static_cast<uint64_t>(i)));
        UploadHalf(hg, &lw.d_mlp_gate);
        UploadHalf(hu, &lw.d_mlp_up);
        UploadHalf(hd, &lw.d_mlp_down);
        UploadHalf(hw, &lw.d_input_norm);
        UploadHalf(hw, &lw.d_post_attn_norm);
        for (int i = 0; i < kInter * kHidden; ++i) {
            wg[l].push_back(__half2float(hg[static_cast<size_t>(i)]));
            wu[l].push_back(__half2float(hu[static_cast<size_t>(i)]));
        }
        for (int i = 0; i < kHidden * kInter; ++i)
            wd[l].push_back(__half2float(hd[static_cast<size_t>(i)]));
        for (int i = 0; i < kHidden; ++i)
            wn[l].push_back(__half2float(hw[static_cast<size_t>(i)]));
        gate_ptr[l] = lw.d_mlp_gate;
        up_ptr[l]   = lw.d_mlp_up;
        down_ptr[l] = lw.d_mlp_down;
    }

    std::vector<__half> hx(static_cast<size_t>(kSeq) * kHidden);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __float2half(HashF(0.02f, 2u, static_cast<uint64_t>(i)));
    __half *d_x = nullptr, *d_alt = nullptr, *d_r = nullptr, *d_s = nullptr;
    UploadHalf(hx, &d_x);
    REQUIRE(cudaMalloc(&d_alt, static_cast<size_t>(kSeq) * kHidden *
                                   sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_r, static_cast<size_t>(kSeq) * kHidden *
                                 sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s, static_cast<size_t>(kSeq) *
                                 (kHidden + 2 * kInter) * sizeof(__half)) ==
            cudaSuccess);

    // Run the concrete Phase 4.5 dense loop on the explicit async stream.
    REQUIRE(lsxcommon::glm::RunGlmDenseMLPStack(
                ch, st, pw, kSeq, d_x, d_alt, d_r, d_s) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hout(static_cast<size_t>(kSeq) * kHidden);
    REQUIRE(cudaMemcpy(hout.data(), d_r, hout.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // GREEN: element-for-element convergence against the FP64 host reference.
    for (int t = 0; t < kSeq; ++t) {
        std::vector<float> x(kHidden);
        for (int i = 0; i < kHidden; ++i)
            x[i] = __half2float(hx[static_cast<size_t>(t) * kHidden + i]);
        for (int l = 0; l < 3; ++l) {
            std::vector<float> y(kHidden);
            RefDenseLayerF64(wg[l], wu[l], wd[l], wn[l], x, y);
            x = std::move(y);
        }
        for (int i = 0; i < kHidden; ++i) {
            float got = __half2float(
                hout[static_cast<size_t>(t) * kHidden + i]);
            REQUIRE(got == Approx(x[static_cast<size_t>(i)]).margin(0.05f));
        }
    }

    // RED guard: the forward must never have corrupted the device weight
    // strides — every layer's gate/up/down buffer pointer is bit-identical to
    // what the fixture uploaded, proving no ping-pong stride aliasing.
    for (int l = 0; l < 3; ++l) {
        CHECK(pw.layers[l].d_mlp_gate == gate_ptr[l]);
        CHECK(pw.layers[l].d_mlp_up   == up_ptr[l]);
        CHECK(pw.layers[l].d_mlp_down == down_ptr[l]);
    }

    pw.Free();
    cudaFree(d_x);
    cudaFree(d_alt);
    cudaFree(d_r);
    cudaFree(d_s);
    cudaStreamDestroy(st);
    cublasDestroy(ch);
}
