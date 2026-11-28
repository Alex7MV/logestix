#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"
#include <catch2/catch_approx.hpp>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/glm_moe_inference.h"
#include "lsxcommon/glm_moe_load.h"

using Catch::Approx;

namespace {

constexpr int kHidden = 6144;
constexpr int kInter  = 2048;
constexpr int kSeq    = 4;
constexpr float kLimit = 10.0f;

float Silu(float v) { return v / (1.0f + std::exp(-v)); }

void FillHalf(std::vector<__half> &h, int n, float scale, int seed)
{
    h.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float v = std::sin(0.001f * i + static_cast<float>(seed)) * scale;
        h[static_cast<size_t>(i)] = __float2half(v);
    }
}

void UploadHalf(const std::vector<__half> &buf, __half **dst)
{
    cudaError_t e = cudaMalloc(dst, buf.size() * sizeof(__half));
    REQUIRE(e == cudaSuccess);
    e = cudaMemcpy(*dst, buf.data(), buf.size() * sizeof(__half),
                   cudaMemcpyHostToDevice);
    REQUIRE(e == cudaSuccess);
}

void RefDenseLayer(const std::vector<float> &wg, const std::vector<float> &wu,
                   const std::vector<float> &wd, const std::vector<float> &wn,
                   const std::vector<float> &x, std::vector<float> &y)
{
    const int hidden = static_cast<int>(x.size());
    const int inter  = static_cast<int>(wg.size()) / hidden;
    double ss = 0.0;
    for (float v : x) ss += static_cast<double>(v) * v;
    const float rinv = 1.0f / std::sqrt(static_cast<float>(ss / hidden) + 1e-5f);
    std::vector<float> n(hidden), g(inter), u(inter), a(inter);
    for (int i = 0; i < hidden; ++i)
        n[i] = x[i] * rinv * wn[i];
    for (int m = 0; m < inter; ++m) {
        const float *wgv = &wg[static_cast<size_t>(m) * hidden];
        const float *wuv = &wu[static_cast<size_t>(m) * hidden];
        float sg = 0.0f, su = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            sg += wgv[k] * n[k];
            su += wuv[k] * n[k];
        }
        g[m] = sg;
        u[m] = su;
    }
    for (int m = 0; m < inter; ++m) {
        float gv = g[m]; if (gv > kLimit) gv = kLimit;
        float uv = u[m]; if (uv > kLimit) uv = kLimit;
                          if (uv < -kLimit) uv = -kLimit;
        a[m] = Silu(gv) * uv;
    }
    y.assign(static_cast<size_t>(hidden), 0.0f);
    for (int k = 0; k < hidden; ++k) {
        const float *wdv = &wd[static_cast<size_t>(k) * inter];
        float acc = 0.0f;
        for (int m = 0; m < inter; ++m) acc += wdv[m] * a[m];
        y[k] = acc + x[k];
    }
}

// FP64 host reference for the real-model dense stack: dense layers consume
// x through post_attention_layernorm only (attention branch is branched out),
// then gate_proj -> silu*up_proj -> down_proj -> residual. Exact same
// algorithm as RefDenseLayer, accumulated in double.
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
        n[i] = static_cast<double>(x[i]) * rinv * wn[i];
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
        for (int m = 0; m < inter; ++m) acc += static_cast<double>(wdv[m]) * a[m];
        y[static_cast<size_t>(k)] = static_cast<float>(acc + x[k]);
    }
}

} // namespace

TEST_CASE("GLM dense MLP layers 0-2: pure FP16 matmul, host strides intact",
          "[glm][compute]")
{
    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    lsxcommon::glm::GlmPinnedWeights pw;
    pw.layers.resize(3);
    std::vector<std::vector<float>> wg(3), wu(3), wd(3), wn(3);
    std::vector<const void *> gate_ptr(3), up_ptr(3), down_ptr(3);

    for (int l = 0; l < 3; ++l) {
        auto &lw = pw.layers[l];
        std::vector<__half> hg, hu, hd, hw;
        FillHalf(hg, kInter * kHidden, 0.01f, 7 + 13 * l);
        FillHalf(hu, kInter * kHidden, 0.01f, 11 + 19 * l);
        FillHalf(hd, kHidden * kInter, 0.01f, 3 + 29 * l);
        FillHalf(hw, kHidden, 0.02f, 5 + 31 * l);
        UploadHalf(hg, &lw.d_mlp_gate);
        UploadHalf(hu, &lw.d_mlp_up);
        UploadHalf(hd, &lw.d_mlp_down);
        UploadHalf(hw, &lw.d_input_norm);
        UploadHalf(hw, &lw.d_post_attn_norm);
        for (int i = 0; i < kInter * kHidden; ++i) {
            wg[l].push_back(__half2float(hg[i]));
            wu[l].push_back(__half2float(hu[i]));
        }
        for (int i = 0; i < kHidden * kInter; ++i)
            wd[l].push_back(__half2float(hd[i]));
        for (int i = 0; i < kHidden; ++i)
            wn[l].push_back(__half2float(hw[i]));
        gate_ptr[l] = lw.d_mlp_gate;
        up_ptr[l]   = lw.d_mlp_up;
        down_ptr[l] = lw.d_mlp_down;
    }

    std::vector<__half> hx;
    FillHalf(hx, kSeq * kHidden, 0.02f, 2);
    __half *d_x = nullptr, *d_alt = nullptr, *d_r = nullptr, *d_s = nullptr;
    UploadHalf(hx, &d_x);
    REQUIRE(cudaMalloc(&d_alt, kSeq * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_r,   kSeq * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s,   kSeq * (kHidden + 2 * kInter) * sizeof(__half))
            == cudaSuccess);

    REQUIRE(lsxcommon::glm::RunGlmDenseMLPStack(
                st, pw, kSeq, d_x, d_alt, d_r, d_s) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hout(kSeq * kHidden);
    REQUIRE(cudaMemcpy(hout.data(), d_r, hout.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int t = 0; t < kSeq; ++t) {
        std::vector<float> x(kHidden);
        for (int i = 0; i < kHidden; ++i)
            x[i] = __half2float(hx[static_cast<size_t>(t) * kHidden + i]);
        for (int l = 0; l < 3; ++l) {
            std::vector<float> y(kHidden);
            RefDenseLayer(wg[l], wu[l], wd[l], wn[l], x, y);
            x = std::move(y);
        }
        for (int i = 0; i < kHidden; ++i) {
            float got = __half2float(hout[static_cast<size_t>(t) * kHidden + i]);
            REQUIRE(got == Approx(x[i]).margin(0.05f));
        }
    }

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
}

TEST_CASE("GLM real dense stack layers 0-2: loader filter, finite, FP64 ref",
          "[glm][compute][real-model]")
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    std::filesystem::path model =
        env ? std::filesystem::path(env)
            : manifest_query::ConvertedDirFor("glm-5.2");
    if (!std::filesystem::exists(model / "pinned"))
        SKIP(std::string("GLM real model directory not present: ") + model.string());

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    lsxcommon::glm::GlmPinnedWeights pw;
    std::vector<int> dense_only{0, 1, 2};
    REQUIRE(lsxcommon::glm::LoadGlmPinnedWeights(model.string(), pw, st,
                                                 &dense_only) == cudaSuccess);
    REQUIRE(pw.hidden == kHidden);
    REQUIRE(pw.layers.size() == 79);

    std::vector<std::vector<__half>> wg(3), wu(3), wd(3), wn(3);
    for (int l = 0; l < 3; ++l) {
        auto &lw = pw.layers[l];
        REQUIRE(lw.d_mlp_gate != nullptr);
        REQUIRE(lw.d_mlp_up != nullptr);
        REQUIRE(lw.d_mlp_down != nullptr);
        REQUIRE(lw.d_post_attn_norm != nullptr);
        wg[l].assign(static_cast<size_t>(kInter) * kHidden, __half{});
        wu[l].assign(static_cast<size_t>(kInter) * kHidden, __half{});
        wd[l].assign(static_cast<size_t>(kHidden) * kInter, __half{});
        wn[l].assign(static_cast<size_t>(kHidden), __half{});
        REQUIRE(cudaMemcpy(wg[l].data(), lw.d_mlp_gate,
                           wg[l].size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(wu[l].data(), lw.d_mlp_up,
                           wu[l].size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(wd[l].data(), lw.d_mlp_down,
                           wd[l].size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(wn[l].data(), lw.d_post_attn_norm,
                           wn[l].size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    }

    std::vector<__half> hx;
    FillHalf(hx, kSeq * kHidden, 0.02f, 2);
    __half *d_x = nullptr, *d_alt = nullptr, *d_r = nullptr, *d_s = nullptr;
    UploadHalf(hx, &d_x);
    REQUIRE(cudaMalloc(&d_alt, kSeq * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_r,   kSeq * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s,   kSeq * (kHidden + 2 * kInter) * sizeof(__half))
            == cudaSuccess);

    REQUIRE(lsxcommon::glm::RunGlmDenseMLPStack(
                st, pw, kSeq, d_x, d_alt, d_r, d_s) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hout(kSeq * kHidden);
    REQUIRE(cudaMemcpy(hout.data(), d_r, hout.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    auto tof = [](const std::vector<__half> &h) {
        std::vector<float> v(h.size());
        for (size_t i = 0; i < h.size(); ++i)
            v[i] = __half2float(h[i]);
        return v;
    };
    std::vector<std::vector<float>> fwg(3), fwu(3), fwd(3), fwn(3);
    for (int l = 0; l < 3; ++l) {
        fwg[l] = tof(wg[l]);
        fwu[l] = tof(wu[l]);
        fwd[l] = tof(wd[l]);
        fwn[l] = tof(wn[l]);
    }

    for (int t = 0; t < kSeq; ++t) {
        std::vector<float> x(kHidden);
        for (int i = 0; i < kHidden; ++i)
            x[i] = __half2float(hx[static_cast<size_t>(t) * kHidden + i]);
        for (int l = 0; l < 3; ++l) {
            std::vector<float> y(kHidden);
            RefDenseLayerF64(fwg[l], fwu[l], fwd[l], fwn[l], x, y);
            x = std::move(y);
        }
        for (int i = 0; i < kHidden; ++i) {
            float got = __half2float(hout[static_cast<size_t>(t) * kHidden + i]);
            REQUIRE(std::isfinite(got));
            REQUIRE(got == Approx(x[i]).margin(0.10f));
        }
    }

    pw.Free();
    cudaFree(d_x);
    cudaFree(d_alt);
    cudaFree(d_r);
    cudaFree(d_s);
    cudaStreamDestroy(st);
}