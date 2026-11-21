// Neutral AttnRes depth-mixer kernel tests (dedup phase 3, Task 2).
//
// The kernel is shared by alice (MaxSrc=64) and kimi-k3 (MaxSrc=16); the
// family launchers keep their own n_src guards. This test pins the neutral
// contract against a double-precision CPU reference: per-source RMS-dot
// score, softmax over sources, weighted combine. All arithmetic in the
// kernel uses explicit *_rn intrinsics, so the result must be identical
// under any -fmad setting; this target is FMAD_OFF like the family TUs.
#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsx_kernels_attnres.cuh"

#include <cmath>
#include <vector>

namespace {

__half H(float v) { return __float2half(v); }
float F(__half v) { return __half2float(v); }

// Double-precision reference of the depth mixer (formula pinned by the
// alice/kimi-k3 host-reference kernel tests): score_s = dot(rmsnorm(src_s)
// * norm_w, proj_w); softmax over scores; out = sum_s softmax_s * src_s.
std::vector<double> Reference(const std::vector<__half> &src, int n_src,
                              const std::vector<__half> &nw,
                              const std::vector<__half> &pw, int hidden,
                              float eps)
{
    std::vector<double> score(n_src, 0.0);
    for (int s = 0; s < n_src; ++s) {
        double sq = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double v = F(src[s * hidden + i]);
            sq += v * v;
        }
        const double rstd = 1.0 / std::sqrt(sq / hidden + eps);
        double dot = 0.0;
        for (int i = 0; i < hidden; ++i)
            dot += F(src[s * hidden + i]) * rstd * F(nw[i]) * F(pw[i]);
        score[s] = dot;
    }
    double mx = score[0];
    for (int s = 1; s < n_src; ++s) mx = std::max(mx, score[s]);
    double z = 0.0;
    for (int s = 0; s < n_src; ++s) z += std::exp(score[s] - mx);
    std::vector<double> out(hidden, 0.0);
    for (int i = 0; i < hidden; ++i) {
        double acc = 0.0;
        for (int s = 0; s < n_src; ++s)
            acc += std::exp(score[s] - mx) / z * F(src[s * hidden + i]);
        out[i] = acc;
    }
    return out;
}

template <int MaxSrc>
void RunCase(int n_src, int hidden, float eps)
{
    INFO("MaxSrc=" << MaxSrc << " n_src=" << n_src << " hidden=" << hidden);
    std::vector<__half> src(static_cast<size_t>(n_src) * hidden), nw(hidden),
        pw(hidden), out(hidden);
    for (int s = 0; s < n_src; ++s)
        for (int i = 0; i < hidden; ++i)
            src[static_cast<size_t>(s) * hidden + i] =
                H(0.1f * static_cast<float>((s + 1) * (i % 7) - 3));
    for (int i = 0; i < hidden; ++i) {
        nw[i] = H(0.5f + 0.01f * (i % 11));
        pw[i] = H(0.03f * static_cast<float>((i % 9) - 4));
    }
    __half *d_src = nullptr, *d_nw = nullptr, *d_pw = nullptr, *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_src, src.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_nw, hidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_pw, hidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, hidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_src, src.data(), src.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_nw, nw.data(), hidden * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_pw, pw.data(), hidden * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    lsxcommon::kernels::AttnResDepthMixKernel<MaxSrc>
        <<<1, 256>>>(d_src, n_src, d_nw, d_pw, d_out, hidden, eps);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(out.data(), d_out, hidden * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_src);
    cudaFree(d_nw);
    cudaFree(d_pw);
    cudaFree(d_out);

    const auto ref = Reference(src, n_src, nw, pw, hidden, eps);
    for (int i = 0; i < hidden; ++i)
        REQUIRE(std::fabs(F(out[i]) - ref[i]) < 6e-3);
}

} // namespace

TEST_CASE("attnres depth mix <64> matches the CPU reference", "[attnres]")
{
    RunCase<64>(1, 8, 1e-6f);
    RunCase<64>(3, 64, 1e-6f);
    RunCase<64>(5, 2048, 1e-5f);
    RunCase<64>(64, 64, 1e-6f); // n_src at the alice cap
}

TEST_CASE("attnres depth mix <16> matches the CPU reference", "[attnres]")
{
    RunCase<16>(1, 8, 1e-5f);
    RunCase<16>(3, 32, 1e-5f);
    RunCase<16>(16, 64, 1e-6f); // n_src at the kimi-k3 cap
}