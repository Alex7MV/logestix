// TDD — MiMo non-GEMM kernel parity (RMSNorm / router / SwiGLU) against the
// family host reference. CUDA OBJECT lib, tolerance class for transcendentals.
#include <catch2/catch_test_macros.hpp>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "lsxcommon/mimo_moe_kernels.h"
#include "mimo_moe_device_math.h"

using namespace lsxcommon::mimo;
namespace ref = lsxcommon::mimo::ref;

namespace {

void FillHalf(std::vector<__half> &v, float scale, int seed)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = __float2half(std::sin(static_cast<float>(i + seed)) * scale);
}

bool Close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

} // namespace

TEST_CASE("mimo rmsnorm parity vs host reference", "[mimo][kernels]")
{
    const int rows = 3, hidden = 64;
    const float eps = 1e-6f;
    std::vector<__half> x(rows * hidden), w(hidden), out(rows * hidden);
    FillHalf(x, 1.0f, 1);
    FillHalf(w, 0.5f, 7);

    std::vector<float> xf(x.size()), wf(w.size()), ref_out(x.size());
    for (size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);
    for (size_t i = 0; i < w.size(); ++i) wf[i] = __half2float(w[i]);
    for (int r = 0; r < rows; ++r)
        ref::RmsNorm(xf.data() + r * hidden, wf.data(), hidden, eps,
                     ref_out.data() + r * hidden);

    __half *dx, *dw, *dout;
    REQUIRE(cudaMalloc(&dx, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, w.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, out.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunMimoRmsNorm(nullptr, dx, dw, rows, hidden, eps, dout) == cudaSuccess);
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t i = 0; i < out.size(); ++i)
        REQUIRE(Close(__half2float(out[i]), ref_out[i], 2e-3f));
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

TEST_CASE("mimo router parity vs host reference", "[mimo][kernels]")
{
    const int rows = 2, hidden = 64, experts = 8, top_k = 3;
    std::vector<__half> x(rows * hidden);
    FillHalf(x, 1.0f, 3);
    std::vector<float> gw_f(experts * hidden);
    for (size_t i = 0; i < gw_f.size(); ++i)
        gw_f[i] = std::cos(static_cast<float>(i)) * 0.3f;
    std::vector<__nv_bfloat16> gw(gw_f.size());
    for (size_t i = 0; i < gw.size(); ++i) gw[i] = __float2bfloat16(gw_f[i]);
    std::vector<float> bias(experts);
    for (int e = 0; e < experts; ++e) bias[e] = 0.1f * e;

    // Host reference (uses bf16-rounded weights to match the kernel).
    std::vector<float> gw_rounded(gw.size());
    for (size_t i = 0; i < gw.size(); ++i) gw_rounded[i] = __bfloat162float(gw[i]);
    std::vector<float> xf(x.size());
    for (size_t i = 0; i < x.size(); ++i) xf[i] = __half2float(x[i]);

    std::vector<int> idx; std::vector<float> w;
    std::vector<int> all_idx(rows * top_k); std::vector<float> all_w(rows * top_k);
    __half *dx; __nv_bfloat16 *dgw; float *dbias; int *didx; float *dw_out;
    REQUIRE(cudaMalloc(&dx, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgw, gw.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dbias, bias.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&didx, rows * top_k * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw_out, rows * top_k * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dgw, gw.data(), gw.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dbias, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunMimoRouter(nullptr, dx, dgw, dbias, rows, hidden, experts, top_k,
                          didx, dw_out) == cudaSuccess);
    REQUIRE(cudaMemcpy(all_idx.data(), didx, rows * top_k * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(all_w.data(), dw_out, rows * top_k * 4, cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r)
    {
        ref::RouterTopK(xf.data() + r * hidden, gw_rounded.data(), bias.data(),
                        hidden, experts, top_k, idx, w);
        bool same_set = true;
        for (int t = 0; t < top_k; ++t)
        {
            bool found = false;
            for (int u = 0; u < top_k; ++u)
                found = found || (all_idx[r * top_k + u] == idx[t]);
            same_set = same_set && found;
        }
        REQUIRE(same_set);
        for (int t = 0; t < top_k; ++t)
        {
            // Match the same expert id and compare the weight.
            for (int u = 0; u < top_k; ++u)
                if (all_idx[r * top_k + u] == idx[t])
                    REQUIRE(Close(all_w[r * top_k + u], w[t], 2e-3f));
        }
    }
    cudaFree(dx); cudaFree(dgw); cudaFree(dbias); cudaFree(didx); cudaFree(dw_out);
}

TEST_CASE("mimo swiglu parity vs host reference", "[mimo][kernels]")
{
    const int64_t n = 128;
    std::vector<__half> g(n), u(n), out(n);
    FillHalf(g, 1.0f, 11);
    FillHalf(u, 2.0f, 13);
    std::vector<float> gf(n), uf(n), rf(n);
    for (int64_t i = 0; i < n; ++i) { gf[i] = __half2float(g[i]); uf[i] = __half2float(u[i]); }
    ref::SwiGLU(gf.data(), uf.data(), n, rf.data());

    __half *dg, *du, *dout;
    REQUIRE(cudaMalloc(&dg, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, n * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dg, g.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(du, u.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunMimoSwiGLU(nullptr, dg, du, n, dout) == cudaSuccess);
    REQUIRE(cudaMemcpy(out.data(), dout, n * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int64_t i = 0; i < n; ++i)
        REQUIRE(Close(__half2float(out[i]), rf[i], 3e-3f));
    cudaFree(dg); cudaFree(du); cudaFree(dout);
}
