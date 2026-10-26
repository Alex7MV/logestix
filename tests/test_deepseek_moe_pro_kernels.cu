// test_deepseek_moe_pro_kernels.cu — deepseek_pro::gpu::rmsnorm_float (DeepSeek-V4-Pro-0813).
//
// The kernel runs in deepseek_moe_inference_pipeline.cu and
// deepseek_moe_inference_attn.cu, i.e. on every decode token of the model, and
// had no test of its own: the family suite covers the MoE/attention layers but
// never this norm.
//
// The kernel is one block per row with a two-phase warp-shuffle reduction and a
// float4 vectorised body plus a scalar tail. The cases below target exactly
// that structure: hidden_dim divisible by 4 (pure vector path), not divisible
// (the tail), a single column (one active lane per warp), and multi-row
// batches. Every numeric case compares against an independent FP64 host
// reference.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/deepseek_moe_pro_inference_csa.h"
#include "lsxcommon/deepseek_moe_pro_norm_kernels.h"

#include "deepseek_moe_pro_load.cuh"  // LayerWeights

using Catch::Approx;

namespace {

// Independent FP64 reference: out = x * rsqrt(mean(x^2) + eps) * gamma.
std::vector<double> Reference(const std::vector<float> &x,
                              const std::vector<float> &gamma, int batch,
                              int hidden, double eps)
{
    std::vector<double> out(static_cast<size_t>(batch) * hidden);
    for (int r = 0; r < batch; ++r) {
        double sum = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double v = x[static_cast<size_t>(r) * hidden + i];
            sum += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum / hidden + eps);
        for (int i = 0; i < hidden; ++i)
            out[static_cast<size_t>(r) * hidden + i] =
                x[static_cast<size_t>(r) * hidden + i] * inv * gamma[i];
    }
    return out;
}

// Deterministic non-trivial input (no rand(), so a failure reproduces).
std::vector<float> MakeInput(int n, float seed)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>((i * 37) % 101) * 0.017f + seed;
        v[i] = std::sin(t) * 2.5f + std::cos(t * 0.5f);
    }
    return v;
}

struct DeviceBuffers {
    float *in = nullptr;
    float *gamma = nullptr;
    float *out = nullptr;
    ~DeviceBuffers()
    {
        if (in) cudaFree(in);
        if (gamma) cudaFree(gamma);
        if (out) cudaFree(out);
    }
};

DeviceBuffers Upload(const std::vector<float> &a, const std::vector<float> &g)
{
    DeviceBuffers d;
    cudaMalloc(&d.in, a.size() * sizeof(float));
    cudaMalloc(&d.gamma, g.size() * sizeof(float));
    cudaMalloc(&d.out, a.size() * sizeof(float));
    cudaMemcpy(d.in, a.data(), a.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d.gamma, g.data(), g.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d;
}

// Runs the kernel and returns the output, or {} when the launch was rejected.
std::vector<float> Run(int batch, int hidden, const std::vector<float> &x,
                       const std::vector<float> &g, float eps)
{
    DeviceBuffers d = Upload(x, g);
    cudaMemset(d.out, 0xff, x.size() * sizeof(float));
    const cudaError_t rc =
        lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, batch, hidden, d.in, d.gamma,
                                      eps, d.out);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    cudaDeviceSynchronize();
    std::vector<float> out(x.size());
    cudaMemcpy(out.data(), d.out, out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    return out;
}

void Compare(const std::vector<float> &got, const std::vector<double> &want,
             float tol)
{
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i)
        REQUIRE(got[i] == Approx(static_cast<float>(want[i])).margin(tol));
}

} // namespace

TEST_CASE("deepseek norm kernels: rmsnorm_float matches the host reference",
          "[deepseek][pro][kernels]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    // 512 is divisible by 4: the pure float4 path, batch=1 as the pipeline
    // always uses.
    const int batch = 1, hidden = 512;
    const auto x = MakeInput(batch * hidden, 0.25f);
    std::vector<float> g(hidden, 1.0f);
    for (int i = 0; i < hidden; ++i) g[i] = 0.5f + 0.01f * (i % 17);

    const auto got = Run(batch, hidden, x, g, 1e-6f);
    REQUIRE(!got.empty());
    Compare(got, Reference(x, g, batch, hidden, 1e-6), 2e-5f);
}

TEST_CASE("deepseek norm kernels: rmsnorm_float covers every block width the "
          "model uses",
          "[deepseek][pro][kernels]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    // run_rmsnorm (deepseek_moe_inference_pipeline.cu) always calls with
    // batch=1 and the frozen hidden dim (4096, a multiple of 256), so the
    // whole grid is one block and the float4 body never has a tail. Sweep the
    // widths around that to exercise the warp-shuffle reduction with a
    // different number of active warps.
    for (const int hidden : {4, 128, 512, 4096}) {
        CAPTURE(hidden);
        const int batch = 1;
        const auto x = MakeInput(batch * hidden, 0.75f);
        std::vector<float> g(hidden);
        for (int i = 0; i < hidden; ++i) g[i] = 1.0f + 0.002f * (i % 7);

        const auto got = Run(batch, hidden, x, g, 1e-5f);
        REQUIRE(!got.empty());
        Compare(got, Reference(x, g, batch, hidden, 1e-5), 2e-5f);
    }
}

TEST_CASE("deepseek norm kernels: rmsnorm_float honours the documented "
          "hidden_dim >= 1 bound",
          "[deepseek][pro][kernels]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    // The header states hidden_dim must be >= 1. With one column the float4
    // body is empty and only the scalar tail runs, so this pins the degenerate
    // grid shape the contract promises to accept.
    const int batch = 1, hidden = 1;
    std::vector<float> x{1.0f};
    std::vector<float> g{2.0f};
    const auto got = Run(batch, hidden, x, g, 0.0f);
    REQUIRE(!got.empty());
    CHECK(got[0] == Approx(2.0f).margin(1e-5f));
}

TEST_CASE("deepseek norm kernels: rmsnorm_float keeps an all-zero row finite",
          "[deepseek][pro][kernels]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    // mean(x^2) == 0, so inv_rms = rsqrt(eps): the output must be 0 and not
    // NaN/Inf. eps is what keeps this finite, so a dropped eps would show here.
    const int batch = 2, hidden = 64;
    std::vector<float> x(batch * hidden, 0.0f);
    std::vector<float> g(hidden, 1.0f);

    const auto got = Run(batch, hidden, x, g, 1e-6f);
    REQUIRE(!got.empty());
    for (float v : got) {
        CHECK(std::isfinite(v));
        CHECK(v == Approx(0.0f).margin(1e-6f));
    }
}

TEST_CASE("deepseek norm kernels: rmsnorm_float rejects invalid arguments",
          "[deepseek][pro][kernels]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    const int batch = 2, hidden = 16;
    const auto x = MakeInput(batch * hidden, 0.5f);
    DeviceBuffers d = Upload(x, std::vector<float>(hidden, 1.0f));

    // Null pointers.
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, batch, hidden, nullptr,
                                        d.gamma, 1e-6f, d.out) ==
          cudaErrorInvalidValue);
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, batch, hidden, d.in, nullptr,
                                        1e-6f, d.out) == cudaErrorInvalidValue);
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, batch, hidden, d.in, d.gamma,
                                        1e-6f, nullptr) == cudaErrorInvalidValue);
    // Non-positive extents.
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, 0, hidden, d.in, d.gamma,
                                        1e-6f, d.out) == cudaErrorInvalidValue);
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, batch, 0, d.in, d.gamma,
                                        1e-6f, d.out) == cudaErrorInvalidValue);
    CHECK(lsxcommon::deepseek_pro::gpu::rmsnorm_float(nullptr, -1, hidden, d.in, d.gamma,
                                        1e-6f, d.out) == cudaErrorInvalidValue);
}
// ── run_csa_attention: reachable contract checks ──────────────────────────
// The numeric path of run_csa_attention is NOT covered here. It is not a single
// kernel but the whole attention pipeline (q/kv FP8 projections, RMSNorm,
// per-head norm, RoPE, CSA compression, KV cache, sinks), and a test would have
// to re-implement that pipeline independently to be worth anything — copying it
// would only assert that the code equals itself.
//
// What IS reachable and cheap to pin: the argument guards run before any
// dereference, so an incompletely populated LayerWeights must be rejected
// instead of faulting inside a kernel. deepseek_moe_inference_attn.cu marks the
// norm companions as "a nullptr here becomes an in-kernel illegal address",
// which is exactly the failure these guards prevent.
TEST_CASE("deepseek norm kernels: run_csa_attention rejects an empty layer",
          "[deepseek][pro][kernels][csa]")
{
    if (cudaFree(0) != cudaSuccess) SKIP("no CUDA device");

    const lsxcommon::deepseek_pro::gpu::LayerWeights empty{};
    lsxcommon::deepseek_pro::gpu::CSAState state;
    // stream, lw, d_x, d_kv_cache_layer, csa_state, start_pos + the 11
    // scratch/output pointers.
    CHECK(lsxcommon::deepseek_pro::gpu::run_csa_attention(nullptr, empty, nullptr, nullptr,
                                            state, 0, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr, nullptr,
                                            nullptr) == cudaErrorInvalidValue);
}
