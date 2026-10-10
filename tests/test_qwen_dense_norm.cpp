#include <catch2/catch_all.hpp>

#include "lsxcommon/lsx_attn_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

using namespace lsxcommon::qwen;

static void rmsnorm_ref_fp64(const std::vector<float> &x, const std::vector<float> &w,
                             std::vector<float> &out, int rows, int hidden, double eps)
{
    for (int i = 0; i < rows; ++i) {
        double sum = 0;
        for (int j = 0; j < hidden; ++j)
            sum += (double)x[i * hidden + j] * x[i * hidden + j];
        double inv = 1.0 / sqrt(sum / hidden + eps);
        for (int j = 0; j < hidden; ++j)
            out[i * hidden + j] = (float)((double)x[i * hidden + j] * inv * (1.0 + (double)w[j]));
    }
}

TEST_CASE("RMSNorm offset form matches CPU FP64")
{
    constexpr int hidden = 256, rows = 8;
    std::vector<float> h_x(rows * hidden), h_w(hidden), h_ref(rows * hidden);
    for (int i = 0; i < rows * hidden; ++i) h_x[i] = (float)((i % 7) * 0.1);
    for (int j = 0; j < hidden; ++j) h_w[j] = (float)((j % 5) * 0.01);
    rmsnorm_ref_fp64(h_x, h_w, h_ref, rows, hidden, 1e-6);

    std::vector<__half> d_half(rows * hidden), w_half(hidden);
    for (size_t i = 0; i < h_x.size(); ++i) d_half[i] = __float2half(h_x[i]);
    for (size_t j = 0; j < h_w.size(); ++j) w_half[j] = __float2half(h_w[j]);

    __half *d_x, *d_w, *d_out;
    REQUIRE(cudaMalloc(&d_x, d_half.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w, sizeof(__half) * hidden) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, sizeof(__half) * rows * hidden) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, d_half.data(), d_half.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w, w_half.data(), sizeof(__half) * hidden,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK(RunAttnRMSNorm(stream, d_x, d_w, d_out, rows, hidden, 1e-6) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> h_out(rows * hidden);
    REQUIRE(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < rows * hidden; ++i)
        CHECK(std::abs((float)h_out[i] - h_ref[i]) < 2e-3f);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    cudaStreamDestroy(stream);
}

TEST_CASE("RMSNorm rejects invalid args")
{
    __half *d = nullptr;
    CHECK(RunAttnRMSNorm(nullptr, nullptr, nullptr, nullptr, 0, 0, 1e-6) == cudaErrorInvalidValue);
    CHECK(d == nullptr);
}