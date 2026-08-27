#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

__global__ void test_swiglu_kernel(const __half *a, const __half *b, __half *y, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = __half2float(a[i]);
    float fb = __half2float(b[i]);
    float sig = 1.0f / (1.0f + expf(-fa));
    y[i] = __float2half(fa * sig * fb);
}

TEST_CASE("SwiGLU elementwise matches CPU FP64")
{
    constexpr int64_t N = 4096;
    std::vector<float> h_a(N), h_b(N), h_ref(N);
    for (int64_t i = 0; i < N; ++i) {
        h_a[i] = (i % 7) * 0.5f - 1.5f;
        h_b[i] = (i % 11) * 0.3f + 0.1f;
        double sig = 1.0 / (1.0 + exp(-(double)h_a[i]));
        h_ref[i] = (float)((double)h_a[i] * sig * (double)h_b[i]);
    }

    __half *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(__half));
    cudaMalloc(&d_b, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));

    std::vector<__half> h_a_h(N), h_b_h(N);
    for (int64_t i = 0; i < N; ++i) { h_a_h[i] = __float2half(h_a[i]); h_b_h[i] = __float2half(h_b[i]); }
    cudaMemcpy(d_a, h_a_h.data(), N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_h.data(), N * sizeof(__half), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    test_swiglu_kernel<<<blocks, threads>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    std::vector<__half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < N; ++i)
        REQUIRE(std::abs((float)h_y[i] - h_ref[i]) < 2e-3f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}

TEST_CASE("SwiGLU zero input produces zero output")
{
    constexpr int64_t N = 1024;
    __half *d_a, *d_b, *d_y;
    cudaMalloc(&d_a, N * sizeof(__half));
    cudaMalloc(&d_b, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));
    cudaMemset(d_a, 0, N * sizeof(__half));
    cudaMemset(d_b, 0, N * sizeof(__half));

    test_swiglu_kernel<<<4, 256>>>(d_a, d_b, d_y, N);
    cudaDeviceSynchronize();

    std::vector<__half> h_y(N);
    cudaMemcpy(h_y.data(), d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (auto v : h_y) CHECK(v == __float2half(0.0f));

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_y);
}