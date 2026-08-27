#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

__constant__ float kE2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

__global__ void test_nvfp4_decode_kernel(const uint8_t *packed, const uint8_t *scales,
    float global_scale, __half *out, int num_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;
    int byte_idx = i >> 1;
    int nibble = (i & 1) ? (packed[byte_idx] >> 4) : (packed[byte_idx] & 0xF);
    int group = i / 16;
    float scale = ((float)scales[group] - 128.0f) * global_scale;
    out[i] = __float2half(kE2M1Table[nibble] * scale);
}

__global__ void test_nvfp4_gemm_kernel(const __half *x, const __half *w,
    __half *y, int M, int K, int N)
{
    int m = blockIdx.x;
    int n = threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(x[m * K + k]) * __half2float(w[n * K + k]);
    y[m * N + n] = __float2half(acc);
}

TEST_CASE("NVFP4 E2M1 decode table roundtrip")
{
    constexpr float kE2M1_expected[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };

    constexpr int N = 64;
    uint8_t h_packed[32];
    uint8_t h_scales[4];
    for (int i = 0; i < 64; ++i) {
        int nibble = (i % 8) + 8;
        int byte_idx = i >> 1;
        if (i & 1) h_packed[byte_idx] = (h_packed[byte_idx] & 0x0F) | (nibble << 4);
        else       h_packed[byte_idx] = (h_packed[byte_idx] & 0xF0) | nibble;
    }
    for (int g = 0; g < 4; ++g) h_scales[g] = 128 + g;

    uint8_t *d_packed, *d_scales;
    __half *d_out;
    cudaMalloc(&d_packed, 32);
    cudaMalloc(&d_scales, 4);
    cudaMalloc(&d_out, N * sizeof(__half));
    cudaMemcpy(d_packed, h_packed, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, 4, cudaMemcpyHostToDevice);

    test_nvfp4_decode_kernel<<<1, 64>>>(d_packed, d_scales, 1.0f, d_out, N);
    cudaDeviceSynchronize();

    std::vector<__half> h_out(N);
    cudaMemcpy(h_out.data(), d_out, N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        int nibble = (i % 8) + 8;
        int group = i / 16;
        float scale = (float)(group);
        float expected = kE2M1_expected[nibble] * scale;
        CHECK(std::abs((float)h_out[i] - expected) < 1e-4f);
    }

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_out);
}

TEST_CASE("NVFP4 GEMM: 4x8x8 multiply")
{
    constexpr int M = 4, K = 8, N = 8;
    float h_x[M * K], h_w[N * K], h_ref[M * N];
    for (int i = 0; i < M * K; ++i) h_x[i] = (i % 3) * 2.0f;
    for (int i = 0; i < N * K; ++i) h_w[i] = (i % 5) * 1.5f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double sum = 0;
            for (int k = 0; k < K; ++k) sum += h_x[m * K + k] * h_w[n * K + k];
            h_ref[m * N + n] = (float)sum;
        }

    __half *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, M * K * sizeof(__half));
    cudaMalloc(&d_w, N * K * sizeof(__half));
    cudaMalloc(&d_y, M * N * sizeof(__half));

    std::vector<__half> h_x_h(M * K), h_w_h(N * K);
    for (int i = 0; i < M * K; ++i) h_x_h[i] = __float2half(h_x[i]);
    for (int i = 0; i < N * K; ++i) h_w_h[i] = __float2half(h_w[i]);
    cudaMemcpy(d_x, h_x_h.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w_h.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice);

    test_nvfp4_gemm_kernel<<<M, N>>>(d_x, d_w, d_y, M, K, N);
    cudaDeviceSynchronize();

    std::vector<__half> h_y(M * N);
    cudaMemcpy(h_y.data(), d_y, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; ++i)
        CHECK(std::abs((float)h_y[i] - h_ref[i]) < 1e-3f);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
}