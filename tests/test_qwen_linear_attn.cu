#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

// Gated delta recurrence: state[t] = state[t-1] * exp(-g[t]) + beta[t] * kv
// where g[t], beta[t] are scalars, kv is a scalar product
__global__ void test_gated_delta_recurrence_kernel(
    const float *g, const float *beta,
    const float *qk, const float *v,
    float *out, int seq)
{
    // Simplified: one thread handles all positions, scalar state
    // state_{t} = state_{t-1} * exp(-g_t) + beta_t * qk_t * v_t
    float state = 0.0f;
    for (int t = 0; t <= blockIdx.x; ++t) {
        float decay = expf(-g[t]);
        state = state * decay + beta[t] * qk[t] * v[t];
    }
    out[blockIdx.x] = state;
}

void gated_delta_ref_fp64(const float *g, const float *beta,
    const float *qk, const float *v, float *out, int seq)
{
    double state = 0.0;
    for (int t = 0; t < seq; ++t) {
        double decay = exp(-(double)g[t]);
        state = state * decay + (double)beta[t] * (double)qk[t] * (double)v[t];
        out[t] = (float)state;
    }
}

TEST_CASE("Gated delta recurrence matches CPU FP64")
{
    constexpr int seq = 8;
    float h_g[seq], h_beta[seq], h_qk[seq], h_v[seq], h_ref[seq];
    for (int t = 0; t < seq; ++t) {
        h_g[t] = (t + 1) * 0.1f;          // positive gates -> decay
        h_beta[t] = 1.0f / (1.0f + expf(-(float)(t - 3))); // sigmoid
        h_qk[t] = (t % 3) * 0.5f;
        h_v[t] = (t % 5) * 0.3f + 0.1f;
    }
    gated_delta_ref_fp64(h_g, h_beta, h_qk, h_v, h_ref, seq);

    float *d_g, *d_beta, *d_qk, *d_v, *d_out;
    cudaMalloc(&d_g, seq * sizeof(float));
    cudaMalloc(&d_beta, seq * sizeof(float));
    cudaMalloc(&d_qk, seq * sizeof(float));
    cudaMalloc(&d_v, seq * sizeof(float));
    cudaMalloc(&d_out, seq * sizeof(float));
    cudaMemcpy(d_g, h_g, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qk, h_qk, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, seq * sizeof(float), cudaMemcpyHostToDevice);

    test_gated_delta_recurrence_kernel<<<seq, 1>>>(d_g, d_beta, d_qk, d_v, d_out, seq);
    cudaDeviceSynchronize();

    std::vector<float> h_out(seq);
    cudaMemcpy(h_out.data(), d_out, seq * sizeof(float), cudaMemcpyDeviceToHost);
    for (int t = 0; t < seq; ++t)
        CHECK(std::abs(h_out[t] - h_ref[t]) < 1e-5f);

    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_qk); cudaFree(d_v); cudaFree(d_out);
}

TEST_CASE("Gated delta with zero gate produces no decay")
{
    constexpr int seq = 4;
    float h_g[seq] = {0,0,0,0};
    float h_beta[seq] = {1,1,1,1};
    float h_qk[seq] = {2,2,2,2};
    float h_v[seq] = {3,3,3,3};
    float h_ref[seq];
    gated_delta_ref_fp64(h_g, h_beta, h_qk, h_v, h_ref, seq);

    float *d_g, *d_beta, *d_qk, *d_v, *d_out;
    cudaMalloc(&d_g, seq * sizeof(float));
    cudaMalloc(&d_beta, seq * sizeof(float));
    cudaMalloc(&d_qk, seq * sizeof(float));
    cudaMalloc(&d_v, seq * sizeof(float));
    cudaMalloc(&d_out, seq * sizeof(float));
    cudaMemcpy(d_g, h_g, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qk, h_qk, seq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, seq * sizeof(float), cudaMemcpyHostToDevice);

    test_gated_delta_recurrence_kernel<<<seq, 1>>>(d_g, d_beta, d_qk, d_v, d_out, seq);
    cudaDeviceSynchronize();

    std::vector<float> h_out(seq);
    cudaMemcpy(h_out.data(), d_out, seq * sizeof(float), cudaMemcpyDeviceToHost);
    for (int t = 0; t < seq; ++t)
        CHECK(std::abs(h_out[t] - h_ref[t]) < 1e-6f);

    cudaFree(d_g); cudaFree(d_beta); cudaFree(d_qk); cudaFree(d_v); cudaFree(d_out);
}