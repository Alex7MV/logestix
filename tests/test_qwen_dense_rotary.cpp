#include <catch2/catch_all.hpp>

#include "lsxcommon/lsx_attn_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

using namespace lsxcommon::qwen;

TEST_CASE("Partial rotary (64 of 256 dims) matches CPU reference")
{
    constexpr int seq = 4, heads = 2, head_dim = 256, rotary_dim = 64;
    constexpr double theta = 10000.0;

    std::vector<int32_t> pos = {0, 1, 2, 3};
    std::vector<__half> h_q(seq * heads * head_dim), h_k(seq * heads * head_dim);
    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = __float2half((float)(i % 13) * 0.3f);
    for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = __float2half((float)((i % 11) * 0.2f - 1.0f));

    // CPU reference
    std::vector<float> qc(seq * heads * head_dim), kc(seq * heads * head_dim);
    std::vector<std::pair<float,float>> ct(seq * (rotary_dim/2));
    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < rotary_dim / 2; ++d) {
            double ang = pos[t] / std::pow(theta, (double)(2*d)/(double)rotary_dim);
            ct[t * (rotary_dim/2) + d] = {
                (float)__float2half((float)std::cos(ang)),
                (float)__float2half((float)std::sin(ang))};
        }
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            // HF rotate_half pairing: (j, j + rotary_dim/2) share cos/sin[j].
            for (int j = 0; j < rotary_dim / 2; ++j) {
                int idx = (t * heads + h) * head_dim;
                auto [c, s] = ct[t * (rotary_dim/2) + j];
                float q1 = (float)h_q[idx + j], q2 = (float)h_q[idx + j + rotary_dim/2];
                float k1 = (float)h_k[idx + j], k2 = (float)h_k[idx + j + rotary_dim/2];
                qc[idx + j]                = (float)__float2half(q1*c - q2*s);
                qc[idx + j + rotary_dim/2] = (float)__float2half(q2*c + q1*s);
                kc[idx + j]                = (float)__float2half(k1*c - k2*s);
                kc[idx + j + rotary_dim/2] = (float)__float2half(k2*c + k1*s);
            }
            // pass-through
            for (int d = rotary_dim; d < head_dim; ++d) {
                int idx = (t * heads + h) * head_dim + d;
                qc[idx] = (float)h_q[idx];
                kc[idx] = (float)h_k[idx];
            }
        }

    __half *d_q, *d_k, *d_cos, *d_sin, *d_qo, *d_ko;
    int32_t *d_pos;
    REQUIRE(cudaMalloc(&d_pos, seq * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_q, h_q.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_k, h_k.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_cos, seq * rotary_dim * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sin, seq * rotary_dim * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_qo, h_q.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ko, h_k.size() * sizeof(__half)) == cudaSuccess);
    cudaMemcpy(d_pos, pos.data(), seq * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CHECK(RunAttnRopeCosSin(stream, d_pos, seq, rotary_dim, theta, d_cos, d_sin) == cudaSuccess);
    CHECK(RunAttnPartialRope(stream, d_q, d_k, d_cos, d_sin, d_qo, d_ko,
                             seq, heads, head_dim, rotary_dim) == cudaSuccess);
    cudaStreamSynchronize(stream);

    std::vector<__half> h_qo(h_q.size()), h_ko(h_k.size());
    cudaMemcpy(h_qo.data(), d_qo, h_qo.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ko.data(), d_ko, h_ko.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h_qo.size(); ++i) {
        CHECK(std::abs((float)h_qo[i] - qc[i]) < 2e-3f);
        CHECK(std::abs((float)h_ko[i] - kc[i]) < 2e-3f);
    }

    cudaFree(d_pos); cudaFree(d_q); cudaFree(d_k); cudaFree(d_cos); cudaFree(d_sin);
    cudaFree(d_qo); cudaFree(d_ko);
    cudaStreamDestroy(stream);
}