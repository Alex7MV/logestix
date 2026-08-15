#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

__global__ void test_causal_softmax_attn_kernel(
    const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int heads, int head_dim, float scale)
{
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (t >= seq || h >= heads || d >= head_dim) return;

    float acc = 0.0f, denom = 0.0f;
    for (int s = 0; s <= t; ++s) {
        float score = 0.0f;
        for (int dd = 0; dd < head_dim; ++dd) {
            float qv = __half2float(q[((t * heads + h) * head_dim) + dd]);
            float kv = __half2float(k[((s * heads + h) * head_dim) + dd]);
            score += qv * kv;
        }
        score *= scale;
        float exp_s = expf(score);
        denom += exp_s;
        acc += exp_s * __half2float(v[((s * heads + h) * head_dim) + d]);
    }
    out[((t * heads + h) * head_dim) + d] = __float2half(acc / (denom + 1e-10f));
}

void attn_ref_fp64(const float *q, const float *k, const float *v,
    float *out, int seq, int heads, int head_dim, float scale)
{
    for (int t = 0; t < seq; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0, denom = 0.0;
                for (int s = 0; s <= t; ++s) {
                    double score = 0.0;
                    for (int dd = 0; dd < head_dim; ++dd)
                        score += q[(t * heads + h) * head_dim + dd] *
                                 k[(s * heads + h) * head_dim + dd];
                    score *= scale;
                    double exp_s = exp(score);
                    denom += exp_s;
                    acc += exp_s * v[(s * heads + h) * head_dim + d];
                }
                out[(t * heads + h) * head_dim + d] = (float)(acc / (denom + 1e-10));
            }
        }
    }
}

TEST_CASE("Causal softmax attention matches CPU FP64")
{
    constexpr int seq = 4, heads = 2, head_dim = 8;
    float scale = 1.0f / sqrtf((float)head_dim);
    int num_elements = seq * heads * head_dim;

    std::vector<float> h_q(num_elements), h_k(num_elements), h_v(num_elements), h_ref(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        h_q[i] = (i * 7 + 1) % 13 * 0.2f - 1.2f;
        h_k[i] = (i * 11 + 3) % 17 * 0.3f - 1.5f;
        h_v[i] = (i * 5 + 2) % 11 * 0.4f - 0.8f;
    }
    attn_ref_fp64(h_q.data(), h_k.data(), h_v.data(), h_ref.data(), seq, heads, head_dim, scale);

    __half *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, num_elements * sizeof(__half));
    cudaMalloc(&d_k, num_elements * sizeof(__half));
    cudaMalloc(&d_v, num_elements * sizeof(__half));
    cudaMalloc(&d_out, num_elements * sizeof(__half));

    std::vector<__half> h_q_h(num_elements), h_k_h(num_elements), h_v_h(num_elements);
    for (int i = 0; i < num_elements; ++i) {
        h_q_h[i] = __float2half(h_q[i]);
        h_k_h[i] = __float2half(h_k[i]);
        h_v_h[i] = __float2half(h_v[i]);
    }
    cudaMemcpy(d_q, h_q_h.data(), num_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k_h.data(), num_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v_h.data(), num_elements * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 grid(seq, heads);
    test_causal_softmax_attn_kernel<<<grid, head_dim>>>(d_q, d_k, d_v, d_out, seq, heads, head_dim, scale);
    cudaDeviceSynchronize();

    std::vector<__half> h_out(num_elements);
    cudaMemcpy(h_out.data(), d_out, num_elements * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elements; ++i) {
        float diff = std::abs((float)h_out[i] - h_ref[i]);
        CHECK(diff < 1e-1f);
    }

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
}