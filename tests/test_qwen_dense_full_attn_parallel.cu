// Production full-attention (causal GQA) parallel-softmax parity harness.
//
// Direction A (Blackwell inference optimization): the production
// full_attn_softmax_kernel is reworked from thread-0-only scores to a
// block-wide chunked parallel softmax. These tests verify that the public
// enrollment RunQwenFullAttnKernel matches an exact FP64 host reference for a
// small case (kv_base=0) and a long-context incremental-decode case
// (kv_base=2000). Numeric contract: FP32 accumulate, tolerance envelope.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"

namespace {

// Exact FP64 reference: out[t][h][d] = sum_{s<=ctx} exp((q·k_s)*scale) * v_s[d]
//                       / (sum_{s<=ctx} exp((q·k_s)*scale) + 1e-10).
// kv_base offsets the k/v row base (incremental-decode persistent cache);
// s ranges [0, kv_base+t] (causal within the global attended rows).
void AttnRefFp64(const __half *q, const __half *k, const __half *v,
                 __half *out, int seq, int heads, int head_dim,
                 int kv_heads, float scale, int kv_base)
{
    const int kvh_ratio = (heads + kv_heads - 1) / kv_heads;
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const int kvh = h / kvh_ratio;
            const int ctx = kv_base + t;
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0, denom = 0.0;
                for (int s = 0; s <= ctx; ++s) {
                    double score = 0.0;
                    const __half *qr = q + ((int64_t)t * heads + h) * head_dim;
                    const __half *kr = k + ((int64_t)s * kv_heads + kvh) * head_dim;
                    for (int dd = 0; dd < head_dim; ++dd)
                        score += __half2float(qr[dd]) * __half2float(kr[dd]);
                    score *= scale;
                    const double ex = std::exp(score);
                    denom += ex;
                    acc += ex * __half2float(
                        v[((int64_t)s * kv_heads + kvh) * head_dim + d]);
                }
                out[((int64_t)t * heads + h) * head_dim + d] =
                    __float2half((float)(acc / (denom + 1e-10)));
            }
        }
}

}  // namespace

TEST_CASE("RunQwenFullAttnKernel small case matches FP64 ref", "[qwen][attn][full]")
{
    constexpr int seq = 4, heads = 2, kv_heads = 1, head_dim = 8, kv_base = 0;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int nq = seq * heads * head_dim;
    const int nkv = (seq + kv_base) * kv_heads * head_dim;

    std::vector<__half> hq(nq), hk(nkv), href(nq);
    std::vector<__half> hv(nkv);
    for (int i = 0; i < nq; ++i) hq[i] = __float2half((float)((i * 7 + 1) % 13) * 0.2f - 1.2f);
    for (int i = 0; i < nkv; ++i) hk[i] = __float2half((float)((i * 11 + 3) % 17) * 0.3f - 1.5f);
    for (int i = 0; i < nkv; ++i) hv[i] = __float2half((float)((i * 5 + 2) % 11) * 0.4f - 0.8f);
    AttnRefFp64(hq.data(), hk.data(), hv.data(), href.data(),
                seq, heads, head_dim, kv_heads, scale, kv_base);

    __half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, nq * 2); cudaMalloc(&dk, nkv * 2); cudaMalloc(&dv, nkv * 2);
    cudaMalloc(&dout, nq * 2);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nkv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nkv * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenFullAttnKernel(
                nullptr, dq, dk, dv, dout, seq, heads, head_dim,
                kv_heads, scale, kv_base) == cudaSuccess);

    std::vector<__half> hout(nq);
    cudaMemcpy(hout.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nq; ++i)
        CHECK(std::abs(__half2float(hout[i]) - __half2float(href[i])) < 1e-2f);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}

TEST_CASE("RunQwenFullAttnKernel long-context matches FP64 ref", "[qwen][attn][full]")
{
    constexpr int seq = 64, heads = 2, kv_heads = 1, head_dim = 8, kv_base = 2000;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const int nq = seq * heads * head_dim;
    const int nkv = (seq + kv_base) * kv_heads * head_dim;

    std::vector<__half> hq(nq), hk(nkv), href(nq);
    std::vector<__half> hv(nkv);
    for (int i = 0; i < nq; ++i) hq[i] = __float2half((float)((i * 7 + 1) % 13) * 0.2f - 1.2f);
    for (int i = 0; i < nkv; ++i) hk[i] = __float2half((float)((i * 11 + 3) % 17) * 0.3f - 1.5f);
    for (int i = 0; i < nkv; ++i) hv[i] = __float2half((float)((i * 5 + 2) % 11) * 0.4f - 0.8f);
    AttnRefFp64(hq.data(), hk.data(), hv.data(), href.data(),
                seq, heads, head_dim, kv_heads, scale, kv_base);

    __half *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, nq * 2); cudaMalloc(&dk, nkv * 2); cudaMalloc(&dv, nkv * 2);
    cudaMalloc(&dout, nq * 2);
    cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nkv * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nkv * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenFullAttnKernel(
                nullptr, dq, dk, dv, dout, seq, heads, head_dim,
                kv_heads, scale, kv_base) == cudaSuccess);

    std::vector<__half> hout(nq);
    cudaMemcpy(hout.data(), dout, nq * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nq; ++i)
        CHECK(std::abs(__half2float(hout[i]) - __half2float(href[i])) < 1e-2f);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}