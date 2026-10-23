#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "lsxcommon/lsx_attn_inference.h"

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
// ── Split-KV decode attention (long-context decode path) ───────────────────
// Real 35B-A3B geometry: NH=16, NKV=2 (GQA 8:1), HD=256. The split-KV path is
// numerically independent from the legacy single-block kernel (chunk-rescaled
// online softmax) — parity holds within an FP16 tolerance envelope; run-to-run
// bits must be identical (determinism contract).

namespace {

void FillAttnHalf(std::vector<__half> &dst, int n, int seed)
{
    dst.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float v =
            ((static_cast<float>((i * 37 + seed * 91) % 251)) / 251.0f) - 0.5f;
        dst[static_cast<size_t>(i)] = __float2half(v);
    }
}

} // namespace

TEST_CASE("Split-KV decode attention matches legacy within tolerance",
          "[qwen-full-attn]")
{
    constexpr int heads = 16, kv_heads = 2, head_dim = 256;
    constexpr int kv_base = 2048;          // nscores = 2049 (multi-chunk)
    constexpr int nscores = kv_base + 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int qn = heads * head_dim;
    const int kvn = nscores * kv_heads * head_dim;

    std::vector<__half> hq, hk, hv;
    FillAttnHalf(hq, qn, 1);
    FillAttnHalf(hk, kvn, 2);
    FillAttnHalf(hv, kvn, 3);

    __half *d_q, *d_k, *d_v, *d_legacy, *d_split;
    cudaMalloc(&d_q, qn * sizeof(__half));
    cudaMalloc(&d_k, kvn * sizeof(__half));
    cudaMalloc(&d_v, kvn * sizeof(__half));
    cudaMalloc(&d_legacy, qn * sizeof(__half));
    cudaMalloc(&d_split, qn * sizeof(__half));
    cudaMemcpy(d_q, hq.data(), qn * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, hk.data(), kvn * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, hv.data(), kvn * sizeof(__half), cudaMemcpyHostToDevice);

    // Legacy reference (single-block kernel, public numeric entry).
    REQUIRE(lsxcommon::qwen::RunAttnFullAttnKernel(
                0, d_q, d_k, d_v, d_legacy, 1, heads, head_dim, kv_heads,
                scale, kv_base) == cudaSuccess);
    // Split-KV path.
    REQUIRE(lsxcommon::qwen::RunAttnFullAttnKernelSplitKV(
                0, d_q, d_k, d_v, d_split, 1, heads, head_dim, kv_heads,
                scale, kv_base) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hl(static_cast<size_t>(qn)),
        hs(static_cast<size_t>(qn));
    cudaMemcpy(hl.data(), d_legacy, qn * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(), d_split, qn * sizeof(__half), cudaMemcpyDeviceToHost);

    double max_abs = 0.0;
    for (int i = 0; i < qn; ++i) {
        const float a = __half2float(hl[static_cast<size_t>(i)]);
        const float b = __half2float(hs[static_cast<size_t>(i)]);
        max_abs = std::max(max_abs, static_cast<double>(std::abs(a - b)));
    }
    // Envelope: FP16 output granularity + chunk-rescale rounding.
    CHECK(max_abs < 2e-2);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_legacy); cudaFree(d_split);
}

TEST_CASE("Split-KV decode attention is run-to-run deterministic",
          "[qwen-full-attn]")
{
    constexpr int heads = 16, kv_heads = 2, head_dim = 256;
    constexpr int kv_base = 1024;          // nscores = 1025 (chunk boundary +1)
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int qn = heads * head_dim;
    const int kvn = (kv_base + 1) * kv_heads * head_dim;

    std::vector<__half> hq, hk, hv;
    FillAttnHalf(hq, qn, 5);
    FillAttnHalf(hk, kvn, 6);
    FillAttnHalf(hv, kvn, 7);

    __half *d_q, *d_k, *d_v, *d_a, *d_b;
    cudaMalloc(&d_q, qn * sizeof(__half));
    cudaMalloc(&d_k, kvn * sizeof(__half));
    cudaMalloc(&d_v, kvn * sizeof(__half));
    cudaMalloc(&d_a, qn * sizeof(__half));
    cudaMalloc(&d_b, qn * sizeof(__half));
    cudaMemcpy(d_q, hq.data(), qn * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, hk.data(), kvn * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, hv.data(), kvn * sizeof(__half), cudaMemcpyHostToDevice);

    auto run = [&]( __half *dst ) {
        REQUIRE(lsxcommon::qwen::RunAttnFullAttnKernelSplitKV(
                    0, d_q, d_k, d_v, dst, 1, heads, head_dim, kv_heads,
                    scale, kv_base) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    };
    run(d_a);
    run(d_b);

    std::vector<__half> ha(static_cast<size_t>(qn)), hb(static_cast<size_t>(qn));
    cudaMemcpy(ha.data(), d_a, qn * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), d_b, qn * sizeof(__half), cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(ha.data(), hb.data(), qn * sizeof(__half)) == 0);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_a); cudaFree(d_b);
}

TEST_CASE("Split-KV decode attention rejects unsupported geometry",
          "[qwen-full-attn]")
{
    // The entry must reject before any launch; buffers are sized for the
    // WORST-CASE index layout of each config so a pre-fix launch (if the
    // guard regresses) stays memory-safe and fails the contract cleanly
    // instead of corrupting the CUDA context.
    __half *d_q, *d_k, *d_v, *d_out;
    constexpr int kQKvElems =
        (1024 + 1) * 6 * 128;   // nscores x max(kv reads) x max(head_dim)
    constexpr int kQoElems = 6 * 128;
    cudaMalloc(&d_q, kQoElems * sizeof(__half));
    cudaMalloc(&d_k, kQKvElems * sizeof(__half));
    cudaMalloc(&d_v, kQKvElems * sizeof(__half));
    cudaMalloc(&d_out, kQoElems * sizeof(__half));

    // Non-pow2 head_dim: the sreduce tree reduction (partial + combine) and
    // the one-thread-per-dim value pass assume a power-of-two blockDim; a
    // non-pow2 block silently corrupts the softmax denominator.
    CHECK(lsxcommon::qwen::RunAttnFullAttnKernelSplitKV(
              0, d_q, d_k, d_v, d_out, 1, 4, 96, 1, 0.1f, 1024) ==
          cudaErrorInvalidValue);
    // GQA mapping kvh = h / (heads / kv_heads) requires exact division:
    // heads % kv_heads != 0 reads past the kv-head table.
    CHECK(lsxcommon::qwen::RunAttnFullAttnKernelSplitKV(
              0, d_q, d_k, d_v, d_out, 1, 6, 128, 4, 0.1f, 1024) ==
          cudaErrorInvalidValue);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
}

TEST_CASE("Non-split attention rejects non-power-of-two head_dim",
          "[qwen-full-attn]")
{
    // The unsplit kernel reduces the softmax denominator with a `threads/2`
    // tree; for head_dim=192 the tree does not cover the row (audit X7).
    // Split-KV already rejects it; the non-split entry must too.
    constexpr int kHeads = 4, kHd = 192, kKvHeads = 1;
    __half *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, (size_t)kHeads * kHd * sizeof(__half));
    cudaMalloc(&d_k, (size_t)kKvHeads * kHd * sizeof(__half));
    cudaMalloc(&d_v, (size_t)kKvHeads * kHd * sizeof(__half));
    cudaMalloc(&d_out, (size_t)kHeads * kHd * sizeof(__half));

    CHECK(lsxcommon::qwen::RunAttnFullAttnKernel(
              0, d_q, d_k, d_v, d_out, 1, kHeads, kHd, kKvHeads, 0.1f, 0) ==
          cudaErrorInvalidValue);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
}
