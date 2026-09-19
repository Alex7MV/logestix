// Gemma-4 attention tests: sliding band-mask + full attention (GQA).
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "lsxcommon/gemma_moe_inference.h"

namespace {

inline bool BfTol(float got, double want)
{
    return std::abs(got - want) <=
           std::max(0.03, 0.01 * std::abs(want));
}

std::vector<__nv_bfloat16> RandHalves(size_t n, std::mt19937 &rng)
{
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> h(n);
    for (auto &v : h) v = __float2bfloat16(nd(rng));
    return h;
}

} // namespace

TEST_CASE("sliding window band mask ignores out-of-window junk",
          "[gemma][kernels]") {
    constexpr int seq = 12, slots = 16, pos_base = 0;
    constexpr int heads = 1, kv_heads = 1, head_dim = 4, window = 8;
    std::mt19937 rng(5);

    auto k = RandHalves(static_cast<size_t>(slots) * kv_heads * head_dim, rng);
    auto v = RandHalves(static_cast<size_t>(slots) * kv_heads * head_dim, rng);

    // Query rows: deterministic; the query at t=11 attends to slots 4..11.
    std::vector<__nv_bfloat16> q(seq * heads * head_dim);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = __float2bfloat16(((i % 13) - 6) * 0.3f);

    auto run = [&](const std::vector<__nv_bfloat16> &kk,
                   const std::vector<__nv_bfloat16> &vv) {
        __nv_bfloat16 *dq, *dk, *dv, *dout;
        cudaMalloc(&dq, q.size() * 2);
        cudaMalloc(&dk, kk.size() * 2);
        cudaMalloc(&dv, vv.size() * 2);
        cudaMalloc(&dout, q.size() * 2);
        cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, kk.data(), kk.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, vv.data(), vv.size() * 2, cudaMemcpyHostToDevice);
        REQUIRE(lsxcommon::gemma::RunGemmaAttention(
                    dq, dk, dv, dout, seq, slots, pos_base, heads, kv_heads,
                    head_dim, window, /*scale=*/0.5f, nullptr) == cudaSuccess);
        std::vector<__nv_bfloat16> out(q.size());
        REQUIRE(cudaMemcpy(out.data(), dout, out.size() * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
        return out;
    };

    auto a = run(k, v);

    // Corrupt ONLY slots < 11-window+1 = 4 (out of band for t=11).
    auto k2 = k;
    auto v2 = v;
    for (int s = 0; s < 4; ++s)
        for (int d = 0; d < head_dim; ++d) {
            k2[s * head_dim + d] = __float2bfloat16(7.7f);
            v2[s * head_dim + d] = __float2bfloat16(-9.9f);
        }
    auto b = run(k2, v2);

    const int64_t row11 = static_cast<int64_t>(11) * heads * head_dim;
    for (int d = 0; d < head_dim; ++d)
        INFO("d=" << d << " a=" << __bfloat162float(a[row11 + d]) << " b="
                  << __bfloat162float(b[row11 + d]));
    // Bitwise identical: out-of-band values never entered the reduction.
    CHECK(std::memcmp(a.data() + row11, b.data() + row11,
                      head_dim * sizeof(__nv_bfloat16)) == 0);
}

TEST_CASE("full attention matches CPU FP64 (GQA)", "[gemma][kernels]") {
    constexpr int seq = 5, slots = 8, pos_base = 0;
    constexpr int heads = 4, kv_heads = 2, head_dim = 8, window = 0;
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> qf(seq * heads * head_dim), kf(slots * kv_heads * head_dim),
        vf(slots * kv_heads * head_dim);
    for (auto &x : qf) x = nd(rng);
    for (auto &x : kf) x = nd(rng);
    for (auto &x : vf) x = nd(rng);
    const float scale = 0.35f;

    // CPU FP64 reference.
    std::vector<float> ref(seq * heads * head_dim, 0.f);
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < heads; ++h) {
            const int kvh = h / (heads / kv_heads);
            const int last = pos_base + t;
            std::vector<double> scores(last + 1);
            double M = -1e300;
            for (int s = 0; s <= last; ++s) {
                double dot = 0;
                for (int d = 0; d < head_dim; ++d)
                    dot += double(qf[(t * heads + h) * head_dim + d]) *
                           double(kf[(static_cast<int64_t>(s) * kv_heads +
                                      kvh) *
                                     head_dim +
                                     d]);
                scores[s] = dot * scale;
                M = std::max(M, scores[s]);
            }
            double Z = 0;
            for (int s = 0; s <= last; ++s) Z += std::exp(scores[s] - M);
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0;
                for (int s = 0; s <= last; ++s)
                    acc += std::exp(scores[s] - M) / Z *
                           double(vf[(static_cast<int64_t>(s) * kv_heads +
                                      kvh) *
                                     head_dim +
                                     d]);
                ref[(t * heads + h) * head_dim + d] =
                    static_cast<float>(acc);
            }
        }

    std::vector<__nv_bfloat16> q(qf.size()), k(kf.size()), v(vf.size());
    for (size_t i = 0; i < qf.size(); ++i) q[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kf.size(); ++i) {
        k[i] = __float2bfloat16(kf[i]);
        v[i] = __float2bfloat16(vf[i]);
    }

    __nv_bfloat16 *dq, *dk, *dv, *dout;
    REQUIRE(cudaMalloc(&dq, q.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, k.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dv, v.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, q.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(lsxcommon::gemma::RunGemmaAttention(
                dq, dk, dv, dout, seq, slots, pos_base, heads, kv_heads,
                head_dim, window, scale, nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> out(q.size());
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < ref.size(); ++i) {
        INFO("i=" << i << " got=" << __bfloat162float(out[i]) << " want="
                  << ref[i]);
        CHECK(BfTol(__bfloat162float(out[i]), ref[i]));
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}



// Split-KV flash-decode path (seq==1): chunk-rescaled online softmax.
// Parity against the CPU FP64 reference including a >1-chunk context, plus
// the sliding-window band predicate on the split path.
TEST_CASE("split-KV decode attention matches CPU FP64 (GQA + band)",
          "[gemma][kernels][split_kv]") {
    const int seq = 1, heads = 4, kv_heads = 2, head_dim = 8;
    const int window = 12;                     // sliding band
    const int slots = 2000, pos_base = 1990;   // 11 valid rows, >1 chunk of 8
    constexpr int kChunkProbe = 8;
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> qf(heads * head_dim), kf(slots * kv_heads * head_dim),
        vf(slots * kv_heads * head_dim);
    for (auto &x : qf) x = nd(rng);
    for (auto &x : kf) x = nd(rng);
    for (auto &x : vf) x = nd(rng);
    const float scale = 0.35f;

    const int last = pos_base + seq - 1;
    const int first = last - window + 1 > 0 ? last - window + 1 : 0;

    // CPU FP64 reference over the band [first..last].
    std::vector<float> ref(heads * head_dim, 0.f);
    for (int h = 0; h < heads; ++h) {
        const int kvh = h / (heads / kv_heads);
        std::vector<double> scores;
        for (int s = first; s <= last; ++s) {
            double dot = 0;
            for (int d = 0; d < head_dim; ++d)
                dot += double(qf[h * head_dim + d]) *
                       double(kf[(static_cast<int64_t>(s) * kv_heads + kvh) *
                                     head_dim + d]);
            scores.push_back(dot * scale);
        }
        double M = *std::max_element(scores.begin(), scores.end());
        double Z = 0;
        for (double s : scores) Z += std::exp(s - M);
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0;
            for (size_t j = 0; j < scores.size(); ++j)
                acc += std::exp(scores[j] - M) / Z *
                       double(vf[(static_cast<int64_t>(first + j) * kv_heads +
                                  kvh) * head_dim + d]);
            ref[h * head_dim + d] = static_cast<float>(acc);
        }
    }

    std::vector<__nv_bfloat16> q(qf.size()), k(kf.size()), v(vf.size());
    for (size_t i = 0; i < qf.size(); ++i) q[i] = __float2bfloat16(qf[i]);
    for (size_t i = 0; i < kf.size(); ++i) {
        k[i] = __float2bfloat16(kf[i]);
        v[i] = __float2bfloat16(vf[i]);
    }

    __nv_bfloat16 *dq, *dk, *dv, *dout;
    REQUIRE(cudaMalloc(&dq, q.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, k.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dv, v.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, q.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    (void)kChunkProbe;
    REQUIRE(lsxcommon::gemma::RunGemmaAttentionSplitKV(
                dq, dk, dv, dout, seq, slots, pos_base, heads, kv_heads,
                head_dim, window, scale, nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> out(q.size());
    REQUIRE(cudaMemcpy(out.data(), dout, out.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < ref.size(); ++i) {
        INFO("i=" << i << " got=" << __bfloat162float(out[i]) << " want="
                  << ref[i]);
        CHECK(BfTol(__bfloat162float(out[i]), ref[i]));
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}
