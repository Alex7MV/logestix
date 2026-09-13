// tests/test_deepseek_moe_v41_kernels.cu — DeepSeek-V4.1-Flash device-math
// parity: RMSNorm (eps 1e-20), YaRN RoPE, Hyper-Connections (mixes/Sinkhorn,
// hc_pre, hc_post) against the host reference in
// deepseek_moe_v41_device_math.h.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "deepseek_moe_v41_device_math.h"
#include "deepseek_moe_v41_inference.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using lsxcommon::deepseek_v41::HcPostRef;
using lsxcommon::deepseek_v41::HcPreRef;
using lsxcommon::deepseek_v41::HcSplitSinkhornRef;
using lsxcommon::deepseek_v41::RmsNormRef;
using lsxcommon::deepseek_v41::RopeApplyRef;
using lsxcommon::deepseek_v41::YarnFreqsRef;

namespace {

uint16_t H(float v) { return __half_as_ushort(__float2half_rn(v)); }
float F(uint16_t b) { return __half2float(__ushort_as_half(b)); }

std::vector<uint16_t> ToHalf(const std::vector<float> &f)
{
    std::vector<uint16_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) out[i] = H(f[i]);
    return out;
}

std::vector<float> ToFloat(const std::vector<uint16_t> &h)
{
    std::vector<float> out(h.size());
    for (size_t i = 0; i < h.size(); ++i) out[i] = F(h[i]);
    return out;
}

} // namespace

TEST_CASE("deepseek-v41 rmsnorm kernel matches the host reference",
          "[deepseek-v41][kernels]")
{
    const int n = 64;
    std::vector<float> x(n), g(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 0.01f * static_cast<float>(i) - 0.3f;
        g[i] = 1.0f + 1e-3f * static_cast<float>(i);
    }
    std::vector<uint16_t> xh = ToHalf(x), gh = ToHalf(g), oh(n);
    uint16_t *dx, *dg, *do_;
    REQUIRE(cudaMalloc(&dx, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dg, n * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&do_, n * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, xh.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dg, gh.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::deepseek_v41::LaunchRmsNormHalf(dx, dg, do_, n, 1e-20f,
                                                       nullptr) == cudaSuccess);
    REQUIRE(cudaMemcpy(oh.data(), do_, n * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    std::vector<float> xf = ToFloat(xh), gf = ToFloat(gh), ref(n);
    RmsNormRef(xf.data(), gf.data(), ref.data(), n, 1e-20f);
    for (int i = 0; i < n; ++i)
        CHECK(F(oh[i]) == Catch::Approx(ref[i]).margin(2e-3f));

    cudaFree(dx); cudaFree(dg); cudaFree(do_);
}

TEST_CASE("deepseek-v41 rope kernel matches the host reference",
          "[deepseek-v41][kernels]")
{
    const int rope_dim = 8, row_len = 16, rows = 3;
    std::vector<float> cos_t, sin_t;
    YarnFreqsRef(rope_dim, 1, 65536, 160000.0f, 16.0f, 32, 1, &cos_t,
                 &sin_t);

    std::vector<float> base(rows * row_len);
    for (size_t i = 0; i < base.size(); ++i)
        base[i] = 0.05f * static_cast<float>(i) - 1.0f;
    std::vector<uint16_t> xh = ToHalf(base);

    uint16_t *dx;
    REQUIRE(cudaMalloc(&dx, xh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchRopeHalf(dx, rows, row_len, rope_dim,
                                                    cos_t.data(), sin_t.data(),
                                                    false, nullptr) ==
            cudaSuccess);
    std::vector<uint16_t> got(xh.size());
    REQUIRE(cudaMemcpy(got.data(), dx, xh.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<float> xf = ToFloat(xh);
    for (int r = 0; r < rows; ++r)
        RopeApplyRef(xf.data() + r * row_len, rope_dim, cos_t.data(),
                     sin_t.data(), false);
    for (size_t i = 0; i < xf.size(); ++i) CHECK(got[i] == H(xf[i]));

    cudaFree(dx);
}

TEST_CASE("deepseek-v41 hyper-connection kernels match the host reference",
          "[deepseek-v41][kernels]")
{
    const int hc = 4, dim = 8, iters = 20, eps = 1e-6f;
    const int mix_hc = (2 + hc) * hc;
    const int total = hc * dim;

    std::vector<float> x(total);
    for (int i = 0; i < total; ++i) x[i] = 0.02f * static_cast<float>(i) - 1.0f;
    std::vector<float> fn(static_cast<size_t>(mix_hc) * total);
    for (size_t i = 0; i < fn.size(); ++i)
        fn[i] = 0.001f * static_cast<float>(static_cast<int>(i % 37) - 18);
    const float scale[3] = {0.5f, 0.7f, 0.3f};
    std::vector<float> hbase(mix_hc);
    for (int i = 0; i < mix_hc; ++i) hbase[i] = 0.01f * static_cast<float>(i);

    float *dx, *dfn, *dscale, *dbase, *dpre, *dpost, *dcomb;
    REQUIRE(cudaMalloc(&dx, total * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dfn, fn.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dscale, sizeof(scale)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dbase, mix_hc * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dpre, hc * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dpost, hc * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dcomb, hc * hc * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, x.data(), total * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dfn, fn.data(), fn.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dscale, scale, sizeof(scale), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dbase, hbase.data(), mix_hc * 4, cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::deepseek_v41::LaunchHcMixes(
                dx, dfn, dscale, dbase, hc, dim, iters, eps, dpre, dpost, dcomb,
                nullptr) == cudaSuccess);

    std::vector<float> pre(hc), post(hc), comb(hc * hc);
    REQUIRE(cudaMemcpy(pre.data(), dpre, hc * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(post.data(), dpost, hc * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(comb.data(), dcomb, hc * hc * 4, cudaMemcpyDeviceToHost) == cudaSuccess);

    // Host reference: recompute mixes (sequential dot, float) then split/sinkhorn.
    std::vector<float> mixes(mix_hc);
    double ss = 0.0;
    for (int i = 0; i < total; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float rs = 1.0f / std::sqrt(static_cast<float>(ss / total) + eps);
    for (int m = 0; m < mix_hc; ++m) {
        float d = 0.0f;
        for (int k = 0; k < total; ++k) d += fn[m * total + k] * x[k];
        mixes[m] = d * rs;
    }
    std::vector<float> rpre(hc), rpost(hc), rcomb(hc * hc);
    HcSplitSinkhornRef(mixes.data(), scale, hbase.data(), hc, iters, eps,
                       rpre.data(), rpost.data(), rcomb.data());
    for (int i = 0; i < hc; ++i) {
        CHECK(pre[i] == Catch::Approx(rpre[i]).margin(2e-4f));
        CHECK(post[i] == Catch::Approx(rpost[i]).margin(2e-4f));
    }
    for (int i = 0; i < hc * hc; ++i)
        CHECK(comb[i] == Catch::Approx(rcomb[i]).margin(2e-4f));

    // hc_pre / hc_post are order-preserving linear maps -> tight parity.
    std::vector<float> premix = {0.25f, 0.5f, 0.125f, 0.125f};
    float *dpre_out = nullptr, *dpost_out = nullptr, *dsub = nullptr;
    REQUIRE(cudaMalloc(&dpre_out, dim * 4) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchHcPre(dx, premix.data(), dpre_out,
                                                 hc, dim, nullptr) ==
            cudaSuccess);
    std::vector<float> pre_out(dim), ref_out(dim);
    REQUIRE(cudaMemcpy(pre_out.data(), dpre_out, dim * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    HcPreRef(x.data(), premix.data(), ref_out.data(), hc, dim);
    for (int i = 0; i < dim; ++i)
        CHECK(pre_out[i] == Catch::Approx(ref_out[i]).margin(1e-6f));

    REQUIRE(cudaMalloc(&dpost_out, total * 4) == cudaSuccess);
    std::vector<float> sub(dim);
    for (int i = 0; i < dim; ++i) sub[i] = 0.3f + 0.01f * static_cast<float>(i);
    REQUIRE(cudaMalloc(&dsub, dim * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dsub, sub.data(), dim * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchHcPost(dsub, dx, dpost, dcomb,
                                                  dpost_out, hc, dim,
                                                  nullptr) == cudaSuccess);
    std::vector<float> post_out(total), ref_post(total);
    REQUIRE(cudaMemcpy(post_out.data(), dpost_out, total * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    HcPostRef(sub.data(), x.data(), post.data(), comb.data(), ref_post.data(),
              hc, dim);
    for (int i = 0; i < total; ++i)
        CHECK(post_out[i] == Catch::Approx(ref_post[i]).margin(1e-6f));

    cudaFree(dx); cudaFree(dfn); cudaFree(dscale); cudaFree(dbase);
    cudaFree(dpre); cudaFree(dpost); cudaFree(dcomb);
    cudaFree(dpre_out); cudaFree(dpost_out); cudaFree(dsub);
}

TEST_CASE("deepseek-v41 sparse attention matches a float reference",
          "[deepseek-v41][kernels]")
{
    const int heads = 2, d = 32, q_rows = 1, kv_rows = 4, topk = 4;
    std::vector<float> qf(q_rows * heads * d), kvf(kv_rows * d);
    for (size_t i = 0; i < qf.size(); ++i)
        qf[i] = 0.1f * static_cast<float>(static_cast<int>(i % 17) - 8);
    for (size_t i = 0; i < kvf.size(); ++i)
        kvf[i] = 0.05f * static_cast<float>(static_cast<int>(i % 23) - 11);
    const float sink[heads] = {0.5f, -0.25f};
    const int32_t idx[topk] = {0, 2, -1, 3};

    std::vector<uint16_t> qh = ToHalf(qf), kvh = ToHalf(kvf);
    uint16_t *dq, *dkv, *dout;
    int32_t *didx;
    float *dsink;
    REQUIRE(cudaMalloc(&dq, qh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dkv, kvh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, qh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&didx, sizeof(idx)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dsink, sizeof(sink)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dkv, kvh.data(), kvh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(didx, idx, sizeof(idx), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dsink, sink, sizeof(sink), cudaMemcpyHostToDevice) == cudaSuccess);

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    REQUIRE(lsxcommon::deepseek_v41::LaunchSparseAttnHalf(
                dq, q_rows, heads, d, dkv, kv_rows, dsink, didx, topk, dout,
                scale, nullptr) == cudaSuccess);
    std::vector<uint16_t> got(qh.size());
    REQUIRE(cudaMemcpy(got.data(), dout, qh.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // Float reference over the same FP16-rounded inputs.
    std::vector<float> qf16 = ToFloat(qh), kvf16 = ToFloat(kvh);
    for (int h = 0; h < heads; ++h) {
        std::vector<float> score(topk);
        float m = -INFINITY;
        for (int t = 0; t < topk; ++t) {
            if (idx[t] < 0) { score[t] = -INFINITY; continue; }
            float dot = 0.0f;
            for (int k = 0; k < d; ++k)
                dot += qf16[h * d + k] * kvf16[idx[t] * d + k];
            score[t] = dot * scale;
            m = std::max(m, score[t]);
        }
        float s = std::exp(sink[h] - m);
        std::vector<float> acc(d, 0.0f);
        for (int t = 0; t < topk; ++t) {
            if (idx[t] < 0) continue;
            const float p = std::exp(score[t] - m);
            s += p;
            for (int k = 0; k < d; ++k) acc[k] += p * kvf16[idx[t] * d + k];
        }
        for (int k = 0; k < d; ++k)
            CHECK(F(got[h * d + k]) ==
                  Catch::Approx(acc[k] / s).margin(3e-3f));
    }

    cudaFree(dq); cudaFree(dkv); cudaFree(dout); cudaFree(didx);
    cudaFree(dsink);
}

TEST_CASE("deepseek-v41 compressor softmax-pool matches the host reference",
          "[deepseek-v41][kernels]")
{
    const int d = 8, ratio = 2, tokens = 4;
    std::vector<float> kv(tokens * d), score(tokens * d);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < d; ++j) {
            kv[t * d + j] = 0.1f * static_cast<float>(t) -
                            0.05f * static_cast<float>(j) + 0.02f;
            score[t * d + j] =
                0.3f * static_cast<float>((t + 1) * (j + 1) % 7) - 1.0f;
        }

    float *dkv, *dscore, *dout;
    REQUIRE(cudaMalloc(&dkv, kv.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dscore, score.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, 2 * d * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dkv, kv.data(), kv.size() * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dscore, score.data(), score.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Group 0 = tokens {0,1}; group 1 = tokens {2,3}.
    REQUIRE(lsxcommon::deepseek_v41::LaunchCompressorPool(
                dkv, dscore, 1, ratio, d, 0, dout, nullptr) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchCompressorPool(
                dkv, dscore, 1, ratio, d, 2, dout + d, nullptr) == cudaSuccess);

    std::vector<float> got(2 * d);
    REQUIRE(cudaMemcpy(got.data(), dout, 2 * d * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<float> ref0(d), ref1(d);
    lsxcommon::deepseek_v41::CompressorPoolGroupRef(kv.data(), score.data(),
                                                    ratio, d, ref0.data());
    lsxcommon::deepseek_v41::CompressorPoolGroupRef(
        kv.data() + 2 * d, score.data() + 2 * d, ratio, d, ref1.data());
    for (int j = 0; j < d; ++j) {
        CHECK(got[j] == Catch::Approx(ref0[j]).margin(1e-6f));
        CHECK(got[d + j] == Catch::Approx(ref1[j]).margin(1e-6f));
    }

    cudaFree(dkv); cudaFree(dscore); cudaFree(dout);
}
