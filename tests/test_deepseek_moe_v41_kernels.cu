// tests/test_deepseek_moe_v41_kernels.cu — DeepSeek-V4.1-Flash device-math
// parity: RMSNorm (eps 1e-20), YaRN RoPE, Hyper-Connections (mixes/Sinkhorn,
// hc_pre, hc_post) against the host reference in
// deepseek_moe_v41_device_math.h.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "deepseek_moe_v41_device_math.h"
#include "deepseek_moe_v41_inference.h"
#include "deepseek_moe_v41_inference_csa.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

TEST_CASE("deepseek-v41 indexer top-k and candidate blocks (host)",
          "[deepseek-v41][kernels]")
{
    const float inf = INFINITY;
    const float score[8] = {0.1f, 0.5f, -inf, 0.9f, 0.2f, -inf, -inf, -inf};
    int idxs[3] = {0, 0, 0};
    lsxcommon::deepseek_v41::IndexTopkRef(score, 8, /*num_valid=*/5, /*topk=*/3,
                                          /*offset=*/100, idxs);
    CHECK(idxs[0] == 101);
    CHECK(idxs[1] == 103);
    CHECK(idxs[2] == 104);

    // Candidate blocks: 20 positions, 3 blocks of 8; block 0 max 0.2,
    // block 1 (newest) max 0.1, block 2 all -inf.
    std::vector<float> cs(20, -inf);
    cs[0] = 0.2f;
    cs[8] = 0.1f;
    std::vector<uint8_t> mask(20, 0);
    lsxcommon::deepseek_v41::SelectCandidateBlocksRef(cs.data(), 20,
                                                      /*num_valid=*/10,
                                                      /*topk_blocks=*/2,
                                                      /*block_size=*/8,
                                                      mask.data());
    for (int i = 0; i < 16; ++i) CHECK(mask[i] == 1);
    for (int i = 16; i < 20; ++i) CHECK(mask[i] == 0);
}

TEST_CASE("deepseek-v41 indexer score kernel matches the host reference",
          "[deepseek-v41][kernels]")
{
    const int heads = 2, dim = 32, n_pos = 3;
    std::vector<float> qf(heads * dim), kf(n_pos * dim);
    for (size_t i = 0; i < qf.size(); ++i)
        qf[i] = 0.1f * static_cast<float>(static_cast<int>(i % 13) - 6);
    for (size_t i = 0; i < kf.size(); ++i)
        kf[i] = 0.05f * static_cast<float>(static_cast<int>(i % 17) - 8);
    const float w[heads] = {0.5f, 1.5f};
    std::vector<uint16_t> qh, kh;
    for (float v : qf) qh.push_back(H(v));
    for (float v : kf) kh.push_back(H(v));

    uint16_t *dq, *dk;
    float *dw, *dout;
    REQUIRE(cudaMalloc(&dq, qh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, kh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, sizeof(w)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, n_pos * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, qh.data(), qh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dk, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, w, sizeof(w), cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::deepseek_v41::LaunchIndexerScores(
                dq, heads, dim, dk, n_pos, dw, dout, nullptr) == cudaSuccess);
    std::vector<float> got(n_pos);
    REQUIRE(cudaMemcpy(got.data(), dout, n_pos * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    std::vector<float> qff = ToFloat(qh), kff = ToFloat(kh);
    for (int p = 0; p < n_pos; ++p) {
        float acc = 0.0f;
        for (int h = 0; h < heads; ++h) {
            float d = 0.0f;
            for (int j = 0; j < dim; ++j)
                d += qff[h * dim + j] * kff[p * dim + j];
            acc += std::max(d, 0.0f) * w[h];
        }
        CHECK(got[p] == Catch::Approx(acc).margin(1e-3f));
    }

    cudaFree(dq); cudaFree(dk); cudaFree(dw); cudaFree(dout);
}

TEST_CASE("deepseek-v41 CED shared-attention runtime contract",
          "[deepseek-v41][kernels]")
{
    using lsxcommon::deepseek_v41::CsaRuntime;
    using lsxcommon::deepseek_v41::OwnsIndexK;
    using lsxcommon::deepseek_v41::UsesCandidates;
    using lsxcommon::deepseek_v41::UsesSharedTopk;

    // Ownership follows the frozen source sets.
    CHECK(OwnsIndexK(2));
    CHECK(OwnsIndexK(20));
    CHECK_FALSE(OwnsIndexK(24));   // index source, but not a kv/compressor layer
    CHECK_FALSE(UsesSharedTopk(2));
    CHECK_FALSE(UsesSharedTopk(24));
    CHECK(UsesSharedTopk(3));      // between sources -> reuses layer 2's topk
    CHECK(UsesSharedTopk(25));

    // Two-level indexer: only downstream of the candidate source (layer 20).
    CHECK_FALSE(UsesCandidates(2));
    CHECK_FALSE(UsesCandidates(20));
    CHECK(UsesCandidates(21));
    CHECK(UsesCandidates(24));
    CHECK(UsesCandidates(36));

    // Publish/read within the in-order walk: later sources overwrite the slot.
    CsaRuntime rt;
    float kv_a[2] = {1.0f, 2.0f};
    float kv_b[2] = {3.0f, 4.0f};
    rt.PublishCompressKv(kv_a, 1);
    CHECK(rt.compress_kv == kv_a);
    CHECK(rt.compress_len == 1);
    rt.PublishCompressKv(kv_b, 2);
    CHECK(rt.compress_kv == kv_b);
    CHECK(rt.compress_len == 2);

    int32_t idx_a[2] = {1, 2};
    int32_t idx_b[2] = {3, 4};
    rt.PublishTopk(idx_a, 2);
    CHECK(rt.topk_idxs == idx_a);
    rt.PublishTopk(idx_b, 2);
    CHECK(rt.topk_idxs == idx_b);

    rt.Reset();
    CHECK(rt.compress_kv == nullptr);
    CHECK(rt.topk_idxs == nullptr);
    CHECK(rt.candidates == nullptr);
    CHECK(rt.compress_len == 0);
}

TEST_CASE("deepseek-v41 router gate matches the reference selection",
          "[deepseek-v41][kernels]")
{
    using lsxcommon::deepseek_v41::GateRouteRef;
    using lsxcommon::deepseek_v41::SoftplusSqrtRef;

    const float scores[4] = {1.0f, 3.0f, 2.0f, 0.0f};
    const float bias[4] = {0.0f, -10.0f, 5.0f, 0.0f};
    int idx[2] = {-1, -1};
    float w[2] = {0.0f, 0.0f};
    GateRouteRef(scores, bias, 4, 2, 1.0f, 1.5f, /*norm_topk_prob=*/true, idx,
                 w);

    // Bias steers selection: expert 2 (bias +5) and expert 0 win over the
    // higher raw score at expert 1 (bias -10).
    CHECK(idx[0] == 2);
    CHECK(idx[1] == 0);

    // Weights come from the unbiased scores, renormalized and scaled.
    const float a = SoftplusSqrtRef(2.0f);
    const float b = SoftplusSqrtRef(1.0f);
    CHECK(w[0] == Catch::Approx(1.5f * a / (a + b)).margin(1e-5f));
    CHECK(w[1] == Catch::Approx(1.5f * b / (a + b)).margin(1e-5f));
    CHECK((w[0] + w[1]) == Catch::Approx(1.5f).margin(1e-5f));
}

TEST_CASE("deepseek-v41 expert swiglu clamps match the host reference",
          "[deepseek-v41][kernels]")
{
    using lsxcommon::deepseek_v41::ExpertSwigluRef;
    const int n = 64;
    std::vector<float> gate(n), up(n);
    for (int i = 0; i < n; ++i) {
        // Include values beyond the +-10 clamp range.
        gate[i] = 0.9f * static_cast<float>((i * 5) % 29) - 13.0f;
        up[i] = 0.8f * static_cast<float>((i * 7) % 31) - 12.5f;
    }
    float *dg, *du, *dout;
    REQUIRE(cudaMalloc(&dg, n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, n * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dg, gate.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(du, up.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchExpertSwiglu(dg, du, dout, n, 10.0f,
                                                        nullptr) == cudaSuccess);
    std::vector<float> got(n), ref(n);
    REQUIRE(cudaMemcpy(got.data(), dout, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    ExpertSwigluRef(gate.data(), up.data(), ref.data(), n, 10.0f);
    for (int i = 0; i < n; ++i)
        CHECK(got[i] == Catch::Approx(ref[i]).margin(1e-5f));
    cudaFree(dg); cudaFree(du); cudaFree(dout);
}

TEST_CASE("deepseek-v41 fp16 gemm and head logits match host",
          "[deepseek-v41][kernels]")
{
    cublasHandle_t h = nullptr;
    REQUIRE(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS);

    // C[2,3] = A[2,4] * B[4,3].
    const int M = 2, N = 3, K = 4;
    std::vector<float> Af(M * K), Bf(K * N);
    for (size_t i = 0; i < Af.size(); ++i)
        Af[i] = 0.25f * static_cast<float>(static_cast<int>(i % 9) - 4);
    for (size_t i = 0; i < Bf.size(); ++i)
        Bf[i] = 0.2f * static_cast<float>(static_cast<int>(i % 7) - 3);

    std::vector<uint16_t> Ah, Bh;
    for (float v : Af) Ah.push_back(H(v));
    for (float v : Bf) Bh.push_back(H(v));
    uint16_t *dA, *dB, *dC;
    REQUIRE(cudaMalloc(&dA, Ah.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dB, Bh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dC, M * N * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dA, Ah.data(), Ah.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dB, Bh.data(), Bh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchGemmFp16(h, nullptr, dA, dB, dC, M,
                                                    N, K) == cudaSuccess);
    std::vector<uint16_t> Ch(M * N);
    REQUIRE(cudaMemcpy(Ch.data(), dC, M * N * 2, cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<float> Aff = ToFloat(Ah), Bff = ToFloat(Bh);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += Aff[m * K + k] * Bff[k * N + n];
            CHECK(F(Ch[m * N + n]) == Catch::Approx(acc).margin(3e-3f));
        }

    // Head logits: logits[3] = W[3,4] * x[4], F32 output.
    const int V = 3, Hd = 4;
    std::vector<float> Wf(V * Hd), xf(Hd);
    for (size_t i = 0; i < Wf.size(); ++i)
        Wf[i] = 0.3f * static_cast<float>(static_cast<int>(i % 5) - 2);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = 0.1f * static_cast<float>(i) + 0.05f;
    std::vector<uint16_t> Wh, xh;
    for (float v : Wf) Wh.push_back(H(v));
    for (float v : xf) xh.push_back(H(v));
    uint16_t *dW, *dx;
    float *dlog;
    REQUIRE(cudaMalloc(&dW, Wh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, xh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dlog, V * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dW, Wh.data(), Wh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchHeadLogits(h, nullptr, dW, dx, dlog,
                                                      V, Hd) == cudaSuccess);
    std::vector<float> logits(V);
    REQUIRE(cudaMemcpy(logits.data(), dlog, V * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<float> Wff = ToFloat(Wh), xff = ToFloat(xh);
    for (int v = 0; v < V; ++v) {
        float acc = 0.0f;
        for (int j = 0; j < Hd; ++j) acc += Wff[v * Hd + j] * xff[j];
        CHECK(logits[v] == Catch::Approx(acc).margin(1e-4f));
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dW); cudaFree(dx); cudaFree(dlog);
    cublasDestroy(h);
}

TEST_CASE("deepseek-v41 embed lookup and hc expand",
          "[deepseek-v41][kernels]")
{
    const int vocab = 4, hidden = 8;
    std::vector<float> ef(vocab * hidden);
    for (int r = 0; r < vocab; ++r)
        for (int j = 0; j < hidden; ++j)
            ef[r * hidden + j] = 0.1f * static_cast<float>(r) +
                                 0.01f * static_cast<float>(j);
    std::vector<uint16_t> eh;
    for (float v : ef) eh.push_back(H(v));
    const int32_t ids[2] = {2, 0};
    uint16_t *de, *dout;
    int32_t *did;
    REQUIRE(cudaMalloc(&de, eh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, 2 * hidden * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&did, sizeof(ids)) == cudaSuccess);
    REQUIRE(cudaMemcpy(de, eh.data(), eh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(did, ids, sizeof(ids), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchEmbedLookup(de, vocab, hidden, did, 2,
                                                       dout, nullptr) == cudaSuccess);
    std::vector<uint16_t> got(2 * hidden);
    REQUIRE(cudaMemcpy(got.data(), dout, 2 * hidden * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int t = 0; t < 2; ++t)
        for (int j = 0; j < hidden; ++j)
            CHECK(got[t * hidden + j] == eh[ids[t] * hidden + j]);

    // HC expand: 3 identical F32 copies of a 4-wide vector.
    const int dim = 4, hc = 3;
    std::vector<float> xf = {1.5f, -2.0f, 0.25f, 3.0f};
    std::vector<uint16_t> xh;
    for (float v : xf) xh.push_back(H(v));
    uint16_t *dhx;
    float *dhout;
    REQUIRE(cudaMalloc(&dhx, xh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dhout, hc * dim * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dhx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchHcExpand(dhx, dim, hc, dhout,
                                                    nullptr) == cudaSuccess);
    std::vector<float> hgot(hc * dim);
    REQUIRE(cudaMemcpy(hgot.data(), dhout, hc * dim * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int c = 0; c < hc; ++c)
        for (int d = 0; d < dim; ++d)
            CHECK(hgot[c * dim + d] == F(xh[d]));

    cudaFree(de); cudaFree(dout); cudaFree(did);
    cudaFree(dhx); cudaFree(dhout);
}

TEST_CASE("deepseek-v41 attention decode plumbing runs",
          "[deepseek-v41][kernels]")
{
    const int hidden = 32, q_lora = 32, heads = 2, head_dim = 32, rope_dim = 8,
              o_groups = 2, o_lora = 32, window = 4;
    const int q_dim = heads * head_dim;
    const int per_group = head_dim * (heads / o_groups);

    auto pat = [](int i) {
        return H(0.05f * static_cast<float>((i % 13) - 6));
    };
    auto mk = [&](int n) {
        std::vector<uint16_t> h(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) h[i] = pat(i);
        uint16_t *d = nullptr;
        REQUIRE(cudaMalloc(&d, static_cast<size_t>(n) * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, h.data(), static_cast<size_t>(n) * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        return d;
    };
    uint16_t *wq_a = mk(q_lora * hidden);
    uint16_t *q_norm = mk(q_lora);
    uint16_t *wq_b = mk(q_dim * q_lora);
    uint16_t *wkv = mk(head_dim * hidden);
    uint16_t *kv_norm = mk(head_dim);
    uint16_t *wo_a = mk(o_groups * o_lora * per_group);
    uint16_t *wo_b = mk(hidden * o_groups * o_lora);
    uint16_t *x = mk(hidden);
    uint16_t *out = mk(hidden);
    uint16_t *kv_cache = nullptr;
    REQUIRE(cudaMalloc(&kv_cache, static_cast<size_t>(window) * head_dim * 2) ==
            cudaSuccess);
    REQUIRE(cudaMemset(kv_cache, 0, static_cast<size_t>(window) * head_dim * 2) ==
            cudaSuccess);
    float *sink = nullptr;
    REQUIRE(cudaMalloc(&sink, heads * 4) == cudaSuccess);
    std::vector<float> sink_h(heads, 0.1f);
    REQUIRE(cudaMemcpy(sink, sink_h.data(), heads * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    float *cos_t = nullptr, *sin_t = nullptr;
    const int half_rope = rope_dim / 2;
    std::vector<float> cos_h(half_rope, 1.0f), sin_h(half_rope, 0.0f);
    REQUIRE(cudaMalloc(&cos_t, half_rope * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&sin_t, half_rope * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(cos_t, cos_h.data(), half_rope * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(sin_t, sin_h.data(), half_rope * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cublasHandle_t h = nullptr;
    REQUIRE(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS);

    for (int pos = 1; pos <= 2; ++pos) {
        REQUIRE(lsxcommon::deepseek_v41::AttentionForwardDecode(
                    h, nullptr, x, hidden, wq_a, q_lora, q_norm, wq_b, heads,
                    head_dim, rope_dim, wkv, kv_norm, sink, wo_a, o_groups,
                    o_lora, wo_b, cos_t, sin_t, kv_cache, window, pos,
                    out) == cudaSuccess);
    }
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<uint16_t> out_h(hidden);
    REQUIRE(cudaMemcpy(out_h.data(), out, hidden * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (int i = 0; i < hidden; ++i)
        CHECK(std::isfinite(F(out_h[i])));

    cudaFree(wq_a); cudaFree(q_norm); cudaFree(wq_b); cudaFree(wkv);
    cudaFree(kv_norm); cudaFree(wo_a); cudaFree(wo_b); cudaFree(x);
    cudaFree(out);     cudaFree(kv_cache); cudaFree(sink); cudaFree(cos_t);
    cudaFree(sin_t);
    cublasDestroy(h);
}

TEST_CASE("deepseek-v41 shared expert ffn matches a float reference",
          "[deepseek-v41][kernels]")
{
    const int hidden = 8, inter = 8;
    std::vector<float> xf(hidden), w1f(inter * hidden), w3f(inter * hidden),
        w2f(hidden * inter);
    for (int j = 0; j < hidden; ++j) xf[j] = 0.2f * static_cast<float>(j) - 0.5f;
    for (int i = 0; i < inter * hidden; ++i) {
        w1f[i] = 0.15f * static_cast<float>((i % 11) - 5);
        w3f[i] = 0.12f * static_cast<float>((i % 7) - 3);
        w2f[i] = 0.1f * static_cast<float>((i % 9) - 4);
    }
    auto to_h = [](const std::vector<float> &f) {
        std::vector<uint16_t> h;
        for (float v : f) h.push_back(H(v));
        return h;
    };
    std::vector<uint16_t> xh = to_h(xf), w1h = to_h(w1f), w3h = to_h(w3f),
                         w2h = to_h(w2f);
    auto dev = [](const std::vector<uint16_t> &h) {
        uint16_t *d = nullptr;
        REQUIRE(cudaMalloc(&d, h.size() * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice) ==
                cudaSuccess);
        return d;
    };
    uint16_t *dx = dev(xh), *dw1 = dev(w1h), *dw3 = dev(w3h), *dw2 = dev(w2h);
    uint16_t *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, hidden * 2) == cudaSuccess);

    cublasHandle_t h = nullptr;
    REQUIRE(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(lsxcommon::deepseek_v41::SharedExpertForward(
                h, nullptr, dx, hidden, dw1, dw3, dw2, inter, 10.0f, dout) ==
            cudaSuccess);
    std::vector<uint16_t> got(hidden);
    REQUIRE(cudaMemcpy(got.data(), dout, hidden * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    auto Fv = [](const std::vector<uint16_t> &h) {
        std::vector<float> f;
        for (uint16_t b : h) f.push_back(F(b));
        return f;
    };
    std::vector<float> xff = Fv(xh), w1ff = Fv(w1h), w3ff = Fv(w3h),
                       w2ff = Fv(w2h);
    std::vector<float> mid(inter);
    for (int i = 0; i < inter; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < hidden; ++j) {
            g += w1ff[i * hidden + j] * xff[j];
            u += w3ff[i * hidden + j] * xff[j];
        }
        const float uc = std::clamp(u, -10.0f, 10.0f);
        const float gc = std::min(g, 10.0f);
        mid[i] = (gc / (1.0f + std::exp(-gc))) * uc;
    }
    std::vector<uint16_t> midh;
    for (float v : mid) midh.push_back(H(v));
    for (int i = 0; i < hidden; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < inter; ++j)
            acc += w2ff[i * inter + j] * F(midh[j]);
        CHECK(F(got[i]) == Catch::Approx(acc).margin(3e-3f));
    }

    cudaFree(dx); cudaFree(dw1); cudaFree(dw3); cudaFree(dw2); cudaFree(dout);
    cublasDestroy(h);
}
