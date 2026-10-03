// Xing4.0-29B-A4B kernels test (stages 5-6). The mHC block is new to this
// codebase, so the device implementation is checked against an INDEPENDENT host
// reference written straight from modeling_xing4_0.py — not against a
// re-derivation of the kernel's own arithmetic. Tolerances are tight (1e-5
// relative) because a wrong Sinkhorn order or a missing clamp still produces a
// plausible-looking matrix.
//
// Label: fast is wrong here (this needs a GPU), so it joins the inference
// sweep; the target is skipped when no device is present.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/xing_moe_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace lsxcommon::xing;
using Catch::Approx;

namespace {

constexpr int kHcMult = 4;
constexpr int kHidden = 64;      // small enough to brute-force on the host
constexpr int kExperts = 16;
constexpr int kTopK = 4;
constexpr float kEps = 1e-6f;
constexpr float kClampMin = -30.0f;
constexpr float kClampMax = 30.0f;
constexpr float kPostGain = 2.0f;
constexpr int kSinkhorn = 20;

bool HasDevice()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// ── host reference (modeling_xing4_0.py, verbatim structure) ───────────────
void ReferenceHyperConn(const std::vector<float> &hidden,
                        const std::vector<__nv_bfloat16> &hc_fn,
                        const std::vector<__nv_bfloat16> &hc_base,
                        const std::vector<float> &hc_scale,
                        std::vector<float> &collapsed, std::vector<float> &post,
                        std::vector<float> &comb)
{
    const int n = kHcMult * kHidden;
    const int mix = (2 + kHcMult) * kHcMult;

    double sumsq = 0.0;
    for (float v : hidden) sumsq += static_cast<double>(v) * v;
    const float inv = static_cast<float>(1.0 / std::sqrt(sumsq / n + kEps));
    std::vector<float> flat(n);
    for (int i = 0; i < n; ++i) flat[i] = hidden[i] * inv;

    auto bf = [](__nv_bfloat16 v) { return __bfloat162float(v); };
    std::vector<float> t(static_cast<size_t>(mix), 0.0f);
    for (int k = 0; k < mix; ++k)
    {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) acc += bf(hc_fn[k * n + j]) * flat[j];
        t[k] = acc;
    }

    collapsed.assign(kHidden, 0.0f);
    post.assign(kHcMult, 0.0f);
    comb.assign(static_cast<size_t>(kHcMult) * kHcMult, 0.0f);

    std::vector<float> pre(kHcMult);
    for (int d = 0; d < kHcMult; ++d)
    {
        pre[d] = 1.0f / (1.0f + std::exp(-(t[d] * hc_scale[0] +
                                           bf(hc_base[d]))));
        post[d] = kPostGain * (1.0f / (1.0f + std::exp(
                                       -(t[kHcMult + d] * hc_scale[1] +
                                         bf(hc_base[kHcMult + d])))));
    }
    const int off = 2 * kHcMult;
    std::vector<float> c(static_cast<size_t>(kHcMult) * kHcMult, 0.0f);
    float mx = -INFINITY;
    for (int i = 0; i < kHcMult * kHcMult; ++i)
    {
        c[i] = std::min(std::max(t[off + i] * hc_scale[2] + bf(hc_base[off + i]),
                                 kClampMin),
                        kClampMax);
        mx = std::max(mx, c[i]);
    }
    for (int i = 0; i < kHcMult * kHcMult; ++i) c[i] = std::exp(c[i] - mx);
    for (int it = 0; it < kSinkhorn; ++it)
    {
        for (int r = 0; r < kHcMult; ++r)
        {
            float s = 0.0f;
            for (int col = 0; col < kHcMult; ++col)
                s += c[r * kHcMult + col];
            s = 1.0f / (s + kEps);
            for (int col = 0; col < kHcMult; ++col) c[r * kHcMult + col] *= s;
        }
        for (int col = 0; col < kHcMult; ++col)
        {
            float s = 0.0f;
            for (int r = 0; r < kHcMult; ++r) s += c[r * kHcMult + col];
            s = 1.0f / (s + kEps);
            for (int r = 0; r < kHcMult; ++r) c[r * kHcMult + col] *= s;
        }
    }
    comb = c;
    for (int j = 0; j < kHidden; ++j)
    {
        float acc = 0.0f;
        for (int d = 0; d < kHcMult; ++d) acc += pre[d] * hidden[d * kHidden + j];
        collapsed[j] = acc;
    }
}

void ReferenceMix(const std::vector<float> &hidden, const std::vector<float> &out,
                  const std::vector<float> &post, const std::vector<float> &comb,
                  std::vector<float> &dst)
{
    dst.assign(hidden.size(), 0.0f);
    for (int d = 0; d < kHcMult; ++d)
        for (int j = 0; j < kHidden; ++j)
        {
            float acc = post[d] * out[j];
            for (int k = 0; k < kHcMult; ++k)
                acc += comb[d * kHcMult + k] * hidden[k * kHidden + j];
            dst[d * kHidden + j] = acc;
        }
}

void ReferenceRouter(const std::vector<__nv_bfloat16> &weight,
                     const std::vector<float> &bias, const std::vector<float> &x,
                     std::vector<float> &weights, std::vector<int> &ids)
{
    weights.assign(kTopK, 0.0f);
    ids.assign(kTopK, -1);
    std::vector<float> score(kExperts, 0.0f);
    for (int e = 0; e < kExperts; ++e)
    {
        float acc = 0.0f;
        for (int i = 0; i < kHidden; ++i)
            acc += __bfloat162float(weight[e * kHidden + i]) * x[i];
        score[e] = 1.0f / (1.0f + std::exp(-acc));
    }
    std::vector<bool> taken(kExperts, false);
    float wsum = 0.0f;
    for (int k = 0; k < kTopK; ++k)
    {
        int best = -1;
        float best_score = 0.0f;
        for (int e = 0; e < kExperts; ++e)
        {
            if (taken[e]) continue;
            const float s = score[e] + bias[e];
            if (best < 0 || s > best_score)
            {
                best = e;
                best_score = s;
            }
        }
        taken[best] = true;
        ids[k] = best;
        weights[k] = score[best];
        wsum += score[best];
    }
    const float denom = 1.0f / (wsum + 1e-20f);
    for (int k = 0; k < kTopK; ++k) weights[k] *= denom * 2.0f;  // routed_scale
}

// Round-trip helper for the float/bfloat16 device uploads.
template <typename T>
T *Upload(const std::vector<T> &v)
{
    T *p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

std::vector<float> Download(const float *p, size_t n)
{
    std::vector<float> v(n, 0.0f);
    cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

}  // namespace

TEST_CASE("Xing mHC block matches the host reference", "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");

    std::mt19937 rng(20261003);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int n = kHcMult * kHidden;
    const int mix = (2 + kHcMult) * kHcMult;

    std::vector<float> hidden(n);
    for (auto &v : hidden) v = dist(rng);
    std::vector<__nv_bfloat16> hc_fn(static_cast<size_t>(mix) * n);
    for (auto &v : hc_fn) v = __float2bfloat16(dist(rng) * 0.3f);
    std::vector<__nv_bfloat16> hc_base(mix);
    for (auto &v : hc_base) v = __float2bfloat16(dist(rng) * 0.1f);
    const std::vector<float> hc_scale = {0.7f, -0.4f, 1.3f};

    std::vector<float> ref_collapsed, ref_post, ref_comb;
    ReferenceHyperConn(hidden, hc_fn, hc_base, hc_scale, ref_collapsed,
                       ref_post, ref_comb);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_hidden = Upload(hidden);
    auto *d_fn = Upload(hc_fn);
    auto *d_base = Upload(hc_base);
    auto *d_scale = Upload(hc_scale);
    float *d_collapsed = nullptr, *d_post = nullptr, *d_comb = nullptr;
    REQUIRE(cudaMalloc(&d_collapsed, kHidden * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_post, kHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comb, kHcMult * kHcMult * sizeof(float)) == cudaSuccess);

    REQUIRE(LaunchXingHyperConnection(d_hidden, d_fn, d_base, d_scale, kHcMult,
                                      kHidden, kEps, kSinkhorn, kClampMin,
                                      kClampMax, kPostGain, d_collapsed, d_post,
                                      d_comb, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto collapsed = Download(d_collapsed, kHidden);
    const auto post = Download(d_post, kHcMult);
    const auto comb = Download(d_comb, kHcMult * kHcMult);

    // Near-doubly-stochastic: 20 Sinkhorn iterations is what the checkpoint was
    // trained with, so rows/columns approach 1 geometrically rather than hitting
    // it exactly (observed residual <1% on random logits). The exact-parity
    // assertions below against the host reference are the real gate; this only
    // catches a dropped normalisation pass or a transposed matrix.
    for (int r = 0; r < kHcMult; ++r)
    {
        float rs = 0.0f;
        for (int c = 0; c < kHcMult; ++c) rs += comb[r * kHcMult + c];
        REQUIRE(rs == Approx(1.0f).margin(2e-2));
    }
    for (int c = 0; c < kHcMult; ++c)
    {
        float cs = 0.0f;
        for (int r = 0; r < kHcMult; ++r) cs += comb[r * kHcMult + c];
        REQUIRE(cs == Approx(1.0f).margin(2e-2));
    }
    // Row and column normalisation must BOTH run: a matrix normalised only by
    // rows has column sums that differ from each other.
    {
        float lo = 1e9f, hi = 0.0f;
        for (int c = 0; c < kHcMult; ++c)
        {
            float cs = 0.0f;
            for (int r = 0; r < kHcMult; ++r) cs += comb[r * kHcMult + c];
            lo = std::min(lo, cs);
            hi = std::max(hi, cs);
        }
        REQUIRE(hi - lo < 5e-2f);
    }
    for (int d = 0; d < kHcMult; ++d)
        REQUIRE(post[d] == Approx(ref_post[d]).margin(1e-5));
    for (int i = 0; i < kHidden; ++i)
        REQUIRE(collapsed[i] == Approx(ref_collapsed[i]).margin(1e-5));
    for (int i = 0; i < kHcMult * kHcMult; ++i)
        REQUIRE(comb[i] == Approx(ref_comb[i]).margin(1e-5));

    // The mix step: h_new = post ⊗ out + comb @ h.
    std::vector<float> out(kHidden);
    for (auto &v : out) v = dist(rng);
    float *d_out = Upload(out);
    float *d_mixed = nullptr;
    REQUIRE(cudaMalloc(&d_mixed, n * sizeof(float)) == cudaSuccess);
    REQUIRE(LaunchXingMix(d_hidden, d_out, d_post, d_comb, kHcMult, kHidden,
                          d_mixed, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> ref_mixed;
    ReferenceMix(hidden, out, ref_post, ref_comb, ref_mixed);
    const auto mixed = Download(d_mixed, n);
    for (int i = 0; i < n; ++i)
        REQUIRE(mixed[i] == Approx(ref_mixed[i]).margin(1e-5));

    // In-place aliasing (hidden_new == hidden) must produce the same result:
    // the decode loop reuses one scratch buffer across both mHC blocks.
    float *d_hidden_inplace = Upload(hidden);
    REQUIRE(LaunchXingMix(d_hidden_inplace, d_out, d_post, d_comb, kHcMult,
                          kHidden, d_hidden_inplace, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto mixed_inplace = Download(d_hidden_inplace, n);
    for (int i = 0; i < n; ++i)
        REQUIRE(mixed_inplace[i] == Approx(ref_mixed[i]).margin(1e-5));

    cudaFree(d_hidden); cudaFree(d_fn); cudaFree(d_base); cudaFree(d_scale);
    cudaFree(d_collapsed); cudaFree(d_post); cudaFree(d_comb); cudaFree(d_out);
    cudaFree(d_mixed); cudaFree(d_hidden_inplace);
    cudaStreamDestroy(stream);
}

TEST_CASE("Xing unweighted RMSNorm covers all four streams at once",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const int n = kHcMult * kHidden;
    std::vector<float> x(n);
    for (auto &v : x) v = dist(rng);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_x = Upload(x);
    float *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, n * sizeof(float)) == cudaSuccess);
    REQUIRE(LaunchXingUnweightedRmsNorm4(d_x, d_out, kHcMult, kHidden, kEps,
                                         stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto out = Download(d_out, n);

    // One scalar for the CONCATENATION, not one per stream: this is what makes
    // the mHC collapse a single gate over all four streams.
    double sumsq = 0.0;
    for (float v : x) sumsq += static_cast<double>(v) * v;
    const float inv = static_cast<float>(1.0 / std::sqrt(sumsq / n + kEps));
    for (int i = 0; i < n; ++i)
        REQUIRE(out[i] == Approx(x[i] * inv).margin(1e-6));
    double norm = 0.0;
    for (float v : out) norm += static_cast<double>(v) * v;
    REQUIRE(std::sqrt(norm / n) == Approx(1.0f).margin(1e-4));

    cudaFree(d_x); cudaFree(d_out);
    cudaStreamDestroy(stream);
}

TEST_CASE("Xing sigmoid noaux_tc router matches the host reference",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> weight(static_cast<size_t>(kExperts) * kHidden);
    for (auto &v : weight) v = __float2bfloat16(dist(rng));
    std::vector<float> bias(kExperts);
    for (auto &v : bias) v = dist(rng) * 0.05f;
    std::vector<float> x(kHidden);
    for (auto &v : x) v = dist(rng);

    std::vector<float> ref_w;
    std::vector<int> ref_ids;
    ReferenceRouter(weight, bias, x, ref_w, ref_ids);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_w = Upload(weight);
    auto *d_b = Upload(bias);
    auto *d_x = Upload(x);
    float *d_ow = nullptr;
    int32_t *d_oe = nullptr;
    REQUIRE(cudaMalloc(&d_ow, 16 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_oe, 16 * sizeof(int32_t)) == cudaSuccess);

    REQUIRE(LaunchXingRouter(d_w, d_b, d_x, kExperts, kHidden, kTopK, true,
                             2.0f, d_ow, d_oe, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto ow = Download(d_ow, 16);
    std::vector<int32_t> oe(16, -1);
    cudaMemcpy(oe.data(), d_oe, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int k = 0; k < kTopK; ++k)
    {
        REQUIRE(oe[k] == ref_ids[k]);
        REQUIRE(ow[k] == Approx(ref_w[k]).margin(1e-6));
    }
    // No duplicates among the selected experts.
    for (int i = 0; i < kTopK; ++i)
        for (int j = i + 1; j < kTopK; ++j) REQUIRE(oe[i] != oe[j]);
    // Unused lanes are defined (the caller's buffers are 16 wide).
    for (int k = kTopK; k < 16; ++k)
    {
        REQUIRE(ow[k] == 0.0f);
        REQUIRE(oe[k] == -1);
    }
    // Weights are normalised then scaled by routed_scaling_factor.
    float sum = 0.0f;
    for (int k = 0; k < kTopK; ++k) sum += ow[k];
    REQUIRE(sum == Approx(2.0f).margin(1e-5));

    cudaFree(d_w); cudaFree(d_b); cudaFree(d_x); cudaFree(d_ow); cudaFree(d_oe);
    cudaStreamDestroy(stream);
}