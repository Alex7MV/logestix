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

#include "test_kernel_budget.h"

#include <lsxcommon/lsx_inference_nvfp4.h>

#include <cstring>

#include "lsxcommon/xing_moe_inference.h"
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
constexpr int kMixSmall = (2 + kHcMult) * kHcMult;
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
    const int hidden_dim = static_cast<int>(hidden.size()) / kHcMult;
    const int n = kHcMult * hidden_dim;
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

    collapsed.assign(static_cast<size_t>(hidden_dim), 0.0f);
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
    for (int j = 0; j < hidden_dim; ++j)
    {
        float acc = 0.0f;
        for (int d = 0; d < kHcMult; ++d)
            acc += pre[d] * hidden[d * hidden_dim + j];
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
    float *d_collapsed = nullptr, *d_post = nullptr, *d_comb = nullptr,
          *d_rows = nullptr;
    REQUIRE(cudaMalloc(&d_collapsed, kHidden * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_post, kHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comb, kHcMult * kHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_rows, (kMixSmall + 8) * sizeof(float)) == cudaSuccess);

    REQUIRE(LaunchXingHyperConnection(d_hidden, d_fn, d_base, d_scale, kHcMult,
                                      kHidden, kEps, kSinkhorn, kClampMin,
                                      kClampMax, kPostGain, d_collapsed, d_post,
                                      d_comb, d_rows, stream) == cudaSuccess);
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

// Host reference for Xing4_0TopkRouter (modeling_xing4_0.py): sigmoid scores,
// noaux_tc bias steering the CHOICE only, lowest-index tie-break.
void ReferenceRouter(const std::vector<__nv_bfloat16> &weight,
                     const std::vector<float> &bias,
                     const std::vector<__nv_bfloat16> &x, int experts,
                     int hidden, int top_k, bool norm_topk_prob,
                     double routed_scale, std::vector<float> &out_w,
                     std::vector<int32_t> &out_ids)
{
    const auto bf = [](__nv_bfloat16 v) { return __bfloat162float(v); };
    std::vector<float> score(static_cast<size_t>(experts));
    for (int e = 0; e < experts; ++e)
    {
        float acc = 0.0f;
        for (int i = 0; i < hidden; ++i)
            acc += bf(weight[static_cast<size_t>(e) * hidden + i]) * bf(x[i]);
        score[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-acc));
    }
    std::vector<bool> taken(static_cast<size_t>(experts), false);
    out_w.assign(16, 0.0f);
    out_ids.assign(16, -1);
    float wsum = 0.0f;
    for (int k = 0; k < top_k; ++k)
    {
        int best = -1;
        float best_score = 0.0f;
        for (int e = 0; e < experts; ++e)
        {
            if (taken[static_cast<size_t>(e)]) continue;
            const float sv = score[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)];
            if (best < 0 || sv > best_score)
            {
                best = e;
                best_score = sv;
            }
        }
        taken[static_cast<size_t>(best)] = true;
        out_ids[static_cast<size_t>(k)] = best;
        out_w[static_cast<size_t>(k)] = score[static_cast<size_t>(best)];
        wsum += score[static_cast<size_t>(best)];
    }
    const float denom = norm_topk_prob ? 1.0f / (wsum + 1e-20f) : 1.0f;
    for (int k = 0; k < top_k; ++k)
        out_w[static_cast<size_t>(k)] *= denom * static_cast<float>(routed_scale);
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
    // The activation reaches the router as BF16: the device-independent neutral
    // GEMV dispatches matching dtypes only, so the router's input is converted.
    std::vector<__nv_bfloat16> x_bf(kHidden);
    for (size_t i = 0; i < x.size(); ++i)
        x_bf[i] = __float2bfloat16(x[i]);

    std::vector<float> ref_w;
    std::vector<int32_t> ref_ids;
    ReferenceRouter(weight, bias, x_bf, kExperts, kHidden, kTopK, true, 2.0,
                    ref_w, ref_ids);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_w = Upload(weight);
    auto *d_b = Upload(bias);
    auto *d_x = Upload(x_bf);
    float *d_ow = nullptr;
    int32_t *d_oe = nullptr;
    float *d_scores = nullptr;
    REQUIRE(cudaMalloc(&d_ow, 16 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_oe, 16 * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scores, kExperts * sizeof(float)) == cudaSuccess);
    REQUIRE(LaunchXingRouter(d_w, d_b, d_x, kExperts, kHidden, kTopK, true, 2.0f,
                             d_ow, d_oe, d_scores, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto ow = Download(d_ow, 16);
    std::vector<int32_t> oe(16, -1);
    cudaMemcpy(oe.data(), d_oe, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int k = 0; k < kTopK; ++k)
    {
        REQUIRE(oe[k] == ref_ids[static_cast<size_t>(k)]);
        REQUIRE(ow[k] == Approx(ref_w[static_cast<size_t>(k)]).margin(1e-6));
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

    cudaFree(d_ow); cudaFree(d_oe);
    cudaStreamDestroy(stream);
}

// The router at the PRODUCTION shape (64 experts, hidden 3584). The case above
// uses 16 experts of 64 dims, where the launch geometry and the serial top-k
// scan cost nothing to see.
TEST_CASE("Xing router matches the host reference at the 29B shape",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    constexpr int kE = 64, kH = 3584;
    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> weight(static_cast<size_t>(kE) * kH);
    for (auto &v : weight) v = __float2bfloat16(dist(rng) * 0.05f);
    std::vector<float> bias(kE);
    for (auto &v : bias) v = dist(rng) * 0.05f;
    std::vector<__nv_bfloat16> x(kH);
    for (auto &v : x) v = __float2bfloat16(dist(rng));

    std::vector<float> ref_w;
    std::vector<int32_t> ref_ids;
    ReferenceRouter(weight, bias, x, kE, kH, kTopK, true, 2.0, ref_w, ref_ids);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_w = Upload(weight);
    auto *d_b = Upload(bias);
    auto *d_x = Upload(x);
    float *d_ow = nullptr;
    int32_t *d_oe = nullptr;
    REQUIRE(cudaMalloc(&d_ow, 16 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_oe, 16 * sizeof(int32_t)) == cudaSuccess);
    float *d_scores = nullptr;
    REQUIRE(cudaMalloc(&d_scores, kE * sizeof(float)) == cudaSuccess);
    REQUIRE(LaunchXingRouter(d_w, d_b, d_x, kE, kH, kTopK, true, 2.0f, d_ow, d_oe,
                             d_scores, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto ow = Download(d_ow, 16);
    std::vector<int32_t> oe(16, -1);
    cudaMemcpy(oe.data(), d_oe, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int k = 0; k < kTopK; ++k)
    {
        REQUIRE(oe[k] == ref_ids[static_cast<size_t>(k)]);
        REQUIRE(ow[k] == Approx(ref_w[static_cast<size_t>(k)]).margin(1e-6));
    }
    cudaFree(d_ow); cudaFree(d_oe);
    cudaStreamDestroy(stream);
}

// Latency budget at the production shape. nsys measured 129 us x 38 launches =
// ~4.9 ms per token - the second-largest serial cost after the mHC block, for a
// [64, 3584] GEMV plus a 4-deep selection over 64 candidates.
TEST_CASE("Xing router stays inside its latency budget at the 29B shape",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    constexpr int kE = 64, kH = 3584;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> weight(static_cast<size_t>(kE) * kH);
    for (auto &v : weight) v = __float2bfloat16(dist(rng) * 0.05f);
    std::vector<float> bias(kE, 0.01f);
    std::vector<__nv_bfloat16> x(kH);
    for (auto &v : x) v = __float2bfloat16(dist(rng));
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto *d_w = Upload(weight);
    auto *d_b = Upload(bias);
    auto *d_x = Upload(x);
    float *d_ow = nullptr;
    int32_t *d_oe = nullptr;
    REQUIRE(cudaMalloc(&d_ow, 16 * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_oe, 16 * sizeof(int32_t)) == cudaSuccess);
    float *d_scores = nullptr;
    REQUIRE(cudaMalloc(&d_scores, kE * sizeof(float)) == cudaSuccess);
    const auto launch = [&] {
        return LaunchXingRouter(d_w, d_b, d_x, kE, kH, kTopK, true, 2.0f, d_ow,
                                d_oe, d_scores, stream);
    };
    for (int i = 0; i < 20; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaEvent_t a, b;
    REQUIRE(cudaEventCreate(&a) == cudaSuccess);
    REQUIRE(cudaEventCreate(&b) == cudaSuccess);
    const int iters = 200;
    REQUIRE(cudaEventRecord(a, stream) == cudaSuccess);
    for (int i = 0; i < iters; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaEventRecord(b, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, a, b) == cudaSuccess);
    const double us = ms * 1000.0 / iters;
    INFO("router launch " << us << " us at 64 experts x " << kH);
#if defined(NDEBUG)
    kernel_budget::Require("xing.router.decode", us, kernel_budget::Kind::kMax,
                           "projection one block per expert (64 blocks x 32"
                           " lanes) + top-k block; 64 experts x hidden 3584");
#else
    WARN("latency budget is a Release-only gate; measured " << us << " us");
#endif
    cudaEventDestroy(a); cudaEventDestroy(b); cudaStreamDestroy(stream);
}

// Interleaved YaRN rotation, per head. Every head gets its OWN rope angle
// content (cos/sin differ per head via the phase), so a launcher that applies
// one shared element offset to all heads - the shape of bug this test was
// written for - shows up as "only one head moved".
TEST_CASE("Xing interleaved rope rotates every head in place", "[xing][kernels]")
{
    if (!HasDevice()) return;
    constexpr int kHeadsT = 6;
    constexpr int kNopeT = 8;
    constexpr int kPairs = 5;                 // 10 rope dims per head
    constexpr int kQkT = kNopeT + 2 * kPairs; // 18
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    std::vector<float> q(static_cast<size_t>(kHeadsT) * kQkT, 0.0f);
    for (int h = 0; h < kHeadsT; ++h)
        for (int j = 0; j < kPairs; ++j)
        {
            q[h * kQkT + kNopeT + 2 * j] = static_cast<float>(h + 1) * 0.5f + 0.01f * j;
            q[h * kQkT + kNopeT + 2 * j + 1] = 0.25f * static_cast<float>(j + 1);
        }
    std::vector<float> cos_t(kPairs), sin_t(kPairs);
    for (int j = 0; j < kPairs; ++j)
    {
        const float a = 0.3f * static_cast<float>(j + 1);
        cos_t[j] = std::cos(a);
        sin_t[j] = std::sin(a);
    }
    std::vector<float> krot(2 * kPairs);
    for (int j = 0; j < 2 * kPairs; ++j) krot[j] = 0.5f * static_cast<float>(j + 1);

    float *d_q = nullptr, *d_krot = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    REQUIRE(cudaMalloc(&d_q, q.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_krot, krot.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_cos, cos_t.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sin, sin_t.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_krot, krot.data(), krot.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(LaunchXingRopeQ(d_q, kHeadsT, kQkT, kPairs, d_cos, d_sin, stream) ==
            cudaSuccess);
    REQUIRE(LaunchXingRopeK(d_krot, kPairs, d_cos, d_sin, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto got = Download(d_q, static_cast<int>(q.size()));
    // Host reference, straight from apply_rotary_pos_emb_interleave.
    for (int h = 0; h < kHeadsT; ++h)
    {
        for (int j = 0; j < kPairs; ++j)
        {
            const float a = q[static_cast<size_t>(h) * kQkT + kNopeT + 2 * j];
            const float b = q[static_cast<size_t>(h) * kQkT + kNopeT + 2 * j + 1];
            REQUIRE(got[static_cast<size_t>(h) * kQkT + kNopeT + 2 * j] ==
                    Approx(a * cos_t[j] - b * sin_t[j]).margin(1e-6));
            REQUIRE(got[static_cast<size_t>(h) * kQkT + kNopeT + 2 * j + 1] ==
                    Approx(a * sin_t[j] + b * cos_t[j]).margin(1e-6));
        }
        // The nope subspace is untouched.
        for (int i = 0; i < kNopeT; ++i)
            REQUIRE(got[static_cast<size_t>(h) * kQkT + i] ==
                    Approx(q[static_cast<size_t>(h) * kQkT + i]));
    }
    const auto gk = Download(d_krot, 2 * kPairs);
    for (int j = 0; j < kPairs; ++j)
    {
        const float a = krot[2 * j], b = krot[2 * j + 1];
        REQUIRE(gk[2 * j] == Approx(a * cos_t[j] - b * sin_t[j]).margin(1e-6));
        REQUIRE(gk[2 * j + 1] == Approx(a * sin_t[j] + b * cos_t[j]).margin(1e-6));
    }

    cudaFree(d_q); cudaFree(d_krot); cudaFree(d_cos); cudaFree(d_sin);
    cudaStreamDestroy(stream);
}

// The mHC block at the REAL 29B shape (hidden 3584, hc_mult 4, mix 24) - the
// small brute-forceable shape above cannot see the launch geometry cost, which
// is what made this family 11x slower than the tuned ones.
namespace real_shape {

constexpr int kRealHidden = 3584;   // Xing4.0-29B hidden_size
constexpr int kRealHcMult = 4;      // hc_mult
constexpr int kRealMix = (2 + kRealHcMult) * kRealHcMult;

struct Inputs
{
    std::vector<float> streams;
    std::vector<__nv_bfloat16> fn, base;
    std::vector<float> scale = {0.7f, -0.4f, 1.3f};
};

Inputs MakeInputs(unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Inputs in;
    in.streams.resize(static_cast<size_t>(kRealHcMult) * kRealHidden);
    for (auto &v : in.streams) v = dist(rng);
    in.fn.resize(static_cast<size_t>(kRealMix) * kRealHcMult * kRealHidden);
    for (auto &v : in.fn) v = __float2bfloat16(dist(rng) * 0.3f);
    in.base.resize(kRealMix);
    for (auto &v : in.base) v = __float2bfloat16(dist(rng) * 0.1f);
    return in;
}

struct Outputs
{
    std::vector<float> collapsed, post, comb;
};

Outputs RunDevice(const Inputs &in, cudaStream_t stream)
{
    const int n = kRealHcMult * kRealHidden;
    auto *d_streams = Upload(in.streams);
    auto *d_fn = Upload(in.fn);
    auto *d_base = Upload(in.base);
    auto *d_scale = Upload(in.scale);
    float *d_collapsed = nullptr, *d_post = nullptr, *d_comb = nullptr,
          *d_rows = nullptr;
    REQUIRE(cudaMalloc(&d_collapsed, kRealHidden * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_post, kRealHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comb, kRealHcMult * kRealHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_rows, (kRealMix + 8) * sizeof(float)) == cudaSuccess);
    REQUIRE(LaunchXingHyperConnection(
                d_streams, d_fn, d_base, d_scale, kRealHcMult, kRealHidden, kEps, kSinkhorn,
                kClampMin, kClampMax, kPostGain, d_collapsed, d_post, d_comb,
                d_rows, stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    Outputs out;
    out.collapsed = Download(d_collapsed, kRealHidden);
    out.post = Download(d_post, kRealHcMult);
    out.comb = Download(d_comb, kRealHcMult * kRealHcMult);
    cudaFree(d_collapsed);
    cudaFree(d_post);
    cudaFree(d_comb);
    return out;
}

} // namespace real_shape

// Parity at the production shape. The rewrite that follows changes the partial
// -sum grouping inside the hc_fn projection, so this is the gate that keeps the
// mHC math honest at hidden=3584 rather than only at hidden=64.
TEST_CASE("Xing mHC matches the host reference at the 29B shape",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    using namespace real_shape;
    const Inputs in = MakeInputs(4242);
    std::vector<float> ref_collapsed, ref_post, ref_comb;
    ReferenceHyperConn(in.streams, in.fn, in.base, in.scale, ref_collapsed,
                       ref_post, ref_comb);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const Outputs got = RunDevice(in, stream);
    for (size_t i = 0; i < ref_collapsed.size(); ++i)
        REQUIRE(got.collapsed[i] == Approx(ref_collapsed[i]).margin(2e-5));
    for (size_t i = 0; i < ref_post.size(); ++i)
        REQUIRE(got.post[i] == Approx(ref_post[i]).margin(2e-5));
    for (size_t i = 0; i < ref_comb.size(); ++i)
        REQUIRE(got.comb[i] == Approx(ref_comb[i]).margin(2e-5));
    cudaStreamDestroy(stream);
}

// Latency budget at the production shape. nsys on the real 29B checkpoint put
// 80% of ALL kernel time in this one launch: 24 sequential 256-thread tree
// reductions (192 dependent barriers) inside a SINGLE block, i.e. one SM of 170.
// 80 launches per token x ~229 us = ~41 ms of a 69 ms token.
TEST_CASE("Xing mHC stays inside its latency budget at the 29B shape",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    using namespace real_shape;
    const Inputs in = MakeInputs(99);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    // Stage the buffers ONCE: the budget is the LAUNCH, so the 688 KiB hc_fn
    // upload and the readback must not be inside the timed region (doing that
    // measures cudaMemcpy, not the kernel - the first version of this test
    // reported 180 us for a 60 us kernel).
    auto *d_streams = Upload(in.streams);
    auto *d_fn = Upload(in.fn);
    auto *d_base = Upload(in.base);
    auto *d_scale = Upload(in.scale);
    float *d_collapsed = nullptr, *d_post = nullptr, *d_comb = nullptr;
    REQUIRE(cudaMalloc(&d_collapsed, kRealHidden * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_post, kRealHcMult * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_comb, kRealHcMult * kRealHcMult * sizeof(float)) ==
            cudaSuccess);
    float *d_rows = nullptr;
    REQUIRE(cudaMalloc(&d_rows, kRealMix * sizeof(float)) == cudaSuccess);
    constexpr int mhc_iters = kSinkhorn;
    const auto launch = [&] {
        return LaunchXingHyperConnection(
            d_streams, d_fn, d_base, d_scale, kRealHcMult, kRealHidden, kEps,
            mhc_iters, kClampMin, kClampMax, kPostGain,
            d_collapsed, d_post, d_comb, d_rows, stream);
    };
    for (int i = 0; i < 20; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    cudaEvent_t a, b;
    REQUIRE(cudaEventCreate(&a) == cudaSuccess);
    REQUIRE(cudaEventCreate(&b) == cudaSuccess);
    const int iters = 200;
    REQUIRE(cudaEventRecord(a, stream) == cudaSuccess);
    for (int i = 0; i < iters; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaEventRecord(b, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, a, b) == cudaSuccess);
    const double us = ms * 1000.0 / iters;
    INFO("mHC launch " << us << " us at hidden=" << kRealHidden);
#if defined(NDEBUG)
    // Release-only gate: a debug build's device code is not the shipped one.
    // Budget is per-gpu (tests/perf/kernel_budgets.json): these three numbers
    // were calibrated on an RTX 5090, and the helper's own header records why a
    // hard-coded kernel budget carries no information on another machine.
    kernel_budget::Require("xing.mhc.decode", us, kernel_budget::Kind::kMax,
                           "3 launches, hidden=3584, hc_mult=4 (proj 24 blocks +"
                           " warp-Sinkhorn gates + wide collapse)");
#else
    WARN("latency budget is a Release-only gate; measured " << us << " us");
#endif
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_collapsed);
    cudaFree(d_post);
    cudaFree(d_comb);
    cudaStreamDestroy(stream);
}

// Attention at a long context. The decode step's cost is dominated by this:
// us/token = 14.3 ms + 16.5 ms per 1000 cached positions on the real 29B
// checkpoint (64.2 t/s at 128 tokens, 32.0 at 2048), and the whole context part
// is xing_attn_kernel - one block per head, so 32 of 170 SMs, with the softmax
// and the value accumulation serial over t <= pos.
namespace attn_shape {

constexpr int kHeads = 32;
constexpr int kQk = 192;   // 128 nope + 64 rope
constexpr int kVh = 128;
constexpr int kPos = 2047;
constexpr int kTestAttnSplits = 8; // canonical 2048-token context
constexpr float kScale = 0.14467962580268925f;  // mscale^2 / sqrt(192)

struct Cache
{
    std::vector<float> q, k, v;
    Cache(unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        q.resize(static_cast<size_t>(kHeads) * kQk);
        k.resize(static_cast<size_t>(kPos + 1) * kHeads * kQk);
        v.resize(static_cast<size_t>(kPos + 1) * kHeads * kVh);
        for (auto &x : q) x = dist(rng);
        for (auto &x : k) x = dist(rng);
        for (auto &x : v) x = dist(rng);
    }
};

// Xing4_0Attention's eager path: softmax(qk^T * scaling) @ v over t <= pos.
std::vector<float> ReferenceAttention(const Cache &c)
{
    std::vector<float> out(static_cast<size_t>(kHeads) * kVh, 0.0f);
    for (int h = 0; h < kHeads; ++h)
    {
        std::vector<float> logits(kPos + 1);
        float mx = -INFINITY;
        for (int t = 0; t <= kPos; ++t)
        {
            const float *kt =
                c.k.data() + (static_cast<size_t>(t) * kHeads + h) * kQk;
            const float *qt = c.q.data() + static_cast<size_t>(h) * kQk;
            float acc = 0.0f;
            for (int i = 0; i < kQk; ++i) acc += qt[i] * kt[i];
            logits[static_cast<size_t>(t)] = acc * kScale;
            mx = std::max(mx, logits[static_cast<size_t>(t)]);
        }
        float sum = 0.0f;
        for (auto &l : logits)
        {
            l = std::exp(l - mx);
            sum += l;
        }
        for (int d = 0; d < kVh; ++d)
        {
            float acc = 0.0f;
            for (int t = 0; t <= kPos; ++t)
            {
                const float w = logits[static_cast<size_t>(t)] / sum;
                acc += w * c.v[(static_cast<size_t>(t) * kHeads + h) * kVh + d];
            }
            out[static_cast<size_t>(h) * kVh + d] = acc;
        }
    }
    return out;
}

struct Device
{
    float *q = nullptr, *k = nullptr, *v = nullptr, *scores = nullptr,
          *out = nullptr;
    float *m = nullptr;     // [heads * splits]
    float *l = nullptr;
    float *part = nullptr;  // [heads * splits * vh]
};

Device Upload(const Cache &c, int capacity)
{
    Device d;
    REQUIRE(cudaMalloc(&d.q, c.q.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d.k, c.k.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d.v, c.v.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d.scores,
                       static_cast<size_t>(capacity) * kHeads * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d.out, static_cast<size_t>(kHeads) * kVh * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d.m, static_cast<size_t>(kHeads) * kTestAttnSplits * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d.l, static_cast<size_t>(kHeads) * kTestAttnSplits * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&d.part,
                       static_cast<size_t>(kHeads) * kTestAttnSplits * kVh *
                           sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d.q, c.q.data(), c.q.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d.k, c.k.data(), c.k.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d.v, c.v.data(), c.v.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

void Free(Device &d)
{
    cudaFree(d.q); cudaFree(d.k); cudaFree(d.v); cudaFree(d.scores);
    cudaFree(d.out);
    cudaFree(d.m);
    cudaFree(d.l);
    cudaFree(d.part);
}

} // namespace attn_shape

TEST_CASE("Xing attention matches the host reference at a 2k context",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    using namespace attn_shape;
    const Cache c(20261004);
    const int capacity = kPos + 1;
    const auto ref = ReferenceAttention(c);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    Device d = Upload(c, capacity);
    // Both forms have to match the eager reference: the decode step always runs
    // the split-K one, but `splits` collapses to the single-block kernel at short
    // contexts (one chunk), so the fallback is on the live path too.
    for (const int splits : {kTestAttnSplits, 1})
    {
        REQUIRE(LaunchXingAttention(d.q, d.k, d.v, kPos, kHeads, kQk, kVh,
                                    kScale, d.scores, d.out, capacity, kPos + 1,
                                    splits, d.m, d.l, d.part, nullptr,
                                    stream) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        const auto got = Download(d.out, kHeads * kVh);
        INFO("splits=" << splits);
        for (size_t i = 0; i < ref.size(); ++i)
            REQUIRE(got[i] == Approx(ref[i]).margin(1e-5));
    }
    Free(d);
    cudaStreamDestroy(stream);
}

TEST_CASE("Xing attention stays inside its latency budget at a 2k context",
          "[xing][kernels]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    using namespace attn_shape;
    const Cache c(11);
    const int capacity = kPos + 1;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    Device d = Upload(c, capacity);
    const auto launch = [&] {
        return LaunchXingAttention(d.q, d.k, d.v, kPos, kHeads, kQk, kVh,
                                   kScale, d.scores, d.out, capacity, kPos + 1,
                                   kTestAttnSplits, d.m, d.l, d.part, nullptr,
                                   stream);
    };
    for (int i = 0; i < 3; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaEvent_t a, b;
    REQUIRE(cudaEventCreate(&a) == cudaSuccess);
    REQUIRE(cudaEventCreate(&b) == cudaSuccess);
    const int iters = 20;
    REQUIRE(cudaEventRecord(a, stream) == cudaSuccess);
    for (int i = 0; i < iters; ++i) REQUIRE(launch() == cudaSuccess);
    REQUIRE(cudaEventRecord(b, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, a, b) == cudaSuccess);
    const double us = ms * 1000.0 / iters;
    INFO("attention launch " << us << " us at pos=" << kPos);
#if defined(NDEBUG)
    // 90 us was the split-K target: heads x kTestAttnSplits = 256 blocks over
    // 170 SMs, against 32 blocks (32 of 170 SMs) for the single-block form,
    // which measured 209 us at this shape.
    kernel_budget::Require("xing.attention.decode", us,
                           kernel_budget::Kind::kMax,
                           "split-K, 32 heads x 8 position chunks at pos=2047");
#else
    WARN("latency budget is a Release-only gate; measured " << us << " us");
#endif
    cudaEventDestroy(a); cudaEventDestroy(b);
    Free(d);
    cudaStreamDestroy(stream);
}

namespace {

std::vector<__half> DownloadHalves(__half *d, int n)
{
    std::vector<__half> out(static_cast<size_t>(n));
    REQUIRE(cudaMemcpy(out.data(), d, sizeof(__half) * static_cast<size_t>(n),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return out;
}

} // namespace

// The device-table expert batch is the precondition for CUDA-graph capture of a
// decode step: it resolves the routed ids on the device, so no D2H and no
// cudaStreamSynchronize sits inside a layer. It is a different code path from the
// by-value batch, so "different path" is exactly what must NOT mean "different
// numbers": every row of both outputs is compared bit for bit.
TEST_CASE("NVFP4 device-table expert batch is bit-identical to the by-value batch",
          "[nvfp4][fast]")
{
    if (!HasDevice()) SKIP("no CUDA device");
    // The streaming ids kernel only dispatches when BOTH toggles are on, and a
    // test binary starts with them off - so without this the case would only ever
    // cover the host-side fallback and the kernel itself would ship untested.
    lsxcommon::lsx_nvfp4::LsxFastMathScope fast(true, true);
    // A shape that takes the streaming kernel (K % 32 == 0, K >= 256) and a
    // small one that must fall back, so both branches of the entry point are
    // covered; the fallback resolves ids on the host by design.
    for (const int K : {512, 96})
    {
        constexpr int kE = 8, kCount = 4, kN = 64, kLayers = 3, kLayer = 2;
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
        const size_t packed_bytes = static_cast<size_t>(kN) * (K / 2);
        const size_t scale_bytes = static_cast<size_t>(kN) * (K / 16);
        std::vector<std::vector<uint8_t>> packed(kE), scales(kE);
        std::vector<float> gs(kE);
        for (int e = 0; e < kE; ++e)
        {
            packed[e].resize(packed_bytes);
            scales[e].resize(scale_bytes);
            for (auto &v : packed[e]) v = static_cast<uint8_t>(rng() & 0xff);
            for (auto &v : scales[e])
                v = static_cast<uint8_t>(0x30 + (rng() % 4));
            gs[e] = 0.5f + 0.1f * e;
        }
        std::vector<__half> x(static_cast<size_t>(K));
        for (auto &v : x) v = __float2half(dist(rng));
        const std::vector<int32_t> ids = {5, 1, 7, 2};

        cudaStream_t stream = nullptr;
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
        auto *d_x = Upload(x);
        auto *d_ids = Upload(ids);
        // kLayers layers of rows, and the dispatch is asked for layer kLayer: the
        // first version of the kernel ignored table_offset and read layer 0,
        // which in the real model is a DENSE layer whose rows are all zero - so it
        // dereferenced a null packed pointer and faulted. With kLayer != 0 and a
        // distinct row per (layer, expert) this case now fails if the offset is
        // dropped.
        std::vector<lsxcommon::lsx_nvfp4::LsxExpertPtrRow> rows(kLayers * kE);
        std::vector<uint8_t *> d_packed(kLayers * kE), d_scales(kLayers * kE);
        for (int l = 0; l < kLayers; ++l)
        {
            for (int e = 0; e < kE; ++e)
            {
                const size_t at = static_cast<size_t>(l) * kE + e;
                REQUIRE(cudaMalloc(&d_packed[at], packed_bytes) == cudaSuccess);
                REQUIRE(cudaMalloc(&d_scales[at], scale_bytes) == cudaSuccess);
                // distinct bytes per (layer, expert) so a wrong layer is visible
                std::vector<uint8_t> p = packed[e];
                for (auto &v : p) v = static_cast<uint8_t>(v + 7 * l);
                REQUIRE(cudaMemcpy(d_packed[at], p.data(), packed_bytes,
                                    cudaMemcpyHostToDevice) == cudaSuccess);
                REQUIRE(cudaMemcpy(d_scales[at], scales[e].data(), scale_bytes,
                                    cudaMemcpyHostToDevice) == cudaSuccess);
                rows[at] = {d_packed[at], d_scales[at],
                            gs[e] * (1.0f + 0.25f * static_cast<float>(l))};
            }
        }
        auto *d_rows = Upload(rows);
        // by-value inputs
        std::vector<const uint8_t *> vp(kCount), vs(kCount);
        std::vector<float> vg(kCount);
        for (int i = 0; i < kCount; ++i)
        {
            const size_t at = static_cast<size_t>(kLayer) * kE +
                              static_cast<size_t>(ids[i]);
            vp[i] = d_packed[at];
            vs[i] = d_scales[at];
            vg[i] = rows[at].gs;
        }
        __half *y_a = nullptr, *y_b = nullptr;
        REQUIRE(cudaMalloc(&y_a, sizeof(__half) * kCount * kN) == cudaSuccess);
        REQUIRE(cudaMalloc(&y_b, sizeof(__half) * kCount * kN) == cudaSuccess);
        REQUIRE(cudaMemset(y_a, 0, sizeof(__half) * kCount * kN) == cudaSuccess);
        REQUIRE(cudaMemset(y_b, 0, sizeof(__half) * kCount * kN) == cudaSuccess);
        REQUIRE(lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, d_x, vp.data(), vs.data(), vg.data(), kCount, 0, y_a,
                    K, kN) == cudaSuccess);
        REQUIRE(lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatchIds(
                    stream, d_x, d_ids, d_rows, kLayer * kE, kE, kCount, 0, y_b,
                    K, kN) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto a = DownloadHalves(y_a, kCount * kN);
        const auto b = DownloadHalves(y_b, kCount * kN);
        INFO("K=" << K);
        REQUIRE(a.size() == b.size());
        REQUIRE(!a.empty());
        // Compare the raw 16 bits (memcmp), not the values: a same-value
        // comparison would hide a NaN payload difference, which is exactly what a
        // store-type change can introduce.
        REQUIRE(a.size() * sizeof(__half) == b.size() * sizeof(__half));
        REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0);
        // Sanity that the id really selects the expert: the same GEMV against a
        // DIFFERENT row must differ. Without this, a kernel that ignored d_ids
        // and always read row 0 would still pass the comparison above whenever
        // ids[0] happened to be 0 - and it is not.
        {
            const std::vector<int32_t> other = {0, 1, 7, 2};
            auto *d_other = Upload(other);
            __half *y_c = nullptr;
            REQUIRE(cudaMalloc(&y_c, sizeof(__half) * kCount * kN) ==
                    cudaSuccess);
            REQUIRE(cudaMemset(y_c, 0, sizeof(__half) * kCount * kN) ==
                    cudaSuccess);
            REQUIRE(lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatchIds(
                        stream, d_x, d_other, d_rows, kLayer * kE, kE, kCount, 0,
                        y_c, K, kN) == cudaSuccess);
            REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
            const auto c = DownloadHalves(y_c, kCount * kN);
            const bool differs =
                std::memcmp(b.data(), c.data(), b.size() * sizeof(__half)) != 0;
            if (K >= 256) REQUIRE(differs);
            cudaFree(y_c);
            cudaFree(d_other);
        }
        for (size_t i = 0; i < d_packed.size(); ++i)
        {
            cudaFree(d_packed[i]);
            cudaFree(d_scales[i]);
        }
        cudaFree(y_a);
        cudaFree(y_b);
        cudaFree(d_x);
        cudaFree(d_ids);
        cudaFree(d_rows);
        cudaStreamDestroy(stream);
    }
}
