// Recursive Flint numeric kernels (Model Ingestion Protocol stages 5/6):
// the learned-sink softmax, the sliding-window bounds, the sigmoid top-k router,
// rotary (applied only on sliding layers), row/per-head RMSNorm and the FP32
// routed+shared combine — each against a host reference. Label `inference`;
// SKIPs (4) when no CUDA device is present.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/flint_moe_device_math.h"
#include "lsxcommon/flint_moe_inference_attn.h"
#include "lsxcommon/flint_moe_kernels.h"

using namespace lsxcommon::flint;

namespace {

bool HaveCuda()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

std::vector<float> FPattern(int64_t n, float scale, int64_t seed)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t k = (i * 7 + seed * 13) % 23;
        v[static_cast<size_t>(i)] = static_cast<float>(k - 11) * scale;
    }
    return v;
}

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, sizeof(T) * v.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), sizeof(T) * v.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

template <typename T>
void FromDevice(const T *d, std::vector<T> &v)
{
    REQUIRE(cudaMemcpy(v.data(), d, sizeof(T) * v.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

} // namespace

TEST_CASE("flint sink softmax puts the sink only in the denominator",
          "[flint][kernels]")
{
    // A huge sink must swallow the whole mass: the value weights shrink toward
    // zero (the sink multiplies no value), while a tiny sink leaves the scores
    // untouched.
    const float scores[3] = {1.5f, -0.25f, 0.75f};
    std::vector<float> w;
    ref::SinkSoftmaxWeights(scores, 3, 40.0f, w);
    float sum = 0.0f;
    for (float x : w) sum += x;
    CHECK(sum < 1e-6f);

    ref::SinkSoftmaxWeights(scores, 3, -40.0f, w);
    sum = 0.0f;
    for (float x : w) sum += x;
    CHECK(sum == Catch::Approx(1.0f).margin(1e-6));

    // The softmax denominator includes exp(sink - max) exactly once.
    ref::SinkSoftmaxWeights(scores, 3, 0.5f, w);
    float m = 0.5f;
    for (float s : scores) m = std::max(m, s);
    double denom = std::exp(0.5 - m);
    for (float s : scores) denom += std::exp(s - m);
    for (int i = 0; i < 3; ++i)
        CHECK(w[static_cast<size_t>(i)] ==
              Catch::Approx(std::exp(scores[i] - m) / denom).margin(1e-7));
}

TEST_CASE("flint sliding window bounds", "[flint][kernels]")
{
    CHECK(ref::WindowStart(0, 513) == 0);
    CHECK(ref::WindowStart(512, 513) == 0);
    CHECK(ref::WindowStart(513, 513) == 1);
    CHECK(ref::WindowStart(2047, 513) == 1535);
    CHECK(ref::WindowStart(2047, 0) == 0);   // full-attention layer
}

TEST_CASE("flint decode attention matches the host reference",
          "[flint][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int heads = 4, kv_heads = 2, head_dim = 8, cap = 40;
    const int pos = 20;
    const float scale = 0.35f;
    auto qf = FPattern(static_cast<int64_t>(heads) * head_dim, 0.2f, 1);
    auto kf = FPattern(static_cast<int64_t>(cap) * kv_heads * head_dim, 0.2f, 2);
    auto vf = FPattern(static_cast<int64_t>(cap) * kv_heads * head_dim, 0.3f, 3);
    auto sf = FPattern(heads, 0.9f, 4);

    std::vector<__half> q(qf.size()), k(kf.size()), v(vf.size()), s(sf.size());
    for (size_t i = 0; i < qf.size(); ++i) q[i] = H(qf[i]);
    for (size_t i = 0; i < kf.size(); ++i) k[i] = H(kf[i]);
    for (size_t i = 0; i < vf.size(); ++i) v[i] = H(vf[i]);
    for (size_t i = 0; i < sf.size(); ++i) s[i] = H(sf[i]);

    // The kernel reads FP16 inputs; the reference consumes the same rounded
    // values so the comparison isolates the arithmetic, not the storage.
    std::vector<float> qr(q.size()), kr(k.size()), vr(v.size()), sr(s.size());
    for (size_t i = 0; i < q.size(); ++i) qr[i] = F(q[i]);
    for (size_t i = 0; i < k.size(); ++i) kr[i] = F(k[i]);
    for (size_t i = 0; i < v.size(); ++i) vr[i] = F(v[i]);
    for (size_t i = 0; i < s.size(); ++i) sr[i] = F(s[i]);

    __half *dq = ToDevice(q), *dk = ToDevice(k), *dv = ToDevice(v),
           *ds = ToDevice(s);
    __half *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, sizeof(__half) * q.size()) == cudaSuccess);

    for (const int window : {0, 6, 513}) {
        REQUIRE(RunFlintAttnDecode(nullptr, dq, pos, ds, heads, kv_heads,
                                   head_dim, window, scale, dk, dv,
                                   dout) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__half> out(q.size());
        FromDevice(dout, out);
        std::vector<float> want(q.size());
        ref::DecodeAttention(qr.data(), heads, kv_heads, head_dim, window, pos,
                             scale, sr.data(), kr.data(), vr.data(),
                             want.data());
        for (size_t i = 0; i < out.size(); ++i)
            CHECK(F(out[i]) == Catch::Approx(want[i]).margin(2e-3));
    }

    // Sink dominance: with a huge sink the output must collapse toward zero
    // rather than reproduce the value average (the bug the vendor's FA4 patch
    // fixed).
    {
        std::vector<__half> big(s.size());
        for (size_t i = 0; i < big.size(); ++i) big[i] = H(80.0f);
        __half *dbig = ToDevice(big);
        REQUIRE(RunFlintAttnDecode(nullptr, dq, pos, dbig, heads, kv_heads,
                                   head_dim, 0, scale, dk, dv,
                                   dout) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__half> out(q.size());
        FromDevice(dout, out);
        for (auto x : out) CHECK(std::fabs(F(x)) < 1e-4f);
        cudaFree(dbig);
    }

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(ds);
    cudaFree(dout);
}

TEST_CASE("flint router is sigmoid top-k with renormalized weights",
          "[flint][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int E = 12, K = 4, Hd = 16;
    auto wf = FPattern(static_cast<int64_t>(E) * Hd, 0.08f, 5);
    auto xf = FPattern(Hd, 0.15f, 6);
    std::vector<__half> x(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) x[i] = H(xf[i]);

    float *dw = ToDevice(wf);
    __half *dx = ToDevice(x);
    FlintRoutedSlot *d_slots = nullptr;
    int32_t *d_ids = nullptr;
    REQUIRE(cudaMalloc(&d_slots, sizeof(FlintRoutedSlot) * K) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ids, sizeof(int32_t) * K) == cudaSuccess);
    REQUIRE(RunFlintRouter(nullptr, dw, dx, 1, E, K, Hd, d_slots, d_ids) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<FlintRoutedSlot> slots(K);
    std::vector<int32_t> ids(K);
    FromDevice(d_slots, slots);
    FromDevice(d_ids, ids);

    std::vector<float> xr(x.size());
    for (size_t i = 0; i < x.size(); ++i) xr[i] = F(x[i]);
    std::vector<int32_t> want_ids;
    std::vector<float> want_coeffs;
    ref::Router(wf.data(), xr.data(), E, K, Hd, want_ids, want_coeffs);

    // Ids ascend (deterministic accumulation order) and match the reference set.
    CHECK(std::is_sorted(ids.begin(), ids.end()));
    CHECK(ids == want_ids);
    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        CHECK(slots[static_cast<size_t>(k)].eid == ids[static_cast<size_t>(k)]);
        CHECK(slots[static_cast<size_t>(k)].coeff ==
              Catch::Approx(want_coeffs[static_cast<size_t>(k)]).margin(1e-6));
        total += slots[static_cast<size_t>(k)].coeff;
    }
    // No routed scale factor: the coefficients sum to one.
    CHECK(total == Catch::Approx(1.0f).margin(1e-6));

    cudaFree(dw);
    cudaFree(dx);
    cudaFree(d_slots);
    cudaFree(d_ids);
}

TEST_CASE("flint rotary matches the rotate-half reference", "[flint][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int heads = 2, head_dim = 8, pos = 3;
    auto base = FPattern(static_cast<int64_t>(heads) * head_dim, 0.3f, 7);
    std::vector<__half> q(base.size());
    for (size_t i = 0; i < base.size(); ++i) q[i] = H(base[i]);
    __half *dq = ToDevice(q);
    REQUIRE(RunFlintRope(nullptr, dq, nullptr, 1, heads, 0, head_dim, 10000.0f,
                         pos) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> out(q.size());
    FromDevice(dq, out);
    for (int h = 0; h < heads; ++h) {
        std::vector<float> want(static_cast<size_t>(head_dim));
        for (int d = 0; d < head_dim; ++d)
            want[static_cast<size_t>(d)] = F(q[static_cast<size_t>(h * head_dim + d)]);
        ref::RopeHalf(want.data(), head_dim, 10000.0f, pos);
        for (int d = 0; d < head_dim; ++d)
            CHECK(F(out[static_cast<size_t>(h * head_dim + d)]) ==
                  Catch::Approx(want[static_cast<size_t>(d)]).margin(2e-3));
    }
    cudaFree(dq);
}

TEST_CASE("flint norm, expert accumulate and shared combine", "[flint][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int rows = 2, hidden = 16, K = 3;

    // Plain RMSNorm (weight form).
    auto xf = FPattern(static_cast<int64_t>(rows) * hidden, 0.2f, 8);
    auto wf = FPattern(hidden, 0.05f, 9);
    std::vector<__half> x(xf.size()), w(wf.size());
    for (size_t i = 0; i < xf.size(); ++i) x[i] = H(xf[i]);
    for (size_t i = 0; i < wf.size(); ++i) w[i] = H(1.0f + wf[i]);
    __half *dx = ToDevice(x), *dw = ToDevice(w);
    __half *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, sizeof(__half) * x.size()) == cudaSuccess);
    REQUIRE(RunFlintNorm(nullptr, dx, dw, dout, rows, hidden, 1e-6f) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> out(x.size());
    FromDevice(dout, out);
    for (int r = 0; r < rows; ++r) {
        double sq = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const float v = F(x[static_cast<size_t>(r * hidden + i)]);
            sq += static_cast<double>(v) * v;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(sq / hidden + 1e-6));
        for (int i = 0; i < hidden; ++i) {
            const float v = F(x[static_cast<size_t>(r * hidden + i)]);
            const float want = v * inv * F(w[static_cast<size_t>(i)]);
            CHECK(F(out[static_cast<size_t>(r * hidden + i)]) ==
                  Catch::Approx(want).margin(3e-3));
        }
    }

    // Routed accumulate + FP32 shared combine.
    auto yf = FPattern(static_cast<int64_t>(K) * hidden, 0.4f, 10);
    auto shf = FPattern(hidden, 0.25f, 11);
    std::vector<__half> y(yf.size()), sh(shf.size());
    for (size_t i = 0; i < yf.size(); ++i) y[i] = H(yf[i]);
    for (size_t i = 0; i < shf.size(); ++i) sh[i] = H(shf[i]);
    std::vector<FlintRoutedSlot> slots(K);
    for (int k = 0; k < K; ++k)
        slots[static_cast<size_t>(k)] = {k, 0.25f * static_cast<float>(k + 1)};
    float *dacc = nullptr;
    __half *dy = ToDevice(y), *dsh = ToDevice(sh);
    float *dcmb = nullptr;
    REQUIRE(cudaMalloc(&dacc, sizeof(float) * hidden) == cudaSuccess);
    // The combined branch is FP32: the MoE output reaches 1.6e5 on the
    // massive-activation layers, which FP16 cannot hold.
    REQUIRE(cudaMalloc(&dcmb, sizeof(float) * hidden) == cudaSuccess);
    FlintRoutedSlot *dslots = ToDevice(slots);
    REQUIRE(RunFlintExpertAccumulate(nullptr, dacc, dy, dslots, K, hidden) ==
            cudaSuccess);
    REQUIRE(RunFlintMoECombine(nullptr, dacc, dsh, dcmb, 1, hidden) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> cmb(hidden);
    FromDevice(dcmb, cmb);
    for (int i = 0; i < hidden; ++i) {
        float want = F(sh[static_cast<size_t>(i)]);
        for (int k = 0; k < K; ++k)
            want += slots[static_cast<size_t>(k)].coeff *
                    F(y[static_cast<size_t>(k * hidden + i)]);
        CHECK(cmb[static_cast<size_t>(i)] ==
              Catch::Approx(want).margin(2e-3));
    }

    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
    cudaFree(dy);
    cudaFree(dsh);
    cudaFree(dcmb);
    cudaFree(dacc);
    cudaFree(dslots);
}
