// TDD — Nemotron-3-Ultra-550B device kernels vs the host reference.
//
// Every case runs the launcher and the ref:: function on the same input and
// compares. The router is the one that matters most: Ultra routes through
// n_group 8 / topk_group 1, and a missing group mask still produces a valid
// top-k that answers "Paris" — so selection (not just weights) is compared, on
// random draws and on the crafted tie vector.
//
// Device math is compiled with -fmad=false in this target (see CMakeLists), so
// relu2-half is required to be bit-identical to float relu2 then round-to-half.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "nemotron_moe_ultra_device_math.h"
#include "nemotron_moe_ultra_kernels.h"

using namespace lsxcommon::nemotron_ultra;
using Catch::Approx;

namespace {

template <typename T>
struct Dev {
    T *p = nullptr;
    explicit Dev(std::size_t n) { REQUIRE(cudaMalloc(&p, n * sizeof(T)) == cudaSuccess); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    void Put(const std::vector<T> &v)
    {
        REQUIRE(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    }
    void Get(std::vector<T> &v) const
    {
        REQUIRE(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost) ==
                cudaSuccess);
    }
};

// Deterministic pseudo-random floats in [-4, 4); no <random> so the sequence is
// identical across platforms and libc++ versions.
std::vector<float> Rand(int n, std::uint32_t seed)
{
    std::vector<float> v(static_cast<std::size_t>(n));
    std::uint32_t s = seed | 1u;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[static_cast<std::size_t>(i)] =
            (static_cast<float>(s >> 8) / 16777216.0f) * 8.0f - 4.0f;
    }
    return v;
}

} // namespace

TEST_CASE("ultra kernels: grouped router matches the reference selection on the real shape",
          "[ultra][kernels]")
{
    // The production geometry: 512 experts, n_group 8, topk_group 1, top-22.
    constexpr int E = 512, k = 22, n_group = 8, topk_group = 1;
    for (std::uint32_t seed = 1; seed <= 50; ++seed) {
        const auto logits = Rand(E, seed * 7919u);
        const auto bias = Rand(E, seed * 104729u);

        float w_ref[k];
        int i_ref[k];
        ref::RouterTopKGrouped(logits.data(), bias.data(), E, k, n_group, topk_group,
                               5.0f, w_ref, i_ref);

        Dev<float> d_logits(E), d_bias(E), d_w(k);
        Dev<int> d_i(k);
        d_logits.Put(logits);
        d_bias.Put(bias);
        RunUltraRouter(d_logits.p, d_bias.p, E, k, n_group, topk_group, 5.0f, d_w.p,
                       d_i.p, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> w(k);
        std::vector<int> idx(k);
        d_w.Get(w);
        d_i.Get(idx);

        // Selection is a SET comparison: the reference's topk does not promise an
        // order, and the kernel sorts by weight.
        std::vector<int> a(idx), b(i_ref, i_ref + k);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        INFO("seed " << seed);
        CHECK(a == b);
        std::vector<float> wa(w), wb(w_ref, w_ref + k);
        std::sort(wa.begin(), wa.end());
        std::sort(wb.begin(), wb.end());
        for (int t = 0; t < k; ++t) CHECK(wa[t] == Approx(wb[t]).margin(2e-5));
    }
}

TEST_CASE("ultra kernels: the production routing shape (n_group 1) matches the reference",
          "[ultra][kernels]")
{
    // THIS checkpoint's routing geometry: n_group 1 / topk_group 1 over 512
    // experts with top-22. The group stage is then the identity (one group holds
    // all experts), which is a DIFFERENT code path from the grouped cases below:
    // a mask that wrongly excludes the single group would select nothing.
    constexpr int E = 512, k = 22, n_group = 1, topk_group = 1;
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        const auto logits = Rand(E, seed * 3571u);
        const auto bias = Rand(E, seed * 6151u);

        float w_ref[k];
        int i_ref[k];
        ref::RouterTopKGrouped(logits.data(), bias.data(), E, k, n_group, topk_group,
                               5.0f, w_ref, i_ref);

        Dev<float> d_logits(E), d_bias(E), d_w(k);
        Dev<int> d_i(k);
        d_logits.Put(logits);
        d_bias.Put(bias);
        RunUltraRouter(d_logits.p, d_bias.p, E, k, n_group, topk_group, 5.0f, d_w.p,
                       d_i.p, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> w(k);
        std::vector<int> idx(k);
        d_w.Get(w);
        d_i.Get(idx);

        std::vector<int> a(idx), b(i_ref, i_ref + k);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        INFO("seed " << seed);
        CHECK(a == b);
        // The 22 weights must be the normalized, scaled sigmoid scores: a null
        // selection would give an all -inf / NaN vector and sum to garbage.
        float sum = 0.0f;
        for (float v : w) sum += v;
        CHECK(sum == Approx(5.0f).margin(1e-3));
    }
}

TEST_CASE("ultra kernels: grouped router keeps only the winning group", "[ultra][kernels]")
{
    // Group 0 holds the best single expert, group 2 wins on the top-2 sum.
    const float logits[8] = {4.0f, 0.0f, 0.0f, 0.0f, 3.0f, 2.9f, 0.0f, 0.0f};
    float bias[8] = {0};
    float w_ref[2];
    int i_ref[2];
    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w_ref, i_ref);

    Dev<float> d_logits(8), d_bias(8), d_w(2);
    Dev<int> d_i(2);
    d_logits.Put(std::vector<float>(logits, logits + 8));
    d_bias.Put(std::vector<float>(bias, bias + 8));
    RunUltraRouter(d_logits.p, d_bias.p, 8, 2, 4, 1, 5.0f, d_w.p, d_i.p, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> w(2);
    std::vector<int> idx(2);
    d_w.Get(w);
    d_i.Get(idx);

    CHECK(((idx[0] == 4 && idx[1] == 5) || (idx[0] == 5 && idx[1] == 4)));
    CHECK((idx[0] == i_ref[0] || idx[0] == i_ref[1]));
    CHECK(w[0] + w[1] == Approx(5.0f).margin(1e-4));
}

TEST_CASE("ultra kernels: grouped router resolves exact ties to the lowest index",
          "[ultra][kernels]")
{
    float logits[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float bias[8] = {0};
    float w_ref[2];
    int i_ref[2];
    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w_ref, i_ref);

    Dev<float> d_logits(8), d_bias(8), d_w(2);
    Dev<int> d_i(2);
    d_logits.Put(std::vector<float>(logits, logits + 8));
    d_bias.Put(std::vector<float>(bias, bias + 8));
    RunUltraRouter(d_logits.p, d_bias.p, 8, 2, 4, 1, 5.0f, d_w.p, d_i.p, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<int> idx(2);
    d_i.Get(idx);
    // The device result must equal the host reference exactly, order included:
    // both walk the same lowest-index-first scan, so a divergence here is a real
    // nondeterminism.
    CHECK(idx[0] == i_ref[0]);
    CHECK(idx[1] == i_ref[1]);
}

TEST_CASE("ultra kernels: depthwise conv step matches the reference", "[ultra][kernels]")
{
    constexpr int conv_dim = 32, K = 4, S = K - 1;
    const auto xBC = Rand(conv_dim, 11);
    const auto w = Rand(conv_dim * K, 12);
    const auto b = Rand(conv_dim, 13);
    const auto st = Rand(conv_dim * S, 14);

    std::vector<float> out_ref(conv_dim), st_ref(conv_dim * S);
    // Host reference for the same single step (test-local: the conv is a plain
    // FIR, and keeping it here means the device kernel is compared to the
    // formula, not to another kernel).
    for (int c = 0; c < conv_dim; ++c) {
        float acc = b[static_cast<std::size_t>(c)];
        for (int k = 0; k < S; ++k)
            acc += w[c * K + k] * st[c * S + k];
        acc += w[c * K + (K - 1)] * xBC[static_cast<std::size_t>(c)];
        for (int k = 0; k < S - 1; ++k) st_ref[c * S + k] = st[c * S + k + 1];
        st_ref[c * S + (S - 1)] = xBC[static_cast<std::size_t>(c)];
        out_ref[static_cast<std::size_t>(c)] = ref::Silu(acc);
    }

    Dev<float> d_xBC(conv_dim), d_w(conv_dim * K), d_b(conv_dim), d_st(conv_dim * S);
    Dev<float> d_stout(conv_dim * S), d_out(conv_dim);
    d_xBC.Put(xBC); d_w.Put(w); d_b.Put(b); d_st.Put(st);
    RunUltraConvStep(d_xBC.p, d_w.p, d_b.p, d_st.p, d_stout.p, d_out.p, conv_dim, K,
                     nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> out(conv_dim), stout(conv_dim * S);
    d_out.Get(out);
    d_stout.Get(stout);
    for (int c = 0; c < conv_dim; ++c) {
        CHECK(out[static_cast<std::size_t>(c)] == Approx(out_ref[static_cast<std::size_t>(c)]).margin(1e-6));
        for (int k = 0; k < S; ++k)
            CHECK(stout[c * S + k] == Approx(st_ref[c * S + k]).margin(1e-9));
    }
}

TEST_CASE("ultra kernels: selective state update matches the reference in both dt modes",
          "[ultra][kernels]")
{
    constexpr int heads = 4, head_dim = 8, groups = 2, state_size = 4;
    const auto st = Rand(heads * head_dim * state_size, 21);
    const auto xs = Rand(heads * head_dim, 22);
    const auto B = Rand(groups * state_size, 23);
    const auto C = Rand(groups * state_size, 24);
    const auto D = Rand(heads, 25);
    const auto dt = Rand(heads, 26);
    auto dt_bias = Rand(heads, 27);
    dt_bias[0] = -60.0f;  // forces the clamp path to be visible
    auto A_log = Rand(heads, 28);
    A_log[1] = 4.0f;      // A = -exp(4): the clamp changes the state a lot

    for (int clamp = 0; clamp <= 1; ++clamp) {
        std::vector<float> st_ref(heads * head_dim * state_size);
        std::vector<float> out_ref(heads * head_dim);
        ref::SelectiveStateUpdate(st.data(), xs.data(), B.data(), C.data(), D.data(),
                                  dt.data(), dt_bias.data(), A_log.data(), heads,
                                  head_dim, groups, state_size, clamp != 0,
                                  st_ref.data(), out_ref.data());

        Dev<float> d_st(st.size()), d_xs(xs.size()), d_B(B.size()), d_C(C.size()),
            d_D(D.size()), d_dt(dt.size()), d_dtb(dt_bias.size()),
            d_A(A_log.size()), d_stout(st.size()), d_out(out_ref.size());
        d_st.Put(st); d_xs.Put(xs); d_B.Put(B); d_C.Put(C); d_D.Put(D);
        d_dt.Put(dt); d_dtb.Put(dt_bias); d_A.Put(A_log);
        RunUltraSelectiveStateStep(d_st.p, d_xs.p, d_B.p, d_C.p, d_D.p, d_dt.p,
                                   d_dtb.p, d_A.p, heads, head_dim, groups,
                                   state_size, clamp, d_stout.p, d_out.p, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> st_got(st.size()), out_got(out_ref.size());
        d_stout.Get(st_got);
        d_out.Get(out_got);
        for (std::size_t i = 0; i < st_got.size(); ++i)
            CHECK(st_got[i] == Approx(st_ref[i]).margin(1e-5));
        for (std::size_t i = 0; i < out_got.size(); ++i)
            CHECK(out_got[i] == Approx(out_ref[i]).margin(1e-5));
    }
}

TEST_CASE("ultra kernels: gated RMSNorm matches the reference at group_size 2048",
          "[ultra][kernels]")
{
    // The production group is 2048 (> 1024 threads), so this also proves the
    // strided path rather than one thread per element.
    constexpr int group = 2048, groups = 2, n = group * groups;
    const auto x = Rand(n, 31);
    const auto gate = Rand(n, 32);
    const auto weight = Rand(n, 33);
    std::vector<float> out_ref(n);
    ref::GatedRmsNorm(x.data(), gate.data(), weight.data(), n, group, 1e-5f,
                      out_ref.data());

    Dev<float> d_x(n), d_g(n), d_w(n), d_out(n);
    d_x.Put(x); d_g.Put(gate); d_w.Put(weight);
    RunUltraGatedRmsNorm(d_x.p, d_g.p, d_w.p, n, group, 1e-5f, d_out.p, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> out(n);
    d_out.Get(out);
    for (int i = 0; i < n; ++i) CHECK(out[static_cast<std::size_t>(i)] ==
                                       Approx(out_ref[static_cast<std::size_t>(i)]).margin(1e-4));
}

TEST_CASE("ultra kernels: NoPE attention decode matches a host softmax", "[ultra][kernels]")
{
    constexpr int heads = 8, kv_heads = 2, head_dim = 16, seq = 12, stride = 16;
    const auto q = Rand(heads * head_dim, 41);
    const auto kc = Rand(kv_heads * stride * head_dim, 42);
    const auto vc = Rand(kv_heads * stride * head_dim, 43);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Host expectation: GQA (heads/kv_heads replication), softmax in F32, NO
    // rotary anywhere (these layers are NoPE).
    std::vector<float> out_ref(heads * head_dim);
    for (int h = 0; h < heads; ++h) {
        const int hk = h / (heads / kv_heads);
        std::vector<float> sc(seq);
        float m = -INFINITY;
        for (int t = 0; t < seq; ++t) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                const std::size_t kb =
                    (static_cast<std::size_t>(hk) * stride + t) * head_dim;
                acc += q[h * head_dim + d] * kc[kb + d];
            }
            sc[static_cast<std::size_t>(t)] = acc * scale;
            m = std::max(m, sc[static_cast<std::size_t>(t)]);
        }
        float l = 0.0f;
        for (int t = 0; t < seq; ++t) {
            sc[static_cast<std::size_t>(t)] = std::exp(sc[static_cast<std::size_t>(t)] - m);
            l += sc[static_cast<std::size_t>(t)];
        }
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < seq; ++t) {
                const std::size_t kb =
                    (static_cast<std::size_t>(hk) * stride + t) * head_dim;
                acc += sc[static_cast<std::size_t>(t)] * vc[kb + d];
            }
            out_ref[h * head_dim + d] = acc / l;
        }
    }

    Dev<float> d_q(q.size()), d_k(kc.size()), d_v(vc.size()), d_o(out_ref.size());
    d_q.Put(q); d_k.Put(kc); d_v.Put(vc);
    RunUltraAttnDecode(d_q.p, d_k.p, d_v.p, heads, kv_heads, head_dim, seq, stride,
                       scale, d_o.p, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> out(out_ref.size());
    d_o.Get(out);
    for (std::size_t i = 0; i < out.size(); ++i)
        CHECK(out[i] == Approx(out_ref[i]).margin(2e-5));
}

TEST_CASE("ultra kernels: relu2 in FP16 is bit-identical to float relu2 rounded",
          "[ultra][kernels]")
{
    constexpr int n = 257;
    auto v = Rand(n, 51);
    // Cover the boundaries as well: exact zero, tiny negatives, large positives.
    v[0] = 0.0f; v[1] = -0.0f; v[2] = -1e-7f; v[3] = 1e4f; v[4] = -3.5f;

    std::vector<__half> h(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) h[static_cast<std::size_t>(i)] = __float2half(v[static_cast<std::size_t>(i)]);
    std::vector<__half> want(h);
    for (int i = 0; i < n; ++i) {
        const float f = __half2float(h[static_cast<std::size_t>(i)]);
        const float r = f > 0.0f ? f : 0.0f;
        want[static_cast<std::size_t>(i)] = __float2half(r * r);
    }

    Dev<__half> d_h(static_cast<std::size_t>(n));
    d_h.Put(h);
    RunUltraRelu2Half(d_h.p, n, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> got(static_cast<std::size_t>(n));
    d_h.Get(got);
    for (int i = 0; i < n; ++i) {
        std::uint16_t a, b;
        std::memcpy(&a, &got[static_cast<std::size_t>(i)], 2);
        std::memcpy(&b, &want[static_cast<std::size_t>(i)], 2);
        INFO("index " << i);
        CHECK(a == b);
    }
}

TEST_CASE("ultra kernels: the expert weighted sum matches the host loop", "[ultra][kernels]")
{
    constexpr int count = 22, n = 64;  // the real top-k
    std::vector<__half> y(static_cast<std::size_t>(count) * n);
    for (int i = 0; i < count * n; ++i)
        y[static_cast<std::size_t>(i)] = __float2half(0.01f * static_cast<float>(i % 37));
    const auto w = Rand(count, 61);

    std::vector<float> acc_ref(static_cast<std::size_t>(n), 0.0f);
    for (int j = 0; j < n; ++j) {
        float s = 0.0f;
        for (int e = 0; e < count; ++e)
            s += w[static_cast<std::size_t>(e)] *
                 __half2float(y[static_cast<std::size_t>(e) * n + j]);
        acc_ref[static_cast<std::size_t>(j)] = s;
    }

    Dev<__half> d_y(y.size());
    Dev<float> d_w(count), d_acc(n);
    std::vector<float> zero(static_cast<std::size_t>(n), 0.0f);
    d_y.Put(y);
    d_w.Put(w);
    d_acc.Put(zero);
    RunUltraMoeWeightedSum(d_y.p, d_w.p, count, n, d_acc.p, /*add=*/false, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> acc(static_cast<std::size_t>(n));
    d_acc.Get(acc);
    for (int j = 0; j < n; ++j)
        CHECK(acc[static_cast<std::size_t>(j)] ==
              Approx(acc_ref[static_cast<std::size_t>(j)]).margin(2e-5));
}
