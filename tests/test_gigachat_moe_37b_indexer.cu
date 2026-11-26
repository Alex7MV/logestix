// TDD — GigaChat4-37B-A4B DSA lightning indexer. Each stage that can be pinned
// exactly against a double host reference is pinned exactly (non-interleaved
// rope, plain RMSNorm, the score dot, the exact top-k with ties); the composite
// step is pinned by invariants plus determinism, and its projection accuracy by
// the model-level MLA parity test.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_37b_constants.h"
#include "lsxcommon/gigachat_moe_37b_inference_indexer.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

using bf16 = uint16_t;

float Bf16ToF32(bf16 b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bf16 F32ToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<bf16>(bits >> 16);
}

// The launchers return `cudaGetLastError()`, so a test must start from a clean
// error state or an unrelated earlier call is reported as the launcher's.
void ClearCudaError()
{
    while (cudaGetLastError() != cudaSuccess)
    {
    }
}

bool HaveDevice()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct DevBuffer {
    void *p = nullptr;
    explicit DevBuffer(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuffer()
    {
        if (p) cudaFree(p);
    }
    DevBuffer(const DevBuffer &) = delete;
    DevBuffer &operator=(const DevBuffer &) = delete;
    template <typename T>
    T *as() { return static_cast<T *>(p); }
};

} // namespace

TEST_CASE("gigachat4 indexer rope is non-interleaved", "[gigachat4][indexer]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kHeads = 2, kDim = 8, kRope = 4, kHalf = kRope / 2;
    constexpr int kCap = 4;
    std::vector<float> q(kHeads * kDim), k(kDim);
    for (int i = 0; i < kHeads * kDim; ++i)
        q[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
    for (int i = 0; i < kDim; ++i) k[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i + 1);

    std::vector<float> cos_t(static_cast<size_t>(kCap) * kHalf),
        sin_t(static_cast<size_t>(kCap) * kHalf);
    for (int p = 0; p < kCap; ++p)
        for (int j = 0; j < kHalf; ++j)
        {
            const double a = 0.3 * (p + 1) * (j + 1);
            cos_t[static_cast<size_t>(p) * kHalf + j] = static_cast<float>(std::cos(a));
            sin_t[static_cast<size_t>(p) * kHalf + j] = static_cast<float>(std::sin(a));
        }

    DevBuffer d_q(kHeads * kDim * 4), d_k(kDim * 4), d_cos(cos_t.size() * 4),
        d_sin(sin_t.size() * 4);
    cudaMemcpy(d_q.p, q.data(), kHeads * kDim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k.p, k.data(), kDim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos.p, cos_t.data(), cos_t.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin.p, sin_t.data(), sin_t.size() * 4, cudaMemcpyHostToDevice);

    const int pos = 2;
    ClearCudaError();
    REQUIRE(lsxcommon::gigachat::RunGigaChatDsaIndexerRope(
                d_q.as<float>(), d_k.as<float>(), kHeads, kDim, kRope,
                d_cos.as<float>(), d_sin.as<float>(), pos, nullptr) ==
            cudaSuccess);

    std::vector<float> got_q(kHeads * kDim), got_k(kDim);
    cudaMemcpy(got_q.data(), d_q.p, kHeads * kDim * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_k.data(), d_k.p, kDim * 4, cudaMemcpyDeviceToHost);

    // Reference: rotate_half over the half split of the first `rope` lanes; the
    // remaining lanes are untouched.
    auto expect = [&](const std::vector<float> &x, int base) {
        std::vector<float> y(x);
        for (int j = 0; j < kHalf; ++j)
        {
            const float c = cos_t[static_cast<size_t>(pos) * kHalf + j];
            const float s = sin_t[static_cast<size_t>(pos) * kHalf + j];
            const float a = x[static_cast<size_t>(base + j)];
            const float b = x[static_cast<size_t>(base + j + kHalf)];
            y[static_cast<size_t>(base + j)] = a * c - b * s;
            y[static_cast<size_t>(base + j + kHalf)] = b * c + a * s;
        }
        return y;
    };
    for (int h = 0; h < kHeads; ++h)
    {
        const auto want = expect(q, h * kDim);
        for (int i = 0; i < kDim; ++i)
            CHECK(got_q[static_cast<size_t>(h * kDim + i)] ==
                  Catch::Approx(want[static_cast<size_t>(h * kDim + i)])
                      .margin(1e-6));
        // Lanes at/after `rope` are untouched.
        for (int i = kRope; i < kDim; ++i)
            CHECK(got_q[static_cast<size_t>(h * kDim + i)] ==
                  q[static_cast<size_t>(h * kDim + i)]);
    }
    const auto want_k = expect(k, 0);
    for (int i = 0; i < kDim; ++i)
        CHECK(got_k[static_cast<size_t>(i)] ==
              Catch::Approx(want_k[static_cast<size_t>(i)]).margin(1e-6));
    // A launch that fails asynchronously must show up here, not as a stale
    // error in the next test case.
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("gigachat4 indexer k-norm is a plain RMSNorm", "[gigachat4][indexer]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kDim = 16;
    std::vector<float> k(kDim);
    std::vector<bf16> w(kDim);
    for (int i = 0; i < kDim; ++i)
    {
        k[static_cast<size_t>(i)] = 0.1f * static_cast<float>((i % 5) - 2);
        w[static_cast<size_t>(i)] = F32ToBf16(0.5f + 0.1f * static_cast<float>(i % 3));
    }
    double ss = 0.0;
    for (int i = 0; i < kDim; ++i) ss += double(k[static_cast<size_t>(i)]) * k[static_cast<size_t>(i)];
    const double inv = 1.0 / std::sqrt(ss / kDim + 1e-6);
    std::vector<double> want(kDim);
    for (int i = 0; i < kDim; ++i)
        want[static_cast<size_t>(i)] = k[static_cast<size_t>(i)] * inv *
                                       double(Bf16ToF32(w[static_cast<size_t>(i)]));

    DevBuffer d_k(kDim * 4), d_w(kDim * 2);
    REQUIRE(d_k.p != nullptr);
    REQUIRE(d_w.p != nullptr);
    REQUIRE(cudaMemcpy(d_k.p, k.data(), kDim * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(d_w.p, w.data(), kDim * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    ClearCudaError();
    REQUIRE(lsxcommon::gigachat::RunGigaChatDsaIndexerKNorm(
                d_k.as<float>(), static_cast<const uint8_t *>(d_w.p), kDim,
                1e-6f, nullptr) == cudaSuccess);
    std::vector<float> got(kDim);
    cudaMemcpy(got.data(), d_k.p, kDim * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kDim; ++i)
        CHECK(got[static_cast<size_t>(i)] ==
              Catch::Approx(want[static_cast<size_t>(i)]).margin(1e-5));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("gigachat4 indexer scores match the reference dot", "[gigachat4][indexer]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kHeads = 3, kDim = 8, kN = 5;
    std::vector<float> q(kHeads * kDim), kc(kN * kDim), w(kHeads);
    for (int i = 0; i < kHeads * kDim; ++i)
        q[static_cast<size_t>(i)] = 0.1f * static_cast<float>((i % 7) - 3);
    for (int i = 0; i < kN * kDim; ++i)
        kc[static_cast<size_t>(i)] = 0.05f * static_cast<float>((i % 11) - 5);
    for (int h = 0; h < kHeads; ++h)
        w[static_cast<size_t>(h)] = 0.2f * static_cast<float>(h + 1);

    const float scale = 0.08838834764831845f;  // 128^-0.5 is the real value
    std::vector<float> want(kN, 0.0f);
    for (int t = 0; t < kN; ++t)
    {
        float acc = 0.0f;
        for (int h = 0; h < kHeads; ++h)
        {
            float dot = 0.0f;
            for (int d = 0; d < kDim; ++d)
                dot += q[static_cast<size_t>(h * kDim + d)] *
                       kc[static_cast<size_t>(t * kDim + d)];
            acc += w[static_cast<size_t>(h)] * std::max(dot, 0.0f);
        }
        want[static_cast<size_t>(t)] = acc * scale;
    }

    DevBuffer d_q(q.size() * 4), d_k(kc.size() * 4), d_w(w.size() * 4),
        d_s(kN * 4);
    cudaMemcpy(d_q.p, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k.p, kc.data(), kc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.p, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    ClearCudaError();
    REQUIRE(lsxcommon::gigachat::RunGigaChatDsaIndexerScores(
                d_q.as<float>(), d_k.as<float>(), d_w.as<float>(), kHeads, kDim,
                scale, kN, d_s.as<float>(), nullptr) == cudaSuccess);
    std::vector<float> got(kN);
    cudaMemcpy(got.data(), d_s.p, kN * 4, cudaMemcpyDeviceToHost);
    for (int t = 0; t < kN; ++t)
        CHECK(got[static_cast<size_t>(t)] ==
              Catch::Approx(want[static_cast<size_t>(t)]).margin(1e-6));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("gigachat4 indexer top-k is exact, ascending and tie-safe",
          "[gigachat4][indexer]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    // Ties on purpose: three 1.0 entries with k = 2 must return the two lowest
    // indices, and a strictly-greater entry must never be crowded out.
    std::vector<float> scores = {1.0f, 5.0f, 1.0f, 9.0f, 1.0f, 2.0f, 0.5f};
    constexpr int kN = 7;
    DevBuffer d_s(kN * 4), d_idx(kN * 4), d_cnt(4);
    REQUIRE(cudaMemcpy(d_s.p, scores.data(), kN * 4, cudaMemcpyHostToDevice) ==
            cudaSuccess);


    auto select = [&](int k) {
        cudaMemset(d_idx.p, 0xFF, kN * 4);
        ClearCudaError();
        return lsxcommon::gigachat::RunGigaChatDsaTopkSelect(
            d_s.as<float>(), kN, k, d_idx.as<int32_t>(), d_cnt.as<int>(),
            nullptr);
    };

    REQUIRE(select(3) == cudaSuccess);
    int cnt = 0;
    cudaMemcpy(&cnt, d_cnt.p, 4, cudaMemcpyDeviceToHost);
    CHECK(cnt == 3);
    std::vector<int32_t> idx(3);
    cudaMemcpy(idx.data(), d_idx.p, 3 * 4, cudaMemcpyDeviceToHost);
    // Ascending index order (the selector's contract), not score order.
    CHECK(idx == std::vector<int32_t>{1, 3, 5});
    CHECK(std::is_sorted(idx.begin(), idx.end()));

    // k >= n is the identity selection: every past position.
    REQUIRE(select(kN) == cudaSuccess);
    cudaMemcpy(&cnt, d_cnt.p, 4, cudaMemcpyDeviceToHost);
    CHECK(cnt == kN);
    std::vector<int32_t> all(kN);
    cudaMemcpy(all.data(), d_idx.p, kN * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kN; ++i) CHECK(all[static_cast<size_t>(i)] == i);

    // Determinism: the same scores select the same set twice.
    REQUIRE(select(3) == cudaSuccess);
    std::vector<int32_t> again(3);
    cudaMemcpy(again.data(), d_idx.p, 3 * 4, cudaMemcpyDeviceToHost);
    CHECK(again == idx);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

TEST_CASE("gigachat4 indexer step caches post-rope keys and selects",
          "[gigachat4][indexer]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    using namespace lsxcommon::gigachat;
    constexpr int kHidden = 16, kQLora = 8, kHeads = 2, kDim = 8, kRope = 4,
                  kTopk = 3, kCap = 8;
    constexpr int kHalf = kRope / 2;

    // wq_b: [heads*head_dim, q_lora] with a single 1.0 per row, quantized by the
    // engine's own NVFP4 quantizer (the path the converter uses).
    std::vector<float> wq(static_cast<size_t>(kHeads * kDim) * kQLora, 0.0f);
    for (int r = 0; r < kHeads * kDim; ++r)
        wq[static_cast<size_t>(r) * kQLora + (r % kQLora)] = 1.0f;
    std::vector<bf16> wq_bf(wq.size());
    for (size_t i = 0; i < wq.size(); ++i) wq_bf[i] = F32ToBf16(wq[i]);
    std::vector<uint8_t> packed, scales;
    float gs = 1.0f;
    REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
        reinterpret_cast<const uint8_t *>(wq_bf.data()),
        static_cast<int64_t>(wq_bf.size()), "BF16", nullptr,
        {kHeads * kDim, kQLora}, {}, packed, scales, gs));

    GcDsaIndexerWeights w;
    w.wq_b.rows = kHeads * kDim;
    w.wq_b.cols = kQLora;
    w.wq_b.global_scale = gs;
    w.wq_b.packed_bytes = static_cast<int64_t>(packed.size());
    w.wq_b.scale_bytes = static_cast<int64_t>(scales.size());
    {
        void *p = nullptr, *sc = nullptr;
        REQUIRE(cudaMalloc(&p, packed.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&sc, scales.size()) == cudaSuccess);
        REQUIRE(cudaMemcpy(p, packed.data(), packed.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(sc, scales.data(), scales.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        w.wq_b.d_packed = static_cast<const uint8_t *>(p);
        w.wq_b.d_scales = static_cast<const uint8_t *>(sc);
    }

    std::vector<bf16> wk(static_cast<size_t>(kDim) * kHidden),
        wp(static_cast<size_t>(kHeads) * kHidden), kn(kDim);
    std::vector<float> wk_f(wk.size()), wp_f(wp.size()), kn_f(kn.size());
    for (size_t i = 0; i < wk.size(); ++i)
    {
        const float v = 0.02f * static_cast<float>((i % 9) - 4);
        wk[i] = F32ToBf16(v);
        wk_f[i] = Bf16ToF32(wk[i]);
    }
    for (size_t i = 0; i < wp.size(); ++i)
    {
        const float v = 0.05f * static_cast<float>((i % 5) - 2);
        wp[i] = F32ToBf16(v);
        wp_f[i] = Bf16ToF32(wp[i]);
    }
    for (int i = 0; i < kDim; ++i)
    {
        kn[i] = F32ToBf16(0.9f + 0.05f * static_cast<float>(i % 3));
        kn_f[static_cast<size_t>(i)] = Bf16ToF32(kn[static_cast<size_t>(i)]);
    }
    std::vector<bf16> hidden(kHidden);
    std::vector<float> hidden_f(kHidden);
    for (int i = 0; i < kHidden; ++i)
    {
        const float v = 0.1f * static_cast<float>((i % 7) - 3);
        hidden[static_cast<size_t>(i)] = F32ToBf16(v);
        hidden_f[static_cast<size_t>(i)] = Bf16ToF32(hidden[static_cast<size_t>(i)]);
    }
    std::vector<float> ql(kQLora);
    for (int i = 0; i < kQLora; ++i)
        ql[static_cast<size_t>(i)] = 0.3f * static_cast<float>(i + 1);

    std::vector<float> cos_t(static_cast<size_t>(kCap) * kHalf),
        sin_t(static_cast<size_t>(kCap) * kHalf);
    for (int p = 0; p < kCap; ++p)
        for (int j = 0; j < kHalf; ++j)
        {
            const double a = 0.2 * (p + 1) * (j + 1);
            cos_t[static_cast<size_t>(p) * kHalf + j] = static_cast<float>(std::cos(a));
            sin_t[static_cast<size_t>(p) * kHalf + j] = static_cast<float>(std::sin(a));
        }

    auto dev_copy = [](const void *src, size_t bytes) {
        void *d = nullptr;
        REQUIRE(cudaMalloc(&d, bytes) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        return d;
    };
    void *d_wk = dev_copy(wk.data(), wk.size() * 2);
    void *d_wp = dev_copy(wp.data(), wp.size() * 2);
    void *d_kn = dev_copy(kn.data(), kn.size() * 2);
    void *d_hid = dev_copy(hidden.data(), hidden.size() * 2);
    void *d_cos = dev_copy(cos_t.data(), cos_t.size() * 4);
    void *d_sin = dev_copy(sin_t.data(), sin_t.size() * 4);
    std::vector<float> ql_scaled(ql);
    float *d_ql = static_cast<float *>(dev_copy(ql_scaled.data(), ql.size() * 4));
    float *d_kcache = static_cast<float *>(dev_copy(
        std::vector<float>(static_cast<size_t>(kCap) * kDim, 0.0f).data(),
        static_cast<size_t>(kCap) * kDim * 4));
    int32_t *d_idx = static_cast<int32_t *>(
        dev_copy(std::vector<int32_t>(kCap, -1).data(), kCap * 4));
    int *d_cnt = static_cast<int *>(dev_copy(std::vector<int>(1, -1).data(), 4));
    w.d_wk = static_cast<const uint8_t *>(d_wk);
    w.d_weights_proj = static_cast<const uint8_t *>(d_wp);
    w.d_k_norm = static_cast<const uint8_t *>(d_kn);

    GcDsaIndexerDims d;
    d.hidden = kHidden;
    d.q_lora = kQLora;
    d.heads = kHeads;
    d.head_dim = kDim;
    d.rope = kRope;
    d.topk = kTopk;
    GcDsaIndexerScratch sc;
    REQUIRE(sc.Alloc(d, kCap));

    // Host reference for the cached key: k = k_norm(wk(x)) then non-interleaved
    // rope at this position.
    auto expected_key = [&](int pos) {
        std::vector<double> k(kDim, 0.0);
        for (int r = 0; r < kDim; ++r)
        {
            double a = 0.0;
            for (int i = 0; i < kHidden; ++i)
                a += double(wk_f[static_cast<size_t>(r * kHidden + i)]) *
                     double(hidden_f[static_cast<size_t>(i)]);
            k[static_cast<size_t>(r)] = a;
        }
        double ss = 0.0;
        for (int r = 0; r < kDim; ++r) ss += k[static_cast<size_t>(r)] * k[static_cast<size_t>(r)];
        const double inv = 1.0 / std::sqrt(ss / kDim + 1e-6);
        std::vector<double> out(kDim);
        for (int r = 0; r < kDim; ++r)
            out[static_cast<size_t>(r)] = k[static_cast<size_t>(r)] * inv *
                                          double(kn_f[static_cast<size_t>(r)]);
        for (int j = 0; j < kHalf; ++j)
        {
            const float c = cos_t[static_cast<size_t>(pos) * kHalf + j];
            const float s = sin_t[static_cast<size_t>(pos) * kHalf + j];
            const double a = out[static_cast<size_t>(j)];
            const double b = out[static_cast<size_t>(j + kHalf)];
            out[static_cast<size_t>(j)] = a * c - b * s;
            out[static_cast<size_t>(j + kHalf)] = b * c + a * s;
        }
        return out;
    };

    for (int pos = 0; pos < 5; ++pos)
    {
        REQUIRE(RunGigaChatDsaIndexerStep(
                    d_ql, static_cast<const uint8_t *>(d_hid), w, d,
                    static_cast<const float *>(d_cos),
                    static_cast<const float *>(d_sin), pos, kCap, d_kcache, sc,
                    d_idx, d_cnt, nullptr) == cudaSuccess);

        std::vector<float> cache(static_cast<size_t>(kCap) * kDim);
        cudaMemcpy(cache.data(), d_kcache, cache.size() * 4,
                   cudaMemcpyDeviceToHost);
        const auto want = expected_key(pos);
        for (int r = 0; r < kDim; ++r)
            CHECK(cache[static_cast<size_t>(pos * kDim + r)] ==
                  Catch::Approx(want[static_cast<size_t>(r)]).margin(2e-3));

        // The selection is exactly min(topk, pos+1) ascending indices inside the
        // causal prefix, and never empty.
        int cnt = 0;
        cudaMemcpy(&cnt, d_cnt, 4, cudaMemcpyDeviceToHost);
        CHECK(cnt == std::min(kTopk, pos + 1));
        std::vector<int32_t> idx(static_cast<size_t>(cnt));
        cudaMemcpy(idx.data(), d_idx, static_cast<size_t>(cnt) * 4,
                   cudaMemcpyDeviceToHost);
        CHECK(std::is_sorted(idx.begin(), idx.end()));
        for (int32_t v : idx)
        {
            CHECK(v >= 0);
            CHECK(v <= pos);
        }
        if (cnt == pos + 1)
            for (int i = 0; i < cnt; ++i)
                CHECK(idx[static_cast<size_t>(i)] == i);

        // Determinism: the same position twice selects the same set.
        std::vector<int32_t> again(static_cast<size_t>(cnt));
        REQUIRE(RunGigaChatDsaIndexerStep(
                    d_ql, static_cast<const uint8_t *>(d_hid), w, d,
                    static_cast<const float *>(d_cos),
                    static_cast<const float *>(d_sin), pos, kCap, d_kcache, sc,
                    d_idx, d_cnt, nullptr) == cudaSuccess);
        cudaMemcpy(again.data(), d_idx, static_cast<size_t>(cnt) * 4,
                   cudaMemcpyDeviceToHost);
        CHECK(again == idx);
    }

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    sc.Free();
    w.Free();
    for (void *p : {d_wk, d_wp, d_kn, d_hid, d_cos, d_sin,
                    static_cast<void *>(d_ql),
                    static_cast<void *>(d_kcache),
                    static_cast<void *>(d_idx),
                    static_cast<void *>(d_cnt)})
        cudaFree(p);
}
