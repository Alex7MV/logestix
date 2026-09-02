#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "qwen_moe_exp_hostref.h"
#include "lsxcommon/qwen_moe_exp_inference.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct CudaBuf {
    void *p = nullptr;
    size_t bytes = 0;
    explicit CudaBuf(size_t b) : bytes(b)
    {
        cudaMalloc(&p, b ? b : 1);
        cudaMemset(p, 0, b ? b : 1);
    }
    ~CudaBuf() { if (p) cudaFree(p); }
    CudaBuf(const CudaBuf &) = delete;
    CudaBuf &operator=(const CudaBuf &) = delete;
    template <typename T> T *as() { return reinterpret_cast<T *>(p); }
};

} // namespace

TEST_CASE("hc GatedResidual F32 kernel parity vs host replica",
          "[qwen-exp-kernels]")
{
    namespace hr = lsxcommon::qwen_moe_exp::hostref;
    constexpr int H = 64, hc = 4, lr = 32, n = H * hc;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> hx(n), down(lr * n), up(n * lr), inject(hc * n);
    for (auto &v : hx) v = dist(rng);
    for (auto &v : down) v = dist(rng);
    for (auto &v : up) v = dist(rng);
    for (auto &v : inject) v = dist(rng);

    std::vector<float> ref_mixed(H), ref_inj(hc);
    hr::GatedResidualHost(hx.data(), down.data(), up.data(), inject.data(),
                          H, hc, lr, 1e-6, ref_mixed.data(), ref_inj.data());

    CudaBuf d_hx(n * 4), d_down(lr * n * 4), d_up(n * lr * 4),
        d_inj(hc * n * 4);
    CudaBuf d_mixed(H * 4), d_inj_out(hc * 4);
    cudaMemcpy(d_hx.p, hx.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_down.p, down.data(), lr * n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up.p, up.data(), n * lr * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inj.p, inject.data(), hc * n * 4,
                cudaMemcpyHostToDevice);

    auto err = lsxcommon::qwen_moe_exp::RunExpHcGatedResidualF32(
        nullptr, d_hx.as<float>(), d_down.as<float>(), d_up.as<float>(),
        d_inj.as<float>(), /*seq=*/1, H, hc, lr, d_mixed.as<float>(),
        d_inj_out.as<float>());
    REQUIRE(err == cudaSuccess);
    std::vector<float> got_mixed(H), got_inj(hc);
    cudaMemcpy(got_mixed.data(), d_mixed.p, H * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_inj.data(), d_inj_out.p, hc * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < H; ++i)
        REQUIRE(got_mixed[i] == Catch::Approx(ref_mixed[i]).epsilon(2e-3));
    for (int i = 0; i < hc; ++i)
        REQUIRE(got_inj[i] == Catch::Approx(ref_inj[i]).epsilon(2e-3));
}

TEST_CASE("hc inject scatter kernel parity", "[qwen-exp-kernels]")
{
    constexpr int H = 64, hc = 4, n = H * hc;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> hx(n), bo(H), w(hc);
    for (auto &v : hx) v = dist(rng);
    for (auto &v : bo) v = dist(rng);
    for (auto &v : w) v = 0.5f + dist(rng);
    std::vector<__half> hxh(n);
    for (int i = 0; i < n; ++i) hxh[i] = __float2half(hx[i]);
    std::vector<__half> ref(n);
    for (int c = 0; c < hc; ++c)
        for (int i = 0; i < H; ++i)
            ref[c * H + i] = __float2half(__half2float(hxh[c * H + i]) +
                                          bo[i] * w[c]);

    std::vector<__half> boh(H);
    for (int i = 0; i < H; ++i) boh[i] = __float2half(bo[i]);
    CudaBuf d_hx(n * 2), d_bo(H * 2), d_w(hc * 4), d_out(n * 2);
    cudaMemcpy(d_hx.p, hxh.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bo.p, boh.data(), H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.p, w.data(), hc * 4, cudaMemcpyHostToDevice);
    auto err = lsxcommon::qwen_moe_exp::RunExpHcInjectScatter(
        nullptr, d_hx.as<__half>(), d_bo.as<__half>(), d_w.as<float>(),
        /*seq=*/1, H, hc, d_out.as<__half>());
    REQUIRE(err == cudaSuccess);
    std::vector<__half> got(n);
    cudaMemcpy(got.data(), d_out.p, n * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        REQUIRE(std::abs(__half2float(got[i]) - __half2float(ref[i])) <
                2e-3f);
}

TEST_CASE("QSA indexer degenerate path selects all tokens when ctx<=budget",
          "[qwen-exp-kernels]")
{
    // past_ctx=5, seq=1, budget 8 (block_topk 4 x compress 2): ctx=6 <= 8
    // -> full mask fast path.
    constexpr int HD = 8, CTX = 6;
    lsxcommon::qwen_moe_exp::ExpQsaIndexerState st;
    REQUIRE(lsxcommon::qwen_moe_exp::ExpQsaIndexerAlloc(st, 64, HD) ==
            cudaSuccess);
    CudaBuf d_q(4 * HD * 2), d_newk(HD * 2), d_mask(CTX * 4);
    auto err = lsxcommon::qwen_moe_exp::RunExpQsaIndexer(
        nullptr, st, d_q.as<__half>(), d_newk.as<__half>(),
        /*past_ctx=*/5, /*seq=*/1, HD, /*compress=*/2, /*block_topk=*/4,
        d_mask.as<int>());
    REQUIRE(err == cudaSuccess);
    std::vector<int> mask(CTX);
    cudaMemcpy(mask.data(), d_mask.p, CTX * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < CTX; ++i) REQUIRE(mask[i] == 1);
    lsxcommon::qwen_moe_exp::ExpQsaIndexerFree(st);
}

TEST_CASE("QSA indexer block scoring parity vs host", "[qwen-exp-kernels]")
{
    // compress=2, past_ctx=4, seq=1 -> 2 complete blocks; block_topk=1 forces
    // a real top-1 selection verified against host-recomputed relu-score.
    constexpr int HD = 8, PAST = 4, COMPRESS = 2;
    lsxcommon::qwen_moe_exp::ExpQsaIndexerState st;
    REQUIRE(lsxcommon::qwen_moe_exp::ExpQsaIndexerAlloc(st, 64, HD) ==
            cudaSuccess);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__half> keys(PAST * HD), q(4 * HD);
    for (auto &v : keys) v = __float2half(dist(rng));
    for (auto &v : q) v = __float2half(dist(rng));
    // Seed the pooled-key cache directly (test seam): block b pools
    // keys[2b..2b+1].
    std::vector<float> pooled(PAST / COMPRESS * HD);
    for (int b = 0; b < PAST / COMPRESS; ++b)
        for (int i = 0; i < HD; ++i)
            pooled[b * HD + i] =
                0.5f * (__half2float(keys[(2 * b) * HD + i]) +
                        __half2float(keys[(2 * b + 1) * HD + i]));
    cudaMemcpy(st.d_pooled_keys, pooled.data(), pooled.size() * 4,
               cudaMemcpyHostToDevice);
    std::vector<int> block_starts{0, 2};
    cudaMemcpy(st.d_block_starts, block_starts.data(), 2 * 4,
               cudaMemcpyHostToDevice);
    st.host_pooled_count = PAST / COMPRESS;  // test seam: pooled rows valid

    CudaBuf d_q(q.size() * 2), d_newk(HD * 2), d_mask(PAST * 4);
    cudaMemcpy(d_q.p, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    auto err = lsxcommon::qwen_moe_exp::RunExpQsaIndexer(
        nullptr, st, d_q.as<__half>(), d_newk.as<__half>(), PAST, 1, HD,
        COMPRESS, /*block_topk=*/1, d_mask.as<int>());
    REQUIRE(err == cudaSuccess);
    std::vector<int> mask(PAST);
    cudaMemcpy(mask.data(), d_mask.p, PAST * 4, cudaMemcpyDeviceToHost);

    // Host oracle: score_b = sum_h relu(q_h · pooled_b) / sqrt(HD).
    float best = -1.f;
    int best_b = -1;
    for (int b = 0; b < PAST / COMPRESS; ++b) {
        float s = 0.f;
        for (int h = 0; h < 4; ++h) {
            float dot = 0.f;
            for (int d = 0; d < HD; ++d)
                dot += __half2float(q[h * HD + d]) * pooled[b * HD + d];
            s += std::fmaxf(dot, 0.f);
        }
        s /= std::sqrt(static_cast<float>(HD));
        if (s > best) {
            best = s;
            best_b = b;
        }
    }
    // Only the winning block's tokens are selected.
    int sel = 0;
    for (int i = 0; i < PAST; ++i) sel += mask[i];
    REQUIRE(sel == COMPRESS);
    REQUIRE(mask[best_b * COMPRESS] == 1);
    lsxcommon::qwen_moe_exp::ExpQsaIndexerFree(st);
}
