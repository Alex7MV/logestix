// Тесты Gemma-4 MoE: router top-k (детерминизм, tie-break, скейлы),
// dense MLP (GELU-tanh) и NVFP4 expert FFN против CPU-референсов.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <random>
#include <vector>

#include "lsxcommon/gemma_moe_inference.h"

namespace {

std::vector<__nv_bfloat16> ToHalf(const std::vector<float> &v)
{
    std::vector<__nv_bfloat16> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2bfloat16(v[i]);
    return h;
}

float GeluTanh(float v)
{
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v *
           (1.f + std::tanh(k0 * (v + 0.044715f * v * v * v)));
}

inline bool BfTol(float got, double want)
{
    return std::abs(got - want) <=
           std::max(0.05, 0.0125 * std::abs(want));
}

} // namespace

TEST_CASE("RunGemmaDenseMLP matches CPU (gelu tanh geglu)", "[gemma][moe]") {
    constexpr int seq = 2, H = 32, I = 24;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf(seq * H), gf(I * H), uf(I * H), df(H * I);
    for (auto &x : xf) x = nd(rng);
    for (auto &x : gf) x = 0.2f * nd(rng);
    for (auto &x : uf) x = 0.2f * nd(rng);
    for (auto &x : df) x = 0.2f * nd(rng);

    auto hx = ToHalf(xf);
    auto hg = ToHalf(gf);
    auto hu = ToHalf(uf);
    auto hd = ToHalf(df);

    __nv_bfloat16 *dx, *dg, *du, *dd, *dout;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dg, hg.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, hu.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dd, hd.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, seq * H * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dg, hg.data(), hg.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(du, hu.data(), hu.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dd, hd.data(), hd.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    REQUIRE(lsxcommon::gemma::RunGemmaDenseMLP(dg, du, dd, I, H, dx, dout, seq,
                                               nullptr) == cudaSuccess);

    std::vector<__nv_bfloat16> hout(seq * H);
    REQUIRE(cudaMemcpy(hout.data(), dout, hout.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int t = 0; t < seq; ++t) {
        std::vector<double> act(I);
        for (int i = 0; i < I; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < H; ++c) {
                g += double(xf[t * H + c]) * double(gf[i * H + c]);
                u += double(xf[t * H + c]) * double(uf[i * H + c]);
            }
            act[i] = GeluTanh(static_cast<float>(g)) * u;
        }
        for (int c = 0; c < H; ++c) {
            double acc = 0;
            for (int i = 0; i < I; ++i)
                acc += act[i] * double(df[c * I + i]);
            INFO("t=" << t << " c=" << c << " got="
                      << __bfloat162float(hout[t * H + c]) << " want=" << acc);
            CHECK(BfTol(__bfloat162float(hout[t * H + c]), acc));
        }
    }

    cudaFree(dx); cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dout);
}

TEST_CASE("router top-k deterministic with lower-index tie-break",
          "[gemma][moe]") {
    constexpr int E = 8, H = 8, K = 2;
    // Crafted proj rows: experts 3 and 5 dominate; experts 1 and 6 tie exactly.
    std::vector<float> pf(E * H, 0.f);
    for (int c = 0; c < H; ++c) {
        pf[3 * H + c] = 0.5f;
        pf[5 * H + c] = 0.5f;
        pf[1 * H + c] = 0.1f;
        pf[6 * H + c] = 0.1f;
    }
    std::vector<float> xf(H, 1.0f);
    auto hp = ToHalf(pf);
    auto hx = ToHalf(xf);

    __nv_bfloat16 *dp, *dx;
    int32_t *dsel;
    float *dw;
    REQUIRE(cudaMalloc(&dp, hp.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dsel, sizeof(int32_t) * K) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, sizeof(float) * K) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    auto run_once = [&]() {
        REQUIRE(lsxcommon::gemma::RunGemmaRouterTopK(
                    dp, nullptr, dx, E, H, K, dsel, dw, 1, nullptr) ==
                cudaSuccess);
        std::vector<int32_t> sel(K);
        std::vector<float> w(K);
        REQUIRE(cudaMemcpy(sel.data(), dsel, sizeof(int32_t) * K,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(w.data(), dw, sizeof(float) * K,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        return std::make_pair(sel, w);
    };

    auto r1 = run_once();
    auto r2 = run_once();
    // Deterministic across runs.
    CHECK(r1.first == r2.first);
    // Winners are 3 and 5 (exact ties between 1/6 must not leak in).
    CHECK(r1.first[0] == 3);
    CHECK(r1.first[1] == 5);
    // Softmax over the two selected logits: equal -> 0.5 / 0.5.
    CHECK(std::abs(r1.second[0] - 0.5f) < 1e-3);
    CHECK(std::abs(r1.second[1] - 0.5f) < 1e-3);

    cudaFree(dp); cudaFree(dx); cudaFree(dsel); cudaFree(dw);
}



// Batched gather-dequant must be BIT-IDENTICAL to sequential per-expert
// dequant (same decode LUTs, same scale indexing) — the batched decode
// fast-path relies on it for bit-preserving parity with the per-expert path.
using lsxcommon::gemma::GemmaDequantBatch;
TEST_CASE("gemma batched dequant matches single-expert bits", "[gemma][moe][batch]")
{
    const int K = 5;
    const int64_t elems = 256 * 2 + 32;  // even and block-aligned (/16)
    const int64_t packed_bytes = elems / 2;
    const int64_t scales_bytes = elems / 16;

    std::mt19937_64 rng(12345);
    std::vector<std::vector<uint8_t>> packs(K), scales(K);
    GemmaDequantBatch batch{};
    float host_gs[16];
    for (int k = 0; k < K; ++k) {
        packs[k].resize(packed_bytes);
        scales[k].resize(scales_bytes);
        for (auto &b : packs[k]) b = static_cast<uint8_t>(rng() & 0xFF);
        // Include a zero-scale byte (e4==0 -> 0.0f decode contract).
        for (auto &b : scales[k]) b = static_cast<uint8_t>(rng() % 0x7F);
        scales[k][k % scales_bytes] = 0x00;
        batch.packs[k] = nullptr;   // filled after cudaMalloc below
        batch.scales[k] = nullptr;
        batch.global_scales[k] = 1.0f + 0.25f * (k + 1);
        host_gs[k] = batch.global_scales[k];
    }

    uint8_t *d_pack_all = nullptr, *d_scale_all = nullptr;
    REQUIRE(cudaMalloc(&d_pack_all, K * packed_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scale_all, K * scales_bytes) == cudaSuccess);
    for (int k = 0; k < K; ++k) {
        cudaMemcpy(d_pack_all + k * packed_bytes, packs[k].data(),
                   packed_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scale_all + k * scales_bytes, scales[k].data(),
                   scales_bytes, cudaMemcpyHostToDevice);
        batch.packs[k] = d_pack_all + k * packed_bytes;
        batch.scales[k] = d_scale_all + k * scales_bytes;
    }

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    __nv_bfloat16 *d_ref = nullptr, *d_got = nullptr;
    REQUIRE(cudaMalloc(&d_ref, K * elems * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_got, K * elems * 2) == cudaSuccess);

    // Reference: K sequential single-expert dequants.
    for (int k = 0; k < K; ++k) {
        REQUIRE(lsxcommon::gemma::RunGemmaSingleDequant(
                    stream, batch.packs[k], packed_bytes, batch.scales[k],
                    scales_bytes, host_gs[k],
                    d_ref + static_cast<size_t>(k) * elems,
                    elems) == cudaSuccess);
    }
    // Candidate: one gathered launch.
    REQUIRE(lsxcommon::gemma::RunGemmaBatchedDequant(
                stream, batch, elems, K, d_got) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__nv_bfloat16> h_ref(K * elems), h_got(K * elems);
    cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_got.data(), d_got, h_got.size() * 2, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(h_ref.data(), h_got.data(),
                      h_ref.size() * sizeof(__nv_bfloat16)) == 0);

    cudaFree(d_pack_all); cudaFree(d_scale_all);
    cudaFree(d_ref); cudaFree(d_got);
    cudaStreamDestroy(stream);
}

// Two-phase lm-head argmax must match a sequential per-row fp32 reference
// (same ascending-d accumulation order, deterministic tie-break).
TEST_CASE("gemma logits argmax two-phase matches sequential reference",
          "[gemma][moe][logits]")
{
    const int vocab = 1024 * 3 + 17;  // not a multiple of block/grid span
    const int hidden = 64;
    std::mt19937_64 rng(777);
    std::vector<__nv_bfloat16> embed(static_cast<size_t>(vocab) * hidden);
    for (auto &v : embed) v = __float2bfloat16((rng() % 2000 - 1000) / 512.0f);
    // Plant one clear global max row (positive, aligned embedding).
    const int best_v = vocab - 5;
    for (int d = 0; d < hidden; ++d)
        embed[static_cast<size_t>(best_v) * hidden + d] =
            __float2bfloat16(4.0f);
    std::vector<__nv_bfloat16> x(hidden);
    for (auto &v : x) v = __float2bfloat16(0.25f);

    __nv_bfloat16 *d_embed = nullptr, *d_x = nullptr;
    float *d_ref_logits = nullptr, *d_blkbest = nullptr;
    int32_t *d_idx = nullptr, *d_blkidx = nullptr;
    REQUIRE(cudaMalloc(&d_embed, embed.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ref_logits, static_cast<size_t>(vocab) *
                                           sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_blkbest,
                       (static_cast<size_t>(vocab) + 4096) *
                           sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_blkidx,
                       (static_cast<size_t>(vocab) + 4096) *
                           sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_idx, sizeof(int32_t)) == cudaSuccess);
    cudaMemcpy(d_embed, embed.data(), embed.size() * 2,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(lsxcommon::gemma::RunGemmaLogitsArgmax(
                stream, d_embed, d_x, vocab, hidden, /*cap=*/0.f, d_idx,
                nullptr, d_ref_logits, d_blkbest, d_blkidx) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    int got = -1;
    cudaMemcpy(&got, d_idx, sizeof(int), cudaMemcpyDeviceToHost);
    // CPU reference: identical ascending-d fp32 accumulation.
    float ref_best = -1e30f;
    int ref_arg = -1;
    std::vector<float> hl(static_cast<size_t>(vocab));
    for (int v = 0; v < vocab; ++v) {
        float acc = 0.f;
        for (int d = 0; d < hidden; ++d)
            acc += __bfloat162float(embed[static_cast<size_t>(v) * hidden +
                                           static_cast<size_t>(d)]) *
                   __bfloat162float(x[static_cast<size_t>(d)]);
        hl[static_cast<size_t>(v)] = acc;
        if (acc > ref_best) { ref_best = acc; ref_arg = v; }
    }
    WARN("got=" << got << " ref_arg=" << ref_arg
                << " best_v=" << best_v);
    CHECK(got == ref_arg);

    cudaFree(d_embed); cudaFree(d_x); cudaFree(d_ref_logits);
    cudaFree(d_blkbest); cudaFree(d_blkidx); cudaFree(d_idx);
    cudaStreamDestroy(stream);
}
