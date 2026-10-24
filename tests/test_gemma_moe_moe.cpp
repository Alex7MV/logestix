// Gemma-4 MoE tests: router top-k (determinism, tie-break, scales),
// dense MLP (GELU-tanh) and NVFP4 expert FFN against CPU references.
#include <catch2/catch_test_macros.hpp>

#include "test_kernel_budget.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "lsxcommon/gemma_moe_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

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

TEST_CASE("router top-k rejects more experts than the launch block covers",
          "[gemma][moe]") {
    // The phase-2 softmax/top-k kernel launches with one thread per expert
    // (blockDim.x) and indexes the per-expert shared scratch by expert id;
    // num_experts above that block size would read past the covered range.
    constexpr int E = 129, H = 8, K = 1;
    std::vector<float> pf(E * H, 0.f);
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

    CHECK(lsxcommon::gemma::RunGemmaRouterTopK(dp, nullptr, dx, E, H, K, dsel,
                                               dw, 1, nullptr) ==
          cudaErrorInvalidValue);

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

// ── Dense MLP NVFP4 GEMV chain parity (Task 3 production path) ──────────────
// gate/up GEMV -> geGLU -> down GEMV, against a host reference that mirrors
// the repo's NVFP4 decode contract (E2M1 low/high nibbles, E8M0-style scale
// per 16 elems, e4==0 -> 0, global_scale multiply).
TEST_CASE("gemma dense MLP NVFP4 GEMV chain parity vs host reference",
          "[gemma][moe][nvfp4]") {
    using lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV;
    using lsxcommon::gpu::quantize_tensor_gpu;
    constexpr int H = 256, I = 128;   // 32-multiple shapes
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<uint8_t> gate_raw(I * H * 2), up_raw(I * H * 2),
        down_raw(H * I * 2);
    auto fill_bf16 = [&](std::vector<uint8_t> &v, float scale) {
        for (size_t i = 0; i < v.size() / 2; ++i) {
            __nv_bfloat16 b = __float2bfloat16(scale * nd(rng));
            uint16_t bits;
            std::memcpy(&bits, &b, 2);
            v[i * 2] = static_cast<uint8_t>(bits & 0xff);
            v[i * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        }
    };
    fill_bf16(gate_raw, 0.2f);
    fill_bf16(up_raw, 0.2f);
    fill_bf16(down_raw, 0.2f);

    std::vector<int64_t> gate_shape{I, H}, down_shape{H, I};
    std::vector<uint8_t> gp, gsc, up, usc, dp, dsc;
    float ggs, ugs, dgs;
    REQUIRE(quantize_tensor_gpu(gate_raw.data(), (int64_t)I * H, "BF16",
                                nullptr, gate_shape, {}, gp, gsc, ggs));
    REQUIRE(quantize_tensor_gpu(up_raw.data(), (int64_t)I * H, "BF16",
                                nullptr, gate_shape, {}, up, usc, ugs));
    REQUIRE(quantize_tensor_gpu(down_raw.data(), (int64_t)H * I, "BF16",
                                nullptr, down_shape, {}, dp, dsc, dgs));

    // x activations (BF16 -> FP16 on device, mirroring ForwardLayer)
    std::vector<float> xf(H);
    for (auto &v : xf) v = nd(rng);
    std::vector<__half> xh(H);
    for (int i = 0; i < H; ++i)
        xh[i] = __float2half(__bfloat162float(__float2bfloat16(xf[i])));

    uint8_t *dgp, *dgsc, *dup, *dusc, *ddp, *ddsc;
    __half *dxh, *dgu, *du, *dh, *dy;
    REQUIRE(cudaMalloc(&dgp, gp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgsc, gsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dup, up.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dusc, usc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddp, dp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddsc, dsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dxh, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgu, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dh, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dy, (size_t)H * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dgp, gp.data(), gp.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dgsc, gsc.data(), gsc.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dup, up.data(), up.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dusc, usc.data(), usc.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ddp, dp.data(), dp.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ddsc, dsc.data(), dsc.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dxh, xh.data(), H * 2, cudaMemcpyHostToDevice) == cudaSuccess);

    // Production entry (Task 3): full gate->geGLU->down chain on device.
    REQUIRE(lsxcommon::gemma::RunGemmaDenseMlpNvfp4Chain(
                nullptr, dgp, dgsc, ggs, dup, dusc, ugs, ddp, ddsc, dgs,
                dxh, dgu, du, dh, dy, H, I) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(H);
    REQUIRE(cudaMemcpy(hy.data(), dy, H * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // Host reference (nvfp4_gemv_math contract).
    auto e2m1 = [](int nib) {
        static const float t[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f,
                                    -4.f, -6.f};
        return t[nib & 0xF];
    };
    // Canonical block-scale decode (complete E4M3FN, single source
    // nvfp4_scale.h) — this replica mapped e4 == 0 to zero, dropping the
    // subnormal range the runtime now decodes.
    auto e4m3 = [](uint8_t b) {
        return lsxcommon::e4m3_to_float(b);
    };
    auto row_dot = [&](const std::vector<uint8_t> &packed,
                       const std::vector<uint8_t> &scales, float gs,
                       int64_t row, int cols,
                       const std::vector<__half> &xin) {
        float a = 0.f;
        const int64_t poff = row * (cols / 2);
        const int64_t soff = row * (cols / 16);
        for (int k = 0; k < cols; ++k) {
            uint8_t byte = packed[poff + k / 2];
            int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            a += __half2float(xin[k]) * e2m1(nib) * e4m3(scales[soff + k / 16]) *
                 gs;
        }
        return a;
    };
    float maxerr = 0.f;
    for (int c = 0; c < H; ++c) {
        std::vector<__half> hh(I);
        for (int j = 0; j < I; ++j) {
            // Mirror the device chain exactly: gate/up dots round to FP16
            // (half buffers) BEFORE the geGLU product, and the product
            // rounds to FP16 again.
            float g = __half2float(__float2half(
                row_dot(gp, gsc, ggs, j, H, xh)));
            float u = __half2float(__float2half(
                row_dot(up, usc, ugs, j, H, xh)));
            hh[j] = __float2half(GeluTanh(g) * u);
        }
        float ref = row_dot(dp, dsc, dgs, c, I, hh);
        float got = __half2float(hy[c]);
        maxerr = std::max(maxerr, std::abs(got - ref));
    }
    CHECK(maxerr < 0.05f);

    cudaFree(dgp); cudaFree(dgsc); cudaFree(dup); cudaFree(dusc);
    cudaFree(ddp); cudaFree(ddsc); cudaFree(dxh); cudaFree(dgu);
    cudaFree(du); cudaFree(dh); cudaFree(dy);
}

// The residual plane is decoded as a SECOND GEMV with acc=true, so each
// projection must equal "base dot rounded to FP16, then + residual dot rounded
// once more" -- not one dot over an exactly summed weight. Pin that arithmetic
// (it is what makes 4-bit + 4-bit behave like ~8 bits instead of like a noisy
// single 4-bit plane).
TEST_CASE("gemma dense MLP chain accumulates the residual planes with acc=true",
          "[gemma][moe][nvfp4][residual]") {
    using lsxcommon::gpu::quantize_tensor_gpu;
    using lsxcommon::gpu::quantize_residual_tensor_gpu;
    constexpr int H = 256, I = 128;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<uint8_t> gate_raw(I * H * 2), up_raw(I * H * 2),
        down_raw(H * I * 2);
    auto fill_bf16 = [&](std::vector<uint8_t> &v, float scale) {
        for (size_t i = 0; i < v.size() / 2; ++i) {
            __nv_bfloat16 b = __float2bfloat16(scale * nd(rng));
            uint16_t bits;
            std::memcpy(&bits, &b, 2);
            v[i * 2] = static_cast<uint8_t>(bits & 0xff);
            v[i * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        }
    };
    fill_bf16(gate_raw, 0.2f);
    fill_bf16(up_raw, 0.2f);
    fill_bf16(down_raw, 0.2f);

    const std::vector<int64_t> gate_shape{I, H}, down_shape{H, I};
    std::vector<uint8_t> gp, gsc, up, usc, dp, dsc;
    std::vector<uint8_t> gr, grsc, ur, ursc, dr, drsc;
    float ggs, ugs, dgs, ggs_r, ugs_r, dgs_r;
    REQUIRE(quantize_tensor_gpu(gate_raw.data(), (int64_t)I * H, "BF16",
                                nullptr, gate_shape, {}, gp, gsc, ggs));
    REQUIRE(quantize_tensor_gpu(up_raw.data(), (int64_t)I * H, "BF16", nullptr,
                                gate_shape, {}, up, usc, ugs));
    REQUIRE(quantize_tensor_gpu(down_raw.data(), (int64_t)H * I, "BF16",
                                nullptr, down_shape, {}, dp, dsc, dgs));
    REQUIRE(quantize_residual_tensor_gpu(gate_raw.data(), (int64_t)I * H,
                                         "BF16", gate_shape, gp, gsc, ggs, gr,
                                         grsc, ggs_r));
    REQUIRE(quantize_residual_tensor_gpu(up_raw.data(), (int64_t)I * H, "BF16",
                                         gate_shape, up, usc, ugs, ur, ursc,
                                         ugs_r));
    REQUIRE(quantize_residual_tensor_gpu(down_raw.data(), (int64_t)H * I,
                                         "BF16", down_shape, dp, dsc, dgs, dr,
                                         drsc, dgs_r));

    std::vector<float> xf(H);
    for (auto &v : xf) v = nd(rng);
    std::vector<__half> xh(H);
    for (int i = 0; i < H; ++i)
        xh[i] = __float2half(__bfloat162float(__float2bfloat16(xf[i])));

    uint8_t *dgp, *dgsc, *dup, *dusc, *ddp, *ddsc;
    uint8_t *dgr, *dgrsc, *dur, *dursc, *ddr, *ddrsc;
    __half *dxh, *dgu, *du, *dh, *dy;
    REQUIRE(cudaMalloc(&dgp, gp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgsc, gsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dup, up.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dusc, usc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddp, dp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddsc, dsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgr, gr.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgrsc, grsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dur, ur.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dursc, ursc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddr, dr.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ddrsc, drsc.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dxh, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dgu, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&du, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dh, (size_t)I * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dy, (size_t)H * 2) == cudaSuccess);
    auto up2 = [](void *dst, const std::vector<uint8_t> &v) {
        REQUIRE(cudaMemcpy(dst, v.data(), v.size(), cudaMemcpyHostToDevice) ==
                cudaSuccess);
    };
    up2(dgp, gp); up2(dgsc, gsc); up2(dup, up); up2(dusc, usc);
    up2(ddp, dp); up2(ddsc, dsc);
    up2(dgr, gr); up2(dgrsc, grsc); up2(dur, ur); up2(dursc, ursc);
    up2(ddr, dr); up2(ddrsc, drsc);
    REQUIRE(cudaMemcpy(dxh, xh.data(), H * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);

    lsxcommon::lsx_nvfp4::LsxNvfp4Weight g_res{}, u_res{}, d_res{};
    g_res.d_packed = dgr; g_res.d_scales = dgrsc; g_res.global_scale = ggs_r;
    u_res.d_packed = dur; u_res.d_scales = dursc; u_res.global_scale = ugs_r;
    d_res.d_packed = ddr; d_res.d_scales = ddrsc; d_res.global_scale = dgs_r;

    REQUIRE(lsxcommon::gemma::RunGemmaDenseMlpNvfp4Chain(
                nullptr, dgp, dgsc, ggs, dup, dusc, ugs, ddp, ddsc, dgs,
                dxh, dgu, du, dh, dy, H, I, &g_res, &u_res, &d_res) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(H);
    REQUIRE(cudaMemcpy(hy.data(), dy, H * 2, cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    auto e2m1 = [](int nib) {
        static const float t[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f,
                                    -4.f, -6.f};
        return t[nib & 0xF];
    };
    auto e4m3 = [](uint8_t b) { return lsxcommon::e4m3_to_float(b); };
    auto row_dot = [&](const std::vector<uint8_t> &packed,
                       const std::vector<uint8_t> &scales, float gs,
                       int64_t row, int cols, const std::vector<__half> &xin) {
        float a = 0.f;
        const int64_t poff = row * (cols / 2);
        const int64_t soff = row * (cols / 16);
        for (int k = 0; k < cols; ++k) {
            const uint8_t byte = packed[poff + k / 2];
            const int nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
            a += __half2float(xin[k]) * e2m1(nib) *
                 e4m3(scales[soff + k / 16]) * gs;
        }
        return a;
    };
    float maxerr = 0.f;
    for (int c = 0; c < H; ++c) {
        std::vector<__half> hh(I);
        for (int j = 0; j < I; ++j) {
            // Two-pass decode: base dot into FP16, then acc the residual.
            float g = __half2float(__float2half(
                row_dot(gp, gsc, ggs, j, H, xh)));
            g = __half2float(__float2half(
                g + row_dot(gr, grsc, ggs_r, j, H, xh)));
            float u = __half2float(__float2half(
                row_dot(up, usc, ugs, j, H, xh)));
            u = __half2float(__float2half(
                u + row_dot(ur, ursc, ugs_r, j, H, xh)));
            hh[j] = __float2half(GeluTanh(g) * u);
        }
        float ref = __half2float(__float2half(row_dot(dp, dsc, dgs, c, I, hh)));
        ref = __half2float(
            __float2half(ref + row_dot(dr, drsc, dgs_r, c, I, hh)));
        maxerr = std::max(maxerr, std::abs(__half2float(hy[c]) - ref));
    }
    CHECK(maxerr < 0.05f);

    cudaFree(dgp); cudaFree(dgsc); cudaFree(dup); cudaFree(dusc);
    cudaFree(ddp); cudaFree(ddsc); cudaFree(dgr); cudaFree(dgrsc);
    cudaFree(dur); cudaFree(dursc); cudaFree(ddr); cudaFree(ddrsc);
    cudaFree(dxh); cudaFree(dgu); cudaFree(du); cudaFree(dh); cudaFree(dy);
}

// The NVFP4 rows kernel decodes weight = E2M1(nibble) * e4m3_block_scale *
// table_global_scale. The latency test below passes gs = 1, where a dropped
// global scale is invisible -- exactly how gemma-4-26b shipped every logit
// inflated by 1/gs (gs = 0.1426, i.e. ~7x) until the softcap buried the
// near-ties and FSI-TEST-01/02 + CODE-TEST-02 flipped. Pin the numeric
// contract with gs != 1 against a hand-written CPU table, independent of the
// device codec.
TEST_CASE("gemma NVFP4 logits apply the table global scale",
          "[gemma][moe][logits]")
{
    constexpr int V = 8, H = 64;
    constexpr float gs = 0.25f;
    // Row r: every nibble is (r % 7) + 1 -> E2M1 magnitudes
    // {0.5, 1, 1.5, 2, 3, 4, 6}; every block scale is 1.0 (e4m3 0x38), x is
    // all ones, so logit[r] = H * 0.5_code(nibble) * gs.
    const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::vector<uint8_t> hp(static_cast<size_t>(V) * (H / 2));
    std::vector<uint8_t> hs(static_cast<size_t>(V) * (H / 16), 0x38);
    for (int r = 0; r < V; ++r) {
        const uint8_t nib = static_cast<uint8_t>((r % 7) + 1);
        const uint8_t byte = static_cast<uint8_t>(nib | (nib << 4));
        for (int i = 0; i < H / 2; ++i)
            hp[static_cast<size_t>(r) * (H / 2) + i] = byte;
    }
    std::vector<__nv_bfloat16> hx(H, __float2bfloat16(1.0f));

    uint8_t *dp = nullptr, *ds = nullptr;
    __nv_bfloat16 *dx = nullptr;
    float *d_logits = nullptr, *d_blkbest = nullptr, *d_max = nullptr;
    int *d_blkidx = nullptr;
    int32_t *d_argmax = nullptr;
    REQUIRE(cudaMalloc(&dp, hp.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, hs.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits, static_cast<size_t>(V) * 4) == cudaSuccess);
    const size_t blk_n = static_cast<size_t>(V) + V / 8 + 4096;
    REQUIRE(cudaMalloc(&d_blkbest, blk_n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_blkidx, blk_n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_max, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_argmax, 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), H * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(lsxcommon::gemma::RunGemmaLogitsArgmaxNvfp4(
                stream, dp, ds, gs, dx, V, H, /*cap=*/0.f, d_argmax, d_max,
                d_logits, d_blkbest, d_blkidx) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<float> logits(V);
    REQUIRE(cudaMemcpy(logits.data(), d_logits, V * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    int32_t best = -1;
    float maxv = 0.f;
    REQUIRE(cudaMemcpy(&best, d_argmax, 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, d_max, 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    for (int r = 0; r < V; ++r) {
        const double want =
            static_cast<double>(H) * kE2M1[(r % 7) + 1] * gs;
        INFO("row " << r << " got " << logits[r] << " want " << want);
        CHECK(std::abs(logits[r] - want) <= 1e-3 * std::abs(want));
    }
    CHECK(best == 6);      // the 6.0 table row (nibble 7)
    CHECK(std::abs(maxv - static_cast<float>(H) * 6.0f * gs) <= 0.05f);

    cudaStreamDestroy(stream);
    cudaFree(dp); cudaFree(ds); cudaFree(dx); cudaFree(d_logits);
    cudaFree(d_blkbest); cudaFree(d_blkidx); cudaFree(d_max);
    cudaFree(d_argmax);
}

// The LM head is one warp-per-vocab-row NVFP4 GEMV over the whole 262144-row
// embedding per token, i.e. the second-largest single kernel in gemma-4
// decode. Pin both the argmax contract and the streaming rate: the byte-scalar
// weight walk this guards against stalls the memory pipe at ~0.5 TB/s while
// uint4-chunk loads reach ~1.1 TB/s on the same bytes.
TEST_CASE("gemma NVFP4 logits decode latency budget",
          "[gemma][moe][logits][budget]")
{
#if defined(NDEBUG)
    constexpr int H = 2816, V = 262144;
    const size_t pb = static_cast<size_t>(V) * (H / 2);
    const size_t sb = static_cast<size_t>(V) * (H / 16);
    std::vector<uint8_t> hp(pb, 0x00);   // E2M1 0 -> zero logits
    std::vector<uint8_t> hs(sb, 0x08);   // 2^-6, irrelevant where nibble==0
    const int planted = V - 3;
    // Row `planted`: nibble 7 (6.0) with the max pow2 scale (2^8) dominates.
    for (size_t i = static_cast<size_t>(planted) * (H / 2);
         i < static_cast<size_t>(planted + 1) * (H / 2); ++i)
        hp[i] = 0x77;
    for (size_t i = static_cast<size_t>(planted) * (H / 16);
         i < static_cast<size_t>(planted + 1) * (H / 16); ++i)
        hs[i] = 0xF8;                     // e4=15 -> 2^8
    std::vector<__nv_bfloat16> hx(H, __float2bfloat16(1.0f));

    uint8_t *dp, *ds;
    __nv_bfloat16 *dx;
    float *d_logits, *d_blkbest, *d_max;
    int *d_blkidx;
    int32_t *d_argmax;
    REQUIRE(cudaMalloc(&dp, pb) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, sb) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits, static_cast<size_t>(V) * 4) == cudaSuccess);
    // The rows kernel stages one winner per block at blk_best[vocab + block],
    // so the buffers span vocab + grid (grid = V/8 here).
    const size_t blk_n = static_cast<size_t>(V) + V / 8 + 4096;
    REQUIRE(cudaMalloc(&d_blkbest, blk_n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_blkidx, blk_n * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_max, 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_argmax, 4) == cudaSuccess);
    cudaMemcpy(dp, hp.data(), pb, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), sb, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), H * 2, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    REQUIRE(lsxcommon::gemma::RunGemmaLogitsArgmaxNvfp4(
                stream, dp, ds, 1.0f, dx, V, H, /*cap=*/0.f, d_argmax, d_max,
                d_logits, d_blkbest, d_blkidx) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    int got = -1;
    cudaMemcpy(&got, d_argmax, 4, cudaMemcpyDeviceToHost);
    CHECK(got == planted);
    for (int i = 0; i < 20; ++i)
        REQUIRE(lsxcommon::gemma::RunGemmaLogitsArgmaxNvfp4(
                    stream, dp, ds, 1.0f, dx, V, H, 0.f, d_argmax, d_max,
                    d_logits, d_blkbest, d_blkidx) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    constexpr int kIters = 20;
    REQUIRE(cudaEventRecord(e0, stream) == cudaSuccess);
    for (int i = 0; i < kIters; ++i)
        REQUIRE(lsxcommon::gemma::RunGemmaLogitsArgmaxNvfp4(
                    stream, dp, ds, 1.0f, dx, V, H, 0.f, d_argmax, d_max,
                    d_logits, d_blkbest, d_blkidx) == cudaSuccess);
    REQUIRE(cudaEventRecord(e1, stream) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
    const double us = static_cast<double>(ms) * 1000.0 / kIters;
    INFO("NVFP4 LM-head latency = " << us << " us (" <<
         (static_cast<double>(pb + sb) / (us * 1e-6) / 1e9) << " GB/s)");
    // Byte-scalar weight walk ~957 us; uint4-chunk walk ~430 us.
    kernel_budget::Require("gemma.moe.decode", us, kernel_budget::Kind::kMax,
                           "gemma routed+shared MoE decode step");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(stream);
    cudaFree(dp); cudaFree(ds); cudaFree(dx); cudaFree(d_logits);
    cudaFree(d_blkbest); cudaFree(d_blkidx); cudaFree(d_max); cudaFree(d_argmax);
#endif
}
