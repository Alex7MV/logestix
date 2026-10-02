// GLM-5.3-Flash CUDA kernel unit tests (bit-parity against host refs).
#include <catch2/catch_test_macros.hpp>

#include "test_kernel_budget.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "glm_moe_flash_device_math.h"
#include "lsxcommon/glm_moe_flash_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/qwen_moe_inference.h"

#include <cstdio>

using namespace lsxcommon;
using namespace lsxcommon::glm_moe_flash;

namespace {

struct CudaGuard {
    CudaGuard() { REQUIRE(cudaFree(0) == cudaSuccess); }
};

std::mt19937 Rng() { return std::mt19937(1234); }

} // namespace

TEST_CASE("flash conv1d: causal depthwise window")
{
    CudaGuard g;
    // decode path: window of last (K-1) rows + current row.
    constexpr int C = 8, K = 4;
    std::vector<float> w(C * K), x(K * C);
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    for (auto &v : w) v = ud(rng);
    for (auto &v : x) v = ud(rng);
    // Host reference: y[c] = sum_k w[c*K + k] * x[k*C + c]
    std::vector<float> ref(C, 0.f);
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k) ref[c] += w[c * K + k] * x[k * C + c];
    // Device path via RunFlashConv1dF32 (test twin).
    float *d_x, *d_w, *d_y;
    REQUIRE(cudaMalloc(&d_x, x.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w, w.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_y, C * 4) == cudaSuccess);
    REQUIRE(
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice)
        == cudaSuccess);
    REQUIRE(
        cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice)
        == cudaSuccess);
    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    REQUIRE(RunFlashConv1dF32(s, d_x, d_w, d_y, C, K) == cudaSuccess);
    std::vector<float> y(C);
    REQUIRE(cudaMemcpyAsync(y.data(), d_y, C * 4, cudaMemcpyDeviceToHost, s)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    for (int c = 0; c < C; ++c) {
        INFO("c=" << c << " ref=" << ref[c] << " got=" << y[c]);
        REQUIRE(std::fabs(y[c] - ref[c]) < 1e-5f);
    }
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaStreamDestroy(s);
}

TEST_CASE("flash conv decode + fused silu is bit-exact")
{
    CudaGuard g;
    constexpr int C = 16, K = 4;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.5f, 1.5f);
    std::vector<__half> w(C * K), hist((K - 1) * C), x(C);
    for (auto &v : w) v = __float2half(ud(rng));
    for (auto &v : hist) v = __float2half(ud(rng));
    for (auto &v : x) v = __float2half(ud(rng));

    __half *d_w, *d_hist, *d_x, *d_ref, *d_fused;
    REQUIRE(cudaMalloc(&d_w, w.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hist, hist.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ref, C * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_fused, C * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w, w.data(), w.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hist, hist.data(), hist.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    // Reference: separate conv launch then in-place silu launch.
    REQUIRE(RunFlashConvDecodeTest(s, d_hist, d_x, d_w, d_ref, C, K, false)
            == cudaSuccess);
    REQUIRE(RunFlashSiluTest(s, d_ref, C) == cudaSuccess);
    // Fused: single launch with post-conv silu.
    REQUIRE(RunFlashConvDecodeTest(s, d_hist, d_x, d_w, d_fused, C, K, true)
            == cudaSuccess);

    std::vector<__half> ref(C), fused(C);
    REQUIRE(cudaMemcpyAsync(ref.data(), d_ref, C * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(fused.data(), d_fused, C * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    REQUIRE(std::memcmp(ref.data(), fused.data(), C * sizeof(__half)) == 0);

    cudaFree(d_w);
    cudaFree(d_hist);
    cudaFree(d_x);
    cudaFree(d_ref);
    cudaFree(d_fused);
    cudaStreamDestroy(s);
}

TEST_CASE("flash rmsnorm gated + fused gate act is bit-exact")
{
    CudaGuard g;
    constexpr int Hh = 4, D = 8;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    std::vector<float> core(Hh * D);
    std::vector<__half> gate(Hh * D), normw(D);
    for (auto &v : core) v = ud(rng);
    for (auto &v : gate) v = __float2half(ud(rng));
    for (auto &v : normw) v = __float2half(ud(rng));

    float *d_core;
    __half *d_gate, *d_gate_sep, *d_normw, *d_ref, *d_fused;
    REQUIRE(cudaMalloc(&d_core, core.size() * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate, gate.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate_sep, gate.size() * sizeof(__half))
            == cudaSuccess);
    REQUIRE(cudaMalloc(&d_normw, normw.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ref, Hh * D * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_fused, Hh * D * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_core, core.data(), core.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_normw, normw.data(), normw.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate_sep, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    for (const bool silu : {false, true}) {
        // Reference: in-place gate activation, then the norm kernel.
        REQUIRE(RunFlashGateActTest(s, d_gate_sep, Hh * D, silu)
                == cudaSuccess);
        REQUIRE(RunFlashRmsnormGatedTest(s, d_core, d_gate_sep, d_normw, d_ref,
                                         Hh, D, 1e-5f, false, silu)
                == cudaSuccess);
        // Fused: norm kernel applies the activation on the fly.
        REQUIRE(RunFlashRmsnormGatedTest(s, d_core, d_gate, d_normw, d_fused,
                                         Hh, D, 1e-5f, true, silu)
                == cudaSuccess);
        std::vector<__half> ref(Hh * D), fused(Hh * D);
        REQUIRE(cudaMemcpyAsync(ref.data(), d_ref, Hh * D * sizeof(__half),
                                cudaMemcpyDeviceToHost, s) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(fused.data(), d_fused, Hh * D * sizeof(__half),
                                cudaMemcpyDeviceToHost, s) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
        REQUIRE(std::memcmp(ref.data(), fused.data(),
                            Hh * D * sizeof(__half)) == 0);
        // Restore the raw gate for the next toggle iteration.
        REQUIRE(cudaMemcpy(d_gate_sep, gate.data(), gate.size() * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }

    cudaFree(d_core);
    cudaFree(d_gate);
    cudaFree(d_gate_sep);
    cudaFree(d_normw);
    cudaFree(d_ref);
    cudaFree(d_fused);
    cudaStreamDestroy(s);
}

TEST_CASE("flash l2norm q/k fused launch is bit-exact")
{
    CudaGuard g;
    constexpr int Hh = 4, D = 8;
    constexpr float q_scale = 0.08838834764831845f;  // 1/sqrt(128)
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    std::vector<__half> q(Hh * D), k(Hh * D);
    for (auto &v : q) v = __float2half(ud(rng));
    for (auto &v : k) v = __float2half(ud(rng));

    __half *d_q, *d_k, *d_qf, *d_kf;
    REQUIRE(cudaMalloc(&d_q, q.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_k, k.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_qf, q.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_kf, k.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q, q.data(), q.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_k, k.data(), k.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_qf, q.data(), q.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_kf, k.data(), k.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    // Reference: two separate single-plane launches.
    REQUIRE(RunFlashL2normTest(s, d_q, D, q_scale, Hh) == cudaSuccess);
    REQUIRE(RunFlashL2normTest(s, d_k, D, 1.0f, Hh) == cudaSuccess);
    // Fused: one launch covering q and k rows.
    REQUIRE(RunFlashL2normQKTest(s, d_qf, d_kf, D, q_scale, Hh) == cudaSuccess);

    std::vector<__half> qr(Hh * D), qf(Hh * D), kr(Hh * D), kf(Hh * D);
    REQUIRE(cudaMemcpyAsync(qr.data(), d_q, qr.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(qf.data(), d_qf, qf.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(kr.data(), d_k, kr.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(kf.data(), d_kf, kf.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    REQUIRE(std::memcmp(qr.data(), qf.data(), qr.size() * sizeof(__half)) == 0);
    REQUIRE(std::memcmp(kr.data(), kf.data(), kr.size() * sizeof(__half)) == 0);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_qf);
    cudaFree(d_kf);
    cudaStreamDestroy(s);
}

TEST_CASE("flash gated delta: device vs host parity (per-channel decay)")
{
    CudaGuard g;
    // Tiny geometry: heads=2, head_dim=4, seq=3, per-channel decay.
    constexpr int Hh = 2, D = 4, S = 3;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
    std::vector<float> q(S * Hh * D), k(S * Hh * D), v(S * Hh * D),
        beta(S * Hh), gdecay(S * Hh * D);
    for (auto &x : q) x = ud(rng);
    for (auto &x : k) x = ud(rng);
    for (auto &x : v) x = ud(rng);
    for (auto &x : beta) x = 1.f / (1.f + std::exp(-ud(rng) * 2.f));
    for (auto &x : gdecay) x = std::exp(-std::fabs(ud(rng)) * 0.3f);
    // Host reference.
    std::vector<float> ref_state(Hh * D * D, 0.f), ref_out(S * Hh * D);
    FlashGatedDeltaHost(q.data(), k.data(), v.data(), beta.data(),
                        gdecay.data(), S, Hh, D, ref_state.data(),
                        ref_out.data());
    // Device twin (raw F32 buffers, no FP16 conversions).
    float *dq, *dk, *dv, *db, *dg, *s_out, *o_out;
    REQUIRE(cudaMalloc(&dq, q.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dk, k.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dv, v.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&db, beta.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&dg, gdecay.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&s_out, ref_state.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&o_out, ref_out.size() * 4) == cudaSuccess);
    REQUIRE(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(db, beta.data(), beta.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dg, gdecay.data(), gdecay.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    cudaStream_t s;
    cudaStreamCreate(&s);
    REQUIRE(RunFlashGatedDeltaF32(s, dq, dk, dv, db, dg, S, Hh, D, s_out,
                                  o_out) == cudaSuccess);
    std::vector<float> got_state(ref_state.size()), got_out(ref_out.size());
    REQUIRE(cudaMemcpyAsync(got_state.data(), s_out, got_state.size() * 4,
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(got_out.data(), o_out, got_out.size() * 4,
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    for (size_t i = 0; i < got_state.size(); ++i) {
        INFO("state i=" << i << " ref=" << ref_state[i] << " got="
                        << got_state[i]);
        REQUIRE(std::fabs(got_state[i] - ref_state[i]) < 1e-4f);
    }
    for (size_t i = 0; i < got_out.size(); ++i) {
        INFO("out i=" << i << " ref=" << ref_out[i] << " got="
                      << got_out[i]);
        REQUIRE(std::fabs(got_out[i] - ref_out[i]) < 1e-4f);
    }
    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(db);
    cudaFree(dg);
    cudaFree(s_out);
    cudaFree(o_out);
    cudaStreamDestroy(s);
}

TEST_CASE("flash router: sigmoid noaux_tc top-8 + renorm + scaling")
{
    CudaGuard g;
    constexpr int E = 8, TOK = 2, HH = 64;
    std::vector<uint16_t> gate(E * HH);   // BF16 bits [E, H]
    std::vector<float> bias(E);
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-2.f, 2.f);
    for (auto &v : gate) {
        uint32_t u;
        float f = ud(rng);
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    for (auto &v : bias) v = ud(rng);
    // Hidden input (router consumes [TOK, H]); the kernel l2-dot with the
    // BF16 gate — parity via the same host ref math.
    constexpr int H = 64;
    std::vector<uint16_t> x(TOK * HH);
    for (auto &v : x) {
        uint32_t u;
        float f = ud(rng);
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    // Host reference.
    auto slots_ref = FlashRouterTop8Host(
        reinterpret_cast<const uint8_t *>(gate.data()), bias.data(),
        reinterpret_cast<const uint8_t *>(x.data()), TOK, H, E);
    REQUIRE(slots_ref.size() == TOK * 8);
    // Device.
    uint8_t *d_gate, *d_x;
    float *d_bias;
    qwen_moe::QwenMoeRoutedSlot *d_slots;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_bias, bias.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_slots, TOK * 8 * sizeof(qwen_moe::QwenMoeRoutedSlot))
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_bias, bias.data(), bias.size() * 4,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    cudaStream_t s;
    cudaStreamCreate(&s);
    REQUIRE(RunFlashRouterTop8(s, d_gate, d_bias, d_x, TOK, H, E, d_slots)
            == cudaSuccess);
    std::vector<qwen_moe::QwenMoeRoutedSlot> got(TOK * 8);
    REQUIRE(cudaMemcpyAsync(got.data(), d_slots,
                            got.size() * sizeof(got[0]),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    for (int t = 0; t < TOK; ++t) {
        float sum = 0;
        for (int kk = 0; kk < 8; ++kk) {
            INFO("t=" << t << " kk=" << kk
                      << " ref=" << slots_ref[t * 8 + kk].eid
                      << " got=" << got[t * 8 + kk].eid);
            REQUIRE(got[t * 8 + kk].eid == slots_ref[t * 8 + kk].eid);
            sum += got[t * 8 + kk].coeff;
        }
        // Renorm + routed_scaling: sum of coeff == 2.5.
        REQUIRE(std::fabs(sum - 2.5f) < 1e-3f);
    }
    // HF semantics pin: with bias == 0, coeff == choice score; with nonzero
    // bias the selected experts must still match the host reference (which
    // selects on choice but coeffs raw sigmoid). The host/device agreement
    // above plus the raw-sigmoid renorm invariant (sum == 2.5 regardless of
    // bias magnitudes) pins the semantics.
    {
        // Recompute raw sigmoid scores on host for token 0 and verify that
        // every device coeff matches raw-sigmoid renorm, NOT bias-added.
        // Addressing matches the kernel byte layout: row stride = 2*H bytes,
        // element (e, h) at uint16 index e*H + h.
        constexpr int H = 64, E = 8;
        std::vector<float> raw(E);
        for (int e = 0; e < E; ++e) {
            float dot = 0;
            for (int h = 0; h < H; ++h) {
                // BF16 bits -> f32 (shift left 16), matching the device dot.
                uint32_t gb = static_cast<uint32_t>(gate[e * H + h]) << 16;
                uint32_t xb = static_cast<uint32_t>(x[0 * H + h]) << 16;
                float g, xv;
                std::memcpy(&g, &gb, 4);
                std::memcpy(&xv, &xb, 4);
                dot += g * xv;
            }
            raw[e] = 1.0f / (1.0f + std::exp(-dot));
        }
        // Sum over the top-8 selected (by choice) of raw scores.
        float raw_sel_sum = 0;
        for (int kk = 0; kk < 8; ++kk) raw_sel_sum += raw[slots_ref[kk].eid];
        // Device coeffs must be raw-sigmoid renormed * 2.5 (NOT choice-score
        // renormed): compare directly per slot; skip near-zero raw values
        // where the division-free comparison is stable anyway.
        for (int kk = 0; kk < 8; ++kk) {
            const int eid = slots_ref[kk].eid;
            const float expect =
                raw[eid] / raw_sel_sum * 2.5f;
            INFO("kk=" << kk << " eid=" << eid << " raw=" << raw[eid]
                       << " expect=" << expect
                       << " got=" << got[kk].coeff);
            REQUIRE(std::fabs(got[kk].coeff - expect) < 1e-3f);
        }
    }
    cudaFree(d_gate);
    cudaFree(d_bias);
    cudaFree(d_x);
    cudaFree(d_slots);
    cudaStreamDestroy(s);
}


TEST_CASE("flash router F16: full-size 288 experts smoke")
{
    CudaGuard g;
    constexpr int E = 288, TOK = 1, HH = 4096;
    std::vector<uint16_t> gate(E * HH), x(TOK * HH);
    std::vector<float> bias(E);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-2.f, 2.f);
    for (auto &v : gate) {
        uint32_t u;
        float f = ud(rng);
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    for (auto &v : x) {
        uint32_t u;
        float f = ud(rng);
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    for (auto &v : bias) v = ud(rng);
    uint8_t *d_gate, *d_x;
    float *d_bias;
    qwen_moe::QwenMoeRoutedSlot *d_slots;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_bias, E * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_slots, TOK * 8 * sizeof(qwen_moe::QwenMoeRoutedSlot))
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_bias, bias.data(), E * 4, cudaMemcpyHostToDevice)
            == cudaSuccess);
    cudaStream_t s;
    cudaStreamCreate(&s);
    REQUIRE(RunFlashRouterTop8F16(s, d_gate, d_bias,
                                  reinterpret_cast<const __half *>(d_x), TOK,
                                  HH, E, d_slots) == cudaSuccess);
    std::vector<qwen_moe::QwenMoeRoutedSlot> got(TOK * 8);
    REQUIRE(cudaMemcpyAsync(got.data(), d_slots,
                            got.size() * sizeof(got[0]),
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    float sum = 0;
    for (int kk = 0; kk < 8; ++kk) {
        INFO("kk=" << kk << " eid=" << got[kk].eid);
        REQUIRE(got[kk].eid >= 0);
        REQUIRE(got[kk].eid < E);
        sum += got[kk].coeff;
    }
    REQUIRE(std::fabs(sum - 2.5f) < 1e-3f);
    cudaFree(d_gate);
    cudaFree(d_x);
    cudaFree(d_bias);
    cudaFree(d_slots);
    cudaStreamDestroy(s);
}

TEST_CASE("flash mla attention: single-token softmax sanity")
{
    CudaGuard g;
    constexpr int Heads = 4, D = 256;
    // q row: ones; k row: ones -> score = 256 * scale = 1.0; v row: ones
    // -> out = 1.0 exactly.
    std::vector<uint16_t> q(Heads * D, 0x3C00);      // fp16 1.0 bits
    std::vector<uint16_t> k(Heads * D, 0x3C00);
    std::vector<uint16_t> v(Heads * D, 0x3C00);
    __half *d_q, *d_k, *d_v, *d_out;
    REQUIRE(cudaMalloc(&d_q, q.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_k, k.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_v, v.size() * 2) == cudaSuccess);

    REQUIRE(cudaMalloc(&d_out, Heads * D * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float scale = 1.0f / sqrtf(256.0f);
    const size_t smem = (2 * D + 1 + 4096) * sizeof(float);
    REQUIRE(RunFlashMlaAttnTest(s, d_q, d_k, d_v, d_out, Heads, D, scale,
                                smem) == cudaSuccess);
    std::vector<uint16_t> out(Heads * D);
    REQUIRE(cudaMemcpyAsync(out.data(), d_out, out.size() * 2,
                            cudaMemcpyDeviceToHost, s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    auto f16 = [](uint16_t bits) {
        // FP16 -> float: __half2float-equivalent host conversion.
        const int sign = (bits >> 15) & 1;
        const int exp = (bits >> 10) & 0x1F;
        const int man = bits & 0x3FF;
        float v;
        if (exp == 0) v = std::ldexp(static_cast<float>(man), -24);
        else if (exp == 31) v = man ? NAN : INFINITY;
        else v = std::ldexp(static_cast<float>(man | 0x400), exp - 25);
        return sign ? -v : v;
    };
    for (int i = 0; i < Heads * D; ++i) {
        INFO("i=" << i << " head=" << (i / D) << " out=" << f16(out[i]));
        REQUIRE(std::fabs(f16(out[i]) - 1.0f) < 5e-2f);
        REQUIRE(!std::isnan(f16(out[i])));
    }
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaStreamDestroy(s);
}

TEST_CASE("flash_expf_glibc: exp(0)==1 on device")
{
    CudaGuard g;
    // Exercised indirectly through the router (scores finite) — here via a
    // device kernel writing exp(0) into a buffer.
    float *d_out;
    REQUIRE(cudaMalloc(&d_out, 64) == cudaSuccess);
    cudaStream_t s;
    cudaStreamCreate(&s);
    // The probe kernels write only the slots they print; zero the buffer so
    // the device->host copy never reads uninitialized device memory.
    REQUIRE(cudaMemsetAsync(d_out, 0, 64, s) == cudaSuccess);
    REQUIRE(RunFlashExp0Probe(s, d_out) == cudaSuccess);
    float hv[16] = {};
    REQUIRE(cudaMemcpyAsync(hv, d_out, 64, cudaMemcpyDeviceToHost, s)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    INFO("exp(0)=" << hv[0] << " exp(1)=" << hv[1] << " exp(-1)=" << hv[2]
                   << " builtin exp(0)=" << hv[3]);
    INFO("pieces: s=" << hv[4] << " kd=" << hv[5] << " ki=" << hv[6]);
    INFO("dbg: abstop=" << hv[15] << " fast=" << hv[6+8] << " z=" << hv[5+8]
                        << " kd=" << hv[4+8] << " s=" << hv[3+8]);
    REQUIRE(std::fabs(hv[0] - 1.0f) < 1e-6f);
    REQUIRE(std::fabs(hv[1] - 2.718281828f) < 1e-4f);
    REQUIRE(std::fabs(hv[2] - 0.36787944f) < 1e-4f);
    INFO("pieces: s=" << hv[4] << " kd=" << hv[5] << " ki=" << hv[6]);
    cudaFree(d_out);
    cudaStreamDestroy(s);
}

// ── Routed-FFN real-shape harness (TDD: batched expert chain) ──────────────
// Builds a synthetic 1-layer QwenMoeExpertStore at the frozen GLM routed
// geometry (H=4096, I=2048, E=8) plus an 8-slot device cache, then exercises
// the production RunFlashRoutedFFN entry. The per-expert path launches 8x4
// kernels per call; the batched chain does it in 4. Budget test pins the
// routed-chain latency so the batching lever cannot silently regress.
namespace {

struct RealRoutedHarness {
    qwen_moe::QwenMoeExpertStore store;
    qwen_moe::QwenMoeExpertDeviceCache cache;
    int H = 4096, I = 2048, S = 8;
    const uint8_t *slab = nullptr;
    __half *dx = nullptr, *yg = nullptr, *yu = nullptr, *y2 = nullptr;
    __half *ygu = nullptr, *hbuf = nullptr, *ypart = nullptr;
    float *acc = nullptr;
    qwen_moe::QwenMoeGvSlotDev *tbl = nullptr;
    int64_t pk = 0, sk = 0;

    bool Build()
    {
        const int64_t E = S;
        pk = static_cast<int64_t>(I) * H / 2;
        sk = static_cast<int64_t>(I) * H / 16;
        const int64_t per_expert = 3 * pk + 3 * sk;
        const size_t slab_bytes = static_cast<size_t>(per_expert) * E;
        uint8_t *pinned = nullptr;
        if (cudaMallocHost(reinterpret_cast<void **>(&pinned), slab_bytes) !=
            cudaSuccess)
            return false;
        std::memset(pinned, 0x21, slab_bytes);
        slab = pinned;
        store.num_layers = 1;
        store.num_experts = S;
        store.h_slabs.push_back(pinned);
        store.slab_bytes.push_back(slab_bytes);
        store.records.assign(1, std::vector<qwen_moe::QwenMoeExpertRecord>(E));
        for (int e = 0; e < E; ++e) {
            const int64_t base = e * per_expert;
            auto &r = store.records[0][e];
            r.w1_off = base;
            r.w1_bytes = pk;
            r.w2_off = base + pk;
            r.w2_bytes = pk;
            r.w3_off = base + 2 * pk;
            r.w3_bytes = pk;
            r.s1_off = base + 3 * pk;
            r.s1_bytes = sk;
            r.s2_off = base + 3 * pk + sk;
            r.s2_bytes = sk;
            r.s3_off = base + 3 * pk + 2 * sk;
            r.s3_bytes = sk;
            r.gs1 = r.gs2 = r.gs3 = 1.0f;
        }
        std::string err;
        if (!qwen_moe::AllocateQwenMoeExpertCache(store, S, cache, err))
            return false;
        if (cudaMalloc(&dx, static_cast<size_t>(H) * 2) != cudaSuccess ||
            cudaMalloc(&yg, static_cast<size_t>(2 * I) * 2) != cudaSuccess ||
            cudaMalloc(&yu, static_cast<size_t>(2 * I) * 2) != cudaSuccess ||
            cudaMalloc(&y2, static_cast<size_t>(H) * 2) != cudaSuccess ||
            cudaMalloc(&ygu, static_cast<size_t>(S) * 2 * I * 2) !=
                cudaSuccess ||
            cudaMalloc(&hbuf, static_cast<size_t>(S) * I * 2) != cudaSuccess ||
            cudaMalloc(&ypart, static_cast<size_t>(S) * H * 2) != cudaSuccess ||
            cudaMalloc(&acc, static_cast<size_t>(H) * 4) != cudaSuccess ||
            cudaMalloc(&tbl, sizeof(qwen_moe::QwenMoeGvSlotDev) * S) !=
                cudaSuccess)
            return false;
        cudaMemset(dx, 0, static_cast<size_t>(H) * 2);
        return true;
    }

    void Free()
    {
        qwen_moe::FreeQwenMoeExpertCache(cache);
        if (slab) cudaFreeHost(const_cast<uint8_t *>(slab));
        cudaFree(dx); cudaFree(yg); cudaFree(yu); cudaFree(y2);
        cudaFree(ygu); cudaFree(hbuf); cudaFree(ypart); cudaFree(acc);
        cudaFree(tbl);
        store.h_slabs.clear();
    }

    cudaError_t Call(cudaStream_t s)
    {
        qwen_moe::QwenMoeRoutedSlot slots[8];
        for (int k = 0; k < S; ++k) {
            slots[k].eid = k;
            slots[k].coeff = 1.0f / S;
        }
        cudaMemsetAsync(acc, 0, static_cast<size_t>(H) * 4, s);
        std::string err;
        return RunFlashRoutedFFN(s, store, cache, 0, /*kMoeFirstLayer=*/3,
                                 slots, S, dx, H, I, acc, yg, yu, y2, err,
                                 nullptr, nullptr, nullptr, ygu, hbuf, ypart,
                                 tbl);
    }
};

} // namespace

TEST_CASE("GLM routed FFN real-shape latency + determinism")
{
    using namespace lsxcommon::lsx_nvfp4;
    CudaGuard g;
    RealRoutedHarness h;
    REQUIRE(h.Build());
    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(true);

    for (int i = 0; i < 5; ++i) REQUIRE(h.Call(s) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    constexpr int kIters = 50;
    cudaEventRecord(a, s);
    for (int i = 0; i < kIters; ++i) REQUIRE(h.Call(s) == cudaSuccess);
    cudaEventRecord(b, s);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1000.0 / kIters;
    std::fprintf(stderr, "[ROUTED-FFN] H=%d I=%d S=%d avg=%.1f us\n", h.H,
                 h.I, h.S, us);

#ifdef NDEBUG
    // Release-only budget: the batched chain lands ~130 us; the legacy
    // per-expert chain (8x4 launches) is ~190 us.
    kernel_budget::Require("glm_flash.gdn.decode", us, kernel_budget::Kind::kMax,
                           "Gated DeltaNet decode step");
#endif

    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);
    h.Free();
    cudaStreamDestroy(s);
}
