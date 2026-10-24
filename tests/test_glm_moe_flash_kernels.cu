// GLM-5.3-Flash CUDA kernel unit tests (bit-parity against host refs).
#include <catch2/catch_test_macros.hpp>

#include "test_kernel_budget.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "glm_moe_flash_constants.h"
#include "glm_moe_flash_device_math.h"
#include "lsxcommon/glm_moe_flash_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_moe_inference.h"

#include <cstdio>

using namespace lsxcommon::glm_moe_flash;
// NB: no blanket `using namespace lsxcommon` here — it makes the C math
// ::log ambiguous inside CUDA's own math_functions.hpp now that the neutral
// MoE header pulls Arrow in. Qualify the few names this test needs instead.
namespace moe = lsxcommon::moe;
namespace qwen_moe = lsxcommon::qwen_moe;
namespace gpu = lsxcommon::gpu;

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
    moe::MoERoutedSlot *d_slots;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_bias, bias.size() * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_slots, TOK * 8 * sizeof(moe::MoERoutedSlot))
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
    std::vector<moe::MoERoutedSlot> got(TOK * 8);
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
    moe::MoERoutedSlot *d_slots;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_x, x.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_bias, E * 4) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_slots, TOK * 8 * sizeof(moe::MoERoutedSlot))
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
    std::vector<moe::MoERoutedSlot> got(TOK * 8);
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
// Builds a synthetic 1-layer lsxcommon::moe::MoEExpertStore at the frozen GLM routed
// geometry (H=4096, I=2048, E=8) plus an 8-slot device cache, then exercises
// the production RunFlashRoutedFFN entry. The per-expert path launches 8x4
// kernels per call; the batched chain does it in 4. Budget test pins the
// routed-chain latency so the batching lever cannot silently regress.
namespace {

struct RealRoutedHarness {
    moe::MoEExpertStore store;
    qwen_moe::QwenMoeExpertDeviceCache cache;
    int H = 4096, I = 2048, S = 8;
    const uint8_t *slab = nullptr;
    __half *dx = nullptr, *yg = nullptr, *yu = nullptr, *y2 = nullptr;
    __half *ygu = nullptr, *hbuf = nullptr, *ypart = nullptr;
    float *acc = nullptr;
    moe::MoEGvSlotDev *tbl = nullptr;
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
        store.records.assign(1, std::vector<lsxcommon::moe::MoEExpertRecord>(E));
        for (int e = 0; e < E; ++e) {
            const int64_t base = e * per_expert;
            auto &r = store.records[0][e];
            r.slab_w1_off = base;
            r.w1_bytes = pk;
            r.slab_w2_off = base + pk;
            r.w2_bytes = pk;
            r.slab_w3_off = base + 2 * pk;
            r.w3_bytes = pk;
            r.slab_s1_off = base + 3 * pk;
            r.s1_bytes = sk;
            r.slab_s2_off = base + 3 * pk + sk;
            r.s2_bytes = sk;
            r.slab_s3_off = base + 3 * pk + 2 * sk;
            r.s3_bytes = sk;
            r.gs1 = r.gs2 = r.gs3 = 1.0f;
        }
        std::string err;
        if (!moe::AllocateMoEExpertCache(store, S, cache, err))
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
            cudaMalloc(&tbl, sizeof(moe::MoEGvSlotDev) * S) !=
                cudaSuccess)
            return false;
        cudaMemset(dx, 0, static_cast<size_t>(H) * 2);
        return true;
    }

    void Free()
    {
        moe::FreeMoEExpertCache(cache);
        if (slab) cudaFreeHost(const_cast<uint8_t *>(slab));
        cudaFree(dx); cudaFree(yg); cudaFree(yu); cudaFree(y2);
        cudaFree(ygu); cudaFree(hbuf); cudaFree(ypart); cudaFree(acc);
        cudaFree(tbl);
        store.h_slabs.clear();
    }

    cudaError_t Call(cudaStream_t s)
    {
        moe::MoERoutedSlot slots[8];
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

TEST_CASE("flash embed cache rebuilds when the weight epoch changes")
{
    CudaGuard g;
    constexpr int64_t n = 8;
    std::vector<__nv_bfloat16> a(n), b(n);
    for (int64_t i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = __float2bfloat16(1.0f);
        b[static_cast<size_t>(i)] = __float2bfloat16(2.0f);
    }
    __nv_bfloat16 *da = nullptr, *db = nullptr;
    REQUIRE(cudaMalloc(&da, n * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&db, n * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMemcpy(da, a.data(), n * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(db, b.data(), n * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    FlashEmbedCache cache;
    REQUIRE(cache.Ensure(0, da, n, /*weight_epoch=*/1) == cudaSuccess);
    __half *host = nullptr;
    REQUIRE(cudaMallocHost(&host, n * sizeof(__half)) == cudaSuccess);
    auto read_back = [&]() {
        REQUIRE(cudaMemcpy(host, cache.data, n * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        return __half2float(host[0]);
    };
    const __half *first = cache.data;
    CHECK(read_back() == 1.0f);

    // Same epoch + same source: reuse the table.
    REQUIRE(cache.Ensure(0, da, n, /*weight_epoch=*/1) == cudaSuccess);
    CHECK(cache.data == first);
    CHECK(read_back() == 1.0f);

    // Same epoch but a different live instance/source: rebuild. The epoch is
    // process-global, so it cannot distinguish two coexisting models.
    REQUIRE(cache.Ensure(0, db, n, /*weight_epoch=*/1) == cudaSuccess);
    CHECK(read_back() == 2.0f);

    // A new weight lifetime rebuilds even when the allocator hands back the
    // same address (pointer identity cannot be the oracle; the contents are).
    REQUIRE(cache.Ensure(0, da, n, /*weight_epoch=*/2) == cudaSuccess);
    CHECK(read_back() == 1.0f);
    cache.Free();
    CHECK(cache.data == nullptr);
    cudaFreeHost(host);
    cudaFree(da);
    cudaFree(db);
}

// ── DSA indexer (sparse attention) ─────────────────────────────────────────
// The indexer is its own lightweight attention: per-position keys (LayerNorm)
// and per-channel pool gates, softmax pooling with the learned `ape` offset,
// relu(dot)*dim^-0.5 summed over the indexer heads, then a top-k over pools
// expanded back to token indices. These tests pin the host replica of that
// math against the production kernels, and the selection contract (score
// order, visible tail, -1 padding, full coverage at `index_topk`).
namespace {

uint16_t ToBf16(float v)
{
    const __nv_bfloat16 b = __float2bfloat16_rn(v);
    uint16_t bits;
    std::memcpy(&bits, &b, sizeof(bits));
    return bits;
}

float FromBf16(uint16_t bits)
{
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &wide, sizeof(f));
    return f;
}

// pool[c] = sum_i softmax_i(gate_i[c] + ape[i][c]) * k_i[c] over the pool that
// starts at token `first`.
std::vector<double> IndexerPoolHost(const std::vector<__half> &keys,
                                    const std::vector<__half> &gates,
                                    const std::vector<uint16_t> &ape, int dim,
                                    int kpool, int64_t first)
{
    std::vector<double> out(static_cast<size_t>(dim));
    std::vector<double> lg(static_cast<size_t>(kpool));
    for (int c = 0; c < dim; ++c) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < kpool; ++i) {
            lg[static_cast<size_t>(i)] =
                static_cast<double>(__half2float(gates[(first + i) * dim + c])) +
                static_cast<double>(FromBf16(ape[i * dim + c]));
            mx = std::max(mx, lg[static_cast<size_t>(i)]);
        }
        double sum = 0.0, acc = 0.0;
        for (int i = 0; i < kpool; ++i) {
            const double p = std::exp(lg[static_cast<size_t>(i)] - mx);
            sum += p;
            acc += p * static_cast<double>(__half2float(keys[(first + i) * dim + c]));
        }
        out[static_cast<size_t>(c)] = acc / sum;
    }
    return out;
}

// scores[p] = sum_h w[h] * relu(dot(q_h, pool_p) * dim^-0.5).
std::vector<double> IndexerScoresHost(const std::vector<float> &q,
                                      const std::vector<float> &w,
                                      const std::vector<float> &pools,
                                      int heads, int dim, int n_pools)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<double> out(static_cast<size_t>(n_pools));
    for (int p = 0; p < n_pools; ++p) {
        double acc = 0.0;
        for (int h = 0; h < heads; ++h) {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d)
                dot += static_cast<double>(q[h * dim + d]) *
                       static_cast<double>(pools[p * dim + d]);
            acc += static_cast<double>(w[h]) *
                   std::max(dot * scale, 0.0);
        }
        out[static_cast<size_t>(p)] = acc;
    }
    return out;
}

// Pool indices in descending score order (ties by index).
std::vector<int> TopKPoolsHost(const std::vector<double> &scores, int k)
{
    std::vector<int> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });
    idx.resize(static_cast<size_t>(k));
    return idx;
}

struct DeviceBuf {
    void *p = nullptr;
    size_t bytes = 0;
    DeviceBuf() = default;
    explicit DeviceBuf(size_t n) : bytes(n)
    {
        REQUIRE(cudaMalloc(&p, n == 0 ? 1 : n) == cudaSuccess);
    }
    ~DeviceBuf() { cudaFree(p); }
    DeviceBuf(const DeviceBuf &) = delete;
    DeviceBuf &operator=(const DeviceBuf &) = delete;
    template <typename T>
    T *as() { return reinterpret_cast<T *>(p); }
};

cudaStream_t MakeStream()
{
    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    return s;
}

template <typename T>
void Upload(void *dst, const std::vector<T> &src)
{
    if (src.empty()) return;
    REQUIRE(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
}

template <typename T>
void Download(std::vector<T> &dst, const void *src)
{
    if (dst.empty()) return;
    REQUIRE(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

} // namespace

TEST_CASE("flash DSA indexer: pool softmax matches the host replica")
{
    CudaGuard g;
    constexpr int D = frozen::kIndexerHeadDim;
    constexpr int Kp = frozen::kIndexerKpool;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.5f, 1.5f);
    std::vector<__half> keys(D * Kp), gates(D * Kp);
    std::vector<uint16_t> ape(D * Kp);
    for (auto &v : keys) v = __float2half_rn(ud(rng));
    for (auto &v : gates) v = __float2half_rn(ud(rng) * 2.0f);
    for (auto &v : ape) v = ToBf16(ud(rng) * 0.5f);
    const std::vector<double> ref = IndexerPoolHost(keys, gates, ape, D, Kp, 0);

    DeviceBuf d_keys(keys.size() * sizeof(__half)), d_gates(gates.size() * sizeof(__half));
    DeviceBuf d_ape(ape.size() * sizeof(uint16_t)), d_pool(D * sizeof(float));
    Upload(d_keys.p, keys);
    Upload(d_gates.p, gates);
    Upload(d_ape.p, ape);
    const cudaStream_t s = MakeStream();

    // Twin: the pooling body on raw key/gate rows.
    REQUIRE(RunFlashIndexerPoolTest(s, d_keys.as<__half>(), d_gates.as<__half>(),
                                    d_ape.as<uint16_t>(), d_pool.as<float>(), D,
                                    Kp) == cudaSuccess);
    std::vector<float> got(D);
    Download(got, d_pool.p);
    for (int c = 0; c < D; ++c) {
        INFO("twin c=" << c << " ref=" << ref[static_cast<size_t>(c)]
                       << " got=" << got[static_cast<size_t>(c)]);
        REQUIRE(std::fabs(static_cast<double>(got[static_cast<size_t>(c)]) -
                          ref[static_cast<size_t>(c)]) < 3e-3);
    }

    // Softmax dominance pins the mix: with token 0 winning every channel by
    // 24 logits the pool is just token 0's key row.
    std::vector<__half> dom_gates(D * Kp);
    for (int i = 0; i < Kp; ++i)
        for (int c = 0; c < D; ++c)
            dom_gates[i * D + c] = __float2half_rn(i == 0 ? 12.0f : -12.0f);
    Upload(d_gates.p, dom_gates);
    REQUIRE(RunFlashIndexerPoolTest(s, d_keys.as<__half>(), d_gates.as<__half>(),
                                    d_ape.as<uint16_t>(), d_pool.as<float>(), D,
                                    Kp) == cudaSuccess);
    Download(got, d_pool.p);
    for (int c = 0; c < D; ++c) {
        INFO("dominant c=" << c << " key0=" << __half2float(keys[static_cast<size_t>(c)])
                           << " got=" << got[static_cast<size_t>(c)]);
        REQUIRE(std::fabs(static_cast<double>(got[static_cast<size_t>(c)]) -
                          static_cast<double>(__half2float(keys[static_cast<size_t>(c)])))
                < 2e-3);
    }

    // Production driver: same host reference through FlashIndexerState, which
    // also pins n_pools bookkeeping (one pool advances the counter once).
    Upload(d_gates.p, gates);
    FlashIndexerState st;
    REQUIRE(FlashIndexerStateAlloc(st, /*max_ctx=*/8) == cudaSuccess);
    REQUIRE(st.n_pools == 0);
    REQUIRE(st.pool_cap == 8 / Kp + 1);
    Upload(st.d_k, keys);
    Upload(st.d_gate, gates);
    FlashLayerWeights lw{};
    lw.d_idx_ape = d_ape.as<uint16_t>();
    REQUIRE(RunFlashIndexerPool(s, st, lw, /*first_tok=*/0) == cudaSuccess);
    REQUIRE(st.n_pools == 1);
    std::vector<float> prod(D);
    Download(prod, st.d_pool);
    for (int c = 0; c < D; ++c) {
        INFO("driver c=" << c << " ref=" << ref[static_cast<size_t>(c)]
                         << " got=" << prod[static_cast<size_t>(c)]);
        REQUIRE(std::fabs(static_cast<double>(prod[static_cast<size_t>(c)]) -
                          ref[static_cast<size_t>(c)]) < 3e-3);
    }
    FlashIndexerStateFree(st);
    cudaStreamDestroy(s);
}

TEST_CASE("flash DSA indexer: relu comes before the head weight")
{
    CudaGuard g;
    constexpr int Heads = frozen::kIndexerHeads;
    constexpr int D = frozen::kIndexerHeadDim;
    constexpr int Pools = 17;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> q(Heads * D), w(Heads), pools(Pools * D);
    for (auto &v : q) v = ud(rng);
    for (auto &v : w) v = ud(rng) * 2.0f;
    for (auto &v : pools) v = ud(rng);
    const std::vector<double> ref = IndexerScoresHost(q, w, pools, Heads, D, Pools);

    DeviceBuf d_q(q.size() * 4), d_w(w.size() * 4), d_pools(pools.size() * 4);
    DeviceBuf d_scores(Pools * 4);
    Upload(d_q.p, q);
    Upload(d_w.p, w);
    Upload(d_pools.p, pools);
    const cudaStream_t s = MakeStream();
    REQUIRE(RunFlashIndexerScoresTest(s, d_q.as<float>(), d_w.as<float>(),
                                      d_pools.as<float>(), Heads, D, Pools,
                                      d_scores.as<float>()) == cudaSuccess);
    std::vector<float> got(Pools);
    Download(got, d_scores.p);
    for (int p = 0; p < Pools; ++p) {
        INFO("p=" << p << " ref=" << ref[static_cast<size_t>(p)]
                  << " got=" << got[static_cast<size_t>(p)]);
        REQUIRE(std::fabs(static_cast<double>(got[static_cast<size_t>(p)]) -
                          ref[static_cast<size_t>(p)]) <
                2e-4 * std::max(1.0, std::fabs(ref[static_cast<size_t>(p)])));
    }

    SECTION("a negative dot with a negative weight contributes nothing")
    {
        // relu(dot) before the weighting keeps the contribution at 0; relu
        // after the weighting would add |dot| = 11.31 here.
        std::vector<float> qn(Heads * D, 1.0f), wn(Heads, 0.0f);
        std::vector<float> pn(Pools * D, 0.0f);
        for (int d = 0; d < D; ++d) pn[static_cast<size_t>(d)] = -1.0f;
        wn[0] = -1.0f;
        Upload(d_q.p, qn);
        Upload(d_w.p, wn);
        Upload(d_pools.p, pn);
        REQUIRE(RunFlashIndexerScoresTest(s, d_q.as<float>(), d_w.as<float>(),
                                          d_pools.as<float>(), Heads, D, Pools,
                                          d_scores.as<float>()) == cudaSuccess);
        Download(got, d_scores.p);
        for (int p = 0; p < Pools; ++p) {
            INFO("relu-before-weighting p=" << p << " got="
                                            << got[static_cast<size_t>(p)]);
            REQUIRE(std::fabs(got[static_cast<size_t>(p)]) < 1e-6f);
        }
    }

    SECTION("a positive dot carries the weighted, dim-scaled magnitude")
    {
        std::vector<float> qn(Heads * D, 1.0f), wn(Heads, 0.0f);
        std::vector<float> pn(Pools * D, 0.0f);
        for (int d = 0; d < D; ++d) pn[static_cast<size_t>(d)] = 1.0f;
        wn[0] = 2.0f;
        Upload(d_q.p, qn);
        Upload(d_w.p, wn);
        Upload(d_pools.p, pn);
        REQUIRE(RunFlashIndexerScoresTest(s, d_q.as<float>(), d_w.as<float>(),
                                          d_pools.as<float>(), Heads, D, Pools,
                                          d_scores.as<float>()) == cudaSuccess);
        Download(got, d_scores.p);
        // 2 * (128 * 128^-0.5) = 2 * sqrt(128) for pool 0; the other pools are
        // zero rows, so their dot (and therefore the score) is 0.
        const double expect = 2.0 * std::sqrt(static_cast<double>(D));
        INFO("scaled p=0 expect=" << expect << " got=" << got[0]);
        REQUIRE(std::fabs(static_cast<double>(got[0]) - expect) < 1e-4 * expect);
        for (int p = 1; p < Pools; ++p) {
            INFO("scaled zero-row p=" << p << " got=" << got[static_cast<size_t>(p)]);
            REQUIRE(std::fabs(got[static_cast<size_t>(p)]) < 1e-6f);
        }
    }
    cudaStreamDestroy(s);
}

TEST_CASE("flash DSA indexer: selection contract (order, tail, padding)")
{
    CudaGuard g;
    constexpr int Topk = frozen::kIndexerTopk;   // 2048
    constexpr int Kp = frozen::kIndexerKpool;    // 4
    constexpr int Width = Topk + Kp - 1;         // 2051
    const cudaStream_t s = MakeStream();
    DeviceBuf d_scores(600 * 4), d_sel((Topk + Kp) * 4), d_count(4);

    auto run = [&](const std::vector<float> &scores, int select_k,
                   int64_t visible) {
        REQUIRE(static_cast<int>(scores.size()) <= 600);
        Upload(d_scores.p, scores);
        REQUIRE(cudaMemsetAsync(d_count.p, 0, 4, s) == cudaSuccess);
        REQUIRE(RunFlashIndexerSelectKTest(s, d_scores.as<float>(),
                                           static_cast<int>(scores.size()),
                                           select_k, Kp, visible, Topk,
                                           d_sel.as<int32_t>(),
                                           d_count.as<int32_t>()) == cudaSuccess);
        std::vector<int32_t> sel(static_cast<size_t>(Topk + Kp), -7);
        std::vector<int32_t> count(1, -7);
        Download(sel, d_sel.p);
        Download(count, d_count.p);
        return std::make_pair(sel, count[0]);
    };

    SECTION("at index_topk the selection is the whole context, once")
    {
        // 512 pools == 2048 visible tokens; the 2049th token is the tail.
        std::vector<float> scores(512);
        for (int p = 0; p < 512; ++p) scores[static_cast<size_t>(p)] = 512.0f - p;
        auto [sel, count] = run(scores, 512, /*visible=*/2049);
        REQUIRE(count == 2049);
        std::vector<int> seen(2049, 0);
        for (int i = 0; i < count; ++i) {
            INFO("i=" << i << " sel=" << sel[static_cast<size_t>(i)]);
            REQUIRE(sel[static_cast<size_t>(i)] >= 0);
            REQUIRE(sel[static_cast<size_t>(i)] < 2049);
            ++seen[static_cast<size_t>(sel[static_cast<size_t>(i)])];
        }
        for (int i = 0; i < 2049; ++i) {
            INFO("coverage i=" << i << " seen=" << seen[static_cast<size_t>(i)]);
            REQUIRE(seen[static_cast<size_t>(i)] == 1);
        }
        // Descending scores expand in pool order: identity.
        for (int i = 0; i < count; ++i)
            REQUIRE(sel[static_cast<size_t>(i)] == i);
        // Padding fills the rest of the fixed-width window.
        for (int i = count; i < Width; ++i)
            REQUIRE(sel[static_cast<size_t>(i)] == -1);
    }

    SECTION("the visible tail is appended after the selected pools")
    {
        // 512 full pools == 2048 tokens, 2 more visible: the tail carries
        // both, so the union is still exactly the context.
        std::vector<float> scores(512);
        for (int p = 0; p < 512; ++p) scores[static_cast<size_t>(p)] = 1.0f + p;
        auto [sel, count] = run(scores, 512, /*visible=*/2050);
        REQUIRE(count == 2050);
        std::vector<int> seen(2050, 0);
        for (int i = 0; i < count; ++i) {
            REQUIRE(sel[static_cast<size_t>(i)] >= 0);
            REQUIRE(sel[static_cast<size_t>(i)] < 2050);
            ++seen[static_cast<size_t>(sel[static_cast<size_t>(i)])];
        }
        for (int i = 0; i < 2050; ++i)
            REQUIRE(seen[static_cast<size_t>(i)] == 1);
        // Ascending scores: the last pool wins, expanded in pool order.
        REQUIRE(sel[0] == 511 * Kp);
        REQUIRE(sel[1] == 511 * Kp + 1);
        REQUIRE(sel[count - 2] == 2048);
        REQUIRE(sel[count - 1] == 2049);
        for (int i = count; i < Width; ++i)
            REQUIRE(sel[static_cast<size_t>(i)] == -1);
    }

    SECTION("one pool is dropped once the context passes index_topk")
    {
        // 513 pools == 2052 visible: top 512 pools, no tail, and the
        // lowest-scoring pool (p = 512 here, descending scores) is the one
        // that must not appear.
        std::vector<float> scores(513);
        for (int p = 0; p < 513; ++p) scores[static_cast<size_t>(p)] = 513.0f - p;
        auto [sel, count] = run(scores, 512, /*visible=*/2052);
        REQUIRE(count == 2048);
        std::vector<int> seen(2052, 0);
        for (int i = 0; i < count; ++i) {
            REQUIRE(sel[static_cast<size_t>(i)] >= 0);
            REQUIRE(sel[static_cast<size_t>(i)] < 2048);
            ++seen[static_cast<size_t>(sel[static_cast<size_t>(i)])];
        }
        for (int i = 0; i < 2048; ++i)
            REQUIRE(seen[static_cast<size_t>(i)] == 1);
        for (int i = 2048; i < 2052; ++i)
            REQUIRE(seen[static_cast<size_t>(i)] == 0);
        REQUIRE(sel[2047] == 2047);
    }

    SECTION("random scores: the order matches the host top-k")
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
        std::vector<float> scores(600);
        for (auto &v : scores) v = ud(rng);
        std::vector<double> as_double(scores.begin(), scores.end());
        const std::vector<int> expect = TopKPoolsHost(as_double, 512);
        // The 512th/513th gap must be clear, otherwise the discrete order is
        // not a fair oracle (the reference implementation's topk is
        // unspecified on exact ties too).
        std::vector<double> sorted = as_double;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        REQUIRE(sorted[511] - sorted[512] > 1e-3);
        auto [sel, count] = run(scores, 512, /*visible=*/2052);
        REQUIRE(count == 2048);
        for (int r = 0; r < 512; ++r)
            for (int i = 0; i < Kp; ++i) {
                INFO("r=" << r << " i=" << i << " expect="
                          << expect[static_cast<size_t>(r)] * Kp + i
                          << " got=" << sel[static_cast<size_t>(r * Kp + i)]);
                REQUIRE(sel[static_cast<size_t>(r * Kp + i)] ==
                        expect[static_cast<size_t>(r)] * Kp + i);
            }
        for (int i = count; i < Width; ++i)
            REQUIRE(sel[static_cast<size_t>(i)] == -1);
    }

    SECTION("twenty thousand pools: the footprint does not scale with context")
    {
        // 80k positions of visible context. The selection footprint is
        // constant (256-bin histogram + the 512-entry prefix), so this is the
        // shape that a per-context bitonic sort could not serve.
        constexpr int Pools = 20000;
        std::mt19937 rng2(7);
        std::uniform_real_distribution<float> ud(-1.0f, 5.0f);
        std::vector<float> scores(Pools);
        for (auto &v : scores) v = ud(rng2);
        std::vector<double> as_double(scores.begin(), scores.end());
        const std::vector<int> expect = TopKPoolsHost(as_double, 512);
        DeviceBuf big_scores(scores.size() * 4);
        DeviceBuf big_sel((Topk + Kp) * 4), big_count(4);
        Upload(big_scores.p, scores);
        REQUIRE(cudaMemsetAsync(big_count.p, 0, 4, s) == cudaSuccess);
        REQUIRE(RunFlashIndexerSelectKTest(s, big_scores.as<float>(), Pools, 512,
                                           Kp, static_cast<int64_t>(Pools) * Kp,
                                           Topk, big_sel.as<int32_t>(),
                                           big_count.as<int32_t>()) == cudaSuccess);
        std::vector<int32_t> sel(static_cast<size_t>(Topk + Kp), -7);
        std::vector<int32_t> count(1, -7);
        Download(sel, big_sel.p);
        Download(count, big_count.p);
        REQUIRE(count[0] == 2048);
        for (int r = 0; r < 512; ++r)
            for (int i = 0; i < Kp; ++i) {
                INFO("r=" << r << " i=" << i << " expect="
                          << expect[static_cast<size_t>(r)] * Kp + i
                          << " got=" << sel[static_cast<size_t>(r * Kp + i)]);
                REQUIRE(sel[static_cast<size_t>(r * Kp + i)] ==
                        expect[static_cast<size_t>(r)] * Kp + i);
            }
        for (int i = count[0]; i < Width; ++i)
            REQUIRE(sel[static_cast<size_t>(i)] == -1);
    }
    cudaStreamDestroy(s);
}

TEST_CASE("flash DSA sparse MLA attends only the selected positions")
{
    CudaGuard g;
    constexpr int Heads = 4;
    constexpr int D = 256;
    constexpr int Cap = 128;
    constexpr int NSel = 64;
    constexpr int Pad = 3;
    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<__half> q(Heads * D), kv(static_cast<size_t>(Cap) * Heads * D);
    std::vector<__half> vv(static_cast<size_t>(Cap) * Heads * D);
    for (auto &v : q) v = __float2half_rn(ud(rng));
    for (auto &v : kv) v = __float2half_rn(ud(rng));
    for (auto &v : vv) v = __float2half_rn(ud(rng));

    // A shuffled subset of 64 positions plus -1 padding, as the selection
    // kernel emits it.
    std::vector<int32_t> sel(Cap);
    std::iota(sel.begin(), sel.end(), 0);
    std::shuffle(sel.begin(), sel.end(), rng);
    sel.resize(NSel + Pad);
    for (int i = NSel; i < NSel + Pad; ++i) sel[static_cast<size_t>(i)] = -1;

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    // Host reference over the same selected positions, in the same order.
    std::vector<double> ref(Heads * D, 0.0);
    std::vector<double> sc(NSel + Pad), p(NSel + Pad);
    for (int h = 0; h < Heads; ++h) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < NSel + Pad; ++i) {
            const int t = sel[static_cast<size_t>(i)];
            if (t < 0) continue;
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += static_cast<double>(__half2float(q[h * D + d])) *
                       static_cast<double>(
                           __half2float(kv[(static_cast<size_t>(t) * Heads + h) * D + d]));
            sc[static_cast<size_t>(i)] = dot * scale;
            mx = std::max(mx, sc[static_cast<size_t>(i)]);
        }
        double psum = 0.0;
        for (int i = 0; i < NSel + Pad; ++i) {
            const int t = sel[static_cast<size_t>(i)];
            if (t < 0) continue;
            p[static_cast<size_t>(i)] = std::exp(sc[static_cast<size_t>(i)] - mx);
            psum += p[static_cast<size_t>(i)];
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int i = 0; i < NSel + Pad; ++i) {
                const int t = sel[static_cast<size_t>(i)];
                if (t < 0) continue;
                acc += p[static_cast<size_t>(i)] / psum *
                       static_cast<double>(
                           __half2float(vv[(static_cast<size_t>(t) * Heads + h) * D + d]));
            }
            ref[static_cast<size_t>(h * D + d)] = acc;
        }
    }

    DeviceBuf d_q(q.size() * sizeof(__half));
    DeviceBuf d_k(kv.size() * sizeof(__half)), d_v(vv.size() * sizeof(__half));
    DeviceBuf d_sel(sel.size() * 4), d_count(4);
    DeviceBuf d_out(Heads * D * sizeof(__half));
    Upload(d_q.p, q);
    Upload(d_k.p, kv);
    Upload(d_v.p, vv);
    Upload(d_sel.p, sel);
    const cudaStream_t s = MakeStream();
    const size_t smem = (static_cast<size_t>(D) + NSel + Pad + 256) * sizeof(float);
    REQUIRE(RunFlashMlaAttnSelected(s, d_q.as<__half>(), d_k.as<__half>(),
                                    d_v.as<__half>(), d_sel.as<int32_t>(),
                                    d_count.as<int32_t>(), d_out.as<__half>(),
                                    NSel + Pad, Heads, D, scale,
                                    smem) == cudaSuccess);
    std::vector<__half> got(Heads * D);
    Download(got, d_out.p);
    for (int i = 0; i < Heads * D; ++i) {
        INFO("h=" << i / D << " d=" << i % D
                  << " ref=" << ref[static_cast<size_t>(i)]
                  << " got=" << __half2float(got[static_cast<size_t>(i)]));
        REQUIRE(std::fabs(static_cast<double>(__half2float(got[static_cast<size_t>(i)])) -
                          ref[static_cast<size_t>(i)]) < 3e-3);
    }
    cudaStreamDestroy(s);
}

TEST_CASE("flash DSA indexer end-to-end: token/pool/select chain vs host")
{
    CudaGuard g;
    constexpr int H = frozen::kHidden;
    constexpr int QL = frozen::kQLora;
    constexpr int Heads = frozen::kIndexerHeads;
    constexpr int D = frozen::kIndexerHeadDim;
    constexpr int Kp = frozen::kIndexerKpool;
    constexpr int Tok = 2 * Kp;  // two complete pools

    std::mt19937 rng = Rng();
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<__half> x(Tok * H), resid(QL);
    std::vector<uint16_t> wk(D * H), gate_w(D * H);
    std::vector<uint16_t> wqb(static_cast<size_t>(Heads * D) * QL);
    std::vector<uint16_t> wproj(Heads * H), knw(D), knb(D), ape(Kp * D);
    for (auto &v : x) v = __float2half_rn(ud(rng));
    for (auto &v : resid) v = __float2half_rn(ud(rng));
    for (auto &v : wk) v = ToBf16(ud(rng) * 0.02f);
    for (auto &v : gate_w) v = ToBf16(ud(rng) * 0.02f);
    for (auto &v : wqb) v = ToBf16(ud(rng) * 0.02f);
    for (auto &v : wproj) v = ToBf16(ud(rng) * 0.02f);
    for (auto &v : knw) v = ToBf16(1.0f + ud(rng) * 0.1f);
    for (auto &v : knb) v = ToBf16(ud(rng) * 0.05f);
    for (auto &v : ape) v = ToBf16(ud(rng) * 0.1f);

    // Host replica: BF16 GEMV (FP32 accumulate, FP16 store), LayerNorm(eps
    // 1e-6) with bias, pooling, then the query/weight projections.
    auto gemv = [](const uint16_t *w, const __half *v, int n_dim, int k_dim) {
        std::vector<__half> out(static_cast<size_t>(n_dim));
        for (int n = 0; n < n_dim; ++n) {
            const uint16_t *row = w + static_cast<int64_t>(n) * k_dim;
            double acc = 0.0;
            for (int k = 0; k < k_dim; ++k)
                acc += static_cast<double>(FromBf16(row[k])) *
                       static_cast<double>(__half2float(v[k]));
            out[static_cast<size_t>(n)] = __float2half_rn(static_cast<float>(acc));
        }
        return out;
    };
    auto layer_norm = [](const std::vector<__half> &raw, const uint16_t *w,
                         const uint16_t *b, int dim) {
        std::vector<__half> out(static_cast<size_t>(dim));
        double mean = 0.0;
        for (auto v : raw) mean += static_cast<double>(__half2float(v));
        mean /= static_cast<double>(dim);
        double var = 0.0;
        for (auto v : raw) {
            const double d = static_cast<double>(__half2float(v)) - mean;
            var += d * d;
        }
        var /= static_cast<double>(dim);
        const double inv = 1.0 / std::sqrt(var + 1e-6);
        for (int c = 0; c < dim; ++c) {
            const double v = static_cast<double>(__half2float(raw[static_cast<size_t>(c)]));
            out[static_cast<size_t>(c)] = __float2half_rn(static_cast<float>(
                (v - mean) * inv * static_cast<double>(FromBf16(w[c])) +
                static_cast<double>(FromBf16(b[c]))));
        }
        return out;
    };

    std::vector<__half> k_host(Tok * D), gate_host(Tok * D);
    for (int t = 0; t < Tok; ++t) {
        const __half *xt = x.data() + static_cast<int64_t>(t) * H;
        const std::vector<__half> raw = gemv(wk.data(), xt, D, H);
        const std::vector<__half> gate = gemv(gate_w.data(), xt, D, H);
        const std::vector<__half> norm = layer_norm(raw, knw.data(), knb.data(), D);
        std::copy(norm.begin(), norm.end(), k_host.begin() + static_cast<int64_t>(t) * D);
        std::copy(gate.begin(), gate.end(), gate_host.begin() + static_cast<int64_t>(t) * D);
    }
    std::vector<double> pool_host(static_cast<size_t>(Tok / Kp) * D);
    for (int p = 0; p < Tok / Kp; ++p) {
        const std::vector<double> one =
            IndexerPoolHost(k_host, gate_host, ape, D, Kp, static_cast<int64_t>(p) * Kp);
        std::copy(one.begin(), one.end(),
                  pool_host.begin() + static_cast<int64_t>(p) * D);
    }
    const std::vector<__half> qraw = gemv(wqb.data(), resid.data(), Heads * D, QL);
    std::vector<float> q_host(qraw.size());
    for (size_t i = 0; i < qraw.size(); ++i) q_host[i] = __half2float(qraw[i]);
    const std::vector<__half> wraw =
        gemv(wproj.data(), x.data() + static_cast<int64_t>(Tok - 1) * H, Heads, H);
    const float wscale = 1.0f / std::sqrt(static_cast<float>(Heads));
    std::vector<float> w_host(static_cast<size_t>(Heads));
    for (int h = 0; h < Heads; ++h)
        w_host[static_cast<size_t>(h)] = __half2float(wraw[static_cast<size_t>(h)]) * wscale;

    std::vector<float> pool_f(pool_host.begin(), pool_host.end());
    const std::vector<double> scores_host =
        IndexerScoresHost(q_host, w_host, pool_f, Heads, D, Tok / Kp);
    // The oracle is discrete: require a clear gap before trusting the order.
    REQUIRE(std::fabs(scores_host[0] - scores_host[1]) > 1e-3);
    const std::vector<int> order = TopKPoolsHost(scores_host, Tok / Kp);

    DeviceBuf d_x(x.size() * sizeof(__half)), d_resid(resid.size() * sizeof(__half));
    DeviceBuf d_wk(wk.size() * 2), d_gate_w(gate_w.size() * 2);
    DeviceBuf d_wqb(wqb.size() * 2), d_wproj(wproj.size() * 2);
    DeviceBuf d_knw(knw.size() * 2), d_knb(knb.size() * 2), d_ape(ape.size() * 2);
    Upload(d_x.p, x);
    Upload(d_resid.p, resid);
    Upload(d_wk.p, wk);
    Upload(d_gate_w.p, gate_w);
    Upload(d_wqb.p, wqb);
    Upload(d_wproj.p, wproj);
    Upload(d_knw.p, knw);
    Upload(d_knb.p, knb);
    Upload(d_ape.p, ape);

    FlashLayerWeights lw{};
    lw.d_idx_wk = d_wk.as<uint16_t>();
    lw.d_idx_gate = d_gate_w.as<uint16_t>();
    lw.d_idx_wq_b = d_wqb.as<uint16_t>();
    lw.d_idx_wproj = d_wproj.as<uint16_t>();
    lw.d_idx_knorm_w = d_knw.as<uint16_t>();
    lw.d_idx_knorm_b = d_knb.as<uint16_t>();
    lw.d_idx_ape = d_ape.as<uint16_t>();

    const cudaStream_t s = MakeStream();
    FlashIndexerState st;
    REQUIRE(FlashIndexerStateAlloc(st, /*max_ctx=*/64) == cudaSuccess);
    for (int t = 0; t < Tok; ++t)
        REQUIRE(RunFlashIndexerToken(s, st, lw, H,
                                     d_x.as<__half>() + static_cast<int64_t>(t) * H,
                                     t) == cudaSuccess);
    for (int p = 0; p < Tok / Kp; ++p)
        REQUIRE(RunFlashIndexerPool(s, st, lw, static_cast<int64_t>(p) * Kp)
                == cudaSuccess);
    REQUIRE(st.n_pools == Tok / Kp);
    REQUIRE(RunFlashIndexerSelect(s, st, lw, H,
                                  d_x.as<__half>() + static_cast<int64_t>(Tok - 1) * H,
                                  d_resid.as<__half>(), Tok - 1) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);

    std::vector<__half> k_dev(k_host.size()), gate_dev(gate_host.size());
    std::vector<float> pool_dev(static_cast<size_t>(Tok / Kp) * D);
    std::vector<float> q_dev(static_cast<size_t>(Heads * D)), w_dev(static_cast<size_t>(Heads));
    std::vector<int32_t> sel(Tok, 0), count(1, 0);
    Download(k_dev, st.d_k);
    Download(gate_dev, st.d_gate);
    Download(pool_dev, st.d_pool);
    Download(q_dev, st.d_q);
    Download(w_dev, st.d_w);
    Download(sel, st.d_sel);
    Download(count, st.d_sel_count);

    for (size_t i = 0; i < k_host.size(); ++i) {
        INFO("k i=" << i << " ref=" << __half2float(k_host[i])
                    << " got=" << __half2float(k_dev[i]));
        REQUIRE(std::fabs(static_cast<double>(__half2float(k_dev[i])) -
                          static_cast<double>(__half2float(k_host[i]))) < 4e-3);
    }
    for (size_t i = 0; i < gate_host.size(); ++i) {
        INFO("gate i=" << i << " ref=" << __half2float(gate_host[i])
                       << " got=" << __half2float(gate_dev[i]));
        REQUIRE(std::fabs(static_cast<double>(__half2float(gate_dev[i])) -
                          static_cast<double>(__half2float(gate_host[i]))) < 4e-3);
    }
    for (size_t i = 0; i < pool_dev.size(); ++i) {
        INFO("pool i=" << i << " ref=" << pool_host[i] << " got=" << pool_dev[i]);
        REQUIRE(std::fabs(static_cast<double>(pool_dev[i]) - pool_host[i]) < 4e-3);
    }
    for (size_t i = 0; i < q_dev.size(); ++i) {
        INFO("q i=" << i << " ref=" << q_host[i] << " got=" << q_dev[i]);
        REQUIRE(std::fabs(static_cast<double>(q_dev[i]) -
                          static_cast<double>(q_host[i])) < 1e-2);
    }
    for (int h = 0; h < Heads; ++h) {
        INFO("w h=" << h << " ref=" << w_host[static_cast<size_t>(h)]
                    << " got=" << w_dev[static_cast<size_t>(h)]);
        REQUIRE(std::fabs(static_cast<double>(w_dev[static_cast<size_t>(h)]) -
                          static_cast<double>(w_host[static_cast<size_t>(h)])) < 1e-3);
    }
    REQUIRE(count[0] == Tok);
    for (int r = 0; r < Tok / Kp; ++r)
        for (int i = 0; i < Kp; ++i) {
            const int expect = order[static_cast<size_t>(r)] * Kp + i;
            INFO("sel r=" << r << " i=" << i << " expect=" << expect
                          << " got=" << sel[static_cast<size_t>(r * Kp + i)]);
            REQUIRE(sel[static_cast<size_t>(r * Kp + i)] == expect);
        }
    FlashIndexerStateFree(st);
    cudaStreamDestroy(s);
}
