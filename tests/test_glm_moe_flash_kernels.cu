// GLM-5.3-Flash CUDA kernel unit tests (bit-parity against host refs).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "glm_moe_flash_device_math.h"
#include "lsxcommon/glm_moe_flash_inference.h"

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
