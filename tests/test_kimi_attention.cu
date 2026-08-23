#include <catch2/catch_test_macros.hpp>

#include "kimi_inference_kernels.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using namespace lsxcommon::kimi;

constexpr int H = kKimiHidden;
constexpr int QA = kKimiQLoraRank;
constexpr int QB = kKimiQbRows;
constexpr int KVA = kKimiKvaRows;
constexpr int KVB = kKimiKvbRows;
constexpr int NH = kKimiNumHeads;
constexpr int NOPE = kKimiQkNope;
constexpr int ROPE = kKimiQkRope;
constexpr int HV = kKimiHeadV;

float Frand(int seed, int i)
{
    // Deterministic pseudo-random in [-1, 1].
    uint32_t v = static_cast<uint32_t>(seed * 2654435761u + i * 40503u);
    v ^= v >> 13;
    v *= 1274126177u;
    v ^= v >> 16;
    return (static_cast<float>(v % 20001) - 10000.0f) / 10000.0f;
}

__half *DevHalf(const std::vector<float> &host)
{
    __half *d = nullptr;
    if (cudaMalloc(&d, host.size() * sizeof(__half)) != cudaSuccess)
        return nullptr;
    std::vector<__half> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        tmp[i] = __float2half(host[i]);
    cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    return d;
}

} // namespace

TEST_CASE("Kimi MLA decode matches naive CPU reference", "[kimi][attention]")
{
    const int seq = 1;

    // Synthetic layer weights.
    std::vector<float> qa(H * QA), qb(QA * QB), kva(H * KVA), kvb(KVB * 512),
        op(HV * NH * H);
    for (size_t i = 0; i < qa.size(); ++i)
        qa[i] = 0.02f * Frand(1, static_cast<int>(i));
    for (size_t i = 0; i < qb.size(); ++i)
        qb[i] = 0.02f * Frand(2, static_cast<int>(i));
    for (size_t i = 0; i < kva.size(); ++i)
        kva[i] = 0.02f * Frand(3, static_cast<int>(i));
    for (size_t i = 0; i < kvb.size(); ++i)
        kvb[i] = 0.02f * Frand(4, static_cast<int>(i));
    for (size_t i = 0; i < op.size(); ++i)
        op[i] = 0.02f * Frand(5, static_cast<int>(i));
    std::vector<float> norm_qa(QA, 0.9f), norm_kva(512, 0.9f);

    KimiLayerWeights lw{};
    lw.d_q_a_proj = DevHalf(qa);
    lw.d_q_a_norm = DevHalf(norm_qa);
    lw.d_q_b_proj = DevHalf(qb);
    lw.d_kv_a_proj = DevHalf(kva);
    lw.d_kv_a_norm = DevHalf(norm_kva);
    lw.d_kv_b_proj = DevHalf(kvb);
    lw.d_o_proj = DevHalf(op);
    REQUIRE(lw.d_q_a_proj);

    // Input hidden state.
    std::vector<float> xf(H);
    for (int i = 0; i < H; ++i) xf[i] = 0.5f * Frand(7, i);
    __half *d_x = DevHalf(xf);
    __half *d_out = nullptr;
    cudaMalloc(&d_out, H * sizeof(__half));

    KimiMlaState *d_state = nullptr;
    cudaMalloc(&d_state, sizeof(KimiMlaState));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE(KimiMlaReset(d_state, stream) == cudaSuccess);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    cudaError_t err =
        RunKimiMlaAttention(stream, /*turn=*/0, lw, seq, d_x, d_out,
                            d_state, cublas);
    REQUIRE(err == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> out(H);
    cudaMemcpy(out.data(), d_out, H * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // ── Naive CPU reference ──────────────────────────────────────────────
    std::vector<float> qa_f(QA);
    for (int i = 0; i < QA; ++i) {
        float acc = 0;
        for (int j = 0; j < H; ++j) acc += qa[i * H + j] * xf[j];
        qa_f[i] = acc;
    }
    float ss = 0;
    for (int i = 0; i < QA; ++i) ss += qa_f[i] * qa_f[i];
    float inv = 1.0f / std::sqrt(ss / QA + kKimiRmsEps);
    for (int i = 0; i < QA; ++i) qa_f[i] *= inv * norm_qa[i];

    std::vector<float> qall(QB);
    for (int r = 0; r < QB; ++r) {
        float acc = 0;
        for (int i = 0; i < QA; ++i) acc += qb[r * QA + i] * qa_f[i];
        qall[r] = acc;
    }
    std::vector<float> kvf(KVA);
    for (int s = 0; s < KVA; ++s) {
        float acc = 0;
        for (int j = 0; j < H; ++j) acc += kva[s * H + j] * xf[j];
        kvf[s] = acc;
    }
    ss = 0;
    for (int i = 0; i < 512; ++i) ss += kvf[i] * kvf[i];
    inv = 1.0f / std::sqrt(ss / 512 + kKimiRmsEps);
    for (int i = 0; i < 512; ++i) kvf[i] *= inv * norm_kva[i];

    // Yarn rope (single token at pos 0 => rotation angle 0: values unchanged
    // because cos(0)=1, sin(0)=0). So no-op here; still verify cache state.
    // Attention over exactly one slot.
    std::vector<float> attn(HV * NH);
    const float scale = (0.1f * std::log(64.0f) + 1.0f) /
                        std::sqrt(static_cast<float>(NOPE + ROPE));
    for (int h = 0; h < NH; ++h) {
        std::vector<float> qeff(512);
        for (int j = 0; j < 512; ++j) {
            float acc = 0;
            for (int i = 0; i < NOPE; ++i)
                acc += qall[h * (NOPE + ROPE) + i] *
                       kvb[(h * 256 + i) * 512 + j];
            qeff[j] = acc;
        }
        float score = 0;
        for (int j = 0; j < 512; ++j) score += qeff[j] * kvf[j];
        for (int j = 0; j < ROPE; ++j)
            score += qall[h * (NOPE + ROPE) + NOPE + j] * kvf[512 + j];
        score *= scale;
        float p = std::exp(score - std::max(0.0f, score)); // single-slot softmax
        p /= (std::exp(score - std::max(0.0f, score)) +
              std::exp(-std::max(0.0f, score)));
        (void)p;
        // Single slot: weight is exactly 1 after softmax normalization.
        for (int j2 = 0; j2 < HV; ++j2) {
            float acc = 0;
            for (int j = 0; j < 512; ++j)
                acc += kvb[(h * 256 + 128 + j2) * 512 + j] * kvf[j];
            attn[h * HV + j2] = acc;
        }
    }
    std::vector<float> ref(H);
    for (int o = 0; o < H; ++o) {
        float acc = 0;
        for (int r = 0; r < HV * NH; ++r)
            acc += op[o * (HV * NH) + r] * attn[r];
        ref[o] = xf[o] + acc;
    }

    double max_err = 0.0;
    for (int i = 0; i < H; ++i) {
        double e = std::abs(__half2float(out[i]) - ref[i]);
        max_err = std::max(max_err, e);
    }
    INFO("max abs error: " << max_err);
    CHECK(max_err <= 0.35);

    // Cache must hold exactly one slot.
    KimiMlaState st_host;
    cudaMemcpy(&st_host, d_state, sizeof(KimiMlaState),
               cudaMemcpyDeviceToHost);
    CHECK(st_host.n_used == 1);

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_state);
    cudaFree(lw.d_q_a_proj); cudaFree(lw.d_q_a_norm);
    cudaFree(lw.d_q_b_proj); cudaFree(lw.d_kv_a_proj);
    cudaFree(lw.d_kv_a_norm); cudaFree(lw.d_kv_b_proj);
    cudaFree(lw.d_o_proj);
}
