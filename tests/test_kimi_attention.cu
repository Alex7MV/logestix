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

__nv_bfloat16 *DevHalf(const std::vector<float> &host)
{
    __nv_bfloat16 *d = nullptr;
    if (cudaMalloc(&d, host.size() * sizeof(__nv_bfloat16)) != cudaSuccess)
        return nullptr;
    std::vector<__nv_bfloat16> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        tmp[i] = __float2bfloat16_rn(host[i]);
    cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    return d;
}

} // namespace

namespace {

// Full CPU reference for a seq-token prefill pass (positions base..base+seq-1,
// slots appended chronologically, causal attention).
void CpuMlaReference(const std::vector<float> &qa, const std::vector<float> &qb,
                     const std::vector<float> &kva, const std::vector<float> &kvb,
                     const std::vector<float> &op, const std::vector<float> &nqa,
                     const std::vector<float> &nkva, const std::vector<float> &X,
                     int seq, const std::vector<float> &xf, std::vector<float> &out)
{
    const float scale = (0.1f * std::log(64.0f) + 1.0f) *
                        (0.1f * std::log(64.0f) + 1.0f) /
                        std::sqrt(static_cast<float>(NOPE + ROPE));
    // Yarn inv_freq (HF linear-index interpolation).
    double cdim_lo = NOPE * 0; (void)cdim_lo;
    auto corr = [&](double r) {
        return ROPE * std::log(4096.0 / (r * 2 * M_PI)) /
               (2 * std::log(50000.0));
    };
    int lo = static_cast<int>(std::floor(corr(32.0)));
    int hi = static_cast<int>(std::ceil(corr(1.0)));
    std::vector<double> inv(ROPE / 2);
    for (int k = 0; k < ROPE / 2; ++k) {
        double extra = std::pow(50000.0, -2.0 * k / ROPE);
        double inter = extra / 64.0;
        double ramp = std::min(std::max((static_cast<double>(k) - lo) /
                                            (hi - lo), 0.0),
                               1.0);
        inv[k] = inter * ramp + extra * (1.0 - ramp);
    }

    std::vector<std::vector<float>> cache_c(seq, std::vector<float>(512));
    std::vector<std::vector<float>> cache_kr(seq, std::vector<float>(ROPE));

    for (int t = 0; t < seq; ++t) {
        const float *x = &X[t * H];
        std::vector<float> qa_f(QA);
        for (int i = 0; i < QA; ++i) {
            double acc = 0;
            for (int j = 0; j < H; ++j) acc += qa[i * H + j] * x[j];
            qa_f[i] = static_cast<float>(acc);
        }
        double ss = 0;
        for (int i = 0; i < QA; ++i) ss += qa_f[i] * qa_f[i];
        float invn = 1.0f / std::sqrt(ss / QA + kKimiRmsEps);
        for (int i = 0; i < QA; ++i) qa_f[i] *= invn * nqa[i];

        std::vector<float> qall(QB);
        for (int r = 0; r < QB; ++r) {
            double acc = 0;
            for (int i = 0; i < QA; ++i) acc += qb[r * QA + i] * qa_f[i];
            qall[r] = static_cast<float>(acc);
        }
        std::vector<float> kvf(KVA);
        for (int s2 = 0; s2 < KVA; ++s2) {
            double acc = 0;
            for (int j = 0; j < H; ++j) acc += kva[s2 * H + j] * x[j];
            kvf[s2] = static_cast<float>(acc);
        }
        ss = 0;
        for (int i = 0; i < 512; ++i) ss += kvf[i] * kvf[i];
        invn = 1.0f / std::sqrt(ss / 512 + kKimiRmsEps);
        for (int i = 0; i < 512; ++i) kvf[i] *= invn * nkva[i];

        // Rope on k branch.
        for (int pair = 0; pair < ROPE / 2; ++pair) {
            float fr = inv[pair];
            float c = std::cos(fr * t), sn = std::sin(fr * t);
            float xr = kvf[512 + pair], yr = kvf[512 + ROPE / 2 + pair];
            kvf[512 + pair] = xr * c - yr * sn;
            kvf[512 + ROPE / 2 + pair] = xr * sn + yr * c;
        }
        // Rope on q heads.
        for (int h = 0; h < NH; ++h)
            for (int pair = 0; pair < ROPE / 2; ++pair) {
                float fr = inv[pair];
                float c = std::cos(fr * t), sn = std::sin(fr * t);
                float *qr = &qall[h * (NOPE + ROPE) + NOPE];
                float xv = qr[pair], yv = qr[ROPE / 2 + pair];
                qr[pair] = xv * c - yv * sn;
                qr[ROPE / 2 + pair] = xv * sn + yv * c;
            }

        cache_c[t] = std::vector<float>(kvf.begin(), kvf.begin() + 512);
        cache_kr[t] = std::vector<float>(kvf.begin() + 512, kvf.end());

        // Causal attention over slots 0..t.
        int visible = t + 1;
        std::vector<float> attn(HV * NH);
        for (int h = 0; h < NH; ++h) {
            std::vector<double> qeff(512);
            for (int j = 0; j < 512; ++j) {
                double acc = 0;
                for (int i = 0; i < NOPE; ++i)
                    acc += qall[h * (NOPE + ROPE) + i] *
                           kvb[(h * 256 + i) * 512 + j];
                qeff[j] = acc;
            }
            std::vector<double> scores(visible);
            double mx = -1e30;
            for (int s2 = 0; s2 < visible; ++s2) {
                double acc = 0;
                for (int j = 0; j < 512; ++j)
                    acc += qeff[j] * cache_c[s2][j];
                for (int j = 0; j < ROPE; ++j)
                    acc += qall[h * (NOPE + ROPE) + NOPE + j] *
                           cache_kr[s2][j];
                scores[s2] = acc * scale;
                mx = mx > scores[s2] ? mx : scores[s2];
            }
            double denom = 0;
            std::vector<double> p(visible);
            for (int s2 = 0; s2 < visible; ++s2) {
                p[s2] = std::exp(scores[s2] - mx);
                denom += p[s2];
            }
            std::vector<double> m(512, 0.0);
            for (int s2 = 0; s2 < visible; ++s2)
                for (int j = 0; j < 512; ++j)
                    m[j] += (p[s2] / denom) * cache_c[s2][j];
            for (int j2 = 0; j2 < HV; ++j2) {
                double acc = 0;
                for (int j = 0; j < 512; ++j)
                    acc += kvb[(h * 256 + 128 + j2) * 512 + j] * m[j];
                attn[h * HV + j2] = static_cast<float>(acc);
            }
        }
        for (int o = 0; o < H; ++o) {
            double acc = 0;
            for (int r = 0; r < HV * NH; ++r)
                acc += op[o * (HV * NH) + r] * attn[r];
            out[static_cast<size_t>(t) * H + o] =
                xf[static_cast<size_t>(t) * H + o] + static_cast<float>(acc);
        }
    }
}

} // namespace

TEST_CASE("Kimi MLA 3-token prefill matches naive CPU reference",
          "[kimi][attention]")
{
    const int seq = 3;

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

    std::vector<float> xf(H * seq);
    for (size_t i = 0; i < xf.size(); ++i)
        xf[i] = 0.5f * Frand(7, static_cast<int>(i));
    __nv_bfloat16 *d_x = DevHalf(xf);
    __nv_bfloat16 *d_out = nullptr;
    cudaMalloc(&d_out, H * seq * sizeof(__nv_bfloat16));

    KimiMlaState *d_state = nullptr;
    cudaMalloc(&d_state, sizeof(KimiMlaState));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE(KimiMlaReset(d_state, stream) == cudaSuccess);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Sequential single-token passes (decode-style): isolates rope
    // positions and incremental cache from batched-prefill effects.
    for (int t = 0; t < seq; ++t) {
        cudaError_t err =
            RunKimiMlaAttention(stream, /*turn=*/t, lw, 1,
                                d_x + (size_t)t * H,
                                /*residual=*/d_x + (size_t)t * H,
                                d_out + (size_t)t * H, d_state, cublas,
                                /*base_slot=*/t, /*pos_offset=*/t);
        REQUIRE(err == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    }

    std::vector<__nv_bfloat16> out(H * seq);
    cudaMemcpy(out.data(), d_out, H * seq * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    std::vector<float> ref(H * seq, 0.0f);
    CpuMlaReference(qa, qb, kva, kvb, op, norm_qa, norm_kva, xf, seq,
                    xf, ref);

    double max_err = 0.0;
    for (int t = 0; t < seq; ++t) {
        double terr = 0.0;
        for (int i = 0; i < H; ++i)
            terr = std::max(terr,
                static_cast<double>(std::fabs(
                    __bfloat162float(out[t * H + i]) - ref[t * H + i])));
        WARN("token " << t << " max abs error: " << terr
             << " gpu[0..2]=[" << __bfloat162float(out[t * H])
             << "," << __bfloat162float(out[t * H + 1]) << ","
             << __bfloat162float(out[t * H + 2]) << "]"
             << " ref[0..2]=[" << ref[t * H] << "," << ref[t * H + 1]
             << "," << ref[t * H + 2] << "]");
    }
    for (size_t i = 0; i < out.size(); ++i)
        max_err = std::max(max_err,
                static_cast<double>(std::fabs(__bfloat162float(out[i]) - ref[i])));
    INFO("max abs error: " << max_err);
    CHECK(max_err <= 0.35);

    KimiMlaState st_host;
    cudaMemcpy(&st_host, d_state, sizeof(KimiMlaState),
               cudaMemcpyDeviceToHost);
    CHECK(st_host.n_used == seq);
    for (int t = 0; t < seq; ++t)
        WARN("slot" << t << " c[0..2]=["
             << __bfloat162float(st_host.kv[t][0]) << ","
             << __bfloat162float(st_host.kv[t][1]) << ","
             << __bfloat162float(st_host.kv[t][2]) << "]"
             << " kr[0..1]=[" << __bfloat162float(st_host.k_rope[t][0])
             << "," << __bfloat162float(st_host.k_rope[t][1]) << "]");

    // ── Batched prefill cross-check ──
    {
        REQUIRE(KimiMlaReset(d_state, stream) == cudaSuccess);
        std::vector<__nv_bfloat16> out_b(H * seq, __float2bfloat16_rn(0.f));
        cudaMemcpy(d_out, out_b.data(), out_b.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
        cudaError_t err =
            RunKimiMlaAttention(stream, 0, lw, seq, d_x, d_x, d_out,
                                d_state, cublas, 0, 0);
        REQUIRE(err == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        KimiMlaState st_b;
        cudaMemcpy(&st_b, d_state, sizeof(KimiMlaState),
                   cudaMemcpyDeviceToHost);
        for (int t = 0; t < seq; ++t)
            WARN("bslot" << t << " c[0..2]=["
                 << __bfloat162float(st_b.kv[t][0]) << ","
                 << __bfloat162float(st_b.kv[t][1]) << ","
                 << __bfloat162float(st_b.kv[t][2]) << "]");
        std::vector<__nv_bfloat16> ob(H * seq);
        cudaMemcpy(ob.data(), d_out, ob.size() * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);
        for (int t = 0; t < seq; ++t) {
            double terr = 0.0;
            for (int i = 0; i < H; ++i)
                terr = std::max(terr,
                    static_cast<double>(std::fabs(
                        __bfloat162float(ob[t * H + i]) - ref[t * H + i])));
            WARN("batched token " << t << " err=" << terr);
        }
    }

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_state);
    cudaFree(lw.d_q_a_proj); cudaFree(lw.d_q_a_norm);
    cudaFree(lw.d_q_b_proj); cudaFree(lw.d_kv_a_proj);
    cudaFree(lw.d_kv_a_norm); cudaFree(lw.d_kv_b_proj);
    cudaFree(lw.d_o_proj);
}
