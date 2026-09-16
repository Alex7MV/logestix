#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
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

// Host-side NVFP4 quantization of one projection (round-2 r2 contract):
// per-16 pow2 block scales + global scale. Keeps the CPU reference on the
// ORIGINAL float weights, so the kernel path (packed bytes) is checked
// against the pre-quantization values within tolerance.
struct PackedProj {
    std::vector<uint8_t> packed, scales;
    float global_scale;
    int64_t rows, cols;
};

PackedProj QuantizeProj(const std::vector<float> &w, int64_t rows,
                        int64_t cols)
{
    PackedProj p;
    p.rows = rows;
    p.cols = cols;
    const int64_t numel = rows * cols;
    p.packed.assign(numel / 2, 0);
    p.scales.assign(numel / 16, 0);
    // global scale: 2^ceil(log2(maxabs/448)) pow2 (E4M3-range aligned).
    float maxabs = 0.f;
    for (float v : w) maxabs = std::max(maxabs, std::abs(v));
    int gexp = 0;
    if (maxabs > 0.f) {
        std::frexp(maxabs / 448.f, &gexp);
        if (gexp > 0) gexp -= 1;
    }
    p.global_scale = std::ldexp(1.0f, gexp);
    const float inv_gs = 1.0f / p.global_scale;
    auto enc_pow2 = [](int e) {
        return static_cast<uint8_t>(((e + 7) & 0xF) << 3);
    };
    for (int64_t blk = 0; blk < numel / 16; ++blk) {
        float bmax = 0.f;
        for (int j = 0; j < 16; ++j) {
            const float v = std::abs(w[blk * 16 + j]) * inv_gs;
            if (v > bmax) bmax = v;
        }
        int e6 = 0;
        if (bmax > 0.f) std::frexp(bmax, &e6);   // bmax in [2^(e6-1), 2^e6)
        const int se = e6 < -6 ? -6 : (e6 > 8 ? 8 : e6);   // scale = 2^se
        p.scales[blk] = enc_pow2(se);
        const float sc = std::ldexp(1.0f, se);
        for (int j = 0; j < 16; ++j) {
            const float v = w[blk * 16 + j] * inv_gs / sc;
            const bool neg = v < 0.f;
            const float a = std::abs(v);
            int mag;
            if (a < 0.25f) mag = 0;
            else if (a < 0.75f) mag = 1;
            else if (a < 1.25f) mag = 2;
            else if (a < 1.75f) mag = 3;
            else if (a < 2.5f) mag = 4;
            else if (a < 3.5f) mag = 5;
            else if (a < 5.0f) mag = 6;
            else mag = 7;
            const int nib = neg ? (mag | 8) : mag;
            const int64_t idx = blk * 16 + j;
            p.packed[idx >> 1] |=
                static_cast<uint8_t>(nib << ((idx & 1) ? 4 : 0));
        }
    }
    return p;
}

uint8_t *DevU8(const std::vector<uint8_t> &host)
{
    uint8_t *d = nullptr;
    if (cudaMalloc(&d, host.size()) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice);
    return d;
}

// Fills lw MLA projections from the packed fixtures.
void AttachPacked(KimiLayerWeights &lw, const PackedProj &qa,
                  const PackedProj &qb, const PackedProj &kva,
                  const PackedProj &kvb, const PackedProj &op,
                  float norm_qa_s, float norm_kva_s)
{
    lw.d_q_a_packed = DevU8(qa.packed);
    lw.d_q_a_scale = DevU8(qa.scales);
    lw.q_a_gs = qa.global_scale;
    lw.q_a_packed_bytes = static_cast<int64_t>(qa.packed.size());
    lw.q_a_scale_bytes = static_cast<int64_t>(qa.scales.size());
    lw.d_q_b_packed = DevU8(qb.packed);
    lw.d_q_b_scale = DevU8(qb.scales);
    lw.q_b_gs = qb.global_scale;
    lw.q_b_packed_bytes = static_cast<int64_t>(qb.packed.size());
    lw.q_b_scale_bytes = static_cast<int64_t>(qb.scales.size());
    lw.d_kv_a_packed = DevU8(kva.packed);
    lw.d_kv_a_scale = DevU8(kva.scales);
    lw.kv_a_gs = kva.global_scale;
    lw.kv_a_packed_bytes = static_cast<int64_t>(kva.packed.size());
    lw.kv_a_scale_bytes = static_cast<int64_t>(kva.scales.size());
    lw.d_kv_b_packed = DevU8(kvb.packed);
    lw.d_kv_b_scale = DevU8(kvb.scales);
    lw.kv_b_gs = kvb.global_scale;
    lw.kv_b_packed_bytes = static_cast<int64_t>(kvb.packed.size());
    lw.kv_b_scale_bytes = static_cast<int64_t>(kvb.scales.size());
    lw.d_o_packed = DevU8(op.packed);
    lw.d_o_scale = DevU8(op.scales);
    lw.o_gs = op.global_scale;
    lw.o_packed_bytes = static_cast<int64_t>(op.packed.size());
    lw.o_scale_bytes = static_cast<int64_t>(op.scales.size());
    (void)norm_qa_s;
    (void)norm_kva_s;
}

void FreePacked(KimiLayerWeights &lw)
{
    cudaFree(lw.d_q_a_packed); cudaFree(lw.d_q_a_scale);
    cudaFree(lw.d_q_b_packed); cudaFree(lw.d_q_b_scale);
    cudaFree(lw.d_kv_a_packed); cudaFree(lw.d_kv_a_scale);
    cudaFree(lw.d_kv_b_packed); cudaFree(lw.d_kv_b_scale);
    cudaFree(lw.d_o_packed); cudaFree(lw.d_o_scale);
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

    std::vector<float> qa(H * QA), qb(QA * QB), kva(H * KVA), kvb(KVB * kKimiKvLoraRank),
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
    // Round-2: projections are NVFP4-packed; the CPU reference stays on
    // the original float weights (quantization tolerance below).
    const PackedProj pqa = QuantizeProj(qa, QA, H);
    const PackedProj pqb = QuantizeProj(qb, QB, QA);
    const PackedProj pkva = QuantizeProj(kva, KVA, H);
    const PackedProj pkvb = QuantizeProj(kvb, KVB, kKimiKvLoraRank);
    const PackedProj pop = QuantizeProj(op, H, HV * NH);
    AttachPacked(lw, pqa, pqb, pkva, pkvb, pop, 0.9f, 0.9f);
    lw.d_q_a_norm = DevHalf(norm_qa);
    lw.d_kv_a_norm = DevHalf(norm_kva);
    REQUIRE(lw.d_q_a_packed);

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

    // Sequential single-token passes (decode-style): isolates rope
    // positions and incremental cache from batched-prefill effects.
    for (int t = 0; t < seq; ++t) {
        cudaError_t err =
            RunKimiMlaAttention(stream, /*turn=*/t, lw, 1,
                                d_x + (size_t)t * H,
                                /*residual=*/d_x + (size_t)t * H,
                                d_out + (size_t)t * H, d_state,
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
                                d_state, 0, 0);
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

    cudaStreamDestroy(stream);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_state);
    cudaFree(lw.d_q_a_norm);
    cudaFree(lw.d_kv_a_norm);
    FreePacked(lw);
}
