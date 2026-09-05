// GLM-5.3-Flash host references for CUDA kernel parity tests.
// Numeric contracts mirror the device twins exactly (plain IEEE FP32 ops,
// no FMA reliance, exp/sigmoid implemented as the device does).
// Host-only: include from test .cu host code, never from device functions.
#ifndef LSXCOMMON_GLM_FLASH_DEVICE_MATH_H
#define LSXCOMMON_GLM_FLASH_DEVICE_MATH_H

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "lsxcommon/qwen_moe_inference.h"  // QwenMoeRoutedSlot

namespace lsxcommon {
namespace glm_moe_flash {

// glibc-replica expf (sysdeps/ieee754/flt-32/e_expf.c) usable from host
// and device: guarantees the kernel's sigmoid matches the host reference
// bit-for-bit. Same algorithm as glm::device::GlmExpfGlibc but with
// portable bit reinterpretation (memcpy-based) instead of __device__
// intrinsics.
#ifdef __CUDACC__
#define FLASH_HD __host__ __device__ __forceinline__
#else
#define FLASH_HD inline
#endif

FLASH_HD uint32_t flash_as_uint(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

FLASH_HD float flash_as_float(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

FLASH_HD double flash_as_double(uint64_t u)
{
    double d;
    memcpy(&d, &u, 8);
    return d;
}

FLASH_HD uint64_t flash_as_uint64(double d)
{
    uint64_t u;
    memcpy(&u, &d, 8);
    return u;
}

constexpr int kFlashExpfTableBits = 5;
constexpr int kFlashExpfN = 1 << kFlashExpfTableBits;
#ifdef __CUDACC__
static __device__ constexpr uint64_t kFlashExpfTab[kFlashExpfN] = {
        0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
        0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
        0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
        0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
        0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
        0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
        0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
        0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
        0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
        0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
        0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
    };
#else
constexpr uint64_t kFlashExpfTab[kFlashExpfN] = {
        0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full,
        0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull,
        0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, 0x3fef06fe0a31b715ull,
        0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
        0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull,
        0x3feea47eb03a5585ull, 0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull,
        0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull,
        0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
        0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull,
        0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full,
        0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
    };
#endif

FLASH_HD float flash_expf_glibc(float x)
{
    constexpr int kExp2fTableBits = 5;
    constexpr int kN = 1 << kExp2fTableBits;
    constexpr double kShift = 0x1.8p+52;
    constexpr double kInvLn2N = 0x1.71547652b82fep+5;
    constexpr double kC0 = 0x1.c6af84b912394p-20;
    constexpr double kC1 = 0x1.ebfce50fac4f3p-13;
    constexpr double kC2 = 0x1.62e42ff0c52d6p-6;

    const uint32_t ux = flash_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7ffu) return x + x;
        if (x > 0x1.62e42ep6f) return flash_as_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }

    const double xd = static_cast<double>(x);
    const double z = kInvLn2N * xd;
    // Value-based round-to-nearest-even via the 1.5*2^52 trick (do NOT
    // reinterpret the sum's bits — that yields garbage on device).
    const double kd =
        static_cast<double>(static_cast<uint64_t>(z + kShift)) - kShift;
    const uint64_t ki = static_cast<uint64_t>(static_cast<int64_t>(kd));
    const double r = z - kd;
    uint64_t t = kFlashExpfTab[ki % kFlashExpfN];
    t += ki << (52 - kExp2fTableBits);
    const double s = flash_as_double(t);
    const double zz = kC0 * r + kC1;
    const double r2 = r * r;
    double y = kC2 * r + 1.0;
    y = zz * r2 + y;
    y = y * s;
    return static_cast<float>(y);
}

inline float sigmoid_host(float x)
{
    return 1.0f / (1.0f + flash_expf_glibc(-x));
}

inline float softplus_host(float x)
{
    // FLA convention: log1p(exp(x)) stable form.
    if (x > 20.0f) return x;
    return std::log1p(flash_expf_glibc(x));
}

// BF16 bits -> float (widening: top 16 bits of the F32 pattern).
inline float bf16_to_float_host(const uint8_t *p)
{
    const uint32_t bits =
        (static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8))
        << 16;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// ── Gated delta rule (KDA) with per-channel decay ────────────────────────
// HF/FLA semantics (kimi delta rule, mirrored by the device kernel):
//   for each head h, value/value-dim d column of state S[h, l, d]:
//     col[l] *= gdecay[tt, h*D + l]              (per-channel decay, key dim)
//     kv = sum_l S[l,d] * k[tt, h, l]            (k l2-normalized upstream)
//     delta = beta[tt, h] * (v[tt, h, d] - kv)
//     col[l] += k[tt, h, l] * delta
//     out[tt, h, d] = sum_l S[l,d] * q[tt, h, l] (q l2-normalized + 1/sqrt(D))
// q/k/v/out: [seq, H, D] FP32; beta: [seq, H]; gdecay: [seq, H*D].
// state: [H, D, D] written with the FINAL row's state (incremental decode
// contract matches qwen_gated_delta_kernel col_seed semantics).
inline void FlashGatedDeltaHost(const float *q, const float *k,
                                const float *v, const float *beta,
                                const float *gdecay, int seq, int heads,
                                int dim, float *state, float *out)
{
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
            float col[256];
            for (int l = 0; l < dim; ++l) col[l] = 0.0f;
            for (int t = 0; t < seq; ++t) {
                for (int l = 0; l < dim; ++l)
                    col[l] *= gdecay[(t * heads + h) * dim + l];
                float kv = 0.0f;
                for (int l = 0; l < dim; ++l)
                    kv += col[l] * k[(t * heads + h) * dim + l];
                const float delta =
                    beta[t * heads + h] * (v[(t * heads + h) * dim + d] - kv);
                for (int l = 0; l < dim; ++l)
                    col[l] += k[(t * heads + h) * dim + l] * delta;
                float acc = 0.0f;
                for (int l = 0; l < dim; ++l)
                    acc += col[l] * q[(t * heads + h) * dim + l];
                out[(t * heads + h) * dim + d] = acc;
            }
            for (int l = 0; l < dim; ++l)
                state[(h * dim + l) * dim + d] = col[l];
        }
    }
}

// ── Router: sigmoid + bias (noaux_tc, n_group=1), top-8, renorm, scale ────
// gate: BF16 bits [E, H] (row-major per expert); bias: F32 [E];
// x: BF16 bits [tok, H].
// HF Glm5NextTextTopkRouter semantics: choice score_e = sigmoid(dot) + bias_e;
// top-8 by choice score (n_group == 1 -> no group mask); coeff from the RAW
// sigmoid score; renorm coeff / sum(top8) * routed_scaling (2.5).
inline std::vector<qwen_moe::QwenMoeRoutedSlot> FlashRouterTop8Host(
    const uint8_t *gate_bf16, const float *bias, const uint8_t *x_bf16,
    int tok, int hidden, int experts)
{
    constexpr int kTopK = 8;
    constexpr float kScaling = 2.5f;
    std::vector<qwen_moe::QwenMoeRoutedSlot> out(tok * kTopK);
    for (int t = 0; t < tok; ++t) {
        std::vector<float> score(experts);
        std::vector<float> choice(experts);
        for (int e = 0; e < experts; ++e) {
            float dot = 0.0f;
            for (int h = 0; h < hidden; ++h)
                dot += bf16_to_float_host(gate_bf16 + e * hidden + h * 2)
                     * bf16_to_float_host(x_bf16 + t * hidden + h * 2);
            score[e] = sigmoid_host(dot);
            choice[e] = score[e] + bias[e];
        }
        // Top-8 selection on choice score, tie-break: lower expert index
        // wins on equal score (matches the device linear-scan selection).
        for (int kk = 0; kk < kTopK; ++kk) {
            int best = -1;
            float best_s = -1e30f;
            for (int e = 0; e < experts; ++e) {
                bool taken = false;
                for (int j = 0; j < kk; ++j)
                    if (out[t * kTopK + j].eid == e) taken = true;
                if (taken) continue;
                if (choice[e] > best_s) {
                    best_s = choice[e];
                    best = e;
                }
            }
            out[t * kTopK + kk].eid = best;
            out[t * kTopK + kk].coeff = score[best];
        }
        float sum = 0.0f;
        for (int kk = 0; kk < kTopK; ++kk) sum += out[t * kTopK + kk].coeff;
        for (int kk = 0; kk < kTopK; ++kk)
            out[t * kTopK + kk].coeff =
                out[t * kTopK + kk].coeff / sum * kScaling;
    }
    return out;
}

// ── Depthwise causal conv1d (decode window form) ─────────────────────────
// x: [K, C] window rows oldest..newest; w: [C, K]; y: [C].
// y[c] = sum_k w[c*K + k] * x[k*C + c].
inline void FlashConv1dHost(const float *x, const float *w, int channels,
                            int kernel, float *y)
{
    for (int c = 0; c < channels; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k)
            acc += w[c * kernel + k] * x[k * channels + c];
        y[c] = acc;
    }
}

} // namespace glm_moe_flash
} // namespace lsxcommon

#endif // LSXCOMMON_GLM_FLASH_DEVICE_MATH_H
