// test_deepseek_moe_moe.cu — Unit tests for the DeepSeek-V4-Flash MoE orchestration.
// Covers the four production MoE paths that the kernel-level tests skip:
//   route_token        — gate GEMM → sqrt(softplus) scores → top-k (bias / hash)
//   run_shared_experts_ffn — F8_E4M3 shared-expert FFN (W1/W3 SwiGLU + W2)
//   run_routed_experts     — NVFP4 routed experts, weighted FP32 accumulation
//   layer mix              — route_token + shared FFN + routed experts in the
//                            exact order of deepseek_moe_inference_pipeline.cu's layer loop.
// Every case compares against an independent host-side FP64 reference.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <cstring>
#include "lsxcommon/lsx_inference_nvfp4.h"
#include <cstdint>
#include <algorithm>
#include <limits>
#include <lsxcommon/shared_experts.h>

#include "deepseek_moe_load.cuh"

using Catch::Approx;

namespace lsxcommon::gpu {

// Production entry points (deepseek_moe_inference_moe.cu), forward-declared like the
// existing kernel test does for run_routed_expert_ffn. route_token comes from
// deepseek_moe_load.cuh (included above).
cudaError_t run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float /*w1_gs*/, float /*w2_gs*/, float /*w3_gs*/,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    float *d_intermediate_f32, float *d_gate_f32,
    float *d_dequant, __half *d_output,
    const __half *d_w1_h_cache = nullptr,
    const __half *d_w3_h_cache = nullptr,
    const float *d_w2_f32_cache = nullptr);

cudaError_t run_routed_experts(
    cudaStream_t stream,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    float *d_output,
    __half *d_yg, __half *d_y2,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes);

} // namespace lsxcommon::gpu

namespace {

constexpr float kRouteScale = 1.5f;

// Deterministic pseudo-random source so fixtures are reproducible per run.
struct Lcg {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
};

// ── Host decoders (bit-identical to shared_experts.cu) ─────────────────────
// E4M3 semantics per cuda_fp8.hpp __nv_cvt_fp8_to_halfraw (__NV_E4M3):
// 0x7F -> canonical NaN, e == 0 && m != 0 -> subnormal m*2^-9.

float f8e4m3_decode(uint8_t b) {
    if (!b || b == 0x80) return 0.0f;
    if ((b & 0x7F) == 0x7F) {
        const uint32_t bits = 0x7FC00000u;  // canonical quiet NaN
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    if (e == 0) {
        float v = static_cast<float>(m) * 0.001953125f;  // m * 2^-9
        return s ? -v : v;
    }
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

float e8m0_scale_decode(uint8_t b) {
    if (b == 0) return 0.0f;
    if (b == 0xFF) return std::numeric_limits<float>::max();
    uint32_t f32 = static_cast<uint32_t>(b) << 23;
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

constexpr float kE2M1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

float e2m1_nibble_value(uint8_t nibble) {
    return kE2M1[nibble & 0x0F];
}

// ── Host references ────────────────────────────────────────────────────────

// Reference route_token math:
//   logit_e = Σ_k gate[e,k]*hidden[k]  computed in FP64, rounded to FP16
//   score_e = sqrt(softplus(fp16(logit_e)))
//   selection uses sel = score + bias (when bias present), weights use score.
void ref_route(
    const std::vector<__half> &gate, int hidden_dim, int num_routed,
    const std::vector<__half> &hidden,
    const std::vector<float> *bias,
    const std::vector<int64_t> *tid2eid, int tid2eid_stride, int token_id,
    int topk, std::vector<int> &idx, std::vector<float> &w)
{
    std::vector<float> scores(num_routed);
    for (int e = 0; e < num_routed; ++e) {
        long double acc = 0.0L;
        for (int k = 0; k < hidden_dim; ++k) {
            long double g = static_cast<long double>(__half2float(gate[static_cast<size_t>(e) * hidden_dim + k]));
            long double h = static_cast<long double>(__half2float(hidden[k]));
            acc += g * h;
        }
        __half logit_h = __float2half_rn(static_cast<float>(acc));
        float logit = __half2float(logit_h);
        scores[e] = std::sqrt(std::log1p(std::exp(logit)));
    }

    idx.assign(topk, -1);
    w.assign(topk, 0.0f);
    if (tid2eid) {
        int64_t base = static_cast<int64_t>(token_id) * tid2eid_stride;
        for (int k = 0; k < topk; ++k)
            idx[k] = static_cast<int>((*tid2eid)[base + k]);
    } else {
        std::vector<bool> used(num_routed, false);
        for (int k = 0; k < topk; ++k) {
            float best = -std::numeric_limits<float>::max();
            int bst = -1;
            for (int e = 0; e < num_routed; ++e) {
                if (used[e]) continue;
                float sel = scores[e] + (bias ? (*bias)[e] : 0.0f);
                if (sel > best) { best = sel; bst = e; }
            }
            if (bst >= 0) { used[bst] = true; idx[k] = bst; }
        }
    }
    double sum_w = 0.0;
    for (int k = 0; k < topk; ++k) {
        const double s = scores[idx[k]];
        w[k] = static_cast<float>(s);
        sum_w += s;
    }
    float inv = static_cast<float>(1.0 / (sum_w + 1e-9));
    for (int k = 0; k < topk; ++k)
        w[k] *= inv * kRouteScale;
}

// act_quant reference: replicate act_quant_block_kernel exactly (per-128
// block, power-of-two scale, E4M3 truncation to 3 mantissa bits).
void ref_act_quant(const std::vector<float> &x,
                   std::vector<float> &xq, std::vector<float> &scales)
{
    constexpr int BLK = 128;
    const int n = static_cast<int>(x.size());
    const int blocks = (n + BLK - 1) / BLK;
    xq.assign(n, 0.0f);
    scales.assign(static_cast<size_t>(blocks), 1.0f);
    for (int blk = 0; blk < blocks; ++blk) {
        int base = blk * BLK;
        int cnt = std::min(BLK, n - base);
        float amax = 0.0f;
        for (int i = 0; i < cnt; ++i)
            amax = std::max(amax, std::fabs(x[base + i]));
        float linear = std::max(amax, 1e-4f) / 448.0f;
        int log2_s = 0;
        float vs = linear;
        while (vs > 1.0f) { vs *= 0.5f; log2_s++; }
        while (vs <= 0.5f && log2_s > -149) { vs *= 2.0f; log2_s--; }
        float act_scale = std::exp2(static_cast<float>(log2_s));
        if (act_scale < linear) act_scale *= 2.0f;
        scales[blk] = act_scale;
        float inv = 1.0f / act_scale;
        for (int i = 0; i < cnt; ++i) {
            float v = x[base + i] * inv;
            v = std::max(-448.0f, std::min(448.0f, v));
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            uint32_t sign = (bits >> 31) & 1;
            int32_t exp32 = (bits >> 23) & 0xFF;
            uint32_t mant3 = (bits >> 20) & 0x07;
            int e4 = exp32 - 127 + 7;
            if (e4 < 1) e4 = 0;
            if (e4 > 15) { e4 = 15; mant3 = 7; }
            if (e4 == 0) {
                xq[base + i] = 0.0f;
            } else {
                uint32_t ob = (sign << 31) | (static_cast<uint32_t>(e4 + 120) << 23) | (mant3 << 20);
                float r;
                std::memcpy(&r, &ob, sizeof(r));
                xq[base + i] = r;
            }
        }
    }
}

// Shared-expert FFN reference (run_shared_experts_ffn):
//   dequant F8 (per-128x128 E8M0 tiles), FP32 GEMMs, silu(clamp(g,10))*clamp(u,±10).
// Returns values in FP32 (the production path final-rounds to FP16).
void ref_shared_ffn(
    const std::vector<uint8_t> &w1, const std::vector<uint8_t> &s1,
    const std::vector<uint8_t> &w2, const std::vector<uint8_t> &s2,
    const std::vector<uint8_t> &w3, const std::vector<uint8_t> &s3,
    const std::vector<__half> &hidden,
    int hidden_dim, int inter_dim,
    std::vector<__half> &out)
{
    std::vector<float> hf(hidden_dim);
    for (int k = 0; k < hidden_dim; ++k) hf[k] = __half2float(hidden[k]);

    auto deq = [&](const std::vector<uint8_t> &wb, const std::vector<uint8_t> &sb,
                   int rows, int cols, std::vector<float> &o) {
        o.assign(static_cast<size_t>(rows) * cols, 0.0f);
        const int SC = cols / 128;
        for (int i = 0; i < rows * cols; ++i) {
            int r = i / cols;
            int c = i % cols;
            float v = f8e4m3_decode(wb[i]) * e8m0_scale_decode(sb[(r / 128) * SC + (c / 128)]);
            o[i] = v;
        }
    };

    std::vector<float> w1f, w3f, w2f;
    deq(w1, s1, inter_dim, hidden_dim, w1f);
    deq(w3, s3, inter_dim, hidden_dim, w3f);
    deq(w2, s2, hidden_dim, inter_dim, w2f);

    std::vector<double> inter(inter_dim), gate(inter_dim), act(inter_dim);
    for (int r = 0; r < inter_dim; ++r) {
        long double a = 0.0L, b = 0.0L;
        for (int k = 0; k < hidden_dim; ++k) {
            a += static_cast<long double>(w1f[static_cast<size_t>(r) * hidden_dim + k]) * hf[k];
            b += static_cast<long double>(w3f[static_cast<size_t>(r) * hidden_dim + k]) * hf[k];
        }
        inter[r] = a;
        gate[r] = b;
    }
    for (int r = 0; r < inter_dim; ++r) {
        float g = static_cast<float>(inter[r]);
        float u = static_cast<float>(gate[r]);
        if (g > 10.0f) g = 10.0f;
        if (u > 10.0f) u = 10.0f;
        if (u < -10.0f) u = -10.0f;
        float sig = 1.0f / (1.0f + std::exp(-g));
        act[r] = static_cast<double>(sig) * g * u;
    }
    out.assign(static_cast<size_t>(hidden_dim), __float2half(0.0f));
    for (int o = 0; o < hidden_dim; ++o) {
        long double acc = 0.0L;
        for (int i = 0; i < inter_dim; ++i)
            acc += static_cast<long double>(w2f[static_cast<size_t>(o) * inter_dim + i]) * act[i];
        float v = static_cast<float>(acc);
        v = std::max(-65504.0f, std::min(65504.0f, v));
        out[o] = __float2half_rn(v);
    }
}

// Write a synthetic NVFP4/fp4 packed tensor (E2M1 nibbles, per-32 E8M0 scales)
// whose dequantized FP32 values are deterministic and spread across magnitudes.
void build_fp4_tensor(int64_t elems, Lcg &rng,
                      std::vector<uint8_t> &packed, std::vector<uint8_t> &scales)
{
    packed.assign(static_cast<size_t>((elems + 1) / 2), 0);
    scales.assign(static_cast<size_t>((elems + 31) / 32), 0);
    const uint8_t sc_base = 120; // scales span 2^-7 .. 2^6
    for (int64_t i = 0; i < elems; ++i) {
        uint8_t nib = static_cast<uint8_t>(rng.next() & 0x0F);
        int64_t byte = i >> 1;
        if (i & 1) packed[byte] |= static_cast<uint8_t>(nib << 4);
        else packed[byte] |= nib;
        if ((i & 31) == 0)
            scales[i >> 5] = sc_base + static_cast<uint8_t>(rng.next() % 14u);
    }
}

// Per-16 NVFP4 tensor (logestix-v1-nvfp4): random nibbles, pow2-E4M3
// scale bytes in [121..134] (window m=0, gs=1), low nibble first.
static void build_nvfp4_tensor16(int64_t elems, Lcg &rng,
                                 std::vector<uint8_t> &packed,
                                 std::vector<uint8_t> &scales)
{
    packed.assign(static_cast<size_t>((elems + 1) / 2), 0);
    scales.assign(static_cast<size_t>((elems + 15) / 16), 0);
    for (int64_t i = 0; i < elems; ++i) {
        uint8_t nib = static_cast<uint8_t>(rng.next() & 0x0F);
        int64_t byte = i >> 1;
        if (i & 1) packed[byte] |= static_cast<uint8_t>(nib << 4);
        else packed[byte] |= nib;
        if ((i & 15) == 0)
            // Repacked pow2-E4M3 payload bytes: e4 in [1..14] (byte = e4<<3),
            // matching the vault repack; keeps outputs inside the FP16
            // y contract end to end.
            scales[i >> 4] = static_cast<uint8_t>(
                (1 + static_cast<uint8_t>(rng.next() % 14u)) << 3);
    }
}

// pow2-E4M3 payload decode (e4 field, nvfp4_scale.h semantics).
static float e4m3p2_dec(uint8_t b)
{
    uint32_t e4 = (b >> 3) & 0x0Fu;
    if (!e4) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float v; __builtin_memcpy(&v, &bits, sizeof(v)); return v;
}

static float e2m1_dec(uint8_t nib)
{
    static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return kE2M1[nib & 0x0F];
}

// NVFP4-native routed-expert FFN reference: y = W1x,W3x GEMV (FP64
// accumulate) -> FP32 SwiGLU clamp10 -> *weight -> W2 accumulate.
static void ref_routed_expert_nvfp4(
    const std::vector<uint8_t> &w1, const std::vector<uint8_t> &s1,
    const std::vector<uint8_t> &w2, const std::vector<uint8_t> &s2,
    const std::vector<uint8_t> &w3, const std::vector<uint8_t> &s3,
    const std::vector<float> &x, int hidden_dim, int inter_dim,
    float weight, std::vector<float> &out)
{
    auto deq = [&](const std::vector<uint8_t> &pb,
                   const std::vector<uint8_t> &sb, int64_t elems) {
        std::vector<double> o(static_cast<size_t>(elems));
        for (int64_t i = 0; i < elems; ++i) {
            uint8_t byte = pb[i >> 1];
            uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            o[static_cast<size_t>(i)] =
                static_cast<double>(e2m1_dec(nib)) *
                static_cast<double>(e4m3p2_dec(sb[i >> 4]));
        }
        return o;
    };
    auto gemv = [&](const std::vector<double> &W, int rows, int cols,
                    const std::vector<float> &v) {
        std::vector<double> y(static_cast<size_t>(rows), 0.0);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                y[static_cast<size_t>(r)] +=
                    W[static_cast<size_t>(r) * cols + c] * v[static_cast<size_t>(c)];
        return y;
    };
    const auto W1 = deq(w1, s1, static_cast<int64_t>(inter_dim) * hidden_dim);
    const auto W3 = deq(w3, s3, static_cast<int64_t>(inter_dim) * hidden_dim);
    const auto W2 = deq(w2, s2, static_cast<int64_t>(hidden_dim) * inter_dim);
    const auto g = gemv(W1, inter_dim, hidden_dim, x);
    const auto u = gemv(W3, inter_dim, hidden_dim, x);
    std::vector<float> h(static_cast<size_t>(inter_dim));
    for (int i = 0; i < inter_dim; ++i) {
        float gv = static_cast<float>(g[static_cast<size_t>(i)]);
        float uv = static_cast<float>(u[static_cast<size_t>(i)]);
        gv = std::fminf(gv, 10.0f);
        uv = std::fminf(std::fmaxf(uv, -10.0f), 10.0f);
        h[static_cast<size_t>(i)] =
            (gv / (1.0f + std::expf(-gv))) * uv * weight;
    }
    const auto y2 = gemv(W2, hidden_dim, inter_dim, h);
    for (int i = 0; i < hidden_dim; ++i)
        out[static_cast<size_t>(i)] += static_cast<float>(y2[static_cast<size_t>(i)]);
}

// Routed-expert FFN reference for a single expert (run_routed_expert_ffn):
//   act_quant -> fuse act_scale into dequant W1/W3 -> GEMM -> SwiGLU(clamp10)
//   -> scale by routing weight -> W2 GEMM accumulate.
// Returns the expert's FP32 contribution that indexes into out[o] by +=.
void ref_routed_expert(
    const std::vector<uint8_t> &w1, const std::vector<uint8_t> &s1,
    const std::vector<uint8_t> &w2, const std::vector<uint8_t> &s2,
    const std::vector<uint8_t> &w3, const std::vector<uint8_t> &s3,
    const std::vector<float> &xq, const std::vector<float> &act_scales,
    int hidden_dim, int inter_dim, float weight,
    std::vector<float> &out)
{
    auto deq_fp4 = [&](const std::vector<uint8_t> &pb, const std::vector<uint8_t> &sb,
                       int64_t elems, std::vector<float> &o) {
        o.assign(static_cast<size_t>(elems), 0.0f);
        for (int64_t i = 0; i < elems; ++i) {
            int64_t byte = i >> 1;
            uint8_t nib = (i & 1) ? static_cast<uint8_t>(pb[byte] >> 4)
                                  : static_cast<uint8_t>(pb[byte] & 0x0F);
            o[i] = e2m1_nibble_value(nib) * e8m0_scale_decode(sb[i >> 5]);
        }
    };

    const int64_t w1el = static_cast<int64_t>(inter_dim) * hidden_dim;
    const int64_t w2el = static_cast<int64_t>(hidden_dim) * inter_dim;
    std::vector<float> w1f, w3f, w2f;
    deq_fp4(w1, s1, w1el, w1f);
    deq_fp4(w3, s3, w1el, w3f);
    deq_fp4(w2, s2, w2el, w2f);

    // fuse act_scale into W1/W3 (K = hidden)
    for (int64_t i = 0; i < w1el; ++i) {
        int k = static_cast<int>(i % hidden_dim);
        float sc = act_scales[k / 128];
        w1f[i] *= sc;
        w3f[i] *= sc;
    }

    std::vector<double> gate(inter_dim), up(inter_dim), swig(inter_dim);
    for (int r = 0; r < inter_dim; ++r) {
        long double a = 0.0L, b = 0.0L;
        for (int k = 0; k < hidden_dim; ++k) {
            a += static_cast<long double>(w1f[static_cast<size_t>(r) * hidden_dim + k]) * xq[k];
            b += static_cast<long double>(w3f[static_cast<size_t>(r) * hidden_dim + k]) * xq[k];
        }
        gate[r] = a;
        up[r] = b;
    }
    for (int r = 0; r < inter_dim; ++r) {
        float g = static_cast<float>(gate[r]);
        float u = static_cast<float>(up[r]);
        if (g > 10.0f) g = 10.0f;
        if (u > 10.0f) u = 10.0f;
        if (u < -10.0f) u = -10.0f;
        float sig = 1.0f / (1.0f + std::exp(-g));
        swig[r] = static_cast<double>(sig) * g * u * weight;
    }
    for (int o = 0; o < hidden_dim; ++o) {
        long double acc = 0.0L;
        for (int i = 0; i < inter_dim; ++i)
            acc += static_cast<long double>(w2f[static_cast<size_t>(o) * inter_dim + i]) * swig[i];
        out[o] += static_cast<float>(acc);
    }
}

bool cuda_sync_check() {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        WARN("CUDA error: " << cudaGetErrorString(e));
        return false;
    }
    return true;
}

void check_float16_close(const __half *gpu, const std::vector<__half> &ref,
                         size_t n, const char *what)
{
    size_t fails = 0, total = 0;
    for (size_t i = 0; i < n; ++i) {
        float g = __half2float(gpu[i]);
        float r = __half2float(ref[i]);
        ++total;
        if (!(std::fabs(g - r) <= 1e-3f * std::max(1.0f, std::fabs(r)))) {
            if (fails < 8)
                INFO(what << "[" << i << "] gpu=" << g << " ref=" << r);
            ++fails;
        }
    }
    INFO(what << ": " << fails << "/" << total << " mismatches");
    CHECK(fails == 0);
}

void check_float32_close(const float *gpu, const std::vector<float> &ref,
                         size_t n, const char *what, float tol = 1e-3f)
{
    size_t fails = 0, total = 0;
    float worst = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < n; ++i) {
        float g = gpu[i];
        float r = ref[i];
        ++total;
        float rel = std::fabs(g - r) / std::max(1.0f, std::fabs(r));
        if (rel > worst) { worst = rel; worst_i = i; }
        if (!(std::fabs(g - r) <= tol * std::max(1.0f, std::fabs(r)))) {
            if (fails < 8)
                INFO(what << "[" << i << "] gpu=" << g << " ref=" << r
                          << " rel=" << rel);
            ++fails;
        }
    }
    INFO(what << ": " << fails << "/" << total << " mismatches"
              << " (worst i=" << worst_i << " rel=" << worst
              << " gpu=" << gpu[worst_i] << " ref=" << ref[worst_i] << ")");
    CHECK(fails == 0);
}

} // namespace

// ── route_token: score path, bias path, hash path ───────────────────────────

TEST_CASE("route_token selects top-k by sqrt(softplus) with bias and scales weights by 1.5",
          "[deepseek][moe][router]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    constexpr int kHidden = 4096;
    constexpr int kNumRouted = 64;
    constexpr int kTopk = 6;

    // Deterministic fixture: hidden is constant (sum = 2048 exactly) and each
    // gate row is constant v_e = 0.00005*(e+1), so logit_e = 0.1024*(e+1) lies
    // in [0.10, 6.55] — small enough that FP16 rounding can never flip the
    // top-k order (adjacent scores are ~0.02 apart, FP16 noise ~0.001).
    std::vector<__half> gate(static_cast<size_t>(kHidden) * kNumRouted);
    for (int e = 0; e < kNumRouted; ++e) {
        __half v = __float2half_rn(0.00005f * static_cast<float>(e + 1));
        for (int k = 0; k < kHidden; ++k)
            gate[static_cast<size_t>(e) * kHidden + k] = v;
    }
    std::vector<__half> hidden(static_cast<size_t>(kHidden), __float2half_rn(0.5f));

    __half *d_gate = nullptr, *d_hidden = nullptr;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hidden, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    int topk[kTopk];
    float weights[kTopk];

    // Caller-owned route_token staging (arena-style, per new signature).
    __half *d_logits_scratch = nullptr;
    float *h_logits_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(float)) == cudaSuccess);

    // 1. Score path, no bias.
    std::vector<int> eidx;
    std::vector<float> eweight;
    {
        lsxcommon::gpu::route_token(ch, st, d_gate, d_hidden, kHidden, kNumRouted,
                                    nullptr, 0, 0, nullptr, kTopk,
                                    d_logits_scratch, h_logits_scratch,
                                    topk, weights);
        REQUIRE(cuda_sync_check());
        ref_route(gate, kHidden, kNumRouted, hidden, nullptr, nullptr, 0, 0,
                  kTopk, eidx, eweight);
        for (int k = 0; k < kTopk; ++k) {
            INFO("topk[" << k << "] gpu=" << topk[k] << " ref=" << eidx[k]);
            CHECK(topk[k] == eidx[k]);
            INFO("w[" << k << "] gpu=" << weights[k] << " ref=" << eweight[k]);
            CHECK(weights[k] == Approx(eweight[k]).epsilon(1e-3f));
        }
        // Distinct indices.
        for (int a = 0; a < kTopk; ++a)
            for (int b = a + 1; b < kTopk; ++b)
                CHECK(topk[a] != topk[b]);
        // Weights sum to ~route_scale after normalization.
        float sum = 0.0f;
        for (int k = 0; k < kTopk; ++k) sum += weights[k];
        CHECK(sum == Approx(kRouteScale).epsilon(1e-3f));
    }

    // 2. Bias path: heavily boost a low-score expert; selection must follow the
    //    biased score while the weights still use the unbiased sqrt(softplus).
    {
        std::vector<float> bias(static_cast<size_t>(kNumRouted), 0.0f);
        bias[3] = 100.0f; // expert 3 has an average rank, but bias forces it in
        lsxcommon::gpu::route_token(ch, st, d_gate, d_hidden, kHidden, kNumRouted,
                                    nullptr, 0, 0, bias.data(), kTopk,
                                    d_logits_scratch, h_logits_scratch,
                                    topk, weights);
        REQUIRE(cuda_sync_check());
        ref_route(gate, kHidden, kNumRouted, hidden, &bias, nullptr, 0, 0,
                  kTopk, eidx, eweight);
        bool has3 = false;
        for (int k = 0; k < kTopk; ++k) {
            INFO("bias topk[" << k << "] gpu=" << topk[k] << " ref=" << eidx[k]);
            CHECK(topk[k] == eidx[k]);
            INFO("bias w[" << k << "] gpu=" << weights[k] << " ref=" << eweight[k]);
            CHECK(weights[k] == Approx(eweight[k]).epsilon(1e-3f));
            if (topk[k] == 3) has3 = true;
        }
        CHECK(has3); // bias must have promoted expert 3 into the top-k
        float sum = 0.0f;
        for (int k = 0; k < kTopk; ++k) sum += weights[k];
        CHECK(sum == Approx(kRouteScale).epsilon(1e-3f));
    }

    // 3. Hash path: indices are predetermined by tid2eid (a hash layer).
    {
        constexpr int kTidStride = kTopk;
        std::vector<int64_t> tid2eid(static_cast<size_t>(2) * kTidStride);
        for (int k = 0; k < 2 * kTidStride; ++k)
            tid2eid[k] = (kNumRouted - 1 - k) % kNumRouted; // descending distinct
        lsxcommon::gpu::route_token(ch, st, d_gate, d_hidden, kHidden, kNumRouted,
                                    tid2eid.data(), kTidStride, 1,
                                    nullptr, kTopk,
                                    d_logits_scratch, h_logits_scratch,
                                    topk, weights);
        REQUIRE(cuda_sync_check());
        ref_route(gate, kHidden, kNumRouted, hidden, nullptr, &tid2eid,
                  kTidStride, 1, kTopk, eidx, eweight);
        for (int k = 0; k < kTopk; ++k) {
            INFO("hash topk[" << k << "] gpu=" << topk[k] << " ref=" << eidx[k]);
            CHECK(topk[k] == eidx[k]);
            INFO("hash w[" << k << "] gpu=" << weights[k] << " ref=" << eweight[k]);
            CHECK(weights[k] == Approx(eweight[k]).epsilon(1e-3f));
        }
        float sum = 0.0f;
        for (int k = 0; k < kTopk; ++k) sum += weights[k];
        CHECK(sum == Approx(kRouteScale).epsilon(1e-3f));
    }

    cudaFree(d_gate);
    cudaFree(d_hidden);
    cudaFree(d_logits_scratch);
    cudaFreeHost(h_logits_scratch);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}

// ── route_token split phases (async handshake) ──────────────────────────────
// begin/finish must produce results identical to the fused route_token for all
// three paths; the pipeline relies on this to overlap the shared-expert FFN
// with the router roundtrip without changing any numerics.

TEST_CASE("route_token split begin/finish matches fused route_token exactly",
          "[deepseek][moe][router]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    constexpr int kHidden = 4096;
    constexpr int kNumRouted = 64;
    constexpr int kTopk = 6;

    std::vector<__half> gate(static_cast<size_t>(kHidden) * kNumRouted);
    for (int e = 0; e < kNumRouted; ++e) {
        __half v = __float2half_rn(0.00005f * static_cast<float>(e + 1));
        for (int k = 0; k < kHidden; ++k)
            gate[static_cast<size_t>(e) * kHidden + k] = v;
    }
    std::vector<__half> hidden(static_cast<size_t>(kHidden), __float2half_rn(0.5f));

    __half *d_gate = nullptr, *d_hidden = nullptr;
    REQUIRE(cudaMalloc(&d_gate, gate.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hidden, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    __half *d_logits_scratch = nullptr;
    float *h_logits_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(float)) == cudaSuccess);

    cudaEvent_t ev;
    REQUIRE(cudaEventCreate(&ev) == cudaSuccess);

    std::vector<float> bias(static_cast<size_t>(kNumRouted), 0.0f);
    bias[3] = 100.0f;
    constexpr int kTidStride = kTopk;
    std::vector<int64_t> tid2eid(static_cast<size_t>(2) * kTidStride);
    for (int k = 0; k < 2 * kTidStride; ++k)
        tid2eid[k] = (kNumRouted - 1 - k) % kNumRouted;

    // {label, h_tid2eid, tid2eid_stride, token_id, h_gate_bias}
    struct Case {
        const char *label;
        const int64_t *tid;
        int stride;
        int token_id;
        const float *bias;
    };
    const Case cases[] = {
        {"score", nullptr, 0, 0, nullptr},
        {"bias", nullptr, 0, 0, bias.data()},
        {"hash", tid2eid.data(), kTidStride, 1, nullptr},
    };

    for (const Case &tc : cases) {
        int fused_idx[kTopk];
        float fused_w[kTopk];
        lsxcommon::gpu::route_token(ch, st, d_gate, d_hidden, kHidden, kNumRouted,
                                    tc.tid, tc.stride, tc.token_id, tc.bias, kTopk,
                                    d_logits_scratch, h_logits_scratch,
                                    fused_idx, fused_w);
        REQUIRE(cuda_sync_check());

        int split_idx[kTopk];
        float split_w[kTopk];
        REQUIRE(lsxcommon::gpu::route_token_begin(ch, st, d_gate, d_hidden,
                                                  kHidden, kNumRouted,
                                                  d_logits_scratch,
                                                  h_logits_scratch,
                                                  ev) == cudaSuccess);
        REQUIRE(lsxcommon::gpu::route_token_finish(kNumRouted, tc.token_id,
                                                   tc.tid, tc.stride, tc.bias,
                                                   kTopk, h_logits_scratch,
                                                   split_idx, split_w,
                                                   ev) == cudaSuccess);
        REQUIRE(cuda_sync_check());

        for (int k = 0; k < kTopk; ++k) {
            INFO(tc.label << " topk[" << k << "] fused=" << fused_idx[k]
                          << " split=" << split_idx[k]);
            CHECK(split_idx[k] == fused_idx[k]);
            INFO(tc.label << " w[" << k << "] fused=" << fused_w[k]
                          << " split=" << split_w[k]);
            CHECK(split_w[k] == fused_w[k]); // bit-exact: same host math
        }
    }

    cudaEventDestroy(ev);
    cudaFree(d_gate);
    cudaFree(d_hidden);
    cudaFree(d_logits_scratch);
    cudaFreeHost(h_logits_scratch);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}

// ── run_shared_experts_ffn: F8 shared-expert FFN ────────────────────────────

TEST_CASE("run_shared_experts_ffn computes the F8 SwiGLU FFN correctly",
          "[deepseek][moe][shared-ffn]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    constexpr int kHidden = 4096;
    constexpr int kInter = 2048;
    const int64_t w1el = static_cast<int64_t>(kHidden) * kInter;
    const int64_t w2el = static_cast<int64_t>(kInter) * kHidden;

    Lcg rng;
    std::vector<uint8_t> w1(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w3(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w2(static_cast<size_t>(w2el), 0);
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<uint8_t>(rng.next() & 0xFF);
    for (size_t i = 0; i < w3.size(); ++i) w3[i] = static_cast<uint8_t>(rng.next() & 0xFF);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = static_cast<uint8_t>(rng.next() & 0xFF);

    // Per-128x128 tile E8M0 scales. W1/W3: [inter, hidden] -> SC = hidden/128.
    constexpr int SC1 = kHidden / 128;
    std::vector<uint8_t> s1(static_cast<size_t>(kInter / 128) * SC1, 127);
    std::vector<uint8_t> s3(static_cast<size_t>(kInter / 128) * SC1, 127);
    for (size_t i = 0; i < s1.size(); ++i) s1[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);
    for (size_t i = 0; i < s3.size(); ++i) s3[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);
    // W2: [hidden, inter] -> SC = inter/128
    constexpr int SC2 = kInter / 128;
    std::vector<uint8_t> s2(static_cast<size_t>(kHidden / 128) * SC2, 127);
    for (size_t i = 0; i < s2.size(); ++i) s2[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);

    std::vector<__half> hidden(static_cast<size_t>(kHidden));
    for (int k = 0; k < kHidden; ++k)
        hidden[k] = __float2half_rn(static_cast<float>(rng.next() & 0xFFu) / 128.0f - 0.5f);

    __half *d_hidden = nullptr, *d_out = nullptr;
    uint8_t *d_w1 = nullptr, *d_w2 = nullptr, *d_w3 = nullptr;
    uint8_t *d_s1 = nullptr, *d_s2 = nullptr, *d_s3 = nullptr;
    float *d_inter = nullptr, *d_gate = nullptr, *d_dequant = nullptr;
    REQUIRE(cudaMalloc(&d_hidden, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, static_cast<size_t>(kHidden) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w1, w1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w2, w2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w3, w3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s1, s1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s2, s2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s3, s3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_inter, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    // d_dequant holds W1 (w1el) + hidden fp32 tail + W2 (w2el) + w2 out scratch.
    REQUIRE(cudaMalloc(&d_dequant, (static_cast<size_t>(w1el) + kHidden + static_cast<size_t>(w2el)) * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w1, w1.data(), w1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w2, w2.data(), w2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w3, w3.data(), w3.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s1, s1.data(), s1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s2, s2.data(), s2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s3, s3.data(), s3.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    lsxcommon::gpu::run_shared_experts_ffn(
        ch, st, d_hidden,
        d_w1, d_w2, d_w3,
        d_s1, d_s2, d_s3,
        1.0f, 1.0f, 1.0f,
        static_cast<int64_t>(w1.size()), static_cast<int64_t>(w2.size()), static_cast<int64_t>(w3.size()),
        d_inter, d_gate, d_dequant, d_out);
    REQUIRE(cuda_sync_check());

    std::vector<__half> ref(static_cast<size_t>(kHidden));
    ref_shared_ffn(w1, s1, w2, s2, w3, s3, hidden, kHidden, kInter, ref);

    std::vector<__half> gpu(static_cast<size_t>(kHidden));
    REQUIRE(cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(__half), cudaMemcpyDeviceToHost) == cudaSuccess);
    check_float16_close(gpu.data(), ref, ref.size(), "shared-ffn");

    cudaFree(d_hidden);
    cudaFree(d_out);
    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
    cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_s3);
    cudaFree(d_inter);
    cudaFree(d_gate);
    cudaFree(d_dequant);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}

// ── run_routed_experts: NVFP4 routed experts, weighted accumulation ─────────

TEST_CASE("run_routed_experts accumulates weighted NVFP4 expert FFNs",
          "[deepseek][moe][routed-experts]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    // Production always runs under the resolved perf policy (TC + seq1
    // GEMV on Blackwell); mirror it so the dispatcher takes the same path.
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(true);
      constexpr int kHidden = 4096;
    constexpr int kInter = 2048;
    constexpr int kExpertsPerLayer = 2;
    constexpr int kTopk = 2;
    const int64_t w1el = static_cast<int64_t>(kHidden) * kInter;
    const int64_t w2el = static_cast<int64_t>(kInter) * kHidden;

    Lcg rng;
    std::vector<lsxcommon::gpu::ExpertEntry> db(kExpertsPerLayer);
    std::vector<std::vector<uint8_t>> pw(kExpertsPerLayer * 3);
    std::vector<std::vector<uint8_t>> ps(kExpertsPerLayer * 3);
    for (int e = 0; e < kExpertsPerLayer; ++e) {
        build_nvfp4_tensor16(w1el, rng, pw[3 * e + 0], ps[3 * e + 0]);
        build_nvfp4_tensor16(w2el, rng, pw[3 * e + 1], ps[3 * e + 1]);
        build_nvfp4_tensor16(w1el, rng, pw[3 * e + 2], ps[3 * e + 2]);
        db[e].host_w1 = pw[3 * e + 0].data();
        db[e].host_w2 = pw[3 * e + 1].data();
        db[e].host_w3 = pw[3 * e + 2].data();
        db[e].host_w1_scales = ps[3 * e + 0].data();
        db[e].host_w2_scales = ps[3 * e + 1].data();
        db[e].host_w3_scales = ps[3 * e + 2].data();
        db[e].w1_bytes = static_cast<int64_t>(pw[3 * e + 0].size());
        db[e].w2_bytes = static_cast<int64_t>(pw[3 * e + 1].size());
        db[e].w3_bytes = static_cast<int64_t>(pw[3 * e + 2].size());
        db[e].w1_scales_bytes = static_cast<int64_t>(ps[3 * e + 0].size());
        db[e].w2_scales_bytes = static_cast<int64_t>(ps[3 * e + 1].size());
        db[e].w3_scales_bytes = static_cast<int64_t>(ps[3 * e + 2].size());
    }

    // Routing picks both experts with distinct weights that do not sum to
    // exactly 1.0 (weights are norm*depth-normalized*scale products).
    const int topk_idx[kTopk] = {0, 1};
    const float topk_wt[kTopk] = {0.7f, 0.9f};

    std::vector<__half> hidden(static_cast<size_t>(kHidden));
    for (int k = 0; k < kHidden; ++k)
        hidden[k] = __float2half_rn(static_cast<float>(rng.next() & 0xFFu) / 128.0f - 0.5f);

    // Host act_quant reference.
    std::vector<float> hf(kHidden);
    for (int k = 0; k < kHidden; ++k) hf[k] = __half2float(hidden[k]);
    std::vector<float> xq, asc;
    ref_act_quant(hf, xq, asc);

    // Expected output.
    std::vector<float> ref(static_cast<size_t>(kHidden), 0.0f);
    for (int k = 0; k < kTopk; ++k) {
        int e = topk_idx[k];
        ref_routed_expert_nvfp4(pw[3 * e + 0], ps[3 * e + 0],
                                pw[3 * e + 1], ps[3 * e + 1],
                                pw[3 * e + 2], ps[3 * e + 2],
                                hf, kHidden, kInter, topk_wt[k], ref);
    }

    __half *d_normed = nullptr;
    float *d_out = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_dequant = nullptr;
    float *d_act_quant = nullptr, *d_swiglu = nullptr, *d_act_scales = nullptr;
    uint8_t *d_pack = nullptr, *d_scale = nullptr;
    int64_t pack_bytes = 0, scale_bytes = 0;
    for (int e = 0; e < kExpertsPerLayer; ++e) {
        pack_bytes += db[e].w1_bytes + db[e].w2_bytes + db[e].w3_bytes;
        scale_bytes += db[e].w1_scales_bytes + db[e].w2_scales_bytes + db[e].w3_scales_bytes;
    }
    REQUIRE(cudaMalloc(&d_normed, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, static_cast<size_t>(kHidden) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_up, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_dequant, static_cast<size_t>(w1el) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_act_quant, static_cast<size_t>(kHidden) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_swiglu, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_act_scales, static_cast<size_t>(kHidden / 128) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_pack, static_cast<size_t>(pack_bytes)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scale, static_cast<size_t>(scale_bytes)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_normed, hidden.data(), hidden.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_out, 0, static_cast<size_t>(kHidden) * sizeof(float)) == cudaSuccess);

    lsxcommon::gpu::run_routed_experts(
        st, d_normed,
        db.data(), 0, kExpertsPerLayer,
        topk_idx, topk_wt, kTopk,
        d_out,
        reinterpret_cast<__half *>(d_gate),
        reinterpret_cast<__half *>(d_up),
        d_pack, d_scale, pack_bytes, scale_bytes);
    REQUIRE(cuda_sync_check());

    std::vector<float> gpu(static_cast<size_t>(kHidden), 0.0f);
    REQUIRE(cudaMemcpy(gpu.data(), d_out, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    // GPU is pure FP32 cuBLAS SGEMM; the reference accumulates in 80-bit
    // long double, so FP32 rounding over K=2048 with wide scales (2^-7..2^7)
    // legitimately reaches ~2e-3 relative error on large outputs.
    // Tolerance-class contract (k-order-free GEMV + FP16 SwiGLU): error is
    // bounded relative to the OUTPUT SCALE of the row, not per-element —
    // cancelled outputs legitimately show large per-element relative error.
    {
        float m = 0;
        for (float v : ref) m = std::max(m, std::fabs(v));
        std::vector<float> g2(gpu.size());
        for (size_t i = 0; i < gpu.size(); ++i)
            g2[i] = gpu[i] - ref[i];
        float dmax = 0;
        for (float v : g2) dmax = std::max(dmax, std::fabs(v));
        INFO("routed-experts: max|diff|=" << dmax << " max|ref|=" << m);
        CHECK(dmax <= 2e-2f * m);
    }

    cudaFree(d_normed);
    cudaFree(d_out);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_dequant);
    cudaFree(d_act_quant);
    cudaFree(d_swiglu);
    cudaFree(d_act_scales);
    cudaFree(d_pack);
    cudaFree(d_scale);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}

// ── Layer mix: route_token + shared FFN + routed experts (production order) ─

TEST_CASE("MoE layer mix preserves the shared-expert accumulation (beta=1)",
          "[deepseek][moe][layer-mix]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    // Production always runs under the resolved perf policy (TC + seq1
    // GEMV on Blackwell); mirror it so the dispatcher takes the same path.
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(true);

    constexpr int kHidden = 4096;
    constexpr int kInter = 2048;
    constexpr int kNumRouted = 64;
    constexpr int kExpertsPerLayer = 8;
    constexpr int kTopk = 3;
    const int64_t w1el = static_cast<int64_t>(kHidden) * kInter;
    const int64_t w2el = static_cast<int64_t>(kInter) * kHidden;

    Lcg rng;

    // Shared-experts FFN weights (F8_E4M3, positive so SwiGLU stays well-scaled).
    std::vector<uint8_t> w1(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w3(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w2(static_cast<size_t>(w2el), 0);
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<uint8_t>(rng.next() & 0x7F); // positive F8
    for (size_t i = 0; i < w3.size(); ++i) w3[i] = static_cast<uint8_t>(rng.next() & 0x7F);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = static_cast<uint8_t>(rng.next() & 0x7F);
    constexpr int SC1 = kHidden / 128;
    std::vector<uint8_t> s1(static_cast<size_t>(kInter / 128) * SC1, 124);
    std::vector<uint8_t> s3(static_cast<size_t>(kInter / 128) * SC1, 124);
    constexpr int SC2 = kInter / 128;
    std::vector<uint8_t> s2(static_cast<size_t>(kHidden / 128) * SC2, 124);

    // Routed-expert db.
    std::vector<lsxcommon::gpu::ExpertEntry> db(kExpertsPerLayer);
    std::vector<std::vector<uint8_t>> pw(kExpertsPerLayer * 3);
    std::vector<std::vector<uint8_t>> ps(kExpertsPerLayer * 3);
    for (int e = 1; e <= kTopk; ++e) { // only fill the routed ones, leave rest null
        build_nvfp4_tensor16(w1el, rng, pw[3 * e + 0], ps[3 * e + 0]);
        build_nvfp4_tensor16(w2el, rng, pw[3 * e + 1], ps[3 * e + 1]);
        build_nvfp4_tensor16(w1el, rng, pw[3 * e + 2], ps[3 * e + 2]);
        db[e].host_w1 = pw[3 * e + 0].data();
        db[e].host_w2 = pw[3 * e + 1].data();
        db[e].host_w3 = pw[3 * e + 2].data();
        db[e].host_w1_scales = ps[3 * e + 0].data();
        db[e].host_w2_scales = ps[3 * e + 1].data();
        db[e].host_w3_scales = ps[3 * e + 2].data();
        db[e].w1_bytes = static_cast<int64_t>(pw[3 * e + 0].size());
        db[e].w2_bytes = static_cast<int64_t>(pw[3 * e + 1].size());
        db[e].w3_bytes = static_cast<int64_t>(pw[3 * e + 2].size());
        db[e].w1_scales_bytes = static_cast<int64_t>(ps[3 * e + 0].size());
        db[e].w2_scales_bytes = static_cast<int64_t>(ps[3 * e + 1].size());
        db[e].w3_scales_bytes = static_cast<int64_t>(ps[3 * e + 2].size());
    }

    // Router: deterministic gate that reliably selects experts 1,2,3. Hidden is
    // kept strictly positive so the constant row value v_e → logit_e = v_e·Σh,
    // and v=0.01 for experts {1,2,3} beats v∈{0.003,0.001} for everyone else.
    // logit stays ~20 so sqrt(softplus) never overflows.
    std::vector<__half> gate_w(static_cast<size_t>(kNumRouted) * kHidden);
    for (int e = 0; e < kNumRouted; ++e) {
        __half v = __float2half_rn((e >= 1 && e <= 3) ? 0.01f : (e % 5 == 0 ? 0.003f : 0.001f));
        for (int k = 0; k < kHidden; ++k)
            gate_w[static_cast<size_t>(e) * kHidden + k] = v;
    }

    std::vector<__half> hidden(static_cast<size_t>(kHidden));
    for (int k = 0; k < kHidden; ++k)
        hidden[k] = __float2half_rn(0.5f + static_cast<float>(rng.next() & 0xFFu) / 256.0f);

    // Reference: shared FFN (fp16 out) then fan-out via half->float accumulator,
    // then routed experts accumulated into the SAME accumulator.
    std::vector<__half> shared_ref(static_cast<size_t>(kHidden));
    ref_shared_ffn(w1, s1, w2, s2, w3, s3, hidden, kHidden, kInter, shared_ref);
    std::vector<float> expected(static_cast<size_t>(kHidden));
    for (int o = 0; o < kHidden; ++o)
        expected[o] = __half2float(shared_ref[o]);

    // GPU device buffers.
    __half *d_gate_in = nullptr, *d_hidden_in = nullptr, *d_ffn_out = nullptr;
    float *d_ffn_accum = nullptr, *d_inter = nullptr, *d_gate = nullptr, *d_dequant = nullptr;
    float *d_up = nullptr, *d_act_quant = nullptr, *d_swiglu = nullptr, *d_act_scales = nullptr;
    uint8_t *d_sw1 = nullptr, *d_sw2 = nullptr, *d_sw3 = nullptr;
    uint8_t *d_ss1 = nullptr, *d_ss2 = nullptr, *d_ss3 = nullptr;
    uint8_t *d_pack = nullptr, *d_scale = nullptr;
    int pack_bytes = 0, scale_bytes = 0;
    for (int e = 1; e <= kTopk; ++e) {
        pack_bytes += static_cast<int>(db[e].w1_bytes + db[e].w2_bytes + db[e].w3_bytes);
        scale_bytes += static_cast<int>(db[e].w1_scales_bytes + db[e].w2_scales_bytes + db[e].w3_scales_bytes);
    }

    REQUIRE(cudaMalloc(&d_gate_in, gate_w.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hidden_in, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ffn_out, static_cast<size_t>(kHidden) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ffn_accum, static_cast<size_t>(kHidden) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_inter, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_dequant, (static_cast<size_t>(w1el) + kHidden + static_cast<size_t>(w2el)) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_up, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_act_quant, static_cast<size_t>(kHidden) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_swiglu, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_act_scales, static_cast<size_t>(kHidden / 128) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sw1, w1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sw2, w2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sw3, w3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ss1, s1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ss2, s2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ss3, s3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_pack, static_cast<size_t>(pack_bytes)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scale, static_cast<size_t>(scale_bytes)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_gate_in, gate_w.data(), gate_w.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hidden_in, hidden.data(), hidden.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_sw1, w1.data(), w1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_sw2, w2.data(), w2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_sw3, w3.data(), w3.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_ss1, s1.data(), s1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_ss2, s2.data(), s2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_ss3, s3.data(), s3.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    // 1. Router (score path, no bias).
    int topk[kTopk];
    float weights[kTopk];
    __half *d_logits_scratch = nullptr;
    float *h_logits_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_logits_scratch, static_cast<size_t>(kNumRouted) * sizeof(float)) == cudaSuccess);
    lsxcommon::gpu::route_token(ch, st, d_gate_in, d_hidden_in, kHidden, kNumRouted,
                                nullptr, 0, 0, nullptr, kTopk,
                                d_logits_scratch, h_logits_scratch,
                                topk, weights);
    REQUIRE(cuda_sync_check());
    // The gate design guarantees {1,2,3}; the routed db only filled those.
    for (int k = 0; k < kTopk; ++k) {
        INFO("layer-mix topk[" << k << "]=" << topk[k]);
        CHECK(topk[k] >= 1);
        CHECK(topk[k] <= 3);
        CHECK(weights[k] > 0.0f);
    }

    // Routed reference using the ACTUAL router weights/indices.
    std::vector<float> hf(kHidden);
    for (int k = 0; k < kHidden; ++k) hf[k] = __half2float(hidden[k]);
    for (int k = 0; k < kTopk; ++k) {
        int e = topk[k];
        ref_routed_expert_nvfp4(pw[3 * e + 0], ps[3 * e + 0],
                                pw[3 * e + 1], ps[3 * e + 1],
                                pw[3 * e + 2], ps[3 * e + 2],
                                hf, kHidden, kInter, weights[k], expected);
    }

    // 2. Shared experts FFN -> d_ffn_out (FP16), then half->float accumulator.
    lsxcommon::gpu::run_shared_experts_ffn(
        ch, st, d_hidden_in,
        d_sw1, d_sw2, d_sw3,
        d_ss1, d_ss2, d_ss3,
        1.0f, 1.0f, 1.0f,
        static_cast<int64_t>(w1.size()), static_cast<int64_t>(w2.size()), static_cast<int64_t>(w3.size()),
        d_inter, d_gate, d_dequant, d_ffn_out);
    REQUIRE(cuda_sync_check());
    {
        // half -> float conversion of the shared FFN output into d_ffn_accum
        // (mirrors the production float_to_half_f2h_kernel accumulation step).
        std::vector<__half> h_ffn_out(kHidden);
        REQUIRE(cudaMemcpy(h_ffn_out.data(), d_ffn_out, h_ffn_out.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        std::vector<float> h_accum(kHidden);
        for (int o = 0; o < kHidden; ++o) h_accum[o] = __half2float(h_ffn_out[o]);
        REQUIRE(cudaMemcpy(d_ffn_accum, h_accum.data(), h_accum.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }

    // 3. Routed experts accumulate on top of the shared-expert result.
    lsxcommon::gpu::run_routed_experts(
        st, d_hidden_in,
        db.data(), 0, kExpertsPerLayer,
        topk, weights, kTopk,
        d_ffn_accum,
        reinterpret_cast<__half *>(d_gate),
        reinterpret_cast<__half *>(d_up),
        d_pack, d_scale, pack_bytes, scale_bytes);
    REQUIRE(cuda_sync_check());

    std::vector<float> gpu(kHidden);
    REQUIRE(cudaMemcpy(gpu.data(), d_ffn_accum, gpu.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    // Tolerance-class row-scale criterion (see the routed-experts test).
    {
        float m = 0;
        for (float v : expected) m = std::max(m, std::fabs(v));
        float dmax = 0;
        for (size_t i = 0; i < gpu.size(); ++i)
            dmax = std::max(dmax, std::fabs(gpu[i] - expected[i]));
        INFO("layer-mix: max|diff|=" << dmax << " max|ref|=" << m);
        CHECK(dmax <= 2e-2f * m);
    }

    cudaFree(d_gate_in);
    cudaFree(d_hidden_in);
    cudaFree(d_ffn_out);
    cudaFree(d_ffn_accum);
    cudaFree(d_inter);
    cudaFree(d_gate);
    cudaFree(d_dequant);
    cudaFree(d_up);
    cudaFree(d_act_quant);
    cudaFree(d_swiglu);
    cudaFree(d_act_scales);
    cudaFree(d_sw1); cudaFree(d_sw2); cudaFree(d_sw3);
    cudaFree(d_ss1); cudaFree(d_ss2); cudaFree(d_ss3);
    cudaFree(d_pack);
    cudaFree(d_scale);
    cudaFree(d_logits_scratch);
    cudaFreeHost(h_logits_scratch);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}
// ── shared-expert dequant caching: bit-exact vs per-call dequant ────────────
TEST_CASE("run_shared_experts_ffn dequant cache is bit-exact",
          "[deepseek][moe][shared-cache]")
{
    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    constexpr int kHidden = 4096;
    constexpr int kInter = 2048;
    const int64_t w1el = static_cast<int64_t>(kHidden) * kInter;
    const int64_t w2el = static_cast<int64_t>(kInter) * kHidden;

    Lcg rng;
    std::vector<uint8_t> w1(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w3(static_cast<size_t>(w1el), 0);
    std::vector<uint8_t> w2(static_cast<size_t>(w2el), 0);
    for (size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<uint8_t>(rng.next() & 0xFF);
    for (size_t i = 0; i < w3.size(); ++i) w3[i] = static_cast<uint8_t>(rng.next() & 0xFF);
    for (size_t i = 0; i < w2.size(); ++i) w2[i] = static_cast<uint8_t>(rng.next() & 0xFF);

    constexpr int SC1 = kHidden / 128;
    std::vector<uint8_t> s1(static_cast<size_t>(kInter / 128) * SC1, 127);
    std::vector<uint8_t> s3(static_cast<size_t>(kInter / 128) * SC1, 127);
    for (size_t i = 0; i < s1.size(); ++i) s1[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);
    for (size_t i = 0; i < s3.size(); ++i) s3[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);
    constexpr int SC2 = kInter / 128;
    std::vector<uint8_t> s2(static_cast<size_t>(kHidden / 128) * SC2, 127);
    for (size_t i = 0; i < s2.size(); ++i) s2[i] = 121 + static_cast<uint8_t>(rng.next() % 12u);

    std::vector<__half> hidden(static_cast<size_t>(kHidden));
    for (int k = 0; k < kHidden; ++k)
        hidden[k] = __float2half_rn(static_cast<float>(rng.next() & 0xFFu) / 128.0f - 0.5f);

    __half *d_hidden = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
    uint8_t *d_w1 = nullptr, *d_w2 = nullptr, *d_w3 = nullptr;
    uint8_t *d_s1 = nullptr, *d_s2 = nullptr, *d_s3 = nullptr;
    float *d_inter = nullptr, *d_gate = nullptr, *d_dequant = nullptr;
    __half *d_c1 = nullptr, *d_c3 = nullptr;   // FP16 W1/W3 caches
    float *d_c2 = nullptr;                     // FP32 W2 cache
    REQUIRE(cudaMalloc(&d_hidden, hidden.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out_a, static_cast<size_t>(kHidden) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out_b, static_cast<size_t>(kHidden) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w1, w1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w2, w2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w3, w3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s1, s1.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s2, s2.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_s3, s3.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_inter, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gate, static_cast<size_t>(kInter) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_dequant, (static_cast<size_t>(w1el) + kHidden + static_cast<size_t>(w2el)) * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_c1, static_cast<size_t>(w1el) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_c3, static_cast<size_t>(w1el) * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_c2, static_cast<size_t>(w2el) * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w1, w1.data(), w1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w2, w2.data(), w2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w3, w3.data(), w3.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s1, s1.data(), s1.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s2, s2.data(), s2.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_s3, s3.data(), s3.size(), cudaMemcpyHostToDevice) == cudaSuccess);

    // Legacy (per-call dequant).
    REQUIRE(lsxcommon::gpu::run_shared_experts_ffn(
                ch, st, d_hidden, d_w1, d_w2, d_w3, d_s1, d_s2, d_s3,
                1.0f, 1.0f, 1.0f,
                static_cast<int64_t>(w1.size()), static_cast<int64_t>(w2.size()),
                static_cast<int64_t>(w3.size()),
                d_inter, d_gate, d_dequant, d_out_a,
                nullptr, nullptr, nullptr) == cudaSuccess);

    // Load-time dequant into the caches, then the cached call.
    REQUIRE(lsxcommon::gpu::dequant_shared_f8_half(
                st, d_w1, static_cast<int64_t>(w1.size()), d_s1,
                static_cast<int64_t>(s1.size()), d_c1, w1el, kHidden) == cudaSuccess);
    REQUIRE(lsxcommon::gpu::dequant_shared_f8_half(
                st, d_w3, static_cast<int64_t>(w3.size()), d_s3,
                static_cast<int64_t>(s3.size()), d_c3, w1el, kHidden) == cudaSuccess);
    REQUIRE(lsxcommon::gpu::dequant_shared_f8(
                st, d_w2, static_cast<int64_t>(w2.size()), d_s2,
                static_cast<int64_t>(s2.size()), d_c2, w2el, kInter) == cudaSuccess);
    REQUIRE(lsxcommon::gpu::run_shared_experts_ffn(
                ch, st, d_hidden, d_w1, d_w2, d_w3, d_s1, d_s2, d_s3,
                1.0f, 1.0f, 1.0f,
                static_cast<int64_t>(w1.size()), static_cast<int64_t>(w2.size()), static_cast<int64_t>(w3.size()),
                d_inter, d_gate, d_dequant, d_out_b,
                d_c1, d_c3, d_c2) == cudaSuccess);
    REQUIRE(cuda_sync_check());

    std::vector<__half> a(static_cast<size_t>(kHidden)), b(static_cast<size_t>(kHidden));
    REQUIRE(cudaMemcpy(a.data(), d_out_a, a.size() * sizeof(__half), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(b.data(), d_out_b, b.size() * sizeof(__half), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0);

    cudaFree(d_hidden); cudaFree(d_out_a); cudaFree(d_out_b);
    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
    cudaFree(d_s1); cudaFree(d_s2); cudaFree(d_s3);
    cudaFree(d_inter); cudaFree(d_gate); cudaFree(d_dequant);
    cudaFree(d_c1); cudaFree(d_c3); cudaFree(d_c2);
    cublasDestroy(ch);
    cudaStreamDestroy(st);
}
