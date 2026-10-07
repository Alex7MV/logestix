// TDD Cycle 10 — SKWOS integration regression harness over the frozen 79-layer
// execution loop inside the unified GlmMoEArena (IndexShare attention K-V
// cache + Layer-78 MTP epilogue in one contiguous device block).
//
// The harness drives a multi-turn synthetic prompt block through the complete
// loop (dense 0-2 -> IndexShare attention + MoE 3-77 -> Layer-78 MTP
// epilogue) on REAL weights and asserts:
//   (a) token decoding runs without memory leaks (VRAM free does not shrink
//       across repeated full runs) and without index-boundary truncation
//       (every produced token id lies in [0, vocab) and the decoder KV slot
//       count matches prompt + generated exactly — no MTP KV leak),
//   (b) the final speculative output logits row (layer-78 MTP head) converges
//       element-for-element against the independent CPU FP64 reference
//       (RED/GREEN) and the greedy token stream shows 0 tokens-drift.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/glm_moe_inference.h>
#include "lsxcommon/schema.h"
#include <lsxcommon/glm_moe_load.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/schema.h>

using lsxcommon::glm::GlmPinnedWeights;
using lsxcommon::glm::GlmExpertTables;
using lsxcommon::glm::GlmGenerationResult;
using lsxcommon::glm::GlmExpertDeviceBlock;

namespace {

constexpr int kHidden   = lsxcommon::glm::kGlmHidden;           // 6144
constexpr int kInter    = lsxcommon::glm::kGlmMoeIntermediate;  // 2048
constexpr int kTopK     = lsxcommon::glm::kGlmMoETopK;          // 8
constexpr int kNRouted  = lsxcommon::glm::kGlmNumRouted;        // 256
constexpr int kHeadDim  = lsxcommon::glm::kGlmIndexHeadDim;     // 128
constexpr int kHeads    = lsxcommon::glm::kGlmIndexNHeads;      // 32
constexpr int kQLora    = lsxcommon::glm::kGlmQLoraRank;        // 2048
constexpr int kMaxSlots = lsxcommon::glm::kGlmIndexMaxSlots;    // 1024
constexpr int kVocab    = 154880;
constexpr int64_t kMatEl = static_cast<int64_t>(kInter) * kHidden;

constexpr float kAttnScale = 0.088388347648318440f;             // 1/sqrt(128)
constexpr int kEosIds[3] = {154820, 154827, 154829};

// ── VRAM-budgeted real-weight subset (the frozen "model baseline" contract) ──
// Dense 0-2, early MoE window (3-10) with live attention + routed experts, a
// late MoE tail (76-77) and the full MTP layer 78. Layers not in the pinned set
// degrade their branch to identity (attention still runs over the shared
// IndexShare staging); experts not loaded are zero-contribution slots.
const std::vector<int> kPinnedLayers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        76, 77, 78};
const std::vector<int> kExpertLayers = {3, 4, 6, 8, 78};
constexpr int kExpertsPerLayer = 16;
const std::vector<int32_t> kPrompt = {154820, 258, 1636, 7};
constexpr int kMaxNew = 6;

// ── FP16 exact helpers (mirror __float2half_rn / __half2float) ────────────
inline uint16_t FloatToHalfRne(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const int      exp  = static_cast<int>((u >> 23) & 0xFFu);
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
    const int exp2 = exp - 127;
    if (exp2 >= 15) return static_cast<uint16_t>(sign | 0x7C00u);
    const uint32_t F = 0x800000u | mant;
    if (exp2 >= -14) {
        int e = exp2 + 15;
        uint32_t m10 = (F >> 13) & 0x3FFu;
        const uint32_t rem = F & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u))) {
            ++m10;
            if (m10 == 0x400u) { m10 = 0; ++e; }
        }
        if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign |
            (static_cast<uint16_t>(e) << 10) | static_cast<uint16_t>(m10));
    }
    const int shift = -exp2 - 1;
    if (shift > 24) return sign;
    uint32_t s = F >> shift;
    const uint32_t rem = F & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (s & 1u))) ++s;
    if (s == 0x400u) return static_cast<uint16_t>(sign | 0x0400u);
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(s));
}

inline __half H(float v)
{
    const uint16_t bits = FloatToHalfRne(v);
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return h;
}
inline float F(__half h) { return __half2float(h); }

inline float DequantValue(const std::vector<uint8_t> &packed,
                          const std::vector<uint8_t> &scales,
                          float gs, int64_t i)
{
    constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                2.0f, 3.0f, 4.0f, 6.0f};
    const uint8_t sb = scales[static_cast<size_t>(i / lsxcommon::glm::kGlmNvfp4Block)];
    // Canonical block-scale decode (complete E4M3FN; public wrapper of
    // nvfp4_scale.h) — never a local pow2/field-zero copy.
    const float    sc = lsxcommon::e4m3_to_float(sb);
    const uint8_t  b   = packed[static_cast<size_t>(i >> 1)];
    const uint8_t  nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                 : static_cast<uint8_t>(b & 0x0Fu);
    const float    e2  = kE2M1[nib & 7u];
    return F(H(((nib & 8u) ? -e2 : e2) * sc * gs));
}

// ── Host mirror of the device FP16 weights the loop consumes ─────────────
struct RefLayer {
    bool has_moe = false;
    bool has_dense = false;
    bool has_indexer = false;
    std::vector<__half> post_attn_norm;
    std::vector<__half> gate;
    std::vector<float>  gate_bias;
    std::vector<__half> mlp_gate, mlp_up, mlp_down;
    std::vector<__half> wq_b, wk, wproj, k_norm;
    std::vector<float>  k_bias;
    std::vector<__half> eh_proj, enorm, shared_head_norm;
    std::vector<std::vector<float>> exp_w1, exp_w2, exp_w3;
    std::vector<float> exp_gs1, exp_gs2, exp_gs3;
    std::vector<bool>  exp_active;
};

struct RefRun {
    std::vector<__half> embed;
    std::vector<__half> head;
    std::vector<RefLayer> layers;   // [79]
    bool ExpertActive(int layer, int eid) const
    {
        const auto &rl = layers[static_cast<size_t>(layer)];
        if (eid < 0 || eid >= static_cast<int>(rl.exp_active.size())) return false;
        return rl.exp_active[static_cast<size_t>(eid)];
    }
};

// ── Cycle-5 exact mirrors (IndexShare attention reference) ────────────────
struct RefIdx {
    std::vector<__half> kv;
    std::vector<__half> q_key, key;
    std::vector<int> slot;
    int n = 0;
    int turn_id = -1;
    RefIdx() : q_key(static_cast<size_t>(kMaxSlots) * kHeadDim),
               key(static_cast<size_t>(kMaxSlots) * kHeadDim),
               slot(static_cast<size_t>(kMaxSlots), 0) {}
};

void RefRefresh(RefIdx &R, const RefLayer &W)
{
    for (int s = 0; s < R.n; ++s) {
        const __half *v = R.kv.data() + static_cast<size_t>(s) * kHidden;
        std::vector<float> sc(static_cast<size_t>(kHeads), 0.0f);
        for (int h = 0; h < kHeads; ++h) {
            float acc = 0.0f;
            const __half *wp = W.wproj.data() + static_cast<size_t>(h) * kHidden;
            for (int k = 0; k < kHidden; ++k) acc = fmaf(F(v[k]), F(wp[k]), acc);
            sc[static_cast<size_t>(h)] = acc;
        }
        int best = 0; float bv = sc[0];
        for (int h = 1; h < kHeads; ++h)
            if (sc[static_cast<size_t>(h)] > bv) { bv = sc[static_cast<size_t>(h)]; best = h; }
        R.slot[static_cast<size_t>(s)] = best;
        std::vector<float> kraw(static_cast<size_t>(kHeadDim), 0.0f);
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wr = W.wk.data() + static_cast<size_t>(r) * kHidden;
            for (int k = 0; k < kHidden; ++k) acc = fmaf(F(v[k]), F(wr[k]), acc);
            kraw[static_cast<size_t>(r)] = acc;
        }
        float sq = 0.0f;
        for (int r = 0; r < kHeadDim; ++r)
            sq = fmaf(kraw[static_cast<size_t>(r)], kraw[static_cast<size_t>(r)], sq);
        const float rms = std::sqrt(sq / static_cast<float>(kHeadDim) +
                                    lsxcommon::glm::kGlmRmsEps);
        __half *kd = R.key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            const float kb = W.k_bias.empty() ? 0.0f : W.k_bias[static_cast<size_t>(r)];
            kd[r] = H((kraw[static_cast<size_t>(r)] / rms) *
                      F(W.k_norm[static_cast<size_t>(r)]) + kb);
        }
        __half *qd = R.q_key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wqr = W.wq_b.data() +
                (static_cast<size_t>(best) * kHeadDim + r) * kQLora;
            for (int k = 0; k < kQLora; ++k) acc = fmaf(F(v[k]), F(wqr[k]), acc);
            qd[r] = H(acc);
        }
    }
}

std::vector<__half> RefAttention(RefIdx &R, const std::vector<__half> &x, int seq)
{
    std::vector<__half> out(static_cast<size_t>(seq) * kHidden);
    if (R.n <= 0) return x;
    const int base = R.n - seq;
    for (int t = 0; t < seq; ++t) {
        const __half *qq = R.q_key.data() + static_cast<size_t>(base + t) * kHeadDim;
        std::vector<float> sims(static_cast<size_t>(R.n), 0.0f);
        for (int s = 0; s < R.n; ++s) {
            const __half *ks = R.key.data() + static_cast<size_t>(s) * kHeadDim;
            float acc = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) acc = fmaf(F(qq[d]), F(ks[d]), acc);
            sims[static_cast<size_t>(s)] = acc * kAttnScale;
        }
        float m = -1e30f;
        for (int s = 0; s < R.n; ++s) m = std::max(m, sims[static_cast<size_t>(s)]);
        std::vector<float> p(static_cast<size_t>(R.n), 0.0f);
        float sum = 0.0f;
        for (int s = 0; s < R.n; ++s) { p[static_cast<size_t>(s)] = std::exp(sims[static_cast<size_t>(s)] - m); sum += p[static_cast<size_t>(s)]; }
        const float inv = 1.0f / sum;
        for (int s = 0; s < R.n; ++s) p[static_cast<size_t>(s)] *= inv;
        for (int h = 0; h < kHidden; ++h) {
            float acc = F(x[static_cast<size_t>(t) * kHidden + h]);
            for (int s = 0; s < R.n; ++s)
                acc = fmaf(p[static_cast<size_t>(s)],
                           F(R.kv[static_cast<size_t>(s) * kHidden + h]), acc);
            out[static_cast<size_t>(t) * kHidden + h] = H(acc);
        }
    }
    return out;
}

// ── Router reference (fp32 mirror of the device top-k total order) ────────
struct CpuRouted { int32_t eid; float coeff; };
std::vector<CpuRouted> CpuRouter(const std::vector<__half> &normed,
                                 const std::vector<__half> &gate,
                                 const std::vector<float> &bias)
{
    const bool use_bias = bias.size() == static_cast<size_t>(kNRouted);
    std::vector<float> logit(static_cast<size_t>(kNRouted), 0.0f);
    for (int e = 0; e < kNRouted; ++e) {
        double acc = 0.0;
        const __half *gr = gate.data() + static_cast<size_t>(e) * kHidden;
        for (int k = 0; k < kHidden; ++k)
            acc += static_cast<double>(F(gr[k])) * static_cast<double>(F(normed[k]));
        logit[static_cast<size_t>(e)] = static_cast<float>(acc);
    }
    std::vector<float> sel(static_cast<size_t>(kNRouted), 0.0f);
    for (int e = 0; e < kNRouted; ++e)
        sel[static_cast<size_t>(e)] = use_bias ? logit[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)]
                                               : logit[static_cast<size_t>(e)];
    float lmax = -1e30f;
    for (float v : logit) lmax = std::max(lmax, v);
    float psum = 0.0f;
    for (float v : logit) psum += std::exp(v - lmax);
    const float inv = 1.0f / psum;
    std::vector<float> prob(static_cast<size_t>(kNRouted), 0.0f);
    for (int e = 0; e < kNRouted; ++e)
        prob[static_cast<size_t>(e)] = std::exp(logit[static_cast<size_t>(e)] - lmax) * inv;
    std::vector<CpuRouted> out(static_cast<size_t>(kTopK));
    std::vector<bool> used(static_cast<size_t>(kNRouted), false);
    for (int r = 0; r < kTopK; ++r) {
        float bv = -1e30f; int bi = -1;
        for (int e = 0; e < kNRouted; ++e) {
            if (used[static_cast<size_t>(e)]) continue;
            const float v = sel[static_cast<size_t>(e)];
            if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
        }
        out[static_cast<size_t>(r)].eid = bi;
        out[static_cast<size_t>(r)].coeff = prob[static_cast<size_t>(bi)];
        used[static_cast<size_t>(bi)] = true;
    }
    return out;
}

// ── RMSNorm fp32 mirror ───────────────────────────────────────────────────
#if defined(_MSC_VER)
inline float GlmRsqrtHost(float v) { return 1.0f / std::sqrt(v); }
#else
inline float GlmRsqrtHost(float v) { return rsqrtf(v); }
#endif
std::vector<__half> RmsNormFP32(const std::vector<__half> &x,
                                const std::vector<__half> &w, int dim)
{
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) { const float v = F(x[static_cast<size_t>(i)]); acc += v * v; }
    const float rinv = GlmRsqrtHost(acc / static_cast<float>(dim) + lsxcommon::glm::kGlmRmsEps);
    std::vector<__half> y(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
        y[static_cast<size_t>(i)] = H(F(x[static_cast<size_t>(i)]) * rinv * F(w[static_cast<size_t>(i)]));
    return y;
}

std::vector<__half> MatMulFp16(const std::vector<__half> &w, int m, int k,
                               const std::vector<__half> &x, int n)
{
    std::vector<__half> out(static_cast<size_t>(m) * n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int q = 0; q < k; ++q)
                acc += static_cast<double>(F(w[static_cast<size_t>(i) * k + q])) *
                       static_cast<double>(F(x[static_cast<size_t>(q) * n + j]));
            out[static_cast<size_t>(i) * n + j] = H(static_cast<float>(acc));
        }
    return out;
}

std::vector<__half> CpuForwardLayer(const RefRun &ref, RefIdx &R, int l, int turn,
                                    const std::vector<__half> &x_in)
{
    const int seq = static_cast<int>(x_in.size()) / kHidden;
    const RefLayer &rl = ref.layers[static_cast<size_t>(l)];
    if (turn != R.turn_id) { R.kv.insert(R.kv.end(), x_in.begin(), x_in.end()); R.n += seq; R.turn_id = turn; }
    if (l % 4 == 0 && rl.has_indexer) RefRefresh(R, rl);
    std::vector<__half> attn = RefAttention(R, x_in, seq);

    if (l < 3) {
        if (!rl.has_dense) return attn;
        std::vector<__half> normed = RmsNormFP32(attn, rl.post_attn_norm, kHidden);
        std::vector<__half> gate = MatMulFp16(rl.mlp_gate, kInter, kHidden, normed, 1);
        std::vector<__half> up   = MatMulFp16(rl.mlp_up,   kInter, kHidden, normed, 1);
        std::vector<__half> swig(static_cast<size_t>(kInter));
        for (int i = 0; i < kInter; ++i) {
            float g = F(gate[static_cast<size_t>(i)]), u = F(up[static_cast<size_t>(i)]);
            if (g > 10.0f) g = 10.0f; if (u > 10.0f) u = 10.0f; if (u < -10.0f) u = -10.0f;
            const float s = g / (1.0f + std::exp(-g));
            swig[static_cast<size_t>(i)] = H(s * u);
        }
        std::vector<__half> down = MatMulFp16(rl.mlp_down, kHidden, kInter, swig, 1);
        std::vector<__half> out(static_cast<size_t>(kHidden));
        for (int h = 0; h < kHidden; ++h)
            out[static_cast<size_t>(h)] = H(F(attn[static_cast<size_t>(h)]) + F(down[static_cast<size_t>(h)]));
        return out;
    }

    if (l == 78) {
        if (rl.has_moe && !rl.eh_proj.empty()) {
            std::vector<__half> enlarge = MatMulFp16(rl.eh_proj, kHidden, kHidden, attn, 1);
            std::vector<__half> hmtp = RmsNormFP32(enlarge, rl.enorm, kHidden);
            std::vector<__half> normed = RmsNormFP32(hmtp, rl.post_attn_norm, kHidden);
            const std::vector<CpuRouted> routing = CpuRouter(normed, rl.gate, rl.gate_bias);
            std::vector<__half> moe(static_cast<size_t>(kHidden), H(0.0f));
            for (int r = 0; r < kTopK; ++r) {
                const int eid = routing[static_cast<size_t>(r)].eid;
                const float coeff = routing[static_cast<size_t>(r)].coeff;
                if (!ref.ExpertActive(78, eid)) continue;
                const size_t e = static_cast<size_t>(eid);
                std::vector<__half> w1(kMatEl), w3(kMatEl);
                for (int64_t i = 0; i < kMatEl; ++i) {
                    w1[static_cast<size_t>(i)] = H(rl.exp_w1[e][static_cast<size_t>(i)]);
                    w3[static_cast<size_t>(i)] = H(rl.exp_w3[e][static_cast<size_t>(i)]);
                }
                std::vector<__half> inter = MatMulFp16(w1, kInter, kHidden, normed, 1);
                std::vector<__half> gate_ = MatMulFp16(w3, kInter, kHidden, normed, 1);
                std::vector<__half> swig(static_cast<size_t>(kInter));
                for (int i = 0; i < kInter; ++i) {
                    const float g = F(inter[static_cast<size_t>(i)]);
                    const float u = F(gate_[static_cast<size_t>(i)]);
                    swig[static_cast<size_t>(i)] = H((g / (1.0f + std::exp(-g))) * u);
                }
                std::vector<__half> tmp(kHidden);
                for (int h = 0; h < kHidden; ++h) {
                    double acc = 0.0;
                    for (int j = 0; j < kInter; ++j)
                        acc += static_cast<double>(rl.exp_w2[e][static_cast<size_t>(h) * kInter + j]) *
                               static_cast<double>(F(swig[static_cast<size_t>(j)]));
                    tmp[static_cast<size_t>(h)] = H(static_cast<float>(acc));
                }
                for (int h = 0; h < kHidden; ++h)
                    moe[static_cast<size_t>(h)] = H(F(moe[static_cast<size_t>(h)]) + coeff * F(tmp[static_cast<size_t>(h)]));
            }
            std::vector<__half> h78(static_cast<size_t>(kHidden));
            for (int h = 0; h < kHidden; ++h)
                h78[static_cast<size_t>(h)] = H(F(hmtp[static_cast<size_t>(h)]) + F(moe[static_cast<size_t>(h)]));
            std::vector<__half> hn = RmsNormFP32(h78, rl.shared_head_norm, kHidden);
            std::vector<__half> logits(static_cast<size_t>(kVocab));
            for (int v = 0; v < kVocab; ++v) {
                double acc = 0.0;
                const __half *hr = ref.head.data() + static_cast<size_t>(v) * kHidden;
                for (int h = 0; h < kHidden; ++h)
                    acc += static_cast<double>(F(hr[h])) * static_cast<double>(F(hn[static_cast<size_t>(h)]));
                logits[static_cast<size_t>(v)] = H(static_cast<float>(acc));
            }
            return logits;   // [vocab] raw logits; caller argmaxes
        }
        return attn;
    }

    if (!rl.has_moe) return attn;
    std::vector<__half> normed = RmsNormFP32(attn, rl.post_attn_norm, kHidden);
    const std::vector<CpuRouted> routing = CpuRouter(normed, rl.gate, rl.gate_bias);
    std::vector<__half> moe(static_cast<size_t>(kHidden), H(0.0f));
    for (int r = 0; r < kTopK; ++r) {
        const int eid = routing[static_cast<size_t>(r)].eid;
        const float coeff = routing[static_cast<size_t>(r)].coeff;
        if (!ref.ExpertActive(l, eid)) continue;
        const size_t e = static_cast<size_t>(eid);
        std::vector<__half> w1(kMatEl), w3(kMatEl);
        for (int64_t i = 0; i < kMatEl; ++i) {
            w1[static_cast<size_t>(i)] = H(rl.exp_w1[e][static_cast<size_t>(i)]);
            w3[static_cast<size_t>(i)] = H(rl.exp_w3[e][static_cast<size_t>(i)]);
        }
        std::vector<__half> inter = MatMulFp16(w1, kInter, kHidden, normed, 1);
        std::vector<__half> gate_ = MatMulFp16(w3, kInter, kHidden, normed, 1);
        std::vector<__half> swig(static_cast<size_t>(kInter));
        for (int i = 0; i < kInter; ++i) {
            const float g = F(inter[static_cast<size_t>(i)]);
            const float u = F(gate_[static_cast<size_t>(i)]);
            swig[static_cast<size_t>(i)] = H((g / (1.0f + std::exp(-g))) * u);
        }
        std::vector<__half> tmp(kHidden);
        for (int h = 0; h < kHidden; ++h) {
            double acc = 0.0;
            for (int j = 0; j < kInter; ++j)
                acc += static_cast<double>(rl.exp_w2[e][static_cast<size_t>(h) * kInter + j]) *
                       static_cast<double>(F(swig[static_cast<size_t>(j)]));
            tmp[static_cast<size_t>(h)] = H(static_cast<float>(acc));
        }
        for (int h = 0; h < kHidden; ++h)
            moe[static_cast<size_t>(h)] = H(F(moe[static_cast<size_t>(h)]) + coeff * F(tmp[static_cast<size_t>(h)]));
    }
    std::vector<__half> out(static_cast<size_t>(kHidden));
    for (int h = 0; h < kHidden; ++h)
        out[static_cast<size_t>(h)] = H(F(attn[static_cast<size_t>(h)]) + F(moe[static_cast<size_t>(h)]));
    return out;
}

// ── CPU FP64 baseline: full greedy decode over the SAME frozen subset ─────
struct CpuBaseline {
    std::vector<int32_t> tokens;
    float first_maxv = 0.0f;
    std::vector<float> final_logits;   // [vocab] fp16-rounded logits row
};

CpuBaseline CpuGenerate(const RefRun &ref, const std::vector<int32_t> &prompt,
                        int max_new)
{
    RefIdx R;
    CpuBaseline out;
    const int P = static_cast<int>(prompt.size());
    auto embed_row = [&](int32_t id) {
        return std::vector<__half>(
            ref.embed.begin() + static_cast<ptrdiff_t>(id) * kHidden,
            ref.embed.begin() + static_cast<ptrdiff_t>(id + 1) * kHidden);
    };
    std::vector<__half> block;
    for (int t = 0; t < P; ++t) {
        auto row = embed_row(prompt[static_cast<size_t>(t)]);
        block.insert(block.end(), row.begin(), row.end());
    }
    for (int l = 0; l < 79; ++l) block = CpuForwardLayer(ref, R, l, 0, block);
    std::vector<__half> last_logits = block;
    for (int step = 0; step < max_new; ++step) {
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < kVocab; ++v) {
            const float lv = F(last_logits[static_cast<size_t>(v)]);
            if (lv > bv) { bv = lv; best = v; }
        }
        if (step == 0) out.first_maxv = bv;
        out.tokens.push_back(best);
        if (best == kEosIds[0] || best == kEosIds[1] || best == kEosIds[2]) break;
        std::vector<__half> one = embed_row(best);
        for (int l = 0; l < 79; ++l) one = CpuForwardLayer(ref, R, l, step + 1, one);
        last_logits = one;
    }
    out.final_logits.assign(last_logits.begin(), last_logits.end());
    return out;
}

// ── Device read-back helper ───────────────────────────────────────────────
template <typename T>
std::vector<T> ReadBack(const T *d, size_t n, cudaStream_t st)
{
    std::vector<T> h(n);
    cudaMemcpyAsync(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return h;
}
std::vector<__half> ReadHalf(const __half *d, size_t n, cudaStream_t st)
{ return ReadBack<__half>(d, n, st); }

} // namespace

TEST_CASE("GLM-5.2 full 79-layer SKWOS regression: multi-turn decode, no leak, "
          "no index truncation, logits vs CPU FP64",
          "[glm][compute][real-model]")
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    std::filesystem::path model =
        env ? std::filesystem::path(env)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) / "GLM-5.2";
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    GlmPinnedWeights pw;
    REQUIRE(lsxcommon::glm::LoadGlmPinnedWeights(model.string(), pw, st,
                                                 &kPinnedLayers) == cudaSuccess);
    REQUIRE(pw.hidden == kHidden);
    REQUIRE(pw.vocab_size == kVocab);
    REQUIRE(pw.layers.size() == 79);

    GlmExpertTables experts;
    REQUIRE(lsxcommon::glm::LoadGlmExpertTables(
                model.string(), experts, st, &kExpertLayers,
                kExpertsPerLayer) == cudaSuccess);
    REQUIRE(experts.tables.size() == 79);

    // ── Reference weights (identical frozen subset the device loop sees) ──
    RefRun ref;
    ref.layers.resize(79);
    ref.embed = ReadHalf(pw.d_embed, static_cast<size_t>(kVocab) * kHidden, st);
    ref.head  = ReadHalf(pw.d_head,  static_cast<size_t>(kVocab) * kHidden, st);
    auto read_layer = [&](int l, RefLayer &rl) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        if (lw.d_post_attn_norm)
            rl.post_attn_norm = ReadHalf(lw.d_post_attn_norm, kHidden, st);
        if (lw.d_router_gate)
            rl.gate = ReadHalf(lw.d_router_gate,
                               static_cast<size_t>(kNRouted) * kHidden, st);
        if (lw.h_gate_bias && lw.gate_bias_n == kNRouted)
            rl.gate_bias.assign(lw.h_gate_bias, lw.h_gate_bias + kNRouted);
        if (lw.d_mlp_gate && lw.d_mlp_up && lw.d_mlp_down) {
            rl.has_dense = true;
            rl.mlp_gate = ReadHalf(lw.d_mlp_gate, static_cast<size_t>(kInter) * kHidden, st);
            rl.mlp_up   = ReadHalf(lw.d_mlp_up,   static_cast<size_t>(kInter) * kHidden, st);
            rl.mlp_down = ReadHalf(lw.d_mlp_down, static_cast<size_t>(kHidden) * kInter, st);
        }
        if (lw.d_idx_wq_b && lw.d_idx_wk && lw.d_idx_weights_proj && lw.d_idx_k_norm) {
            rl.has_indexer = true;
            rl.wq_b = ReadHalf(lw.d_idx_wq_b, static_cast<size_t>(kHeads) * kHeadDim * kQLora, st);
            rl.wk = ReadHalf(lw.d_idx_wk, static_cast<size_t>(kHeadDim) * kHidden, st);
            rl.wproj = ReadHalf(lw.d_idx_weights_proj, static_cast<size_t>(kHeads) * kHidden, st);
            rl.k_norm = ReadHalf(lw.d_idx_k_norm, kHeadDim, st);
            if (lw.d_idx_k_norm_bias) rl.k_bias = ReadBack(lw.d_idx_k_norm_bias, kHeadDim, st);
        }
        if (lw.d_eh_proj && lw.d_enorm && lw.d_shared_head_norm) {
            rl.eh_proj = ReadHalf(lw.d_eh_proj, static_cast<size_t>(kHidden) * kHidden, st);
            rl.enorm = ReadHalf(lw.d_enorm, kHidden, st);
            rl.shared_head_norm = ReadHalf(lw.d_shared_head_norm, kHidden, st);
        }
        rl.has_moe = !rl.gate.empty() && !rl.post_attn_norm.empty();
    };
    for (int l : kPinnedLayers)
        read_layer(l, ref.layers[static_cast<size_t>(l)]);
    for (int l : kExpertLayers) {
        RefLayer &rl = ref.layers[static_cast<size_t>(l)];
        rl.exp_w1.resize(static_cast<size_t>(kNRouted));
        rl.exp_w2.resize(static_cast<size_t>(kNRouted));
        rl.exp_w3.resize(static_cast<size_t>(kNRouted));
        rl.exp_gs1.assign(static_cast<size_t>(kNRouted), 1.0f);
        rl.exp_gs2.assign(static_cast<size_t>(kNRouted), 1.0f);
        rl.exp_gs3.assign(static_cast<size_t>(kNRouted), 1.0f);
        rl.exp_active.assign(static_cast<size_t>(kNRouted), false);
        for (int e = 0; e < kExpertsPerLayer; ++e) {
            auto p = model / "logestix" / "experts" /
                     ("layer_" + std::to_string(l)) /
                     ("exp_" + std::to_string(e) + ".arrow");
            if (!std::filesystem::exists(p)) continue;
            auto batch_res = lsxcommon::ReadRecordBatchFromFile(p);
            if (!batch_res.ok()) continue;
            auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) continue;
            const auto &pe = *pe_res;
            if (pe.w1_packed.empty() || pe.w2_packed.empty() || pe.w3_packed.empty() ||
                pe.w1_scales.empty() || pe.w2_scales.empty() || pe.w3_scales.empty())
                continue;
            const size_t eidx = static_cast<size_t>(e);
            rl.exp_w1[eidx].resize(static_cast<size_t>(kMatEl));
            rl.exp_w2[eidx].resize(static_cast<size_t>(kMatEl));
            rl.exp_w3[eidx].resize(static_cast<size_t>(kMatEl));
            for (int64_t i = 0; i < kMatEl; ++i) {
                rl.exp_w1[eidx][static_cast<size_t>(i)] = DequantValue(pe.w1_packed, pe.w1_scales, pe.w1_global_scale, i);
                rl.exp_w2[eidx][static_cast<size_t>(i)] = DequantValue(pe.w2_packed, pe.w2_scales, pe.w2_global_scale, i);
                rl.exp_w3[eidx][static_cast<size_t>(i)] = DequantValue(pe.w3_packed, pe.w3_scales, pe.w3_global_scale, i);
            }
            rl.exp_gs1[eidx] = pe.w1_global_scale;
            rl.exp_gs2[eidx] = pe.w2_global_scale;
            rl.exp_gs3[eidx] = pe.w3_global_scale;
            rl.exp_active[eidx] = true;
        }
    }

    // ── CPU FP64 reference (RED/GREEN oracle) ─────────────────────────────
    CpuBaseline baseline = CpuGenerate(ref, kPrompt, kMaxNew);
    REQUIRE(baseline.tokens.size() == static_cast<size_t>(kMaxNew));
    REQUIRE(baseline.final_logits.size() == static_cast<size_t>(kVocab));

    // ── VRAM leak probe: capture free bytes before the first full run. ────
    size_t free_before = 0, total_vram = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total_vram) == cudaSuccess);

    // ── Device run #1 ─────────────────────────────────────────────────────
    GlmGenerationResult r1;
    REQUIRE(lsxcommon::glm::RunGlmGeneration(st, pw, experts, kPrompt,
                                             kMaxNew, r1) == cudaSuccess);
    REQUIRE(r1.ok);
    REQUIRE(r1.error_msg.empty());
    REQUIRE(r1.tokens.size() == static_cast<size_t>(kMaxNew));
    REQUIRE(r1.final_logits.size() == static_cast<size_t>(kVocab));

    // (a) Index-boundary truncation: every produced token id is in range.
    for (int32_t id : r1.tokens) {
        CHECK(id >= 0);
        CHECK(id < kVocab);
    }
    // (a) KV isolation: no MTP leak into the IndexShare cache — slot growth is
    //     exactly prompt + generated tokens.
    const bool eos1 = !r1.tokens.empty() &&
        (r1.tokens.back() == kEosIds[0] || r1.tokens.back() == kEosIds[1] ||
         r1.tokens.back() == kEosIds[2]);
    const int expect_slots = static_cast<int>(kPrompt.size()) +
                             static_cast<int>(r1.tokens.size()) - (eos1 ? 1 : 0);
    CHECK(r1.kv_slots_used == expect_slots);

    // (b) Greedy decode: 0 tokens-drift vs the FP64 baseline.
    int drift = 0;
    for (size_t i = 0; i < baseline.tokens.size(); ++i)
        if (i < r1.tokens.size() && baseline.tokens[i] != r1.tokens[i]) ++drift;
    CHECK(drift == 0);

    // (b) Element-for-element convergence of the final MTP logits row against
    //     the independent CPU FP64 reference (fp16-rounded both sides), within
    //     the bounded tolerance envelope used for the head-row parity check.
    int worst_logit = -1;
    float worst_abs = 0.0f;
    for (int v = 0; v < kVocab; ++v) {
        const float refv = baseline.final_logits[static_cast<size_t>(v)];
        const float devv = r1.final_logits[static_cast<size_t>(v)];
        const float abs_err = std::fabs(devv - refv);
        if (abs_err > worst_abs) { worst_abs = abs_err; worst_logit = v; }
        // Same bounded envelope the single-head parity check permits.
        CHECK(abs_err <= 1e-2f * (1.0f + std::fabs(refv)) + 0.5f);
    }
    INFO("worst logit index=" << worst_logit << " abs_err=" << worst_abs);

    // First-token argmax logit also agrees within the envelope.
    CHECK(std::isfinite(r1.first_maxv));
    CHECK(std::fabs(r1.first_maxv - baseline.first_maxv) <=
          1e-2f * (1.0f + std::fabs(baseline.first_maxv)) + 0.5f);

    // ── Multi-turn + leak regression: repeated fresh full runs must not
    //    monotonically consume VRAM (the unified arena frees its ONE device
    //    allocation on every exit). ─────────────────────────────────────────
    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total_vram) == cudaSuccess);
    CHECK(free_after >= free_before);   // no net VRAM growth after run #1

    const std::vector<int32_t> prompts[3] = {
        kPrompt,
        {154820, 1636, 258, 7, 258, 1636},
        {5},
    };
    for (const auto &p : prompts) {
        for (int rep = 0; rep < 2; ++rep) {
            GlmGenerationResult rr;
            REQUIRE(lsxcommon::glm::RunGlmGeneration(
                        st, pw, experts, p, kMaxNew, rr) == cudaSuccess);
            REQUIRE(rr.ok);
            for (int32_t id : rr.tokens) { CHECK(id >= 0); CHECK(id < kVocab); }
            const bool e = !rr.tokens.empty() &&
                (rr.tokens.back() == kEosIds[0] || rr.tokens.back() == kEosIds[1] ||
                 rr.tokens.back() == kEosIds[2]);
            const int exp = static_cast<int>(p.size()) +
                            static_cast<int>(rr.tokens.size()) - (e ? 1 : 0);
            CHECK(rr.kv_slots_used == exp);
        }
    }
    size_t free_final = 0;
    REQUIRE(cudaMemGetInfo(&free_final, &total_vram) == cudaSuccess);
    // A leak of the arena's single big allocation (KV + MTP block) would show
    // as a clear drop; allow a small tolerance for driver/allocator jitter.
    CHECK(free_final + (4ull << 20) >= free_before);

    experts.Free();
    pw.Free();
    cudaStreamDestroy(st);
}
