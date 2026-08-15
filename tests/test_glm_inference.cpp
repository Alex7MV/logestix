// TDD Cycle 6 — GLM-5.2 heterogeneous 79-layer execution loop (dense 0-2 ->
// IndexShare attention + MoE expert stack 3-77 -> Layer-78 MTP epilogue) wired
// end-to-end into the model's async generation pass. Greedy multi-token
// generation on REAL weights (VRAM-budgeted subset) must:
//   (a) converge bit-exactly against an independent FP64 host baseline of the
//       SAME frozen subset math and exit with 0 tokens-drift,
//   (b) be byte-identical across two full device runs (determinism),
//   (c) keep the decoder IndexShare KV staging untouched by the MTP block:
//       kv_slots_used must equal prompt_len + generated tokens exactly.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/glm_inference.h>
#include <lsxcommon/glm_inference_load.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/schema.h>

using lsxcommon::glm::GlmPinnedWeights;
using lsxcommon::glm::GlmExpertTables;
using lsxcommon::glm::GlmGenerationResult;
using lsxcommon::glm::GlmExpertDeviceBlock;
using lsxcommon::glm::GlmRoutedSlot;

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
constexpr int64_t kMatEl =
    static_cast<int64_t>(kInter) * kHidden;                     // 12582912

constexpr float kAttnScale = 0.088388347648318440f;             // 1/sqrt(128)

// Real-model EOS ids (config.json eos_token_id).
constexpr int kEosIds[3] = {154820, 154827, 154829};

// ── VRAM-budgeted real-weight subset (the frozen "model baseline" contract) ──
// Dense stack 0-2, an early MoE window (3-10) with live attention + routed
// experts, a late MoE tail (76-77) and the full MTP layer 78. Every layer NOT
// in the pinned set degrades its MoE branch to identity (attention still runs
// over the shared IndexShare staging); every expert NOT loaded is a
// deterministic zero-contribution slot.
const std::vector<int> kPinnedLayers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        76, 77, 78};
const std::vector<int> kExpertLayers = {3, 4, 6, 8, 78};
constexpr int kExpertsPerLayer = 16;

const std::vector<int32_t> kPrompt = {154820, 258, 1636, 7};
constexpr int kMaxNew = 6;

// ── FP16 exact conversion helpers (mirror __float2half_rn / __half2float) ──
inline uint16_t FloatToHalfRne(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const int      exp  = static_cast<int>((u >> 23) & 0xFFu);
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
    const int exp2 = exp - 127;
    if (exp2 >= 15) return static_cast<uint16_t>(sign | 0x7C00u);
    const uint32_t F = 0x800000u | mant;
    if (exp2 >= -14) {
        int      e  = exp2 + 15;
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
    uint32_t s   = F >> shift;
    const uint32_t rem = F & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (s & 1u))) ++s;
    if (s == 0x400u)
        return static_cast<uint16_t>(sign | 0x0400u);
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

// E2M1 decode (cycle-4 fixture): value = E2M1(nibble) * 2^(((s>>3)&0xF)-7) * gs
// for the byte-packed NVFP4 layout the converter wrote.
constexpr float kE2M1Table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
inline float E2M1Value(uint8_t nib)
{
    return (nib & 8u) ? -kE2M1Table[nib & 7u] : kE2M1Table[nib & 7u];
}
inline float DequantValue(const std::vector<uint8_t> &packed,
                          const std::vector<uint8_t> &scales,
                          float gs, int64_t i)
{
    const uint8_t sb = scales[static_cast<size_t>(i / lsxcommon::glm::kGlmNvfp4Block)];
    const int      e4 = (sb >> 3) & 0x0F;
    const float    sc = (sb == 0) ? 0.0f : std::ldexp(1.0f, e4 - 7);
    const uint8_t  b   = packed[static_cast<size_t>(i >> 1)];
    const uint8_t  nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                 : static_cast<uint8_t>(b & 0x0Fu);
    return F(H(E2M1Value(nib) * sc * gs));
}

// ── Host-side mirror of the device FP16 weights the loop consumes ───────────
struct RefLayer {
    bool has_moe  = false;
    bool has_dense = false;
    bool has_indexer = false;
    std::vector<__half> post_attn_norm;  // [hidden]
    std::vector<__half> gate;            // [256, hidden] fp16
    std::vector<float>  gate_bias;       // [256] fp32
    std::vector<__half> mlp_gate, mlp_up, mlp_down;   // dense 0-2
    // indexer (refresh at layer_idx % 4 == 0 when present)
    std::vector<__half> wq_b, wk, wproj, k_norm;
    std::vector<float>  k_bias;
    // MTP (layer 78)
    std::vector<__half> eh_proj, enorm, shared_head_norm;
    // routed experts: dequantized fp16-rounded float matrices [eid][matEl]
    std::vector<std::vector<float>> exp_w1, exp_w2, exp_w3;
    std::vector<float> exp_gs1, exp_gs2, exp_gs3;
    std::vector<bool>  exp_active;
};

struct RefRun {
    std::vector<__half> embed;   // [vocab, hidden]
    std::vector<__half> head;    // [vocab, hidden]
    std::vector<RefLayer> layers;   // [79]

    bool ExpertActive(int layer, int eid) const
    {
        const auto &rl = layers[static_cast<size_t>(layer)];
        if (eid < 0 || eid >= static_cast<int>(rl.exp_active.size()))
            return false;
        return rl.exp_active[static_cast<size_t>(eid)];
    }
};

// ── Cycle-5 exact mirrors (IndexShare attention reference) ──────────────────
struct RefIdx {
    std::vector<__half> kv;        // [n*hidden]
    std::vector<__half> q_key;     // [maxSlots*headDim]
    std::vector<__half> key;       // [maxSlots*headDim]
    std::vector<int>    slot;      // [maxSlots]
    int n = 0;
    int turn_id = -1;
    RefIdx()
        : q_key(static_cast<size_t>(kMaxSlots) * kHeadDim),
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
            for (int k = 0; k < kHidden; ++k)
                acc = fmaf(F(v[k]), F(wp[k]), acc);
            sc[static_cast<size_t>(h)] = acc;
        }
        int best = 0;
        float bv = sc[0];
        for (int h = 1; h < kHeads; ++h)
            if (sc[static_cast<size_t>(h)] > bv) { bv = sc[static_cast<size_t>(h)]; best = h; }
        R.slot[static_cast<size_t>(s)] = best;
        std::vector<float> kraw(static_cast<size_t>(kHeadDim), 0.0f);
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wr = W.wk.data() + static_cast<size_t>(r) * kHidden;
            for (int k = 0; k < kHidden; ++k)
                acc = fmaf(F(v[k]), F(wr[k]), acc);
            kraw[static_cast<size_t>(r)] = acc;
        }
        float sq = 0.0f;
        for (int r = 0; r < kHeadDim; ++r)
            sq = fmaf(kraw[static_cast<size_t>(r)], kraw[static_cast<size_t>(r)], sq);
        const float rms = std::sqrt(sq / static_cast<float>(kHeadDim) +
                                    lsxcommon::glm::kGlmRmsEps);
        __half *kd = R.key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            const float kb = W.k_bias.empty()
                                 ? 0.0f
                                 : W.k_bias[static_cast<size_t>(r)];
            kd[r] = H((kraw[static_cast<size_t>(r)] / rms) *
                      F(W.k_norm[static_cast<size_t>(r)]) + kb);
        }
        __half *qd = R.q_key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wqr = W.wq_b.data() +
                (static_cast<size_t>(best) * kHeadDim + r) * kQLora;
            for (int k = 0; k < kQLora; ++k)
                acc = fmaf(F(v[k]), F(wqr[k]), acc);
            qd[r] = H(acc);
        }
    }
}

// Exact mirror of GlmIdxAttnKernel (per-token deterministic softmax order).
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
            for (int d = 0; d < kHeadDim; ++d)
                acc = fmaf(F(qq[d]), F(ks[d]), acc);
            sims[static_cast<size_t>(s)] = acc * kAttnScale;
        }
        float m = -1e30f;
        for (int s = 0; s < R.n; ++s) m = std::max(m, sims[static_cast<size_t>(s)]);
        std::vector<float> p(static_cast<size_t>(R.n), 0.0f);
        float sum = 0.0f;
        for (int s = 0; s < R.n; ++s) {
            p[static_cast<size_t>(s)] = std::exp(sims[static_cast<size_t>(s)] - m);
            sum += p[static_cast<size_t>(s)];
        }
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

// ── Router reference (fp32 mirror of the device top-k total order) ──────────
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
    // Selection keys: logit + optional bias.
    std::vector<float> sel(static_cast<size_t>(kNRouted), 0.0f);
    for (int e = 0; e < kNRouted; ++e)
        sel[static_cast<size_t>(e)] = use_bias ? logit[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)]
                                               : logit[static_cast<size_t>(e)];
    // Softmax over the UNBIASED logits (device kernel normalizes logit only).
    float lmax = -1e30f;
    for (float v : logit) lmax = std::max(lmax, v);
    float psum = 0.0f;
    for (float v : logit) psum += std::exp(v - lmax);
    const float inv = 1.0f / psum;
    std::vector<float> prob(static_cast<size_t>(kNRouted), 0.0f);
    for (int e = 0; e < kNRouted; ++e)
        prob[static_cast<size_t>(e)] = std::exp(logit[static_cast<size_t>(e)] - lmax) * inv;

    // Total-order top-k: larger key wins, equal key -> smallest eid.
    std::vector<CpuRouted> out(static_cast<size_t>(kTopK));
    std::vector<bool> used(static_cast<size_t>(kNRouted), false);
    for (int r = 0; r < kTopK; ++r) {
        float bv = -1e30f;
        int   bi = -1;
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

// ── RMSNorm fp32 mirror of the device kernel (strided partials + butterfly) ──
// rsqrtf is a CUDA/glibc builtin, absent from MSVC's host <cmath>; on Windows
// fall back to the mathematically identical 1.0f/std::sqrt so this real-model
// test still compiles there (it is SKIP'd without the model anyway). Linux
// keeps calling rsqrtf, so its FP64 parity is unchanged.
#if defined(_MSC_VER)
inline float GlmRsqrtHost(float v) { return 1.0f / std::sqrt(v); }
#else
inline float GlmRsqrtHost(float v) { return rsqrtf(v); }
#endif
std::vector<__half> RmsNormFP32(const std::vector<__half> &x,
                                const std::vector<__half> &w, int dim)
{
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float v = F(x[static_cast<size_t>(i)]);
        acc += v * v;
    }
    const float rinv = GlmRsqrtHost(acc / static_cast<float>(dim) + lsxcommon::glm::kGlmRmsEps);
    std::vector<__half> y(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
        y[static_cast<size_t>(i)] = H(F(x[static_cast<size_t>(i)]) * rinv * F(w[static_cast<size_t>(i)]));
    return y;
}

// FP16 store of a [m, k] x [k, n] product (fp64 accumulate, fp16 RNE result),
// approximating the device cuBLAS fp16-GEMM-with-fp32-compute rounding point.
std::vector<__half> MatMulFp16(const std::vector<__half> &w, int m, int k,
                               const std::vector<__half> &x, int n)
{
    // w row-major [m, k]; x row-major [k, n]; out [m, n]
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

// ── One MoE/dense/identity layer, mirroring the locked heterogeneous loop ───
std::vector<__half> CpuForwardLayer(const RefRun &ref, RefIdx &R,
                                    int l, int turn,
                                    const std::vector<__half> &x_in)
{
    const int seq = static_cast<int>(x_in.size()) / kHidden;
    const RefLayer &rl = ref.layers[static_cast<size_t>(l)];

    if (turn != R.turn_id) {
        R.kv.insert(R.kv.end(), x_in.begin(), x_in.end());
        R.n += seq;
        R.turn_id = turn;
    }
    if (l % 4 == 0 && rl.has_indexer) RefRefresh(R, rl);
    std::vector<__half> attn = RefAttention(R, x_in, seq);

    if (l < 3) {
        // Dense MLP layer 0-2: post-attn RMSNorm -> gate/up -> swiglu -> down -> residual.
        if (!rl.has_dense) return attn;
        std::vector<__half> normed = RmsNormFP32(attn, rl.post_attn_norm, kHidden);
        std::vector<__half> gate = MatMulFp16(rl.mlp_gate, kInter, kHidden, normed, 1);
        std::vector<__half> up   = MatMulFp16(rl.mlp_up,   kInter, kHidden, normed, 1);
        std::vector<__half> swig(static_cast<size_t>(kInter));
        for (int i = 0; i < kInter; ++i) {
            float g = F(gate[static_cast<size_t>(i)]);
            float u = F(up[static_cast<size_t>(i)]);
            if (g > 10.0f) g = 10.0f;
            if (u > 10.0f) u = 10.0f;
            if (u < -10.0f) u = -10.0f;
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
        // Layer-78 MTP epilogue (dense fallback for missing tables -> identity).
        if (rl.has_moe && !rl.eh_proj.empty()) {
            std::vector<__half> enlarge = MatMulFp16(rl.eh_proj, kHidden, kHidden, attn, 1);
            std::vector<__half> hmtp = RmsNormFP32(enlarge, rl.enorm, kHidden);
            std::vector<__half> normed = RmsNormFP32(hmtp, rl.post_attn_norm, kHidden);
            const std::vector<CpuRouted> routing = CpuRouter(normed, rl.gate, rl.gate_bias);
            std::vector<__half> moe(static_cast<size_t>(kHidden), H(0.0f));
            for (int r = 0; r < kTopK; ++r) {
                const int   eid = routing[static_cast<size_t>(r)].eid;
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
            // Head: shared_head.norm -> lm_head^T (fp64), returning the raw
            // logits row so the caller can run the vocab argmax once.
            std::vector<__half> hn = RmsNormFP32(h78, rl.shared_head_norm, kHidden);
            std::vector<__half> logits(static_cast<size_t>(kVocab));
            for (int v = 0; v < kVocab; ++v) {
                double acc = 0.0;
                const __half *hr = ref.head.data() + static_cast<size_t>(v) * kHidden;
                for (int h = 0; h < kHidden; ++h)
                    acc += static_cast<double>(F(hr[h])) * static_cast<double>(F(hn[static_cast<size_t>(h)]));
                logits[static_cast<size_t>(v)] = H(static_cast<float>(acc));
            }
            return logits;  // [vocab] raw logits; caller argmaxes.
        }
        return attn;
    }

    // MoE layers 3-77.
    if (!rl.has_moe) return attn;
    std::vector<__half> normed = RmsNormFP32(attn, rl.post_attn_norm, kHidden);
    const std::vector<CpuRouted> routing = CpuRouter(normed, rl.gate, rl.gate_bias);
    std::vector<__half> moe(static_cast<size_t>(kHidden), H(0.0f));
    for (int r = 0; r < kTopK; ++r) {
        const int   eid = routing[static_cast<size_t>(r)].eid;
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

// ── CPU baseline: full greedy decode over the same frozen subset ────────────
struct CpuBaseline {
    std::vector<int32_t> tokens;
    float first_maxv = 0.0f;
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

    // Prefill: turn 0, seq = P.
    std::vector<__half> block;
    for (int t = 0; t < P; ++t) {
        auto row = embed_row(prompt[static_cast<size_t>(t)]);
        block.insert(block.end(), row.begin(), row.end());
    }
    for (int l = 0; l < 79; ++l)
        block = CpuForwardLayer(ref, R, l, 0, block);

    std::vector<__half> last_logits = block;   // layer-78 raw logits row [vocab]
    for (int step = 0; step < max_new; ++step) {
        int best = 0;
        float bv = -1e30f;
        for (int v = 0; v < kVocab; ++v) {
            const float lv = F(last_logits[static_cast<size_t>(v)]);
            if (lv > bv) { bv = lv; best = v; }
        }
        if (step == 0) out.first_maxv = bv;
        out.tokens.push_back(best);
        if (best == kEosIds[0] || best == kEosIds[1] || best == kEosIds[2])
            break;

        // Decode step: embed the produced token, forward all 79 layers, MTP head.
        std::vector<__half> one = embed_row(best);
        for (int l = 0; l < 79; ++l)
            one = CpuForwardLayer(ref, R, l, step + 1, one);
        last_logits = one;
    }
    return out;
}

// ── Device buffer read-back helper ──────────────────────────────────────────
template <typename T>
std::vector<T> ReadBack(const T *d, size_t n, cudaStream_t st)
{
    std::vector<T> h(n);
    cudaMemcpyAsync(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return h;
}

std::vector<__half> ReadHalf(const __half *d, size_t n, cudaStream_t st)
{
    return ReadBack<__half>(d, n, st);
}

} // namespace

TEST_CASE("GLM-5.2 79-layer loop + MTP epilogue: real-weight generation "
          "bit-exact vs FP64 baseline, 0 tokens-drift, KV isolation",
          "[glm][inference][real-model]")
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    std::filesystem::path model =
        env ? std::filesystem::path(env)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) / "GLM-5.2";
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    cudaFree(0);
    cublasHandle_t ch;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    // ── Load the VRAM-budgeted real-weight subset ───────────────────────────
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
    for (int l : kExpertLayers) {
        const auto &tab = experts.tables[static_cast<size_t>(l)];
        REQUIRE(tab.size() == static_cast<size_t>(kNRouted));
        int active = 0;
        for (const auto &b : tab)
            if (b.d_w1_packed && b.d_w2_packed && b.d_w3_packed) ++active;
        REQUIRE(active > 0);
    }

    // ── Read back every weight the CPU baseline consumes ────────────────────
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
            rl.mlp_gate  = ReadHalf(lw.d_mlp_gate, static_cast<size_t>(kInter) * kHidden, st);
            rl.mlp_up    = ReadHalf(lw.d_mlp_up,   static_cast<size_t>(kInter) * kHidden, st);
            rl.mlp_down  = ReadHalf(lw.d_mlp_down, static_cast<size_t>(kHidden) * kInter, st);
        }
        if (lw.d_idx_wq_b && lw.d_idx_wk && lw.d_idx_weights_proj &&
            lw.d_idx_k_norm) {
            rl.has_indexer = true;
            rl.wq_b = ReadHalf(lw.d_idx_wq_b,
                static_cast<size_t>(kHeads) * kHeadDim * kQLora, st);
            rl.wk = ReadHalf(lw.d_idx_wk,
                static_cast<size_t>(kHeadDim) * kHidden, st);
            rl.wproj = ReadHalf(lw.d_idx_weights_proj,
                static_cast<size_t>(kHeads) * kHidden, st);
            rl.k_norm = ReadHalf(lw.d_idx_k_norm, kHeadDim, st);
            if (lw.d_idx_k_norm_bias)
                rl.k_bias = ReadBack(lw.d_idx_k_norm_bias, kHeadDim, st);
        }
        if (lw.d_eh_proj && lw.d_enorm && lw.d_shared_head_norm) {
            rl.eh_proj = ReadHalf(lw.d_eh_proj,
                static_cast<size_t>(kHidden) * kHidden, st);
            rl.enorm = ReadHalf(lw.d_enorm, kHidden, st);
            rl.shared_head_norm = ReadHalf(lw.d_shared_head_norm, kHidden, st);
        }
        rl.has_moe = !rl.gate.empty() && !rl.post_attn_norm.empty();
    };
    for (int l : kPinnedLayers)
        read_layer(l, ref.layers[static_cast<size_t>(l)]);

    // Routed experts: read the packed arrow files straight from disk and
    // dequantize on host (identical byte stream the loader uploads).
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
            auto exp_dir = model / "logestix" / "experts" /
                           ("layer_" + std::to_string(l));
            auto p = exp_dir / ("exp_" + std::to_string(e) + ".arrow");
            if (!std::filesystem::exists(p)) continue;
            auto batch_res = lsxcommon::ReadRecordBatchFromFile(p);
            if (!batch_res.ok()) continue;
            auto pe_res = lsxcommon::ExpertSchema::UnpackBatch(*batch_res);
            if (!pe_res.ok()) continue;
            const auto &pe = *pe_res;
            if (pe.w1_packed.empty() || pe.w2_packed.empty() ||
                pe.w3_packed.empty() || pe.w1_scales.empty() ||
                pe.w2_scales.empty() || pe.w3_scales.empty())
                continue;
            const size_t eidx = static_cast<size_t>(e);
            rl.exp_w1[eidx].resize(static_cast<size_t>(kMatEl));
            rl.exp_w2[eidx].resize(static_cast<size_t>(kMatEl));
            rl.exp_w3[eidx].resize(static_cast<size_t>(kMatEl));
            for (int64_t i = 0; i < kMatEl; ++i) {
                rl.exp_w1[eidx][static_cast<size_t>(i)] =
                    DequantValue(pe.w1_packed, pe.w1_scales, pe.w1_global_scale, i);
                rl.exp_w2[eidx][static_cast<size_t>(i)] =
                    DequantValue(pe.w2_packed, pe.w2_scales, pe.w2_global_scale, i);
                rl.exp_w3[eidx][static_cast<size_t>(i)] =
                    DequantValue(pe.w3_packed, pe.w3_scales, pe.w3_global_scale, i);
            }
            rl.exp_gs1[eidx] = pe.w1_global_scale;
            rl.exp_gs2[eidx] = pe.w2_global_scale;
            rl.exp_gs3[eidx] = pe.w3_global_scale;
            rl.exp_active[eidx] = true;
        }
    }

    // ── FP64 CPU baseline over the identical frozen subset ─────────────────
    CpuBaseline baseline = CpuGenerate(ref, kPrompt, kMaxNew);
    REQUIRE(baseline.tokens.size() == static_cast<size_t>(kMaxNew));
    INFO("baseline tokens: " << [&]() {
        std::string s;
        for (int32_t t : baseline.tokens) s += " " + std::to_string(t);
        return s;
    }());
    INFO("baseline first_maxv: " << baseline.first_maxv);

    // ── Device generation (first run = the model under test) ────────────────
    GlmGenerationResult r1;
    REQUIRE(lsxcommon::glm::RunGlmGeneration(
                ch, st, pw, experts, kPrompt, kMaxNew, r1) == cudaSuccess);
    REQUIRE(r1.ok);
    REQUIRE(r1.error_msg.empty());
    REQUIRE(r1.tokens.size() == static_cast<size_t>(kMaxNew));
    INFO("device tokens: " << [&]() {
        std::string s;
        for (int32_t t : r1.tokens) s += " " + std::to_string(t);
        return s;
    }());
    INFO("device first_maxv: " << r1.first_maxv);

    // (a) Bit-exact convergence: 0 tokens-drift against the CPU baseline.
    int tokens_drift = 0;
    for (size_t i = 0; i < baseline.tokens.size(); ++i)
        if (baseline.tokens[i] != r1.tokens[i]) ++tokens_drift;
    CHECK(tokens_drift == 0);
    for (int32_t id : r1.tokens) {
        CHECK(id >= 0);
        CHECK(id < kVocab);
    }

    // First-token argmax logit agrees with the reference within a bounded
    // envelope (cuBLAS fp32 head accumulation vs the fp64 reference).
    CHECK(std::isfinite(r1.first_maxv));
    CHECK(std::fabs(r1.first_maxv - baseline.first_maxv) <=
          1e-2f * (1.0f + std::fabs(baseline.first_maxv)) + 0.5f);

    // (c) MTP isolation: the decoder IndexShare staging grew by exactly
    //     prompt_len + generated tokens. A Layer-78 leak would add extra slots.
    //     The EOS token itself is NOT forwarded into the KV cache, so a run
    //     that ended on EOS accounts for one fewer slot than tokens emitted.
    const bool trailing_eos = !r1.tokens.empty() &&
        (r1.tokens.back() == kEosIds[0] || r1.tokens.back() == kEosIds[1] ||
         r1.tokens.back() == kEosIds[2]);
    const int expect_slots = static_cast<int>(kPrompt.size()) +
                             static_cast<int>(r1.tokens.size()) -
                             (trailing_eos ? 1 : 0);
    CHECK(r1.kv_slots_used == expect_slots);

    // (b) Full determinism: a second fresh run must be byte-identical.
    GlmGenerationResult r2;
    REQUIRE(lsxcommon::glm::RunGlmGeneration(
                ch, st, pw, experts, kPrompt, kMaxNew, r2) == cudaSuccess);
    REQUIRE(r2.ok);
    REQUIRE(r2.tokens.size() == r1.tokens.size());
    CHECK(std::memcmp(r2.tokens.data(), r1.tokens.data(),
                      r1.tokens.size() * sizeof(int32_t)) == 0);
    CHECK(r2.kv_slots_used == r1.kv_slots_used);

    // ── Edge: single-token prompt. The loop must still embed, forward all 79
    //    layers and produce exactly max_new tokens; KV growth = 1 + generated.
    {
        const std::vector<int32_t> one = {5};
        GlmGenerationResult rs;
        REQUIRE(lsxcommon::glm::RunGlmGeneration(
                    ch, st, pw, experts, one, 1, rs) == cudaSuccess);
        REQUIRE(rs.ok);
        REQUIRE(rs.tokens.size() == 1);
        const bool eos_one = rs.tokens[0] == kEosIds[0] ||
                             rs.tokens[0] == kEosIds[1] ||
                             rs.tokens[0] == kEosIds[2];
        CHECK(rs.kv_slots_used == 1 + (eos_one ? 0 : 1));
    }

    // ── Edge: capacity guard. prompt + max_new must fit the fixed decoder KV
    //    staging (kGlmIndexMaxSlots); a request that exceeds it is rejected
    //    deterministically with cudaErrorInvalidValue (no device OOB access).
    {
        const std::vector<int32_t> big(static_cast<size_t>(kMaxSlots - 1), 0);
        GlmGenerationResult rb;
        CHECK(lsxcommon::glm::RunGlmGeneration(
                  ch, st, pw, experts, big, 2, rb) == cudaErrorInvalidValue);
        CHECK(!rb.ok);
    }

    // ── Edge: out-of-range prompt ids are rejected before any allocation, so
    //    the embed table can never be indexed past its bounds.
    {
        const std::vector<int32_t> bad0 = {0, 200000};
        GlmGenerationResult rb0;
        CHECK(lsxcommon::glm::RunGlmGeneration(
                  ch, st, pw, experts, bad0, 1, rb0) == cudaErrorInvalidValue);
        CHECK(!rb0.ok);
        const std::vector<int32_t> bad1 = {0, -3};
        GlmGenerationResult rb1;
        CHECK(lsxcommon::glm::RunGlmGeneration(
                  ch, st, pw, experts, bad1, 1, rb1) == cudaErrorInvalidValue);
        CHECK(!rb1.ok);
    }

    experts.Free();
    pw.Free();
    cudaStreamDestroy(st);
    cublasDestroy(ch);
}
