// TDD Cycle 5 — GLM-5.2 IndexShare attention pass (layers 3-77): the central
// memory-mapped shared indexer mask arrays (wq_b / wk / weights_proj) are
// calculated, block-aligned and refreshed ONLY on the structural modulo check
// (layer_idx % 4 == 0). Intermediate layers must completely bypass every indexer
// projection and reuse the cached sparse KV pointers from the pre-allocated
// staging buffers, so (a) a refresh layer, its three reuse siblings and a
// multi-turn refresh all converge element-for-element against the deterministic
// FP16 reference, and (b) bitwise-identical runs with the indexer weights
// corrupted on device prove the reuse path never reads them.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lsxcommon/glm_moe_inference.h"
#include "lsxcommon/glm_moe_load.h"

using lsxcommon::glm::GlmIndexShareState;
using lsxcommon::glm::GlmLayerWeights;

namespace {

constexpr int kHidden   = lsxcommon::glm::kGlmHidden;          // 6144
constexpr int kHeads    = lsxcommon::glm::kGlmIndexNHeads;      // 32
constexpr int kHeadDim  = lsxcommon::glm::kGlmIndexHeadDim;     // 128
constexpr int kQLora    = lsxcommon::glm::kGlmQLoraRank;        // 2048
constexpr int kMaxSlots = lsxcommon::glm::kGlmIndexMaxSlots;    // 1024
constexpr int kSeq      = 2;

// 1/sqrt(128.0f). Both the device attention kernel and the host reference must
// multiply the raw float dot by this exact float literal.
constexpr float kAttnScale = 0.088388347648318440f;

inline uint16_t FloatToHalfRne(float v)
{
    // RNE float -> fp16 bits, mirroring __float2half_rn exactly for the
    // normal-range values this fixture produces (kraw/rms ~ O(1), queries
    // ~ O(0.2), outputs ~ O(0.5)).
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
        int      e   = exp2 + 15;
        uint32_t m10 = (F >> 13) & 0x3FFu;
        const uint32_t rem = F & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u))) {
            ++m10;
            if (m10 == 0x400u) { m10 = 0; ++e; }
        }
        if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(
            sign | (static_cast<uint16_t>(e) << 10) |
            static_cast<uint16_t>(m10));
    }
    if (exp2 < -24) return sign;
    const int shift = -exp2 - 1;
    uint32_t  s     = F >> shift;
    const uint32_t rem  = F & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (s & 1u))) ++s;
    if (s == 0x400u) return static_cast<uint16_t>(sign | 0x0400u);
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(s));
}

inline __half FloatToHalf(float v)
{
    const uint16_t bits = FloatToHalfRne(v);
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return h;
}

// Deterministic hash into an FP16-exact small lattice. Random-walk dot
// products of ~2048/6144 terms then stay O(0.1..1) so the softmax sims never
// approach the expf overflow bound while exercising real mixed arithmetic.
inline __half HashHalf(int64_t i, unsigned seed)
{
    unsigned x = static_cast<unsigned>(i) ^ (0x9E3779B9u * seed);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return __float2half(0.03125f * static_cast<float>((int)(x % 9) - 4));
}

inline float HashFloat(int64_t i, unsigned seed)
{
    unsigned x = static_cast<unsigned>(i) ^ (0x9E3779B9u * seed);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return 0.02f * static_cast<float>((int)(x % 5) - 2);
}

// Reference weights (host copy of the device FP16 uploads).
struct RefWeights {
    std::vector<__half> wq_b;     // [heads*headDim, qLora]   = [4096, 2048]
    std::vector<__half> wk;       // [headDim, hidden]        = [128, 6144]
    std::vector<__half> wproj;    // [heads, hidden]          = [32, 6144]
    std::vector<__half> k_norm;   // [headDim]
    std::vector<float>  k_bias;   // [headDim]
};

// Host-mirror of GlmIndexShareState (values stored as the FP16 the device
// caches, converted back at the exact same rounding points).
struct RefIdx {
    std::vector<__half> kv;       // [n*hidden]
    std::vector<__half> q_key;    // [maxSlots*headDim]
    std::vector<__half> key;      // [maxSlots*headDim]
    std::vector<int>    slot;     // [maxSlots]
    int n = 0;
    int turn_id = -1;
    RefIdx()
        : q_key(static_cast<size_t>(kMaxSlots) * kHeadDim),
          key(static_cast<size_t>(kMaxSlots) * kHeadDim),
          slot(static_cast<size_t>(kMaxSlots), 0) {}
};

// Exact mirror of the refresh-layer device math (all loops ascending, global
// FP32 fmaf accumulation, FP16 only at the store points).
void RefRefresh(RefIdx &R, const RefWeights &W)
{
    for (int s = 0; s < R.n; ++s) {
        const __half *v = R.kv.data() + static_cast<size_t>(s) * kHidden;

        // score[h] = dot(v, wproj[h]) — ascending-k fmaf, FP32.
        std::vector<float> sc(static_cast<size_t>(kHeads), 0.0f);
        for (int h = 0; h < kHeads; ++h) {
            float acc = 0.0f;
            const __half *wp =
                W.wproj.data() + static_cast<size_t>(h) * kHidden;
            for (int k = 0; k < kHidden; ++k)
                acc = fmaf(__half2float(v[k]), __half2float(wp[k]), acc);
            sc[static_cast<size_t>(h)] = acc;
        }
        int   best = 0;
        float bv   = sc[0];
        for (int h = 1; h < kHeads; ++h)
            if (sc[static_cast<size_t>(h)] > bv) {
                bv = sc[static_cast<size_t>(h)];
                best = h;
            }
        R.slot[static_cast<size_t>(s)] = best;

        // key[r] = (kraw[r]/rms) * k_norm[r] + k_bias[r], FP16 store.
        std::vector<float> kraw(static_cast<size_t>(kHeadDim), 0.0f);
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wr = W.wk.data() + static_cast<size_t>(r) * kHidden;
            for (int k = 0; k < kHidden; ++k)
                acc = fmaf(__half2float(v[k]), __half2float(wr[k]), acc);
            kraw[static_cast<size_t>(r)] = acc;
        }
        float sq = 0.0f;
        for (int r = 0; r < kHeadDim; ++r)
            sq = fmaf(kraw[static_cast<size_t>(r)],
                      kraw[static_cast<size_t>(r)], sq);
        const float rms = sqrtf(sq / static_cast<float>(kHeadDim) +
                                lsxcommon::glm::kGlmRmsEps);
        __half *kd = R.key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            const float val =
                (kraw[static_cast<size_t>(r)] / rms) *
                    __half2float(W.k_norm[static_cast<size_t>(r)]) +
                W.k_bias[static_cast<size_t>(r)];
            kd[r] = FloatToHalf(val);
        }

        // q_key[r] = dot(v[0:qLora], wq_b[winner_head][r]) — FP16 store.
        __half *qd = R.q_key.data() + static_cast<size_t>(s) * kHeadDim;
        for (int r = 0; r < kHeadDim; ++r) {
            float acc = 0.0f;
            const __half *wqr = W.wq_b.data() +
                (static_cast<size_t>(best) * kHeadDim + r) * kQLora;
            for (int k = 0; k < kQLora; ++k)
                acc = fmaf(__half2float(v[k]), __half2float(wqr[k]), acc);
            qd[r] = FloatToHalf(acc);
        }
    }
}

// Exact mirror of the device attention kernel (per-token deterministic order).
std::vector<__half> RefAttention(RefIdx &R, const std::vector<__half> &x,
                                 int seq)
{
    std::vector<__half> out(static_cast<size_t>(seq) * kHidden);
    if (R.n <= 0) {
        out = x;
        return out;
    }
    const int base = R.n - seq;
    for (int t = 0; t < seq; ++t) {
        const __half *qq = R.q_key.data() +
            static_cast<size_t>(base + t) * kHeadDim;
        float sims[kMaxSlots];
        for (int s = 0; s < R.n; ++s) {
            const __half *ks =
                R.key.data() + static_cast<size_t>(s) * kHeadDim;
            float acc = 0.0f;
            for (int d = 0; d < kHeadDim; ++d)
                acc = fmaf(__half2float(qq[d]), __half2float(ks[d]), acc);
            sims[s] = acc * kAttnScale;
        }
        float m = -1e30f;
        for (int s = 0; s < R.n; ++s) m = fmaxf(m, sims[s]);
        float p[kMaxSlots];
        float sum = 0.0f;
        for (int s = 0; s < R.n; ++s) {
            p[s] = expf(sims[s] - m);
            sum += p[s];
        }
        const float inv = 1.0f / sum;
        for (int s = 0; s < R.n; ++s) p[s] *= inv;
        for (int h = 0; h < kHidden; ++h) {
            float acc = __half2float(x[static_cast<size_t>(t) * kHidden + h]);
            for (int s = 0; s < R.n; ++s)
                acc = fmaf(p[s],
                           __half2float(R.kv[static_cast<size_t>(s) * kHidden +
                                             h]),
                           acc);
            out[static_cast<size_t>(t) * kHidden + h] = FloatToHalf(acc);
        }
    }
    return out;
}

// Forward-mirror of one RunGlmIndexShareAttention call.
std::vector<__half> RefForward(RefIdx &R, const RefWeights &W,
                               int layer_idx, int turn,
                               const std::vector<__half> &x)
{
    const int seq = static_cast<int>(x.size()) / kHidden;
    if (turn != R.turn_id) {
        R.kv.insert(R.kv.end(), x.begin(), x.end());
        R.n += seq;
        R.turn_id = turn;
    }
    if (layer_idx % 4 == 0) RefRefresh(R, W);
    return RefAttention(R, x, seq);
}

} // namespace

TEST_CASE("GLM IndexShare attention: mod-4 refresh vs cached reuse and "
          "multi-turn FP16 convergence", "[glm][indexshare][attention]")
{
    static_assert(lsxcommon::glm::kGlmIndexHeadDim == 128,
                  "IndexShare locks the indexer head dim to 128");
    static_assert(lsxcommon::glm::kGlmIndexNHeads == 32,
                  "IndexShare locks the indexer head count to 32");
    static_assert(lsxcommon::glm::kGlmQLoraRank == 2048,
                  "IndexShare locks the wq_b input rank to 2048");
    static_assert(lsxcommon::glm::kGlmIndexMaxSlots >= 16,
                  "IndexShare fixture needs at least 16 slots");

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    // ── Build deterministic FP16 indexer weights ───────────────────────────
    const int64_t n_wq = static_cast<int64_t>(kHeads) * kHeadDim * kQLora;
    const int64_t n_wk = static_cast<int64_t>(kHeadDim) * kHidden;
    const int64_t n_wp = static_cast<int64_t>(kHeads) * kHidden;
    RefWeights W;
    W.wq_b.resize(static_cast<size_t>(n_wq));
    W.wk.resize(static_cast<size_t>(n_wk));
    W.wproj.resize(static_cast<size_t>(n_wp));
    W.k_norm.resize(static_cast<size_t>(kHeadDim));
    W.k_bias.resize(static_cast<size_t>(kHeadDim));
    for (int64_t i = 0; i < n_wq; ++i)
        W.wq_b[static_cast<size_t>(i)] = HashHalf(i, 7u);
    for (int64_t i = 0; i < n_wk; ++i)
        W.wk[static_cast<size_t>(i)] = HashHalf(i, 11u);
    for (int64_t i = 0; i < n_wp; ++i)
        W.wproj[static_cast<size_t>(i)] = HashHalf(i, 13u);
    for (int i = 0; i < kHeadDim; ++i)
        W.k_norm[static_cast<size_t>(i)] =
            __float2half(1.0f + 0.25f * static_cast<float>(i % 3));
    for (int i = 0; i < kHeadDim; ++i)
        W.k_bias[static_cast<size_t>(i)] = HashFloat(i, 17u);

    // ── Upload to device (indexer layer kit for the mod-4 group head) ──────
    GlmLayerWeights lw;
    REQUIRE(cudaMalloc(&lw.d_idx_wq_b,
                       static_cast<size_t>(n_wq) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&lw.d_idx_wk,
                       static_cast<size_t>(n_wk) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&lw.d_idx_weights_proj,
                       static_cast<size_t>(n_wp) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&lw.d_idx_k_norm,
                       static_cast<size_t>(kHeadDim) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(&lw.d_idx_k_norm_bias,
                       static_cast<size_t>(kHeadDim) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_wq_b, W.wq_b.data(),
                       static_cast<size_t>(n_wq) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_wk, W.wk.data(),
                       static_cast<size_t>(n_wk) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_weights_proj, W.wproj.data(),
                       static_cast<size_t>(n_wp) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_k_norm, W.k_norm.data(),
                       static_cast<size_t>(kHeadDim) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_k_norm_bias, W.k_bias.data(),
                       static_cast<size_t>(kHeadDim) * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // ── Persistent device state (the pre-allocated indexer stage) ──────────
    GlmIndexShareState *d_state = nullptr;
    REQUIRE(cudaMalloc(&d_state, sizeof(GlmIndexShareState)) == cudaSuccess);
    REQUIRE(lsxcommon::glm::GlmIndexShareReset(d_state, st) == cudaSuccess);

    // ── Token hidden blocks: two per turn, deterministic FP16 lattice ──────
    auto make_x = [&](int t0, unsigned seed) {
        std::vector<__half> x(static_cast<size_t>(kSeq) * kHidden);
        for (int t = 0; t < kSeq; ++t)
            for (int k = 0; k < kHidden; ++k)
                x[static_cast<size_t>(t) * kHidden + k] =
                    HashHalf(static_cast<int64_t>((t0 + t) * 97 + k), seed);
        return x;
    };
    const std::vector<__half> x0  = make_x(0, 3u);   // turn 0 tokens A,B
    const std::vector<__half> x0b = make_x(2, 4u);   // evolved A,B (reuse)
    const std::vector<__half> x1  = make_x(4, 5u);   // turn 1 tokens C,D

    // ── Output buffers with sentinel-guarded tails ─────────────────────────
    constexpr size_t kGuard = 64;
    const size_t out_n = static_cast<size_t>(kSeq) * kHidden;
    std::vector<__half> hout(out_n + kGuard, __float2half(-1.0f));
    __half *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, hout.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_out, hout.data(), hout.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    auto run = [&](int layer_idx, int turn, const std::vector<__half> &x) {
        __half *d_x = nullptr;
        REQUIRE(cudaMalloc(&d_x, x.size() * sizeof(__half)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(lsxcommon::glm::RunGlmIndexShareAttention(
                    st, layer_idx, turn, lw, kSeq, d_x, d_out, d_state) ==
                cudaSuccess);
        cudaFree(d_x);
    };
    auto fetch = [&]() {
        std::fill(hout.begin(), hout.end(), __float2half(-1.0f));
        REQUIRE(cudaMemcpy(hout.data(), d_out, hout.size() * sizeof(__half),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
        return std::vector<__half>(hout.begin(), hout.begin() +
                                   static_cast<ptrdiff_t>(out_n));
    };
    auto check_tail = [&]() {
        for (size_t p = out_n; p < out_n + kGuard; ++p)
            CHECK(__half2float(hout[p]) == -1.0f);
    };
    auto check_match = [&](const std::vector<__half> &got,
                           const std::vector<__half> &want, const char *tag) {
        REQUIRE(got.size() == out_n);
        REQUIRE(want.size() == out_n);
        for (size_t i = 0; i < out_n; ++i) {
            const float g = __half2float(got[i]);
            const float w = __half2float(want[i]);
            CHECK(std::isfinite(g));
            CHECK(std::isfinite(w));
            // FP16 one-ulp boundary flips (~5e-4 relative) from device/host
            // expf ulp divergence are the only tolerated difference; a layout
            // or projection fault shifts values by far more.
            CHECK(std::fabs(g - w) <= 1e-3f * (1.0f + std::fabs(w)));
        }
    };
    auto max_abs = [&](const std::vector<__half> &v) {
        float mx = 0.0f;
        for (__half h : v) mx = std::max(mx, std::fabs(__half2float(h)));
        return mx;
    };

    // ── Host reference, driven through the identical call sequence ─────────
    RefIdx R;
    std::vector<__half> ref4, ref6, ref8;

    // ── Turn 0, layer 4 (4 % 4 == 0): refresh the indexer mask arrays ──────
    REQUIRE(lsxcommon::glm::GlmIndexShareReset(d_state, st) == cudaSuccess);
    ref4 = RefForward(R, W, 4, 0, x0);
    run(4, 0, x0);
    std::vector<__half> out4 = fetch();
    check_match(out4, ref4, "refresh layer-4 output");
    check_tail();
    // The refresh must produce real, non-degenerate attention math.
    const float attn4 = max_abs(ref4) > 0.0f
        ? [&] {
            float mx = 0.0f;
            for (size_t i = 0; i < out_n; ++i)
                mx = std::max(mx,
                              std::fabs(__half2float(ref4[i]) -
                                        __half2float(x0[i])));
            return mx;
          }()
        : 0.0f;
    CHECK(attn4 > 0.005f);

    // ── Turn 0, layers 5-7 (mod-4 != 0): reuse the cached slots ────────────
    // Corrupt every indexer weight on-device: a reuse path that touchingly
    // re-derives masks would produce garbage; the true reuse path is
    // bit-identical to layer 4.
    REQUIRE(cudaMemset(lw.d_idx_wq_b, 0xA5,
                       static_cast<size_t>(n_wq) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(lw.d_idx_wk, 0xA5,
                       static_cast<size_t>(n_wk) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(lw.d_idx_weights_proj, 0xA5,
                       static_cast<size_t>(n_wp) * sizeof(__half)) ==
            cudaSuccess);

    run(5, 0, x0);
    std::vector<__half> out5 = fetch();
    REQUIRE(out5.size() == out4.size());
    CHECK(std::memcmp(out5.data(), out4.data(),
                      out4.size() * sizeof(__half)) == 0);
    // Reference reuse output must equal the cached-mask result too.
    const std::vector<__half> ref5 = RefForward(R, W, 5, 0, x0);
    check_match(out5, ref5, "cached reuse layer-5 output");

    // Evolved hidden (same turn, still layer 7 reuse): attention part stays
    // cached, only the residual changes.
    ref6 = RefForward(R, W, 6, 0, x0b);
    run(6, 0, x0b);
    std::vector<__half> out6 = fetch();
    check_match(out6, ref6, "cached reuse layer-6 output");
    run(7, 0, x0b);
    std::vector<__half> out7 = fetch();
    const std::vector<__half> ref7 = RefForward(R, W, 7, 0, x0b);
    check_match(out7, ref7, "cached reuse layer-7 output");
    REQUIRE(out7.size() == out6.size());
    CHECK(std::memcmp(out7.data(), out6.data(),
                      out6.size() * sizeof(__half)) == 0);

    // ── Turn 1, layer 8 (8 % 4 == 0): multi-turn refresh over 4 KV slots ──
    // Restore the pristine indexer weights before the next refresh point.
    REQUIRE(cudaMemcpy(lw.d_idx_wq_b, W.wq_b.data(),
                       static_cast<size_t>(n_wq) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_wk, W.wk.data(),
                       static_cast<size_t>(n_wk) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(lw.d_idx_weights_proj, W.wproj.data(),
                       static_cast<size_t>(n_wp) * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    ref8 = RefForward(R, W, 8, 1, x1);
    run(8, 1, x1);
    std::vector<__half> out8 = fetch();
    check_match(out8, ref8, "multi-turn refresh layer-8 output");

    // The two-turn KV must actually change the result vs turn-0.
    bool multiturn_diff = false;
    for (size_t i = 0; i < out_n; ++i)
        if (std::fabs(__half2float(ref8[i]) - __half2float(ref4[i])) >
            1e-3f)
            multiturn_diff = true;
    CHECK(multiturn_diff);
    float attn8 = 0.0f;
    for (size_t i = 0; i < out_n; ++i)
        attn8 = std::max(attn8, std::fabs(__half2float(ref8[i]) -
                                          __half2float(x1[i])));
    CHECK(attn8 > 0.005f);

    // Turn 1 reuse layer 9 with the indexer weights corrupted again must be
    // bit-identical to layer 8.
    REQUIRE(cudaMemset(lw.d_idx_wq_b, 0xA5,
                       static_cast<size_t>(n_wq) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(lw.d_idx_wk, 0xA5,
                       static_cast<size_t>(n_wk) * sizeof(__half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(lw.d_idx_weights_proj, 0xA5,
                       static_cast<size_t>(n_wp) * sizeof(__half)) ==
            cudaSuccess);
    run(9, 1, x1);
    std::vector<__half> out9 = fetch();
    REQUIRE(out9.size() == out8.size());
    CHECK(std::memcmp(out9.data(), out8.data(),
                      out8.size() * sizeof(__half)) == 0);

    // Guard tails never written through the whole turn cycle.
    check_tail();

    // ── Device state sanity: 4 registered slots, valid head selection ──────
    int h_n = 0;
    REQUIRE(cudaMemcpy(&h_n, &d_state->n_used, sizeof(int),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    CHECK(h_n == 4);
    std::vector<int> h_slots(static_cast<size_t>(kMaxSlots));
    REQUIRE(cudaMemcpy(h_slots.data(), d_state->slot,
                       h_slots.size() * sizeof(int),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    for (int s = 0; s < h_n; ++s) {
        CHECK(h_slots[static_cast<size_t>(s)] >= 0);
        CHECK(h_slots[static_cast<size_t>(s)] < kHeads);
    }

    cudaFree(d_out);
    cudaFree(d_state);
    lw.Free();
    cudaStreamDestroy(st);
}