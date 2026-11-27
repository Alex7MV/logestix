// TDD Cycle 3 - GLM-5.2 MoE router (layers 3-77): entering Layer 3 must
// populate the 8 active expert slots of every token's routing record without
// index out-of-bounds corruption of the output grid.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>
#include <vector>

#include "lsxcommon/glm_moe_inference.h"
#include "lsxcommon/glm_moe_load.h"
#include "lsxcommon/path_registry.h"

using lsxcommon::glm::GlmPinnedWeights;
using lsxcommon::glm::GlmRoutedSlot;

namespace {

constexpr int kHidden  = lsxcommon::glm::kGlmHidden;    // 6144
constexpr int kNRouted = lsxcommon::glm::kGlmNumRouted; // 256
constexpr int kTopK    = lsxcommon::glm::kGlmMoETopK;   // 8
constexpr int kSeq     = 4;

// Per-expert gate row-sum target r_e. r_0 = r_1 = 400 (deliberate top tie that
// must resolve via smallest-index tie-break), then a strict ramp down through
// r_40 = 10 and a floor of 0 for e >= 41 (a 215-wide exact tie group).
// Gate row e is a constant FP16(r_e/6144) repeated across all k = 6144
// columns, so with an RMS-normed constant hidden input the router logit for
// expert e is proportional (through a positive monotone scale) to r_e, and a
// sign flip per token negates the whole score profile.
float RowSumTarget(int e)
{
    if (e == 0 || e == 1) return 400.0f;
    if (e <= 40) return 400.0f - 10.0f * static_cast<float>(e - 1);
    return 0.0f;
}

// Deterministic host reference: top-k under the same total order the kernel
// implements (larger key wins; equal key resolves toward the smallest expert
// index), applied to the sign-scaled row targets.
std::vector<int> RefTop8(float sign)
{
    std::vector<std::pair<float, int>> kv;
    kv.reserve(static_cast<size_t>(kNRouted));
    for (int e = 0; e < kNRouted; ++e)
        kv.emplace_back(sign * RowSumTarget(e), e);
    std::sort(kv.begin(), kv.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    std::vector<int> out;
    out.reserve(static_cast<size_t>(kTopK));
    for (int i = 0; i < kTopK; ++i)
        out.push_back(kv[static_cast<size_t>(i)].second);
    return out;
}

} // namespace

TEST_CASE("GLM MoE Layer 3 router: top_k=8 slots per token, no OOB corruption",
          "[glm][moe][router]")
{
    static_assert(sizeof(GlmRoutedSlot) == 8u,
                  "GlmRoutedSlot must pack into an 8-byte boundary record");
    static_assert(alignof(GlmRoutedSlot) == 8u,
                  "GlmRoutedSlot must honour 8-byte alignment");

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    // ── Enter Layer 3: build the MoE layer kit (router gate + post-attn norm).
    GlmPinnedWeights pw;
    pw.layers.resize(4);
    auto &l3 = pw.layers[3];

    std::vector<__half> hgate(static_cast<size_t>(kNRouted) * kHidden);
    for (int e = 0; e < kNRouted; ++e) {
        __half v = __float2half(RowSumTarget(e) / static_cast<float>(kHidden));
        for (int k = 0; k < kHidden; ++k)
            hgate[static_cast<size_t>(e) * kHidden + k] = v;
    }
    REQUIRE(cudaMalloc(&l3.d_router_gate,
                       hgate.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(l3.d_router_gate, hgate.data(),
                       hgate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    std::vector<__half> hnorm(static_cast<size_t>(kHidden), __float2half(1.0f));
    REQUIRE(cudaMalloc(&l3.d_post_attn_norm,
                       hnorm.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(l3.d_post_attn_norm, hnorm.data(),
                       hnorm.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Raw activations: constant +1 for tokens 0..1, -1 for tokens 2..3.
    std::vector<__half> hx(static_cast<size_t>(kSeq) * kHidden);
    for (int t = 0; t < kSeq; ++t)
        for (int k = 0; k < kHidden; ++k)
            hx[static_cast<size_t>(t) * kHidden + k] =
                __float2half((t < 2) ? 1.0f : -1.0f);
    __half *d_x = nullptr, *d_normed = nullptr;
    float *d_logits = nullptr;
    REQUIRE(cudaMalloc(&d_x, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_normed, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits,
                       static_cast<size_t>(kSeq) * kNRouted *
                           sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Routing grid with a sentinel-guarded tail: keep the guard 8-byte-aligned
    // too, so any overrun of the [seq, 8] grid is caught verbatim.
    constexpr size_t kGuard = 16;
    const size_t grid_slots = static_cast<size_t>(kSeq) * kTopK;
    std::vector<uint64_t> hbuf(grid_slots + kGuard, 0xdeadbeefdeadbeefULL);
    uint64_t *d_route = nullptr;
    REQUIRE(cudaMalloc(&d_route, hbuf.size() * sizeof(uint64_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_route, hbuf.data(), hbuf.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // ── Layer-3 routing, single explicit async stream, no device sync ──────
    REQUIRE(lsxcommon::glm::RunGlmMoERouter(
                st, l3, kSeq, d_x, nullptr, d_normed, d_logits,
                reinterpret_cast<GlmRoutedSlot *>(d_route)) == cudaSuccess);
    REQUIRE(cudaMemcpy(hbuf.data(), d_route, hbuf.size() * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    const GlmRoutedSlot *slots =
        reinterpret_cast<const GlmRoutedSlot *>(hbuf.data());

    // Every token must carry exactly its 8 expected expert slots.
    for (int t = 0; t < kSeq; ++t) {
        const float sign = (t < 2) ? 1.0f : -1.0f;
        const std::vector<int> expected = RefTop8(sign);
        for (int r = 0; r < kTopK; ++r) {
            const GlmRoutedSlot &s = slots[static_cast<size_t>(t) * kTopK + r];
            CHECK(s.eid >= 0);
            CHECK(s.eid < kNRouted);
            CHECK(std::isfinite(s.coeff));
            CHECK(s.coeff > 0.0f);
            CHECK(s.coeff <= 1.0f);
            CHECK(s.eid == expected[static_cast<size_t>(r)]);
        }
        // Slots are score-sorted: coefficients non-increasing down the record.
        for (int r = 0; r < kTopK - 1; ++r) {
            const GlmRoutedSlot &a = slots[static_cast<size_t>(t) * kTopK + r];
            const GlmRoutedSlot &b = slots[static_cast<size_t>(t) * kTopK + r + 1];
            CHECK(a.coeff >= b.coeff - 1e-5f);
        }
    }

    // Negative-sign tokens route through the exact 215-wide zero-tie group:
    // the 8 winners must carry identical softmax coefficients.
    for (int t = 2; t < kSeq; ++t) {
        const float c0 = slots[static_cast<size_t>(t) * kTopK].coeff;
        for (int r = 1; r < kTopK; ++r) {
            const float cr = slots[static_cast<size_t>(t) * kTopK + r].coeff;
            CHECK(std::fabs(cr - c0) < 1e-6f);
        }
    }

    // The kernel wrote exactly [seq,8] slots: the guard tail is untouched.
    for (size_t p = grid_slots; p < grid_slots + kGuard; ++p)
        CHECK(hbuf[p] == 0xdeadbeefdeadbeefULL);

    pw.Free();
    cudaFree(d_x);
    cudaFree(d_normed);
    cudaFree(d_logits);
    cudaFree(d_route);
    cudaStreamDestroy(st);
}

// CPU reference for the real-model case: RMSNorm a constant ±1 input, project
// through the read-back FP16 gate rows, softmax, then the same total-order
// top-8. The RMSNorm output is forced through FP16 rounding (the device GEMM
// consumes the FP16 d_normed buffer), accumulation stays in double.
std::vector<int> RefTop8Real(const std::vector<__half> &gate,
                             const std::vector<__half> &norm_w, float sign,
                             int hidden, int n_routed)
{
    const float rinv = static_cast<float>(
        1.0 / std::sqrt(1.0 + static_cast<double>(lsxcommon::glm::kGlmRmsEps)));
    std::vector<double> logit(static_cast<size_t>(n_routed), 0.0);
    for (int e = 0; e < n_routed; ++e) {
        double acc = 0.0;
        for (int k = 0; k < hidden; ++k) {
            const float x = sign * static_cast<float>(rinv);
            const __half normed = __float2half(x * __half2float(norm_w[k]));
            acc += static_cast<double>(__half2float(gate[static_cast<size_t>(e) * hidden + k])) *
                   static_cast<double>(__half2float(normed));
        }
        logit[static_cast<size_t>(e)] = acc;
    }
    double mx = -1e300;
    for (double v : logit) mx = std::max(mx, v);
    double psum = 0.0;
    std::vector<double> p(static_cast<size_t>(n_routed));
    for (int e = 0; e < n_routed; ++e) {
        p[static_cast<size_t>(e)] = std::exp(logit[static_cast<size_t>(e)] - mx);
        psum += p[static_cast<size_t>(e)];
    }
    std::vector<std::pair<double, int>> kv;
    kv.reserve(static_cast<size_t>(n_routed));
    for (int e = 0; e < n_routed; ++e)
        kv.emplace_back(p[static_cast<size_t>(e)] / psum, e);
    std::sort(kv.begin(), kv.end(), [](const auto &a, const auto &b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    std::vector<int> out;
    out.reserve(static_cast<size_t>(kTopK));
    for (int i = 0; i < kTopK; ++i)
        out.push_back(kv[static_cast<size_t>(i)].second);
    return out;
}

TEST_CASE("GLM MoE Layer 3 router on real model weights: smoke + determinism",
          "[glm][moe][router][real-model]")
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    std::filesystem::path model =
        env ? std::filesystem::path(env)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) / "GLM-5.2";
    if (!std::filesystem::exists(model / "pinned"))
        SKIP("GLM real model directory not present");

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    GlmPinnedWeights pw;
    std::vector<int> layer3{3};
    REQUIRE(lsxcommon::glm::LoadGlmPinnedWeights(model.string(), pw, st,
                                                 &layer3) == cudaSuccess);
    REQUIRE(pw.hidden == kHidden);
    REQUIRE(pw.layers.size() == 79);
    auto &l3 = pw.layers[3];
    REQUIRE(l3.d_router_gate != nullptr);
    REQUIRE(l3.d_post_attn_norm != nullptr);

    // Two identical constant +1 tokens: the routing records must be bit-identical.
    const int seq = 2;
    std::vector<__half> hx(static_cast<size_t>(seq) * kHidden, __float2half(1.0f));
    __half *d_x = nullptr, *d_normed = nullptr;
    float *d_logits = nullptr;
    REQUIRE(cudaMalloc(&d_x, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_normed, hx.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits, static_cast<size_t>(seq) * kNRouted * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    constexpr size_t kGuard = 16;
    const size_t grid_slots = static_cast<size_t>(seq) * kTopK;
    std::vector<uint64_t> hbuf(grid_slots + kGuard, 0xdeadbeefdeadbeefULL);
    uint64_t *d_route = nullptr;
    REQUIRE(cudaMalloc(&d_route, hbuf.size() * sizeof(uint64_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_route, hbuf.data(), hbuf.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // Read the real FP16 router weights back for the double-precision reference.
    std::vector<__half> hgate(static_cast<size_t>(kNRouted) * kHidden);
    std::vector<__half> hnorm(static_cast<size_t>(kHidden));
    REQUIRE(cudaMemcpyAsync(hgate.data(), l3.d_router_gate,
                            hgate.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, st) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(hnorm.data(), l3.d_post_attn_norm,
                            hnorm.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost, st) == cudaSuccess);

    REQUIRE(lsxcommon::glm::RunGlmMoERouter(
                st, l3, seq, d_x, nullptr, d_normed, d_logits,
                reinterpret_cast<GlmRoutedSlot *>(d_route)) == cudaSuccess);
    REQUIRE(cudaMemcpy(hbuf.data(), d_route, hbuf.size() * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    const GlmRoutedSlot *slots =
        reinterpret_cast<const GlmRoutedSlot *>(hbuf.data());

    // Identical inputs -> the two tokens must carry identical routing records.
    for (int r = 0; r < kTopK; ++r) {
        const GlmRoutedSlot &a = slots[static_cast<size_t>(r)];
        const GlmRoutedSlot &b = slots[static_cast<size_t>(kTopK) + r];
        CHECK(a.eid == b.eid);
        CHECK(std::memcmp(&a.coeff, &b.coeff, sizeof(float)) == 0);
    }

    // Validity: distinct, in-range expert slots with finite sorted coefficients.
    const std::vector<int> ref = RefTop8Real(hgate, hnorm, 1.0f, kHidden, kNRouted);
    bool ref_top1_ok = false;
    int  min_overlap = kTopK; // worst per-token overlap with the CPU reference
    for (int t = 0; t < seq; ++t) {
        bool seen[256] = {false};
        int  overlap = 0;
        for (int r = 0; r < kTopK; ++r) {
            const GlmRoutedSlot &s = slots[static_cast<size_t>(t) * kTopK + r];
            CHECK(s.eid >= 0);
            CHECK(s.eid < kNRouted);
            REQUIRE(!seen[s.eid]);   // distinct within one token's record
            seen[s.eid] = true;
            CHECK(std::isfinite(s.coeff));
            CHECK(s.coeff > 0.0f);
            CHECK(s.coeff <= 1.0f);
            for (int e : ref) overlap += (s.eid == e) ? 1 : 0;
        }
        min_overlap = std::min(min_overlap, overlap);
        if (slots[static_cast<size_t>(t) * kTopK].eid == ref[0])
            ref_top1_ok = true;
        for (int r = 0; r < kTopK - 1; ++r) {
            const GlmRoutedSlot &a = slots[static_cast<size_t>(t) * kTopK + r];
            const GlmRoutedSlot &b = slots[static_cast<size_t>(t) * kTopK + r + 1];
            CHECK(a.coeff >= b.coeff - 1e-5f);
        }
    }
    // The argmax must agree exactly; a single near-boundary expert may flip.
    CHECK(ref_top1_ok);
    CHECK(min_overlap >= 6);

    // Guard tail untouched: exactly [seq,8] slots were written.
    for (size_t p = grid_slots; p < grid_slots + kGuard; ++p)
        CHECK(hbuf[p] == 0xdeadbeefdeadbeefULL);

    pw.Free();
    cudaFree(d_x);
    cudaFree(d_normed);
    cudaFree(d_logits);
    cudaFree(d_route);
    cudaStreamDestroy(st);
}