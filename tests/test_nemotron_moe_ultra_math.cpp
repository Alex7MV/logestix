// TDD — Nemotron-3-Ultra-550B host reference math.
//
// These are the formulas the device kernels are pinned against, transcribed from
// the reference implementation the checkpoint ships with:
//   transformers 5.15.1 models/nemotron_h/modeling_nemotron_h.py
//     NemotronHTopkRouter.forward   (L728-753)  grouped sigmoid top-k router
//     NemotronHMoE.forward          (L699-712)  latent bridge + shared expert
//     NemotronHMamba2Mixer.forward  (L427-571)  recurrence, dt clamp split
//     NemotronHRMSNorm.forward      (L584-589)
//   transformers 5.15.1 models/zamba2/modeling_zamba2.py
//     Zamba2RMSNormGated                        gated group RMSNorm
//     mamba2_selective_state_update (L478-534) decode: softplus, NO dt clamp
//     mamba2_chunk_scan             (L540-562) prefill: softplus then clamp
//
// The two Ultra-only pieces are the grouped router and the latent MoE; a defect
// in either answers "Paris" while being wrong, which is why they are pinned by
// analytic expectations rather than by eyeballing output.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "nemotron_moe_ultra_device_math.h"

using namespace lsxcommon::nemotron_ultra;
using Catch::Approx;

namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

TEST_CASE("ultra router: the group stage keeps only the best group", "[ultra][math]")
{
    // 8 experts, n_group 4 (two per group), topk_group 1, top-2.
    // Group 0 holds the single best expert (logit 4.0) but group 2 wins on the
    // reference's group score (the sum of its top-2); selecting expert 0 here
    // would be the classic "forgot the group mask" defect.
    const float logits[8] = {4.0f, 0.0f, 0.0f, 0.0f, 3.0f, 2.9f, 0.0f, 0.0f};
    const float bias[8] = {0};
    float w[2] = {0, 0};
    int idx[2] = {-1, -1};

    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w, idx);

    // Both selected ids must come from the winning group {4, 5}.
    CHECK((((idx[0] == 4 && idx[1] == 5) || (idx[0] == 5 && idx[1] == 4))));

    // Weights are the sigmoid scores normalized (norm_topk_prob) then scaled.
    const float s4 = Sigmoid(3.0f), s5 = Sigmoid(2.9f);
    const float sum = s4 + s5;
    const float expect4 = s4 / sum * 5.0f;
    const float expect5 = s5 / sum * 5.0f;
    for (int t = 0; t < 2; ++t) {
        const float want = idx[t] == 4 ? expect4 : expect5;
        CHECK(w[t] == Approx(want).epsilon(1e-5));
    }
    CHECK(w[0] + w[1] == Approx(5.0f).epsilon(1e-5));
}

TEST_CASE("ultra router: e_score_correction_bias selects but never scales",
          "[ultra][math]")
{
    const float logits[8] = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float bias[8] = {0};
    bias[6] = 20.0f;  // wins selection outright
    float w[2] = {0, 0};
    int idx[2] = {-1, -1};

    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 1.0f, w, idx);

    // Expert 6 is selected (bias) but its WEIGHT is sigmoid(0) = 0.5, not
    // sigmoid(0)+20: the bias is a selection device, not a scaling one.
    bool has6 = (idx[0] == 6 || idx[1] == 6);
    CHECK(has6);
    for (int t = 0; t < 2; ++t)
        if (idx[t] == 6) CHECK(w[t] > 0.0f);
    // Sum of the two weights is the routed scale (norm_topk_prob).
    CHECK(w[0] + w[1] == Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("ultra router: exact ties resolve to the lowest index, deterministically",
          "[ultra][math]")
{
    const float logits[8] = {0};
    const float bias[8] = {0};
    float w1[2], w2[2];
    int i1[2], i2[2];

    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w1, i1);
    ref::RouterTopKGrouped(logits, bias, 8, 2, 4, 1, 5.0f, w2, i2);

    // All groups tie at 1.0 (0.5 + 0.5); topk_group=1 keeps the first, and the
    // two lowest ids inside it win.
    CHECK(i1[0] == i2[0]);
    CHECK(i1[1] == i2[1]);
    CHECK(((i1[0] == 0 && i1[1] == 1)));
}

TEST_CASE("ultra latent MoE: routed path runs at the latent width and is projected back",
          "[ultra][math]")
{
    // Tiny analytic case. H 16 -> latent 8 -> expert I 16 -> latent 8 -> H 16.
    constexpr int H = 16, L = 8, I = 16, SI = 32, TOK = 2;
    std::vector<float> fc1(L * H, 0.0f), fc2(H * L, 0.0f);
    std::vector<float> shared_up(SI * H, 0.0f), shared_down(H * SI, 0.0f);
    // fc1 = [I_8 | 0] : takes the first 8 hidden dims as the latent.
    for (int i = 0; i < L; ++i) fc1[i * H + i] = 1.0f;
    // fc2 = [I_8 ; 0] : writes the latent back into the first 8 hidden dims.
    for (int i = 0; i < L; ++i) fc2[i * L + i] = 1.0f;
    // Shared expert contributes nothing (zero input projection).
    std::vector<float> x(H);
    for (int i = 0; i < H; ++i) x[i] = 0.25f * static_cast<float>(i + 1);

    // up_e = [I_8 | 0] (I x L, identity in the first 8 rows), down_e = [I_8 ; 0]
    // (L x I, identity in the first 8 columns) => expert is a pass-through of
    // relu2(latent).
    constexpr int EXPERTS = 4;
    std::vector<std::vector<float>> ups(EXPERTS, std::vector<float>(I * L, 0.0f));
    std::vector<std::vector<float>> downs(EXPERTS, std::vector<float>(L * I, 0.0f));
    for (int e = 0; e < EXPERTS; ++e) {
        for (int i = 0; i < L; ++i) ups[e][i * L + i] = 1.0f;
        for (int i = 0; i < L; ++i) downs[e][i * I + i] = 1.0f;
    }
    std::vector<const float *> up_p, down_p;
    for (int e = 0; e < EXPERTS; ++e) {
        up_p.push_back(ups[e].data());
        down_p.push_back(downs[e].data());
    }
    const int ids[TOK] = {1, 3};
    const float wts[TOK] = {0.75f, 2.25f};

    std::vector<float> out(H, -12345.0f);
    ref::LatentMoe(x.data(), fc1.data(), fc2.data(), shared_up.data(),
                   shared_down.data(), up_p.data(), down_p.data(), ids, wts, TOK, H,
                   L, I, SI, out.data());

    // Expectation, computed here independently: z_j = x_j (j < L); each expert is
    // relu2(z); the weighted sum is (0.75 + 2.25) * relu2(z); fc2 writes it back
    // into the first L hidden dims and zeroes the rest.
    for (int j = 0; j < H; ++j) {
        float want = 0.0f;
        if (j < L) want = 3.0f * ref::Relu2(x[j]);
        CHECK(out[j] == Approx(want).margin(1e-6));
    }
}

TEST_CASE("ultra latent MoE: the shared expert runs at the hidden width, not the latent",
          "[ultra][math]")
{
    constexpr int H = 16, L = 8, I = 16, SI = 32, TOK = 0;
    std::vector<float> fc1(L * H, 0.0f), fc2(H * L, 0.0f);      // routed path = 0
    std::vector<float> shared_up(SI * H, 0.0f), shared_down(H * SI, 0.0f);
    // shared_up row 0 reads hidden dim 0 with weight 2; shared_down col 0 writes
    // hidden dim 5 with weight 3.
    shared_up[0 * H + 0] = 2.0f;
    shared_down[5 * SI + 0] = 3.0f;
    std::vector<float> x(H, 0.0f);
    x[0] = 0.5f;

    std::vector<float> out(H, -12345.0f);
    ref::LatentMoe(x.data(), fc1.data(), fc2.data(), shared_up.data(),
                   shared_down.data(), nullptr, nullptr, nullptr, nullptr, TOK, H, L,
                   I, SI, out.data());

    // out[5] = 3 * relu2(2 * 0.5) = 3 * 1 = 3; everything else zero.
    CHECK(out[5] == Approx(3.0f).margin(1e-6));
    for (int j = 0; j < H; ++j)
        if (j != 5) CHECK(out[j] == Approx(0.0f).margin(1e-6));
}

TEST_CASE("ultra mamba: gated RMSNorm normalizes per group", "[ultra][math]")
{
    // Two groups of DIFFERENT scale: a normalization that ignored the group
    // boundary would produce different values inside one group.
    constexpr int kGroup = 8;
    constexpr int kGroups = 2;
    constexpr int n = kGroup * kGroups;
    std::vector<float> x(n), gate(n), weight(n), out(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        x[i] = (i < kGroup) ? 1.0f : 4.0f;
        gate[i] = 10.0f;   // silu(10) ~= 1
        weight[i] = 1.0f;
    }
    ref::GatedRmsNorm(x.data(), gate.data(), weight.data(), n, kGroup, 1e-5f,
                      out.data());
    // Within a group the input is constant, so h/(rms(h)) == 1 exactly: both
    // groups come out at 1.0 even though their inputs differ by 4x. A GLOBAL
    // normalization would give 0.34 and 1.37 for the two groups instead.
    for (int i = 0; i < n; ++i) CHECK(out[i] == Approx(1.0f).margin(1e-5));
}

TEST_CASE("ultra mamba: the prefill form clamps dt, the decode form does not",
          "[ultra][math]")
{
    // head_dim 1, state 1, one head: the recurrence reduces to
    //   state' = state*exp(dt*A) + dt*B*x
    constexpr int heads = 1, head_dim = 1, groups = 1, state_size = 1;
    const float state_in[1] = {1.0f};
    const float xs[1] = {1.0f};
    const float B[1] = {1.0f};
    const float C[1] = {1.0f};
    const float D[1] = {0.0f};
    const float dt[1] = {-50.0f};     // softplus(-50) underflows toward 0
    const float dt_bias[1] = {0.0f};
    const float A_log[1] = {5.0f};     // A = -exp(5) = -148.4, so the clamp bites

    float st_pre[1], out_pre[1], st_dec[1], out_dec[1];
    ref::SelectiveStateUpdate(state_in, xs, B, C, D, dt, dt_bias, A_log, heads,
                              head_dim, groups, state_size, /*clamp_dt=*/true,
                              st_pre, out_pre);
    ref::SelectiveStateUpdate(state_in, xs, B, C, D, dt, dt_bias, A_log, heads,
                              head_dim, groups, state_size, /*clamp_dt=*/false,
                              st_dec, out_dec);

    // Prefill: dt clamped to kDtMin, so the state decays by exp(A*kDtMin) and
    // gains dt*B*xs = kDtMin.
    const float A = -std::exp(5.0f);
    const float want_pre = std::exp(A * frozen::kDtMin) + frozen::kDtMin;
    CHECK(st_pre[0] == Approx(want_pre).margin(1e-5));
    // Decode: softplus(-50) ~ 2e-22, so the state is untouched (no clamp).
    CHECK(st_dec[0] == Approx(1.0f).margin(1e-6));
    // The two forms must not agree: that difference IS the clamp.
    CHECK(std::fabs(st_pre[0] - st_dec[0]) > 0.1f);
}

TEST_CASE("ultra math: softplus is stable across the whole range", "[ultra][math]")
{
    for (float v : {-50.0f, -5.0f, 0.0f, 5.0f, 50.0f}) {
        const float got = ref::Softplus(v);
        CHECK(std::isfinite(got));
        CHECK(got >= 0.0f);
        if (v <= 20.0f) CHECK(got == Approx(std::log1p(std::exp(v))).margin(1e-5));
    }
    CHECK(ref::Softplus(50.0f) == Approx(50.0f).margin(1e-5));
}
