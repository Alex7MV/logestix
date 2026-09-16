// TDD — Nemotron-3.5 host reference math (Model Ingestion Protocol step 5/8).
// Pure host, header-only: pins the Mamba-2 / gated-RMSNorm / router / relu2
// formulas that the CUDA kernels are validated against.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nemotron_moe_device_math.h"

#include <cmath>
#include <vector>

using namespace lsxcommon::nemotron_moe;

TEST_CASE("relu2 squares the positive part", "[nemotron][math]")
{
    CHECK(ref::Relu2(-2.0f) == 0.0f);
    CHECK(ref::Relu2(0.0f) == 0.0f);
    CHECK(ref::Relu2(3.0f) == 9.0f);
}

TEST_CASE("gated rmsnorm applies silu(gate) then per-group rmsnorm",
          "[nemotron][math]")
{
    const std::vector<float> x = {3.0f, 4.0f};
    const std::vector<float> w = {1.0f, 1.0f};
    std::vector<float> out(2);

    // silu(0) == 0 -> the whole row collapses to zero.
    const std::vector<float> g0 = {0.0f, 0.0f};
    ref::GatedRmsNorm(x.data(), g0.data(), w.data(), 2, 2, 1e-5f, out.data());
    CHECK(out[0] == 0.0f);
    CHECK(out[1] == 0.0f);

    // A constant gate cancels in the RMS: out = x / rms(x) = {0.8485, 1.1314}.
    const std::vector<float> g1 = {100.0f, 100.0f};
    ref::GatedRmsNorm(x.data(), g1.data(), w.data(), 2, 2, 0.0f, out.data());
    CHECK(out[0] == Catch::Approx(0.84853f).margin(1e-4f));
    CHECK(out[1] == Catch::Approx(1.13137f).margin(1e-4f));

    // Weight scales the normalized value.
    const std::vector<float> w2 = {0.0f, 2.0f};
    ref::GatedRmsNorm(x.data(), g1.data(), w2.data(), 2, 2, 0.0f, out.data());
    CHECK(out[0] == Catch::Approx(0.0f));
    CHECK(out[1] == Catch::Approx(2.26274f).margin(1e-4f));

    // Non-uniform gate zeroes the gated element: x={3,4}, gate={100,0}.
    const std::vector<float> g2 = {100.0f, 0.0f};
    ref::GatedRmsNorm(x.data(), g2.data(), w.data(), 2, 2, 0.0f, out.data());
    // rms = sqrt((300^2+0)/2) = 212.13; out = 300/212.13 = 1.4142, 0.
    CHECK(out[0] == Catch::Approx(1.41421f).margin(1e-4f));
    CHECK(out[1] == Catch::Approx(0.0f));
}

TEST_CASE("selective state update matches the recurrent Mamba-2 form",
          "[nemotron][math]")
{
    // 1 head, head_dim 1, 1 group, state 2. A_log {0, ln2} -> A {-1, -2}.
    const float state_in[2] = {0.0f, 0.0f};
    const float xs[1] = {2.0f};
    const float Bv[2] = {1.0f, 1.0f};
    const float Cv[2] = {1.0f, 1.0f};
    const float Dv[1] = {0.0f};
    const float dt[1] = {0.0f};
    const float dt_bias[1] = {0.0f};
    const float A_log[2] = {0.0f, std::log(2.0f)};
    float state_out[2] = {0.0f, 0.0f};
    float out[1] = {0.0f};

    ref::SelectiveStateUpdate(state_in, xs, Bv, Cv, Dv, dt, dt_bias, A_log,
                              /*heads=*/1, /*head_dim=*/1, /*groups=*/1,
                              /*state_size=*/2, state_out, out);

    const float d = std::log(2.0f); // softplus(0)
    CHECK(state_out[0] == Catch::Approx(d * 2.0f).margin(1e-5f)); // exp(-d)*0 term
    CHECK(state_out[1] == Catch::Approx(d * 2.0f).margin(1e-5f));
    CHECK(out[0] == Catch::Approx(state_out[0] + state_out[1]).margin(1e-5f));
}

TEST_CASE("router: sigmoid top-k, bias selects, normalized then scaled",
          "[nemotron][math]")
{
    const std::vector<float> logits = {0.0f, 1.0f, 2.0f, -1.0f};
    const std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> w(2);
    std::vector<int> idx(2);
    ref::RouterTopK(logits.data(), bias.data(), 4, 2, 2.5f, w.data(), idx.data());
    // Top-2 by sigmoid are experts 2 then 1; weights sum to routed_scale.
    CHECK(idx[0] == 2);
    CHECK(idx[1] == 1);
    CHECK(w[0] + w[1] == Catch::Approx(2.5f).margin(1e-5f));
    CHECK(w[0] > w[1]); // 2.0 logit dominates
}

TEST_CASE("router e_score_correction_bias changes only selection",
          "[nemotron][math]")
{
    const std::vector<float> logits = {0.0f, 1.0f, 2.0f, -1.0f};
    const std::vector<float> bias = {0.0f, 0.0f, 0.0f, 10.0f}; // push expert 3 in
    std::vector<float> w(1);
    std::vector<int> idx(1);
    ref::RouterTopK(logits.data(), bias.data(), 4, 1, 2.5f, w.data(), idx.data());
    CHECK(idx[0] == 3);                        // selected by bias
    CHECK(w[0] == Catch::Approx(2.5f));        // weight from the raw score
}

TEST_CASE("expert relu2 mlp", "[nemotron][math]")
{
    // H=1, I=1: up=[2] -> relu2(2*1)=4 -> *down 3 -> 12.
    const float x[1] = {1.0f};
    const float up[1] = {2.0f};
    const float down[1] = {3.0f};
    float out[1] = {0.0f};
    ref::ExpertMlpRelu2(x, up, down, 1, 1, out);
    CHECK(out[0] == Catch::Approx(12.0f));

    // Negative pre-activation -> relu2 == 0.
    const float xneg[1] = {-1.0f};
    ref::ExpertMlpRelu2(xneg, up, down, 1, 1, out);
    CHECK(out[0] == 0.0f);
}
