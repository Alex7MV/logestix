// TDD — GigaChat4-37B-A4B host reference math (no CUDA). Pins the derived
// geometry and the variant gating math (non-zero-centred GatedNorm, sigmoid GDN
// output norm, logit softcap) that the device kernels implement.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_37b_constants.h"
#include "gigachat_moe_device_math.h"

#include <cmath>

namespace frozen = lsxcommon::gigachat4::frozen;

TEST_CASE("gigachat4 derived geometry matches the reference", "[gigachat4][fast]")
{
    CHECK(frozen::kQkHeadDim == 192);
    CHECK(frozen::kLinearKeyDim == 2048);
    CHECK(frozen::kLinearValueDim == 4096);
    CHECK(frozen::kLinearConvDim == 8192);
    CHECK(frozen::kLinearQkvzDim == 12288);
    CHECK(frozen::kLinearBaDim == 64);
    CHECK(frozen::kSharedIntermediate == 768);
    CHECK(frozen::kIndexHeads * frozen::kIndexHeadDim == 8192);

    // YaRN: the cos/sin attention_factor is 1.0 (mscale == mscale_all_dim), and
    // the attention scale carries get_mscale(32, 1)^2 / sqrt(192).
    const double mscale = 0.1 * 1.0 * std::log(32.0) + 1.0;
    CHECK(mscale == Catch::Approx(1.3465736).margin(1e-6));
    const double scale = std::pow(192.0, -0.5) * mscale * mscale;
    CHECK(scale == Catch::Approx(0.13086079996).margin(1e-9));
    CHECK(frozen::kAttnScale == Catch::Approx(scale).margin(1e-6));

    CHECK(frozen::kAlphaQ ==
          Catch::Approx(std::sqrt(2560.0 / 1536.0)).margin(1e-6));
    CHECK(frozen::kAlphaKv ==
          Catch::Approx(std::sqrt(2560.0 / 512.0)).margin(1e-6));
    CHECK(frozen::kIndexWeightScale == Catch::Approx(0.125).margin(1e-7));
    CHECK(frozen::kIndexScoreScale ==
          Catch::Approx(std::pow(128.0, -0.5)).margin(1e-7));
}

TEST_CASE("gigachat4 GatedNorm is not zero-centred", "[gigachat4][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    // The norm is `scaled * 2*sigmoid(gate)` where the scale is
    // `xn*(offset+w)`: GC4 offset 0 with w=2 gives 2, the 3.5 offset-1 form 3.
    CHECK(dev::GatedNormScale(1.0f, 2.0f, 0.0f) == Catch::Approx(2.0f));
    CHECK(dev::GatedNormScale(1.0f, 2.0f, 1.0f) == Catch::Approx(3.0f));
    CHECK(dev::GatedNormElement(2.0f, 0.0f) == Catch::Approx(2.0f).margin(1e-6f));
    CHECK(dev::GatedNormElement(3.0f, 0.0f) == Catch::Approx(3.0f).margin(1e-6f));
    // The legacy 3.5 entry point keeps the gate-from-xn form: gate 0 -> the
    // scale is halved by 2*sigmoid(0) = 1.
    CHECK(dev::ZeroCenteredGatedNormElement(1.0f, 2.0f, 0.0f) ==
          Catch::Approx(3.0f).margin(1e-6f));
    // Large negative gate drives the gating to ~0, not to a sign flip.
    CHECK(dev::GatedNormElement(2.0f, -60.0f) ==
          Catch::Approx(0.0f).margin(1e-6f));
}

TEST_CASE("gigachat4 GDN output norm is sigmoid-gated without the offset",
          "[gigachat4][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    // x_norm=1, w=2, z=0, scale=2 -> 1*2*2*0.5 = 2 (offset 0);
    // the zero-centred 3.5 form gives 1*3*2*0.5 = 3.
    CHECK(dev::GdnOutputNormElementOffset(1.0f, 2.0f, 0.0f, 1e-6f, 2.0f, 0.0f) ==
          Catch::Approx(2.0f).margin(1e-6f));
    CHECK(dev::GdnOutputNormElement(1.0f, 2.0f, 0.0f, 1e-6f, 2.0f) ==
          Catch::Approx(3.0f).margin(1e-6f));
}

TEST_CASE("gigachat4 logit softcap saturates like tanh", "[gigachat4][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    CHECK(dev::LogitSoftcapElement(0.0f, 30.0f) == Catch::Approx(0.0f));
    CHECK(dev::LogitSoftcapElement(30.0f, 30.0f) ==
          Catch::Approx(30.0f * std::tanh(1.0f)).margin(1e-6f));
    CHECK(dev::LogitSoftcapElement(-30.0f, 30.0f) ==
          Catch::Approx(-30.0f * std::tanh(1.0f)).margin(1e-6f));
    CHECK(dev::LogitSoftcapElement(1.0e6f, 30.0f) ==
          Catch::Approx(30.0f).margin(1e-3f));
    // cap <= 0 disables the softcap (identity), never divides by zero.
    CHECK(dev::LogitSoftcapElement(5.0f, 0.0f) == Catch::Approx(5.0f));
    // Monotone: a larger pre-cap logit never maps to a smaller post-cap one.
    float prev = -1.0e9f;
    for (float x = -200.0f; x <= 200.0f; x += 5.0f)
    {
        const float y = dev::LogitSoftcapElement(x, 30.0f);
        CHECK(y >= prev);
        CHECK(std::abs(y) <= 30.0f);
        prev = y;
    }
}

TEST_CASE("gigachat4 swiglu clamp and attention gate", "[gigachat4][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    CHECK(dev::SwigluLimitElement(20.0f, 20.0f, 10.0f) ==
          Catch::Approx(99.995f).margin(1e-2f));
    CHECK(dev::SwigluLimitElement(20.0f, 20.0f, 0.0f) ==
          Catch::Approx(20.0f * dev::Silu(20.0f)).margin(1e-2f));
    CHECK(dev::SigmoidGateElement(8.0f, 0.0f) ==
          Catch::Approx(4.0f).margin(1e-6f));
}
