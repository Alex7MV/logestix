// TDD — GigaChat3.5 host reference math (no CUDA). Pins the derived geometry
// the kernels will implement (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "gigachat_moe_constants.h"
#include "gigachat_moe_device_math.h"

#include <cmath>

namespace frozen = lsxcommon::gigachat::frozen;

TEST_CASE("gigachat derived geometry matches the reference", "[gigachat][fast]")
{
    CHECK(frozen::kQkHeadDim == 192);
    CHECK(frozen::kLinearKeyDim == 4096);
    CHECK(frozen::kLinearValueDim == 8192);
    CHECK(frozen::kLinearConvDim == 16384);
    CHECK(frozen::kLinearQkvzDim == 24576);
    CHECK(frozen::kLinearBaDim == 128);
    CHECK(frozen::kSharedIntermediate == 2048);

    const double mscale = 0.1 * 1.0 * std::log(8.0) + 1.0;  // yarn_get_mscale(8,1)
    CHECK(mscale == Catch::Approx(1.20794).margin(1e-5));
    const double scale = std::pow(192.0, -0.5) * mscale * mscale;
    CHECK(scale == Catch::Approx(0.105299).margin(1e-5));

    const double alpha_q = std::sqrt(7168.0 / 1536.0);
    const double alpha_kv = std::sqrt(7168.0 / 512.0);
    CHECK(alpha_q == Catch::Approx(2.160246).margin(1e-5));
    CHECK(alpha_kv == Catch::Approx(3.741657).margin(1e-5));
}

TEST_CASE("gigachat gating reference math", "[gigachat][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    // ZeroCenteredGatedNorm: x_norm*(1+w)*2*sigmoid(gate).
    // x_norm=1, w=0 -> 1; gate=0 -> sigmoid=0.5 -> 1*1*2*0.5 = 1.
    CHECK(dev::ZeroCenteredGatedNormElement(1.0f, 0.0f, 0.0f) ==
          Catch::Approx(1.0f).margin(1e-6f));
    // GDN output norm: x_norm*(1+w)*scale*sigmoid(z) = 1*1*2*0.5.
    CHECK(dev::GdnOutputNormElement(1.0f, 0.0f, 0.0f, 1e-6f, 2.0f) ==
          Catch::Approx(1.0f).margin(1e-6f));
    // SwiGLU limit: gate 20 -> clamp 10; up 20 -> clamp 10; silu(10)*10.
    CHECK(dev::SwigluLimitElement(20.0f, 20.0f, 10.0f) ==
          Catch::Approx(99.995f).margin(1e-2f));
    // Gated attention: y * sigmoid(gate) = 8 * 0.5.
    CHECK(dev::SigmoidGateElement(8.0f, 0.0f) ==
          Catch::Approx(4.0f).margin(1e-6f));
}
