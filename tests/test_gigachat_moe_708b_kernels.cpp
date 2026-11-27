// TDD — GigaChat4-708B-A41B passport. Pins the geometry the checkpoint's
// config.json declares (7168 hidden, 64 layers, 61 MoE layers of 256 experts,
// 16 full-attention DSA layers, GDN 32 key / 64 value heads) and the derived
// values that differ from the shipped 37B sibling (alpha_q, alpha_kv, widths).
// Host-only: no CUDA, no weights.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_708b_constants.h"
#include "gigachat_moe_device_math.h"

#include <cmath>
#include <string>

namespace frozen = lsxcommon::gigachat4_708b::frozen;

TEST_CASE("gigachat4-708b identity is distinct from the 37B row",
          "[gigachat708b][fast]")
{
    CHECK(std::string(frozen::kType) == "gigachat4-708b-a41b");
    // A distinct format tag: a 708B loader must reject a 37B tree before it even
    // reaches the geometry gate.
    CHECK(std::string(frozen::kFormat) == "logestix-gigachat4-708b-nvfp4");
    CHECK(std::string(frozen::kModelName) == "GigaChat4-708B-A41B");
}

TEST_CASE("gigachat4-708b geometry matches config.json", "[gigachat708b][fast]")
{
    CHECK(frozen::kHidden == 7168);
    CHECK(frozen::kLayers == 64);
    CHECK(frozen::kVocab == 128256);
    CHECK(frozen::kFirstKDense == 3);
    CHECK(frozen::kNumRouted == 256);
    CHECK(frozen::kTopK == 8);
    CHECK(frozen::kMoeIntermediate == 2048);
    CHECK(frozen::kDenseIntermediate == 18432);
    CHECK(frozen::kHeads == 64);
    CHECK(frozen::kKVHeads == 64);
    CHECK(frozen::kQLoraRank == 1536);
    CHECK(frozen::kKvLoraRank == 512);
    CHECK(frozen::kLinearNumKeyHeads == 32);
    CHECK(frozen::kLinearNumValueHeads == 64);
    CHECK(frozen::kMtpLayer == 64);
    CHECK(frozen::kNumFullAttn == 16);
    CHECK(frozen::kFullAttnLayers[0] == 3);
    CHECK(frozen::kFullAttnLayers[15] == 63);
    // MoE layers = every layer from first_k_dense_replace to the last one.
    CHECK(frozen::kLayers - frozen::kFirstKDense == 61);
}

TEST_CASE("gigachat4-708b derived widths", "[gigachat708b][fast]")
{
    CHECK(frozen::kQkHeadDim == 192);
    CHECK(frozen::kLinearKeyDim == 4096);
    CHECK(frozen::kLinearValueDim == 8192);
    CHECK(frozen::kLinearConvDim == 16384);
    CHECK(frozen::kLinearQkvzDim == 24576);
    CHECK(frozen::kLinearBaDim == 128);
    CHECK(frozen::kSharedIntermediate == 2048);
    CHECK(frozen::kIndexHeads * frozen::kIndexHeadDim == 8192);
    CHECK(frozen::kGateRank == 16);
}

TEST_CASE("gigachat4-708b alpha latents are re-derived, not inherited",
          "[gigachat708b][fast]")
{
    // alpha = sqrt(hidden / rank). These are the two constants that differ from
    // the 37B row (1.2909944 / 2.2360680): a copy would still decode, wrongly.
    CHECK(frozen::kAlphaQ == Catch::Approx(std::sqrt(7168.0 / 1536.0)).margin(1e-6));
    CHECK(frozen::kAlphaKv == Catch::Approx(std::sqrt(7168.0 / 512.0)).margin(1e-6));
    CHECK(frozen::kAlphaQ == Catch::Approx(2.160246899469287).margin(1e-9));
    CHECK(frozen::kAlphaKv == Catch::Approx(3.7416573867739413).margin(1e-9));
    // YaRN: the cos/sin factor is 1.0 (mscale == mscale_all_dim) and the
    // attention scale carries get_mscale(32, 1)^2 / sqrt(192).
    const double mscale = 0.1 * std::log(32.0) + 1.0;
    CHECK(frozen::kAttnScale ==
          Catch::Approx(std::pow(192.0, -0.5) * mscale * mscale).margin(1e-6));
}

TEST_CASE("gigachat4-708b variant math is the 4.0 form at these widths",
          "[gigachat708b][fast]")
{
    namespace dev = lsxcommon::gigachat::dev;
    // GatedNorm is not zero-centred; the GDN output norm is the plain-scale
    // sigmoid variant; the logits are tanh-capped at 30.
    CHECK(dev::GatedNormScale(1.0f, 2.0f, 0.0f) == Catch::Approx(2.0f));
    CHECK(dev::GdnOutputNormElementOffset(1.0f, 2.0f, 0.0f, 1e-6f, 2.0f, 0.0f) ==
          Catch::Approx(2.0f).margin(1e-6f));
    CHECK(dev::LogitSoftcapElement(1.0e6f, frozen::kLogitSoftcap) ==
          Catch::Approx(30.0f).margin(1e-3f));
}
