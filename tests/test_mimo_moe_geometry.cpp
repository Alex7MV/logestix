// TDD — MiMo-V2.6-Flash-RL frozen geometry passport (host-only).
// Model Ingestion Protocol step 3: frozen geometry is compile-time and
// cross-checked at Load; this test pins the passport invariants.
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "mimo_moe_constants.h"

using namespace lsxcommon::mimo::frozen;

TEST_CASE("mimo frozen geometry is internally consistent", "[mimo][geometry][fast]")
{
    STATIC_REQUIRE(kLayers == 48);
    STATIC_REQUIRE(kHidden == 4096);
    STATIC_REQUIRE(kVocab == 152576);
    STATIC_REQUIRE(kFullAttnLayers.size() == 9);

    STATIC_REQUIRE(kQSize == kHeads * kQkHeadDim);
    STATIC_REQUIRE(kOSize == kHeads * kVHeadDim);
    STATIC_REQUIRE(kQSize + kKSizeGa + kVSizeGa == 13568);
    STATIC_REQUIRE(kQSize + kKSizeSwa + kVSizeSwa == 14848);
    STATIC_REQUIRE(kRopeDim == 64);
    STATIC_REQUIRE(kRopeDim % 2 == 0);

    REQUIRE(std::fabs(static_cast<double>(kAttnScale) - std::pow(192.0, -0.5)) < 1e-7);
    REQUIRE(std::fabs(static_cast<double>(kValueScale) - 0.707) < 1e-6);
}

TEST_CASE("mimo layer roles decode from the frozen pattern", "[mimo][geometry][fast]")
{
    const int full[9] = {0, 5, 11, 17, 23, 29, 35, 41, 47};
    for (int l = 0; l < kLayers; ++l) {
        bool expect_full = false;
        for (int f : full) expect_full = expect_full || (f == l);
        REQUIRE(IsFullAttnLayer(l) == expect_full);
        // Layer 0 is the sole dense FFN; 1..47 are MoE.
        REQUIRE(IsMoeLayer(l) == (l >= 1));
    }
}
