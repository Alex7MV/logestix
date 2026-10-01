// AliceAI-Foundation-80B-A3B model-stage tests. The passport case is host-only
// and runs first; the loader/pipeline cases land in later stages.
#include <catch2/catch_test_macros.hpp>

#include "alice_moe_constants.h"

using namespace lsxcommon::alice::frozen;

TEST_CASE("alice passport matches the audited checkpoint", "[alice][fast]")
{
    STATIC_REQUIRE(kHidden == 2048);
    STATIC_REQUIRE(kLayers == 48);
    STATIC_REQUIRE(kFullAttnInterval == 4);
    STATIC_REQUIRE(kBlockAttnRes == 4);
    STATIC_REQUIRE(kHeads == 16);
    STATIC_REQUIRE(kKVHeads == 2);
    STATIC_REQUIRE(kHeadDim == 256);
    STATIC_REQUIRE(kRotaryDim == 64);
    STATIC_REQUIRE(kLinearKeyHeads == 32);
    STATIC_REQUIRE(kLinearValueHeads == 32);
    STATIC_REQUIRE(kLinearKeyHeadDim == 128);
    STATIC_REQUIRE(kLinearValueHeadDim == 128);
    STATIC_REQUIRE(kLinearConvKernel == 4);
    STATIC_REQUIRE(kLinearKvRepeat == 1);
    STATIC_REQUIRE(kNumExperts == 512);
    STATIC_REQUIRE(kTopK == 10);
    STATIC_REQUIRE(kMoeIntermediate == 512);
    STATIC_REQUIRE(kSharedIntermediate == 512);
    STATIC_REQUIRE(kVocab == 129024);
    STATIC_REQUIRE(kBosId == 1);
    STATIC_REQUIRE(kEosId == 2);
    STATIC_REQUIRE(kMtpLayers == 1);
}
