#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/kimi_inference_load.h"

TEST_CASE("Kimi geometry is compile-time locked", "[kimi][convert]") {
    using namespace lsxcommon::kimi;
    STATIC_REQUIRE(kKimiHidden == 7168);
    STATIC_REQUIRE(kKimiNumLayers == 61);
    STATIC_REQUIRE(kKimiNumHeads == 64);
    STATIC_REQUIRE(kKimiKvLoraRank == 512);
    STATIC_REQUIRE(kKimiQLoraRank == 1536);
    STATIC_REQUIRE(kKimiQkNope == 128);
    STATIC_REQUIRE(kKimiQkRope == 64);
    STATIC_REQUIRE(kKimiHeadV == 128);
    STATIC_REQUIRE(kKimiMoeIntermediate == 2048);
    STATIC_REQUIRE(kKimiDenseIntermediate == 18432);
    STATIC_REQUIRE(kKimiNumRouted == 384);
    STATIC_REQUIRE(kKimiNumShared == 1);
    STATIC_REQUIRE(kKimiTopK == 8);
    STATIC_REQUIRE(kKimiVocab == 163840);
    STATIC_REQUIRE(kKimiFirstKDenseReplace == 1);
}
