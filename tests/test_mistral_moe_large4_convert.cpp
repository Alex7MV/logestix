// Mistral-Large-4-1T-A52B passport + converter tests (Model Ingestion Protocol
// stages 0/1). The passport case is host-only; the synthetic E2E converter case
// runs the real pipeline on a fixture (label `convert`, bounded TIMEOUT,
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY as a RUNTIME env var).

#include <catch2/catch_test_macros.hpp>

#include "mistral_moe_large4_constants.h"

using namespace lsxcommon::mistral_large4::frozen;

TEST_CASE("Mistral-Large-4 geometry is compile-time locked", "[mistral_large4][convert]")
{
    STATIC_REQUIRE(kHidden == 6144);
    STATIC_REQUIRE(kLayers == 72);
    STATIC_REQUIRE(kVocab == 204800);
    STATIC_REQUIRE(kHeads == 64);
    STATIC_REQUIRE(kKvHeads == 64);
    STATIC_REQUIRE(kKvLora == 512);
    STATIC_REQUIRE(kQkNope == 64);
    STATIC_REQUIRE(kQkRope == 64);
    STATIC_REQUIRE(kQkHead == 128);
    STATIC_REQUIRE(kVHead == 128);
    STATIC_REQUIRE(kQSize == 8192);
    STATIC_REQUIRE(kKvAWidthRope == 576);
    STATIC_REQUIRE(kKvAWidthNope == 512);
    STATIC_REQUIRE(kKvBRowsRope == 12288);
    STATIC_REQUIRE(kKvBRowsNope == 16384);
    STATIC_REQUIRE(kOSize == 8192);
    STATIC_REQUIRE(kDenseIntermediate == 24576);
    STATIC_REQUIRE(kMoeIntermediate == 6144);
    STATIC_REQUIRE(kNumRouted == 128);
    STATIC_REQUIRE(kTopK == 4);
    STATIC_REQUIRE(kIndexHeads == 32);
    STATIC_REQUIRE(kIndexHeadDim == 128);
    STATIC_REQUIRE(kIndexTopk == 2048);
    STATIC_REQUIRE(kLeRopeDims == 32);
    STATIC_REQUIRE(kEosId == 2);
    STATIC_REQUIRE(kPadId == 11);
}

TEST_CASE("Mistral-Large-4 layer tables match the checkpoint config",
          "[mistral_large4][convert]")
{
    int rope = 0, nope = 0, shared = 0, full = 0, moe = 0, dense = 0;
    for (int i = 0; i < kLayers; ++i)
    {
        if (kRopeLayer[static_cast<size_t>(i)]) ++rope; else ++nope;
        if (kKvSource[static_cast<size_t>(i)] != i) ++shared;
        if (kIndexerFull[static_cast<size_t>(i)]) ++full;
        if (kMoeLayer[static_cast<size_t>(i)]) ++moe; else ++dense;
    }
    REQUIRE(rope == kNumRopeLayers);
    REQUIRE(nope == kNumNopeLayers);
    REQUIRE(shared == kNumKvShared);
    REQUIRE(full == kNumFullIndexers);
    REQUIRE(moe == 71);
    REQUIRE(dense == 1);
    // Every NoPE layer is index i % 4 == 3 and every KV sharer is a NoPE layer
    // reading layer 35.
    for (int i = 0; i < kLayers; ++i)
    {
        const bool is_nope = kRopeLayer[static_cast<size_t>(i)] == 0;
        REQUIRE(is_nope == (i % 4 == 3));
        if (kKvSource[static_cast<size_t>(i)] != i)
        {
            REQUIRE(is_nope);
            REQUIRE(kKvSource[static_cast<size_t>(i)] == kKvSourceLayer);
        }
    }
    // The dense MLP prefix is exactly layer 0.
    REQUIRE(kMoeLayer[0] == 0);
    REQUIRE(kFirstKDense == 1);
}
