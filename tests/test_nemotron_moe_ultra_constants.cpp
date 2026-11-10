// Frozen-passport guard for NVIDIA-Nemotron-3-Ultra-550B-A55B.
//
// Host-only: no CUDA, no weights. The passport is the compile-time geometry the
// loader cross-checks a converted tree against, so a wrong value here is a
// silent weight-shape mismatch at run time; every claim is asserted against the
// real checkpoint's config.json / safetensors inventory (2026-11-10).
#include <catch2/catch_test_macros.hpp>

#include <cstring>

#include "nemotron_moe_ultra_constants.h"

using namespace lsxcommon::nemotron_ultra;

TEST_CASE("ultra passport: kType is the manifest model_type")
{
    REQUIRE(std::strcmp(frozen::kType, "nemotron-ultra-550b-a55b") == 0);
}

TEST_CASE("ultra passport: layer mix decodes to 48/48/12 with the reference indices")
{
    REQUIRE(frozen::kMambaLayers == 48);
    REQUIRE(frozen::kMoeLayers == 48);
    REQUIRE(frozen::kAttnLayers == 12);
    REQUIRE(frozen::kLayers == 108);
    // layers_block_type of the real checkpoint; the attention layers are the
    // only ones that carry a KV cache, so this list is the cache's shape.
    const int expect[12] = {7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98};
    for (int i = 0; i < frozen::kAttnLayers; ++i)
        REQUIRE(frozen::kAttnLayerIdx[i] == expect[i]);
}

TEST_CASE("ultra passport: mamba geometry is self-consistent")
{
    REQUIRE(frozen::kMambaDInner == frozen::kMambaHeads * frozen::kMambaHeadDim);
    REQUIRE(frozen::kMambaDInner == 16384);
    REQUIRE(frozen::kMambaConvDim ==
            frozen::kMambaDInner + 2 * frozen::kSsmGroups * frozen::kSsmStateSize);
    REQUIRE(frozen::kMambaConvDim == 18432);
    REQUIRE(frozen::kMambaInProj ==
            frozen::kMambaDInner + frozen::kMambaConvDim + frozen::kMambaHeads);
    REQUIRE(frozen::kMambaInProj == 35072);
    REQUIRE(frozen::kMambaGroupSize == frozen::kMambaDInner / frozen::kSsmGroups);
    REQUIRE(frozen::kMambaGroupSize == 2048);
}

TEST_CASE("ultra passport: attention and MoE geometry")
{
    REQUIRE(frozen::kHeads * frozen::kHeadDim == 8192);   // full-width q/o
    REQUIRE(frozen::kKVHeads * frozen::kHeadDim == 256);  // k/v rows
    REQUIRE(frozen::kHeads % frozen::kKVHeads == 0);
    REQUIRE(frozen::kTopK == 22);
    REQUIRE(frozen::kNumRouted == 512);
    REQUIRE(frozen::kNumRouted % frozen::kRouterGroups == 0);
    REQUIRE(frozen::kNumRouted / frozen::kRouterGroups == 64);
    REQUIRE(frozen::kRouterTopkGroup == 1);
    REQUIRE(frozen::kTopK <= frozen::kNumRouted);
    REQUIRE(frozen::kMoeLatent == 2048);
    REQUIRE(frozen::kSharedIntermediate == 10240);
    REQUIRE(frozen::kRoutedScale == 5.0f);
    REQUIRE(frozen::kVocab == 131072);
    REQUIRE(frozen::kRmsEps == 1e-5);
}

TEST_CASE("ultra passport: geometry is distinct from the 30B sibling")
{
    // A copied 30B passport must not satisfy this model's own facts: the two
    // lineages are separate on purpose, and this is what stops one being
    // silently pointed at the other's converted tree.
    REQUIRE(frozen::kHidden != 2688);
    REQUIRE(frozen::kLayers != 52);
    REQUIRE(frozen::kNumRouted != 128);
    REQUIRE(frozen::kTopK != 6);
    REQUIRE(frozen::kMoeLatent != 0);
    REQUIRE(frozen::kMambaDInner != 4096);
}
