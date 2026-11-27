// Frozen passport guard for Inkling-2-Mini (inkling2_loop). Host-only.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Mini: config.json,
// hf_quant_config.json, tokenizer.json added_tokens and the safetensors
// headers) and cross-checked against the reference semantics in transformers
// 5.24.0 `models/inkling2_loop/modeling_inkling2_loop.py`. Load re-derives the
// same numbers from the converted index, so a passport edit without a re-audit
// fails Load instead of silently decoding a different model.
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

#include "../libs/lsxcommon/src/inkling_moe_loop_constants.h"

using namespace lsxcommon::inkling::loop::frozen;

TEST_CASE("inkling-2-mini passport: registry and format identity",
          "[inkling][constants]")
{
    CHECK(std::string(kModelType) == "inkling-2-mini");
    CHECK(std::string(kFormatTag) == "logestix-inkling-2-mini");
    CHECK(std::string(kSourceModelType) == "inkling2_loop");
    CHECK(kTokenizerKind == 2);
}

TEST_CASE("inkling-2-mini passport: text tower geometry", "[inkling][constants]")
{
    CHECK(kHidden == 2304);
    CHECK(kLayers == 24);
    CHECK(kHeads == 24);
    CHECK(kKvHeads == 8);
    CHECK(kHeadDim == 128);
    CHECK(kGqaGroups == 3);
    CHECK(kVocab == 201024);
    CHECK(kBaseVocab == 200058);
    CHECK(kRmsNormEps == 1e-6);
    // head_dim**-0.5, as inkling2 (NOT the older inkling's 1/head_dim).
    CHECK(kAttnScaleDenom == 128.0);
    CHECK(kConvKernel == 4);
}

TEST_CASE("inkling-2-mini passport: the 4 full layers are the (layer+1)%6 ids",
          "[inkling][constants]")
{
    CHECK(kFullLayerCount == 4);
    CHECK(kSlidingLayerCount == 20);
    CHECK(kFullLayers == std::array<int, 4>{5, 11, 17, 23});
    int full = 0;
    for (int l = 0; l < kLayers; ++l)
    {
        const bool expect_full = ((l + 1) % 6 == 0);
        CHECK(IsFullLayer(l) == expect_full);
        CHECK(IsSlidingLayer(l) == !expect_full);
        full += IsFullLayer(l) ? 1 : 0;
    }
    CHECK(full == kFullLayerCount);
    CHECK(kFullLayers.back() == kLayers - 1);
}

TEST_CASE("inkling-2-mini passport: the loop core is layers [6,18) x 3 passes",
          "[inkling][constants]")
{
    CHECK(kLoopStart == 6);
    CHECK(kLoopEnd == 18);
    CHECK(kLoopPasses == 3);
    CHECK(kLoopGateRank == 64);
    CHECK(kLoopLayers == 12);
    // 24 physical layers, 48 applications per token.
    CHECK(kLayerApplications == 48);
    CHECK(kCacheLayers == 48);
    CHECK_FALSE(IsLoopLayer(5));
    CHECK(IsLoopLayer(6));
    CHECK(IsLoopLayer(17));
    CHECK_FALSE(IsLoopLayer(18));
    // The loop bounds are AttnRes block boundaries (config validate_architecture).
    CHECK(kLoopStart % kAttnResBlock == 0);
    CHECK(kLoopEnd % kAttnResBlock == 0);
    CHECK(kLoopStart > 0);
}

TEST_CASE("inkling-2-mini passport: per-pass cache slots", "[inkling][constants]")
{
    // Pass 0 uses the physical layer slot; passes 1/2 append after the stack.
    for (int l = 0; l < kLayers; ++l) CHECK(CacheSlot(l, 0) == l);
    CHECK(CacheSlot(6, 1) == 24);
    CHECK(CacheSlot(17, 1) == 35);
    CHECK(CacheSlot(6, 2) == 36);
    CHECK(CacheSlot(17, 2) == 47);
    // Every (loop layer, pass) pair owns a distinct slot; non-loop layers have
    // exactly one.
    std::set<int> slots;
    for (int l = 0; l < kLayers; ++l)
    {
        const int passes = IsLoopLayer(l) ? kLoopPasses : 1;
        for (int p = 0; p < passes; ++p)
            CHECK(slots.insert(CacheSlot(l, p)).second);
    }
    CHECK(slots.size() == kCacheLayers);
}

TEST_CASE("inkling-2-mini passport: AttnRes blocks are layer%6==0 boundaries",
          "[inkling][constants]")
{
    CHECK(kAttnResBlock == 6);
    CHECK(kAttnResBlocks == 4);
    CHECK(kAttnResMaxSources == 5);
    CHECK(kFullLayerMixQueries == 3);
    CHECK(kSingleMixQueries == 1);
    int boundaries = 0;
    for (int l = 0; l < kLayers; ++l)
        boundaries += IsBlockBoundary(l) ? 1 : 0;
    CHECK(boundaries == kAttnResBlocks);
    CHECK(IsBlockBoundary(0));
    CHECK_FALSE(IsBlockBoundary(1));
    CHECK(IsBlockBoundary(18));
}

TEST_CASE("inkling-2-mini passport: attention constants", "[inkling][constants]")
{
    CHECK(kSlidingWindow == 1024);
    CHECK(kRelExtent == 1024);
    CHECK(kDRel == 16);
    CHECK(kQkL2Clip == 22.627416997969522);
    CHECK(kRelL2Clip == 8.0);
    // Mini log-scales from 8192, NOT Small's 128000: a copied constant disables
    // the path for every context this card can run.
    CHECK(kLogScaleFloor == 8192);
    CHECK(kLogScaleAlpha == 0.1);
}

TEST_CASE("inkling-2-mini passport: routed MoE", "[inkling][constants]")
{
    CHECK(kExperts == 128);
    CHECK(kTopK == 8);
    CHECK(kMoeIntermediate == 1152);
    CHECK(kExpertGateUpRows == 2304);
    CHECK(kSharedExperts == 0);
    CHECK(kRouteScale == 8.0);
    CHECK(kMlpGateSoftcap == 4.0);
    CHECK(kMlpUpSoftcap == 25.0);
    CHECK(kFinalLogitSoftcap == 30.0);
}

TEST_CASE("inkling-2-mini passport: mixed expert codec split", "[inkling][constants]")
{
    CHECK(kQuantBlock == 16);
    CHECK(kOuterScaleDenominator == 2688.0);
    CHECK(kNovfp4ExpertLayerCount == 20);
    CHECK(kMxfp8ExpertLayerCount == 4);
    CHECK(kMxfp8Block == 32);
    int nvfp4 = 0;
    for (int l = 0; l < kLayers; ++l)
        nvfp4 += ExpertIsNovfp4(l) ? 1 : 0;
    CHECK(nvfp4 == kNovfp4ExpertLayerCount);
    // Layers 20..23 are MXFP8 (dtype census + hf_quant_config.json): the last
    // attention group, and the trap is assuming one codec for the whole stack.
    CHECK(ExpertIsNovfp4(19));
    CHECK_FALSE(ExpertIsNovfp4(20));
    CHECK_FALSE(ExpertIsNovfp4(23));
}

TEST_CASE("inkling-2-mini passport: special ids and chat markers",
          "[inkling][constants]")
{
    CHECK(kBosId == -1);
    CHECK(kPadId == 199999);  // generation_config.json pad_token_id
    CHECK(kEosId == 200006);
    CHECK(kStopIdAlt == 199999);
    CHECK(kMarkerUnused == 199998);
    CHECK(kMarkerUser == 200000);
    CHECK(kMarkerModel == 200001);
    CHECK(kMarkerSystem == 200002);
    CHECK(kMarkerTool == 200003);
    CHECK(kMarkerText == 200004);
    CHECK(kMarkerImage == 200005);
    CHECK(kMarkerEndSampling == 200006);
    CHECK(kMarkerThinking == 200008);
    CHECK(kMarkerEndMessage == 200010);
    CHECK(kMarkerToolError == 200022);
    CHECK(kMarkerXml == 200024);
    CHECK(kMarkerBeginOfText == 200028);
    CHECK(kMarkerInvokeJson == 200049);
    CHECK(kMarkerInvokeText == 200057);
    CHECK(kImagePlaceholder == 200054);

    const std::set<int> markers = {
        kMarkerUnused,    kMarkerUser,       kMarkerModel,
        kMarkerSystem,    kMarkerTool,       kMarkerText,
        kMarkerImage,     kMarkerEndSampling, kMarkerThinking,
        kMarkerEndMessage, kMarkerToolError, kMarkerXml,
        kMarkerBeginOfText, kMarkerInvokeJson, kMarkerInvokeText};
    CHECK(markers.size() == 15);
    for (int m : markers) CHECK((m >= kAddedTokenBase && m < kBaseVocab));
}