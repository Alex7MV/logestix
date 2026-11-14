// Frozen passport guard for Inkling-2-Small. Host-only, no CUDA deps.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/huggingface/Inkling-2-Small: config.json,
// hf_quant_config.json, tokenizer.json added_tokens and the safetensors
// headers) and cross-checked against the reference semantics in transformers
// 5.23.0 `models/inkling2/modeling_inkling2.py`. Load re-derives the same
// numbers from the converted index, so a passport edit without a re-audit fails
// Load instead of silently decoding a different model.
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

#include "../libs/lsxcommon/src/inkling_moe_constants.h"

using namespace lsxcommon::inkling::frozen;

TEST_CASE("inkling-2-small passport: registry and format identity",
          "[inkling][constants]")
{
    CHECK(std::string(kModelType) == "inkling-2-small");
    CHECK(std::string(kFormatTag) == "logestix-inkling-2-small-nvfp4");
    CHECK(std::string(kSourceModelType) == "inkling2");
    CHECK(kTokenizerKind == 2);
}

TEST_CASE("inkling-2-small passport: text tower geometry", "[inkling][constants]")
{
    CHECK(kHidden == 4608);
    CHECK(kLayers == 48);
    CHECK(kHeads == 48);
    CHECK(kKvHeads == 8);
    CHECK(kHeadDim == 128);
    CHECK(kGqaGroups == 6);
    CHECK(kVocab == 201024);
    CHECK(kBaseVocab == 200058);
    CHECK(kRmsNormEps == 1e-6);
    // Inkling2 scales by head_dim**-0.5; Inkling-1 (a different family) used
    // 1/head_dim. A copy-paste from the older port silently halves the logits.
    CHECK(kAttnScaleDenom == 128.0);
    CHECK(kConvKernel == 4);
}

TEST_CASE("inkling-2-small passport: the 8 full layers are the (layer+1)%6 ids",
          "[inkling][constants]")
{
    CHECK(kFullLayerCount == 8);
    CHECK(kSlidingLayerCount == 40);
    CHECK(kFullLayers == std::array<int, 8>{5, 11, 17, 23, 29, 35, 41, 47});
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

TEST_CASE("inkling-2-small passport: AttnRes blocks are layer%6==0 boundaries",
          "[inkling][constants]")
{
    CHECK(kAttnResBlock == 6);
    CHECK(kAttnResBlocks == 8);
    CHECK(kAttnResMaxSources == 9);
    CHECK(kFullLayerMixQueries == 3);
    CHECK(kSingleMixQueries == 1);
    int boundaries = 0;
    for (int l = 0; l < kLayers; ++l)
        boundaries += IsBlockBoundary(l) ? 1 : 0;
    CHECK(boundaries == kAttnResBlocks);
    CHECK(IsBlockBoundary(0));
    CHECK_FALSE(IsBlockBoundary(1));
    CHECK(IsBlockBoundary(42));
}

TEST_CASE("inkling-2-small passport: attention constants", "[inkling][constants]")
{
    CHECK(kSlidingWindow == 1024);
    CHECK(kRelExtent == 1024);
    CHECK(kDRel == 16);
    CHECK(kQkL2Clip == 22.627416997969522);
    CHECK(kRelL2Clip == 8.0);
    CHECK(kLogScaleFloor == 128000);
    CHECK(kLogScaleAlpha == 0.1);
}

TEST_CASE("inkling-2-small passport: routed MoE", "[inkling][constants]")
{
    CHECK(kExperts == 384);
    CHECK(kTopK == 8);
    CHECK(kMoeIntermediate == 2304);
    CHECK(kExpertGateUpRows == 4608);
    CHECK(kSharedExperts == 0);
    CHECK(kRouteScale == 8.0);
    CHECK(kMlpGateSoftcap == 4.0);
    CHECK(kMlpUpSoftcap == 25.0);
    CHECK(kFinalLogitSoftcap == 30.0);
}

TEST_CASE("inkling-2-small passport: mixed expert codec split", "[inkling][constants]")
{
    CHECK(kQuantBlock == 16);
    CHECK(kOuterScaleDenominator == 2688.0);
    CHECK(kNovfp4ExpertLayerCount == 40);
    CHECK(kMxfp8ExpertLayerCount == 8);
    CHECK(kMxfp8Block == 32);
    int nvfp4 = 0;
    for (int l = 0; l < kLayers; ++l)
        nvfp4 += ExpertIsNovfp4(l) ? 1 : 0;
    CHECK(nvfp4 == kNovfp4ExpertLayerCount);
    // Layers 40..47 are MXFP8 in hf_quant_config.json, and they are also the
    // last attention group: the trap is assuming one codec for the whole stack.
    CHECK(ExpertIsNovfp4(39));
    CHECK_FALSE(ExpertIsNovfp4(40));
    CHECK_FALSE(ExpertIsNovfp4(47));
}

TEST_CASE("inkling-2-small passport: special ids and chat markers",
          "[inkling][constants]")
{
    CHECK(kBosId == -1);
    CHECK(kPadId == -1);
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

    // Every marker is a distinct added token above the base vocabulary.
    const std::set<int> markers = {
        kMarkerUnused,    kMarkerUser,       kMarkerModel,
        kMarkerSystem,    kMarkerTool,       kMarkerText,
        kMarkerImage,     kMarkerEndSampling, kMarkerThinking,
        kMarkerEndMessage, kMarkerToolError, kMarkerXml,
        kMarkerBeginOfText, kMarkerInvokeJson, kMarkerInvokeText};
    CHECK(markers.size() == 15);
    for (int m : markers) CHECK((m >= kAddedTokenBase && m < kBaseVocab));
}
