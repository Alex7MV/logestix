// Frozen passport guard for TML-d24-varloop-research (`inkling2_loop`,
// is_varloop). Host-only.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/lsx_model/thinkingmachines/TML-d24-varloop-research: config.json,
// hf_quant_config.json, tokenizer.json added_tokens, the safetensors index and
// the model card) on 2026-11-22. The architecture is Inkling-2-Mini's — the
// same `Inkling2LoopForConditionalGeneration` — so the geometry is the same and
// this passport repeats it deliberately (model independence: one frozen
// namespace per model, never a shared header).
//
// The one NEW semantic this row owns is the pass count: `is_varloop: true`
// means the same weights are trained for 1..5 passes and the count is chosen at
// inference time (model card §3), with 3 being the post-trained default the
// checkpoint's own `loop_config.num_passes` declares. A passport that pinned 3
// as the only legal value would be the parent's header with a different name.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <set>
#include <string>

#include "../libs/lsxcommon/src/inkling_moe_varloop_constants.h"

using namespace lsxcommon::inkling::varloop::frozen;

TEST_CASE("tml-d24-varloop passport: registry and format identity",
          "[inkling][varloop][constants]")
{
    CHECK(std::string(kModelType) == "tml-d24-varloop-research");
    CHECK(std::string(kModelCard) == "TML-d24-varloop-research");
    CHECK(std::string(kFormatTag) == "logestix-inkling-2-varloop");
    CHECK(std::string(kSourceModelType) == "inkling2_loop");
    CHECK(kTokenizerKind == 2);
    // The capability this model exists for: the checkpoint declares it, and a
    // passport that says false would silently run the fixed-pass sibling.
    CHECK(kVarLoop);
}

TEST_CASE("tml-d24-varloop passport: text tower geometry is inkling2_loop's",
          "[inkling][varloop][constants]")
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
    CHECK(kAttnScaleDenom == 128.0);
    CHECK(kConvKernel == 4);
}

TEST_CASE("tml-d24-varloop passport: hybrid attention table",
          "[inkling][varloop][constants]")
{
    CHECK(kFullLayerCount == 4);
    CHECK(kFullLayers == std::array<int, 4>{5, 11, 17, 23});
    CHECK(kSlidingWindow == 1024);
    CHECK(kRelExtent == 1024);
    CHECK(kDRel == 16);
    CHECK(kQkL2Clip == 22.627416997969522);
    CHECK(kRelL2Clip == 8.0);
    // 8192 here and in Inkling-2-Mini, NOT Inkling-2-Small's 128000: the constant
    // decides whether the log-scaling path is active at every context this card
    // can run.
    CHECK(kLogScaleFloor == 8192);
    CHECK(kLogScaleAlpha == 0.1);
    for (int l = 0; l < kLayers; ++l)
        CHECK(IsFullLayer(l) == ((l + 1) % 6 == 0));
}

TEST_CASE("tml-d24-varloop passport: the loop core and its pass bounds",
          "[inkling][varloop][constants]")
{
    CHECK(kLoopStart == 6);
    CHECK(kLoopEnd == 18);
    CHECK(kLoopLayers == 12);
    CHECK(kLoopGateRank == 64);
    // The checkpoint's own post-trained default and the model card's range.
    CHECK(kLoopPassesDefault == 3);
    CHECK(kLoopPassesMin == 1);
    CHECK(kLoopPassesMax == 5);
    // 24 physical layers; the default stream is 48 applications per token and
    // the slab is sized for the maximum, 5 passes = 72.
    CHECK(kLayerApplicationsDefault == 48);
    CHECK(kCacheLayersMax == 72);
    CHECK(LayerApplications(1) == 24);
    CHECK(LayerApplications(3) == 48);
    CHECK(LayerApplications(5) == 72);
    CHECK(kLoopStart % kAttnResBlock == 0);
    CHECK(kLoopEnd % kAttnResBlock == 0);
}

TEST_CASE("tml-d24-varloop passport: a requested pass count resolves safely",
          "[inkling][varloop][constants]")
{
    // No request: the count the CHECKPOINT declares (3 for this release, and
    // the shipped `loop_config.num_passes`). The override never invents a
    // default — that would silently change a re-converted tree's behaviour.
    CHECK(ResolvePasses(0, kLoopPassesDefault) == 3);
    CHECK(ResolvePasses(-4, kLoopPassesDefault) == 3);
    CHECK(ResolvePasses(0, 2) == 2);
    // Legal requests pass through unchanged.
    for (int p = kLoopPassesMin; p <= kLoopPassesMax; ++p)
        CHECK(ResolvePasses(p, kLoopPassesDefault) == p);
    // Out of range clamps instead of running an untrained count (the weights
    // were trained with 1..5 passes) or zero passes (no loop at all).
    CHECK(ResolvePasses(6, kLoopPassesDefault) == 5);
    CHECK(ResolvePasses(64, kLoopPassesDefault) == 5);
    CHECK(ResolvePasses(1, kLoopPassesDefault) == 1);
    // Every resolved value indexes a slot the slab actually has.
    for (int p = -2; p < 10; ++p)
        CHECK(LayerApplications(ResolvePasses(p, kLoopPassesDefault)) <=
              kCacheLayersMax);
    CHECK(PassesInTrainedRange(kLoopPassesMin));
    CHECK(PassesInTrainedRange(kLoopPassesDefault));
    CHECK(PassesInTrainedRange(kLoopPassesMax));
    CHECK_FALSE(PassesInTrainedRange(0));
    CHECK_FALSE(PassesInTrainedRange(kLoopPassesMax + 1));
}

TEST_CASE("tml-d24-varloop passport: per-pass cache slots inside the slab",
          "[inkling][varloop][constants]")
{
    // Pass 0 uses the physical layer slot; every extra pass appends a block of
    // kLoopLayers slots after the physical stack. The formula is the release's
    // own `num_hidden_layers + (p-1)*loop_layers - loop_start` cache offset.
    for (int l = 0; l < kLayers; ++l) CHECK(CacheSlot(l, 0) == l);
    CHECK(CacheSlot(kLoopStart, 1) == 24);
    CHECK(CacheSlot(kLoopEnd - 1, 1) == 35);
    CHECK(CacheSlot(kLoopStart, 4) == 60);
    CHECK(CacheSlot(kLoopEnd - 1, 4) == 71);
    // Every (loop layer, pass) pair of the MAX configuration owns a distinct
    // slot, and the whole set is exactly the slab.
    std::set<int> slots;
    for (int l = 0; l < kLayers; ++l) {
        const int passes = IsLoopLayer(l) ? kLoopPassesMax : 1;
        for (int p = 0; p < passes; ++p) CHECK(slots.insert(CacheSlot(l, p)).second);
    }
    CHECK(static_cast<int>(slots.size()) == kCacheLayersMax);
    // A smaller active count uses a PREFIX of the same slab: no slot moves.
    for (int p = 0; p < kLoopPassesDefault; ++p)
        for (int l = kLoopStart; l < kLoopEnd; ++l)
            CHECK(CacheSlot(l, p) < kCacheLayersMax);
}

TEST_CASE("tml-d24-varloop passport: AttnRes depth mixer", "[inkling][varloop][constants]")
{
    CHECK(kAttnResBlock == 6);
    CHECK(kAttnResBlocks == 4);
    CHECK(kAttnResMaxSources == 5);
    CHECK(kFullLayerMixQueries == 3);
    CHECK(kSingleMixQueries == 1);
}

TEST_CASE("tml-d24-varloop passport: routed MoE and softcaps",
          "[inkling][varloop][constants]")
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

TEST_CASE("tml-d24-varloop passport: mixed expert codec split",
          "[inkling][varloop][constants]")
{
    CHECK(kQuantBlock == 16);
    CHECK(kOuterScaleDenominator == 2688.0);
    CHECK(kNovfp4ExpertLayerCount == 20);
    CHECK(kMxfp8ExpertLayerCount == 4);
    CHECK(ExpertIsNovfp4(19));
    CHECK_FALSE(ExpertIsNovfp4(20));
}

TEST_CASE("tml-d24-varloop passport: special ids and chat markers",
          "[inkling][varloop][constants]")
{
    CHECK(kBosId == -1);
    CHECK(kPadId == 199999);
    CHECK(kEosId == 200006);
    CHECK(kStopIdAlt == 199999);
    CHECK(kMarkerUser == 200000);
    CHECK(kMarkerModel == 200001);
    CHECK(kMarkerEndMessage == 200010);
    CHECK(kMarkerInvokeJson == 200049);
    CHECK(kMarkerInvokeText == 200057);
    // Every marker must live in the added-token band of the 200058-token base
    // vocabulary, or the tokenizer would resolve it as ordinary text.
    for (int m : {kMarkerUser, kMarkerModel, kMarkerSystem, kMarkerTool,
                  kMarkerText, kMarkerEndSampling, kMarkerThinking,
                  kMarkerEndMessage, kMarkerInvokeJson, kMarkerInvokeText})
        CHECK((m >= kAddedTokenBase && m < kBaseVocab));
}
