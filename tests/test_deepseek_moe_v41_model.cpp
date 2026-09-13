// tests/test_deepseek_moe_v41_model.cpp — DeepSeek-V4.1-Flash model-stage
// (host-only): frozen passport geometry and the CED per-layer classification.
// The model class itself lands in Plan 6; this stage pins the constants the
// whole contour compiles against (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include "deepseek_moe_v41_constants.h"

using namespace lsxcommon::deepseek_v41;

TEST_CASE("deepseek-v41 frozen geometry is locked", "[deepseek-v41][model]")
{
    CHECK(kHidden == 5120);
    CHECK(kLayers == 40);
    CHECK(kMoeIntermediate == 2304);
    CHECK(kSharedIntermediate == 2304);
    CHECK(kNumRouted == 384);
    CHECK(kTopK == 6);
    CHECK(kHeads == 64);
    CHECK(kHeadDim == 512);
    CHECK(kRopeHeadDim == 64);
    CHECK(kQLoraRank == 1280);
    CHECK(kOLoraRank == 1024);
    CHECK(kOGroups == 8);
    CHECK(kIndexNHeads == 32);
    CHECK(kIndexHeadDim == 128);
    CHECK(kIndexTopk == 512);
    CHECK(kWindow == 128);
    CHECK(kVocab == 129280);
    CHECK(kHcMult == 4);
    CHECK(kHcMix == 24);
    CHECK(kHcDim == 20480);
    CHECK(kSinkhornIters == 20);
    CHECK(kCandidateBlockSize == 8);
    CHECK(kCandidateTopkBlocks == 2048);
    CHECK(kMtpLayers == 3);
}

TEST_CASE("deepseek-v41 CED layer classification matches config",
          "[deepseek-v41][model]")
{
    // compress_ratios: layers 0-1 -> 0, 2..19 -> 2, 20..39 -> 1.
    CHECK(CompressRatio(0) == 0);
    CHECK(CompressRatio(1) == 0);
    CHECK(CompressRatio(2) == 2);
    CHECK(CompressRatio(19) == 2);
    CHECK(CompressRatio(20) == 1);
    CHECK(CompressRatio(39) == 1);

    // kv_source_layers = {2,8,14,20}
    CHECK(IsKvSource(2));
    CHECK(IsKvSource(8));
    CHECK(IsKvSource(14));
    CHECK(IsKvSource(20));
    CHECK_FALSE(IsKvSource(21));

    // index_source_layers = {2,8,14,20,24,28,32,36}
    CHECK(IsIndexSource(2));
    CHECK(IsIndexSource(36));
    CHECK_FALSE(IsIndexSource(35));

    CHECK(kCandidateSourceLayer == 20);
    CHECK(IsCandidateSource(20));
    CHECK_FALSE(IsCandidateSource(21));

    // Every kv source is also an index source; every index source compresses.
    for (int l = 0; l < kLayers; ++l) {
        if (IsKvSource(l)) CHECK(IsIndexSource(l));
        if (IsIndexSource(l)) CHECK(CompressRatio(l) > 0);
    }
    // Candidate source publishes the level-one mask and owns both caches.
    CHECK(IsKvSource(kCandidateSourceLayer));
    CHECK(IsIndexSource(kCandidateSourceLayer));

    const LayerMeta m = LayerMetaAt(20);
    CHECK(m.compress_ratio == 1);
    CHECK(m.is_kv_source);
    CHECK(m.is_index_source);
    CHECK(m.is_candidate_source);
}
