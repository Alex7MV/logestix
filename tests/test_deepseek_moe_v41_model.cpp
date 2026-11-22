// tests/test_deepseek_moe_v41_model.cpp — DeepSeek-V4.1-Flash model-stage
// (host-only): frozen passport geometry, the CED per-layer classification and
// the SIBLING checkpoint's (V4.1-Flash-1017) passport/registry identity.
// The model class itself lands in Plan 6; this stage pins the constants the
// whole contour compiles against (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include <string>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>

#include "deepseek_moe_v41_1017_constants.h"
#include "deepseek_moe_v41_constants.h"

using namespace lsxcommon::deepseek_v41;

namespace {

// `gpu::detect_model_type` (deepseek_moe_load.cu): spaces -> '-', lower case.
std::string NormalizeCardName(std::string s)
{
    for (char &c : s) {
        if (c == ' ')
            c = '-';
        else if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

} // namespace

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

// ── Sibling checkpoint: DeepSeek-V4.1-Flash-1017 ───────────────────────────
//
// Stage 0 triage (2026-11-22): the 1017 tree ships a byte-identical config.json,
// tokenizer.json and model.safetensors.index.json against the parent, and
// re-trained payload bytes. It is therefore a SIBLING VARIANT, not a new family:
// it owns its frozen passport (model independence: one namespace per registered
// model) and reuses the parent's converter, loader, kernels and pipeline.

TEST_CASE("deepseek-v41-1017 passport is the parent checkpoint's geometry",
          "[deepseek-v41][model][sibling]")
{
    namespace sibling = lsxcommon::deepseek_v41_flash1017;

    // Registry identity: detect_model_type normalizes the index's `model`
    // field, which the converter stamps from the input directory's card name.
    CHECK(std::string(sibling::kType) == "deepseek-v4.1-flash-1017");
    CHECK(NormalizeCardName(sibling::kModelName) ==
          std::string(sibling::kType));

    // The source configs are byte-identical, so every frozen number must agree
    // with the parent's passport field by field. A drift here means the sibling
    // is validated against geometry its checkpoint does not have.
    CHECK(sibling::kHidden == kHidden);
    CHECK(sibling::kLayers == kLayers);
    CHECK(sibling::kMoeIntermediate == kMoeIntermediate);
    CHECK(sibling::kSharedIntermediate == kSharedIntermediate);
    CHECK(sibling::kNumRouted == kNumRouted);
    CHECK(sibling::kTopK == kTopK);
    CHECK(sibling::kHeads == kHeads);
    CHECK(sibling::kHeadDim == kHeadDim);
    CHECK(sibling::kRopeHeadDim == kRopeHeadDim);
    CHECK(sibling::kQLoraRank == kQLoraRank);
    CHECK(sibling::kOLoraRank == kOLoraRank);
    CHECK(sibling::kOGroups == kOGroups);
    CHECK(sibling::kIndexNHeads == kIndexNHeads);
    CHECK(sibling::kIndexHeadDim == kIndexHeadDim);
    CHECK(sibling::kIndexTopk == kIndexTopk);
    CHECK(sibling::kWindow == kWindow);
    CHECK(sibling::kVocab == kVocab);
    CHECK(sibling::kHcMult == kHcMult);
    CHECK(sibling::kSinkhornIters == kSinkhornIters);
    CHECK(sibling::kCandidateBlockSize == kCandidateBlockSize);
    CHECK(sibling::kCandidateTopkBlocks == kCandidateTopkBlocks);
    CHECK(sibling::kMtpLayers == kMtpLayers);
    CHECK(sibling::kRmsEps == kRmsEps);
    CHECK(sibling::kRouteScale == kRouteScale);
    CHECK(sibling::kCompressRopeTheta == kCompressRopeTheta);
    CHECK(sibling::kRopeTheta == kRopeTheta);
    CHECK(sibling::kRopeFactor == kRopeFactor);
    CHECK(sibling::kOriginalSeqLen == kOriginalSeqLen);

    // Same CED classification: the sibling runs the parent's contour verbatim.
    for (int l = 0; l < kLayers; ++l) {
        CHECK(sibling::CompressRatio(l) == CompressRatio(l));
        CHECK(sibling::IsKvSource(l) == IsKvSource(l));
        CHECK(sibling::IsIndexSource(l) == IsIndexSource(l));
        CHECK(sibling::IsCandidateSource(l) == IsCandidateSource(l));
    }
}

TEST_CASE("deepseek-v41-1017 is a registered model of its own type",
          "[deepseek-v41][model][sibling]")
{
    // The factory is the only way an operator reaches a model; a kType that is
    // not registered resolves to nullptr and the CLI reports "that model does
    // not exist" — the exact failure two sibling trees collapse into when they
    // share one registry entry.
    lsxcommon::ModelInitConfig sibling_cfg;
    sibling_cfg.model_type = lsxcommon::deepseek_v41_flash1017::kType;
    auto sibling = lsxcommon::LsxModelFactory::Create(sibling_cfg);
    REQUIRE(sibling != nullptr);
    CHECK(sibling->Type() == "deepseek-v4.1-flash-1017");

    // ... and the parent row still resolves to its own class.
    lsxcommon::ModelInitConfig parent_cfg;
    parent_cfg.model_type = lsxcommon::deepseek_v41::kType;
    auto parent = lsxcommon::LsxModelFactory::Create(parent_cfg);
    REQUIRE(parent != nullptr);
    CHECK(parent->Type() == "deepseek-v4.1-flash");
    CHECK(parent->Type() != sibling->Type());
}
