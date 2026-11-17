// Frozen passport guard for Inkling (base). Host-only, no CUDA deps.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/huggingface/Inkling: config.json, tokenizer_config.json
// added_tokens and all 108 safetensors headers, audited 2026-11-17) and
// cross-checked against the reference semantics in transformers 5.15.1
// `models/inkling/modeling_inkling.py` + `conversion_mapping.py`
// ("inkling_mm_model"). This is a DIFFERENT architecture from inkling2
// (Inkling-2-Small): no L2 clips, no attention sinks/gates, no AttnRes depth
// mixing, no expert softcaps, scale 1/head_dim, static convs without a
// normalizer, plain residual, 2 shared-sink experts. Load re-derives the same
// numbers from the converted index, so a passport edit without a re-audit
// fails Load instead of silently decoding a different model.
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

#include "../libs/lsxcommon/src/inkling_moe_base_constants.h"

using namespace lsxcommon::inkling::base::frozen;

TEST_CASE("inkling base passport: registry and format identity",
          "[inkling][constants]")
{
    CHECK(std::string(kModelType) == "inkling");
    CHECK(std::string(kFormatTag) == "logestix-inkling-nvfp4");
    CHECK(std::string(kSourceModelType) == "inkling_mm_model");
    CHECK(kTokenizerKind == 2);
}

TEST_CASE("inkling base passport: text tower geometry", "[inkling][constants]")
{
    CHECK(kHidden == 6144);
    CHECK(kLayers == 66);
    CHECK(kHeads == 64);
    CHECK(kHeadDim == 128);
    CHECK(kKvHeadsFull == 8);
    CHECK(kKvHeadsSliding == 16);
    CHECK(kVocab == 201024);
    CHECK(kBaseVocab == 200058);
    CHECK(kRmsNormEps == 1e-6);
    // Inkling (base) scales by 1/head_dim because q/k are per-head RMS-normed;
    // inkling2 uses head_dim**-0.5. A copy-paste from the sibling port silently
    // rescales every attention logit by sqrt(128).
    CHECK(kAttnScaleDenom == 128.0);
    CHECK(kConvKernel == 4);
}

TEST_CASE("inkling base passport: the 11 full layers are the (layer+1)%6 ids",
          "[inkling][constants]")
{
    CHECK(kFullLayerCount == 11);
    CHECK(kSlidingLayerCount == 55);
    CHECK(kFullLayers ==
          std::array<int, 11>{5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65});
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

TEST_CASE("inkling base passport: attention constants", "[inkling][constants]")
{
    CHECK(kSlidingWindow == 512);
    CHECK(kRelExtent == 1024);
    CHECK(kDRel == 16);
    CHECK(kLogScaleFloor == 128000);
    CHECK(kLogScaleAlpha == 0.1);
}

TEST_CASE("inkling base passport: dense MLP prefix and routed MoE",
          "[inkling][constants]")
{
    // Layers 0..1 are dense SwiGLU (intermediate 24576, output x global_scale);
    // layers 2..65 are sparse. `dense_mlp_idx` in config.json is 2.
    CHECK(kDenseMlpLayers == 2);
    CHECK(kDenseIntermediate == 24576);
    CHECK(kExperts == 256);
    CHECK(kTopK == 6);
    CHECK(kMoeIntermediate == 3072);
    CHECK(kSharedExperts == 2);
    CHECK(kRouteScale == 8.0);
    // The router renormalises over the 6 selected routed + 2 shared logits
    // jointly (softmax of logsigmoid), then scales by route_scale and the
    // gate's own global_scale. The selection bias steers only the top-6.
    CHECK(kRouterRows == kExperts + kSharedExperts);
}

TEST_CASE("inkling base passport: muP head, no softcap", "[inkling][constants]")
{
    // hidden / logits_mup_width_multiplier BEFORE lm_head; the checkpoint's
    // final_logit_softcapping is null.
    CHECK(kLogitsMupDiv == 24.0);
    CHECK(kQuantBlock == 16);
}

TEST_CASE("inkling base passport: special ids and chat markers",
          "[inkling][constants]")
{
    CHECK(kBosId == -1);
    CHECK(kPadId == -1);
    CHECK(kEosId == 200006);
    CHECK(kStopIdAlt == 199999);
    CHECK(kMarkerUser == 200000);
    CHECK(kMarkerModel == 200001);
    CHECK(kMarkerSystem == 200002);
    CHECK(kMarkerTool == 200003);
    CHECK(kMarkerText == 200004);
    CHECK(kMarkerEndSampling == 200006);
    CHECK(kMarkerThinking == 200008);
    CHECK(kMarkerEndMessage == 200010);
    CHECK(kMarkerXml == 200024);
    CHECK(kMarkerInvokeJson == 200049);

    const std::set<int> markers = {
        kMarkerUser,   kMarkerModel,       kMarkerSystem,
        kMarkerTool,   kMarkerText,        kMarkerEndSampling,
        kMarkerThinking, kMarkerEndMessage, kMarkerXml,
        kMarkerInvokeJson};
    CHECK(markers.size() == 10);
    for (int m : markers) CHECK((m >= kAddedTokenBase && m < kBaseVocab));
}