// Frozen passport guard for Muse-Glimmer-30B. Host-only, no CUDA deps.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/lsx_model/meta-models/Muse-Glimmer-30B: config.json text_config,
// tokenizer.json added_tokens and both safetensors headers, audited
// 2026-11-18) and cross-checked against the reference semantics in
// transformers 5.15.1 `models/muse_glimmer/modular_muse_glimmer.py`
// (MuseGlimmerTextAttention(AfmoeAttention), Gemma2DecoderLayer sandwich,
// MuseGlimmerTextNormedEmbedding). This is a NEW dense family: gated
// attention + NoPE full layers + centered norms + weightless embed norm.
// Load re-derives the same numbers from the converted index, so a passport
// edit without a re-audit fails Load instead of silently decoding a
// different model.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <set>
#include <string>

#include "lsxcommon/muse_dense_constants.h"

using namespace lsxcommon::muse_dense::frozen;

TEST_CASE("muse dense passport: registry and format identity",
          "[muse_dense][constants]")
{
    CHECK(std::string(kModelType) == "muse-glimmer-30b");
    CHECK(std::string(kFormatTag) == "logestix-muse-nvfp4");
    CHECK(std::string(kSourceModelType) == "muse_glimmer");
    CHECK(kTokenizerKind == 3);
}

TEST_CASE("muse dense passport: text tower geometry", "[muse_dense][constants]")
{
    CHECK(kHiddenSize == 6656);
    CHECK(kNumLayers == 52);
    CHECK(kIntermediateSize == 19968);
    CHECK(kVocabSize == 202048);
    CHECK(kBaseVocabSize == 200000);
    CHECK(kNumHeads == 32);
    CHECK(kNumKVHeads == 2);
    CHECK(kHeadDim == 128);
    CHECK(kNumHeads % kNumKVHeads == 0);
    // q/o rows are heads*head_dim, NOT hidden_size (4096 != 6656).
    CHECK(kQRows == 4096);
    CHECK(kKvRows == 256);
}

TEST_CASE("muse dense passport: the 13 full layers are the l%4==3 ids",
          "[muse_dense][constants]")
{
    CHECK(kNumFullLayers == 13);
    CHECK(kNumSlidingLayers == 39);
    int full = 0;
    for (int l = 0; l < kNumLayers; ++l)
    {
        const bool expect_full = (l % 4 == 3);
        CHECK(IsFullLayer(l) == expect_full);
        full += IsFullLayer(l) ? 1 : 0;
    }
    CHECK(full == kNumFullLayers);
    CHECK(IsFullLayer(3));
    CHECK(IsFullLayer(7));
    CHECK(IsFullLayer(51));
    CHECK_FALSE(IsFullLayer(0));
    CHECK_FALSE(IsFullLayer(50));
}

TEST_CASE("muse dense passport: RoPE on sliding layers only (NoPE full)",
          "[muse_dense][constants]")
{
    CHECK(kSlidingWindow == 2048);
    CHECK(kRopeTheta == 500000.0);
    // layer_rope_theta[i] == 0 disables rotary on the full-attention layers;
    // the sliding ones carry the global theta. A uniform-RoPE port decodes
    // garbage past the first positions.
    CHECK(RopeThetaForLayer(0) == 500000.0);
    CHECK(RopeThetaForLayer(2) == 500000.0);
    CHECK(RopeThetaForLayer(3) == 0.0);
    CHECK(RopeThetaForLayer(51) == 0.0);
    CHECK(WindowForLayer(0) == 2048);
    CHECK(WindowForLayer(3) == 0);
}

TEST_CASE("muse dense passport: attention scale and gate",
          "[muse_dense][constants]")
{
    // Weightless per-head QK RMSNorm, then Q *= 3.87, then the STANDARD
    // 1/sqrt(head_dim) score scale (unlike gemma's scale-free attention).
    CHECK(kQkScaleFactor == Catch::Approx(3.87f));
    CHECK(kAttentionScale == Catch::Approx(1.0f / std::sqrt(128.0f)).epsilon(1e-7));
    CHECK(kRmsNormEps == 1e-5);
    // post_attention/post_feedforward norms use post_norm_eps, NOT rms_norm_eps.
    CHECK(kPostNormEps == 1e-8);
}

TEST_CASE("muse dense passport: softcapped logits with output multiplier",
          "[muse_dense][constants]")
{
    // logits = cap * tanh(raw * multiplier / cap); the multiplier is
    // 1/sqrt(hidden/256). Dropping it ships ~5.1x hot logits.
    CHECK(kFinalLogitSoftcap == 20.0f);
    CHECK(kOutputMultiplier == Catch::Approx(0.19611613513818404f).epsilon(1e-9));
    CHECK(kOutputMultiplier ==
          Catch::Approx(1.0f / std::sqrt(6656.0f / 256.0f)).epsilon(1e-7));
}

TEST_CASE("muse dense passport: special ids and ATEM markers",
          "[muse_dense][constants]")
{
    CHECK(kBosId == 200000);
    CHECK(kEosIdText == 200001);
    CHECK(kEosIdTurn == 200008);
    CHECK(kPadId == 200018);
    CHECK(kMarkerEom == 200007);
    CHECK(kMarkerStart == 200022);
    CHECK(kMarkerMessage == 200023);
    CHECK(kMarkerImage == 200090);
    CHECK(kMarkerVideo == 200091);
    CHECK(kMarkerPatch == 200092);

    const std::set<int> markers = {kBosId, kEosIdText, kEosIdTurn, kPadId,
                                   kMarkerEom, kMarkerStart, kMarkerMessage};
    CHECK(markers.size() == 7);
    for (int m : markers) CHECK((m >= kBaseVocabSize && m < kVocabSize));
}