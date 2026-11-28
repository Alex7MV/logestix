// gemma-4-12B-it: frozen-passport + registration test (Model Ingestion
// Protocol step 3). Host-only: no CUDA context, no converted checkpoint, no
// weights — this is the stage that decides whether the engine would accept the
// right tree and reject the wrong one, and it must be cheap enough to run on
// every diff.
//
// The 12B is the SMALL sibling of gemma-4-31B-it: same decoder composition
// (5:1 sliding/global attention, per-head QK-norm, K tied to V on the global
// layers, four norms + layer_scalar per layer, GeGLU, tied embed/lm_head,
// softcapped logits) with a different width/depth/head layout. Both are served
// by ONE converter and ONE kernel set, so the only thing that keeps them apart
// is the passport each class validates the converted tree against.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "test_manifest_query_root.h"

#include "lsxcommon/gemma_dense_12b_constants.h"
#include "lsxcommon/gemma_dense_31b_constants.h"
#include "lsxcommon/gemma_dense_constants.h"
#include "lsxcommon/gemma_dense_convert.h"
#include "lsxcommon/model_factory.h"

namespace {

// A `GemmaDenseTopology` carrying exactly the numbers this checkpoint's
// config.json declares. Kept in the test on purpose: if the passport and this
// table ever disagree, the passport is wrong, not the test.
lsxcommon::gemma_dense::GemmaDenseTopology Topology12B()
{
    using namespace lsxcommon::gemma_dense_12b::frozen;
    lsxcommon::gemma_dense::GemmaDenseTopology t;
    t.num_layers = kNumLayers;
    t.hidden_size = kHiddenSize;
    t.intermediate_size = kIntermediateSize;
    t.vocab_size = kVocabSize;
    t.num_attention_heads = kNumHeads;
    t.num_key_value_heads = kNumKVHeads;
    t.num_global_key_value_heads = kNumGlobalKVHeads;
    t.head_dim = kHeadDim;
    t.global_head_dim = kGlobalHeadDim;
    t.sliding_window = kSlidingWindow;
    t.rms_norm_eps = static_cast<float>(kRmsNormEps);
    for (int l = 0; l < kNumLayers; ++l)
        t.layer_types.push_back((l + 1) % kFullLayerPeriod == 0
                                    ? "full_attention"
                                    : "sliding_attention");
    return t;
}

} // namespace

TEST_CASE("gemma-4-12B passport: geometry invariants are what the checkpoint says",
          "[gemma_dense_12b][model]") {
    using namespace lsxcommon::gemma_dense_12b::frozen;
    // config.json (text_config), 2026-11-28.
    CHECK(kHiddenSize == 3840);
    CHECK(kNumLayers == 48);
    CHECK(kNumHeads == 16);
    CHECK(kNumKVHeads == 8);
    CHECK(kNumGlobalKVHeads == 1);
    CHECK(kHeadDim == 256);
    CHECK(kGlobalHeadDim == 512);
    CHECK(kIntermediateSize == 15360);
    CHECK(kVocabSize == 262144);
    CHECK(kSlidingWindow == 1024);
    CHECK(kFullLayerPeriod == 6);
    CHECK(kNumFullLayers == 8);        // 48 / 6
    CHECK(kNumSlidingLayers == 40);
    CHECK(kPartialRotaryFactor == Catch::Approx(0.25));
    CHECK(kFullRotaryAngles == 64);    // 0.25 * 512 / 2
    CHECK(kBosId == 2);
    CHECK(kPadId == 0);
    // A 12B is not a 31B: the fixture would still "work" (same shapes, no
    // assertion) if a sibling constant leaked into this namespace.
    CHECK(kHiddenSize != lsxcommon::gemma_dense::frozen::kHiddenSize);
    CHECK(kNumLayers != lsxcommon::gemma_dense::frozen::kNumLayers);
}

TEST_CASE("gemma-4-12B passport matches its own tree and rejects the 31B's",
          "[gemma_dense_12b][model]") {
    using lsxcommon::gemma_dense::GemmaDenseGeometryMismatch;
    const auto top12 = Topology12B();

    // Its own passport: clean.
    CHECK(GemmaDenseGeometryMismatch(
              lsxcommon::gemma_dense_12b::frozen::kFrozen, top12)
              .empty());

    // The 31B passport against a 12B tree: every width/depth field is wrong,
    // and all of them are reported (not just the first).
    const std::string why = GemmaDenseGeometryMismatch(
        lsxcommon::gemma_dense::frozen::kFrozen, top12);
    CHECK_FALSE(why.empty());
    CHECK(why.find("hidden_size") != std::string::npos);
    CHECK(why.find("num_layers") != std::string::npos);
    CHECK(why.find("intermediate_size") != std::string::npos);

    // One field is enough to refuse a tree.
    auto drift = top12;
    drift.sliding_window = 4096;
    const std::string drift_why = GemmaDenseGeometryMismatch(
        lsxcommon::gemma_dense_12b::frozen::kFrozen, drift);
    CHECK_FALSE(drift_why.empty());
    CHECK(drift_why.find("sliding_window") != std::string::npos);
}

TEST_CASE("embed scale is the bf16-rounded sqrt(hidden), per checkpoint",
          "[gemma_dense_12b][model]") {
    using lsxcommon::gemma_dense::GemmaDenseEmbedScale;
    // Gemma4TextScaledWordEmbedding casts sqrt(hidden_size) to the weight dtype
    // first, so the gather multiplies by the bf16 value: 62.0 for 3840 (not
    // 61.9677) and 73.5 for 5376 (not 73.3212). A shared 31B constant here is
    // exactly how the 12B would scale every embedding by 73.5/62 = 1.19.
    CHECK(GemmaDenseEmbedScale(3840) == 62.0f);
    CHECK(GemmaDenseEmbedScale(5376) == 73.5f);
    CHECK(GemmaDenseEmbedScale(3840) != Catch::Approx(61.96774f).epsilon(1e-4));
}

TEST_CASE("gemma-4-12b-it is reachable through the model factory",
          "[gemma_dense_12b][model]") {
    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_type = "gemma-4-12b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "gemma-4-12b-it");

    // The sibling still resolves to its own class, not to this one.
    lsxcommon::ModelInitConfig cfg31;
    cfg31.model_type = "gemma-4-31b-it";
    auto model31 = lsxcommon::LsxModelFactory::Create(cfg31);
    REQUIRE(model31 != nullptr);
    CHECK(model31->Type() == "gemma-4-31b-it");
    CHECK(model31->Type() != model->Type());
}

// The real config.json is ONE light file: reading it is allowed (and it is the
// only place the passport's numbers can be cross-checked against the actual
// checkpoint). Absence SKIPs with the directory it looked in.
TEST_CASE("gemma-4-12B passport matches the real checkpoint's config.json",
          "[gemma_dense_12b][model][real]") {
    manifest_query::RequireSourceRoot();
    const auto dir = manifest_query::SourceDirFor("gemma-4-12b-it");
    if (dir.empty()) SKIP("gemma-4-12b-it has no manifest row yet");
    const auto cfg = dir / "config.json";
    if (!std::filesystem::exists(cfg))
        SKIP("no checkpoint config at " + cfg.string());

    auto top = lsxcommon::gemma_dense::load_gemma_dense_topology(cfg.string());
    INFO(top.error_msg);
    REQUIRE(top.error_msg.empty());
    CHECK(top.model_name.empty());  // config.json has no "model" key
    const std::string why = lsxcommon::gemma_dense::GemmaDenseGeometryMismatch(
        lsxcommon::gemma_dense_12b::frozen::kFrozen, top);
    INFO(why);
    CHECK(why.empty());
    CHECK(top.eos_ids == std::vector<int>({1, 106, 50}));
    CHECK_FALSE(lsxcommon::gemma_dense::GemmaDenseGeometryMismatch(
                    lsxcommon::gemma_dense::frozen::kFrozen, top)
                    .empty());
}

// The residual plane is declared PER MODEL in the registry, because its cost is
// per model and the two members sit on opposite sides of the card's budget:
//
//   gemma-4-12b-it  4-bit 6.3 GB -> +5.7 GB = 12.0 GB   (fits; FSI 1/3 -> 2/3)
//   gemma-4-31b-it  4-bit 16.5 GiB -> +17 GiB = 33.7 GiB (does not fit 32 GB)
//
// A shared converter constant cannot express that, and getting it wrong is
// expensive in one direction only: a 31B tree with the plane cannot be loaded at
// all. The declaration is therefore a manifest field that the generated row
// carries, and this test pins BOTH directions so neither can drift.
TEST_CASE("the residual plane is declared per model, not per converter",
          "[gemma_dense_12b][model]") {
    const auto *r12 = manifest_query::FindModel("gemma-4-12b-it");
    const auto *r31 = manifest_query::FindModel("gemma-4-31b-it");
    REQUIRE(r12 != nullptr);
    REQUIRE(r31 != nullptr);
    // The 12B NEEDS it: FSI-01 is a near-tie that the 4-bit body flips, and
    // 12.0 GB fits the card.
    CHECK(r12->convert_residual_plane);
    // The 31B must NOT get one by default: 33.7 GiB does not fit, so a fresh
    // conversion would produce an unloadable tree.
    CHECK_FALSE(r31->convert_residual_plane);
}
