// Frozen passport guard for Kimi-K3.1-Pro (KimiLinear contour + Gated DSA).
// Host-only, no CUDA deps.
//
// Values are read off the released checkpoint
// (/mnt/fast_storage/lsx_model/moonshotai/Kimi-K3.1-Pro: config.json + safetensors
// headers). Kimi-K3.1-Pro has the same text tower as Kimi-K3 and replaces the
// 24 full-attention Gated-MLA layers with Gated DSA (the same gated NoPE MLA
// plus a 32-head Lightning Indexer over a 2048-key budget). The load path
// cross-checks the same numbers, so a passport edit without a re-audit fails
// Load rather than silently decoding a different model.
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

#include "../libs/lsxcommon/src/kimi_moe_pro_constants.h"

using namespace lsxcommon::kimi_pro::frozen;

TEST_CASE("kimi-k3.1-pro passport: registry and format identity",
          "[kimi-pro][constants]")
{
    CHECK(std::string(kModelType) == "kimi-k3.1-pro");
    CHECK(std::string(kFormatTag) == "logestix-kimi-k31-pro-nvfp4");
    CHECK(std::string(kSourceModelType) == "kimi_k31");
    CHECK(kTokenizerKind == 2);
}

TEST_CASE("kimi-k3.1-pro passport: text tower geometry",
          "[kimi-pro][constants]")
{
    CHECK(kHidden == 7168);
    CHECK(kLayers == 93);
    CHECK(kDenseLayers == 1);
    CHECK(kAttnResBlock == 12);
    CHECK(kVocab == 163840);
    CHECK(kBaseVocab == 163584);
    CHECK(kBosId == 163584);
    CHECK(kEosId == 163586);
    CHECK(kStopIdAlt == 163585);
    CHECK(kPadId == 163839);
    CHECK(kRmsNormEps == 1e-5);
    CHECK(kExperts == 896);
    CHECK(kTopK == 16);
    CHECK(kMoeIntermediate == 3072);
    CHECK(kLatentDim == 3584);
    CHECK(kMlaHeads == 96);
    CHECK(kQLoraRank == 1536);
    CHECK(kKvLoraRank == 512);
    CHECK(kQkNopeHeadDim == 128);
    CHECK(kQkRopeHeadDim == 64);
    CHECK(kVHeadDim == 128);
    CHECK(kKdaHeads == 96);
    CHECK(kKdaHeadDim == 128);
    CHECK(kSituBeta == 4.0);
    CHECK(kSituLinearBeta == 25.0);
}

TEST_CASE("kimi-k3.1-pro passport: the DSA table is the MLA table",
          "[kimi-pro][constants]")
{
    std::set<int> mla(kMlaLayers.begin(), kMlaLayers.end());
    CHECK(mla.size() == static_cast<size_t>(kMlaLayerCount));
    CHECK(kMlaLayerCount == 24);
    CHECK(kKdaLayerCount == 69);
    CHECK(kDsaLayerCount == kMlaLayerCount);

    std::set<int> dsa(kDsaLayers.begin(), kDsaLayers.end());
    CHECK(dsa == mla);
    for (int l = 0; l < kLayers; ++l)
    {
        CHECK(IsDsaLayer(l) == IsMlaLayer(l));
        CHECK(IsMlaLayer(l) == (mla.count(l) != 0));
        CHECK(IsKdaLayer(l) == (mla.count(l) == 0));
    }
    // 0-based DSA = {3,7,...,91} (23) + the extra final layer 92.
    for (int l = 3; l <= 91; l += 4) CHECK(IsDsaLayer(l));
    CHECK(IsDsaLayer(92));
    CHECK_FALSE(IsDsaLayer(0));
    CHECK_FALSE(IsDsaLayer(1));
    CHECK_FALSE(IsDsaLayer(2));
}

TEST_CASE("kimi-k3.1-pro passport: Lightning Indexer geometry",
          "[kimi-pro][constants]")
{
    CHECK(kIndexHeads == 32);
    CHECK(kIndexHeadDim == 128);
    CHECK(kIndexHeads * kIndexHeadDim == 4096);  // wq_b rows
    CHECK(kIndexTopK == 2048);
    CHECK(kIndexRopeDim == 0);  // index_rope_dim: no rotary on q/k
    CHECK(kIndexQLora == kQLoraRank);
    CHECK(kIndexKvLen == kHidden);
    CHECK(kIndexWeightsRows == kIndexHeads);
    CHECK(kIndexNormHasBias);  // index_k_norm_type == "layernorm"
    CHECK(kIndexPerPositionKeys);  // the key is cached per position
}

TEST_CASE("kimi-k3.1-pro passport: scale constants are derived, not guessed",
          "[kimi-pro][constants]")
{
    // q carries 1/sqrt(index_head_dim) and the head weights 1/sqrt(index_heads).
    CHECK(kIndexQScale == static_cast<float>(1.0 / 11.313708498984761));  // 128^-0.5
    CHECK(kIndexWeightScale == static_cast<float>(1.0 / 5.656854249492381));  // 32^-0.5
}
