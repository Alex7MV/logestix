// Frozen passport guard for Kimi-K3 (KimiLinear: KDA + gated NoPE MLA +
// AttnRes + Stable LatentMoE + SiTU). Host-only, no CUDA deps.
//
// The values here are read off the released checkpoint
// (/mnt/fast_storage/huggingface/Kimi-K3: config.json + safetensors headers +
// modeling_kimi_linear.py); the load path cross-checks the same numbers, so a
// passport edit without a re-audit fails Load rather than silently decoding a
// different model.
#include <catch2/catch_test_macros.hpp>

#include <set>
#include <string>

#include "../libs/lsxcommon/src/kimi_moe_k3_constants.h"

using namespace lsxcommon::kimi_k3::frozen;

TEST_CASE("kimi-k3 passport: registry and format identity", "[kimi-k3][constants]")
{
    CHECK(std::string(kModelType) == "kimi-k3");
    CHECK(std::string(kFormatTag) == "logestix-kimi-k3-nvfp4");
    CHECK(kTokenizerKind == 2);
}

TEST_CASE("kimi-k3 passport: text tower geometry", "[kimi-k3][constants]")
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
}

TEST_CASE("kimi-k3 passport: hybrid layer tables are a partition", "[kimi-k3][constants]")
{
    std::set<int> mla(kMlaLayers.begin(), kMlaLayers.end());
    CHECK(mla.size() == static_cast<size_t>(kMlaLayerCount));
    CHECK(kMlaLayerCount == 24);
    CHECK(kKdaLayerCount == 69);
    CHECK(kMlaLayerCount + kKdaLayerCount == kLayers);

    for (int l = 0; l < kLayers; ++l)
    {
        const bool is_mla = mla.count(l) != 0;
        CHECK(IsMlaLayer(l) == is_mla);
        CHECK(IsKdaLayer(l) == !is_mla);
    }

    // 0-based MLA = {3,7,11,...,91} (23) + the extra final layer 92.
    for (int l = 3; l <= 91; l += 4) CHECK(IsMlaLayer(l));
    CHECK(IsMlaLayer(92));
    CHECK_FALSE(IsMlaLayer(0));
    CHECK_FALSE(IsMlaLayer(1));
    CHECK_FALSE(IsMlaLayer(2));
    CHECK(kAttnResBlock % kFullAttnInterval == 0);
}

TEST_CASE("kimi-k3 passport: gated NoPE MLA", "[kimi-k3][constants]")
{
    CHECK(kMlaHeads == 96);
    CHECK(kMlaKVHeads == 96);
    CHECK(kQLoraRank == 1536);
    CHECK(kKvLoraRank == 512);
    CHECK(kQkNopeHeadDim == 128);
    CHECK(kQkRopeHeadDim == 64);
    CHECK(kVHeadDim == 128);
    CHECK(kQHeadDim == 192);
    CHECK(kMlaLatentWidth == 576);  // kv_lora + qk_rope
    CHECK(kMlaOutputGate);
    CHECK_FALSE(kMlaUsesRotary);    // mla_use_nope: the 64 dims stay unrotated
    CHECK(kMlaScaleDenom == 192);
}

TEST_CASE("kimi-k3 passport: KDA", "[kimi-k3][constants]")
{
    CHECK(kKdaHeads == 96);
    CHECK(kKdaHeadDim == 128);
    CHECK(kKdaConvKernel == 4);
    CHECK(kKdaGateLowerBound == -5.0);
    CHECK(kKdaFullRankGate);
    CHECK(kKdaA_logPerChannel);  // A_log ships with head_dim entries
}

TEST_CASE("kimi-k3 passport: Stable LatentMoE", "[kimi-k3][constants]")
{
    CHECK(kExperts == 896);
    CHECK(kTopK == 16);
    CHECK(kMoeIntermediate == 3072);
    CHECK(kLatentDim == 3584);
    CHECK(kLatentMoeNorm);
    CHECK(kSharedExperts == 2);
    CHECK(kSharedIntermediate == 6144);
    CHECK(kMoeRenormalize);
    CHECK(kRouteScale == 1.0);
    CHECK(kExpertGroup == 1);
    CHECK(kTopkGroup == 1);
}

TEST_CASE("kimi-k3 passport: SiTU-GLU", "[kimi-k3][constants]")
{
    CHECK(kSituBeta == 4.0);
    CHECK(kSituLinearBeta == 25.0);
}
