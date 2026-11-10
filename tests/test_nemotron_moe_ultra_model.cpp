// TDD — Nemotron-3-Ultra-550B model leaf: session sizing, factory resolution,
// load/teardown on the synthetic tree, and a generation that actually runs.
//
// The session case is the one that catches a whole class of silent defects: every
// width the decode loop indexes comes from here, so an off-by-one in the latent
// scratch or the KV stride shows up as a wrong-size allocation rather than as
// subtly wrong logits 200 tokens later.
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_ultra_fixture.h"

#include "lsxcommon/nemotron_moe_ultra_convert.h"
#include "lsxcommon/nemotron_moe_ultra_expert_cache.h"
#include "lsxcommon/nemotron_moe_ultra_inference.h"
#include "lsxcommon/nemotron_moe_ultra_load.h"
#include "nemotron_moe_ultra_model.h"

#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <cmath>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_ultra;

namespace {

struct EnableSyntheticGeometry {
    EnableSyntheticGeometry()
    {
#ifdef _WIN32
        _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
#else
        setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
#endif
    }
} g_enable_synthetic;

bool HaveCuda()
{
    int devs = 0;
    return cudaGetDeviceCount(&devs) == cudaSuccess && devs > 0;
}

fs::path ConvertedFixture(const std::string &tag)
{
    const auto src = ultratest::MakeTempDir(tag + "_src");
    ultratest::WriteUltraCheckpoint(src);
    const auto dst = ultratest::MakeTempDir(tag + "_dst");
    REQUIRE(ConvertNemotronUltraModel(src.string(), dst.string()).ok());
    return dst;
}

} // namespace

TEST_CASE("ultra session: widths come from the index and the scratch is sized for the latent path",
          "[ultra][model]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_sess");
    UltraIndex idx;
    std::string err;
    REQUIRE(LoadUltraIndex(tree.string(), idx, err));

    UltraSession s;
    REQUIRE(AllocUltraSession(idx, /*max_seq=*/64, s, err));
    CHECK(s.H == ultratest::kTH);
    CHECK(s.mamba_layers == 1);
    CHECK(s.attn_layers == 1);
    CHECK(s.moe_latent == ultratest::kTLatent);
    CHECK(s.moe_intermediate == ultratest::kTI);
    CHECK(s.shared_intermediate == ultratest::kTShared);
    CHECK(s.n_group == ultratest::kTGroup);
    CHECK(s.topk_group == ultratest::kTTopkGroup);
    CHECK(s.routed_scale == 5.0f);
    CHECK(s.d_inner == ultratest::kTMH * ultratest::kTMHD);
    CHECK(s.conv_dim == ultratest::kTConvDim);
    REQUIRE(s.mamba_state.size() == 1);
    REQUIRE(s.conv_state.size() == 1);
    REQUIRE(s.k_cache.size() == 1);
    REQUIRE(s.topk_i != nullptr);

    // Reset zeroes the persistent state; the buffers stay allocated.
    ResetUltraSession(s, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    FreeUltraSession(s);
    CHECK(s.h == nullptr);
    CHECK(s.proj16 == nullptr);
    CHECK(s.mamba_state.empty());
    CHECK(s.k_cache.empty());
}

TEST_CASE("ultra pipeline: one token through 108-layer-shaped walk yields finite logits",
          "[ultra][model]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_fwd");
    UltraIndex idx;
    std::string err;
    REQUIRE(LoadUltraIndex(tree.string(), idx, err));
    REQUIRE(ValidateUltraGeometry(idx, err));

    UltraWeights w;
    REQUIRE(LoadUltraWeights(tree.string(), idx, w, nullptr, err));

    UltraExpertCache cache;
    LoadUltraExpertStore(tree.string(), idx.num_layers, ultratest::kTExperts,
                         cache.store);
    cache.experts_per_layer = ultratest::kTExperts;
    cache.moe_index_of_layer.assign(static_cast<std::size_t>(idx.num_layers), -1);
    for (int l = 0; l < idx.num_layers; ++l)
        if (idx.layer_types[static_cast<std::size_t>(l)] == "moe") {
            cache.moe_index_of_layer[static_cast<std::size_t>(l)] = cache.moe_layers++;
        }
    cache.layout = PlanUltraExpertSlotFromStore(cache.store);
    REQUIRE(cache.layout.ok);
    REQUIRE(AllocateUltraExpertCache(cache, ultratest::kTExperts, err));

    UltraSession s;
    REQUIRE(AllocUltraSession(idx, /*max_seq=*/32, s, err));
    ResetUltraSession(s, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Prefill one token, then decode one: both paths must run (the dt clamp is
    // the only thing that differs) and produce finite logits.
    for (int step = 0; step < 2; ++step) {
        const cudaError_t e = UltraForwardToken(idx, w, s, cache, /*token=*/1 + step,
                                               /*pos=*/step, /*prefill=*/step == 0,
                                               nullptr);
        REQUIRE(e == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        std::vector<float> logits(static_cast<std::size_t>(idx.vocab_size));
        REQUIRE(cudaMemcpy(logits.data(), s.head_logits,
                           logits.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        bool any_finite = false;
        for (float v : logits) {
            REQUIRE_FALSE(std::isnan(v));
            if (std::isfinite(v)) any_finite = true;
        }
        CHECK(any_finite);
        // A dead pipeline (all zeros, or one constant) would pass the finite
        // check above, so require spread between the extremes.
        const auto [lo, hi] = std::minmax_element(logits.begin(), logits.end());
        CHECK(*hi > *lo);
    }

    FreeUltraSession(s);
    FreeUltraExpertCache(cache);
    w.Free();
}

using lsxcommon::LsxModelFactory;
using lsxcommon::ModelInitConfig;
using lsxcommon::ModelRequest;
using lsxcommon::NemotronUltra550BModel;

TEST_CASE("ultra factory resolves nemotron-ultra-550b-a55b", "[ultra][model]")
{
    lsxcommon::EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = NemotronUltra550BModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == NemotronUltra550BModel::kType);
}

TEST_CASE("ultra Load+Infer on the synthetic tree generate two tokens",
          "[ultra][model]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const auto tree = ConvertedFixture("nu_infer");
    ModelInitConfig cfg;
    cfg.model_path = tree.string();
    cfg.model_type = NemotronUltra550BModel::kType;
    auto model = std::make_unique<NemotronUltra550BModel>(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    CHECK(model->Type() == NemotronUltra550BModel::kType);

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.max_gen_tokens = 2;
    req.ignore_eos = true;
    const auto r = model->Infer(req);
    INFO("infer error: " << r.error_msg);
    REQUIRE(r.ok);
    CHECK(r.num_tokens == 2);
    CHECK(r.gen_tokens.size() == 2);
    // Logits are the fixture's vocab, taken from the index rather than assumed.
    CHECK(r.final_logits.size() == static_cast<std::size_t>(ultratest::kTVocab));
    model->Unload();
}
