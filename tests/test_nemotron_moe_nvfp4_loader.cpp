// TDD — NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 loader-stage passport
// (Model Ingestion Protocol stage 3). Synthetic converted fixture only: never
// opens the vault. The frozen geometry gate is exercised for real (no synthetic
// bypass), so the passport really is the thing under test.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"

#include "test_nemotron_moe_nvfp4_fixture.h"

#include "lsxcommon/nemotron_moe_nvfp4_load.h"
#include "nemotron_moe_nvfp4_constants.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_nvfp4;

namespace {

// The loader-stage tests validate the REAL passport: force the synthetic
// escape hatch off whatever the surrounding environment says.
void DisableSyntheticGeometry()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0", 1);
#endif
}

} // namespace

TEST_CASE("nvfp4 index loads and matches the frozen passport",
          "[nemotron][nvfp4][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_ok");
    nemotronv4test::WriteNemotronNvfp4Index(root);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK(err.empty());
    CHECK(idx.num_layers == 52);
    CHECK(idx.num_experts == 128);
    CHECK(idx.num_experts_per_tok == 6);
    CHECK(idx.experts_no_gate);
    CHECK(idx.mtp_tensors == 15);
    CHECK(idx.attn_layer_idx == std::vector<int>{5, 12, 19, 26, 33, 42});
    CHECK(ValidateNemotronNvfp4Geometry(idx, err));
    CHECK(err.empty());
}

TEST_CASE("nvfp4 index carries BOTH stop ids from generation_config",
          "[nemotron][nvfp4][load]")
{
    // The vault's generation_config.json declares eos_token_id [2, 11]: id 11 is
    // a real second stop token. A loader that keeps only the first id silently
    // runs past the model's own stop and generates garbage after it - so the
    // LIST has to survive the round trip, and the scalar field has to stay
    // consistent with it for the sibling's single-id readers.
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_eos");
    nemotronv4test::WriteNemotronNvfp4Index(root);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK(idx.eos_token_ids == std::vector<int>{2, 11});
    CHECK(idx.eos_token_id == 2);  // the scalar stays the primary stop
    // The index also records that the FP8 KV-cache scales were consumed by the
    // converter rather than carried as weights.
    CHECK(idx.fp8_kv_scales_dropped == 12);
}

TEST_CASE("nvfp4 index accepts the scalar eos form the sibling writes",
          "[nemotron][nvfp4][load]")
{
    // The BF16 sibling writes a bare int. The shared reader has to keep serving
    // it, or a sibling tree would stop loading.
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_eos_scalar");
    nemotronv4test::WriteNemotronNvfp4Index(root, 2688, true,
                                            lsxcommon::kIndexLayout,
                                            "Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                                            "2");

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK(idx.eos_token_ids == std::vector<int>{2});
    CHECK(idx.eos_token_id == 2);
}

TEST_CASE("nvfp4 index rejects a missing index", "[nemotron][nvfp4][load]")
{
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_missing");
    NemotronIndex idx;
    std::string err;
    CHECK_FALSE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("nvfp4 index rejects an unknown format tag",
          "[nemotron][nvfp4][load]")
{
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_fmt");
    nemotronv4test::WriteNemotronNvfp4Index(root, /*hidden_override=*/2688,
                                            /*experts_no_gate=*/true,
                                            /*layout=*/1);  // wrong generation

    NemotronIndex idx;
    std::string err;
    CHECK_FALSE(LoadNemotronNvfp4Index(root.string(), idx, err));
    INFO("error: " << err);
    CHECK(err.find("layout") != std::string::npos);
}

TEST_CASE("nvfp4 loader refuses the BF16 sibling's converted tree",
          "[nemotron][nvfp4][load]")
{
    // Both distributions emit the SAME `logestix-nemotron-nvfp4` layout - that
    // is the point of reusing the loader. But they are two registered models
    // with two goldens each, so a tree converted from the BF16 distribution
    // must not be loadable under this model's kType just because the geometry
    // happens to match. The `model` field is what tells them apart.
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_sibling");
    nemotronv4test::WriteNemotronNvfp4Index(
        root, /*hidden_override=*/2688, /*experts_no_gate=*/true,
        /*layout=*/lsxcommon::kIndexLayout,
        /*model_name=*/"Nemotron-3.5-Lightning-30B-A3B");

    NemotronIndex idx;
    std::string err;
    CHECK_FALSE(LoadNemotronNvfp4Index(root.string(), idx, err));
    INFO("error: " << err);
    CHECK(err.find("model") != std::string::npos);
}

TEST_CASE("nvfp4 frozen geometry mismatch fails validation",
          "[nemotron][nvfp4][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_geo");
    nemotronv4test::WriteNemotronNvfp4Index(root, /*hidden_override=*/4096);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK_FALSE(ValidateNemotronNvfp4Geometry(idx, err));
    CHECK(err.find("hidden_size") != std::string::npos);
}

TEST_CASE("nvfp4 geometry requires the non-gated expert layout",
          "[nemotron][nvfp4][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_gate");
    nemotronv4test::WriteNemotronNvfp4Index(root, /*hidden_override=*/2688,
                                            /*experts_no_gate=*/false);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK_FALSE(ValidateNemotronNvfp4Geometry(idx, err));
    CHECK(err.find("experts_no_gate") != std::string::npos);
}

TEST_CASE("nvfp4 geometry rejects a second stop id that is not the vault's",
          "[nemotron][nvfp4][load]")
{
    // The eos set is part of the frozen passport for THIS distribution: the
    // vault declares [2, 11]. A tree claiming [2, 99] would run on past the
    // model's real stop, which is exactly the silent failure the list exists to
    // prevent - so it is checked, not merely carried.
    DisableSyntheticGeometry();
    const auto root = nemotronv4test::MakeTempDir("nmv4_idx_eos_bad");
    nemotronv4test::WriteNemotronNvfp4Index(root, 2688, true,
                                            lsxcommon::kIndexLayout,
                                            "Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                                            "[2, 99]");

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronNvfp4Index(root.string(), idx, err));
    CHECK_FALSE(ValidateNemotronNvfp4Geometry(idx, err));
    CHECK(err.find("eos") != std::string::npos);
}

TEST_CASE("nvfp4 passport is self-consistent", "[nemotron][nvfp4][load]")
{
    // The variant re-declares the sibling's geometry in its own namespace (the
    // independence guard requires a unique namespace per model). These
    // assertions are what make the duplication deliberate rather than a drift:
    // if the two passports ever disagree, this goes red.
    CHECK(frozen::kHidden == 2688);
    CHECK(frozen::kLayers == 52);
    CHECK(frozen::kVocab == 131072);
    CHECK(frozen::kMambaLayers + frozen::kMoeLayers + frozen::kAttnLayers ==
          frozen::kLayers);
    CHECK(frozen::kAttnLayerIdx[0] == 5);
    CHECK(frozen::kAttnLayerIdx[frozen::kAttnLayers - 1] == 42);
    CHECK(frozen::kNumRouted == 128);
    CHECK(frozen::kTopK == 6);
    CHECK(frozen::kMoeIntermediate == 1856);
    CHECK(frozen::kSharedIntermediate == 3712);
    CHECK(frozen::kHeads == 32);
    CHECK(frozen::kKVHeads == 2);
    CHECK(frozen::kHeadDim == 128);
    CHECK(frozen::kMambaInProj == 10304);
    // The stop set is part of the passport, not a runtime discovery.
    const std::vector<int> eos(frozen::kEosTokenIds,
                               frozen::kEosTokenIds + frozen::kNumEosTokenIds);
    CHECK(eos == std::vector<int>({2, 11}));
}