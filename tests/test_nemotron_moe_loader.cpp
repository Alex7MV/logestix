// TDD — Nemotron-3.5 loader-stage passport (Model Ingestion Protocol step 1
// output -> residency). Synthetic converted fixture only: never opens the
// vault. The frozen geometry gate is exercised for real (no synthetic bypass).
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_fixture.h"

#include "lsxcommon/nemotron_moe_load.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace lsxcommon::nemotron_moe;

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

TEST_CASE("nemotron index loads and matches the frozen passport",
          "[nemotron][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotrontest::MakeTempDir("nm_idx_ok");
    nemotrontest::WriteNemotronIndex(root);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronIndex(root.string(), idx, err));
    CHECK(err.empty());
    CHECK(idx.num_layers == 52);
    CHECK(idx.num_experts == 128);
    CHECK(idx.num_experts_per_tok == 6);
    CHECK(idx.experts_no_gate);
    CHECK(idx.mtp_tensors == 128);
    CHECK(idx.attn_layer_idx == std::vector<int>{5, 12, 19, 26, 33, 42});
    CHECK(ValidateNemotronGeometry(idx, err));
    CHECK(err.empty());
}

TEST_CASE("nemotron index rejects a missing index", "[nemotron][load]")
{
    const auto root = nemotrontest::MakeTempDir("nm_idx_missing");
    NemotronIndex idx;
    std::string err;
    CHECK_FALSE(LoadNemotronIndex(root.string(), idx, err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("nemotron index rejects an unknown format tag", "[nemotron][load]")
{
    const auto root = nemotrontest::MakeTempDir("nm_idx_fmt");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/2688,
                                     /*experts_no_gate=*/true,
                                     /*format_tag=*/"logestix-glm");

    NemotronIndex idx;
    std::string err;
    CHECK_FALSE(LoadNemotronIndex(root.string(), idx, err));
    INFO("error: " << err);
    CHECK(err.find("format") != std::string::npos);
}

TEST_CASE("nemotron frozen geometry mismatch fails validation",
          "[nemotron][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotrontest::MakeTempDir("nm_idx_geo");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/4096);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronIndex(root.string(), idx, err));
    CHECK_FALSE(ValidateNemotronGeometry(idx, err));
    CHECK(err.find("hidden_size") != std::string::npos);
}

TEST_CASE("nemotron geometry requires the non-gated expert layout",
          "[nemotron][load]")
{
    DisableSyntheticGeometry();
    const auto root = nemotrontest::MakeTempDir("nm_idx_gate");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/2688,
                                     /*experts_no_gate=*/false);

    NemotronIndex idx;
    std::string err;
    REQUIRE(LoadNemotronIndex(root.string(), idx, err));
    CHECK_FALSE(ValidateNemotronGeometry(idx, err));
    CHECK(err.find("experts_no_gate") != std::string::npos);
}
