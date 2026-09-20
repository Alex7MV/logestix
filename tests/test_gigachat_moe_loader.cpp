// TDD — GigaChat3.5 loader-stage passport (converter output -> residency).
// Synthetic index only: never opens the vault. The frozen geometry gate is
// exercised for real (no synthetic bypass).
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "lsxcommon/gigachat_moe_load.h"

#include <cstdlib>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat;

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

TEST_CASE("gigachat index loads and matches the frozen passport",
          "[gigachat][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachattest::MakeTempDir("gc_idx_ok");
    gigachattest::WriteGigaChatIndex(root);

    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));
    CHECK(err.empty());
    CHECK(idx.num_layers == 40);
    CHECK(idx.hidden_size == 7168);
    CHECK(idx.num_experts == 256);
    CHECK(idx.num_experts_per_tok == 8);
    CHECK(idx.full_attention_layers ==
          std::vector<int>{3, 7, 11, 15, 19, 23, 27, 31, 35, 39});
    CHECK(ValidateGigaChatGeometry(idx, err));
    CHECK(err.empty());
}

TEST_CASE("gigachat index rejects a missing index", "[gigachat][load]")
{
    const auto root = gigachattest::MakeTempDir("gc_idx_missing");
    GigaChatIndex idx;
    std::string err;
    CHECK_FALSE(LoadGigaChatIndex(root.string(), idx, err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("gigachat index rejects an unknown format tag", "[gigachat][load]")
{
    const auto root = gigachattest::MakeTempDir("gc_idx_fmt");
    gigachattest::WriteGigaChatIndex(root, /*hidden_override=*/7168,
                                     /*format_tag=*/"logestix-glm");
    GigaChatIndex idx;
    std::string err;
    CHECK_FALSE(LoadGigaChatIndex(root.string(), idx, err));
    CHECK(err.find("format") != std::string::npos);
}

TEST_CASE("gigachat frozen geometry mismatch fails validation",
          "[gigachat][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachattest::MakeTempDir("gc_idx_geo");
    gigachattest::WriteGigaChatIndex(root, /*hidden_override=*/4096);
    GigaChatIndex idx;
    std::string err;
    REQUIRE(LoadGigaChatIndex(root.string(), idx, err));
    CHECK_FALSE(ValidateGigaChatGeometry(idx, err));
    CHECK(err.find("mismatch") != std::string::npos);
}
