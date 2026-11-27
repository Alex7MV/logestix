// TDD — GigaChat4-37B-A4B loader-stage passport (converter output -> residency).
// Synthetic index only: never opens the vault. The frozen geometry gate is
// exercised for real (no synthetic bypass).
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"

#include "lsxcommon/gigachat_moe_37b_load.h"

#include <cstdlib>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat4;

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

TEST_CASE("gigachat4 index loads and matches the frozen passport",
          "[gigachat4][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc4_idx_ok");
    gigachat4test::WriteGigaChat4Index(root);

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4Index(root.string(), idx, err));
    CHECK(err.empty());
    CHECK(idx.num_layers == 32);
    CHECK(idx.hidden_size == 2560);
    CHECK(idx.num_experts == 192);
    CHECK(idx.num_experts_per_tok == 8);
    CHECK(idx.index_heads == 64);
    CHECK(idx.index_head_dim == 128);
    CHECK(idx.index_topk == 2048);
    CHECK(idx.logit_softcap == 30.0);
    CHECK(idx.full_attention_layers ==
          std::vector<int>{3, 7, 11, 15, 19, 23, 27, 31});
    CHECK(ValidateGigaChat4Geometry(idx, err));
    CHECK(err.empty());
}

TEST_CASE("gigachat4 index rejects a missing index", "[gigachat4][load]")
{
    const auto root = gigachat4test::MakeTempDir("gc4_idx_missing");
    Gc4Index idx;
    std::string err;
    CHECK_FALSE(LoadGigaChat4Index(root.string(), idx, err));
    CHECK_FALSE(err.empty());
}

TEST_CASE("gigachat4 index rejects the sibling's format tag",
          "[gigachat4][load]")
{
    // `logestix-gigachat-nvfp4` is the 3.5 layout: it has no indexer tensors and
    // zero-centred norms, so loading it as this model would decode garbage.
    const auto root = gigachat4test::MakeTempDir("gc4_idx_fmt");
    gigachat4test::WriteGigaChat4Index(root, /*hidden_override=*/2560,
                                       /*format_tag=*/"logestix-gigachat-nvfp4");
    Gc4Index idx;
    std::string err;
    CHECK_FALSE(LoadGigaChat4Index(root.string(), idx, err));
    CHECK(err.find("format") != std::string::npos);
}

TEST_CASE("gigachat4 index requires the DSA indexer geometry",
          "[gigachat4][load]")
{
    const auto root = gigachat4test::MakeTempDir("gc4_idx_noidx");
    // A 3.5-shaped index without index_* fields must be refused, not defaulted.
    gigachat4test::WriteGigaChat4Index(root);
    const auto path = root / "model.index.json";
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string json = ss.str();
    in.close();
    for (const char *field : {"\"index_heads\"", "\"index_head_dim\"",
                              "\"index_topk\""})
    {
        const size_t pos = json.find(field);
        REQUIRE(pos != std::string::npos);
        const size_t eol = json.find('\n', pos);
        json.erase(pos, eol - pos + 1);
    }
    std::ofstream out(path);
    out << json;
    out.close();

    Gc4Index idx;
    std::string err;
    CHECK_FALSE(LoadGigaChat4Index(root.string(), idx, err));
    CHECK(err.find("indexer") != std::string::npos);
}

TEST_CASE("gigachat4 frozen geometry mismatch fails validation",
          "[gigachat4][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc4_idx_geo");
    gigachat4test::WriteGigaChat4Index(root, /*hidden_override=*/4096);
    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4Index(root.string(), idx, err));
    CHECK_FALSE(ValidateGigaChat4Geometry(idx, err));
    CHECK(err.find("mismatch") != std::string::npos);
}
