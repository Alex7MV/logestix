// TDD — GigaChat4-708B-A41B loader-stage passport (converter output ->
// residency). Synthetic index only: never opens the vault. The frozen geometry
// gate is exercised for real (no synthetic bypass), including the symmetric
// refusal: the 708B row must not accept a 37B tree and the 37B row must not
// accept a 708B one.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"
#include "gigachat_moe_708b_constants.h"
#include "lsxcommon/gigachat_moe_708b_model.h"
#include "lsxcommon/gigachat_moe_37b_load.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace gc708 = lsxcommon::gigachat4_708b;
namespace frozen708 = lsxcommon::gigachat4_708b::frozen;
using lsxcommon::gigachat4::Gc4FrozenExpect;
using lsxcommon::gigachat4::Gc4Index;
using lsxcommon::gigachat4::LoadGigaChat4IndexWithFormat;
using lsxcommon::gigachat4::ValidateGigaChat4GeometryAgainst;

namespace {

void DisableSyntheticGeometry()
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0", 1);
#endif
}

} // namespace

TEST_CASE("gigachat4-708b expectation is built from its own passport",
          "[gigachat708b][load]")
{
    const Gc4FrozenExpect e = gc708::Gc708BExpect();
    CHECK(std::string(e.format) == "logestix-gigachat4-708b-nvfp4");
    CHECK(e.geo.hidden == 7168);
    CHECK(e.geo.layers == 64);
    CHECK(e.geo.num_experts == 256);
    CHECK(e.geo.moe_intermediate == 2048);
    CHECK(e.geo.first_k_dense == 3);
    REQUIRE(e.full_attention_layers.size() == 16);
    CHECK(e.gate_rank == frozen708::kGateRank);
    // Distinct from the family default, which stays the 37B's.
    const Gc4FrozenExpect d = lsxcommon::gigachat4::Gc4DefaultExpect();
    CHECK(d.geo.layers != e.geo.layers);
    CHECK(std::string(d.format) != std::string(e.format));
}

TEST_CASE("gigachat4-708b index matches its own passport",
          "[gigachat708b][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc708_idx_ok");
    gigachat4test::WriteGigaChat4_708BIndex(root);

    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), frozen708::kFormat, idx,
                                         err));
    CHECK(err.empty());
    CHECK(idx.num_layers == 64);
    CHECK(idx.hidden_size == 7168);
    CHECK(idx.num_attention_heads == 64);
    CHECK(idx.num_experts == 256);
    CHECK(idx.num_experts_per_tok == 8);
    CHECK(idx.moe_intermediate_size == 2048);
    CHECK(idx.dense_intermediate_size == 18432);
    CHECK(idx.first_k_dense_replace == 3);
    CHECK(idx.index_heads == 64);
    CHECK(idx.index_topk == 2048);
    CHECK(idx.full_attention_layers.size() == 16);
    CHECK(idx.full_attention_layers.front() == 3);
    CHECK(idx.full_attention_layers.back() == 63);

    CHECK(ValidateGigaChat4GeometryAgainst(idx, gc708::Gc708BExpect(), err));
    CHECK(err.empty());
    // And the 37B passport refuses it — the refusal is symmetric.
    CHECK_FALSE(lsxcommon::gigachat4::ValidateGigaChat4Geometry(idx, err));
    CHECK(err.find("num_layers") != std::string::npos);
}

TEST_CASE("gigachat4-708b refuses a 37B tree on the format tag",
          "[gigachat708b][load]")
{
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc708_idx_37fmt");
    gigachat4test::WriteGigaChat4Index(root);  // the 37B tree's index
    Gc4Index idx;
    std::string err;
    CHECK_FALSE(LoadGigaChat4IndexWithFormat(root.string(), frozen708::kFormat,
                                             idx, err));
    CHECK(err.find("format") != std::string::npos);
}

TEST_CASE("gigachat4-708b refuses a 37B tree on the geometry",
          "[gigachat708b][load]")
{
    // Even when the tag is right (a hand-edited or mis-converted tree), the
    // geometry gate must stop it: 32 layers against a frozen 64.
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc708_idx_37geo");
    gigachat4test::WriteGigaChat4_708BIndex(root,
                                            /*hidden_override=*/7168,
                                            /*format_tag=*/frozen708::kFormat);
    // Rewrite num_layers to the 37B depth, keeping the 708B tag.
    const auto path = root / "model.index.json";
    std::string json;
    {
        std::ifstream in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        json = ss.str();
    }
    const size_t pos = json.find("\"num_layers\": 64");
    REQUIRE(pos != std::string::npos);
    json.replace(pos, std::string("\"num_layers\": 64").size(),
                 "\"num_layers\": 32");
    {
        std::ofstream out(path);
        out << json;
    }
    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), frozen708::kFormat, idx,
                                         err));
    CHECK_FALSE(ValidateGigaChat4GeometryAgainst(idx, gc708::Gc708BExpect(), err));
    CHECK(err.find("mismatch") != std::string::npos);
}

TEST_CASE("gigachat4-708b geometry gate catches a single wrong width",
          "[gigachat708b][load]")
{
    // A tree converted with 2032 moe_intermediate would load and answer, off by
    // 16 elements per expert row. The gate must name the field.
    DisableSyntheticGeometry();
    const auto root = gigachat4test::MakeTempDir("gc708_idx_width");
    gigachat4test::WriteGigaChat4_708BIndex(root);
    const auto path = root / "model.index.json";
    std::string json;
    {
        std::ifstream in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        json = ss.str();
    }
    const size_t pos = json.find("\"moe_intermediate_size\": 2048");
    REQUIRE(pos != std::string::npos);
    json.replace(pos, std::string("\"moe_intermediate_size\": 2048").size(),
                 "\"moe_intermediate_size\": 2032");
    {
        std::ofstream out(path);
        out << json;
    }
    Gc4Index idx;
    std::string err;
    REQUIRE(LoadGigaChat4IndexWithFormat(root.string(), frozen708::kFormat, idx,
                                         err));
    CHECK_FALSE(ValidateGigaChat4GeometryAgainst(idx, gc708::Gc708BExpect(), err));
    CHECK(err.find("moe_intermediate_size") != std::string::npos);
}
