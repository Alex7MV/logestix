// AliceAI loader topology/format gate tests on a synthetic converted fixture.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/alice_moe_convert.h>
#include <lsxcommon/alice_moe_load.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "test_alice_moe_fixture.h"

namespace fs = std::filesystem;

namespace {

fs::path ConvertFixture()
{
    const fs::path in = fs::temp_directory_path() / "alice_load_in";
    const fs::path out = fs::temp_directory_path() / "alice_load_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);
    alicetest::WriteAliceConfigJson(in);
    alicetest::WriteAliceShard(in);
    auto st = lsxcommon::alice::ConvertAliceModel(in.string(), out.string());
    REQUIRE(st.ok());
    return out;
}

} // namespace

TEST_CASE("alice loader parses topology and gates the format", "[alice][load]")
{
    REQUIRE(std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr);
    const fs::path out = ConvertFixture();

    lsxcommon::alice::AliceMoeTopology top;
    auto st = lsxcommon::alice::LoadAliceTopology(out.string(), top);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    REQUIRE(top.format == "logestix-alice-nvfp4");
    REQUIRE(top.num_layers == alicetest::kLayers);
    REQUIRE(top.num_experts == alicetest::kExperts);
    REQUIRE(top.block_attn_res_block_size == alicetest::kBlockAttnRes);
    REQUIRE(static_cast<int>(top.layer_types.size()) == alicetest::kLayers);

    // Synthetic geometry intentionally differs from the frozen passport, so the
    // strict frozen comparison must fail; production Load wraps it with the
    // SyntheticGeometryAllowed() bypass.
    REQUIRE_FALSE(lsxcommon::alice::AliceTopologyMatchesFrozen(top));

    // Corrupt the format tag -> the gate rejects it.
    const fs::path idx = out / "logestix" / "model.index.json";
    std::string json;
    {
        std::ifstream f(idx);
        json.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    }
    const std::string needle = "logestix-alice-nvfp4";
    json.replace(json.find(needle), needle.size(), "logestix-wrong");
    {
        std::ofstream f(idx, std::ios::trunc);
        f << json;
    }
    lsxcommon::alice::AliceMoeTopology bad;
    REQUIRE_FALSE(
        lsxcommon::alice::LoadAliceTopology(out.string(), bad).ok());
}
