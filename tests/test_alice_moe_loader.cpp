// AliceAI loader topology/format gate tests on a synthetic converted fixture.
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/index_layout.h"
#include <lsxcommon/alice_moe_convert.h>
#include <lsxcommon/alice_moe_load.h>

#include <simdjson.h>

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
    const fs::path out = ConvertFixture();

    lsxcommon::alice::AliceMoeTopology top;
    auto st = lsxcommon::alice::LoadAliceTopology(out.string(), top);
    if (!st.ok()) FAIL(st.ToString());
    REQUIRE(st.ok());
    REQUIRE(top.layout == lsxcommon::kIndexLayout);
    REQUIRE(top.num_layers == alicetest::kLayers);
    REQUIRE(top.num_experts == alicetest::kExperts);
    REQUIRE(top.block_attn_res_block_size == alicetest::kBlockAttnRes);
    REQUIRE(static_cast<int>(top.layer_types.size()) == alicetest::kLayers);

    // Synthetic geometry intentionally differs from the frozen passport, so the
    // strict frozen comparison must fail; production Load wraps it with the
    // SyntheticGeometryAllowed() bypass.
    REQUIRE_FALSE(lsxcommon::alice::AliceTopologyMatchesFrozen(top));

    // Corrupt the format tag -> the gate rejects it.
    const fs::path idx = out / "model.index.json";
    std::string json;
    {
        std::ifstream f(idx);
        json.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    }
    const std::string needle = "\"layout\": 2";
    json.replace(json.find(needle), needle.size(), "\"layout\": 1");
    {
        std::ofstream f(idx, std::ios::trunc);
        f << json;
    }
    lsxcommon::alice::AliceMoeTopology bad;
    REQUIRE_FALSE(
        lsxcommon::alice::LoadAliceTopology(out.string(), bad).ok());
}

TEST_CASE("alice resident bytes count every uploaded weight", "[alice][load]")
{
    // `ResidentWeightBytes()` drives the governor lease and the `resident:`
    // line of lsxbenchmark, so it must be the sum of what the loader actually
    // uploads. It used to be a hand-derived formula that counted only the
    // embed/head, router gates and shared experts, silently omitting every
    // KDA/attention FP16 projection (the dense body is 4.4 GiB on the real
    // checkpoint, the old report was 1.2 GiB). The converter's VRAM profile
    // independently computes the same quantity from the source inventory
    // (`vram.min_bytes` = device-resident weights only), so the two must agree.
    const fs::path out = ConvertFixture();

    std::string json;
    {
        std::ifstream f(out / "model.index.json");
        json.assign((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    }
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json);
    auto doc = parser.iterate(pad);
    int64_t min_bytes = 0;
    REQUIRE(doc["vram"]["min_bytes"].get_int64().get(min_bytes) ==
            simdjson::SUCCESS);

    lsxcommon::alice::AlicePinnedWeights pw;
    REQUIRE(lsxcommon::alice::LoadAlicePinnedWeights(out.string(), pw,
                                                     /*stream=*/nullptr));
    INFO("loader resident=" << pw.resident_bytes
                            << " index min_bytes=" << min_bytes);
    REQUIRE(pw.resident_bytes > 0);
    REQUIRE(static_cast<int64_t>(pw.resident_bytes) == min_bytes);
    lsxcommon::alice::FreeAlicePinnedWeights(pw);
}
