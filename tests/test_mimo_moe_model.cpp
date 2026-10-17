// TDD — MiMo model registration + synthetic Load/Infer (Model Ingestion
// Protocol step 3). Synthetic converted fixture; label inference.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <lsxcommon/mimo_moe_convert.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>

#include "test_mimo_moe_fixture.h"
#include "test_streaming_assert.h"

namespace fs = std::filesystem;

namespace {

fs::path BuildConverted()
{
    const fs::path root = fs::temp_directory_path() / "lsx_mimo_model";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path in = root / "src";
    const fs::path out = root / "dst";
    mimotest::BuildSyntheticMiMoCheckpoint(in);
    REQUIRE(lsxcommon::mimo::ConvertMiMoModel(in.string(), out.string()).ok());
    return out;
}

} // namespace

TEST_CASE("mimo model loads a synthetic checkpoint and infers", "[mimo][model]")
{
    const fs::path out = BuildConverted();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "mimo-v2.6-flash-rl";

    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "mimo-v2.6-flash-rl");
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2, 3, 4};
    req.max_gen_tokens = 3;
    auto r = m->Infer(req);
    INFO(r.error_msg);
    CHECK(r.ok);
    CHECK(static_cast<int>(r.gen_tokens.size()) >= 1);

    m->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("mimo Infer streams one chunk per generated token",
          "[mimo][model][lifecycle]")
{
    // lsxfabric --serve / taskmosaic stream exclusively through
    // req.token_callback: a family that never invokes it delivers an empty
    // stream (only the final DONE frame) to every client.
    const fs::path out = BuildConverted();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "mimo-v2.6-flash-rl";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2, 3, 4};
    req.max_gen_tokens = 4;
    ExpectStreamMatchesTokens(*m, req);

    m->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("mimo Infer honours cooperative cancel", "[mimo][model][lifecycle]")
{
    const fs::path out = BuildConverted();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "mimo-v2.6-flash-rl";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2, 3, 4};
    // OPEN_TASKS §E1a: pre-set and mid-generation abort both must stop.
    ExpectCooperativeCancelStops(*m, req);

    m->Unload();
    fs::remove_all(out.parent_path());
}

TEST_CASE("mimo resident bytes cover the uploaded layer weights",
          "[mimo][model][lifecycle]")
{
    // The governor lease and `lsxbenchmark --bench`'s `resident:` line are both
    // driven by ResidentWeightBytes(). A partial sum under-reports the real
    // device footprint and lets the governor admit a second model.
    const fs::path out = BuildConverted();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "mimo-v2.6-flash-rl";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    mimotest::SynthShape sh;
    const std::uint64_t qkv_ga =
        static_cast<std::uint64_t>(mimotest::GaQkvRows(sh)) * sh.hidden * 2;
    const std::uint64_t qkv_swa =
        static_cast<std::uint64_t>(mimotest::SwaQkvRows(sh)) * sh.hidden * 2;
    const std::uint64_t o_proj = static_cast<std::uint64_t>(sh.hidden) *
                                 sh.heads * sh.v_head_dim * 2;
    // Globals (embed + head) + every layer's qkv and o_proj — the bare minimum
    // that is physically resident after Load.
    const std::uint64_t floor_bytes =
        2ull * sh.vocab * sh.hidden * 2 + qkv_ga + qkv_swa +
        static_cast<std::uint64_t>(sh.layers) * o_proj;
    INFO("resident=" << m->ResidentWeightBytes() << " floor=" << floor_bytes);
    REQUIRE(m->ResidentWeightBytes() >= floor_bytes);

    m->Unload();
    fs::remove_all(out.parent_path());
}
