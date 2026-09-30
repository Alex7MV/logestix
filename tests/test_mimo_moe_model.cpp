// TDD — MiMo model registration + synthetic Load/Infer (Model Ingestion
// Protocol step 3). Synthetic converted fixture; label inference.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include <lsxcommon/mimo_moe_convert.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>

#include "test_mimo_moe_fixture.h"

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
