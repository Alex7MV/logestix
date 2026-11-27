#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "lsxcommon/glm_moe_convert.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"

#include "test_glm_moe_fixture.h"

namespace fs = std::filesystem;

TEST_CASE("GLM factory autodetects and creates glm-5.2 model", "[glm][model]")
{
    // Layout mirrors production: <model>/logestix/model.index.json where
    // <model> is the checkpoint directory (e.g. /mnt/llm_models/models/GLM-5.2).
    auto base = fs::temp_directory_path() / "logestix_glm_model";
    fs::remove_all(base);
    auto model_home = base / "GLM-5.2";
    fs::create_directories(model_home);

    glmtest::WriteSafetensors(model_home, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(model_home);
    glmtest::WriteTokenizer(model_home);
    auto st = lsxcommon::glm::ConvertGlmModel(model_home.string(), model_home);
    REQUIRE(st.ok());

    // Autodetect from logestix/model.index.json ("GLM-5.2" -> "glm-5.2").
    auto type = lsxcommon::LsxModelFactory::DetectType(model_home.string());
    CHECK(type == "glm-5.2");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_home.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "glm-5.2");

    // Load-once lifecycle (ResourceGovernor spec): Load() uploads the pinned
    // weights and validates them against the frozen geometry constants. A
    // synthetic model (hidden=16) fails that validation cleanly.
    REQUIRE_FALSE(m->Load());
    CHECK(!m->Inspect().shards.empty());  // Inspect needs no residency

    // Guard semantics: an unloaded model rejects every request at the
    // ModelBase preamble ("model not loaded") — including SelfCheck's forced
    // max_gen_tokens (the forced value is applied inside the generation call,
    // after the guard).
    lsxcommon::ModelRequest zero;
    zero.input_ids = {154826};
    zero.max_gen_tokens = 0;
    auto r0 = m->Infer(zero);
    CHECK(!r0.ok);
    CHECK(r0.error_msg == "model not loaded");

    auto sc0 = m->SelfCheck(zero);
    CHECK(!sc0.ok);
    CHECK(!sc0.error_msg.empty());

    lsxcommon::ModelRequest empty;
    empty.input_ids = {};
    empty.max_gen_tokens = 8;
    auto rem = m->Infer(empty);
    CHECK(!rem.ok);
    CHECK(rem.error_msg == "model not loaded");

    m->Unload();  // idempotent after failed load
    m->Unload();
    fs::remove_all(base);
}

TEST_CASE("GLM model explicit type create", "[glm][model]")
{
    auto base = fs::temp_directory_path() / "logestix_glm_model_type";
    fs::remove_all(base);
    auto model_home = base / "GLM-5.2";
    fs::create_directories(model_home);

    glmtest::WriteSafetensors(model_home, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(model_home);
    glmtest::WriteTokenizer(model_home);
    REQUIRE(lsxcommon::glm::ConvertGlmModel(model_home.string(), model_home).ok());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_home.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "glm-5.2";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "glm-5.2");

    m->Unload();
    fs::remove_all(base);
}