#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "lsxcommon/converter.h"
#include "lsxcommon/glm_convert.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"

#include "glm_test_fixture.h"

namespace fs = std::filesystem;

TEST_CASE("GLM factory autodetects and creates glm-5.2 model", "[glm][model]")
{
    // Layout mirrors production: <model>/logestix/model.index.json where
    // <model> is the checkpoint directory (e.g. /mnt/llm_models/models/GLM-5.2).
    auto base = fs::temp_directory_path() / "logestix_glm_model";
    fs::remove_all(base);
    auto model_home = base / "GLM-5.2";
    auto out = model_home / "logestix";
    fs::create_directories(model_home);

    glmtest::WriteSafetensors(model_home, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(model_home);
    glmtest::WriteTokenizer(model_home);
    auto st = lsxcommon::glm::ConvertGlmModel(model_home.string(), out);
    REQUIRE(st.ok());

    // Autodetect from logestix/model.index.json ("GLM-5.2" -> "glm-5.2").
    auto type = lsxcommon::LsxModelFactory::DetectType(model_home.string());
    CHECK(type == "glm-5.2");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_home.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "glm-5.2");

    REQUIRE(m->Load());
    CHECK(m->Tokenizer().valid());

    auto info = m->Inspect();
    REQUIRE(!info.shards.empty());

    // Guard semantics (ModelBase::ValidateInferRequest): a loaded model with a
    // synthetic dir and zero max_gen_tokens must reject the request; no real
    // 79-layer weights are needed.
    lsxcommon::ModelRequest zero;
    zero.input_ids = {154826};
    zero.max_gen_tokens = 0;
    auto r0 = m->Infer(zero);
    CHECK(!r0.ok);
    CHECK(r0.error_msg == "max_gen_tokens must be positive");

    auto sc0 = m->SelfCheck(zero);
    CHECK(!sc0.ok);
    // SelfCheck forces max_gen_tokens = max(2, ...) before Infer, so the guard
    // message does not surface here; the synthetic dir then fails at pinned
    // weight load. Only require a non-empty error.
    CHECK(!sc0.error_msg.empty());

    lsxcommon::ModelRequest empty;
    empty.input_ids = {};
    empty.max_gen_tokens = 8;
    auto rem = m->Infer(empty);
    CHECK(!rem.ok);
    CHECK(rem.error_msg == "empty input_ids");

    m->Unload();
    fs::remove_all(base);
}

TEST_CASE("GLM model explicit type create", "[glm][model]")
{
    auto base = fs::temp_directory_path() / "logestix_glm_model_type";
    fs::remove_all(base);
    auto model_home = base / "GLM-5.2";
    auto out = model_home / "logestix";
    fs::create_directories(model_home);

    glmtest::WriteSafetensors(model_home, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(model_home);
    glmtest::WriteTokenizer(model_home);
    REQUIRE(lsxcommon::glm::ConvertGlmModel(model_home.string(), out).ok());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_home.string();
    cfg.model_type = "glm-5.2";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "glm-5.2");

    m->Unload();
    fs::remove_all(base);
}