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
    auto st = lsxcommon::glm::ConvertGlmModel(model_home.string(), out);
    REQUIRE(st.ok());

    // Tokenizer arrows are required for Load(); the real pipeline builds them
    // from <model>/tokenizer.json. Use the repository synthetic tokenizer.
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", out / "pinned").ok());

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

    // Inference pipeline is not yet implemented for GLM.
    lsxcommon::ModelRequest req;
    req.input_ids = {154826};
    req.max_gen_tokens = 0;
    auto r = m->Infer(req);
    CHECK(!r.ok);
    CHECK(!r.error_msg.empty());

    auto sc = m->SelfCheck(req);
    CHECK(!sc.ok);
    CHECK(!sc.error_msg.empty());

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
    REQUIRE(lsxcommon::glm::ConvertGlmModel(model_home.string(), out).ok());
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", out / "pinned").ok());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_home.string();
    cfg.model_type = "glm-5.2";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "glm-5.2");

    m->Unload();
    fs::remove_all(base);
}