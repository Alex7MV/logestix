#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace {
void set_model_root(const char *value)
{
#if defined(_WIN32)
    _putenv_s("LSX_MODEL_ROOT", value ? value : "");
#else
    if (value)
        setenv("LSX_MODEL_ROOT", value, 1);
    else
        unsetenv("LSX_MODEL_ROOT");
#endif
}
} // namespace

TEST_CASE("path_registry platform root base", "[path_registry]")
{
    set_model_root(nullptr);
    const std::string models = lsxcommon::lsx_model_models_dir();
    REQUIRE_FALSE(models.empty());
#if defined(_WIN32)
    CHECK(models == R"(D:\Programs\huggingface.co)");
#else
    CHECK(models == "/mnt/llm_models/models");
#endif
}

TEST_CASE("path_registry LSX_MODEL_ROOT overrides the compile-time base", "[path_registry]")
{
    const std::string override_root =
        (std::filesystem::temp_directory_path() / "lsx-vault").string();
    set_model_root(override_root.c_str());
    CHECK(lsxcommon::lsx_model_models_dir() == override_root);
    const std::filesystem::path meta(lsxcommon::lsx_model_metadata_root());
    CHECK(meta.parent_path().string() == override_root);
    CHECK(meta.filename() == "model-metadata");

    set_model_root(nullptr);
#if defined(_WIN32)
    CHECK(lsxcommon::lsx_model_models_dir() == R"(D:\Programs\huggingface.co)");
#else
    CHECK(lsxcommon::lsx_model_models_dir() == "/mnt/llm_models/models");
#endif
}

TEST_CASE("path_registry vault root carries model-metadata block", "[path_registry]")
{
    const std::string root = lsxcommon::lsx_model_metadata_root();
    const std::filesystem::path p(root);
    REQUIRE(p.filename() == "model-metadata");
    REQUIRE(p.parent_path().string() == lsxcommon::lsx_model_models_dir());
}

TEST_CASE("path_registry appends inherited subpath after block", "[path_registry]")
{
    const std::string pathed = lsxcommon::lsx_model_metadata("manifests/glm.json");
    const std::filesystem::path p(pathed);
    REQUIRE(p.filename() == "glm.json");
    REQUIRE(p.parent_path().filename() == "manifests");
    REQUIRE(p.parent_path().parent_path().filename() == "model-metadata");
}

TEST_CASE("path_registry round-trips into product directory", "[path_registry]")
{
    namespace fs = std::filesystem;
    auto mans = lsxcommon::lsx_model_metadata("manifests/model.index.json");
    auto dir = fs::path(mans).parent_path();
    REQUIRE(dir.string().rfind(lsxcommon::lsx_model_metadata_root(), 0) == 0);
}
