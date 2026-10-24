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

    set_model_root(nullptr);
#if defined(_WIN32)
    CHECK(lsxcommon::lsx_model_models_dir() == R"(D:\Programs\huggingface.co)");
#else
    CHECK(lsxcommon::lsx_model_models_dir() == "/mnt/llm_models/models");
#endif
}

TEST_CASE("path_registry cache root is per-user, never in the vault", "[path_registry]")
{
    const std::string cache = lsxcommon::lsx_cache_dir();
    REQUIRE_FALSE(cache.empty());
    // The cache must not live inside the model vault: a converter or a test
    // writing there must not be able to clobber model data (or vice versa).
    const std::string models = lsxcommon::lsx_model_models_dir();
    REQUIRE(cache.rfind(models, 0) != 0);
}

TEST_CASE("path_registry LSX_CACHE_DIR overrides the cache root", "[path_registry]")
{
    const std::string dir =
        (std::filesystem::temp_directory_path() / "lsx-cache-test").string();
#if defined(_WIN32)
    _putenv_s("LSX_CACHE_DIR", dir.c_str());
#else
    setenv("LSX_CACHE_DIR", dir.c_str(), 1);
#endif
    CHECK(lsxcommon::lsx_cache_dir() == dir);
    const std::string pathed = lsxcommon::lsx_cache_path("perf-cache/rtx 5090.json");
    const std::filesystem::path p(pathed);
    REQUIRE(p.filename() == "rtx 5090.json");
    REQUIRE(p.parent_path().filename() == "perf-cache");
    REQUIRE(p.parent_path().parent_path().string() == dir);
#if defined(_WIN32)
    _putenv_s("LSX_CACHE_DIR", "");
#else
    unsetenv("LSX_CACHE_DIR");
#endif
    CHECK(lsxcommon::lsx_cache_dir() != dir);
}
