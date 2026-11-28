#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/path_registry.h"
#include "lsxstorage/roots.h"

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

// Both spellings of the source root, cleared together: a case that asserts
// "nothing is configured" must not be satisfied by the other variable.
void unset_source_roots()
{
    set_model_root(nullptr);
#if defined(_WIN32)
    _putenv_s("LSX_SOURCE_ROOT", "");
#else
    unsetenv("LSX_SOURCE_ROOT");
#endif
}

void set_source_root(const char *value)
{
#if defined(_WIN32)
    _putenv_s("LSX_SOURCE_ROOT", value ? value : "");
#else
    if (value)
        setenv("LSX_SOURCE_ROOT", value, 1);
    else
        unsetenv("LSX_SOURCE_ROOT");
#endif
}
} // namespace

TEST_CASE("no compiled-in checkpoint root exists at all", "[path_registry]") {
    // The source root is configured or absent — never assumed. This is the
    // Windows half of the same rule: a `D:\Programs\lsx_model` default in the
    // binary is a path the operator never chose, and on Linux it would be a
    // drive letter that cannot exist.
    unset_source_roots();
    CHECK_FALSE(lsx::storage::SourceRoot().has_value());

    // And it becomes configured by the variable alone, with no fallback anywhere.
    set_source_root("/configured/store");
    REQUIRE(lsx::storage::SourceRoot().has_value());
    CHECK(*lsx::storage::SourceRoot() == std::filesystem::path("/configured/store"));
    unset_source_roots();
}

TEST_CASE("path_registry cache root is per-user, never in the vault", "[path_registry]")
{
    const std::string cache = lsxcommon::lsx_cache_dir();
    REQUIRE_FALSE(cache.empty());
    // The cache must not live inside a model store: a converter or a test
    // writing there must not be able to clobber model data (or vice versa).
    // Asserted against the roots that ARE configured, so it holds with no source
    // root at all as well as with one.
    if (const auto source = lsx::storage::SourceRoot(); source.has_value())
        CHECK(cache.rfind(source->string(), 0) != 0);
    if (const auto artifact = lsx::storage::ArtifactRoot(); artifact.has_value())
        CHECK(cache.rfind(artifact->string(), 0) != 0);
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
