// libs/lsxstorage: split source/artifact roots (invariant I5).
//
// The point of this layer is that a converted artifact never lives inside the
// read-only Hugging Face source checkout again: LSX_SOURCE_ROOT is what we read
// checkpoints from, LSX_ARTIFACT_ROOT is where conversions are written. The
// third root, LSX_MODEL_ROOT, is the pre-split single-vault spelling and is kept
// only so SourceRoot() has a default and LegacyDir() can *diagnose* an old
// layout — it is not a loadable path.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/path_registry.h"
#include "lsxstorage/roots.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

// RAII env var override: restores the previous value on scope exit so test
// order cannot leak one case's root into the next.
class ScopedEnv {
public:
    ScopedEnv(const char *name, const char *value) : name_(name)
    {
        if (const char *prev = std::getenv(name)) {
            previous_ = prev;
            had_previous_ = true;
        }
        set(value);
    }
    ~ScopedEnv() { set(had_previous_ ? previous_.c_str() : nullptr); }

    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
    void set(const char *v)
    {
#if defined(_WIN32)
        _putenv_s(name_, v ? v : "");
#else
        if (v)
            setenv(name_, v, 1);
        else
            unsetenv(name_);
#endif
    }
    const char *name_;
    std::string previous_;
    bool had_previous_ = false;
};

} // namespace

TEST_CASE("artifact root is optional and never invented", "[storage_roots]") {
    ScopedEnv a("LSX_ARTIFACT_ROOT", nullptr);
    CHECK_FALSE(lsx::storage::ArtifactRoot().has_value());
}

TEST_CASE("artifact dir joins family/model_type/profile under the artifact root",
          "[storage_roots]") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", "/tmp/art");
    REQUIRE(lsx::storage::ArtifactDir("phi", "phi-4", "bf16") ==
            fs::path("/tmp/art/phi/phi-4/bf16"));
    // model_type, NOT the vault dir name: "Qwen3.5-9B" -> "qwen3.5-9b".
    REQUIRE(lsx::storage::ArtifactDir("qwen", "qwen3.5-9b", "bf16") ==
            fs::path("/tmp/art/qwen/qwen3.5-9b/bf16"));
}

TEST_CASE("artifact dir is empty when no artifact root is configured",
          "[storage_roots]") {
    // A caller that ignores ArtifactRoot() must get an empty path rather than a
    // path under some default: silently writing into the source vault is the
    // bug this whole layer exists to prevent.
    ScopedEnv e("LSX_ARTIFACT_ROOT", nullptr);
    CHECK(lsx::storage::ArtifactDir("phi", "phi-4", "bf16").empty());
}

TEST_CASE("source root falls back to LSX_MODEL_ROOT then the platform default",
          "[storage_roots]") {
    {
        ScopedEnv a("LSX_SOURCE_ROOT", nullptr);
        ScopedEnv b("LSX_MODEL_ROOT", "/m");
        CHECK(lsx::storage::SourceRoot() == fs::path("/m"));
    }
    {
        ScopedEnv a("LSX_SOURCE_ROOT", "/s");
        ScopedEnv b("LSX_MODEL_ROOT", "/m");
        CHECK(lsx::storage::SourceRoot() == fs::path("/s"));
    }
}

TEST_CASE("lsxcommon forwards to lsxstorage verbatim", "[storage_roots]") {
    ScopedEnv a("LSX_MODEL_ROOT", "/m");
    CHECK(lsxcommon::lsx_model_models_dir() == "/m");
}

TEST_CASE("an unset artifact root is never silently replaced by a legacy path",
          "[storage_roots]") {
    ScopedEnv a("LSX_ARTIFACT_ROOT", nullptr);
    ScopedEnv b("LSX_MODEL_ROOT", "/m");
    CHECK_FALSE(lsx::storage::ArtifactRoot().has_value());
    // LegacyDir still answers, but only so Resolve can SAY "source, not artifact".
    CHECK(lsx::storage::LegacyDir("phi-4") == fs::path("/m/phi-4/logestix"));
}

TEST_CASE("cache dir precedence is unchanged by the move", "[storage_roots]") {
    ScopedEnv a("LSX_CACHE_DIR", "/c");
    CHECK(lsx::storage::CacheRoot() == fs::path("/c"));
    CHECK(lsxcommon::lsx_cache_path("perf-cache/x.json") == "/c/perf-cache/x.json");
}

TEST_CASE("a relative artifact root is rejected rather than resolved against cwd",
          "[storage_roots]") {
    // A relative root would make the artifact layout depend on where the
    // process happened to be started.
    ScopedEnv e("LSX_ARTIFACT_ROOT", "artifacts");
    CHECK_FALSE(lsx::storage::ArtifactRoot().has_value());
}
