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
#include "lsxstorage/resolve.h"
#include "lsxstorage/roots.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

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

// Scratch source root, removed on scope exit.
class TempRoot {
public:
    TempRoot()
    {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("lsx-src-" + std::to_string(::getpid()) + "-" +
                 std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    ~TempRoot()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    const std::filesystem::path &path() const { return path_; }
    const char *c_str() const { return path_.c_str(); }
    TempRoot(const TempRoot &) = delete;
    TempRoot &operator=(const TempRoot &) = delete;

private:
    std::filesystem::path path_;
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

TEST_CASE("the source root default moved to the org-nested store",
          "[storage_roots]") {
    // The vault is gone; checkpoints live at <root>/<org>/<model>.
    ScopedEnv a("LSX_SOURCE_ROOT", nullptr);
    ScopedEnv b("LSX_MODEL_ROOT", nullptr);
#if defined(_WIN32)
    CHECK(lsx::storage::SourceRoot() == fs::path(R"(D:\Programs\lsx_model)"));
#else
    CHECK(lsx::storage::SourceRoot() == fs::path("/mnt/fast_storage/lsx_model"));
#endif
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

// ── org-nested source store: <root>/<org>/<model> ─────────────────────────
//
// Checkpoints are no longer flat under the root; every model sits under the
// organisation that published it, which is also how Hugging Face names them.
// Resolution therefore searches one level of nesting instead of building
// <root>/<model> directly.

namespace {

void MakeCheckpoint(const fs::path &dir)
{
    fs::create_directories(dir);
    std::ofstream cfg(dir / "config.json");
    cfg << "{}";
}

} // namespace

TEST_CASE("a checkpoint that ships params.json instead of config.json is a model",
          "[storage_roots]") {
    // Mistral's own download layout has params.json + consolidated*.safetensors
    // and no config.json. Trusting config.json alone would hide those models.
    TempRoot root;
    const auto dir = root.path() / "mistralai" / "Mistral-Small-4-119B";
    fs::create_directories(dir);
    std::ofstream(dir / "params.json") << "{}";
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    const auto found = lsx::storage::FindSourceModel("Mistral-Small-4-119B", &why);
    REQUIRE(found.has_value());
    CHECK(*found == dir);
}

TEST_CASE("a checkpoint that ships only safetensors weights is a model",
          "[storage_roots]") {
    TempRoot root;
    const auto dir = root.path() / "acme" / "weights-only";
    fs::create_directories(dir);
    std::ofstream(dir / "model-00001-of-00001.safetensors") << "x";
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    REQUIRE(lsx::storage::FindSourceModel("weights-only", &why).has_value());
}

TEST_CASE("an org-qualified name is matched exactly, not by search",
          "[storage_roots]") {
    TempRoot root;
    MakeCheckpoint(root.path() / "microsoft" / "phi-4");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    const auto found = lsx::storage::FindSourceModel("microsoft/phi-4", &why);
    REQUIRE(found.has_value());
    CHECK(*found == root.path() / "microsoft" / "phi-4");

    // The wrong org must not resolve just because the model name matches.
    CHECK_FALSE(
        lsx::storage::FindSourceModel("someone-else/phi-4", &why).has_value());
}

TEST_CASE("a source model is found one level down, under its org",
          "[storage_roots]") {
    TempRoot root;
    MakeCheckpoint(root.path() / "microsoft" / "phi-4");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    const auto found = lsx::storage::FindSourceModel("phi-4", &why);
    REQUIRE(found.has_value());
    CHECK(*found == root.path() / "microsoft" / "phi-4");
    CHECK(why.empty());
}

TEST_CASE("a model name absent from every org reports why", "[storage_roots]") {
    TempRoot root;
    MakeCheckpoint(root.path() / "microsoft" / "phi-4");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    CHECK_FALSE(lsx::storage::FindSourceModel("not-a-model", &why).has_value());
    CHECK(why.find("not-a-model") != std::string::npos);
}

TEST_CASE("the same model name under two orgs is ambiguous, never picked",
          "[storage_roots]") {
    // Picking one silently would make the resolved checkpoint depend on
    // directory iteration order.
    TempRoot root;
    MakeCheckpoint(root.path() / "org-a" / "same-name");
    MakeCheckpoint(root.path() / "org-b" / "same-name");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    CHECK_FALSE(lsx::storage::FindSourceModel("same-name", &why).has_value());
    CHECK(why.find("ambiguous") != std::string::npos);
    CHECK(why.find("org-a") != std::string::npos);
    CHECK(why.find("org-b") != std::string::npos);
}

TEST_CASE("a bare org-nested checkpoint path resolves to source-only",
          "[storage_roots]") {
    // The shape a user actually copies out of `ls`: <root>/<org>/<model>.
    TempRoot root;
    const auto ckpt = root.path() / "Qwen" / "Qwen3.5-9B";
    MakeCheckpoint(ckpt);
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    const auto r = lsx::storage::Resolve(ckpt.string());
    CHECK(r.status == lsx::storage::ResolveStatus::kSourceOnly);
    CHECK(r.detail.find("artifact.fb") != std::string::npos);
}

TEST_CASE("listing the source store reports org and model", "[storage_roots]") {
    TempRoot root;
    MakeCheckpoint(root.path() / "microsoft" / "phi-4");
    MakeCheckpoint(root.path() / "Qwen" / "Qwen3.5-9B");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    const auto models = lsx::storage::ListSourceModels();
    REQUIRE(models.size() == 2);
    // Sorted by (org, model) so two runs report the same order.
    CHECK(models[0].org == "Qwen");
    CHECK(models[0].model == "Qwen3.5-9B");
    CHECK(models[1].org == "microsoft");
    CHECK(models[1].model == "phi-4");
}

TEST_CASE("a directory directly under the root is not mistaken for a model",
          "[storage_roots]") {
    // <root>/<org> itself has no config.json, so it is an org, not a model.
    TempRoot root;
    fs::create_directories(root.path() / "microsoft");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());
    CHECK(lsx::storage::ListSourceModels().empty());
}
