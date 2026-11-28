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
#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"
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

TEST_CASE("artifact dir mirrors the source store: org/model/profile",
          "[storage_roots]") {
    ScopedEnv e("LSX_ARTIFACT_ROOT", "/tmp/art");
    // The two path segments are the STORE's spelling, so moving between a
    // checkpoint and its artifacts means changing only the root.
    REQUIRE(lsx::storage::ArtifactDir("microsoft", "phi-4", "nvfp4") ==
            fs::path("/tmp/art/microsoft/phi-4/nvfp4"));
    REQUIRE(lsx::storage::ArtifactDir("Qwen", "Qwen3.5-9B", "nvfp4") ==
            fs::path("/tmp/art/Qwen/Qwen3.5-9B/nvfp4"));
}

TEST_CASE("artifact dir is empty when no artifact root is configured",
          "[storage_roots]") {
    // A caller that ignores ArtifactRoot() must get an empty path rather than a
    // path under some default: silently writing into the source vault is the
    // bug this whole layer exists to prevent.
    ScopedEnv e("LSX_ARTIFACT_ROOT", nullptr);
    CHECK(lsx::storage::ArtifactDir("microsoft", "phi-4", "nvfp4").empty());
}

TEST_CASE("source root falls back to LSX_MODEL_ROOT", "[storage_roots]") {
    {
        ScopedEnv a("LSX_SOURCE_ROOT", nullptr);
        ScopedEnv b("LSX_MODEL_ROOT", "/m");
        REQUIRE(lsx::storage::SourceRoot().has_value());
        CHECK(*lsx::storage::SourceRoot() == fs::path("/m"));
    }
    {
        ScopedEnv a("LSX_SOURCE_ROOT", "/s");
        ScopedEnv b("LSX_MODEL_ROOT", "/m");
        REQUIRE(lsx::storage::SourceRoot().has_value());
        CHECK(*lsx::storage::SourceRoot() == fs::path("/s"));
    }
}

TEST_CASE("an unset source root is a configuration error, not a default",
          "[storage_roots]") {
    // There is no compiled-in checkpoint root. A root the binary invents is a
    // path the operator never chose, and with the org-nested store an invented
    // root silently points at nothing — so the whole tree would SKIP as
    // "weights absent" instead of saying "you forgot to configure a store".
    // `ArtifactRoot()` has followed this rule since the split; the source root
    // was the odd one out.
    ScopedEnv a("LSX_SOURCE_ROOT", nullptr);
    ScopedEnv b("LSX_MODEL_ROOT", nullptr);

    CHECK_FALSE(lsx::storage::SourceRoot().has_value());
    CHECK(lsx::storage::ListSourceModels().empty());

    std::string why;
    CHECK_FALSE(lsx::storage::FindSourceModel("phi-4", &why).has_value());
    CHECK(why.find("LSX_SOURCE_ROOT") != std::string::npos);
    CHECK(why.find("not set") != std::string::npos);
    // The message must not name an invented location.
    CHECK(why.find("/mnt/") == std::string::npos);

    // A directory sweep cannot silently become a sweep of the cwd either.
    CHECK(lsx::storage::ListPlaceholderDirs("").empty());
}

TEST_CASE("a checkpoint path is not resolved from the cwd when no root is set",
          "[storage_roots]") {
    // `Resolve("<org>/<model>")` joins a QUALIFIED name to the source root. With
    // no root configured, that join must not degrade to a path relative to the
    // process's working directory — the caller would get a directory that
    // happens to exist there and a load that fails much later.
    ScopedEnv a("LSX_SOURCE_ROOT", nullptr);
    ScopedEnv b("LSX_MODEL_ROOT", nullptr);
    ScopedEnv c("LSX_ARTIFACT_ROOT", nullptr);

    const auto r = lsx::storage::Resolve("microsoft/phi-4");
    CHECK(r.status == lsx::storage::ResolveStatus::kNotFound);
    CHECK(r.dir.empty());
    CHECK(r.detail.find("LSX_SOURCE_ROOT") != std::string::npos);
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

// A loadable artifact, so a resolution case can exercise the real reader: an
// `artifact.fb` that only exists is not the same thing as one that parses.
void MakeArtifact(const fs::path &dir, const std::string &model_type,
                  const std::string &family, const std::string &org,
                  const std::string &model_dir, const std::string &profile)
{
    lsx::storage::ArtifactSpec spec;
    spec.model_type = model_type;
    spec.family = family;
    spec.profile = profile;
    spec.org = org;
    spec.model_dir = model_dir;
    spec.model_name = model_dir;
    std::string err;
    lsx::storage::EmitArtifact(spec, dir, &err);
    // The return value is the codec mask, which is 0 for an artifact with no
    // planes — so the artifact is verified by reading it back, not by the mask.
    const auto read = lsx::storage::ReadArtifact(dir);
    INFO(err << " — " << read.detail);
    REQUIRE(read.status == lsx::storage::ReadStatus::kOk);
}

} // namespace

TEST_CASE("an org-qualified name resolves under the artifact root",
          "[storage_roots]") {
    // `<org>/<model>` is how the store is laid out, so it is what an operator
    // copies out of a `ls`, and it is the spelling the README and the design
    // use. Resolution answered it only for SOURCE checkpoints: the artifact
    // search compared directory names, so `Qwen/Qwen3.5-9B` — a path relative
    // to the artifact root, not the cwd — fell through to "not found".
    TempRoot art;
    MakeArtifact(art.path() / "Qwen" / "Qwen3.5-9B" / "nvfp4", "qwen3.5-9b",
                 "qwen", "Qwen", "Qwen3.5-9B", "nvfp4");
    ScopedEnv a("LSX_ARTIFACT_ROOT", art.path().c_str());

    const auto model = lsx::storage::Resolve("Qwen/Qwen3.5-9B");
    REQUIRE(model.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(model.org == "Qwen");
    CHECK(model.model_dir == "Qwen3.5-9B");
    CHECK(model.profile == "nvfp4");
    CHECK(model.model_type == "qwen3.5-9b");

    // With the profile spelled out, the same tree and that profile.
    const auto with_profile = lsx::storage::Resolve("Qwen/Qwen3.5-9B/nvfp4");
    REQUIRE(with_profile.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(with_profile.dir == model.dir);
    CHECK(with_profile.profile == "nvfp4");

    // A profile that does not exist is not a silent fallback to another one.
    CHECK(lsx::storage::Resolve("Qwen/Qwen3.5-9B/bf16").status ==
          lsx::storage::ResolveStatus::kNotFound);

    // An org-qualified name that matches nothing says so, and does not become a
    // path relative to the cwd.
    CHECK(lsx::storage::Resolve("Nobody/Nothing").status ==
          lsx::storage::ResolveStatus::kNotFound);

    // And the org is part of the name, not decoration: `WrongOrg/Qwen3.5-9B`
    // must not resolve to Qwen's tree by matching the leaf alone. The store is
    // org-nested precisely so two orgs can hold the same directory name.
    const auto wrong_org = lsx::storage::Resolve("WrongOrg/Qwen3.5-9B");
    CHECK(wrong_org.status == lsx::storage::ResolveStatus::kNotFound);
}

TEST_CASE("a model resolves by its registry identity, not only its store name",
          "[storage_roots]") {
    // The store spells it `Qwen/Qwen3.5-9B`; the registry, the gates and every
    // document spell it `qwen3.5-9b`. Resolve's own contract says both are
    // things a user types — but it compared only the directory name, so the
    // identity the whole tree uses answered "no artifact and no source
    // checkpoint". A CLI default written as an identity could never load.
    TempRoot src, art;
    MakeCheckpoint(src.path() / "Qwen" / "Qwen3.5-9B");
    MakeArtifact(art.path() / "Qwen" / "Qwen3.5-9B" / "nvfp4", "qwen3.5-9b",
                 "qwen", "Qwen", "Qwen3.5-9B", "nvfp4");
    ScopedEnv s("LSX_SOURCE_ROOT", src.path().c_str());
    ScopedEnv a("LSX_ARTIFACT_ROOT", art.path().c_str());

    const auto by_identity = lsx::storage::Resolve("qwen3.5-9b");
    REQUIRE(by_identity.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(by_identity.model_type == "qwen3.5-9b");
    CHECK(by_identity.org == "Qwen");
    CHECK(by_identity.model_dir == "Qwen3.5-9B");
    CHECK(by_identity.profile == "nvfp4");

    // And the store's own spelling still resolves to the same tree.
    const auto by_store_name = lsx::storage::Resolve("Qwen3.5-9B");
    REQUIRE(by_store_name.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(by_store_name.dir == by_identity.dir);
}

TEST_CASE("a model_type with no artifact says 'not converted', not 'not found'",
          "[storage_roots]") {
    // The registry identity is what the gates, the CLI defaults and every
    // document use. Answering "no artifact and no source checkpoint" for a model
    // whose directory is right there sends an operator hunting a typo instead of
    // running the conversion.
    TempRoot art;
    // A model directory that exists with no profile in it.
    fs::create_directories(art.path() / "Qwen" / "Qwen3.5-9B");
    ScopedEnv a("LSX_ARTIFACT_ROOT", art.path().c_str());

    // The store spelling still reports source-only by directory name.
    const auto by_name = lsx::storage::Resolve("Qwen3.5-9B");
    CHECK(by_name.status == lsx::storage::ResolveStatus::kSourceOnly);
    CHECK(by_name.detail.find("no converted artifact") != std::string::npos);

    // And the identity of a model that IS converted elsewhere in the tree.
    TempRoot art2;
    MakeArtifact(art2.path() / "microsoft" / "phi-4" / "nvfp4", "phi-4", "phi",
                 "microsoft", "phi-4", "nvfp4");
    fs::create_directories(art2.path() / "microsoft" / "phi-4" / "bf16");
    ScopedEnv a2("LSX_ARTIFACT_ROOT", art2.path().c_str());
    // `phi-4` resolves to its one real profile; the empty bf16 dir is not one.
    const auto phi = lsx::storage::Resolve("phi-4");
    REQUIRE(phi.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(phi.profile == "nvfp4");
    CHECK(phi.org == "microsoft");
}

TEST_CASE("a path inside an artifact resolves to the artifact", "[storage_roots]") {
    // `ls` gives you file paths: pointing --model at model.index.json (or any
    // file the artifact carries) must name the same tree, not "not found". The
    // loop that was meant to accept this iterated `{given, given}` — the same
    // candidate twice — so the parent path the comment describes was never
    // tried.
    TempRoot art;
    MakeArtifact(art.path() / "Qwen" / "Qwen3.5-9B" / "nvfp4", "qwen3.5-9b",
                 "qwen", "Qwen", "Qwen3.5-9B", "nvfp4");
    ScopedEnv a("LSX_ARTIFACT_ROOT", art.path().c_str());

    const auto inside = lsx::storage::Resolve(
        (art.path() / "Qwen" / "Qwen3.5-9B" / "nvfp4" / "model.index.json").string());
    REQUIRE(inside.status == lsx::storage::ResolveStatus::kArtifact);
    CHECK(inside.profile == "nvfp4");
    CHECK(inside.model_dir == "Qwen3.5-9B");
}

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

// ── Placeholder directories ────────────────────────────────────────────────
//
// An empty <org>/<model> directory records where an evicted checkpoint belongs.
// It is NOT a checkpoint (nothing can load from it), but "absent" is the wrong
// message: the operator created the directory on purpose, and a diagnostic that
// ignores it sends them looking for a typo instead of at a missing download.

TEST_CASE("an empty placeholder directory is reported as a placeholder",
          "[storage_roots]") {
    TempRoot root;
    fs::create_directories(root.path() / "moonshotai" / "Kimi-K3");  // empty
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    CHECK_FALSE(lsx::storage::FindSourceModel("Kimi-K3", &why).has_value());
    CHECK(why.find("placeholder") != std::string::npos);
    CHECK(why.find("moonshotai") != std::string::npos);
    // And it must say what to do, not just that it is empty.
    CHECK(why.find("not downloaded") != std::string::npos);
}

TEST_CASE("a placeholder never becomes a checkpoint", "[storage_roots]") {
    // The distinction matters: ListSourceModels feeds Resolve, and an empty dir
    // resolving to kSourceOnly would send a conversion at a directory with no
    // weights at all.
    TempRoot root;
    fs::create_directories(root.path() / "acme" / "evicted");
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());
    CHECK(lsx::storage::ListSourceModels().empty());

    const auto r = lsx::storage::Resolve("evicted");
    CHECK(r.status == lsx::storage::ResolveStatus::kNotFound);
    CHECK(r.detail.find("placeholder") != std::string::npos);
}

TEST_CASE("a name that matches nothing at all still says just that",
          "[storage_roots]") {
    TempRoot root;
    fs::create_directories(root.path() / "acme" / "present");
    std::ofstream(root.path() / "acme" / "present" / "config.json") << "{}";
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    std::string why;
    CHECK_FALSE(lsx::storage::FindSourceModel("typo-name", &why).has_value());
    CHECK(why.find("placeholder") == std::string::npos);
    CHECK(why.find("typo-name") != std::string::npos);
}

TEST_CASE("ListPlaceholderDirs lists every placeholder and nothing else",
          "[storage_roots]") {
    TempRoot root;
    MakeCheckpoint(root.path() / "Qwen" / "real-model");
    fs::create_directories(root.path() / "zai-org" / "evicted-a");
    fs::create_directories(root.path() / "moonshotai" / "evicted-b");
    fs::create_directories(root.path() / "moonshotai" / "empty-org-less");  // also a placeholder
    ScopedEnv e("LSX_SOURCE_ROOT", root.path().c_str());

    const auto all = lsx::storage::ListPlaceholderDirs("");
    REQUIRE(all.size() == 3);
    // Sorted, and a checkpoint is never listed.
    for (const auto &p : all) {
        INFO(p.string());
        CHECK(p.filename() != "real-model");
    }

    const auto one = lsx::storage::ListPlaceholderDirs("evicted-b");
    REQUIRE(one.size() == 1);
    CHECK(one.front() == root.path() / "moonshotai" / "evicted-b");

    // And a real checkpoint has no placeholder entry.
    CHECK(lsx::storage::ListPlaceholderDirs("real-model").empty());
}
