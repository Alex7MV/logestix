// Target-aware profile selection (design §2.7, invariant I4).
//
// With one profile per model, "which artifact do I load" was answered by
// directory iteration order — unspecified, and the comment beside it claimed the
// sorted order ListProfiles uses. With two profiles it becomes a coin flip, and
// the gate then refuses the arbitrarily chosen one and prints the usable list:
// a good error message standing in for the selection I4 requires.
//
// The storage layer stays codec-agnostic: it takes the capability as a plain
// mask and the preference as an ordered list of names. lsxcommon supplies both
// (SupportedCodecs(ActiveTarget()) and the rank table), so "nvfp4 before bf16"
// is a policy in one place rather than a filesystem accident.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"
#include "lsxstorage/roots.h"
#include "lsxstorage/resolve.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace lsx::storage;

namespace {

constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kI8Sym = 1ull << 3;
constexpr uint64_t kNvFp4 = 1ull << 5;

// RAII env override: selection reads the store root, so every case has to point
// it at a scratch tree and put it back.
class ScopedEnv {
public:
    ScopedEnv(const char *name, const char *value) : name_(name)
    {
        if (const char *old = std::getenv(name)) {
            had_ = true;
            old_ = old;
        }
        if (value == nullptr) {
            unsetenv(name);
            return;
        }
        setenv(name, value, 1);
    }
    ~ScopedEnv()
    {
        if (had_) setenv(name_.c_str(), old_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
    std::string name_;
    std::string old_;
    bool had_ = false;
};

class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-select-" + std::to_string(::getpid()) + "-" +
                 std::to_string(counter++));
        fs::create_directories(path_);
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    const fs::path &path() const { return path_; }
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_;
};

// Writes a real artifact for `<root>/<org>/<model>/<profile>` whose mask is
// `mask`.
//
// The plane files hold no real Arrow payload: the EMITTER does not read tensors
// (by design — it declares, the verifier infers), so it only requires each
// declared plane to exist and digests it. That keeps this fixture about
// SELECTION, which is what it tests; codec inference is covered by
// test_storage_verify.
void MakeProfile(const fs::path &root, const std::string &org,
                 const std::string &model, const std::string &profile,
                 uint64_t mask)
{
    const fs::path dir = root / org / model / profile;
    fs::create_directories(dir);

    ArtifactSpec s;
    s.model_type = model;
    s.family = org;
    s.profile = profile;
    s.org = org;
    s.model_dir = model;
    s.model_name = model;
    s.source_dir = model;
    s.target_id = "nvidia:sm_120";
    // One plane per codec bit: the emitter checks that every declared plane
    // exists on disk, so the tree has to carry them.
    int n = 0;
    for (uint64_t bit : {kF32, kBF16, kI8Sym, kNvFp4}) {
        if ((mask & bit) == 0) continue;
        const std::string rel = "pinned/layer_" + std::to_string(n) + ".arrow";
        fs::create_directories((dir / rel).parent_path());
        std::ofstream f(dir / rel, std::ios::binary);
        f << "plane-" << n;
        s.planes.push_back({"dense.mlp", bit, rel, 7});
        ++n;
    }
    s.layer_codec = {mask};
    s.layer_count = 1;

    std::string err;
    const uint64_t written = EmitArtifact(s, dir, &err);
    INFO(err);
    REQUIRE(written == mask);
}

} // namespace

TEST_CASE("selection prefers the compressed profile the target can execute",
          "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    // bf16 is executable everywhere; nvfp4 only on Blackwell.
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4",
                kF32 | kBF16 | kNvFp4);

    const std::vector<std::string> rank = {"nvfp4", "int8-w8a16", "bf16"};

    const auto blackwell =
        SelectProfile("microsoft", "phi-4", kF32 | kBF16 | kNvFp4, rank);
    REQUIRE(blackwell.has_value());
    CHECK(blackwell->profile == "nvfp4");
    CHECK(blackwell->status == ResolveStatus::kArtifact);

    const auto ampere = SelectProfile("microsoft", "phi-4", kF32 | kBF16, rank);
    REQUIRE(ampere.has_value());
    CHECK(ampere->profile == "bf16");
}

TEST_CASE("no compatible profile is nullopt, never the first one on disk",
          "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4",
                kF32 | kBF16 | kNvFp4);
    // A CPU target cannot execute NVFP4 at all.
    CHECK_FALSE(SelectProfile("microsoft", "phi-4", kF32 | kBF16,
                              {"nvfp4", "bf16"})
                    .has_value());
    // And a model with no artifacts answers the same way.
    CHECK_FALSE(SelectProfile("microsoft", "not-converted", kF32, {"bf16"})
                    .has_value());
}

TEST_CASE("the same store and target answer the same profile every time",
          "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    // Two profiles the SAME target can run, neither first in directory order:
    // without a rank table this is where iteration order decided.
    MakeProfile(root.path(), "Qwen", "Qwen3.5-9B", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "Qwen", "Qwen3.5-9B", "int8-w8a16",
                kF32 | kBF16 | kI8Sym);

    const std::vector<std::string> rank = {"int8-w8a16", "nvfp4", "bf16"};
    std::string first;
    for (int i = 0; i < 10; ++i) {
        const auto r = SelectProfile("Qwen", "Qwen3.5-9B",
                                     kF32 | kBF16 | kI8Sym, rank);
        REQUIRE(r.has_value());
        if (first.empty()) first = r->profile;
        CHECK(r->profile == first);
    }
    CHECK(first == "int8-w8a16");   // rank wins over the name order
}

TEST_CASE("profiles the rank table does not name come last, by name",
          "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "google", "gemma", "bbb", kF32 | kBF16);
    MakeProfile(root.path(), "google", "gemma", "aaa", kF32 | kBF16);
    // Unknown names are still selectable — the rank table is a preference, not
    // a filter — and they sort by name so two runs agree.
    const auto r = SelectProfile("google", "gemma", kF32 | kBF16,
                                 {"nvfp4", "bf16"});
    REQUIRE(r.has_value());
    CHECK(r->profile == "aaa");
}

TEST_CASE("an explicit profile path is a choice and is never re-ranked",
          "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4",
                kF32 | kBF16 | kNvFp4);

    // A path, a path inside a profile, and the org-qualified profile name all
    // name ONE artifact: selection must not second-guess any of them.
    const std::vector<std::string> given_paths = {
        (root.path() / "microsoft/phi-4/bf16").string(),
        (root.path() / "microsoft/phi-4/bf16/model.index.json").string(),
        std::string("microsoft/phi-4/bf16")};
    for (const std::string &given : given_paths) {
        INFO("given: " << given);
        const auto r = Resolve(given);
        INFO(r.detail);
        REQUIRE(r.status == ResolveStatus::kArtifact);
        CHECK(r.profile == "bf16");
    }
}

TEST_CASE("a bare model name selects instead of guessing", "[select][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4",
                kF32 | kBF16 | kNvFp4);

    // Resolve() with the active target: on a Blackwell mask it takes nvfp4; with
    // a bf16-only mask it takes bf16 — the same store, two answers.
    const auto on_blackwell =
        Resolve("phi-4", kF32 | kBF16 | kNvFp4, {"nvfp4", "bf16"});
    REQUIRE(on_blackwell.status == ResolveStatus::kArtifact);
    CHECK(on_blackwell.profile == "nvfp4");

    const auto on_ampere = Resolve("phi-4", kF32 | kBF16, {"nvfp4", "bf16"});
    REQUIRE(on_ampere.status == ResolveStatus::kArtifact);
    CHECK(on_ampere.profile == "bf16");

    // The org-qualified model name behaves the same way.
    const auto qualified = Resolve("microsoft/phi-4", kF32 | kBF16, {"nvfp4", "bf16"});
    REQUIRE(qualified.status == ResolveStatus::kArtifact);
    CHECK(qualified.profile == "bf16");
}
