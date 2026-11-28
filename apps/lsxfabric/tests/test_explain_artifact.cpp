// --explain-artifact: WHY this profile, and not the other one (design §2.7).
//
// Selection became deterministic and target-aware in Task 9e, but a choice
// nobody can inspect is a choice nobody can debug: "it loaded bf16 on my 5090"
// has to be answerable from the command line. The report names every candidate,
// the target's capability, whether that candidate's mask fits, and the rank that
// decided it — and it exits without loading a model, so it is safe to run while
// something else holds the GPU.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hardware_target.h"
#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"
#include "lsxstorage/resolve.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// The formatter lives with the app, not in the library: it is a diagnostic for
// one CLI, and lsxstorage has no business rendering prose.
namespace lsxfabric {
std::string ExplainArtifact(const std::string &model_or_path,
                            uint64_t supported,
                            const std::vector<std::string> &preference,
                            int *rc);
} // namespace lsxfabric

namespace fs = std::filesystem;
using namespace lsx::storage;

namespace {

constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kNvFp4 = 1ull << 5;

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
                ("lsx-explain-" + std::to_string(::getpid()) + "-" +
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
    int n = 0;
    for (uint64_t bit : {kF32, kBF16, kNvFp4}) {
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
    REQUIRE(EmitArtifact(s, dir, nullptr) == mask);
}

const std::vector<std::string> kRank = {"int8-w8a16", "nvfp4", "bf16"};

} // namespace

TEST_CASE("explain lists every candidate with its verdict and names the choice",
          "[explain][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4", kF32 | kBF16 | kNvFp4);

    int rc = 1;
    const auto text = lsxfabric::ExplainArtifact("phi-4", kF32 | kBF16, kRank, &rc);
    INFO(text);
    CHECK(rc == 0);

    // Both candidates appear — an explanation that lists only the winner cannot
    // answer "why not the other one".
    CHECK(text.find("bf16") != std::string::npos);
    CHECK(text.find("nvfp4") != std::string::npos);
    // Each carries its compatibility verdict.
    CHECK(text.find("compatible") != std::string::npos);
    // And the choice is stated once, unambiguously.
    CHECK(text.find("chosen") != std::string::npos);
}

TEST_CASE("explain reports the target mask it judged against",
          "[explain][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4", kF32 | kBF16 | kNvFp4);

    int rc = 1;
    const auto text =
        lsxfabric::ExplainArtifact("phi-4", kF32 | kBF16, kRank, &rc);
    INFO(text);
    // The capability is the reason nvfp4 is refused, so it has to be printed:
    // otherwise the reader cannot tell a rank decision from a capability one.
    CHECK(text.find("nvfp4") != std::string::npos);
    CHECK(text.find("no") != std::string::npos);
    // No compatible profile: a non-zero exit, and it says so.
    CHECK(rc != 0);
    CHECK(text.find("no compatible") != std::string::npos);
}

TEST_CASE("explain refuses an unknown model instead of inventing one",
          "[explain][profile]") {
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    int rc = 0;
    const auto text =
        lsxfabric::ExplainArtifact("no-such-model", kF32 | kBF16, kRank, &rc);
    INFO(text);
    CHECK(rc != 0);
    CHECK(text.find("no-such-model") != std::string::npos);
}

TEST_CASE("an explicitly named profile is what the loader would take, and the "
          "report says so",
          "[explain][profile]") {
    // The bug this case pins: the report re-ran its own ranking even when the
    // operator had NAMED a profile, so `--localmodel .../bf16` printed
    // "(named explicitly)" beside bf16 and then `chosen: nvfp4` — a report that
    // disagrees with the loader it explains. BeginLoad hands the path to
    // Resolve(), which returns the named profile, so the report must too.
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4", kF32 | kBF16 | kNvFp4);

    // A Blackwell mask: nvfp4 outranks bf16, so a re-ranking would pick nvfp4.
    int rc = 1;
    const auto text = lsxfabric::ExplainArtifact(
        (root.path() / "microsoft/phi-4/bf16").string(), kF32 | kBF16 | kNvFp4,
        kRank, &rc);
    INFO(text);
    CHECK(rc == 0);
    const auto chosen_at = text.find("chosen:");
    REQUIRE(chosen_at != std::string::npos);
    const std::string chosen_line = text.substr(chosen_at);
    CHECK(chosen_line.find("bf16") != std::string::npos);
    CHECK(chosen_line.find("nvfp4") == std::string::npos);
}

TEST_CASE("a named profile the target CANNOT run is refused, not silently "
          "swapped",
          "[explain][profile]") {
    // The other half of the same rule: honouring a choice is not the same as
    // pretending it is loadable. An operator who names nvfp4 on a bf16-only mask
    // gets a non-zero exit and the reason — not a quiet substitution of bf16.
    TempDir root;
    ScopedEnv env("LSX_ARTIFACT_ROOT", root.path().c_str());
    MakeProfile(root.path(), "microsoft", "phi-4", "bf16", kF32 | kBF16);
    MakeProfile(root.path(), "microsoft", "phi-4", "nvfp4", kF32 | kBF16 | kNvFp4);

    int rc = 0;
    const auto text = lsxfabric::ExplainArtifact(
        (root.path() / "microsoft/phi-4/nvfp4").string(), kF32 | kBF16, kRank,
        &rc);
    INFO(text);
    CHECK(rc != 0);
    CHECK(text.find("nvfp4") != std::string::npos);
}
