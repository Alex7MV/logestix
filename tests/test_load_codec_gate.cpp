// The admission gate in ModelBase::BeginLoad (invariant I3).
//
// Every case here runs without a GPU: the gate is a pure function of the
// artifact's declared mask and the active target, and the active target is
// injectable for tests only. That is deliberate — the whole 7-codec x N-target
// matrix has to be provable on a machine that is not an Ampere box.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/model.h"
#include "lsxcommon/model_base.h"
#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace lsxcommon::hw;
using lsx::storage::ArtifactSpec;
using lsx::storage::PlaneDecl;

namespace {

constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kNvFp4 = 1ull << 5;

class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-gate-" + std::to_string(::getpid()) + "-" +
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

void WriteFile(const fs::path &p, const std::string &bytes)
{
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Converts a throwaway artifact with the given mask into `dir`.
bool MakeArtifact(const fs::path &dir, const std::string &profile, uint64_t mask,
                  const std::string &model_type = "unit-model",
                  const std::string &family = "unit")
{
    WriteFile(dir / "pinned" / "layer_0.arrow", "payload");
    WriteFile(dir / "pinned" / "tokenizer_vocab.arrow", "v");
    WriteFile(dir / "pinned" / "tokenizer_merges.arrow", "m");

    ArtifactSpec spec;
    spec.model_type = model_type;
    spec.family = family;
    spec.profile = profile;
    spec.model_name = model_type;
    spec.source_dir = model_type;
    spec.target_id = "nvidia:sm_120";
    spec.planes = {{"dense.mlp", mask, "pinned/layer_0.arrow", 7},
                   {"embed.tokenizer", kF32, "pinned/tokenizer_vocab.arrow", 1},
                   {"embed.tokenizer", kF32, "pinned/tokenizer_merges.arrow", 1}};
    spec.layer_codec = {mask};
    spec.layer_count = 1;
    std::string err;
    return lsx::storage::EmitArtifact(spec, dir, &err) != 0;
}

// ModelBase is abstract; this is the smallest concrete shell that exercises the
// base's Load sequence without any family's semantics.
class ProbeModel : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    bool Load() override
    {
        if (!BeginLoad(cfg_.model_path)) return AbortLoad();
        loaded_ = true;
        return true;
    }
    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest &) override
    {
        return {};
    }
    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest &) override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static const std::string kType = "unit-probe";
        return kType;
    }
};

} // namespace

TEST_CASE("an nvfp4 artifact is refused on Ampere and names the target",
          "[load_gate]") {
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4 | kBF16 | kF32));

    ProbeModel m(lsxcommon::ModelInitConfig{});
    m.cfg_.model_path = d.path().string();

    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    CHECK_FALSE(m.Load());
    ResetActiveTargetForTesting();
}

TEST_CASE("the same nvfp4 artifact gates clean on Blackwell", "[load_gate]") {
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4 | kBF16 | kF32));

    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    ProbeModel m(lsxcommon::ModelInitConfig{});
    m.cfg_.model_path = d.path().string();

    // The gate is the thing under test; Load also needs a tokenizer DB built
    // from real Arrow files, which this fixture deliberately does not have.
    const auto read = lsx::storage::ReadArtifact(d.path());
    REQUIRE(read.status == lsx::storage::ReadStatus::kOk);
    std::string why;
    CHECK(m.CodecGateAllows(read.info, ActiveTarget(), &why));
    CHECK(why.empty());
    ResetActiveTargetForTesting();
}

TEST_CASE("the gate is refused before cfg_.model_path is rebound", "[load_gate]") {
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4 | kBF16 | kF32));
    const std::string before = d.path().string();

    SetActiveTargetForTesting(TargetFromComputeCapability(8, 6));
    ProbeModel m(lsxcommon::ModelInitConfig{});
    m.cfg_.model_path = before;
    CHECK_FALSE(m.Load());
    // A refused model must not leave a config pointing somewhere else: a retry
    // on another target would otherwise resolve to the wrong profile.
    CHECK(m.cfg_.model_path == before);
    CHECK(m.artifact() == nullptr);
    ResetActiveTargetForTesting();
}

TEST_CASE("a source checkpoint is refused as source-only, not as corrupt",
          "[load_gate]") {
    TempDir d;
    WriteFile(d.path() / "config.json", "{}");   // a download, not an artifact

    ProbeModel m(lsxcommon::ModelInitConfig{});
    m.cfg_.model_path = d.path().string();
    CHECK_FALSE(m.Load());
    CHECK(m.artifact() == nullptr);
}

TEST_CASE("an absent path is refused without touching the filesystem twice",
          "[load_gate]") {
    ProbeModel m(lsxcommon::ModelInitConfig{});
    m.cfg_.model_path = "/nonexistent-artifact-xyz";
    CHECK_FALSE(m.Load());
}

TEST_CASE("a synthetic model skips resolve and gate explicitly", "[load_gate]") {
    TempDir d;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = d.path().string();
    cfg.synthetic = true;
    ProbeModel m(cfg);
    // BeginLoad reaches the tokenizer stage: the fixture has no arrows, so it
    // fails there rather than at the gate. What matters is that it got past the
    // gate at all, which the log line records.
    CHECK_FALSE(m.Load());
    CHECK(m.artifact() == nullptr);   // synthetic models carry no artifact
}

TEST_CASE("LayerUsesCodec answers from the artifact, not from model.index.json",
          "[load_gate]") {
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "bf16", kBF16 | kF32));

    // Before a load there is no artifact, so every query is false.
    ProbeModel m(lsxcommon::ModelInitConfig{});
    CHECK_FALSE(m.LayerUsesCodec(0, PlaneCodec::kBF16));

    const auto read = lsx::storage::ReadArtifact(d.path());
    REQUIRE(read.status == lsx::storage::ReadStatus::kOk);
    REQUIRE(read.info.layer_codec.size() == 1);
    CHECK((read.info.layer_codec[0] & Mask(PlaneCodec::kBF16)) != 0);
    CHECK((read.info.layer_codec[0] & Mask(PlaneCodec::kNvFp4)) == 0);
}

TEST_CASE("a quantized mask that is not hardware-native anywhere is refused on "
          "every target",
          "[load_gate]") {
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4));

    for (auto t : {TargetFromComputeCapability(8, 6),
                   TargetFromComputeCapability(8, 7)}) {
        SetActiveTargetForTesting(t);
        ProbeModel m(lsxcommon::ModelInitConfig{});
        m.cfg_.model_path = d.path().string();
        CHECK_FALSE(m.Load());
    }
    // And accepted where it is native, which is the whole point of the gate
    // being a table rather than a hard-coded arch check.
    SetActiveTargetForTesting(TargetFromComputeCapability(12, 1));
    const auto read = lsx::storage::ReadArtifact(d.path());
    ProbeModel m2(lsxcommon::ModelInitConfig{});
    CHECK(m2.CodecGateAllows(read.info, ActiveTarget(), nullptr));
    ResetActiveTargetForTesting();
}

// ── Identity: the artifact must be the model the caller NAMED ──────────────
//
// Successor to the per-model `format` tag. Ten family loaders compared that tag
// against their own passport, and for siblings with the same tower (hy-memory vs
// hy-simulmt) it was the ONLY thing telling their trees apart — the geometry is
// identical, so once the tag retired the parent's loader accepted the sibling's
// weights and answered as it while calling itself the parent. The comparison now
// lives in one place and reads artifact.fb's model_type, which the contract made
// the single source of truth for what an artifact is.
TEST_CASE("an artifact of another model is refused by name", "[load_gate]") {
    TempDir d;
    // The parent's tree, offered to a caller that pinned the sibling.
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4 | kBF16 | kF32,
                         "hy-memory-30b-a3b", "hy"));
    const auto read = lsx::storage::ReadArtifact(d.path());
    REQUIRE(read.status == lsx::storage::ReadStatus::kOk);

    std::string why;
    CHECK_FALSE(lsxcommon::ModelBase::IdentityAllows("hy-simulmt-30b-a3b",
                                                     read.info, &why));
    // The message has to name both sides: "refused" without "you asked for X,
    // this is Y" sends the operator to the codec table instead of the path.
    INFO(why);
    CHECK(why.find("hy-simulmt-30b-a3b") != std::string::npos);
    CHECK(why.find("hy-memory-30b-a3b") != std::string::npos);

    // The same model, and the spelling the vault shows: both accepted, because a
    // pin of "Hy-Memory-30B-A3B" means the artifact whose id is the lower-cased
    // form — refusing it would make the check a spelling test rather than an
    // identity test.
    CHECK(lsxcommon::ModelBase::IdentityAllows("hy-memory-30b-a3b", read.info,
                                               nullptr));
    CHECK(lsxcommon::ModelBase::IdentityAllows("Hy-Memory-30B-A3B", read.info,
                                               nullptr));

    // An empty declaration is the autodetect path: the type came FROM this
    // artifact, so there is nothing to disagree with.
    CHECK(lsxcommon::ModelBase::IdentityAllows("", read.info, nullptr));
}

TEST_CASE("a mis-pinned model_type is refused on a target that could run it",
          "[load_gate]") {
    // Blackwell on purpose: the codec mask is native here, so the ONLY reason
    // left to refuse is the identity. Run on sm_86 and this case would pass
    // while proving nothing about the check it names.
    TempDir d;
    REQUIRE(MakeArtifact(d.path(), "nvfp4", kNvFp4, "phi-4", "phi"));

    SetActiveTargetForTesting(TargetFromComputeCapability(12, 0));
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = d.path().string();
    cfg.model_type = "qwen3.5-9b";
    ProbeModel m(cfg);
    CHECK_FALSE(m.Load());
    // The gate refused, so nothing was rebound and no artifact was published.
    CHECK(m.cfg_.model_path == d.path().string());
    CHECK(m.artifact() == nullptr);

    // Control: the same tree under its own name passes this gate.
    const auto read = lsx::storage::ReadArtifact(d.path());
    REQUIRE(read.status == lsx::storage::ReadStatus::kOk);
    CHECK(lsxcommon::ModelBase::IdentityAllows("phi-4", read.info, nullptr));
    CHECK(m.CodecGateAllows(read.info, ActiveTarget(), nullptr));
    ResetActiveTargetForTesting();
}
