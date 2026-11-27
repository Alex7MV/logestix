// VerifyArtifact: rules D1-D5 and the "no ballast" rule.
//
// Each case is a defect that would otherwise ship silently: a converter that
// kept the old ambiguous U8 spelling, a metadata plane that holds weights, a
// mask that no longer describes the bytes, a plane nobody declared, or a file
// the load path cannot reach. With no reference artifacts left in the tree,
// these are the only independent check that a converter wrote what it claimed.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/hardware_target.h"
#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>

#include <cstdint>
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
constexpr uint64_t kMxFp8 = 1ull << 4;
constexpr uint64_t kNvFp4 = 1ull << 5;

class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-verify-" + std::to_string(::getpid()) + "-" +
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

// Writes a PinnedSchema-shaped Arrow file: name/shape/dtype/data per row.
void WritePinnedArrow(const fs::path &path,
                      const std::vector<std::pair<std::string, std::string>> &rows,
                      const std::string &payload = "payload")
{
    auto name_b = std::make_shared<arrow::StringBuilder>();
    auto shape_b = std::make_shared<arrow::StringBuilder>();
    auto dtype_b = std::make_shared<arrow::StringBuilder>();
    auto data_b = std::make_shared<arrow::BinaryBuilder>();

    for (const auto &[name, dtype] : rows) {
        REQUIRE(name_b->Append(name).ok());
        REQUIRE(shape_b->Append("[2]").ok());
        REQUIRE(dtype_b->Append(dtype).ok());
        REQUIRE(data_b->Append(payload.data(),
                               static_cast<int32_t>(payload.size())).ok());
    }
    std::shared_ptr<arrow::Array> name_a, shape_a, dtype_a, data_a;
    REQUIRE(name_b->Finish(&name_a).ok());
    REQUIRE(shape_b->Finish(&shape_a).ok());
    REQUIRE(dtype_b->Finish(&dtype_a).ok());
    REQUIRE(data_b->Finish(&data_a).ok());

    auto schema = arrow::schema({arrow::field("name", arrow::utf8()),
                                arrow::field("shape", arrow::utf8()),
                                arrow::field("dtype", arrow::utf8()),
                                arrow::field("data", arrow::binary())});
    auto batch = arrow::RecordBatch::Make(schema, name_a->length(),
                                          {name_a, shape_a, dtype_a, data_a});

    fs::create_directories(path.parent_path());
    auto out_res = arrow::io::FileOutputStream::Open(path.string());
    REQUIRE(out_res.ok());
    auto out = *out_res;
    auto writer_res = arrow::ipc::MakeFileWriter(out.get(), schema);
    REQUIRE(writer_res.ok());
    auto writer = *writer_res;
    REQUIRE(writer->WriteRecordBatch(*batch).ok());
    REQUIRE(writer->Close().ok());
    REQUIRE(out->Close().ok());
}

// A minimal valid artifact: one bf16 plane, one f32 plane, tokenizer arrows.
ArtifactSpec BaseSpec(const fs::path &dir)
{
    WritePinnedArrow(dir / "pinned" / "layer_0.arrow",
                     {{"attn.q.weight", "BF16"}, {"attn.q.weight.bias", "BF16"}});
    WritePinnedArrow(dir / "pinned" / "layer_1.arrow",
                     {{"norm.weight", "F32"}});
    WritePinnedArrow(dir / "pinned" / "tokenizer_vocab.arrow", {{"v", "F32"}});
    WritePinnedArrow(dir / "pinned" / "tokenizer_merges.arrow", {{"m", "F32"}});

    ArtifactSpec s;
    s.model_type = "phi-4";
    s.family = "phi";
    s.profile = "bf16";
    s.model_name = "Phi-4";
    s.source_dir = "phi-4";
    s.target_id = "nvidia:sm_86";
    s.planes = {
        {"dense.mlp", kBF16, "pinned/layer_0.arrow", 16},
        {"norm", kF32, "pinned/layer_1.arrow", 8},
        {"embed.tokenizer", kF32, "pinned/tokenizer_vocab.arrow", 8},
        {"embed.tokenizer", kF32, "pinned/tokenizer_merges.arrow", 8},
    };
    s.layer_codec = {kBF16, kF32};
    s.layer_count = 2;
    return s;
}

} // namespace

TEST_CASE("a well-formed artifact verifies, shallow and deep", "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) != 0);
    INFO(err);

    const auto shallow = VerifyArtifact(d.path(), false);
    for (const auto &p : shallow.problems) INFO(p);
    CHECK(shallow.ok);

    const auto deep = VerifyArtifact(d.path(), true);
    for (const auto &p : deep.problems) INFO(p);
    CHECK(deep.ok);
}

TEST_CASE("a plane appearing after the mask was declared is caught",
          "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);

    // The exact I1 hole: content the declaration does not mention.
    WritePinnedArrow(d.path() / "pinned" / "layer_9_nvfp4.arrow",
                     {{"extra.weight", "NVFP4"}});
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    CHECK(r.problems[0].find("layer_9_nvfp4.arrow") != std::string::npos);
}

TEST_CASE("a plane whose bytes changed is caught by the deep pass",
          "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    CHECK(VerifyArtifact(d.path(), false).ok);

    WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow",
                     {{"attn.q.weight", "BF16"}}, "different-payload");
    const auto deep = VerifyArtifact(d.path(), true);
    REQUIRE_FALSE(deep.ok);
    bool saw_digest = false;
    for (const auto &p : deep.problems)
        if (p.find("sha256") != std::string::npos) saw_digest = true;
    CHECK(saw_digest);
}

TEST_CASE("D1: a weight plane spelled U8 is rejected, naming NVFP4",
          "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    // The pre-contract spelling: U8 nibbles + U8 scales, codec only inferable
    // from a sibling's existence.
    WritePinnedArrow(d.path() / "pinned" / "layer_2.arrow",
                     {{"mlp.down.weight.packed", "U8"},
                      {"mlp.down.weight.scales", "U8"}});
    spec.planes.push_back({"dense.mlp", kNvFp4, "pinned/layer_2.arrow", 24});
    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) != 0);
    INFO(err);

    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("U8") != std::string::npos &&
            p.find("NVFP4") != std::string::npos)
            named = true;
    CHECK(named);
}

TEST_CASE("D3/D4: metadata may not hold weight-like tensors", "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    WritePinnedArrow(d.path() / "pinned" / "meta.arrow",
                     {{"sneaky.weight", "BF16"}});
    spec.planes.push_back({"meta", kBF16, "pinned/meta.arrow", 8});
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);

    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("role=meta") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("D3: metadata without weight-like content is accepted",
          "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    // An index table: I64, not a weight.
    WritePinnedArrow(d.path() / "pinned" / "idx.arrow", {{"indices", "I64"}});
    spec.planes.push_back({"meta", 0, "pinned/idx.arrow", 16});
    // A meta plane declares no codec bit; the emitter rejects codec_bit == 0
    // for non-meta planes only, so this must be accepted.
    std::string err;
    const uint64_t mask = EmitArtifact(spec, d.path(), &err);
    INFO(err);
    REQUIRE(mask == (kBF16 | kF32));
}

TEST_CASE("a directory the load path cannot reach is a defect (no ballast)",
          "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    // v1 ships no ballast: an MTP head would be unreachable at inference.
    WritePinnedArrow(d.path() / "mtp" / "layer_032.arrow", {{"x", "F8_E4M3"}});
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("mtp/") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("a declared plane that is missing is caught", "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    fs::remove(d.path() / "pinned" / "layer_1.arrow");
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("missing") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("a wrongly-shaped layer_codec is caught", "[storage_verify]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    spec.layer_codec = {kBF16};   // says 2 layers, maps 1
    spec.layer_count = 2;
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("layer_codec") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("the verifier's codec bits match lsxcommon's enum", "[storage_verify]") {
    // lsxstorage sits below lsxcommon and duplicates the bit values; this is
    // the guard that the duplication cannot drift.
    using namespace lsxcommon::hw;
    CHECK(kF32 == Mask(PlaneCodec::kF32));
    CHECK(kBF16 == Mask(PlaneCodec::kBF16));
    CHECK(kI8Sym == Mask(PlaneCodec::kI8Sym));
    CHECK(kMxFp8 == Mask(PlaneCodec::kMxFp8));
    CHECK(kNvFp4 == Mask(PlaneCodec::kNvFp4));
}

// ── ScanArtifactTree: the converter-side half of the contract ─────────────
//
// The forge path has no hand-written plane table per family, so it scans what
// the converter wrote and derives the planes from the bytes. The profile then
// cross-checks: two different inputs, so agreement means something.

TEST_CASE("scan derives the mask and plane list from a tree with no artifact.fb",
          "[storage_verify][scan]") {
    TempDir d;
    BaseSpec(d.path());   // writes pinned/layer_0.arrow (BF16) + layer_1.arrow (F32)
    // The scan runs BEFORE the emitter, so there is deliberately no artifact.fb.
    REQUIRE_FALSE(IsArtifactDir(d.path()));

    const auto s = ScanArtifactTree(d.path());
    for (const auto &p : s.problems) INFO(p);
    REQUIRE(s.ok);
    CHECK((s.mask & kBF16) != 0);
    CHECK((s.mask & kF32) != 0);
    CHECK((s.mask & kNvFp4) == 0);
    CHECK(s.layer_count == 2);
    REQUIRE_FALSE(s.planes.empty());
    for (const auto &p : s.planes) CHECK(p.has_digest);
}

TEST_CASE("scan refuses a U8 weight spelling before the emitter sees it",
          "[storage_verify][scan]") {
    TempDir d;
    BaseSpec(d.path());
    WritePinnedArrow(d.path() / "pinned" / "layer_2.arrow",
                     {{"mlp.down.weight.packed", "U8"}});
    const auto s = ScanArtifactTree(d.path());
    REQUIRE_FALSE(s.ok);
    bool named = false;
    for (const auto &p : s.problems)
        if (p.find("U8") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("scan refuses a file the load path cannot reach", "[storage_verify][scan]") {
    TempDir d;
    BaseSpec(d.path());
    WritePinnedArrow(d.path() / "mtp" / "layer_032.arrow", {{"x", "F8_E4M3"}});
    const auto s = ScanArtifactTree(d.path());
    REQUIRE_FALSE(s.ok);
    bool named = false;
    for (const auto &p : s.problems)
        if (p.find("mtp/") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("scan of a non-directory fails instead of returning an empty tree",
          "[storage_verify][scan]") {
    const auto s = ScanArtifactTree(fs::path("/nonexistent-artifact-root"));
    CHECK_FALSE(s.ok);
    CHECK_FALSE(s.problems.empty());
}
