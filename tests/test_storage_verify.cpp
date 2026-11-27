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

// ── Shape-first classification (the bug the first real conversion exposed) ──
//
// A DenseNvfp4 file carries a `dtype` column that holds the dtype the tensor was
// QUANTIZED FROM ("BF16"), not the storage form. Classifying by that column
// reports an NVFP4 plane as BF16 — which is exactly what happened: phi-4 wrote
// 40 NVFP4 planes and artifact.fb recorded mask=2. The codec is the SCHEMA SHAPE.
void WriteDenseNvfp4Arrow(const fs::path &path,
                          const std::vector<std::string> &source_dtypes)
{
    auto name_b = std::make_shared<arrow::StringBuilder>();
    auto shape_b = std::make_shared<arrow::StringBuilder>();
    auto dtype_b = std::make_shared<arrow::StringBuilder>();
    auto packed_b = std::make_shared<arrow::BinaryBuilder>();
    auto scales_b = std::make_shared<arrow::BinaryBuilder>();
    auto gscale_b = std::make_shared<arrow::FloatBuilder>();

    for (const auto &dt : source_dtypes) {
        REQUIRE(name_b->Append("w").ok());
        REQUIRE(shape_b->Append("[2]").ok());
        REQUIRE(dtype_b->Append(dt).ok());          // the SOURCE dtype
        REQUIRE(packed_b->Append("pp", 2).ok());    // E2M1 nibbles
        REQUIRE(scales_b->Append("s", 1).ok());     // UE4M3 block scales
        REQUIRE(gscale_b->Append(1.0f).ok());
    }
    std::shared_ptr<arrow::Array> n, s, d, p, sc, g;
    REQUIRE(name_b->Finish(&n).ok());
    REQUIRE(shape_b->Finish(&s).ok());
    REQUIRE(dtype_b->Finish(&d).ok());
    REQUIRE(packed_b->Finish(&p).ok());
    REQUIRE(scales_b->Finish(&sc).ok());
    REQUIRE(gscale_b->Finish(&g).ok());

    auto schema = arrow::schema(
        {arrow::field("name", arrow::utf8()), arrow::field("shape", arrow::utf8()),
         arrow::field("dtype", arrow::utf8()), arrow::field("packed", arrow::binary()),
         arrow::field("scales", arrow::binary()),
         arrow::field("global_scale", arrow::float32())});
    auto batch = arrow::RecordBatch::Make(schema, n->length(), {n, s, d, p, sc, g});

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

TEST_CASE("a DenseNvfp4 plane is NVFP4 even though its dtype column says BF16",
          "[storage_verify][shape]") {
    TempDir d;
    auto spec = BaseSpec(d.path());
    // Exactly phi-4's layout: a 6-column packed file whose dtype says BF16.
    WriteDenseNvfp4Arrow(d.path() / "pinned" / "layer_0_nvfp4.arrow", {"BF16"});
    spec.planes.push_back({"dense.nvfp4", kNvFp4, "pinned/layer_0_nvfp4.arrow", 3});
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);

    const auto r = VerifyArtifact(d.path(), false);
    for (const auto &p : r.problems) INFO(p);
    CHECK(r.ok);
    CHECK((r.actual_mask & kNvFp4) != 0);
}

TEST_CASE("the scan reports NVFP4 for a packed plane, not the source dtype",
          "[storage_verify][shape]") {
    // The same defect on the converter side: a scan that trusted dtype would
    // declare a bf16 artifact for a tree full of NVFP4 nibbles.
    TempDir d;
    BaseSpec(d.path());
    WriteDenseNvfp4Arrow(d.path() / "pinned" / "layer_0_nvfp4.arrow", {"BF16", "F32"});
    const auto s = ScanArtifactTree(d.path());
    for (const auto &p : s.problems) INFO(p);
    REQUIRE(s.ok);
    CHECK((s.mask & kNvFp4) != 0);
}

// ── The chat template: a recorded path and a recorded digest, both checked ──
//
// Invariant I5 says the artifact is self-sufficient, so the template it records
// must be inside it. Nothing checked that until the bootstrap conversion of a
// real checkpoint made it visible: the emitter looked for the template under the
// ARTIFACT directory, where no converter writes it, so every real artifact
// recorded chat_template = "" while its source had one — and, because a recorded
// digest that no reader compares is a hollow field, a drifted template verified
// clean. Both halves of that are pinned here.
namespace {
void WriteTemplate(const fs::path &dir, const std::string &text)
{
    fs::create_directories(dir);
    std::ofstream(dir / "chat_template.jinja") << text;
}

ArtifactSpec SpecWithTemplate(const fs::path &dir, const fs::path &src)
{
    WriteTemplate(src, "{{ messages }}");
    auto spec = BaseSpec(dir);
    spec.chat_template_src = (src / "chat_template.jinja").string();
    return spec;
}
} // namespace

TEST_CASE("an artifact whose recorded chat template is gone is refused",
          "[storage_verify][template]") {
    TempDir d, src;
    auto spec = SpecWithTemplate(d.path(), src.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    REQUIRE(ReadArtifact(d.path()).info.chat_template == "chat_template.jinja");
    CHECK(VerifyArtifact(d.path(), false).ok);

    // The artifact still claims to ship it. A loader then fails at the first
    // framed prompt instead of at load time, which is the wrong place to learn.
    fs::remove(d.path() / "chat_template.jinja");
    const auto r = VerifyArtifact(d.path(), false);
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems) {
        INFO(p);
        if (p.find("chat template") != std::string::npos) named = true;
    }
    CHECK(named);
}

TEST_CASE("a template supplied as inline text verifies like a copied one",
          "[storage_verify][template]") {
    // The phi-4 shape end to end on a tree with real Arrow planes: no
    // chat_template.jinja in the source, the template inside tokenizer_config.
    // The emitter writes it into the artifact and digests the bytes it wrote, so
    // the deep pass must agree with it.
    TempDir d;
    auto spec = BaseSpec(d.path());
    spec.chat_template_text = "{{ inline }}";
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    REQUIRE(fs::exists(d.path() / "chat_template.jinja"));

    const auto info = ReadArtifact(d.path()).info;
    CHECK(info.chat_template == "chat_template.jinja");
    CHECK(info.has_chat_template_digest);
    const auto deep = VerifyArtifact(d.path(), true);
    std::string joined;
    for (const auto &p : deep.problems) joined += p + "\n";
    INFO("problems:\n" << joined);
    CHECK(deep.ok);

    // And the digest is of the WRITTEN bytes, not of some source file.
    WriteTemplate(d.path(), "{{ tampered }}");
    const auto after = VerifyArtifact(d.path(), true);
    REQUIRE_FALSE(after.ok);
    CHECK(after.problems.size() == 1);
}

TEST_CASE("a chat template that drifted from its recorded digest is caught",
          "[storage_verify][template]") {
    TempDir d, src;
    auto spec = SpecWithTemplate(d.path(), src.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    CHECK(VerifyArtifact(d.path(), true).ok);

    // Same path, different bytes: the shallow pass has nothing to compare (it
    // does not hash), the deep pass must catch it. Without this the digest the
    // writer stores is decorative.
    WriteTemplate(d.path(), "{{ completely different }}");
    CHECK(VerifyArtifact(d.path(), false).ok);
    const auto deep = VerifyArtifact(d.path(), true);
    REQUIRE_FALSE(deep.ok);
    bool named = false;
    for (const auto &p : deep.problems) {
        INFO(p);
        if (p.find("chat template") != std::string::npos) named = true;
    }
    CHECK(named);
}
