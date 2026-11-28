// Rule 5: model.index.json owns GEOMETRY, artifact.fb owns CODECS.
//
// The index used to declare five per-layer codec fields (mlp_nvfp4, attn_nvfp4,
// head_nvfp4, expert_nvfp4[], expert_fp8[]) and six files read them to pick a
// kernel per layer. artifact.fb::layer_codec replaced them — but two writers
// kept emitting the keys, two loaders kept reading them, and the arrow verifier
// derived an exponent map from them. That left TWO sources of truth for one
// fact, and a disagreement between them passes the gate on an artifact that
// executes an undeclared codec (design §0.5).
//
// This check makes the split enforceable: an index that still carries any of
// the five keys is refused, naming the field and where the map now lives.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>

#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace lsx::storage;

namespace {

class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-index-codec-" + std::to_string(::getpid()) + "-" +
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

void WriteFile(const fs::path &p, const std::string &text)
{
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << text;
}

// A real PinnedSchema Arrow file: an unreadable payload would make the verifier
// refuse for a reason unrelated to rule 5, and a test that fails for the wrong
// reason proves nothing.
void WritePinnedArrow(const fs::path &p)
{
    arrow::StringBuilder name_b, shape_b, dtype_b;
    arrow::BinaryBuilder data_b;
    REQUIRE(name_b.Append("mlp.up.weight").ok());
    REQUIRE(shape_b.Append("[2]").ok());
    REQUIRE(dtype_b.Append("BF16").ok());
    REQUIRE(data_b.Append("payload", 7).ok());
    std::shared_ptr<arrow::Array> name_a, shape_a, dtype_a, data_a;
    REQUIRE(name_b.Finish(&name_a).ok());
    REQUIRE(shape_b.Finish(&shape_a).ok());
    REQUIRE(dtype_b.Finish(&dtype_a).ok());
    REQUIRE(data_b.Finish(&data_a).ok());

    auto schema = arrow::schema({arrow::field("name", arrow::utf8()),
                                arrow::field("shape", arrow::utf8()),
                                arrow::field("dtype", arrow::utf8()),
                                arrow::field("data", arrow::binary())});
    auto batch = arrow::RecordBatch::Make(schema, 1,
                                          {name_a, shape_a, dtype_a, data_a});

    fs::create_directories(p.parent_path());
    auto out_res = arrow::io::FileOutputStream::Open(p.string());
    REQUIRE(out_res.ok());
    auto out = *out_res;
    auto writer_res = arrow::ipc::MakeFileWriter(out.get(), schema);
    REQUIRE(writer_res.ok());
    auto writer = *writer_res;
    REQUIRE(writer->WriteRecordBatch(*batch).ok());
    REQUIRE(writer->Close().ok());
    REQUIRE(out->Close().ok());
}

// A minimal tree: one codec-bearing Arrow file plus an index the caller writes.
ArtifactSpec MinimalSpec()
{
    ArtifactSpec s;
    s.model_type = "phi-4";
    s.family = "phi";
    s.profile = "nvfp4";
    s.model_name = "Phi-4";
    s.source_dir = "phi-4";
    s.target_id = "nvidia:sm_120";
    s.planes = {{"dense.mlp", 1ull << 1, "pinned/layer_0.arrow", 7}};
    s.layer_codec = {1ull << 1};
    s.layer_count = 1;
    return s;
}

// The index the converters write: geometry only. `extra` is spliced in to
// reproduce a pre-rule-5 index.
std::string IndexJson(const std::string &extra = "")
{
    std::string j = R"({"layout": 2, "num_layers": 1, "hidden_size": 2560,
      "eos_token_id": 2)";
    if (!extra.empty()) j += ", " + extra;
    j += "}";
    return j;
}

std::string Joined(const VerifyReport &r)
{
    std::string out;
    for (const auto &p : r.problems) out += "\n  " + p;
    return out;
}

} // namespace

TEST_CASE("rule 5: an index that still declares a codec is refused",
          "[index_codec][rule5]") {
    // Each of the five fields, one case each: a single-key check would pass the
    // other four, and all five were written by real converters.
    for (const char *key :
         {"mlp_nvfp4", "attn_nvfp4", "head_nvfp4", "expert_nvfp4", "expert_fp8"}) {
        INFO("key: " << key);
        TempDir d;
        WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow");
        auto spec = MinimalSpec();
        std::string err;
        REQUIRE(EmitArtifact(spec, d.path(), &err) != 0);
        INFO(err);

        // The index is written AFTER the artifact so the failure is the rule's,
        // not a missing-plane error.
        WriteFile(d.path() / "model.index.json",
                  IndexJson(std::string("\"") + key + "\": true"));

        const auto r = VerifyArtifact(d.path(), false);
        INFO(Joined(r));
        REQUIRE_FALSE(r.ok);
        bool named = false;
        for (const auto &p : r.problems)
            if (p.find(key) != std::string::npos) named = true;
        CHECK(named);
        // And it says where the map lives now, so the fix is not a guess.
        bool points = false;
        for (const auto &p : r.problems)
            if (p.find("layer_codec") != std::string::npos) points = true;
        CHECK(points);
    }
}

TEST_CASE("rule 5: a geometry-only index verifies clean",
          "[index_codec][rule5]") {
    TempDir d;
    WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow");
    auto spec = MinimalSpec();
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    WriteFile(d.path() / "model.index.json", IndexJson());
    const auto r = VerifyArtifact(d.path(), false);
    INFO(Joined(r));
    CHECK(r.ok);
}

TEST_CASE("rule 5: a key inside a string VALUE is not a declaration",
          "[index_codec][rule5]") {
    // The field names are ordinary prose too — "the per-layer expert_nvfp4 map"
    // in a note must not fail verification. This is why the check parses JSON
    // keys instead of grepping text.
    TempDir d;
    WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow");
    auto spec = MinimalSpec();
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    WriteFile(d.path() / "model.index.json",
              IndexJson(R"("notes": "codecs live in artifact.fb, not expert_nvfp4")"));
    const auto r = VerifyArtifact(d.path(), false);
    INFO(Joined(r));
    CHECK(r.ok);
}

TEST_CASE("rule 5: an unparseable index is refused, not ignored",
          "[index_codec][rule5]") {
    // An index the verifier cannot read cannot be shown to be codec-free, and
    // a check that answers "fine" when it cannot read the file is exactly the
    // silent second source of truth rule 5 exists to remove.
    TempDir d;
    WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow");
    auto spec = MinimalSpec();
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    WriteFile(d.path() / "model.index.json", R"({"layout": 2, "num_layers": )");
    const auto r = VerifyArtifact(d.path(), false);
    INFO(Joined(r));
    REQUIRE_FALSE(r.ok);
    bool named = false;
    for (const auto &p : r.problems)
        if (p.find("model.index.json") != std::string::npos) named = true;
    CHECK(named);
}

TEST_CASE("rule 5: an artifact with no index at all is still verified",
          "[index_codec][rule5]") {
    // Pre-contract trees and syntax fixtures may carry no index; the rule is
    // about an index that EXISTS and disagrees, not about requiring one.
    TempDir d;
    WritePinnedArrow(d.path() / "pinned" / "layer_0.arrow");
    auto spec = MinimalSpec();
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    std::error_code ec;
    fs::remove(d.path() / "model.index.json", ec);
    const auto r = VerifyArtifact(d.path(), false);
    INFO(Joined(r));
    CHECK(r.ok);
}
