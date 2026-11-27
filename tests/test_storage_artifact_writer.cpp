// artifact.fb writer: the mask it records must equal the content it wrote.
//
// With every pre-existing converted tree deleted there is no reference artifact
// to compare a converter against, so these cases are the only thing standing
// between a converter bug and a plausible-looking artifact. The emitter is
// required to distrust its own input: a declared plane that is absent, or a
// mask that would not survive a re-read, is a failure and not a warning.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/artifact.h"
#include "lsxstorage/artifact_writer.h"
#include "sha256.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace lsx::storage;

namespace {

constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kNvFp4 = 1ull << 5;
constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kMxFp8 = 1ull << 4;

class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-emit-" + std::to_string(::getpid()) + "-" +
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

// Two weight planes plus the tokenizer arrows the loader expects.
ArtifactSpec MinimalSpec(const fs::path &dir)
{
    WriteFile(dir / "pinned" / "layer_0.arrow", "bf16-weights");
    WriteFile(dir / "pinned" / "layer_1.arrow", "f32-scales");
    WriteFile(dir / "pinned" / "tokenizer_vocab.arrow", "vocab");
    WriteFile(dir / "pinned" / "tokenizer_merges.arrow", "merges");

    ArtifactSpec s;
    s.model_type = "phi-4";
    s.family = "phi";
    s.profile = "bf16";
    s.model_name = "Phi-4";
    s.source_dir = "phi-4";
    s.target_id = "nvidia:sm_86";
    s.planes = {
        {"dense.mlp", kBF16, "pinned/layer_0.arrow", 12},
        {"norm", kF32, "pinned/layer_1.arrow", 10},
    };
    s.layer_codec = {kBF16, kF32};
    s.layer_count = 2;
    s.pinned_tensors = 2;
    return s;
}

} // namespace

TEST_CASE("emit records the OR of its planes and reads back identically",
          "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    std::string err;
    const uint64_t mask = EmitArtifact(spec, d.path(), &err);
    INFO(err);
    REQUIRE(mask == (kBF16 | kF32));

    const auto r = ReadArtifact(d.path());
    REQUIRE(r.status == ReadStatus::kOk);
    CHECK(r.info.used_codecs == mask);
    CHECK(r.info.model_type == "phi-4");
    CHECK(r.info.family == "phi");
    CHECK(r.info.profile == "bf16");
    CHECK(r.info.layer_codec == std::vector<uint64_t>({kBF16, kF32}));
    REQUIRE(r.info.planes.size() == 2);
    CHECK(r.info.planes[0].has_digest);
    CHECK(r.info.planes[1].bytes == 10);
}

TEST_CASE("emit copies the chat template in and digests it", "[storage_writer]") {
    TempDir d, src;
    WriteFile(src.path() / "chat_template.jinja", "{{ messages }}");
    auto spec = MinimalSpec(d.path());
    spec.chat_template_src = (src.path() / "chat_template.jinja").string();
    spec.tokenizer_json_src = (src.path() / "tokenizer.json").string();
    WriteFile(src.path() / "tokenizer.json", "{}");

    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) != 0);
    INFO(err);
    // Self-sufficiency (I5): the artifact must load with the source unmounted.
    CHECK(fs::exists(d.path() / "chat_template.jinja"));
    CHECK(fs::exists(d.path() / "tokenizer.json"));
    CHECK(ReadArtifact(d.path()).info.chat_template == "chat_template.jinja");
}

TEST_CASE("a declared plane missing from disk fails the conversion",
          "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    spec.planes.push_back({"expert.routed", kMxFp8, "experts/layer_0.arrow", 1});
    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) == 0);
    CHECK(err.find("experts/layer_0.arrow") != std::string::npos);
    // Nothing was published: a failed conversion must not leave a loadable tree.
    CHECK_FALSE(fs::exists(d.path() / "artifact.fb"));
}

TEST_CASE("a weight plane with no codec bit fails instead of producing mask 0",
          "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    // A real weight plane that forgot to name its codec would under-report the
    // mask, letting the gate pass an artifact whose codec it never checked.
    spec.planes.push_back({"expert.routed", 0, "pinned/layer_0.arrow", 12});
    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) == 0);
    CHECK(err.find("codec") != std::string::npos);
}

TEST_CASE("a meta plane is allowed to declare no codec bit", "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    WriteFile(d.path() / "pinned" / "idx.arrow", "index-table");
    spec.planes.push_back({"meta", 0, "pinned/idx.arrow", 11});
    std::string err;
    const uint64_t mask = EmitArtifact(spec, d.path(), &err);
    INFO(err);
    REQUIRE(mask == (kBF16 | kF32));   // the meta plane adds nothing
}

TEST_CASE("emitting twice is idempotent in identity fields", "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    const uint64_t first = EmitArtifact(spec, d.path(), nullptr);
    REQUIRE(first != 0);
    const auto a = ReadArtifact(d.path());
    const uint64_t second = EmitArtifact(spec, d.path(), nullptr);
    REQUIRE(second == first);
    const auto b = ReadArtifact(d.path());
    CHECK(a.info.used_codecs == b.info.used_codecs);
    CHECK(a.info.artifact_id == b.info.artifact_id);
    CHECK(a.info.planes[0].sha256 == b.info.planes[0].sha256);
}

TEST_CASE("a plane's digest changes when its bytes change", "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    const auto before = ReadArtifact(d.path()).info.planes[0].sha256;

    WriteFile(d.path() / "pinned" / "layer_0.arrow", "different-weights");
    REQUIRE(EmitArtifact(spec, d.path(), nullptr) != 0);
    const auto after = ReadArtifact(d.path()).info.planes[0].sha256;
    CHECK(before != after);
}

TEST_CASE("a missing chat template source fails rather than shipping a template-less "
          "artifact",
          "[storage_writer]") {
    TempDir d;
    auto spec = MinimalSpec(d.path());
    spec.chat_template_src = (d.path() / "nope.jinja").string();
    std::string err;
    REQUIRE(EmitArtifact(spec, d.path(), &err) == 0);
    CHECK(err.find("chat template") != std::string::npos);
}

TEST_CASE("preflight refuses a conversion that cannot fit", "[storage_writer]") {
    TempDir d;
    std::string err;
    // 1 TiB will not be free anywhere this test can run.
    CHECK_FALSE(PreflightDisk(d.path(), 1ull << 40, &err));
    CHECK(err.find("free") != std::string::npos);
    // And a tiny request against a real directory succeeds.
    CHECK(PreflightDisk(d.path(), 1, nullptr));
}

// The digests are the only corruption detector artifact.fb has, so the hash
// itself is pinned against the FIPS 180-4 vectors — a subtly wrong SHA-256 would
// still produce stable, comparable-looking digests and hide real damage.
TEST_CASE("sha256 matches the standard vectors", "[storage_writer][sha256]") {
    using lsx::storage::Sha256;
    using lsx::storage::Sha256Hex;

    CHECK(Sha256Hex(Sha256::OfBytes("", 0)) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(Sha256Hex(Sha256::OfBytes("abc", 3)) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const char *two_block =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(Sha256Hex(Sha256::OfBytes(two_block, std::strlen(two_block))) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // Streaming must agree with one-shot: the emitter streams files, never
    // whole-file buffers.
    Sha256 h;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) h.Update(chunk.data(), chunk.size());
    CHECK(Sha256Hex(h.Final()) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
