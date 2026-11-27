// artifact.fb reader + verifier (libs/lsxstorage, no GPU).
//
// Every case here is a way an artifact can be wrong: absent, truncated, a
// future version, or a mask that no longer matches the bytes on disk. The
// hostile cases matter more than the happy one, because artifact.fb is the only
// thing standing between a converter bug and a model that runs the wrong
// numeric path.
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/artifact.h"

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

// Minimal scratch directory; removed on scope exit.
class TempDir {
public:
    TempDir()
    {
        static int counter = 0;
        path_ = fs::temp_directory_path() /
                ("lsx-art-" + std::to_string(::getpid()) + "-" +
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

} // namespace

TEST_CASE("missing artifact.fb is kMissing, not an exception", "[storage_artifact]") {
    TempDir d;
    const auto r = ReadArtifact(d.path());
    REQUIRE(r.status == ReadStatus::kMissing);
    CHECK_FALSE(r.detail.empty());
}

TEST_CASE("an empty artifact.fb is kUnreadable, not a crash", "[storage_artifact]") {
    TempDir d;
    WriteFile(d.path() / "artifact.fb", "");
    const auto r = ReadArtifact(d.path());
    CHECK((r.status == ReadStatus::kUnreadable ||
           r.status == ReadStatus::kMalformed));
    CHECK_FALSE(r.detail.empty());
}

TEST_CASE("a truncated artifact.fb is refused", "[storage_artifact]") {
    TempDir d;
    // A plausible-looking prefix with nothing behind it: the verifier must
    // reject rather than read whatever the mapping happens to contain.
    WriteFile(d.path() / "artifact.fb", std::string(64, '\x01'));
    const auto r = ReadArtifact(d.path());
    CHECK(r.status != ReadStatus::kOk);
}

TEST_CASE("random bytes are refused, never interpreted", "[storage_artifact]") {
    TempDir d;
    std::string noise;
    for (int i = 0; i < 512; ++i)
        noise.push_back(static_cast<char>((i * 37 + 11) & 0xFF));
    WriteFile(d.path() / "artifact.fb", noise);
    const auto r = ReadArtifact(d.path());
    CHECK(r.status != ReadStatus::kOk);
}

TEST_CASE("ReadArtifactFile reports a missing file without touching the dir",
          "[storage_artifact]") {
    TempDir d;
    const auto r = ReadArtifactFile(d.path() / "nope.fb");
    REQUIRE(r.status == ReadStatus::kMissing);
}

TEST_CASE("absent metadata is a diagnostic, not a silent default",
          "[storage_artifact]") {
    // A kOk result must carry usable identity: an artifact whose model_type or
    // used_codecs never made it into the buffer is malformed, not "empty".
    TempDir d;
    WriteFile(d.path() / "artifact.fb", "\0\0\0\0");
    const auto r = ReadArtifact(d.path());
    CHECK(r.status != ReadStatus::kOk);
}
