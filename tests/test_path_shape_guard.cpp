// Path-shape guard: no source file may spell a `logestix` path segment.
//
// Every converted tree used to live at `<base>/<Model>/logestix/` inside the
// read-only source vault, so 540 sites wrote `fs::path(x) / "logestix"`. The
// artifact store makes that segment redundant — the profile directory IS the
// artifact — and it was actively harmful, because three detection sites used the
// folder NAME as the "is this a converted tree?" marker instead of artifact.fb.
//
// Shape of the guard mirrors test_logging_guard: scan the sources, allowlist the
// one legitimate use (the per-user cache root), and fail listing every violation.
// Without this, a single missed site sends a loader to a directory that no longer
// exists, and the failure reads as "missing weights" rather than "wrong path".
#include <catch2/catch_test_macros.hpp>

#include "lsxstorage/artifact.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

// The only file allowed to spell `logestix` as a path segment: it builds the
// per-user host cache (~/.cache/logestix), which is host state, not a model path.
const std::vector<std::string> &Allowlist()
{
    static const std::vector<std::string> v = {"path_registry.cpp"};
    return v;
}

bool IsSourceFile(const fs::path &p)
{
    const auto ext = p.extension().string();
    return ext == ".cpp" || ext == ".cu" || ext == ".cuh" || ext == ".h";
}

std::vector<fs::path> SourceTree(const char *root)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto &entry : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        if (!IsSourceFile(entry.path())) continue;
        out.push_back(entry.path());
    }
    return out;
}

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Remove // and /* */ comments so a documented mention does not count as a use.
std::string StripComments(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    bool line = false, block = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        const char n = i + 1 < s.size() ? s[i + 1] : '\0';
        if (line) {
            if (c == '\n') { line = false; out.push_back(c); }
        } else if (block) {
            if (c == '*' && n == '/') { block = false; ++i; }
        } else if (c == '/' && n == '/') {
            line = true; ++i;
        } else if (c == '/' && n == '*') {
            block = true; ++i;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

} // namespace

TEST_CASE("no source file spells a logestix path segment", "[path_shape]")
{
    std::vector<std::string> violations;
    for (const char *dir : {"libs", "apps", "tests"}) {
        for (const auto &path : SourceTree((fs::path(LSX_SOURCE_DIR) / dir).c_str())) {
            if (std::find(Allowlist().begin(), Allowlist().end(),
                          path.filename().string()) != Allowlist().end())
                continue;
            const std::string text = StripComments(ReadFile(path));
            // Both spellings a join can take: `x / "logestix"` and the rarer
            // leading form `"logestix" / y`.
            if (text.find("/ \"logestix\"") != std::string::npos ||
                text.find("\"logestix\" /") != std::string::npos) {
                violations.push_back(fs::relative(path, LSX_SOURCE_DIR).string());
            }
        }
    }
    for (const auto &v : violations) INFO(v);
    CHECK(violations.empty());
}

TEST_CASE("an artifact is detected by artifact.fb, not by a directory name",
          "[path_shape]")
{
    // The three detection sites this replaced tested for a folder called
    // `logestix`. IsArtifactDir tests for the contract instead.
    // Assembled at runtime so this guard's own source does not spell the
    // literal it forbids (the scanner cannot tell an intentional probe from a
    // real join).
    const std::string kOldMarker = std::string("loge") + "stix";
    const auto dir = fs::temp_directory_path() / "lsx-pathshape-probe";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "pinned", ec);
    fs::create_directories(dir / kOldMarker, ec);   // the OLD marker
    CHECK_FALSE(lsx::storage::IsArtifactDir(dir));
    {
        std::ofstream(dir / "artifact.fb") << "x";
    }
    CHECK(lsx::storage::IsArtifactDir(dir));
    fs::remove_all(dir, ec);
}
