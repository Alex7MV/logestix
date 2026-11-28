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
#include <cctype>
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

// Exempt from the pre-split MODEL segment: where the name is not a model path
// (the per-user cache root, ~/.cache/logestix) and where recognising an old tree
// is the function's whole job (`tools/lsx_paths.py`, so a reference run captured
// against a pre-split tree can still be reproduced).
const std::vector<std::string> &LegacyAllowlist()
{
    static const std::vector<std::string> v = {"path_registry.cpp",
                                               "lsx_paths.py",
                                               // asserts the segment is ABSENT
                                               "test_lsx_paths.py"};
    return v;
}

bool IsSourceFile(const fs::path &p)
{
    const auto ext = p.extension().string();
    return ext == ".cpp" || ext == ".cu" || ext == ".cuh" || ext == ".h";
}

bool IsPythonFile(const fs::path &p) { return p.extension() == ".py"; }

std::vector<fs::path> SourceTree(const char *root,
                                 bool (*accept)(const fs::path &) = IsSourceFile)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto &entry : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        if (!accept(entry.path())) continue;
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

// Python's comments start with '#'; its strings stay intact, which is what makes
// the quoted needle below meaningful.
std::string StripHashComments(const std::string &s)
{
    std::string out;
    bool line = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (line) {
            if (c == '\n') { line = false; out.push_back(c); }
            continue;
        }
        if (c == '#') { line = true; continue; }
        out.push_back(c);
    }
    return out;
}

// ── Rule: nothing builds a pre-split `logestix/<subdir>` path ──────────────
//
// C++ spelled it `x / "logestix"`; Python spells the same thing as
// `os.path.join(x, "logestix", ...)` or `Path(x) / "logestix"`. Both are the
// same defect — a directory that no conversion writes any more, because the
// profile directory IS the artifact — and five operator scripts
// (hy4_flash_reference, kimi_layer_oracle, three inkling readers) still built
// it, so every reference run answered "no such directory" against a tree that
// was there.
//
// The needle is the QUOTED segment, so a prose mention ("no `logestix/` segment
// any more") does not count. Two files are exempt by name: the cache root, where
// `logestix` is a per-user directory name and not a model path, and
// `tools/lsx_paths.py`, the one function allowed to recognise an OLD tree so a
// reference run captured against one can still be reproduced.
bool BuildsLegacySegment(const std::string &text, bool python)
{
    if (!python)
        return text.find("/ \"logestix\"") != std::string::npos ||
               text.find("\"logestix\" /") != std::string::npos;
    return text.find("\"logestix\"") != std::string::npos ||
           text.find("'logestix'") != std::string::npos;
}

std::vector<std::string> LegacySegmentViolations(const fs::path &root,
                                                bool python)
{
    std::vector<std::string> out;
    for (const auto &path :
         SourceTree(root.string().c_str(), python ? IsPythonFile : IsSourceFile)) {
        if (std::find(LegacyAllowlist().begin(), LegacyAllowlist().end(),
                      path.filename().string()) != LegacyAllowlist().end())
            continue;
        const std::string text =
            python ? StripHashComments(ReadFile(path)) : StripComments(ReadFile(path));
        if (BuildsLegacySegment(text, python))
            out.push_back(fs::relative(path, LSX_SOURCE_DIR).string());
    }
    return out;
}

} // namespace

// ── Rule 4: a Python tool never learns the store root itself ───────────────
//
// The tools are operator code, and they carried the same defect in a second
// spelling: `default="/mnt/fast_storage/lsx_model/<org>/<model>"`. The org was
// right there, so nothing looked broken — but the ROOT was frozen in the
// source, so `LSX_SOURCE_ROOT` (a store on another disk, a copy kept for a
// comparison run) was ignored and the tool read a directory that had moved.
// One resolver, `tools/lsx_paths.py`, is the only file allowed to name a root;
// everything else asks it where a model is.
TEST_CASE("no Python tool hardcodes a store root", "[path_shape]")
{
    // The needles are assembled, so this guard's own source does not spell the
    // roots it forbids — the suite guard scans the same file (a scanner that
    // reads as a violation is a broken scanner, not a strict one).
    const std::string source_root = std::string("/mnt/fast_storage/lsx_") + "model";
    const std::string artifact_root = std::string("/mnt/fast_storage/lsx_") + "artifact";
    std::vector<std::string> violations;
    for (const auto &path : SourceTree((fs::path(LSX_SOURCE_DIR) / "tools").c_str(),
                                       IsPythonFile)) {
        const std::string name = path.filename().string();
        // The resolver itself, its self-test, and the generator — whose
        // docstring is where a reader should meet the two roots.
        if (name == "lsx_paths.py" || name == "test_lsx_paths.py" ||
            name == "gen_model_manifest.py")
            continue;
        const std::string text = StripHashComments(ReadFile(path));
        if (text.find(source_root) != std::string::npos ||
            text.find(artifact_root) != std::string::npos)
            violations.push_back(fs::relative(path, LSX_SOURCE_DIR).string());
    }
    std::string report;
    for (const auto &v : violations) report += "\n    " + v;
    INFO(report);
    CHECK(violations.empty());
}

TEST_CASE("no source file spells a logestix path segment", "[path_shape]")
{
    std::vector<std::string> violations;
    for (const char *dir : {"libs", "apps", "tests"})
        for (auto &v : LegacySegmentViolations((fs::path(LSX_SOURCE_DIR) / dir).c_str(), false))
            violations.push_back(v);
    // The operator tools are code too: a reference oracle that reads a directory
    // nothing writes is as broken as a test that does.
    for (auto &v : LegacySegmentViolations((fs::path(LSX_SOURCE_DIR) / "tools").c_str(), true))
        violations.push_back(v);
    std::string report;
    for (const auto &v : violations) report += "\n    " + v;
    INFO(report);
    CHECK(violations.empty());
}

TEST_CASE("the legacy-segment scanner reads both languages and skips prose",
          "[path_shape]")
{
    // Non-vacuity, per language: a code join is a violation, a comment is not,
    // and the two exempt files are exempt by name.
    const auto dir = fs::temp_directory_path() / "lsx-pathshape-legacy";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    {
        std::ofstream(dir / "bad.py")
            << "p = os.path.join(root, \"logestix\", \"pinned\")\n";
        std::ofstream(dir / "commented.py")
            << "# this used to join \"logestix\" here\n"
            << "q = lsx_paths.artifact_dir(root)\n";
        std::ofstream(dir / "lsx_paths.py")
            << "def artifact_dir(x):\n    return x / \"logestix\"\n";
    }
    const auto py = LegacySegmentViolations(dir, true);
    REQUIRE(py.size() == 1);
    CHECK(fs::path(py[0]).filename() == "bad.py");

    {
        std::ofstream(dir / "bad.cpp") << "auto p = root / \"logestix\";\n";
        std::ofstream(dir / "commented.cpp")
            << "// the old shape was root / \"logestix\"\n";
    }
    const auto cxx = LegacySegmentViolations(dir, false);
    REQUIRE(cxx.size() == 1);
    CHECK(fs::path(cxx[0]).filename() == "bad.cpp");
    fs::remove_all(dir, ec);
}

// ── Rule 3: one spelling of the source root, and it is the resolver's ──────
//
// The store is ORG-NESTED (`<root>/<org>/<model>`), and the org lives only in
// the registry — so a caller that builds `<root>/<Model>` itself drops the org
// and points at a directory that never exists.
//
// Thirty-eight sites did exactly that (`fs::path(lsx_model_models_dir()) /
// "Kimi-K2.7-Code"`). Every one of them read as "the weights are not on disk":
// the SKIP message printed the flat path, and twenty-two of them were pointing
// at a model whose weights were right there under its org. The literal-root
// scanner in test_model_suite_guard could not see them, because the root
// arrived through the API rather than as a string.
//
// The rule that closes it: the source-root accessor is called in exactly one
// place — the resolver (`manifest_query::SourceDirFor` / `ConvertedDirFor` /
// `SourceDir(row)`) — plus the files whose SUBJECT is the store layer. A test
// that wants a model path asks for it by model_type; an app asks
// `lsx::storage::Resolve` (which searches artifacts by name) or the registry.
//
// The accessor is `lsx::storage::SourceRoot()`, which returns std::optional and
// has NO default: since the compiled-in root was removed a caller cannot even get
// a path without first checking that a store is configured, so this rule guards
// the join rather than the missing check.

namespace {

// Comments become spaces, newlines stay, so a reported line is the real one.
std::string MaskComments(const std::string &s)
{
    std::string out = s;
    bool line = false, block = false;
    for (size_t i = 0; i < out.size(); ++i) {
        const char c = out[i];
        const char n = i + 1 < out.size() ? out[i + 1] : '\0';
        if (line) {
            if (c == '\n') line = false; else out[i] = ' ';
        } else if (block) {
            if (c == '*' && n == '/') { block = false; out[i] = out[i + 1] = ' '; ++i; }
            else if (c != '\n') out[i] = ' ';
        } else if (c == '/' && n == '/') {
            line = true; out[i] = ' ';
        } else if (c == '/' && n == '*') {
            block = true; out[i] = out[i + 1] = ' '; ++i;
        }
    }
    return out;
}

const std::vector<std::string> &SourceRootAllowlist()
{
    static const std::vector<std::string> v = {
        "roots.h",                    // declares it
        "roots.cpp",                  // implements it and the search over it
        "resolve.cpp",                // joins a qualified name to it
        "test_manifest_query_root.h", // the resolver every test asks
        "test_manifest_sync.cpp",     // subject IS the store layer
        "test_storage_roots.cpp",
        "test_path_registry.cpp",
        // Asserts the per-user cache is NOT under the source root: the root's
        // own string is the subject of the assertion, not a model path.
        "test_gpu_perf.cpp",
    };
    return v;
}

// `file:line` for every call to the source-root accessor in @p dir.
std::vector<std::string> SourceRootCalls(const fs::path &dir)
{
    std::vector<std::string> out;
    for (const auto &path : SourceTree(dir.string().c_str())) {
        if (std::find(SourceRootAllowlist().begin(), SourceRootAllowlist().end(),
                      path.filename().string()) != SourceRootAllowlist().end())
            continue;
        const std::string text = MaskComments(ReadFile(path));
        // Assembled: this scanner's own needle must not read as a call site.
        // A substring match covers both spellings — `lsx::storage::SourceRoot()`
        // outside the library and the unqualified call inside it — but it must
        // NOT match a longer identifier that merely ends in the same letters
        // (`manifest_query::RequireSourceRoot()` skips when no store is
        // configured; it is not a caller of the accessor).
        const std::string needle = std::string("SourceRoo") + "t()";
        size_t at = text.find(needle);
        while (at != std::string::npos) {
            const bool standalone =
                at == 0 ||
                !(std::isalnum(static_cast<unsigned char>(text[at - 1])) ||
                  text[at - 1] == '_');
            if (standalone) {
                const size_t line =
                    1 + std::count(text.begin(), text.begin() + at, '\n');
                out.push_back(fs::relative(path, LSX_SOURCE_DIR).string() + ":" +
                              std::to_string(line));
            }
            at = text.find(needle, at + 1);
        }
    }
    return out;
}

} // namespace

TEST_CASE("the source root is spelled in one place, the resolver", "[path_shape]")
{
    std::vector<std::string> violations;
    for (const char *dir : {"libs", "apps", "tests"})
        for (const auto &v : SourceRootCalls((fs::path(LSX_SOURCE_DIR) / dir)))
            violations.push_back(v);
    // One message, not one INFO per site: Catch2 prints a lone INFO, and the
    // whole list is what tells an operator how far the migration got.
    std::string report;
    for (const auto &v : violations) report += "\n    " + v;
    INFO(report);
    CHECK(violations.empty());
}

TEST_CASE("the source-root scanner is not vacuous", "[path_shape]")
{
    // A guard that reports nothing because it looks at nothing is worse than no
    // guard: it reads as compliance. Pin the three behaviours it must have.
    const auto dir = fs::temp_directory_path() / "lsx-pathshape-roots";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const std::string call = std::string("SourceRoo") + "t()";

    {
        // Two calls on lines 1 and 3: the scanner must report both, with the
        // line a reader can open.
        std::ofstream(dir / "code.cpp")
            << "auto p = fs::path(" << call << ") / \"M\";\n"
            << "// a comment between them\n"
            << "auto q = " << call << ";\n";
        // A documented mention is not a use.
        std::ofstream(dir / "commented.cpp")
            << "// this file used to call " << call << " here\n"
            << "/* and here: " << call << " */\n";
        // The resolver and the store-layer tests are allowed to call it.
        std::ofstream(dir / "test_storage_roots.cpp") << "auto p = " << call << ";\n";
    }

    auto hits = SourceRootCalls(dir);
    std::sort(hits.begin(), hits.end());
    REQUIRE(hits.size() == 2);
    const auto ends = [](const std::string &s, const std::string &suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    CHECK(ends(hits[0], "code.cpp:1"));
    CHECK(ends(hits[1], "code.cpp:3"));
    fs::remove_all(dir, ec);
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
