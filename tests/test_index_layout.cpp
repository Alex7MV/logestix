// model.index.json declares its LAYOUT, not a per-model format tag.
//
// The `format` field was a per-model identity string ("logestix-phi",
// "logestix-alice-nvfp4", …): 18 spellings for one schema family. It duplicated
// three things that now have owners — the model identity (artifact.fb
// model_type, and the index's own "model"), the codec (artifact.fb
// used_codecs), and the schema layout (this field).
//
// It also broke multi-profile support outright: six of the seven tags a loader
// GATES on embedded the codec ("…-nvfp4"), so a bf16 artifact of the same model
// would pass the codec gate and then be refused by its own loader.
//
// `layout` is one integer for the whole tree schema generation. Old trees
// (layout 1, the `format`-tag era) are deleted and deliberately NOT loadable, so
// there is no fallback: a missing or different layout is a refusal, not a guess.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/index_layout.h"

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

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<fs::path> Sources(const char *root)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const auto &e : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        const auto ext = e.path().extension().string();
        if (ext == ".cpp" || ext == ".cu" || ext == ".cuh" || ext == ".h")
            out.push_back(e.path());
    }
    return out;
}

} // namespace

TEST_CASE("the layout constant is a positive generation number",
          "[index_layout]") {
    // 1 was the `format`-tag generation, which is no longer readable; 2 is this
    // one. The value is an on-disk contract, so it is pinned.
    CHECK(lsxcommon::kIndexLayout == 2);
}

TEST_CASE("index layout support is exact, never a range", "[index_layout]") {
    using lsxcommon::IndexLayoutSupported;
    CHECK(IndexLayoutSupported(lsxcommon::kIndexLayout));
    CHECK_FALSE(IndexLayoutSupported(0));    // absent
    CHECK_FALSE(IndexLayoutSupported(1));    // the deleted generation
    CHECK_FALSE(IndexLayoutSupported(3));    // a future one this build cannot read
    CHECK_FALSE(IndexLayoutSupported(-1));
}

TEST_CASE("the refusal message names both the expectation and what was found",
          "[index_layout]") {
    const auto msg = lsxcommon::IndexLayoutError(1);
    CHECK(msg.find("2") != std::string::npos);   // what this build reads
    CHECK(msg.find("1") != std::string::npos);   // what the tree declared
    CHECK(msg.find("reconvert") != std::string::npos);

    // An absent field reads as 0 and says so, rather than "layout 0".
    const auto absent = lsxcommon::IndexLayoutError(0);
    CHECK(absent.find("no") != std::string::npos);
    CHECK(absent.find("layout") != std::string::npos);
}

TEST_CASE("no converter writes a per-model format tag into model.index.json",
          "[index_layout]") {
    // The goldens (tests/fixtures/*.json) keep their own `format` field — they
    // are a separate artifact class with their own reader — so only libs/ is
    // scanned. What this forbids is the tag inside a MODEL INDEX.
    std::vector<std::string> violations;
    for (const auto &p : Sources((fs::path(LSX_SOURCE_DIR) / "libs").c_str())) {
        const std::string text = ReadFile(p);
        if (text.find("\\\"format\\\":") != std::string::npos)
            violations.push_back("writes an index format tag: " +
                                 fs::relative(p, LSX_SOURCE_DIR).string());
    }
    for (const auto &v : violations) INFO(v);
    CHECK(violations.empty());
}

TEST_CASE("no index reader still reads a per-model format tag",
          "[index_layout]") {
    // The property that matters: nothing parses `format` out of a MODEL INDEX.
    // (The selfcheck readers parse it out of a GOLDEN, a different artifact
    // class with its own tag, so they are exempt.)
    std::vector<std::string> offenders;
    for (const auto &p : Sources((fs::path(LSX_SOURCE_DIR) / "libs").c_str())) {
        const auto name = p.filename().string();
        // The exemption is the CLASS, not a list of three names: every
        // `<kind>_selfcheck.cpp` reads a GOLDEN, a separate artifact class with
        // its own `format` tag and its own reader. Spelling the three names out
        // meant the fourth gate (mcp) tripped this guard on the day it landed —
        // a guard that has to be edited in the same commit as every new member
        // of a family is a guard that will eventually be edited wrongly.
        static const std::string kGoldenReader = "_selfcheck.cpp";
        if (name.size() > kGoldenReader.size() &&
            name.compare(name.size() - kGoldenReader.size(),
                         kGoldenReader.size(), kGoldenReader) == 0)
            continue;   // golden readers
        if (ReadFile(p).find("[\"format\"]") != std::string::npos)
            offenders.push_back(fs::relative(p, LSX_SOURCE_DIR).string());
    }
    for (const auto &o : offenders) INFO(o);
    CHECK(offenders.empty());
}

TEST_CASE("the layout gate is used by more than one family", "[index_layout]") {
    // A single user would mean the rest quietly lost their schema check.
    int users = 0;
    for (const auto &p : Sources((fs::path(LSX_SOURCE_DIR) / "libs").c_str())) {
        const auto text = ReadFile(p);
        if (text.find("IndexLayoutSupported") != std::string::npos) ++users;
    }
    CHECK(users >= 10);
}


