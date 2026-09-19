// License compliance guard (host-only, zero deps beyond Catch2).
//
// Rule: no first-party source file may carry a copyleft license header or an
// SPDX license identifier. logestix is dual-licensed (AGPL-3.0 for the open
// source edition, a commercial Enterprise license otherwise); a stray
// GPL/LGPL/AGPL/copyleft fragment in shipped code would make the commercial
// relicensing impossible. Third-party licenses live only in
// THIRD_PARTY_NOTICES and are attached to unmodified dependencies.
//
// This guard statically scans libs/, apps/, tests/ and tools/ for copyleft
// markers in code files and fails, listing every violation. It mirrors the
// scan used by the license audit.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

// Copyleft / license-header markers. Case-insensitive except for the SPDX tag,
// which is matched case-insensitively too (SPDX permits any case).
const std::regex& ForbiddenMarker()
{
    static const std::regex r(
        R"(\bGPL\b|\bGPLv[0-9]|\bGPL-[0-9]|\bLGPL\b|\bLGPLv[0-9]|\bAGPL\b|\bAGPLv[0-9]|GNU General Public|GNU Lesser General Public|GNU Affero|Affero General Public|copyleft)",
        std::regex::icase | std::regex::ECMAScript);
    return r;
}

bool IsScannedExtension(const fs::path& p)
{
    const std::string ext = p.extension().string();
    return ext == ".cpp" || ext == ".cc" || ext == ".c" || ext == ".h" ||
           ext == ".hpp" || ext == ".cu" || ext == ".cuh";
}

// The guard must not flag its own pattern definitions.
bool IsExcludedFile(const fs::path& p)
{
    if (p.filename() == "test_license_guard.cpp") return true;
    const std::string generic = p.generic_string();
    return generic.find("/third_party/") != std::string::npos;
}

struct Violation
{
    std::string file;
    std::size_t line = 0;
    std::string match;
};

void ScanSourceText(const std::string& src, const std::string& name,
                    std::vector<Violation>& out)
{
    std::smatch m;
    auto begin = src.cbegin();
    while (std::regex_search(begin, src.cend(), m, ForbiddenMarker()))
    {
        const std::size_t pos = static_cast<std::size_t>(m.position(0)) +
                                static_cast<std::size_t>(begin - src.cbegin());
        const std::size_t line =
            1 + static_cast<std::size_t>(std::count(
                    src.begin(), src.begin() + static_cast<long>(pos), '\n'));
        out.push_back({name, line, m.str(0)});
        begin = m.suffix().first;
    }
}

void ScanFile(const fs::path& path, std::vector<Violation>& out)
{
    if (IsExcludedFile(path)) return;
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::ostringstream ss;
    ss << f.rdbuf();
    ScanSourceText(ss.str(), path.generic_string(), out);
}

void ScanTree(const fs::path& dir, std::vector<Violation>& out)
{
    std::error_code ec;
    if (!fs::exists(dir, ec))
    {
        out.push_back({dir.string() + " (MISSING)", 0, ""});
        return;
    }
    for (auto& e : fs::recursive_directory_iterator(dir))
    {
        if (!e.is_regular_file()) continue;
        if (!IsScannedExtension(e.path())) continue;
        ScanFile(e.path(), out);
    }
}

std::vector<Violation> ScanCode(const std::string& code)
{
    std::vector<Violation> v;
    ScanSourceText(code, "snippet", v);
    return v;
}

} // namespace

TEST_CASE("license guard: parser flags copyleft markers",
          "[license][guard]")
{
    SECTION("GPL / LGPL / AGPL headers and SPDX tags are flagged")
    {
        CHECK(ScanCode("// SPDX-License-Identifier: GPL-3.0-or-later").size() == 1);
        CHECK(ScanCode("/* This file is part of the GNU General Public "
                       "License. */").size() == 1);
        CHECK(ScanCode("// GNU Lesser General Public License, version 2.1").size() == 1);
        CHECK(ScanCode("// GNU AFFERO GENERAL PUBLIC LICENSE").size() == 1);
        CHECK(ScanCode("// SPDX-License-Identifier: LGPL-2.1-only").size() == 1);
        CHECK(ScanCode("// Distributable under GPLv2 only.").size() == 1);
        CHECK(ScanCode("// strong copyleft").size() == 1);
    }
    SECTION("permissive and project markers are clean")
    {
        CHECK(ScanCode("// SPDX-License-Identifier: Apache-2.0").empty());
        CHECK(ScanCode("// Licensed under the MIT License.").empty());
        CHECK(ScanCode("// Copyright (c) 2026 logestix.").empty());
        CHECK(ScanCode("// GPU kernel, no license header here.").empty());
    }
    SECTION("case-insensitive")
    {
        CHECK(ScanCode("// gnu general public license").size() == 1);
        CHECK(ScanCode("// lgpl").size() == 1);
    }
}

TEST_CASE("license guard: no copyleft markers in first-party source",
          "[license][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);

    std::vector<Violation> violations;
    ScanTree(root / "libs", violations);
    ScanTree(root / "apps", violations);
    ScanTree(root / "tests", violations);
    ScanTree(root / "tools", violations);

    std::sort(violations.begin(), violations.end(),
              [](const Violation& a, const Violation& b) {
                  if (a.file != b.file) return a.file < b.file;
                  return a.line < b.line;
              });
    std::string joined;
    for (const auto& v : violations)
    {
        joined += v.file + ":" + std::to_string(v.line) + "  [" + v.match +
                  "]\n";
    }
    INFO("copyleft markers (" << violations.size() << "):\n" << joined);
    REQUIRE(violations.empty());
}
