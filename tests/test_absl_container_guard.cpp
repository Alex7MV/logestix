// absl container / device-TU guard (host-only static scan; zero deps beyond
// Catch2 + absl containers).
//
// Bug class: abseil containers (flat_*, btree_*, everything under
// absl/container/internal/*) do not compile under NVCC — raw_hash_map breaks
// inside a .cu with MSVC-in-NVCC C2794/C2938. So no .cu/.cuh translation unit
// may transitively include <absl/container/...>.
//
// This machine-checks the transitive include closure over libs/, apps/ and
// tests/ (includes resolved by basename, so it over-approximates; 0 hits is
// sound). Host-only code (.cpp/.h) may use absl containers freely.
#include <catch2/catch_test_macros.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <algorithm>
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

bool IsScanned(const fs::path &p)
{
    const std::string ext = p.extension().string();
    return ext == ".h" || ext == ".hpp" || ext == ".cuh" || ext == ".cu" ||
           ext == ".cpp";
}

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> Includes(const std::string &text)
{
    static const std::regex re(R"(^\s*#\s*include\s*[<"]([^">]+)[">])",
                               std::regex_constants::multiline);
    std::vector<std::string> out;
    for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end;
         ++it)
        out.push_back((*it)[1].str());
    return out;
}

} // namespace

TEST_CASE("absl container guard: no .cu/.cuh reaches an absl container header",
          "[absl][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<fs::path> files;
    for (const char *dir : {"libs", "apps", "tests"}) {
        std::error_code ec;
        const fs::path base = root / dir;
        if (!fs::exists(base, ec)) continue;
        for (auto &e : fs::recursive_directory_iterator(base, ec))
            if (e.is_regular_file() && IsScanned(e.path()))
                files.push_back(e.path());
    }

    absl::flat_hash_map<std::string, std::vector<fs::path>> by_name;
    for (const auto &p : files) by_name[p.filename().string()].push_back(p);

    absl::flat_hash_set<std::string> absloop;
    absl::flat_hash_map<std::string, std::vector<std::string>> include_names;
    for (const auto &p : files) {
        const std::string text = ReadFile(p);
        if (text.find("#include <absl/container") != std::string::npos ||
            text.find("#include \"absl/container") != std::string::npos)
            absloop.insert(p.string());
        auto &names = include_names[p.string()];
        for (const auto &inc : Includes(text))
            names.push_back(fs::path(inc).filename().string());
    }

    // Sanity: the scan must find the known absl-container users, otherwise it
    // is silently broken and would pass vacuously.
    REQUIRE_FALSE(absloop.empty());

    std::vector<std::string> violations;
    for (const auto &p : files) {
        const std::string ext = p.extension().string();
        if (ext != ".cu" && ext != ".cuh") continue;
        absl::flat_hash_set<std::string> seen;
        std::vector<fs::path> stack{p};
        while (!stack.empty()) {
            const fs::path f = stack.back();
            stack.pop_back();
            if (!seen.insert(f.string()).second) continue;
            if (absloop.count(f.string())) {
                violations.push_back(p.string());
                break;
            }
            for (const auto &nm : include_names[f.string()]) {
                auto it = by_name.find(nm);
                if (it != by_name.end())
                    for (const auto &d : it->second) stack.push_back(d);
            }
        }
    }

    std::sort(violations.begin(), violations.end());
    std::string joined;
    for (const auto &v : violations) joined += v + "\n";
    INFO("device TUs reaching an absl container header:\n" << joined);
    REQUIRE(violations.empty());
}
