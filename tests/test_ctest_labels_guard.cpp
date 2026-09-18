// CTest category-label discipline guard (host-only, zero deps beyond Catch2).
// Categories: fast | inference | perf. Spec:
// docs/superpowers/specs/2026-09-10-test-categories-design.md
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

const std::set<std::string>& Categories()
{
    static const std::set<std::string> v = {"fast", "convert", "inference",
                                            "perf"};
    return v;
}

std::string ReadFileOrThrow(const fs::path& p)
{
    std::ifstream in(p);
    if (!in.is_open())
        throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Remove // and /* */ comments so source scans do not match documented
// mentions of a macro (e.g. this guard's own "SKIP()" prose).
std::string StripComments(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    bool line = false, block = false;
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        if (line)
        {
            if (s[i] == '\n')
            {
                line = false;
                out += '\n';
            }
            continue;
        }
        if (block)
        {
            if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '/')
            {
                block = false;
                ++i;
            }
            continue;
        }
        if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/')
        {
            line = true;
            ++i;
            continue;
        }
        if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '*')
        {
            block = true;
            ++i;
            continue;
        }
        out += s[i];
    }
    return out;
}

std::vector<std::string> RegisteredTestNames(const std::string& text)
{
    std::vector<std::string> names;
    static const std::regex add_re(
        R"(add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
    // foreach(<var> <items...>): every item is a registered test name.
    // Skip foreach(... IN LISTS <var>) / RANGE forms — variable-driven,
    // not statically resolvable.
    static const std::regex fe_re(
        R"(foreach\s*\(\s*\w+((?:\s+(?!IN\b|LISTS\b|RANGE\b)[A-Za-z0-9_]+)+)\s*\))");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, add_re))
    {
        names.push_back(m[1]);
        begin = m.suffix().first;
    }
    begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, fe_re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok)
            names.push_back(tok);
        begin = m.suffix().first;
    }
    return names;
}

struct LabelCall
{
    std::vector<std::string> names;
    std::string label;
};

// Recognizes set_tests_properties(<names...> PROPERTIES LABELS <cat>) calls,
// single- or multi-line (parens body scan up to the first ')').
std::vector<LabelCall> ParseLabelCalls(const std::string& text)
{
    std::vector<LabelCall> calls;
    static const std::regex open_re(R"(set_tests_properties\s*\()");
    static const std::regex label_re(R"(PROPERTIES\s+LABELS\s+(\w+)\s*$)");
    static const std::regex props_re(R"(PROPERTIES\s+LABELS)");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, open_re))
    {
        const auto body_begin = m.suffix().first;
        const auto body_end = std::find(body_begin, text.cend(), ')');
        const std::string body(body_begin, body_end);
        std::smatch lm;
        if (std::regex_search(body, lm, label_re))
        {
            LabelCall c;
            c.label = lm[1];
            std::smatch pm;
            std::regex_search(body, pm, props_re);
            std::istringstream is(body.substr(0, pm.position()));
            std::string tok;
            while (is >> tok)
                c.names.push_back(tok);
            calls.push_back(std::move(c));
        }
        begin = (body_end == text.cend()) ? body_end : body_end + 1;
    }
    return calls;
}

std::vector<std::string> CheckLabelCoverage(
    const std::vector<std::string>& tests, const std::vector<LabelCall>& calls)
{
    std::vector<std::string> v;
    std::map<std::string, std::string> label_of;
    for (const auto& c : calls)
    {
        if (!Categories().count(c.label))
            v.push_back("unknown label: " + c.label);
        for (const auto& n : c.names)
        {
            const auto it = label_of.find(n);
            if (it != label_of.end())
                v.push_back("duplicate label for " + n + " (" + it->second +
                            " and " + c.label + ")");
            else
                label_of.emplace(n, c.label);
        }
    }
    for (const auto& t : tests)
        if (!label_of.count(t))
            v.push_back("unlabeled test: " + t);
    for (const auto& [n, l] : label_of)
        if (std::find(tests.begin(), tests.end(), n) == tests.end())
            v.push_back("label for unknown test: " + n);
    return v;
}

std::vector<std::string> CheckPerfSkip(const std::vector<LabelCall>& calls,
                                       const std::string& text)
{
    std::vector<std::string> v;
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    for (const auto& c : calls)
    {
        if (c.label != "perf")
            continue;
        for (const auto& n : c.names)
        {
            bool found = false;
            std::smatch m;
            auto begin = text.cbegin();
            while (std::regex_search(begin, text.cend(), m, re))
            {
                std::istringstream is(m[1].str());
                std::string tok;
                while (is >> tok)
                    if (tok == n)
                    {
                        found = true;
                        break;
                    }
                if (found)
                    break;
                begin = m.suffix().first;
            }
            if (!found)
                v.push_back("perf test without SKIP_RETURN_CODE 4: " + n);
        }
    }
    return v;
}

std::vector<std::string> CheckPresets(const std::string& json_text)
{
    static const std::vector<std::string> kBase = {
        "windows-debug", "windows-release", "linux-debug",
        "linux-release", "linux-release-znver4"};
    static const std::vector<std::string> kSuffix = {
        "fast", "convert", "inference", "perf"};
    std::vector<std::string> v;
    static const std::regex label_re(R"rx("label"\s*:\s*"([^"]+)")rx");
    std::smatch m;
    auto begin = json_text.cbegin();
    while (std::regex_search(begin, json_text.cend(), m, label_re))
    {
        std::string inner = m[1].str();
        if (!inner.empty() && inner.front() == '^')
            inner.erase(0, 1);
        if (!inner.empty() && inner.back() == '$')
            inner.pop_back();
        if (!Categories().count(inner))
            v.push_back("CMakePresets.json: invalid label filter: " +
                        m[1].str());
        begin = m.suffix().first;
    }
    for (const auto& b : kBase)
        for (const auto& s : kSuffix)
            if (json_text.find("\"" + b + "-" + s + "\"") ==
                std::string::npos)
                v.push_back("CMakePresets.json: missing testPreset " + b +
                            "-" + s);
    return v;
}

// tools/test_router.cmake declares: lsx_router_rule(<id> "<labels>" "<regex>")
// — one rule per line, labels are a quoted space-separated list of
// fast|convert|inference (perf is never auto-run).
std::vector<std::string> CheckRouter(const std::string& router_text)
{
    std::vector<std::string> v;
    static const std::regex rule_re(
        R"rx(lsx_router_rule\s*\(\s*(\w+)\s+"([^"]*)"\s+")rx");
    bool any = false;
    std::istringstream in(router_text);
    std::string line;
    while (std::getline(in, line))
    {
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
            continue;
        if (line.find("lsx_router_rule") == std::string::npos)
            continue;
        if (line.rfind("macro(", 0) == 0)
            continue; // the macro definition itself
        std::smatch m;
        if (!std::regex_search(line, m, rule_re))
        {
            v.push_back("tools/test_router.cmake: malformed rule: " + line);
            continue;
        }
        any = true;
        const std::string id = m[1];
        std::istringstream labels(m[2].str());
        std::string lb;
        while (labels >> lb)
            if (!Categories().count(lb) || lb == "perf")
                v.push_back("tools/test_router.cmake: rule '" + id +
                            "' invalid label: " + lb);
    }
    if (!any)
        v.push_back("tools/test_router.cmake: no lsx_router_rule entries");
    return v;
}

// Every file in tests/ (top level) follows the test_* prefix: test_*.cpp,
// test_*.cu, test_*_fixture.h.
std::vector<std::string> CheckTestsNaming(const fs::path& tests_dir)
{
    std::vector<std::string> v;
    static const std::regex ok(R"(test_.+\.(cpp|cu|h)$)");
    std::error_code ec;
    for (auto& e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file())
            continue;
        const std::string name = e.path().filename().string();
        if (name == "CMakeLists.txt")
            continue;
        if (!std::regex_match(name, ok))
            v.push_back("tests/ naming violation (expected test_*): " + name);
    }
    return v;
}

// Any test target whose C++/CUDA source calls Catch2's SKIP() must declare
// SKIP_RETURN_CODE 4; otherwise CTest reports the expected skip (missing real
// weights / no GPU support) as a failure. Source is located by the convention
// test_<name>.cpp|test_<name>.cu in tests/.
std::vector<std::string> CheckSkipCapableHaveCode(
    const std::vector<std::string>& tests, const std::string& cmake_text,
    const fs::path& tests_dir)
{
    std::vector<std::string> v;
    std::set<std::string> has_code;
    // foreach(<var> <items...>): a later set_tests_properties(${var} ...) with
    // SKIP_RETURN_CODE 4 applies to every item.
    std::map<std::string, std::vector<std::string>> loop_items;
    {
        static const std::regex fe_re(
            R"(foreach\s*\(\s*(\w+)((?:\s+(?!IN\b|LISTS\b|RANGE\b)[A-Za-z0-9_]+)+)\s*\))");
        std::smatch fm;
        auto b = cmake_text.cbegin();
        while (std::regex_search(b, cmake_text.cend(), fm, fe_re))
        {
            std::istringstream is(fm[2].str());
            std::string tok;
            auto &items = loop_items[fm[1].str()];
            while (is >> tok)
                items.push_back(tok);
            b = fm.suffix().first;
        }
    }
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    std::smatch m;
    auto begin = cmake_text.cbegin();
    while (std::regex_search(begin, cmake_text.cend(), m, re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok)
        {
            if (tok.size() > 3 && tok.front() == '$' && tok[1] == '{' &&
                tok.back() == '}')
            {
                const auto it = loop_items.find(tok.substr(2, tok.size() - 3));
                if (it != loop_items.end())
                    has_code.insert(it->second.begin(), it->second.end());
            }
            else
            {
                has_code.insert(tok);
            }
        }
        begin = m.suffix().first;
    }
    static const std::regex skip_call(R"(\bSKIP\s*\()");
    for (const auto& t : tests)
    {
        if (has_code.count(t))
            continue;
        for (const char* ext : {".cpp", ".cu"})
        {
            const fs::path p = tests_dir / (t + ext);
            std::error_code ec;
            if (!fs::exists(p, ec))
                continue;
            if (std::regex_search(StripComments(ReadFileOrThrow(p)), skip_call))
                v.push_back("skip-capable test without SKIP_RETURN_CODE 4: " +
                            t);
            break;
        }
    }
    return v;
}

// test_perf_baseline measures a Release build against the committed reference;
// a Debug run silently measures an unoptimized binary and reports a bogus
// DEGRADED verdict. The harness must SKIP under a non-NDEBUG build instead.
std::vector<std::string> CheckPerfBaselineReleaseGuard(const fs::path& tests_dir)
{
    std::vector<std::string> v;
    const std::string src =
        ReadFileOrThrow(tests_dir / "test_perf_baseline.cpp");
    if (src.find("NDEBUG") == std::string::npos)
    {
        v.push_back("test_perf_baseline.cpp: no NDEBUG Release guard "
                    "(a Debug run reports a bogus DEGRADED verdict)");
        return v;
    }
    static const std::regex skip_call(R"(\bSKIP\s*\()");
    if (!std::regex_search(src, skip_call))
        v.push_back("test_perf_baseline.cpp: NDEBUG guard never calls SKIP");
    return v;
}

// Converter tests are unified on synthetic fixtures and must stay fast:
// every convert-labeled test declares a bounded ctest TIMEOUT (seconds).
// A converter test without a TIMEOUT is exactly how a full-model
// conversion sneaks back into the suite unnoticed.
std::vector<std::string> CheckConvertTimeout(
    const std::vector<LabelCall>& calls, const std::string& text)
{
    std::vector<std::string> v;
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)TIMEOUT\s+(\d+))");
    for (const auto& c : calls)
    {
        if (c.label != "convert")
            continue;
        for (const auto& n : c.names)
        {
            bool found = false;
            std::smatch m;
            auto begin = text.cbegin();
            while (std::regex_search(begin, text.cend(), m, re))
            {
                std::istringstream is(m[1].str());
                std::string tok;
                while (is >> tok)
                {
                    if (tok != n)
                        continue;
                    found = true;
                    const int seconds = std::stoi(m[2].str());
                    if (seconds <= 0 || seconds > 600)
                        v.push_back("convert test TIMEOUT out of range "
                                    "(1..600 s): " + n);
                    break;
                }
                if (found)
                    break;
                begin = m.suffix().first;
            }
            if (!found)
                v.push_back("convert test without TIMEOUT: " + n);
        }
    }
    return v;
}

} // namespace

TEST_CASE("ctest labels guard parser: synthetic fixtures",
          "[labels][guard][unit]")
{
    SECTION("unlabeled test is a violation")
    {
        CHECK(!CheckLabelCoverage({"t1"}, {}).empty());
    }
    SECTION("duplicate label is a violation")
    {
        CHECK(!CheckLabelCoverage(
                  {"t1"}, {{{"t1"}, "fast"}, {{"t1"}, "inference"}})
                  .empty());
    }
    SECTION("unknown label is a violation")
    {
        CHECK(!CheckLabelCoverage({"t1"}, {{{"t1"}, "perf2"}}).empty());
    }
    SECTION("phantom label (unknown test) is a violation")
    {
        CHECK(!CheckLabelCoverage({}, {{{"ghost"}, "fast"}}).empty());
    }
    SECTION("clean labeling passes")
    {
        CHECK(CheckLabelCoverage(
                  {"t1", "t2"}, {{{"t1"}, "fast"}, {{"t2"}, "perf"}})
                  .empty());
    }
    SECTION("multi-line label call parses")
    {
        const std::string cm = R"(set_tests_properties(
    t1
    t2
    PROPERTIES LABELS fast)
set_tests_properties(t3 PROPERTIES LABELS perf)
set_tests_properties(t3 PROPERTIES SKIP_RETURN_CODE 4
    ))";
        const auto calls = ParseLabelCalls(cm);
        REQUIRE(calls.size() == 2);
        CHECK(calls[0].label == "fast");
        CHECK(calls[0].names == std::vector<std::string>{"t1", "t2"});
        CHECK(calls[1].label == "perf");
        CHECK(CheckLabelCoverage({"t1", "t2", "t3"}, calls).empty());
        CHECK(CheckPerfSkip(calls, cm).empty());
    }
}

TEST_CASE("ctest labels guard: every test labeled exactly once",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> texts{
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt")};
    // App-side test registrations are under the same label discipline.
    const fs::path apps = root / "apps";
    std::error_code ec;
    for (auto& e : fs::directory_iterator(apps, ec))
    {
        const fs::path tcm = e.path() / "tests" / "CMakeLists.txt";
        if (fs::exists(tcm, ec))
            texts.push_back(ReadFileOrThrow(tcm));
    }
    std::vector<std::string> tests;
    std::vector<LabelCall> calls;
    for (const auto& t : texts)
    {
        auto n = RegisteredTestNames(t);
        tests.insert(tests.end(), n.begin(), n.end());
        auto c = ParseLabelCalls(t);
        calls.insert(calls.end(), c.begin(), c.end());
    }
    INFO("registered tests: " << tests.size());
    REQUIRE(tests.size() >= 126);
    auto v = CheckLabelCoverage(tests, calls);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("label violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: perf tests declare SKIP_RETURN_CODE 4",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto calls = ParseLabelCalls(cmake_text);
    auto v = CheckPerfSkip(calls, cmake_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("perf skip violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: skip-capable tests declare SKIP_RETURN_CODE 4",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    auto tests = RegisteredTestNames(cmake_text);
    auto v = CheckSkipCapableHaveCode(tests, cmake_text, root / "tests");
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("skip-code violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: perf baseline is Release-gated",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckPerfBaselineReleaseGuard(root / "tests");
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("perf release-guard violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: convert tests declare bounded TIMEOUT",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto calls = ParseLabelCalls(cmake_text);
    auto v = CheckConvertTimeout(calls, cmake_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("convert timeout violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: presets reference valid labels",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string json_text =
        ReadFileOrThrow(root / "CMakePresets.json");
    auto v = CheckPresets(json_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("preset violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: router rules valid", "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string router_text =
        ReadFileOrThrow(root / "tools" / "test_router.cmake");
    auto v = CheckRouter(router_text);
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("router violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("ctest labels guard: tests/ file naming uniformity",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckTestsNaming(root / "tests");
    std::string joined;
    for (const auto& s : v)
        joined += s + "\n";
    INFO("naming violations:\n" << joined);
    REQUIRE(v.empty());
}
