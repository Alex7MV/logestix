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
    static const std::set<std::string> v = {"fast", "inference", "perf"};
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

std::vector<std::string> RegisteredTestNames(const std::string& text)
{
    std::vector<std::string> names;
    static const std::regex add_re(
        R"(add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
    // foreach(<var> <items...>): every item is a registered test name.
    static const std::regex fe_re(
        R"(foreach\s*\(\s*\w+((?:\s+[A-Za-z0-9_]+)+)\s*\))");
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
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto tests = RegisteredTestNames(cmake_text);
    INFO("registered tests: " << tests.size());
    REQUIRE(tests.size() >= 126);
    const auto calls = ParseLabelCalls(cmake_text);
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
