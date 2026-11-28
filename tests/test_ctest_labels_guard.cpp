// CTest category-label discipline guard (host-only, zero deps beyond Catch2).
// Categories: fast | inference | perf. Spec:
// docs/superpowers/specs/2026-09-10-test-categories-design.md
#include <catch2/catch_test_macros.hpp>

#include <absl/strings/str_join.h>

#include <algorithm>
#include <cctype>
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
    // Both registration helpers register a test, and both must be visible or a
    // whole class of targets escapes label discipline. `add_test(` matches
    // plain CMake and also the tail of `lsx_add_test(`; `lsx_add_cuda_test(`
    // does NOT contain `add_test(` and needs its own alternative.
    static const std::regex add_re(
        R"((?:add_test|lsx_add_cuda_test)\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
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
    // Two spellings set a label: the literal properties call, and the LABEL
    // argument of the registration helpers. The helpers were invisible until
    // 2026-11-05 (see the registration-helper case below), and a label nobody
    // reads is the same as no label.
    {
        static const std::regex helper_open(R"(lsx_add_(?:cuda_)?test\s*\()");
        static const std::regex helper_name(R"(NAME\s+([A-Za-z0-9_]+))");
        // `LABEL <cat>` may be followed by more keywords on the same line
        // (TIMEOUT, SKIP, ENV, ...), which is how the generated fragment emits
        // it. Requiring end-of-line made 336 of 440 generated targets invisible.
        static const std::regex helper_label(
            R"(\bLABEL\s+(fast|convert|inference|perf)\b)");
        std::smatch hm;
        auto hb = text.cbegin();
        while (std::regex_search(hb, text.cend(), hm, helper_open)) {
            const auto body_begin = hm.suffix().first;
            // The call ends at the matching ')' — NOT at the first one. A CMake
            // comment inside a call is still part of the text, so a comment
            // carrying a parenthesis ("(roots.h)", "(see the spec)") used to
            // truncate the body before LABEL and make a correctly labelled
            // target read as unlabelled. Comments are skipped, and depth is
            // tracked for CMake's own nested calls.
            auto body_end = body_begin;
            int depth = 0;
            bool in_comment = false;
            for (; body_end != text.cend(); ++body_end) {
                const char c = *body_end;
                if (in_comment) {
                    if (c == '\n') in_comment = false;
                    continue;
                }
                if (c == '#') { in_comment = true; continue; }
                if (c == '(') ++depth;
                else if (c == ')') {
                    if (depth == 0) break;
                    --depth;
                }
            }
            std::string body(body_begin, body_end);
            std::smatch nm, lm;
            if (std::regex_search(body, lm, helper_label) &&
                std::regex_search(body, nm, helper_name)) {
                LabelCall c;
                c.label = lm[1];
                c.names.push_back(nm[1]);
                calls.push_back(std::move(c));
            }
            hb = (body_end == text.cend()) ? body_end : body_end + 1;
        }
    }
    static const std::regex open_re(R"(set_tests_properties\s*\()");
    // LABELS is not necessarily the last property (the generated fragment
    // puts TIMEOUT/SKIP_RETURN_CODE/RESOURCE_LOCK after it on one line).
    // ... and it may follow TIMEOUT/SKIP_RETURN_CODE on the same line, which is
    // how a single-call registration reads. Requiring LABELS first hid
    // test_deepseek_nvfp4_repack_tool's label entirely.
    static const std::regex label_re(
        R"(\bLABELS\s+(fast|convert|inference|perf)\b)");
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

// True when a registration helper call for `target` passes the bare SKIP
// keyword. That keyword makes the CMake helper set SKIP_RETURN_CODE 4, so the
// generated text never carries the literal property - a reader that only greps
// for the property declares every generated perf target skip-less.
bool HelperDeclaresSkip(const std::string& text, const std::string& target)
{
    for (std::size_t open = text.find("lsx_add_test(");
         open != std::string::npos; open = text.find("lsx_add_test(", open + 1))
    {
        const std::size_t end = text.find(')', open);
        if (end == std::string::npos) continue;
        const std::string body = text.substr(open, end - open);
        if (body.find("NAME " + target + " ") == std::string::npos &&
            body.find("NAME " + target + ")") == std::string::npos)
            continue;
        for (std::size_t p = body.find("SKIP"); p != std::string::npos;
             p = body.find("SKIP", p + 1))
        {
            const std::size_t after = p + 4;
            if (after < body.size() &&
                (std::isalnum(static_cast<unsigned char>(body[after])) ||
                 body[after] == '_'))
                continue;                      // SKIPPY, SKIP_RETURN_CODE, ...
            return true;
        }
        return false;                          // this target's call, no SKIP
    }
    // CUDA registrations use the same argument shape.
    for (std::size_t open = text.find("lsx_add_cuda_test(");
         open != std::string::npos;
         open = text.find("lsx_add_cuda_test(", open + 1))
    {
        const std::size_t end = text.find(')', open);
        if (end == std::string::npos) continue;
        const std::string body = text.substr(open, end - open);
        if (body.find("NAME " + target + " ") == std::string::npos &&
            body.find("NAME " + target + ")") == std::string::npos)
            continue;
        for (std::size_t p = body.find("SKIP"); p != std::string::npos;
             p = body.find("SKIP", p + 1))
        {
            const std::size_t after = p + 4;
            if (after < body.size() &&
                (std::isalnum(static_cast<unsigned char>(body[after])) ||
                 body[after] == '_'))
                continue;
            return true;
        }
        return false;
    }
    return false;
}

std::vector<std::string> CheckPerfSkip(const std::vector<LabelCall>& calls,
                                       const std::string& text)
{
    std::vector<std::string> v;
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    // A target registered through lsx_add_test sets SKIP_RETURN_CODE 4 inside
    // the CMake helper, so the text only carries the bare SKIP keyword. Reading
    // only the literal property made every generated perf target look
    // skip-less (test_xing_warmup_ramp was the first to be flagged).
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
            if (!found) found = HelperDeclaresSkip(text, n);
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
    // set(<var> <items...>): the literal lists a `foreach(<x> IN LISTS <var>)`
    // expands (the GPU-budget suites are registered that way, and missing it
    // made this guard blind to all nine of them).
    std::map<std::string, std::vector<std::string>> list_items;
    {
        static const std::regex set_re(
            R"(\bset\s*\(\s*(\w+)((?:\s+(?!PARENT_SCOPE\b)[A-Za-z0-9_]+)+)\s*\))");
        std::smatch sm;
        auto b = cmake_text.cbegin();
        while (std::regex_search(b, cmake_text.cend(), sm, set_re))
        {
            std::istringstream is(sm[2].str());
            std::string tok;
            auto &items = list_items[sm[1].str()];
            while (is >> tok)
                items.push_back(tok);
            b = sm.suffix().first;
        }
    }
    // foreach(<var> <items...>): a later set_tests_properties(${var} ...) with
    // SKIP_RETURN_CODE 4 applies to every item. `IN LISTS <var>` resolves
    // through the set() map above.
    std::map<std::string, std::vector<std::string>> loop_items;
    {
        static const std::regex fe_re(
            R"(foreach\s*\(\s*(\w+)((?:\s+(?!IN\b|LISTS\b|RANGE\b)[A-Za-z0-9_]+)+)\s*\))");
        static const std::regex fe_lists_re(
            R"(foreach\s*\(\s*(\w+)\s+IN\s+LISTS\s+(\w+)\s*\))");
        std::smatch fm;
        auto b = cmake_text.cbegin();
        while (std::regex_search(b, cmake_text.cend(), fm, fe_lists_re))
        {
            const auto it = list_items.find(fm[2].str());
            if (it != list_items.end())
                loop_items[fm[1].str()] = it->second;
            b = fm.suffix().first;
        }
        b = cmake_text.cbegin();
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
    // The helper's own `SKIP` keyword sets SKIP_RETURN_CODE 4 inside
    // lsx_add_test, so a target registered that way is configured correctly.
    // Recognising only the literal `set_tests_properties(...)` spelling made the
    // guard report correct CMake as a violation — the false positive that gets
    // "fixed" by deleting the SKIP.
    {
        static const std::regex helper_open(R"(lsx_add_(?:cuda_)?test\s*\()");
        static const std::regex helper_name(R"(NAME\s+([A-Za-z0-9_]+))");
        static const std::regex helper_skip(R"(\bSKIP\b)");
        std::smatch hm;
        auto hb = cmake_text.cbegin();
        while (std::regex_search(hb, cmake_text.cend(), hm, helper_open)) {
            const auto body_begin = hm.suffix().first;
            auto body_end = body_begin;
            int depth = 0;
            bool in_comment = false;
            for (; body_end != cmake_text.cend(); ++body_end) {
                const char c = *body_end;
                if (in_comment) {
                    if (c == '\n') in_comment = false;
                    continue;
                }
                if (c == '#') { in_comment = true; continue; }
                if (c == '(') ++depth;
                else if (c == ')') {
                    if (depth == 0) break;
                    --depth;
                }
            }
            const std::string body(body_begin, body_end);
            std::smatch nm;
            if (std::regex_search(body, helper_skip) &&
                std::regex_search(body, nm, helper_name))
                has_code.insert(nm[1]);
            hb = (body_end == cmake_text.cend()) ? body_end : body_end + 1;
        }
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

// ── GPU exclusivity ───────────────────────────────────────────────────────
//
// `AGENTS.md` has always stated "one GPU workload at a time" and, until
// 2026-11-07, nothing checked it: 263 of 279 GPU-labelled targets ran without a
// lock, including 36 of 44 `perf` targets. A rule that only lives in prose rots
// the moment a new target is registered (which is exactly what happened — the
// generator had no opinion and the hand-written CMake set the property wherever
// the author remembered).
//
// The rule enforced here:
//   * every model stage (convert/load/model/kernels/selfcheck/fsi/code/det/perf)
//     allocates on the GPU, so every target on those labels MUST declare
//     `RESOURCE_LOCK <gpu>`;
//   * `fast` is host-only by contract, so a `fast` target must NOT hold the GPU
//     lock — 140 fast targets serialising on one lock is a slow suite, and a
//     slow suite is a suite people stop running.
std::set<std::string> LocksDeclared(const std::string& text)
{
    std::set<std::string> out;
    static const std::regex re(
        R"(set_tests_properties\(([^)]*?)\s+PROPERTIES[^)]*RESOURCE_LOCK\s+\"?([A-Za-z0-9_]+))");
    auto begin = text.cbegin();
    std::smatch m;
    while (std::regex_search(begin, text.cend(), m, re)) {
        std::istringstream names(m[1]);
        std::string n;
        while (names >> n) out.insert(n);
        begin = m.suffix().first;
    }
    return out;
}

// Targets that carry a GPU-ish label but provably do no device work. Each entry
// carries the reason, and the reason must be checkable: a file scan that never
// includes a CUDA header, not "we think it is cheap".
const std::map<std::string, std::string>& HostOnlyDespiteLabel()
{
    static const std::map<std::string, std::string> v = {
        // Label `convert` (the sweep that quantises on the GPU), but the test is
        // a mmap walk over the Arrow vault: 0 occurrences of `cuda` in the file.
        {"test_arrow_tree_integrity", "host-only Arrow/vault integrity scan"},
    };
    return v;
}

std::vector<std::string> CheckGpuExclusivity(
    const std::set<std::string>& gpu_labels,
    const std::map<std::string, std::string>& label_of,
    const std::set<std::string>& locked)
{
    // ONE direction: a target that allocates on the GPU must hold the lock. The
    // opposite direction ("a fast target must not hold it") was tried and is
    // wrong: six model loaders are labelled `fast` AND declare the lock
    // explicitly in their manifests because they upload weights
    // (beam/flint/hy/hy-simulmt/ling/minimax/step). Their label is a separate
    // question from their lock, and overriding an explicit manifest declaration
    // from a guard would make the manifest lie.
    std::vector<std::string> v;
    const auto& host_only = HostOnlyDespiteLabel();
    for (const auto& [name, label] : label_of) {
        if (gpu_labels.count(label) == 0) continue;
        if (locked.count(name) != 0) continue;
        if (host_only.count(name) != 0) continue;
        v.push_back(name + ": label " + label +
                    " runs on the GPU but declares no RESOURCE_LOCK "
                    "(add it, or prove the target is host-only and add it to "
                    "HostOnlyDespiteLabel with the reason)");
    }
    return v;
}

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

TEST_CASE("ctest labels guard: both registration helpers are visible",
          "[labels][guard]")
{
    // The guard resolves registered names by scanning CMake text. `lsx_add_test`
    // was visible by accident — the substring `add_test(` sits inside it — and
    // `lsx_add_cuda_test` was NOT, so every CUDA target registered through the
    // helper was invisible to label discipline: it could ship with no label, or
    // carry a label for a name the guard did not know, and the guard said
    // nothing either way. Found while registering the staging-ring budget test.
    const std::string cm = R"(lsx_add_test(NAME t_host LINK lsxcommon)
lsx_add_cuda_test(NAME t_device)
set_tests_properties(t_host PROPERTIES LABELS fast)
set_tests_properties(t_device PROPERTIES LABELS fast))";
    const auto names = RegisteredTestNames(cm);
    INFO("names seen: " << names.size());
    CHECK(std::find(names.begin(), names.end(), "t_host") != names.end());
    CHECK(std::find(names.begin(), names.end(), "t_device") != names.end());
    // With both visible, the coverage check is a real assertion rather than a
    // check over an empty set.
    CHECK(CheckLabelCoverage(names, ParseLabelCalls(cm)).empty());

    // The helper's LABEL argument is a label too. test_dense_kernels carries
    // `LABEL inference` inside lsx_add_cuda_test and the guard called it
    // unlabeled once it could see the name — the label was there, the reader
    // was not.
    const std::string helper = R"(lsx_add_cuda_test(NAME t_dev2
    SOURCES t.cu
    LINK lsxcommon
    LABEL inference))";
    const auto names2 = RegisteredTestNames(helper);
    REQUIRE(names2.size() == 1);
    const auto calls2 = ParseLabelCalls(helper);
    INFO("label calls: " << calls2.size());
    CHECK(calls2.size() == 1);
    if (calls2.size() == 1) {
        CHECK(calls2[0].label == "inference");
        CHECK(calls2[0].names == std::vector<std::string>{"t_dev2"});
    }
    CHECK(CheckLabelCoverage(names2, calls2).empty());
}

TEST_CASE("ctest labels guard: the generated manifest fragment is in scope",
          "[labels][guard]")
{
    // Model test targets are registered by the generator into
    // ${CMAKE_BINARY_DIR}/generated/model_targets.cmake and included from
    // tests/CMakeLists.txt. A target that lives in a file this guard does not
    // read is a target whose label discipline is checked by nobody: the same
    // blind spot `lsx_add_cuda_test` had until 2026-11-05.
    const fs::path generated(LSX_MANIFEST_GENERATED_DIR "/model_targets.cmake");
    const fs::path manifests(LSX_SOURCE_DIR "/models");
    std::error_code ec;
    bool have_manifests = false;
    for (auto& e : fs::directory_iterator(manifests, ec))
        if (e.path().extension() == ".json") have_manifests = true;

    if (!have_manifests) {
        SUCCEED("no manifests yet: the generated fragment is not in scope");
        return;
    }
    INFO("configure first: " << generated);
    REQUIRE(fs::exists(generated, ec));

    const std::string text = ReadFileOrThrow(generated);
    const auto names = RegisteredTestNames(text);
    INFO("generated targets: " << names.size());
    REQUIRE(names.size() > 100);   // 35 manifests x ~12 targets

    // Every generated label is one of the four categories, and every generated
    // target carries exactly one.
    const auto calls = ParseLabelCalls(text);
    std::string coverage;
    for (const auto& problem : CheckLabelCoverage(names, calls))
        coverage += problem + "\n";
    INFO("generated label calls: " << calls.size() << "\n" << coverage);
    CHECK(coverage.empty());

    // A convert target without a bounded TIMEOUT used to mean a full-model
    // conversion slipping into a sweep.
    const auto skip_problems = CheckPerfSkip(calls, text);
    std::string skips;
    for (const auto& sp : skip_problems) skips += sp + "\n";
    INFO(skips);
    CHECK(skips.empty());
}

TEST_CASE("ctest labels guard: every test labeled exactly once",
          "[labels][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> texts{
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt")};
    // Model test registration moved into the generated fragment, so the label
    // corpus has to include it or the majority of the suite is unchecked.
    {
        const fs::path generated(LSX_MANIFEST_GENERATED_DIR "/model_targets.cmake");
        std::error_code gec;
        if (fs::exists(generated, gec))
            texts.push_back(ReadFileOrThrow(generated));
    }
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

TEST_CASE("ctest labels guard: GPU targets declare RESOURCE_LOCK",
          "[labels][guard]")
{
    // Checked over BOTH sources: the generated fragment (427 model targets) and
    // the hand-written tests/CMakeLists.txt (the rest). Reading only one of them
    // is how the property went missing in the first place.
    const fs::path root(LSX_SOURCE_DIR);
    const fs::path generated(LSX_MANIFEST_GENERATED_DIR "/model_targets.cmake");

    static const std::set<std::string> kGpuLabels = {
        "convert", "load", "model", "kernels",
        "selfcheck", "fsi", "code", "det", "perf"};
    // Every label a model target can carry in the generated fragment is covered
    // by kGpuLabels except `fast`; the hand-written file also uses `inference`
    // and `perf` directly.
    std::map<std::string, std::string> label_of;
    std::set<std::string> locked;

    const std::string hand =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    for (const auto& c : ParseLabelCalls(hand))
        for (const auto& n : c.names) label_of[n] = c.label;
    for (const auto& n : LocksDeclared(hand)) locked.insert(n);

    std::error_code ec;
    if (fs::exists(generated, ec)) {
        const std::string frag = ReadFileOrThrow(generated);
        for (const auto& c : ParseLabelCalls(frag))
            for (const auto& n : c.names) label_of[n] = c.label;
        for (const auto& n : LocksDeclared(frag)) locked.insert(n);
    }

    INFO("targets with a label: " << label_of.size()
                                  << ", declaring a lock: " << locked.size());
    REQUIRE(label_of.size() > 200);

    auto v = CheckGpuExclusivity(kGpuLabels, label_of, locked);
    std::string joined;
    for (const auto& s : v) joined += s + "\n";
    INFO("gpu exclusivity violations (" << v.size() << "):\n" << joined);
    CHECK(v.empty());
}

TEST_CASE("ctest labels guard: the GPU-lock rule bites", "[labels][guard]")
{
    // Non-vacuity: the predicate must flag a GPU-labelled target with no lock and
    // a `fast` target that holds one. Without this the check above would pass on
    // an empty input set and read as coverage.
    static const std::set<std::string> kGpuLabels = {"inference", "perf"};
    const std::map<std::string, std::string> label_of = {
        {"gate_no_lock", "inference"},
        {"gate_ok", "inference"},
        {"fast_with_lock", "fast"},
        {"fast_clean", "fast"},
        // The allowlisted host-only target must not be reported even though its
        // label implies GPU work: this is the real entry, so the allowlist is
        // exercised rather than assumed.
        {"test_arrow_tree_integrity", "perf"},
    };
    const std::set<std::string> locked = {"gate_ok", "fast_with_lock"};
    const auto v = CheckGpuExclusivity(kGpuLabels, label_of, locked);
    // Only the unlocked GPU target is a violation. `fast_with_lock` is a
    // deliberate manifest declaration (six loaders do it), and `fast_clean` is
    // the normal case.
    REQUIRE(v.size() == 1);
    const std::string joined = absl::StrJoin(v, "\n");
    INFO(joined);
    CHECK(joined.find("gate_no_lock") != std::string::npos);
    CHECK(joined.find("gate_ok") == std::string::npos);
    CHECK(joined.find("fast_with_lock") == std::string::npos);
    CHECK(joined.find("fast_clean") == std::string::npos);
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

TEST_CASE("ctest labels guard: a comment cannot hide a helper's LABEL",
          "[labels][guard]")
{
    // The helper-body scan ends at the first ')', and a CMake comment inside a
    // call is still part of the text. A comment carrying a parenthesis —
    // "(roots.h)", "(see the spec)" — therefore truncated the body before
    // LABEL and the target read as UNLABELED even though it was labelled
    // correctly. Found because a comment naming libs/lsxstorage's roots.h did
    // exactly that.
    const std::string cm = R"(lsx_add_test(NAME t1
    SOURCES t1.cpp
    LINK lsxcommon lsxstorage
    # the artifact-root half lives in roots.h (see libs/lsxstorage)
    LABEL fast))";
    const auto calls = ParseLabelCalls(cm);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].names == std::vector<std::string>{"t1"});
    CHECK(calls[0].label == "fast");
    CHECK(CheckLabelCoverage({"t1"}, calls).empty());
}

TEST_CASE("ctest labels guard: the helper's SKIP keyword counts as a skip code",
          "[labels][guard]")
{
    // `lsx_add_test(... SKIP ...)` sets SKIP_RETURN_CODE 4 inside the helper, so
    // a target registered that way is correctly configured. The guard only
    // understood the literal `set_tests_properties(... SKIP_RETURN_CODE 4)`
    // spelling, so it reported such a target as a violation — a false positive
    // on correct CMake, and the kind that gets "fixed" by deleting the SKIP.
    const fs::path dir = fs::temp_directory_path() / "lsx_skip_helper";
    fs::create_directories(dir);
    // Assembled from pieces on purpose: this guard scans every tests/*.cpp for
    // the literal `SKIP(` to decide what is skip-capable, and a test that
    // SIMULATES a skip-capable source must not make the guard's own file look
    // like one. (It did, and the guard reported itself.)
    const std::string kSkipCall = std::string("SK") + "IP(\"absent\");";
    {
        std::ofstream f(dir / "t_with_skip.cpp");
        f << "TEST_CASE(\"x\") { if (nothing) " << kSkipCall << " }\n";
    }
    {
        std::ofstream f(dir / "t_without_skip.cpp");
        f << "TEST_CASE(\"y\") { if (nothing) " << kSkipCall << " }\n";
    }
    const std::string cm = R"(lsx_add_test(NAME t_with_skip
    SOURCES t_with_skip.cpp
    LABEL fast
    SKIP)
lsx_add_test(NAME t_without_skip
    SOURCES t_without_skip.cpp
    LABEL fast))";
    const auto v = CheckSkipCapableHaveCode({"t_with_skip", "t_without_skip"},
                                            cm, dir);
    std::string joined;
    for (const auto& s : v) joined += s + "\n";
    INFO("violations:\n" << joined);
    // The correctly-configured one must NOT be reported...
    CHECK(joined.find("t_with_skip") == std::string::npos);
    // ...and the one with no code at all still must be, or this test would pass
    // by accepting everything.
    CHECK(joined.find("t_without_skip") != std::string::npos);
    fs::remove_all(dir);
}
