// Infrastructure guard: the centralizations of 2026-11-03 (dependency include
// paths, the shared tensor reader/writers, the CUDA arch variable) must not be
// undone one call site at a time.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

// Injected by tests/CMakeLists.txt: the cmake that configured this build, so the
// probe uses the same binary (and the same version) the validator lives in.
#ifndef LSX_CMAKE_COMMAND
#define LSX_CMAKE_COMMAND "cmake"
#endif

namespace {

std::string ReadFile(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// CMake comments start with '#'.
std::string StripHashComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool comment = false;
    for (char c : text) {
        if (comment) {
            if (c == '\n') {
                comment = false;
                out += c;
            }
            continue;
        }
        if (c == '#') {
            comment = true;
            continue;
        }
        out += c;
    }
    return out;
}

// Strips // and /* */ comments so a commented-out signature cannot trip the
// definition scan. A state machine, not a regex: line numbers must survive
// multi-line block comments.
std::string StripCppComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool line_comment = false;
    bool block_comment = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (line_comment) {
            if (c == '\n') {
                line_comment = false;
                out += c;
            }
            continue;
        }
        if (block_comment) {
            if (c == '*' && i + 1 < text.size() && text[i + 1] == '/') {
                block_comment = false;
                ++i;
            } else if (c == '\n') {
                out += c;
            }
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            line_comment = true;
            ++i;
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            block_comment = true;
            ++i;
            continue;
        }
        out += c;
    }
    return out;
}

std::vector<std::string> Lines(const std::string &text)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

// Rule 1: no CMake call site may spell the FetchContent build directory.
std::vector<int> FindDependencyPaths(const std::string &text)
{
    static const std::regex re(
        R"(\$\{CMAKE_BINARY_DIR\}/_deps|(^|[^A-Za-z0-9_])_deps/)");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        if (std::regex_search(line, re)) hits.push_back(line_no);
    }
    return hits;
}

// Rule 2: each shared helper is DEFINED once (a call or an '=' initializer is
// not a definition, so the match is anchored on the return type).
std::vector<int> FindDuplicateDefinitions(const std::string &text)
{
    static const std::regex re(
        R"((?:std::vector\s*<\s*uint8_t\s*>|arrow::Status|bool|)"
        R"(std::filesystem::path)\s+)"
        R"((ReadTensorData|read_tensor_data|WriteBatch|write_batch|)"
        R"(WriteExpertArrow|WriteExpertFile|write_expert_file|ExpertArrowPath|)"
        R"(QuantizeSlice|QuantizeSliceBf16|quantize_slice)\s*\()");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripCppComments(text))) {
        ++line_no;
        if (std::regex_search(line, re)) hits.push_back(line_no);
    }
    return hits;
}

// Rule 3: device codegen arch is LSX_CUDA_ARCH or OFF only.
std::vector<int> FindArchLiterals(const std::string &text)
{
    static const std::regex call(R"(CUDA_ARCHITECTURES\s+([^\s\)]+))");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        std::smatch m;
        auto begin = line.cbegin();
        while (std::regex_search(begin, line.cend(), m, call)) {
            std::string arg = m[1].str();
            if (arg.size() >= 2 && arg.front() == '"' && arg.back() == '"')
                arg = arg.substr(1, arg.size() - 2);
            if (arg != "OFF" &&
                arg.find("${LSX_CUDA_ARCH}") == std::string::npos)
                hits.push_back(line_no);
            begin = m.suffix().first;
        }
    }
    return hits;
}

// Rule 8: a build tree records the toolkit it was configured with, so the next
// configure after a toolkit upgrade stops instead of silently reusing objects
// compiled against the previous headers. CUDA's own headers carry *package*
// mtimes, so after an upgrade they are OLDER than the objects built before it
// and no build-system rule considers them changed — this is the only rule that
// can see the swap.
//
// The rule reads the BODY of `lsx_stamp_cuda_toolkit()`, not the file: the root
// CMakeLists already contains file(READ/file(WRITE/STREQUAL for unrelated
// reasons (dependency patches), so a whole-file scan would be satisfied by code
// that has nothing to do with the stamp.
std::vector<std::string> StampRuleProblems(const std::string &text)
{
    std::vector<std::string> problems;
    const std::string fn = "function(lsx_stamp_cuda_toolkit";
    const auto begin = text.find(fn);
    if (begin == std::string::npos) {
        problems.push_back("no function(lsx_stamp_cuda_toolkit ...)");
        return problems;
    }
    const auto finish = text.find("endfunction()", begin);
    if (finish == std::string::npos) {
        problems.push_back("lsx_stamp_cuda_toolkit() has no endfunction()");
        return problems;
    }
    const std::string body = text.substr(begin, finish - begin);
    const auto need = [&](const char *marker, const char *what) {
        if (body.find(marker) == std::string::npos) problems.push_back(what);
    };
    need("lsx_cuda_toolkit_stamp.txt", "the body does not name the stamp file");
    need("file(WRITE", "the body never writes the stamp");
    need("file(READ", "the body never reads the stamp back");
    need("STREQUAL", "the body never compares the stamp");
    need("FATAL_ERROR", "a changed stamp cannot stop the configure");
    return problems;
}

std::vector<fs::path> CmakeFiles(const fs::path &root)
{
    std::vector<fs::path> out{root / "CMakeLists.txt"};
    std::error_code ec;
    if (fs::exists(root / "cmake", ec)) {
        for (const auto &e : fs::directory_iterator(root / "cmake", ec))
            if (e.is_regular_file() && e.path().extension() == ".cmake")
                out.push_back(e.path());
    }
    for (const char *dir : {"libs", "apps", "tests"}) {
        ec.clear();
        for (fs::recursive_directory_iterator it(root / dir, ec), end;
             it != end; it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file() &&
                it->path().filename() == "CMakeLists.txt")
                out.push_back(it->path());
        }
    }
    return out;
}

// Production sources only: test fixtures may legitimately name these symbols.
std::vector<fs::path> SourceFiles(const fs::path &root)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const char *dir : {"libs", "apps"}) {
        ec.clear();
        for (fs::recursive_directory_iterator it(root / dir, ec), end;
             it != end; it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file()) continue;
            const auto ext = it->path().extension().string();
            if (ext == ".cpp" || ext == ".cu" || ext == ".h" || ext == ".cuh")
                out.push_back(it->path());
        }
    }
    return out;
}

std::string Rel(const fs::path &root, const fs::path &p)
{
    return fs::relative(p, root).generic_string();
}

std::string JoinHits(const std::vector<std::string> &hits)
{
    std::string joined;
    for (const auto &h : hits) joined += h + "\n";
    return joined;
}

} // namespace

TEST_CASE("infra guard: dependency-path scanner", "[guard][infra]")
{
    CHECK(FindDependencyPaths(
              "target_include_directories(t PRIVATE "
              "${CMAKE_BINARY_DIR}/_deps/arrow-src)")
              .size() == 1);
    CHECK(FindDependencyPaths(
              "include_directories(${CMAKE_BINARY_DIR}/_deps/b/include)")
              .size() == 1);
    CHECK(FindDependencyPaths(
              "include_directories(${FETCHCONTENT_BASE_DIR}/arrow-src/cpp/src)")
              .empty());
    CHECK(FindDependencyPaths("# legacy ${CMAKE_BINARY_DIR}/_deps/c").empty());
}

TEST_CASE("infra guard: duplicate-definition scanner", "[guard][infra]")
{
    CHECK(FindDuplicateDefinitions(
              "std::vector<uint8_t> ReadTensorData(const T &a, const T &b)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "arrow::Status WriteExpertFile(const P &p, int l, int e)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "bool QuantizeSlice(const uint8_t *d, int64_t n)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "std::filesystem::path ExpertArrowPath(const P &p, int l, int e)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "auto d = ReadTensorData(input_dir, shard, t);")
              .empty());
    CHECK(FindDuplicateDefinitions("if (QuantizeSlice(x, 4, dt, r)) {").empty());
    CHECK(FindDuplicateDefinitions(
              "// bool QuantizeSlice(const uint8_t *d, int64_t n)")
              .empty());
}

TEST_CASE("infra guard: arch-literal scanner", "[guard][infra]")
{
    CHECK(FindArchLiterals(
              "set_target_properties(t PROPERTIES CUDA_ARCHITECTURES "
              "\"120-real;121-real\")")
              .size() == 1);
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES \"90\"").size() == 1);
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES ${LSX_CUDA_ARCH}").empty());
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES \"${LSX_CUDA_ARCH}\"").empty());
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES OFF").empty());
}

TEST_CASE("infra guard: no dependency build paths in CMake", "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> bad;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindDependencyPaths(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("dependency path hits (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}

// A fixture/golden directory macro that falls back to a RELATIVE path is resolved
// against the process CWD, and ctest runs from the build dir — so the test finds no
// fixture, SKIPs, and the skip reads as "not applicable" while the file sits in the
// source tree. That is how both granite-switch model targets lost their real-weight
// tool round-trip without a single red target (docs/OPEN_TASKS.md G7).
//
// The rule: if a .cpp has `#ifndef X / #define X "<relative>"` for a directory
// macro, every target that compiles that source must pass X from CMake with an
// absolute path. Fallbacks are still useful (a developer can compile a single
// target), they just must not be the one that ships.
TEST_CASE("infra guard: a fixture dir macro is never satisfied by its relative "
          "fallback", "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const fs::path tests = root / "tests";

    // Macro name -> sources carrying a relative fallback for it.
    const std::regex fallback(
        R"RE(#ifndef (LSX_[A-Z0-9_]*_DIR)\s*\n#define \1 "([a-z][^"]*)")RE");
    std::map<std::string, std::vector<std::string>> by_macro;
    for (const auto &p : fs::directory_iterator(tests)) {
        if (p.path().extension() != ".cpp") continue;
        const std::string text = ReadFile(p.path());
        for (std::sregex_iterator it(text.begin(), text.end(), fallback), end;
             it != end; ++it)
            by_macro[(*it)[1].str()].push_back(p.path().filename().string());
    }
    // The parity gate is the exception by construction: it #errors when
    // LSX_GOLDEN_DIR is absent rather than falling back, so it has no entry here.
    REQUIRE(by_macro.find("LSX_GOLDEN_DIR") == by_macro.end());
    // Non-vacuity: a scan that finds nothing would pass every assertion below.
    // Ten directory macros carried a relative fallback when this rule was
    // written; the floor is 5 so removing one deliberately does not force an edit.
    INFO("macros with a relative fallback: " << by_macro.size());
    REQUIRE(by_macro.size() >= 5);

    const std::string cmake = ReadFile(tests / "CMakeLists.txt");
    // Every add_executable block, up to the next one.
    const std::regex exe(R"(add_executable\((\w+)\s+([\w.]+))");
    std::vector<std::pair<std::string, std::string>> targets;   // name, source
    std::vector<std::size_t> starts;
    for (std::sregex_iterator it(cmake.begin(), cmake.end(), exe), end; it != end;
         ++it) {
        targets.emplace_back((*it)[1].str(), (*it)[2].str());
        starts.push_back(static_cast<std::size_t>((*it).position()));
    }
    starts.push_back(cmake.size());

    INFO("add_executable blocks scanned: " << targets.size());
    REQUIRE(targets.size() >= 100);      // likewise: an empty scan proves nothing

    std::vector<std::string> gaps;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const auto &[name, src] = targets[i];
        const auto found = by_macro.end();
        for (auto m = by_macro.begin(); m != by_macro.end(); ++m) {
            if (std::find(m->second.begin(), m->second.end(), src) == m->second.end())
                continue;
            const std::string body = cmake.substr(starts[i], starts[i + 1] - starts[i]);
            if (body.find(m->first) == std::string::npos)
                gaps.push_back(name + " compiles " + src + " without defining " +
                               m->first);
        }
        (void)found;
    }
    INFO("targets relying on a relative fixture fallback (" << gaps.size() << "):\n"
         << [&] {
                std::string j;
                for (const auto &g : gaps) j += "  " + g + "\n";
                return j;
            }());
    REQUIRE(gaps.empty());
}

TEST_CASE("infra guard: shared converter helpers have one definition",
          "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    // converter.cpp implements the helpers; converter.h and safetensors.h
    // declare them. Everywhere else a matching signature is a revived copy.
    const fs::path canonical =
        fs::weakly_canonical(root / "libs/lsxcommon/src/converter.cpp");
    const fs::path canonical_header = fs::weakly_canonical(
        root / "libs/lsxcommon/include/lsxcommon/converter.h");
    const fs::path canonical_reader_header = fs::weakly_canonical(
        root / "libs/lsxcommon/include/lsxcommon/safetensors.h");
    std::vector<std::string> bad;
    for (const auto &p : SourceFiles(root)) {
        const fs::path abs = fs::weakly_canonical(p);
        if (abs == canonical || abs == canonical_header ||
            abs == canonical_reader_header)
            continue;
        for (int line : FindDuplicateDefinitions(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    }
    INFO("duplicate definitions (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}

TEST_CASE("infra guard: device codegen arch is centralized", "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> bad;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindArchLiterals(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("arch literal hits (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}

TEST_CASE("infra guard: build-toolkit stamp scanner", "[guard][infra]")
{
    const std::string body =
        "function(lsx_stamp_cuda_toolkit)\n"
        "  set(f \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\")\n"
        "  if(EXISTS \"${f}\")\n"
        "    file(READ \"${f}\" old)\n"
        "    if(NOT old STREQUAL new)\n"
        "      message(FATAL_ERROR \"wipe the build directory\")\n"
        "    endif()\n"
        "  endif()\n"
        "  file(WRITE \"${f}\" \"${new}\")\n"
        "endfunction()\n";
    CHECK(StampRuleProblems(body).empty());
    // Negative controls: each half is caught on its own, so the rule cannot be
    // satisfied by a comment or by write-only "decoration".
    CHECK(StampRuleProblems("").size() == 1);
    CHECK(StampRuleProblems("function(lsx_stamp_cuda_toolkit)").size() == 1);
    CHECK(StampRuleProblems(
              "function(lsx_stamp_cuda_toolkit)\n"
              "  file(WRITE \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\" x)\n"
              "endfunction()\n")
              .size() == 3);
    CHECK(StampRuleProblems(
              "function(lsx_stamp_cuda_toolkit)\n"
              "  file(WRITE \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\" x)\n"
              "  file(READ \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\" y)\n"
              "endfunction()\n")
              .size() == 2);
    CHECK(StampRuleProblems(
              "function(lsx_stamp_cuda_toolkit)\n"
              "  file(WRITE \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\" x)\n"
              "  file(READ \"${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt\" y)\n"
              "  if(NOT y STREQUAL x)\n  endif()\n"
              "endfunction()\n")
              .size() == 1);
    // The body is what counts: the same markers OUTSIDE the function are not
    // the rule (this is the whole-file-scan false pass the rule exists to avoid).
    CHECK(StampRuleProblems(
              "file(WRITE ${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt x)\n"
              "file(READ ${CMAKE_BINARY_DIR}/lsx_cuda_toolkit_stamp.txt y)\n"
              "if(y STREQUAL x)\nendif()\nmessage(FATAL_ERROR \"x\")\n"
              "function(lsx_stamp_cuda_toolkit)\nendfunction()\n")
              .size() == 5);
}

TEST_CASE("infra guard: the root CMakeLists stamps the build toolkit",
          "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const auto problems = StampRuleProblems(ReadFile(root / "CMakeLists.txt"));
    INFO("stamp rule problems (" << problems.size() << "):\n"
                                 << JoinHits(problems));
    REQUIRE(problems.empty());
}

// Rule 9: the family-specific arch suffix must never reach CUDA_ARCHITECTURES.
// CMake 4.x does not know `f` (measured on CMake 4.2.3): given "120f" — or even
// "120-real;121f-real" — it silently emits `compute_75`, which still runs through
// PTX JIT and is SLOWER than the plain sm_120 code it replaced. The expansion
// into nvcc's own --generate-code flags is the only spelling CMake passes
// through unharmed, so the value must stay OUT of set_target_properties.
std::vector<int> FindFamilySuffixInArch(const std::string &text)
{
    static const std::regex call(R"(CUDA_ARCHITECTURES\s+([^\s\)]+))");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        std::smatch m;
        auto begin = line.cbegin();
        while (std::regex_search(begin, line.cend(), m, call)) {
            std::string arg = m[1].str();
            if (arg.size() >= 2 && arg.front() == '"' && arg.back() == '"')
                arg = arg.substr(1, arg.size() - 2);
            // Any family spelling (120f / 121f / 120f-real) reaching this
            // property is the silent-downgrade bug. So is a suffixed Ampere
            // arch: sm_86f/sm_87a do not exist, and if one reached this property
            // CMake 4.x would mangle it exactly as it mangles `120f`. Plain
            // `86`/`87` are legitimate in CUDA_ARCHITECTURES (CMake knows them),
            // so only the SUFFIXED form is flagged — the rule is about the
            // suffix, not about the arch family.
            if (arg == "OFF") { begin = m.suffix().first; continue; }
            static const std::regex fam(
                R"((^|[^0-9])(12[01][af]?f|8[67][af])([^0-9]|$))");
            if (std::regex_search(arg, fam)) hits.push_back(line_no);
            begin = m.suffix().first;
        }
    }
    return hits;
}

// The same silent downgrade hides in a literal --generate-code line: a target
// that spells `compute_120,code=sm_120` by hand bypasses LSX_CUDA_ARCH entirely
// and keeps emitting non-family SASS (measured 2026-11-05: 30 test targets did
// exactly that while the rest of the tree moved to `f`). Two spellings are
// legitimate and stay literal: `compute_12xa` (architecture-specific, required
// by tcgen05 in lsx_tc_gemm.cu) and `compute_12xf` (the family form itself).
//
// Ampere is in the pattern for the same reason the family suffix is: the moment
// 86/87 became supported values, `--generate-code=arch=compute_86` became a way
// to opt one target out of LSX_CUDA_ARCH without anyone noticing. The scanner
// matched only `compute_(12[0-9])` while Ampere was unsupported, which was true
// then and is a hole now.
std::vector<int> FindNonFamilyGencodeLiterals(const std::string &text)
{
    static const std::regex lit(
        R"(--generate-code=arch=compute_(12[0-9]|8[67])([af]?),code=)");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        std::smatch m;
        auto begin = line.cbegin();
        while (std::regex_search(begin, line.cend(), m, lit)) {
            // Only `a` (tcgen05) and `f` (the family spelling) are sanctioned.
            if (m[2].str() != "a" && m[2].str() != "f")
                hits.push_back(line_no);
            begin = m.suffix().first;
        }
    }
    return hits;
}

// ── The arch validator, exercised for real ────────────────────────────────
//
// Everything above in this file INSPECTS CMake text. That is the right tool for
// "nobody spelled a literal gencode flag", and the wrong tool for "does the
// validator accept 86". A scanner cannot answer the second question: it passed
// for two years while `^12[01][af]?$` still rejected every Ampere arch, because
// finding the string "12[01]" is not evaluating it.
//
// So this runs the function itself, through `cmake -P`, and reports its exit
// code plus combined stdout/stderr. The probe script includes the SAME module
// the build includes — one copy of the pattern, so this test cannot go green
// against a stale duplicate (which is exactly how a hand-copied regex drifts).
struct ArchProbe {
    int rc = -1;
    std::string out;      // stdout + stderr, as the operator would see it
};

ArchProbe ProbeArch(const std::string &arch)
{
    ArchProbe r;
    const fs::path script =
        fs::path(LSX_SOURCE_DIR) / "tests" / "cmake" / "probe_arch.cmake";
    const fs::path module = fs::path(LSX_SOURCE_DIR) / "cmake" / "LsxArch.cmake";
    const std::string cmd = std::string("\"") + LSX_CMAKE_COMMAND + "\" -D PROBE_ARCH=\"" +
                            arch + "\" -D LsxArch=\"" + module.string() +
                            "\" -P \"" + script.string() + "\" 2>&1";
#if defined(_WIN32)
    FILE *pipe = _popen(cmd.c_str(), "r");
#else
    FILE *pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) return r;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) r.out += buf;
#if defined(_WIN32)
    r.rc = _pclose(pipe);
#else
    const int status = pclose(pipe);
    // WEXITSTATUS without <sys/wait.h>: a plain shell exit code is what cmake
    // reports, and the low byte carries it.
    r.rc = (status >> 8) & 0xff;
#endif
    return r;
}

TEST_CASE("infra guard: family arch suffix scanner", "[guard][infra]")
{
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES \"120f\"").size() == 1);
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES \"120f;121f\"").size() == 1);
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES 120f-real").size() == 1);
    CHECK(FindFamilySuffixInArch(
              "CUDA_ARCHITECTURES \"120-real;121f-real\"")
              .size() == 1);
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES OFF").empty());
    CHECK(FindFamilySuffixInArch(
              "CUDA_ARCHITECTURES \"120-real;121-real\"")
              .empty());
    CHECK(FindFamilySuffixInArch(
              "--generate-code=arch=compute_120f,code=[compute_120f,sm_120f]")
              .empty());
}

TEST_CASE("infra guard: family arches reach nvcc, never CUDA_ARCHITECTURES",
          "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> bad;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindFamilySuffixInArch(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("family-suffix hits (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());

    // And the mechanism must exist: a single cache value plus the expansion.
    const std::string root_cmake = ReadFile(root / "CMakeLists.txt");
    REQUIRE(root_cmake.find("LSX_CUDA_GENCODE_FLAGS") != std::string::npos);
    REQUIRE(root_cmake.find("lsx_cuda_gencode_flags") != std::string::npos);

    // The default spelling is the measured one: family targets were neutral, so
    // the shipped value must not silently become `f` (see the design doc).
    REQUIRE(root_cmake.find("set(LSX_CUDA_ARCH \"120;121\"") !=
            std::string::npos);

    // Literal --generate-code lines are the same downgrade in disguise.
    std::vector<std::string> literal;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindNonFamilyGencodeLiterals(ReadFile(p)))
            literal.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("literal gencode hits (" << literal.size() << "):\n"
                                  << JoinHits(literal));
    REQUIRE(literal.empty());
}

TEST_CASE("infra guard: gencode-literal scanner", "[guard][infra]")
{
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_120,code=sm_120")
              .size() == 1);
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_121,code=sm_121")
              .size() == 1);
    // Sanctioned spellings.
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_120a,code=sm_120a")
              .empty());
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_120f,code=[compute_120f,sm_120f]")
              .empty());
    CHECK(FindNonFamilyGencodeLiterals("${LSX_CUDA_GENCODE_FLAGS}").empty());
}

// ── Task 11: Ampere is a build target ──────────────────────────────────────
//
// `LSX_CUDA_ARCH=86;87` must configure AND emit both real SASS images. The
// assertions name the flags the function actually produces
// (`--generate-code=arch=compute_86,code=[compute_86,sm_86]`): what has to be
// true is that each arch is requested as a virtual arch AND that its SASS is
// named, because a PTX-only entry still "works" through JIT — far slower than
// the SASS it silently replaced, which is the failure mode this whole mechanism
// exists to prevent.

TEST_CASE("infra guard: the arch validator accepts 86;87 and emits both SASS",
          "[guard][infra][arch]")
{
    const ArchProbe r = ProbeArch("86;87");
    INFO(r.out);
    INFO("exit=" << r.rc);
    REQUIRE(r.rc == 0);
    CHECK(r.out.find("arch=compute_86") != std::string::npos);
    CHECK(r.out.find("arch=compute_87") != std::string::npos);
    // The real image, per arch — not just the virtual one.
    CHECK(r.out.find("sm_86") != std::string::npos);
    CHECK(r.out.find("sm_87") != std::string::npos);
}

TEST_CASE("infra guard: the Blackwell default still expands unchanged",
          "[guard][infra][arch]")
{
    // Regression in both directions: widening the pattern must not alter what
    // the shipped default emits. Same shape as before Task 11.
    const ArchProbe r = ProbeArch("120;121");
    INFO(r.out);
    REQUIRE(r.rc == 0);
    CHECK(r.out.find("arch=compute_120") != std::string::npos);
    CHECK(r.out.find("sm_120") != std::string::npos);
    CHECK(r.out.find("arch=compute_121") != std::string::npos);
    CHECK(r.out.find("sm_121") != std::string::npos);
    // Family spellings are still accepted on 12x (the tcgen05 experiment).
    const ArchProbe fam = ProbeArch("120f;121f");
    INFO(fam.out);
    CHECK(fam.rc == 0);
    CHECK(fam.out.find("sm_120f") != std::string::npos);
}

TEST_CASE("infra guard: a forbidden arch is still refused, and says why",
          "[guard][infra][arch]")
{
    // Hopper, Ada and Ampere's datacenter sibling stay out (AGENTS.md); so do
    // architectures no shipped toolkit can emit. Each must fail with a reason an
    // operator can act on, not a bare "not supported".
    for (const char *arch : {"90", "89", "80", "75", "100"}) {
        INFO("arch=" << arch);
        const ArchProbe r = ProbeArch(arch);
        INFO(r.out);
        CHECK(r.rc != 0);
        CHECK(r.out.find("supported") != std::string::npos);
        // And it names what IS supported, including the Ampere rows.
        CHECK(r.out.find("86") != std::string::npos);
        CHECK(r.out.find("87") != std::string::npos);
    }
}

TEST_CASE("infra guard: an Ampere arch refuses the 12x-only suffixes",
          "[guard][infra][arch]")
{
    // There is no sm_86a and no sm_86f: GA10x has no family-specific target and
    // no tcgen05. Accepting the suffix here would hand nvcc an arch it rejects
    // with a message about the toolkit rather than about the spelling.
    for (const char *arch : {"86a", "86f", "87a", "87f"}) {
        INFO("arch=" << arch);
        const ArchProbe r = ProbeArch(arch);
        INFO(r.out);
        CHECK(r.rc != 0);
        CHECK(r.out.find("suffix") != std::string::npos);
    }
    // A mixed list fails on the offending entry, not silently on the list.
    const ArchProbe mixed = ProbeArch("87;86f");
    INFO(mixed.out);
    CHECK(mixed.rc != 0);
}

TEST_CASE("infra guard: a gencode literal that names an Ampere arch is caught",
          "[guard][infra][arch]")
{
    // The scanner used to match only `compute_(12[0-9])`, so a target that
    // spelled `--generate-code=arch=compute_86` by hand bypassed LSX_CUDA_ARCH
    // entirely — the same bypass the 12x scanner was written for, newly
    // reachable the moment 86 became a supported value.
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_86,code=sm_86")
              .size() == 1);
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_87,code=[compute_87,sm_87]")
              .size() == 1);
    // A family suffix on an Ampere arch is refused by the validator, so the
    // scanner does not need to special-case it — but it must not flag the
    // sanctioned 12x spellings either.
    CHECK(FindNonFamilyGencodeLiterals(
              "--generate-code=arch=compute_121a,code=sm_121a")
              .empty());
}

TEST_CASE("infra guard: a family suffix on an Ampere arch is flagged in "
          "CUDA_ARCHITECTURES",
          "[guard][infra][arch]")
{
    // `86f` does not exist, but if it ever reached CUDA_ARCHITECTURES CMake 4.x
    // would mangle it the way it mangles `120f`. Flag the suffix, not the plain
    // arch: `CUDA_ARCHITECTURES 86` is a legitimate CMake spelling.
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES 86f").size() == 1);
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES \"86f;87f\"").size() == 1);
    CHECK(FindFamilySuffixInArch("CUDA_ARCHITECTURES \"86;87\"").empty());
}

// ── Task 12: the arch-'a' object libs exist only when a 12x arch is targeted ─
//
// `lsxcommon_tc_cu` compiles src/lsx_tc_gemm.cu with `--generate-code=
// arch=compute_120a`, because the `kind::mxf4nvf4` PTX does not assemble on a
// plain 120/121 target — and certainly not on sm_86. So an Ampere-only build must
// not create it at all: on CUDA 13.5 it merely wastes a compile (nvcc still knows
// compute_120a), and on a pre-12.0 toolkit — JetPack 6.2.1 on the Orin, which is
// half of what this branch is for — it fails the build outright.
//
// Two questions, and the first one is the one that can rot silently:

struct Has12xProbe {
    int rc = -1;
    bool has12x = false;
    std::string tc_lib;
    std::string out;
};

Has12xProbe ProbeHas12x(const std::string &arches)
{
    Has12xProbe r;
    const fs::path script =
        fs::path(LSX_SOURCE_DIR) / "tests" / "cmake" / "probe_arch.cmake";
    const fs::path module = fs::path(LSX_SOURCE_DIR) / "cmake" / "LsxArch.cmake";
    const std::string cmd = std::string("\"") + LSX_CMAKE_COMMAND +
                            "\" -D PROBE_HAS12X=\"" + arches + "\" -D LsxArch=\"" +
                            module.string() + "\" -P \"" + script.string() +
                            "\" 2>&1";
#if defined(_WIN32)
    FILE *pipe = _popen(cmd.c_str(), "r");
#else
    FILE *pipe = popen(cmd.c_str(), "r");
#endif
    if (pipe == nullptr) return r;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) r.out += buf;
#if defined(_WIN32)
    r.rc = _pclose(pipe);
#else
    r.rc = (pclose(pipe) >> 8) & 0xff;
#endif
    r.has12x = r.out.find("HAS12X=TRUE") != std::string::npos;
    const auto at = r.out.find("TC_LIB=");
    if (at != std::string::npos) {
        auto end = r.out.find('\n', at);
        r.tc_lib = r.out.substr(at + 7, end == std::string::npos ? std::string::npos
                                                                : end - at - 7);
        while (!r.tc_lib.empty() &&
               (r.tc_lib.back() == '\r' || r.tc_lib.back() == ' ' ||
                r.tc_lib.back() == ';'))
            r.tc_lib.pop_back();
    }
    return r;
}

TEST_CASE("infra guard: an Ampere-only arch list targets no 12x arch",
          "[guard][infra][arch]")
{
    // The predicate the object-lib branches read. FALSE for Ampere-only, TRUE
    // the moment any 12x entry is present — including the family spelling.
    for (const char *arch : {"86", "87", "86;87", "87;86"}) {
        INFO("arch=" << arch);
        const auto r = ProbeHas12x(arch);
        INFO(r.out);
        REQUIRE(r.rc == 0);
        CHECK_FALSE(r.has12x);
        // ... and it names the STUB, not the arch-'a' object lib.
        CHECK(r.tc_lib == "lsxcommon_tc_stub");
    }
    for (const char *arch : {"120", "121", "120;121", "86;87;120", "120f",
                             "121a", "86;120f"}) {
        INFO("arch=" << arch);
        const auto r = ProbeHas12x(arch);
        INFO(r.out);
        REQUIRE(r.rc == 0);
        CHECK(r.has12x);
        CHECK(r.tc_lib == "lsxcommon_tc_cu");
    }
}

TEST_CASE("infra guard: the arch-'a' object libs are inside the 12x branch",
          "[guard][infra][arch]")
{
    // The structural half. The predicate above can only be trusted if the
    // targets that need it are actually guarded by it, so this walks the CMake
    // text and requires every 12x-only gencode literal to sit inside an
    // `if(lsx_arch_has_12x())` block. A scanner IS the right tool here: "is this
    // statement guarded" is a property of the text, unlike "does this regex
    // accept 86", which is a property of an evaluation.
    const auto unguarded = [](const std::string &text) {
        std::vector<int> bad;
        int depth = 0;
        int guard_depth = -1;   // >0 while inside an if(lsx_arch_has_12x()) block
        int line_no = 0;
        for (const auto &raw : Lines(text)) {
            ++line_no;
            const std::string line = StripHashComments(raw);
            // Two accepted forms of the same guard: the predicate called
            // directly, or the configure-time variable the root file resolves
            // from it. Both are real guards; anything else is not.
            const bool opens_guard =
                line.find("if(lsx_arch_has_12x(") != std::string::npos ||
                line.find("if (lsx_arch_has_12x(") != std::string::npos ||
                line.find("if(LSX_ARCH_HAS_12X)") != std::string::npos ||
                line.find("if (LSX_ARCH_HAS_12X)") != std::string::npos;
            if (opens_guard) guard_depth = depth;
            if (guard_depth >= 0 && line.find("endif(") != std::string::npos &&
                depth == guard_depth) {
                guard_depth = -1;
            }
            if (line.find("if(") != std::string::npos ||
                line.find("foreach(") != std::string::npos ||
                line.find("function(") != std::string::npos)
                ++depth;
            if (line.find("endif(") != std::string::npos ||
                line.find("endforeach(") != std::string::npos ||
                line.find("endfunction(") != std::string::npos)
                --depth;
            // A 12x-only codegen spelling outside the guard is the defect.
            if (line.find("compute_120a") != std::string::npos ||
                line.find("compute_121a") != std::string::npos) {
                if (guard_depth < 0) bad.push_back(line_no);
            }
        }
        return bad;
    };

    // Self-check the scanner first: a positive and a negative case, so a broken
    // walker cannot report "clean" for everything.
    CHECK(unguarded("--generate-code=arch=compute_120a,code=sm_120a").size() == 1);
    CHECK(unguarded(
              "if(lsx_arch_has_12x())\n"
              "  --generate-code=arch=compute_120a,code=sm_120a\n"
              "endif()\n")
              .empty());

    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> hits;
    for (const auto &p : CmakeFiles(root))
        for (int line : unguarded(ReadFile(p)))
            hits.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("12x gencode outside if(lsx_arch_has_12x()) ("
         << hits.size() << "):\n" << JoinHits(hits));
    REQUIRE(hits.empty());
}
