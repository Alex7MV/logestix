// Infrastructure guard: the centralizations of 2026-11-03 (dependency include
// paths, the shared tensor reader/writers, the CUDA arch variable) must not be
// undone one call site at a time.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
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
            // property is the silent-downgrade bug.
            if (arg == "OFF") { begin = m.suffix().first; continue; }
            static const std::regex fam(R"((^|[^0-9])(12[01][af]?f)([^0-9]|$))");
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
std::vector<int> FindNonFamilyGencodeLiterals(const std::string &text)
{
    static const std::regex lit(
        R"(--generate-code=arch=compute_(12[0-9])([af]?),code=)");
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
