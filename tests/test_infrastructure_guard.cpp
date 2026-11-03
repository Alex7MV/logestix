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
