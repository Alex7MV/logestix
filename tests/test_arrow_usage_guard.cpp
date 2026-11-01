// Arrow usage guard (host-only static scan; zero deps beyond Catch2).
//
// The audit's machine-checkable rules (docs/ARROW_AUDIT_2026-11-01.md):
//
//   R-A1  Reading goes through MemoryMappedFile; ReadableFile is forbidden.
//   R-A2  Every owning *Schema::UnpackBatch call site has a recorded verdict
//         (transient copy vs retained legacy arena), and every mmap open site
//         is sanctioned.
//   R-A6  Artifacts are written through schema.cpp's atomic
//         WriteRecordBatchToFile; no converter drives MakeFileWriter itself.
//   R-A8  Every adapted model has a row in the audit matrix, and the converter
//         and loader TUs that row names actually exist.
//
// Adding a model, a loader or a new Arrow read site without updating the
// matrix/allowlist is a test failure, not a silent hole.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

bool ScannedExtension(const fs::path &p)
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

std::vector<fs::path> SourceTree()
{
    std::vector<fs::path> out;
    for (const char *root : {"libs", "apps"}) {
        const fs::path dir = fs::path(LSX_SOURCE_DIR) / root;
        if (!fs::exists(dir)) continue;
        for (const auto &e : fs::recursive_directory_iterator(dir)) {
            if (e.is_regular_file() && ScannedExtension(e.path()))
                out.push_back(e.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string RelativeToSource(const fs::path &p)
{
    return fs::relative(p, fs::path(LSX_SOURCE_DIR)).generic_string();
}

std::size_t CountOccurrences(const std::string &haystack,
                             const std::string &needle)
{
    std::size_t n = 0;
    for (std::size_t pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + needle.size()))
        ++n;
    return n;
}

// R-A1 allowlist: the only files permitted to open a memory map. Every one of
// them keeps the mapping alive for the pointers it publishes (schema.cpp's
// view keepalive, ExpertStore's mmaps/batches, tokenizer/shared-experts' db
// owner) or unpacks into transient owning records (the gemma copiers).
const std::set<std::string> kMmapOpenAllowed = {
    "libs/lsxcommon/src/schema.cpp",
    "libs/lsxcommon/include/lsxcommon/expert_store.h",
    "libs/lsxcommon/src/tokenizer.cpp",
    "libs/lsxcommon/src/tokenizer_arrows.cpp",
    "libs/lsxcommon/src/model_integrity.cpp",
    "libs/lsxcommon/src/shared_experts.cu",
    "libs/lsxcommon/src/gemma_moe_load.cpp",
    "libs/lsxcommon/src/gemma_dense_load.cpp",
    "libs/lsxcommon/src/deepseek_moe_load.cu",
    "libs/lsxcommon/src/deepseek_moe_pro_load.cu",
};

// R-A2 verdict per file that calls an owning *Schema::UnpackBatch.
//   transient        — unpacked, uploaded and freed in the same scope.
//   retained-legacy  — copied into a pinned/slab arena that lives for the
//                      session (documented debt: docs/OPEN_TASKS.md).
const std::map<std::string, std::string> kUnpackVerdict = {
    {"alice_moe_load.cpp", "retained-legacy"},
    {"deepseek_moe_repack.cpp", "transient"},
    {"deepseek_moe_v41_load.cu", "transient"},
    {"flint_moe_load.cpp", "retained-legacy"},
    {"gemma_dense_load.cpp", "transient"},
    {"gemma_moe_load.cpp", "transient"},
    {"glm_moe_flash_load.cpp", "retained-legacy"},
    {"glm_moe_load.cpp", "transient"},
    {"hy_moe_load.cpp", "retained-legacy"},
    {"ling_moe_load.cpp", "retained-legacy"},
    {"mistral_moe_load.cpp", "retained-legacy"},
    {"nemotron_moe_nvfp4_convert.cpp", "transient"},
    {"qwen_moe_exp_load.cpp", "retained-legacy"},
    {"xing_moe_load.cpp", "retained-legacy"},
};

// R-A8: every adapted model checked by the audit, and the scan roots below
// resolve the converter/loader files the report row names.
const std::vector<std::string> kModels = {
    "AliceAI-Foundation-80B-A3B-Base",
    "Beam-1.0-501B-A23B",
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro-0813",
    "DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash",
    "GigaChat3.5-432B-A28B-Reasoning",
    "Hy-Memory-30B-A3B",
    "Hy-SimulMT-30B-A3B",
    "Kimi-K2.7-Code",
    "Ling-3.1-flash",
    "MiniMax-M3",
    "MiniMax-M3.1-Flash",
    "Mistral-Small-4-119B-2603-NVFP4",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "Ornith-1.5-35B-A3B",
    "Qwen3.5-35B-A3B",
    "Qwen3.5-9B",
    "Qwen3.6-35B-A3B",
    "Qwen3.8-27B",
    "Qwen3.8-Flash-Next",
    "Step-5-Preview",
    "Xing4.0-29B-A4B",
    "flint",
    "gemma-4-26B-A4B-it",
    "gemma-4-31B-it",
    "granite-4.2-30b",
    "granite-switch-4.1-30b",
    "phi-4",
};

std::string Trim(std::string s)
{
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::vector<std::string> SplitColumns(const std::string &line)
{
    std::vector<std::string> cols;
    std::size_t start = 0;
    while (start <= line.size()) {
        const std::size_t bar = line.find('|', start);
        const std::size_t end = bar == std::string::npos ? line.size() : bar;
        cols.push_back(Trim(line.substr(start, end - start)));
        if (bar == std::string::npos) break;
        start = bar + 1;
    }
    if (!cols.empty() && cols.front().empty()) cols.erase(cols.begin());
    if (!cols.empty() && cols.back().empty()) cols.pop_back();
    return cols;
}

struct MatrixRow {
    std::string model;
    std::string converter;
    std::string loader;
    std::string verdict;
};

std::vector<MatrixRow> ReadMatrix()
{
    const fs::path report = fs::path(LSX_SOURCE_DIR) /
                            "docs" / "ARROW_AUDIT_2026-11-01.md";
    REQUIRE(fs::exists(report));
    const std::string text = ReadFile(report);
    const std::string start_marker = "<!-- arrow-matrix:start -->";
    const std::string end_marker = "<!-- arrow-matrix:end -->";
    const auto start = text.find(start_marker);
    const auto end = text.find(end_marker);
    REQUIRE(start != std::string::npos);
    REQUIRE(end != std::string::npos);
    REQUIRE(start < end);

    std::vector<MatrixRow> rows;
    std::istringstream in(text.substr(start + start_marker.size(),
                                      end - start - start_marker.size()));
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '|') continue;
        const auto cols = SplitColumns(line);
        if (cols.size() != 6) continue;
        if (cols[0] == "model" || cols[0].find("---") != std::string::npos)
            continue;
        rows.push_back({cols[0], cols[1], cols[2], cols[5]});
    }
    return rows;
}

bool FileExistsUnderSource(const std::string &name)
{
    for (const auto &e : SourceTree())
        if (e.filename().string() == name) return true;
    return false;
}

std::vector<std::string> SplitFileList(const std::string &field)
{
    std::vector<std::string> out;
    std::istringstream in(field);
    std::string item;
    while (std::getline(in, item, ',')) {
        const auto t = Trim(item);
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

const std::set<std::string> kReadPaths = {
    "views+ExpertStore", "views+PinChunked", "views+resident-slabs",
    "views+dense", "unpack+pinned-arena", "unpack+copy",
};

} // namespace

TEST_CASE("Arrow maps are mmap-backed: ReadableFile is forbidden",
          "[arrow][guard]")
{
    for (const auto &p : SourceTree()) {
        const std::string text = ReadFile(p);
        if (text.find("arrow::io::ReadableFile::Open") != std::string::npos)
            FAIL("ReadableFile::Open in " << RelativeToSource(p)
                 << " — use MemoryMappedFile (R-A1)");
    }
}

TEST_CASE("Every MemoryMappedFile::Open site is sanctioned", "[arrow][guard]")
{
    std::vector<std::string> unlisted;
    for (const auto &p : SourceTree()) {
        const std::string text = ReadFile(p);
        if (CountOccurrences(text, "MemoryMappedFile::Open") == 0) continue;
        const std::string rel = RelativeToSource(p);
        if (kMmapOpenAllowed.count(rel) == 0) unlisted.push_back(rel);
    }
    CHECK(unlisted.empty());
}

TEST_CASE("Every owning UnpackBatch call site has a recorded verdict",
          "[arrow][guard]")
{
    const std::regex call(
        R"((Expert|SharedExpert|DenseNvfp4|Pinned|TokenizerVocab|TokenizerMerges)Schema::UnpackBatch\()");
    std::map<std::string, int> seen;
    for (const auto &p : SourceTree()) {
        if (p.filename() == "schema.h" || p.filename() == "schema.cpp")
            continue;
        const std::string text = ReadFile(p);
        const auto n = std::distance(std::sregex_iterator(text.begin(), text.end(), call),
                                     std::sregex_iterator());
        if (n > 0) seen[p.filename().string()] += static_cast<int>(n);
    }
    for (const auto &[file, count] : seen) {
        INFO(file << " has " << count << " UnpackBatch call site(s)");
        CHECK(kUnpackVerdict.count(file) == 1);
    }
    for (const auto &[file, verdict] : kUnpackVerdict) {
        INFO(file);
        CHECK(seen.count(file) == 1);
        CHECK((verdict == "transient" || verdict == "retained-legacy"));
    }
}

TEST_CASE("Arrow IPC readers keep the zero-copy alignment default",
          "[arrow][guard]")
{
    // IpcReadOptions::ensure_alignment defaults to kAnyAlignment, which hands
    // back buffers that point into the mapping. Setting it (k8Byte/k32Byte/
    // k64Byte) makes Arrow copy every misaligned buffer into the memory pool,
    // silently destroying the zero-copy contract the loaders rely on.
    for (const auto &p : SourceTree()) {
        const std::string text = ReadFile(p);
        for (const char *forbidden : {"ensure_alignment", "Alignment::k8Byte",
                                      "Alignment::k32Byte",
                                      "Alignment::k64Byte"}) {
            if (text.find(forbidden) != std::string::npos)
                FAIL(RelativeToSource(p) << " sets " << forbidden
                     << " — that breaks zero-copy reads");
        }
    }
}

TEST_CASE("Arrow IPC writers are confined to schema.cpp", "[arrow][guard]")
{
    for (const auto &p : SourceTree()) {
        const std::string text = ReadFile(p);
        if (text.find("arrow::ipc::MakeFileWriter") == std::string::npos) continue;
        const std::string rel = RelativeToSource(p);
        INFO(rel);
        CHECK(rel == "libs/lsxcommon/src/schema.cpp");
    }
}

TEST_CASE("The audit matrix covers every model with resolvable TUs",
          "[arrow][guard]")
{
    const auto rows = ReadMatrix();
    std::map<std::string, int> by_model;
    for (const auto &row : rows) {
        ++by_model[row.model];
        INFO(row.model);
        CHECK(kReadPaths.count(row.verdict) == 1);
        for (const auto &f : SplitFileList(row.converter)) {
            INFO(f);
            CHECK(FileExistsUnderSource(f));
        }
        for (const auto &f : SplitFileList(row.loader)) {
            INFO(f);
            CHECK(FileExistsUnderSource(f));
        }
    }
    for (const auto &model : kModels) {
        INFO(model);
        CHECK(by_model[model] == 1);
    }
    CHECK(rows.size() == kModels.size());
}
