// artifact verification (design §1.3 rules D1-D5, §2.4.3).
//
// The verifier walks the tree, decides each file's codec from its CONTENT (the
// Arrow `dtype` column or the schema shape) and from its name, unions those
// bits into an actual mask, and compares that with what artifact.fb declares.
//
// This is the one place a codec is inferred rather than declared, and it exists
// only to police the declaration — the runtime never does this. Because there
// are no reference artifacts left in the tree, the inference here is also the
// only independent check that a converter wrote what it said it wrote.
#include "lsxstorage/artifact.h"

#include "sha256.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace lsx::storage {

namespace {

// Bit values mirrored from lsxcommon::hw::PlaneCodec. Duplicated deliberately:
// lsxstorage sits below lsxcommon, and a drift between the two is caught by
// tests/test_storage_verify.cpp comparing against the enum.
constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kF16 = 1ull << 2;
constexpr uint64_t kI8Sym = 1ull << 3;
constexpr uint64_t kMxFp8 = 1ull << 4;
constexpr uint64_t kNvFp4 = 1ull << 5;
constexpr uint64_t kMxFp4 = 1ull << 6;

// Directories that hold planes the load path reaches.
const std::set<std::string> &ContentDirs()
{
    static const std::set<std::string> v = {"pinned",  "experts", "router",
                                            "shared",  "ple"};
    return v;
}

bool EndsWith(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool StartsWith(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}

// A companion tensor: it scales a plane and never establishes a codec alone.
bool IsCompanionName(const std::string &name)
{
    return EndsWith(name, ".scale") || EndsWith(name, ".scales") ||
           EndsWith(name, ".global_scale") || EndsWith(name, "_scale") ||
           EndsWith(name, "_scales");
}

// Arrow dtype spelling -> codec bit. 0 means "this dtype establishes no codec":
// companions, metadata and shapes that only make sense in context.
uint64_t CodecForDtype(const std::string &dtype)
{
    if (dtype == "BF16" || dtype == "bf16") return kBF16;
    if (dtype == "F32" || dtype == "float32") return kF32;
    if (dtype == "F16" || dtype == "float16") return kF16;
    if (dtype == "I8" || dtype == "int8") return kI8Sym;
    if (dtype == "F8_E4M3") return kMxFp8;
    if (dtype == "NVFP4") return kNvFp4;
    // F8_E8M0 is an MXFP8 exponent companion; I64/BOOL/U8 are metadata or the
    // old ambiguous NVFP4 spelling, handled by the caller with context.
    return 0;
}

struct FileFacts {
    bool has_dtype_column = false;
    std::vector<std::string> names;
    std::vector<std::string> dtypes;
    int num_rows = 0;
};

// Reads the small columns only. These files are Arrow IPC record batches whose
// tensor payloads live in the `data`/`packed`/`scales` columns; we never touch
// those, so verification does not pay for the weights.
bool ReadFileFacts(const fs::path &path, FileFacts *facts, std::string *error)
{
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap;
    auto opened = arrow::io::MemoryMappedFile::Open(path.string(),
                                                   arrow::io::FileMode::READ);
    if (!opened.ok()) {
        if (error != nullptr) *error = opened.status().ToString();
        return false;
    }
    mmap = *opened;
    auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap);
    if (!reader_res.ok()) {
        if (error != nullptr) *error = reader_res.status().ToString();
        return false;
    }
    const auto reader = *reader_res;
    if (reader->num_record_batches() != 1) {
        if (error != nullptr)
            *error = "expected 1 record batch, found " +
                     std::to_string(reader->num_record_batches());
        return false;
    }
    auto batch_res = reader->ReadRecordBatch(0);
    if (!batch_res.ok()) {
        if (error != nullptr) *error = batch_res.status().ToString();
        return false;
    }
    const auto batch = *batch_res;
    facts->num_rows = batch->num_rows();

    const auto schema = batch->schema();
    const int dtype_idx = schema->GetFieldIndex("dtype");
    const int name_idx = schema->GetFieldIndex("name");
    facts->has_dtype_column = dtype_idx >= 0;
    if (!facts->has_dtype_column) return true;

    const auto dtypes = std::static_pointer_cast<arrow::StringArray>(
        batch->column(dtype_idx));
    for (int64_t i = 0; i < dtypes->length(); ++i)
        facts->dtypes.push_back(dtypes->GetString(i));

    if (name_idx >= 0) {
        const auto names = std::static_pointer_cast<arrow::StringArray>(
            batch->column(name_idx));
        for (int64_t i = 0; i < names->length(); ++i)
            facts->names.push_back(names->GetString(i));
    }
    return true;
}

} // namespace

VerifyReport VerifyArtifact(const fs::path &artifact_dir, bool deep)
{
    VerifyReport report;

    const auto read = ReadArtifact(artifact_dir);
    if (read.status != ReadStatus::kOk) {
        report.problems.push_back(std::string("artifact.fb: ") +
                                  ReadStatusName(read.status) + ": " + read.detail);
        return report;
    }
    report.declared_mask = read.info.used_codecs;

    std::map<std::string, const PlaneInfo *> declared;
    for (const auto &p : read.info.planes) {
        if (p.path.empty()) continue;  // a meta plane inside model.index.json
        declared[p.path] = &p;
    }

    // ── Walk the tree: every file must be either a declared plane or a
    // sanctioned sidecar. A plane the load path cannot reach is a defect, which
    // is why v1 ships no ballast.
    std::set<std::string> sidecars = {"artifact.fb", "model.index.json",
                                      "chat_template.jinja", "tokenizer.json"};
    std::error_code ec;
    for (const auto &entry : fs::recursive_directory_iterator(artifact_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto rel = fs::relative(entry.path(), artifact_dir).generic_string();
        if (sidecars.count(rel) != 0) continue;

        const auto top = rel.substr(0, rel.find('/'));
        const bool known_dir = ContentDirs().count(top) != 0;
        const bool declared_here = declared.count(rel) != 0;

        if (!known_dir && !declared_here) {
            report.problems.push_back("file is not reachable from the load path "
                                      "and not a declared plane: " + rel);
            continue;
        }
        if (!declared_here) {
            report.problems.push_back("file exists but no plane declares it: " + rel);
            continue;
        }

        // D1-D5 on this file's tensors.
        FileFacts facts;
        std::string ferr;
        if (!EndsWith(rel, ".arrow")) {
            // Non-Arrow payloads (a raw template, say) establish no codec.
            continue;
        }
        if (!ReadFileFacts(entry.path(), &facts, &ferr)) {
            report.problems.push_back("cannot read " + rel + ": " + ferr);
            continue;
        }
        if (!facts.has_dtype_column) {
            // No dtype column: the codec is implied by the schema shape, which
            // the declared plane already states. Nothing to infer.
            report.actual_mask |= declared[rel]->codec_bit;
            continue;
        }
        for (size_t i = 0; i < facts.dtypes.size(); ++i) {
            const auto &dtype = facts.dtypes[i];
            const std::string name =
                i < facts.names.size() ? facts.names[i] : std::string{};
            if (dtype == "U8") {
                // D1: U8 was the old spelling of NVFP4 nibbles, meaning
                // "NVFP4 only if a .scales sibling exists". The new format
                // names the codec, so a U8 WEIGHT here is a converter that was
                // not migrated. Metadata `U8` under role "meta" is fine.
                if (declared[rel]->role == "meta") continue;
                report.problems.push_back(
                    "weight plane " + rel + " spells its codec as U8"
                    " (dtype \"" + dtype + "\"" + (name.empty() ? "" :
                     " for tensor '" + name + "'") +
                    "); rule D1 requires NVFP4/F8_E4M3/I8/BF16/F32");
                continue;
            }
            if (declared[rel]->role == "meta") {
                // D3/D4: metadata may not carry weight-looking dtypes.
                const uint64_t as_weight = CodecForDtype(dtype);
                if (as_weight != 0 && !IsCompanionName(name)) {
                    report.problems.push_back(
                        "plane " + rel + " is declared role=meta but holds a "
                        "weight-like tensor '" + name + "' (dtype " + dtype + ")");
                }
                continue;
            }
            // D2: companions add no bit of their own.
            if (IsCompanionName(name)) continue;
            // D5: union, never sum.
            report.actual_mask |= CodecForDtype(dtype);
        }
    }

    // Every declared plane must exist.
    for (const auto &[rel, plane] : declared) {
        if (!fs::is_regular_file(artifact_dir / rel, ec))
            report.problems.push_back("declared plane is missing on disk: " + rel);
        (void)plane;
    }

    // The mask must equal the content it describes.
    if (report.actual_mask != report.declared_mask) {
        std::ostringstream os;
        os << "mask mismatch: artifact.fb declares " << report.declared_mask
           << " but the tree contains " << report.actual_mask;
        report.problems.push_back(os.str());
    }

    // Per-layer map shape.
    if (read.info.layer_count > 0 &&
        static_cast<int>(read.info.layer_codec.size()) != read.info.layer_count) {
        report.problems.push_back(
            "layer_codec has " + std::to_string(read.info.layer_codec.size()) +
            " entries, layer_count is " + std::to_string(read.info.layer_count));
    }

    // Digests.
    if (deep) {
        for (const auto &p : read.info.planes) {
            if (p.path.empty() || !p.has_digest) continue;
            std::string herr;
            const auto d = Sha256::OfFile((artifact_dir / p.path).string(), &herr);
            if (!herr.empty()) {
                report.problems.push_back(herr);
                continue;
            }
            if (!std::equal(d.begin(), d.end(), p.sha256.begin()))
                report.problems.push_back("sha256 mismatch for plane " + p.path);
        }
    }

    report.ok = report.problems.empty();
    return report;
}

} // namespace lsx::storage
