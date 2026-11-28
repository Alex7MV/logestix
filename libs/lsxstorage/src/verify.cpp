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

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
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

// The on-disk SCHEMA SHAPE. This is what decides a file's storage codec; the
// `dtype` column cannot, because a DenseNvfp4 file records the dtype of the
// tensor it was QUANTIZED FROM ("BF16"), not the form it is stored in.
enum class SchemaShape {
    kPinned,     // name, shape, dtype, data              -> codec from dtype
    kDenseNvfp4, // name, shape, dtype, packed, scales, global_scale -> NVFP4
    kExpert,     // w1_weight .. w3_global_scale + dtype  -> codec from dtype
    kExpertUndeclared, // the same nine columns with NO dtype: a defect (D1)
    kVocab,      // token_id, token_bytes, is_special     -> no codec
    kMerges,     // merge_left .. merge_token_id          -> no codec
    kUnknown,
};

const char *SchemaShapeName(SchemaShape s)
{
    switch (s) {
    case SchemaShape::kPinned: return "pinned";
    case SchemaShape::kDenseNvfp4: return "dense-nvfp4";
    case SchemaShape::kExpert: return "expert";
    case SchemaShape::kExpertUndeclared: return "expert-undeclared";
    case SchemaShape::kVocab: return "vocab";
    case SchemaShape::kMerges: return "merges";
    case SchemaShape::kUnknown: return "unknown";
    }
    return "unknown";
}

bool ColumnsAre(const std::shared_ptr<arrow::Schema> &schema,
                std::initializer_list<const char *> names)
{
    if (!schema || schema->num_fields() != static_cast<int>(names.size()))
        return false;
    int i = 0;
    for (const char *n : names) {
        if (schema->field(i)->name() != n) return false;
        ++i;
    }
    return true;
}

SchemaShape ClassifyShape(const std::shared_ptr<arrow::Schema> &schema)
{
    if (ColumnsAre(schema, {"name", "shape", "dtype", "packed", "scales",
                            "global_scale"}))
        return SchemaShape::kDenseNvfp4;
    if (ColumnsAre(schema, {"name", "shape", "dtype", "data"}))
        return SchemaShape::kPinned;
    // A routed/shared expert plane: the nine packed/scale columns PLUS the
    // `dtype` the record's codec is declared in. The nine-column form is NOT an
    // expert plane of this contract: it says what the bytes are shaped like and
    // nothing about how they are encoded, which is the ambiguity D1 retires.
    if (ColumnsAre(schema, {"w1_weight", "w2_weight", "w3_weight", "w1_scale",
                            "w2_scale", "w3_scale", "w1_global_scale",
                            "w2_global_scale", "w3_global_scale", "dtype"}))
        return SchemaShape::kExpert;
    if (ColumnsAre(schema, {"w1_weight", "w2_weight", "w3_weight", "w1_scale",
                            "w2_scale", "w3_scale", "w1_global_scale",
                            "w2_global_scale", "w3_global_scale"}))
        return SchemaShape::kExpertUndeclared;
    if (ColumnsAre(schema, {"token_id", "token_bytes", "is_special"}))
        return SchemaShape::kVocab;
    if (ColumnsAre(schema, {"merge_left", "merge_right", "merge_rank",
                            "merge_token_id"}))
        return SchemaShape::kMerges;
    return SchemaShape::kUnknown;
}

struct FileFacts {
    SchemaShape shape = SchemaShape::kUnknown;
    bool has_dtype_column = false;
    std::vector<std::string> names;
    std::vector<std::string> dtypes;
    int num_rows = 0;
};

// The codec an expert plane declares, from the `dtype` column of its records.
// Every row of one file is one plane of one encoding, and D5 makes the answer a
// union rather than a sum, so a file that (wrongly) mixes two dtypes still
// yields a well-defined mask — but the mixture is reported, because a single
// file describing two encodings is a converter defect, not a codec.
//
// Returns 0 and records a problem when no codec can be established.
uint64_t ExpertCodecFromDtypes(const std::string &rel, const FileFacts &facts,
                              VerifyReport *report)
{
    uint64_t mask = 0;
    for (const auto &dtype : facts.dtypes) {
        if (dtype == "U8") {
            report->problems.push_back(
                "expert plane " + rel + " declares dtype U8; rule D1 requires "
                "the codec to be named (NVFP4/F8_E4M3/I8/BF16/F32)");
            return 0;
        }
        const uint64_t bit = CodecForDtype(dtype);
        if (bit == 0) {
            report->problems.push_back(
                "expert plane " + rel + " declares an unknown dtype '" + dtype +
                "'; rule D1 requires NVFP4/F8_E4M3/I8/BF16/F32");
            return 0;
        }
        mask |= bit;
    }
    if (mask == 0) {
        report->problems.push_back(
            "expert plane " + rel + " has an empty dtype column: its codec "
            "cannot be established (rule D1)");
        return 0;
    }
    return mask;
}

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
    facts->shape = ClassifyShape(schema);
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

// ── Rule 5: the index owns geometry, artifact.fb owns codecs ──────────────
//
// The five fields below used to live in model.index.json and were read by six
// files to pick a kernel per layer. artifact.fb::layer_codec replaced them, so
// an index that still carries one is a SECOND source of truth for the same
// fact — and if the two disagree, the gate passes an artifact that executes a
// codec it never declared (design §0.5).
//
// The check parses the JSON and looks at KEYS, not text: the field names are
// ordinary prose as well ("the per-layer expert_nvfp4 map"), and a text scan
// would both miss a nested declaration and refuse a harmless note. An index the
// parser cannot read is a problem rather than a pass — a check that answers
// "fine" when it cannot read the file is exactly the hole it exists to close.
constexpr const char *kIndexCodecKeys[] = {"mlp_nvfp4", "attn_nvfp4",
                                           "head_nvfp4", "expert_nvfp4",
                                           "expert_fp8"};

void CheckIndexDeclaresNoCodec(const fs::path &artifact_dir,
                               VerifyReport *report)
{
    const fs::path index = artifact_dir / "model.index.json";
    std::error_code ec;
    if (!fs::is_regular_file(index, ec)) return;   // nothing to contradict

    // mmap, like every other JSON/Arrow read on the host path (R-A1).
    auto file = arrow::io::MemoryMappedFile::Open(index.string(),
                                                 arrow::io::FileMode::READ);
    if (!file.ok()) {
        report->problems.push_back("cannot open model.index.json: " +
                                   file.status().ToString());
        return;
    }
    auto size = (*file)->GetSize();
    if (!size.ok() || *size <= 0) {
        report->problems.push_back("cannot size model.index.json: " +
                                   size.status().ToString());
        return;
    }
    // An index is geometry: kilobytes. A megabyte is already a defect, and
    // reading it would be unbounded work inside a verification pass.
    constexpr int64_t kMaxIndexBytes = int64_t{4} << 20;
    if (*size > kMaxIndexBytes) {
        report->problems.push_back(
            "model.index.json is " + std::to_string(*size) +
            " bytes, which is not an index (limit 4 MiB)");
        return;
    }
    auto buf = arrow::AllocateBuffer(*size);
    if (!buf.ok()) {
        report->problems.push_back("cannot allocate for model.index.json: " +
                                   buf.status().ToString());
        return;
    }
    if (!(*file)->Read(*size, (*buf)->mutable_data()).ok()) {
        report->problems.push_back("cannot read model.index.json");
        return;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const auto view = std::string_view(
        reinterpret_cast<const char *>((*buf)->data()),
        static_cast<size_t>(*size));
    if (parser.parse(view).get(doc) != simdjson::SUCCESS) {
        report->problems.push_back(
            "model.index.json is not readable JSON, so it cannot be shown to "
            "declare no codec (rule 5: codecs live in artifact.fb::layer_codec)");
        return;
    }

    for (const char *key : kIndexCodecKeys) {
        simdjson::dom::element value;
        if (doc[key].get(value) != simdjson::SUCCESS) continue;
        report->problems.push_back(
            std::string("model.index.json still declares codec '") + key +
            "': the per-layer codec map lives in artifact.fb::layer_codec "
            "(design §2.4.3 rule 5) — stop writing it in the converter");
    }
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
        // The SHAPE decides first. A DenseNvfp4 file carries a `dtype` column
        // holding the dtype it was quantized FROM ("BF16"), so trusting it would
        // report an NVFP4 plane as BF16 — which is exactly the bug this
        // classification exists to catch.
        switch (facts.shape) {
        case SchemaShape::kDenseNvfp4:
            report.actual_mask |= kNvFp4;
            continue;
        case SchemaShape::kExpert: {
            // The plane declares its own codec; there is nothing to infer.
            // A "U8" spelling is the retired NVFP4 nibble form and stays a
            // failure, exactly as it is on the pinned path (D1).
            const uint64_t bit = ExpertCodecFromDtypes(rel, facts, &report);
            if (bit == 0) continue;   // a problem was recorded
            report.actual_mask |= bit;
            continue;
        }
        case SchemaShape::kExpertUndeclared:
            report.problems.push_back(
                "plane " + rel + " is a nine-column expert file with no dtype "
                "column: its codec cannot be established. Rule D1 requires every "
                "expert record to declare NVFP4/F8_E4M3/I8/BF16 in a trailing "
                "`dtype` field (ExpertSchema::kIdxDtype)");
            continue;
        case SchemaShape::kVocab:
        case SchemaShape::kMerges:
            continue;  // payload tables: no codec of their own
        case SchemaShape::kUnknown:
            report.problems.push_back(
                "plane " + rel + " has an unrecognised Arrow schema (" +
                std::to_string(facts.dtypes.size()) + " tensors); the codec "
                "cannot be verified from its shape");
            continue;
        case SchemaShape::kPinned:
            break;  // fall through to the dtype column
        }
        if (!facts.has_dtype_column) {
            report.problems.push_back(
                "plane " + rel + " is a 4-column PinnedSchema with no dtype "
                "column: its codec cannot be established");
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

    // The chat template the artifact CLAIMS to ship. Same rule as a plane: the
    // recorded path must exist, because an artifact is self-sufficient or it is
    // not (design §1.5). Checked before the digest loop so the cheap "missing"
    // answer wins over the expensive hash.
    const auto template_digest_ok = [&] {
        if (read.info.chat_template.empty() || !read.info.has_chat_template_digest)
            return;
        const auto p = artifact_dir / read.info.chat_template;
        if (!fs::is_regular_file(p, ec)) {
            report.problems.push_back(
                "artifact.fb records chat template '" + read.info.chat_template +
                "' but the file is not in the artifact (invariant I5: an "
                "artifact must load with the source store unmounted)");
            return;
        }
        if (!deep) return;
        std::string herr;
        const auto d = Sha256::OfFile(p.string(), &herr);
        if (!herr.empty()) {
            report.problems.push_back(herr);
            return;
        }
        if (!std::equal(d.begin(), d.end(),
                        read.info.chat_template_sha256.begin()))
            report.problems.push_back(
                "sha256 mismatch for chat template " + read.info.chat_template);
    };
    template_digest_ok();

    // Rule 5: the index may not declare a codec (design §2.4.3).
    CheckIndexDeclaresNoCodec(artifact_dir, &report);

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

// ── Scan a freshly converted tree ──────────────────────────────────────────
//
// Same classification as the verifier, pointed at a tree with no artifact.fb
// yet. The point is that the emitter's plane list and the converter's intent
// are computed from DIFFERENT inputs — one from the bytes, one from the
// profile — so comparing them is a real check rather than a restatement.
ScanReport ScanArtifactTree(const fs::path &tree_root)
{
    ScanReport out;

    if (!fs::is_directory(tree_root)) {
        out.problems.push_back("not a directory: " + tree_root.string());
        return out;
    }

    std::set<std::string> sidecars = {"artifact.fb", "model.index.json",
                                      "chat_template.jinja", "tokenizer.json",
                                      "artifact.fb.tmp"};
    std::map<std::string, uint64_t> per_file_mask;  // rel path -> OR of codecs
    std::map<std::string, std::string> per_file_role;

    std::error_code ec;
    for (const auto &entry : fs::recursive_directory_iterator(tree_root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const auto rel = fs::relative(entry.path(), tree_root).generic_string();
        if (sidecars.count(rel) != 0) continue;

        const auto top = rel.substr(0, rel.find('/'));
        if (ContentDirs().count(top) == 0) {
            out.problems.push_back(
                "converted tree holds a file the load path cannot reach: " + rel +
                " (v1 ships no ballast: an unreachable plane is not written)");
            continue;
        }

        // Tokenizer arrows are planes in their own right (the loader mmaps
        // them), but they carry no codec of their own — they are uint8 payload
        // tables — so they are recorded under role "meta".
        const bool is_tokenizer = rel.find("tokenizer") != std::string::npos;
        if (is_tokenizer) {
            per_file_role[rel] = "meta";
            continue;
        }

        if (!EndsWith(rel, ".arrow")) {
            per_file_role[rel] = "meta";
            continue;
        }

        FileFacts facts;
        std::string ferr;
        if (!ReadFileFacts(entry.path(), &facts, &ferr)) {
            out.problems.push_back("cannot read " + rel + ": " + ferr);
            continue;
        }

        uint64_t mask = 0;
        // Shape first, dtype second: see the note in VerifyArtifact.
        if (facts.shape == SchemaShape::kDenseNvfp4) {
            per_file_role[rel] = "dense.nvfp4";
            per_file_mask[rel] = kNvFp4;
            continue;
        }
        if (facts.shape == SchemaShape::kExpert) {
            // Role follows the SHAPE (an expert plane is expert.routed in every
            // encoding); the codec follows the declared dtype.
            per_file_role[rel] = "expert.routed";
            VerifyReport scratch;   // problems are copied into `out` below
            const uint64_t bit = ExpertCodecFromDtypes(rel, facts, &scratch);
            for (const auto &problem : scratch.problems)
                out.problems.push_back(problem);
            if (bit == 0) continue;
            per_file_mask[rel] = bit;
            continue;
        }
        if (facts.shape == SchemaShape::kExpertUndeclared) {
            per_file_role[rel] = "expert.routed";
            out.problems.push_back(
                "converter wrote " + rel + " as a nine-column expert file with "
                "no dtype column; rule D1 requires every expert record to "
                "declare its codec (NVFP4/F8_E4M3/I8/BF16) in a trailing "
                "`dtype` field");
            continue;
        }
        if (facts.shape == SchemaShape::kVocab || facts.shape == SchemaShape::kMerges) {
            per_file_role[rel] = "meta";
            continue;
        }
        if (!facts.has_dtype_column) {
            out.problems.push_back(
                "cannot establish a codec for " + rel + " (unknown schema shape)");
            continue;
        }
        {
            bool any_weight = false;
            for (size_t i = 0; i < facts.dtypes.size(); ++i) {
                const auto &dtype = facts.dtypes[i];
                const std::string name =
                    i < facts.names.size() ? facts.names[i] : std::string{};
                if (IsCompanionName(name)) continue;
                if (dtype == "U8") {
                    out.problems.push_back(
                        "converter wrote " + rel + " with dtype U8 for '" + name +
                        "'; rule D1 requires the codec to be named "
                        "(NVFP4/F8_E4M3/I8/BF16/F32)");
                    continue;
                }
                const uint64_t bit = CodecForDtype(dtype);
                if (bit != 0 && !StartsWith(name, "indices")) {
                    mask |= bit;
                    any_weight = true;
                }
            }
            per_file_role[rel] = any_weight ? "dense" : "meta";
        }
        per_file_mask[rel] = mask;
    }

    // Build the plane list: one entry per file, skipping pure-metadata files
    // that hold no codec evidence at all (nothing to declare).
    for (const auto &[rel, mask] : per_file_mask) {
        PlaneInfo plane;
        plane.codec_bit = mask;
        plane.role = per_file_role[rel];
        plane.path = rel;
        std::error_code size_ec;
        plane.bytes = fs::file_size(tree_root / rel, size_ec);
        std::string herr;
        const auto d = Sha256::OfFile((tree_root / rel).string(), &herr);
        if (!herr.empty()) {
            out.problems.push_back(herr);
            continue;
        }
        plane.sha256 = d;
        plane.has_digest = true;
        out.mask |= mask;
        out.planes.push_back(std::move(plane));
    }
    for (const auto &[rel, role] : per_file_role) {
        if (role != "meta" || per_file_mask.count(rel) != 0) continue;
        // A tokenizer/sidecar arrow: declare it with no codec so the verifier
        // can see it is accounted for.
        PlaneInfo plane;
        plane.codec_bit = 0;
        plane.role = "meta";
        plane.path = rel;
        std::error_code size_ec;
        plane.bytes = fs::file_size(tree_root / rel, size_ec);
        std::string herr;
        const auto d = Sha256::OfFile((tree_root / rel).string(), &herr);
        if (!herr.empty()) {
            out.problems.push_back(herr);
            continue;
        }
        plane.sha256 = d;
        plane.has_digest = true;
        out.planes.push_back(std::move(plane));
    }

    // Recover the layer count AND the per-layer codec map from the file naming
    // the converters use (`layer_<n>.arrow`, `layer_<nnn>.arrow`,
    // `exp_<m>.arrow` inside `layer_<n>/`). One pass, because both answers come
    // from the same fact: which layer a classified file belongs to.
    //
    // The map is per LAYER, not "the tree mask repeated": a mixed tree has to
    // report each layer's own codec, and the readers of artifact.fb::layer_codec
    // ask exactly that question.
    std::vector<uint64_t> per_layer;
    const auto note_layer = [&per_layer](int idx, uint64_t mask) {
        if (idx < 0) return;
        if (per_layer.size() <= static_cast<size_t>(idx))
            per_layer.resize(static_cast<size_t>(idx) + 1, 0);
        per_layer[static_cast<size_t>(idx)] |= mask;
    };
    std::error_code name_ec;
    for (const auto &[rel, mask] : per_file_mask) {
        // Layer index from either component: the file stem (`layer_007.arrow`)
        // or the directory (`experts/layer_7/exp_0.arrow`).
        int layer_idx = -1;
        const auto file = fs::path(rel).filename().string();
        const auto us = file.rfind('_');
        if (us != std::string::npos) {
            std::string digits;
            for (size_t i = us + 1; i < file.size() &&
                 std::isdigit(static_cast<unsigned char>(file[i])); ++i)
                digits.push_back(file[i]);
            if (!digits.empty()) layer_idx = std::atoi(digits.c_str());
        }
        const auto parent = fs::path(rel).parent_path().filename().string();
        const auto pus = parent.rfind('_');
        if (pus != std::string::npos) {
            std::string pd;
            for (size_t i = pus + 1; i < parent.size() &&
                 std::isdigit(static_cast<unsigned char>(parent[i])); ++i)
                pd.push_back(parent[i]);
            // A directory index wins: `experts/layer_7/exp_3.arrow` must be
            // layer 7, not layer 3.
            if (!pd.empty()) layer_idx = std::atoi(pd.c_str());
        }
        if (layer_idx < 0) continue;
        out.layer_count = std::max(out.layer_count, layer_idx + 1);
        note_layer(layer_idx, mask);
        (void)name_ec;
    }
    per_layer.resize(static_cast<size_t>(out.layer_count), 0);
    out.layer_codec = std::move(per_layer);

    out.ok = out.problems.empty();
    return out;
}

} // namespace lsx::storage
