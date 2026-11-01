// Arrow IPC tree integrity — validates the CONVERTED artifacts of every
// adapted model (docs/ARROW_AUDIT_2026-11-01.md, R-A3/R-A4/R-A5/R-A8).
//
// Two phases per tree:
//   * footer-only: open every .arrow, require exactly one record batch, a
//     known schema (by field names) and a non-empty schema — cheap even for a
//     500k-file vault;
//   * deep sample: read batch 0 for a deterministic subset (every directory
//     contributes its first and last file; global/tokenizer files always) and
//     validate list offsets, buffer ranges and NVFP4/raw size invariants.
//     Bounded by LSX_ARROW_AUDIT_MAX_BYTES (default 256 MiB per tree);
//     LSX_ARROW_AUDIT_FULL=1 removes the budget.
//
// The vault is optional: absent root / no converted tree -> CTest SKIP (4).
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "lsxcommon/arrow_validate.h"
#include "lsxcommon/schema.h"

namespace fs = std::filesystem;
namespace av = lsxcommon::arrow_check;

namespace {

enum class Kind {
    kExpert,
    kSharedExpert,
    kDenseNvfp4,
    kPinned,
    kVocab,
    kMerges,
    kUnknown,
};

std::string KindName(Kind k)
{
    switch (k) {
    case Kind::kExpert: return "expert";
    case Kind::kSharedExpert: return "shared-expert";
    case Kind::kDenseNvfp4: return "dense-nvfp4";
    case Kind::kPinned: return "pinned";
    case Kind::kVocab: return "tokenizer-vocab";
    case Kind::kMerges: return "tokenizer-merges";
    default: return "unknown";
    }
}

bool FieldsAre(const std::shared_ptr<arrow::Schema> &schema,
               std::initializer_list<const char *> names)
{
    if (!schema || schema->num_fields() !=
                       static_cast<int>(names.size()))
        return false;
    int i = 0;
    for (const char *name : names) {
        if (schema->field(i)->name() != name) return false;
        ++i;
    }
    return true;
}

Kind Classify(const std::shared_ptr<arrow::Schema> &schema)
{
    if (FieldsAre(schema, {"w1_weight", "w2_weight", "w3_weight",
                           "w1_scale", "w2_scale", "w3_scale",
                           "w1_global_scale", "w2_global_scale",
                           "w3_global_scale"}))
        return Kind::kExpert;
    if (FieldsAre(schema, {"name", "shape", "dtype", "packed", "scales",
                           "global_scale"}))
        return Kind::kDenseNvfp4;
    if (FieldsAre(schema, {"name", "shape", "dtype", "data"}))
        return Kind::kPinned;
    if (FieldsAre(schema, {"token_id", "token_bytes", "is_special"}))
        return Kind::kVocab;
    if (FieldsAre(schema, {"merge_left", "merge_right", "merge_rank",
                           "merge_token_id"}))
        return Kind::kMerges;
    return Kind::kUnknown;
}

struct TreeStats {
    int64_t files = 0;
    int64_t footer_ok = 0;
    int64_t deep = 0;
    int64_t deep_bytes = 0;
    int64_t budget_skips = 0;
    std::map<Kind, int64_t> kinds;
    std::map<int64_t, int64_t> block_hist;   // implied elems-per-scale
    std::vector<std::string> problems;
};

void Problem(TreeStats &st, const std::string &msg)
{
    st.problems.push_back(msg);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ReadFirstBatch(const fs::path &path,
               std::shared_ptr<arrow::io::MemoryMappedFile> *out_mmap,
               std::shared_ptr<arrow::ipc::RecordBatchFileReader> *out_reader)
{
    auto mmap = arrow::io::MemoryMappedFile::Open(path.string(),
                                                  arrow::io::FileMode::READ);
    if (!mmap.ok()) return mmap.status();
    auto reader = arrow::ipc::RecordBatchFileReader::Open(mmap->get());
    if (!reader.ok()) return reader.status();
    if ((*reader)->num_record_batches() != 1)
        return arrow::Status::Invalid("expected exactly one record batch, got ",
                                      (*reader)->num_record_batches());
    auto batch = (*reader)->ReadRecordBatch(0);
    if (!batch.ok()) return batch.status();
    if (out_mmap) *out_mmap = *mmap;
    if (out_reader) *out_reader = *reader;
    return *batch;
}

arrow::Status ValidateBinaryColumn(const arrow::RecordBatch &batch, int col,
                                   const char *what)
{
    auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        batch.column(col));
    if (!bin) return arrow::Status::TypeError(what, " is not large_binary");
    for (int64_t i = 0; i < batch.num_rows(); ++i) {
        auto v = bin->GetView(i);
        ARROW_RETURN_NOT_OK(av::CheckViewRange(
            v.data(), static_cast<int64_t>(v.size()), bin->value_data(), what,
            std::to_string(i)));
    }
    return arrow::Status::OK();
}

arrow::Status ValidateScaleColumn(const arrow::RecordBatch &batch, int col,
                                  const char *what)
{
    auto lst = std::dynamic_pointer_cast<arrow::ListArray>(batch.column(col));
    if (!lst) return arrow::Status::TypeError(what, " is not a list");
    auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
    auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
    if (!u8 || !offs)
        return arrow::Status::TypeError(what, " is not list<uint8>");
    for (int64_t i = 0; i < batch.num_rows(); ++i)
        ARROW_RETURN_NOT_OK(
            av::CheckListSlice(*offs, i, u8->length(), what, std::to_string(i)));
    return arrow::Status::OK();
}

arrow::Status ValidateKind(Kind kind, const arrow::RecordBatch &batch,
                           std::map<int64_t, int64_t> *block_hist)
{
    if (batch.num_rows() <= 0)
        return arrow::Status::Invalid("batch has no rows");

    switch (kind) {
    case Kind::kExpert:
    case Kind::kSharedExpert: {
        for (int col : {0, 1, 2})
            ARROW_RETURN_NOT_OK(
                ValidateBinaryColumn(batch, col, "expert weight"));
        for (int col : {3, 4, 5})
            ARROW_RETURN_NOT_OK(
                ValidateScaleColumn(batch, col, "expert scale"));
        // Exact converted-tree contract: one E4M3 scale per 16 elements.
        {
            auto bin = [&](int col) {
                return std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
                    batch.column(col));
            };
            auto scales = [&](int col) {
                return std::dynamic_pointer_cast<arrow::ListArray>(
                    batch.column(col));
            };
            for (int w = 0; w < 3; ++w) {
                auto b = bin(w);
                auto l = scales(w + 3);
                if (!b || !l)
                    return arrow::Status::TypeError(
                        "expert payload/scale column type mismatch");
                auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(
                    l->offsets());
                if (!offs)
                    return arrow::Status::TypeError(
                        "expert scale offsets not int32");
                for (int64_t i = 0; i < batch.num_rows(); ++i)
                    ARROW_RETURN_NOT_OK(av::CheckExpertScaleCount(
                        static_cast<int64_t>(b->GetView(i).size()),
                        offs->Value(i + 1) - offs->Value(i), 16,
                        std::string("expert[w") + std::to_string(w + 1) +
                            "][" + std::to_string(i) + "]"));
            }
        }
        auto gs = batch.column(6);
        for (int col : {6, 7, 8}) {
            auto f = std::dynamic_pointer_cast<arrow::FloatArray>(
                batch.column(col));
            if (!f)
                return arrow::Status::TypeError(
                    "expert global_scale is not float32");
            for (int64_t i = 0; i < batch.num_rows(); ++i)
                if (!std::isfinite(f->Value(i)) || f->Value(i) <= 0.0f)
                    return arrow::Status::Invalid(
                        "expert has an invalid global_scale ", f->Value(i));
        }
        if (block_hist) {
            for (int64_t i = 0; i < batch.num_rows(); ++i) {
                auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
                    batch.column(0));
                auto lst = std::dynamic_pointer_cast<arrow::ListArray>(
                    batch.column(3));
                if (!bin || !lst) continue;
                const int64_t packed = static_cast<int64_t>(bin->GetView(i).size());
                auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
                auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
                if (!u8 || !offs) continue;
                const int64_t scales = offs->Value(i + 1) - offs->Value(i);
                if (packed > 0 && scales > 0)
                    (*block_hist)[(packed * 2 + scales - 1) / scales] += 1;
            }
        }
        (void)gs;
        return arrow::Status::OK();
    }
    case Kind::kDenseNvfp4: {
        auto names = std::dynamic_pointer_cast<arrow::StringArray>(
            batch.column(0));
        auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(
            batch.column(1));
        auto packeds = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
            batch.column(3));
        auto scales = std::dynamic_pointer_cast<arrow::ListArray>(
            batch.column(4));
        auto gs = std::dynamic_pointer_cast<arrow::FloatArray>(batch.column(5));
        if (!names || !shapes || !packeds || !scales || !gs)
            return arrow::Status::TypeError("dense-nvfp4 column type mismatch");
        auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(
            shapes->values());
        auto shape_offsets = std::dynamic_pointer_cast<arrow::Int32Array>(
            shapes->offsets());
        auto flat_scales = std::dynamic_pointer_cast<arrow::UInt8Array>(
            scales->values());
        auto scale_offsets = std::dynamic_pointer_cast<arrow::Int32Array>(
            scales->offsets());
        if (!flat_shapes || !shape_offsets || !flat_scales || !scale_offsets)
            return arrow::Status::TypeError(
                "dense-nvfp4 list value/offset type mismatch");
        for (int64_t i = 0; i < batch.num_rows(); ++i) {
            const std::string name = names->GetString(i);
            ARROW_RETURN_NOT_OK(av::CheckListSlice(
                *shape_offsets, i, flat_shapes->length(), "shape", name));
            ARROW_RETURN_NOT_OK(av::CheckListSlice(
                *scale_offsets, i, flat_scales->length(), "scales", name));
            auto pv = packeds->GetView(i);
            ARROW_RETURN_NOT_OK(av::CheckViewRange(
                pv.data(), static_cast<int64_t>(pv.size()),
                packeds->value_data(), "packed payload", name));
            std::vector<int64_t> shape;
            for (int64_t j = shape_offsets->Value(i);
                 j < shape_offsets->Value(i + 1); ++j)
                shape.push_back(flat_shapes->Value(j));
            const int64_t rank = static_cast<int64_t>(shape.size());
            const int64_t scale_bytes = scale_offsets->Value(i + 1) -
                                        scale_offsets->Value(i);
            // The payload is always NVFP4; `dtype` is the source dtype.
            ARROW_RETURN_NOT_OK(av::CheckNvfp4Size(
                shape.data(), rank, static_cast<int64_t>(pv.size()),
                scale_bytes, gs->Value(i), name));
        }
        return arrow::Status::OK();
    }
    case Kind::kPinned: {
        auto names = std::dynamic_pointer_cast<arrow::StringArray>(
            batch.column(0));
        auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(
            batch.column(2));
        auto datas = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
            batch.column(3));
        auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(
            batch.column(1));
        if (!names || !dtypes || !datas || !shapes)
            return arrow::Status::TypeError("pinned column type mismatch");
        auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(
            shapes->values());
        auto offsets = std::dynamic_pointer_cast<arrow::Int32Array>(
            shapes->offsets());
        if (!flat_shapes || !offsets)
            return arrow::Status::TypeError(
                "pinned shape list value/offset type mismatch");
        for (int64_t i = 0; i < batch.num_rows(); ++i) {
            const std::string name = names->GetString(i);
            ARROW_RETURN_NOT_OK(av::CheckListSlice(
                *offsets, i, flat_shapes->length(), "shape", name));
            auto v = datas->GetView(i);
            ARROW_RETURN_NOT_OK(av::CheckViewRange(
                v.data(), static_cast<int64_t>(v.size()), datas->value_data(),
                "tensor data", name));
            std::vector<int64_t> shape;
            for (int64_t j = offsets->Value(i); j < offsets->Value(i + 1); ++j)
                shape.push_back(flat_shapes->Value(j));
            ARROW_RETURN_NOT_OK(av::CheckPinnedTensor(
                name, dtypes->GetString(i), shape.data(),
                static_cast<int64_t>(shape.size()),
                static_cast<int64_t>(v.size())));
        }
        return arrow::Status::OK();
    }
    case Kind::kVocab:
    case Kind::kMerges:
        return arrow::Status::OK();
    default:
        return arrow::Status::Invalid("unknown Arrow schema");
    }
}

bool IsDeepByDefault(const fs::path &rel)
{
    const std::string s = rel.generic_string();
    if (s.find("tokenizer_") != std::string::npos) return true;
    if (s.find("global.arrow") != std::string::npos) return true;
    // Experts are the bulk: the deterministic sample below takes one file per
    // layer directory, so skip them in the default set.
    if (s.find("experts/") != std::string::npos) return false;
    return true;
}

void ValidateTree(const fs::path &root, TreeStats &st, bool full,
                  int64_t budget_bytes)
{
    std::vector<fs::path> files;
    for (const auto &e : fs::recursive_directory_iterator(root)) {
        if (e.is_regular_file() && e.path().extension() == ".arrow")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        Problem(st, "no .arrow files");
        return;
    }

    // Deterministic deep set: default deep files, plus the first and last
    // expert file of every experts/layer_* directory.
    std::set<std::string> deep;
    for (const auto &p : files)
        if (IsDeepByDefault(fs::relative(p, root))) deep.insert(p.string());
    std::map<std::string, std::pair<fs::path, fs::path>> expert_edges;
    for (const auto &p : files) {
        const std::string rel = fs::relative(p, root).generic_string();
        if (rel.find("experts/") == std::string::npos) continue;
        const std::string dir = fs::path(rel).parent_path().generic_string();
        auto &edge = expert_edges[dir];
        if (edge.first.empty()) edge.first = p;
        edge.second = p;
    }
    for (const auto &[dir, edge] : expert_edges) {
        deep.insert(edge.first.string());
        deep.insert(edge.second.string());
    }

    for (const auto &p : files) {
        ++st.files;
        const std::string rel = fs::relative(p, root).generic_string();
        std::shared_ptr<arrow::io::MemoryMappedFile> mmap;
        std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
        auto mmap_res = arrow::io::MemoryMappedFile::Open(
            p.string(), arrow::io::FileMode::READ);
        if (!mmap_res.ok()) {
            Problem(st, rel + ": mmap failed: " + mmap_res.status().ToString());
            continue;
        }
        auto reader_res = arrow::ipc::RecordBatchFileReader::Open(
            mmap_res->get());
        if (!reader_res.ok()) {
            Problem(st, rel + ": reader open failed: " +
                        reader_res.status().ToString());
            continue;
        }
        if ((*reader_res)->num_record_batches() != 1) {
            Problem(st, rel + ": expected one record batch, got " +
                        std::to_string((*reader_res)->num_record_batches()));
            continue;
        }
        const Kind kind = Classify((*reader_res)->schema());
        if (kind == Kind::kUnknown) {
            std::ostringstream os;
            os << rel << ": unknown schema [";
            for (int i = 0; i < (*reader_res)->schema()->num_fields(); ++i) {
                if (i) os << ",";
                os << (*reader_res)->schema()->field(i)->name();
            }
            os << "]";
            Problem(st, os.str());
            continue;
        }
        Kind effective = kind;
        if (kind == Kind::kExpert &&
            fs::path(rel).filename() == "shared_experts.arrow")
            effective = Kind::kSharedExpert;

        // CountRows reads only the per-batch message metadata (not the body),
        // so every one of the 270k+ files can be checked without payload I/O.
        const auto rows = (*reader_res)->CountRows();
        if (!rows.ok()) {
            Problem(st, rel + ": count_rows failed: " + rows.status().ToString());
            continue;
        }
        if (*rows <= 0) {
            Problem(st, rel + ": record batch has no rows");
            continue;
        }
        if (effective == Kind::kExpert && *rows != 1) {
            Problem(st, rel + ": expert file has " + std::to_string(*rows) +
                        " rows, expected 1");
            continue;
        }
        ++st.footer_ok;
        ++st.kinds[effective];

        if (!full && deep.count(p.string()) == 0) continue;
        int64_t size = 0;
        std::error_code ec;
        size = static_cast<int64_t>(fs::file_size(p, ec));
        if (ec) size = 0;
        if (!full && budget_bytes > 0 && st.deep_bytes + size > budget_bytes) {
            ++st.budget_skips;
            continue;
        }
        auto batch = (*reader_res)->ReadRecordBatch(0);
        if (!batch.ok()) {
            Problem(st, rel + ": read failed: " + batch.status().ToString());
            continue;
        }
        const auto vst = ValidateKind(effective, **batch, &st.block_hist);
        if (!vst.ok())
            Problem(st, rel + ": " + vst.ToString());
        ++st.deep;
        st.deep_bytes += size;
    }
}

std::string RootDir()
{
    if (const char *env = std::getenv("LSX_ARROW_AUDIT_ROOT"))
        return env;
    return "/mnt/fast_storage/huggingface";
}

int64_t BudgetBytes()
{
    if (const char *env = std::getenv("LSX_ARROW_AUDIT_MAX_BYTES")) {
        const long long v = std::atoll(env);
        if (v > 0) return v;
    }
    return int64_t{256} << 20;
}

bool FullMode()
{
    const char *env = std::getenv("LSX_ARROW_AUDIT_FULL");
    return env && env[0] == '1';
}

} // namespace

TEST_CASE("Every converted model tree has a valid Arrow IPC layout",
          "[arrow][tree][real-model]")
{
    const fs::path root = RootDir();
    if (!fs::exists(root))
        SKIP("Arrow audit root not present: " << root.string());

    std::vector<fs::path> trees;
    for (const auto &e : fs::directory_iterator(root)) {
        if (!e.is_directory()) continue;
        if (fs::exists(e.path() / "logestix"))
            trees.push_back(e.path() / "logestix");
    }
    if (trees.empty())
        SKIP("no converted logestix/ tree under " << root.string());
    std::sort(trees.begin(), trees.end());

    const bool full = FullMode();
    const int64_t budget = BudgetBytes();
    std::ostringstream summary;
    std::ostringstream problems;
    int failures = 0;
    for (const auto &tree : trees) {
        TreeStats st;
        ValidateTree(tree, st, full, budget);
        const std::string model =
            fs::path(tree).parent_path().filename().string();
        summary << model << ": files=" << st.files
                << " footer_ok=" << st.footer_ok << " deep=" << st.deep
                << " deep_MiB=" << (st.deep_bytes >> 20)
                << " budget_skips=" << st.budget_skips
                << " problems=" << st.problems.size();
        for (const auto &[kind, n] : st.kinds)
            summary << " " << KindName(kind) << "=" << n;
        if (!st.block_hist.empty()) {
            summary << " block={";
            bool first = true;
            for (const auto &[b, n] : st.block_hist) {
                if (!first) summary << ",";
                first = false;
                summary << b << ":" << n;
            }
            summary << "}";
        }
        summary << "\n";
        if (!st.problems.empty()) {
            ++failures;
            for (std::size_t i = 0; i < st.problems.size() && i < 20; ++i)
                problems << "\n  " << model << ": " << st.problems[i];
            if (st.problems.size() > 20)
                problems << "\n  " << model << ": ... and "
                         << (st.problems.size() - 20) << " more";
        }
    }
    WARN(summary.str());
    INFO(problems.str());
    CHECK(failures == 0);
}
