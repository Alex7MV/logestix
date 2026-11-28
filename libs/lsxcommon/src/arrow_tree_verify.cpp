// Arrow IPC tree verification — the shared implementation used by the
// audit test and by `lsxforge --verify` (invariants: lsxcommon/arrow_validate.h;
// read-path matrix: docs/ARROW_MATRIX.md).
#include "lsxcommon/arrow_tree_verify.h"
#include "lsxcommon/arrow_validate.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>

#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace lsxcommon {
namespace arrow_check {
namespace {
namespace fs = std::filesystem;

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

TreeFileKind Classify(const std::shared_ptr<arrow::Schema> &schema)
{
    if (FieldsAre(schema, {"w1_weight", "w2_weight", "w3_weight",
                           "w1_scale", "w2_scale", "w3_scale",
                           "w1_global_scale", "w2_global_scale",
                           "w3_global_scale"}))
        return TreeFileKind::kExpert;
    if (FieldsAre(schema, {"name", "shape", "dtype", "packed", "scales",
                           "global_scale"}))
        return TreeFileKind::kDenseNvfp4;
    if (FieldsAre(schema, {"name", "shape", "dtype", "data"}))
        return TreeFileKind::kPinned;
    if (FieldsAre(schema, {"token_id", "token_bytes", "is_special"}))
        return TreeFileKind::kVocab;
    if (FieldsAre(schema, {"merge_left", "merge_right", "merge_rank",
                           "merge_token_id"}))
        return TreeFileKind::kMerges;
    return TreeFileKind::kUnknown;
}

// Per-layer expert codec of one tree, from `<root>/model.index.json`
// (`expert_nvfp4` / `expert_fp8`, which every converter writes). 2 = NVFP4
// nibbles, 1 = MXFP8 bytes, 0 = unknown (the check then accepts either exact
// convention). A missing/unreadable index is not a problem here — the pinned
// plane and the expert planes are validated on their own — so this cannot turn
// an unrelated index defect into an expert-plane failure.
struct ExpertCodecMap {
    std::vector<int32_t> elements_per_byte;   // per physical layer

    int32_t ForLayer(int64_t layer) const
    {
        if (layer < 0 ||
            layer >= static_cast<int64_t>(elements_per_byte.size()))
            return 0;
        return elements_per_byte[static_cast<size_t>(layer)];
    }
};

ExpertCodecMap LoadExpertCodec(const std::filesystem::path &root)
{
    ExpertCodecMap out;
    const auto index = root / "model.index.json";
    std::error_code ec;
    if (!std::filesystem::exists(index, ec) || ec) return out;
    // mmap, like every other Arrow/JSON read in the tree (R-A1): a converted
    // tree can be hundreds of GB and this runs on the host path.
    auto file = arrow::io::MemoryMappedFile::Open(index.string(),
                                                 arrow::io::FileMode::READ);
    if (!file.ok()) return out;
    auto size = (*file)->GetSize();
    if (!size.ok() || *size <= 0 || *size > (int64_t{4} << 20)) return out;
    auto buf = arrow::AllocateBuffer(*size);
    if (!buf.ok()) return out;
    if (!(*file)->Read(*size, (*buf)->mutable_data()).ok()) return out;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const auto view = std::string_view(
        reinterpret_cast<const char *>((*buf)->data()),
        static_cast<size_t>(*size));
    if (parser.parse(view).get(doc) != simdjson::SUCCESS) return out;
    simdjson::dom::array fp8, nvfp4;
    if (doc["expert_fp8"].get_array().get(fp8) != simdjson::SUCCESS) return out;
    const bool have_nvfp4 =
        doc["expert_nvfp4"].get_array().get(nvfp4) == simdjson::SUCCESS;
    std::vector<bool> is_fp8;
    for (auto v : fp8) {
        bool b = false;
        if (v.get_bool().get(b) != simdjson::SUCCESS) return out;
        is_fp8.push_back(b);
    }
    std::vector<bool> is_nvfp4;
    if (have_nvfp4)
        for (auto v : nvfp4) {
            bool b = false;
            if (v.get_bool().get(b) != simdjson::SUCCESS) return out;
            is_nvfp4.push_back(b);
        }
    out.elements_per_byte.assign(is_fp8.size(), 0);
    for (size_t l = 0; l < is_fp8.size(); ++l) {
        const bool nv = have_nvfp4 && l < is_nvfp4.size() && is_nvfp4[l];
        if (is_fp8[l]) out.elements_per_byte[l] = 1;
        else if (nv) out.elements_per_byte[l] = 2;
    }
    return out;
}

// `experts/layer_<n>/exp_<e>.arrow` -> n; -1 when the path is not an expert
// file (shared_experts.arrow, for instance, uses a different contract).
int64_t ExpertLayerFromPath(const std::string &rel)
{
    const std::string marker = "experts/layer_";
    const auto at = rel.find(marker);
    if (at == std::string::npos) return -1;
    auto pos = at + marker.size();
    int64_t layer = 0;
    bool any = false;
    while (pos < rel.size() && rel[pos] >= '0' && rel[pos] <= '9') {
        layer = layer * 10 + (rel[pos] - '0');
        ++pos;
        any = true;
    }
    if (!any || pos >= rel.size() || rel[pos] != '/') return -1;
    return layer;
}

void Problem(TreeVerifyStats &st, const std::string &msg)
{
    st.problems.push_back(msg);
}

arrow::Status ValidateBinaryColumn(const arrow::RecordBatch &batch, int col,
                                   const char *what)
{
    auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        batch.column(col));
    if (!bin) return arrow::Status::TypeError(what, " is not large_binary");
    for (int64_t i = 0; i < batch.num_rows(); ++i) {
        auto v = bin->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
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
            CheckListSlice(*offs, i, u8->length(), what, std::to_string(i)));
    return arrow::Status::OK();
}

arrow::Status ValidateKind(TreeFileKind kind, const arrow::RecordBatch &batch,
                           std::map<int64_t, int64_t> *block_hist,
                           int32_t expert_elements_per_byte)
{
    if (batch.num_rows() <= 0)
        return arrow::Status::Invalid("batch has no rows");

    switch (kind) {
    case TreeFileKind::kExpert:
    case TreeFileKind::kSharedExpert: {
        for (int col : {0, 1, 2})
            ARROW_RETURN_NOT_OK(
                ValidateBinaryColumn(batch, col, "expert weight"));
        for (int col : {3, 4, 5})
            ARROW_RETURN_NOT_OK(
                ValidateScaleColumn(batch, col, "expert scale"));
        // Exact converted-tree contract for ROUTED experts: one E4M3 scale
        // per 16 elements. Shared experts are excluded: DeepSeek stores them
        // as raw F8_E4M3 with E8M0 128x128 tile scales.
        if (kind == TreeFileKind::kExpert) {
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
                    ARROW_RETURN_NOT_OK(CheckExpertScaleCountBits(
                        static_cast<int64_t>(b->GetView(i).size()),
                        offs->Value(i + 1) - offs->Value(i), 16,
                        expert_elements_per_byte,
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
    case TreeFileKind::kDenseNvfp4: {
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
            ARROW_RETURN_NOT_OK(CheckListSlice(
                *shape_offsets, i, flat_shapes->length(), "shape", name));
            ARROW_RETURN_NOT_OK(CheckListSlice(
                *scale_offsets, i, flat_scales->length(), "scales", name));
            auto pv = packeds->GetView(i);
            ARROW_RETURN_NOT_OK(CheckViewRange(
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
            ARROW_RETURN_NOT_OK(CheckNvfp4Size(
                shape.data(), rank, static_cast<int64_t>(pv.size()),
                scale_bytes, gs->Value(i), name));
        }
        return arrow::Status::OK();
    }
    case TreeFileKind::kPinned: {
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
            ARROW_RETURN_NOT_OK(CheckListSlice(
                *offsets, i, flat_shapes->length(), "shape", name));
            auto v = datas->GetView(i);
            ARROW_RETURN_NOT_OK(CheckViewRange(
                v.data(), static_cast<int64_t>(v.size()), datas->value_data(),
                "tensor data", name));
            std::vector<int64_t> shape;
            for (int64_t j = offsets->Value(i); j < offsets->Value(i + 1); ++j)
                shape.push_back(flat_shapes->Value(j));
            ARROW_RETURN_NOT_OK(CheckPinnedTensor(
                name, dtypes->GetString(i), shape.data(),
                static_cast<int64_t>(shape.size()),
                static_cast<int64_t>(v.size())));
        }
        return arrow::Status::OK();
    }
    case TreeFileKind::kVocab:
    case TreeFileKind::kMerges:
        return arrow::Status::OK();
    default:
        return arrow::Status::Invalid("unknown Arrow schema");
    }
}

bool IsDeepByDefault(const std::filesystem::path &rel)
{
    const std::string s = rel.generic_string();
    if (s.find("tokenizer_") != std::string::npos) return true;
    if (s.find("global.arrow") != std::string::npos) return true;
    // Experts are the bulk: the deterministic sample below takes one file per
    // layer directory, so skip them in the default set.
    if (s.find("experts/") != std::string::npos) return false;
    return true;
}

void ValidateTree(const std::filesystem::path &root, TreeVerifyStats &st, bool full,
                  int64_t budget_bytes)
{
    const ExpertCodecMap codec = LoadExpertCodec(root);
    std::vector<std::filesystem::path> files;
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
    std::map<std::string, std::pair<std::filesystem::path, std::filesystem::path>> expert_edges;
    for (const auto &p : files) {
        const std::string rel = fs::relative(p, root).generic_string();
        if (rel.find("experts/") == std::string::npos) continue;
        const std::string dir = std::filesystem::path(rel).parent_path().generic_string();
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
        const TreeFileKind kind = Classify((*reader_res)->schema());
        if (kind == TreeFileKind::kUnknown) {
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
        TreeFileKind effective = kind;
        if (kind == TreeFileKind::kExpert &&
            std::filesystem::path(rel).filename() == "shared_experts.arrow")
            effective = TreeFileKind::kSharedExpert;

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
        if (effective == TreeFileKind::kExpert && *rows != 1) {
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
        int32_t elements_per_byte = 0;
        if (effective == TreeFileKind::kExpert)
            elements_per_byte = codec.ForLayer(ExpertLayerFromPath(rel));
        const auto vst = ValidateKind(effective, **batch, &st.block_hist,
                                      elements_per_byte);
        if (!vst.ok())
            Problem(st, rel + ": " + vst.ToString());
        ++st.deep;
        st.deep_bytes += size;
    }
}

} // namespace

TreeVerifyStats VerifyArrowTree(const std::filesystem::path &converted_root,
                                const TreeVerifyOptions &opts)
{
    TreeVerifyStats st;
    if (!std::filesystem::exists(converted_root)) {
        st.problems.push_back("tree root does not exist: " +
                              converted_root.string());
        return st;
    }
    ValidateTree(converted_root, st, opts.full, opts.budget_bytes);
    return st;
}

std::vector<std::filesystem::path>
FindConvertedTrees(const std::filesystem::path &artifact_root)
{
    std::vector<std::filesystem::path> trees;
    if (!std::filesystem::exists(artifact_root)) return trees;

    // An artifact is detected by `artifact.fb`, never by a directory name
    // (design §2.4.2). This used to be "<vault>/<model>/logestix for every model
    // that has a converted tree", which reported EVERY subdirectory as a tree:
    // correct for the flat pre-split vault, and after the store became
    // org-nested it reported each org ("microsoft", "Qwen") as a model tree with
    // no .arrow files in it. The store is <root>/<org>/<model>/<profile>, so the
    // walk is two levels deep and the predicate is the artifact itself.
    const auto is_artifact = [](const std::filesystem::path &dir) {
        std::error_code ec;
        return std::filesystem::is_regular_file(dir / "artifact.fb", ec);
    };
    std::error_code org_ec;
    for (const auto &org : std::filesystem::directory_iterator(artifact_root, org_ec)) {
        if (org_ec) break;
        if (!org.is_directory()) continue;
        std::error_code model_ec;
        for (const auto &model :
             std::filesystem::directory_iterator(org.path(), model_ec)) {
            if (model_ec) break;
            if (!model.is_directory()) continue;
            std::error_code profile_ec;
            for (const auto &profile :
                 std::filesystem::directory_iterator(model.path(), profile_ec)) {
                if (profile_ec) break;
                if (!profile.is_directory()) continue;
                if (is_artifact(profile.path()))
                    trees.push_back(profile.path());
            }
        }
    }
    std::sort(trees.begin(), trees.end());
    return trees;
}

} // namespace arrow_check
} // namespace lsxcommon
