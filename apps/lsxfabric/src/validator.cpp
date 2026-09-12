#include "validator.h"
#include <simdjson.h>
#include <fstream>
#include <filesystem>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include "lsxcommon/schema.h"
#include "lsxcommon/log.h"
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <cmath>

namespace lsxfabric {

// ── Model index parser ───────────────────────────────────────────────────────

ModelInfo LoadModelIndex(const std::string &index_path)
{
    ModelInfo info;
    std::ifstream f(index_path);
    if (!f) {
        lsxcommon::log::error(absl::StrFormat("Cannot open index: %s", index_path));
        return info;
    }
    std::string json_str((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_str.data(), json_str.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
        lsxcommon::log::error("JSON parse error in index");
        return info;
    }

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        lsxcommon::log::error("Index root not an object");
        return info;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) == simdjson::SUCCESS)
        info.format = std::string(fmt);
    std::string_view name;
    if (root["model"].get_string().get(name) == simdjson::SUCCESS)
        info.model_name = std::string(name);

    int64_t nl = 0, epl = 0, tp = 0, te = 0;
    if (root["num_layers"].get_int64().get(nl) == simdjson::SUCCESS)
        info.num_layers = static_cast<int>(nl);
    if (root["experts_per_layer"].get_int64().get(epl) == simdjson::SUCCESS)
        info.experts_per_layer = static_cast<int>(epl);
    if (root["total_pinned_tensors"].get_int64().get(tp) == simdjson::SUCCESS)
        info.total_pinned = tp;
    if (root["total_experts"].get_int64().get(te) == simdjson::SUCCESS)
        info.total_experts = te;

    return info;
}

// ── DenseNvfp4Schema validation (for logestix-qwen / logestix-glm) ─────────

ValidationResult ValidateDenseNvfp4Schema(const std::string &arrow_path)
{
    ValidationResult r;

    auto result = arrow::io::ReadableFile::Open(arrow_path);
    if (!result.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot open %s", arrow_path)); return r; }
    auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
    if (!reader.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot read batch from %s", arrow_path)); return r; }
    auto batch = (*reader)->ReadRecordBatch(0);
    if (!batch.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot read record from %s", arrow_path)); return r; }

    auto rb = *batch;
    using D = lsxcommon::DenseNvfp4Schema;
    const int expected_cols = 6;
    if (rb->num_columns() != expected_cols) {
        lsxcommon::log::error(absl::StrFormat("  DenseNvfp4Schema: expected %d columns, got %d", expected_cols, rb->num_columns()));
        r.add_error(); return r;
    }

    int64_t n = rb->num_rows();
    lsxcommon::log::info(absl::StrFormat("  DenseNvfp4Schema: %d tensors in %s", n, arrow_path));

    // Validate packed binary (E2M1 nibbles).
    auto packed_arr = std::static_pointer_cast<arrow::LargeBinaryArray>(rb->column(D::kIdxPacked));
    for (int64_t i = 0; i < n; ++i) {
        auto view = packed_arr->GetView(i);
        if (!ValidateNVFP4Data(reinterpret_cast<const uint8_t*>(view.data()),
                               static_cast<int64_t>(view.size()))) {
            lsxcommon::log::error(absl::StrFormat("  Packed data invalid at row %d", i));
            r.add_error();
        }
    }

    // Validate scales (E4M3).
    auto scales_arr = std::static_pointer_cast<arrow::ListArray>(rb->column(D::kIdxScales));
    auto scales_u8 = std::static_pointer_cast<arrow::UInt8Array>(scales_arr->values());
    const int32_t *offsets = scales_arr->raw_value_offsets();
    for (int64_t i = 0; i < n; ++i) {
        int64_t count = offsets[i + 1] - offsets[i];
        if (!ValidateE4M3Scales(scales_u8->raw_values() + offsets[i], count)) {
            lsxcommon::log::error(absl::StrFormat("  E4M3 scales invalid at row %d", i));
            r.add_error();
        }
    }

    // Validate global scale.
    auto gs_arr = std::static_pointer_cast<arrow::FloatArray>(rb->column(D::kIdxGlobalScale));
    for (int64_t i = 0; i < n; ++i) {
        if (!ValidateGlobalScale(gs_arr->Value(i))) {
            lsxcommon::log::error(absl::StrFormat("  Global scale invalid at row %d: %f", i, gs_arr->Value(i)));
            r.add_error();
        }
    }

    return r;
}

// ── Pinned tensor validation ─────────────────────────────────────────────────

ValidationResult ValidatePinnedTensors(const std::string &pinned_dir, int64_t expected_count)
{
    namespace fs = std::filesystem;
    ValidationResult r;
    int64_t loaded = 0;

    auto check_batch = [&](const std::string &path) {
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot open %s", path)); return; }
        auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
        if (!reader.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot read batch from %s", path)); return; }
        auto batch = (*reader)->ReadRecordBatch(0);
        if (!batch.ok()) { r.add_error(); lsxcommon::log::error(absl::StrFormat("  Cannot read record from %s", path)); return; }
        int ncol = (*batch)->num_columns();
        lsxcommon::log::info(absl::StrFormat("  %s: %d columns, %d rows", path, ncol, (*batch)->num_rows()));
        loaded += (*batch)->num_rows();
    };

    lsxcommon::log::info(absl::StrFormat("Validating pinned tensors from %s...", pinned_dir));

    auto global_path = fs::path(pinned_dir) / "global.arrow";
    if (fs::exists(global_path))
        check_batch(global_path.string());

    for (const auto &entry : fs::directory_iterator(pinned_dir)) {
        if (entry.path().extension() != ".arrow") continue;
        const std::string fname = entry.path().filename().string();
        if (fname == "global.arrow") continue;
        if (fname.rfind("tokenizer_", 0) == 0) continue;
        if (fname == "shared_experts.arrow") continue;
        check_batch(entry.path().string());
    }

    if (expected_count > 0 && loaded != expected_count) {
        lsxcommon::log::warn(absl::StrFormat("  Expected %d pinned tensors, loaded %d", expected_count, loaded));
        r.add_warning();
    }
    return r;
}

// ── NVFP4 data validation ────────────────────────────────────────────────────

bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes)
{
    for (int64_t i = 0; i < nbytes; ++i) {
        // All 16 nibble values 0x00-0x0F are valid NVFP4 E2M1 codes
        uint8_t lo = data[i] & 0x0F;
        uint8_t hi = (data[i] >> 4) & 0x0F;
        if (lo > 0x0F || hi > 0x0F) return false;
    }
    return true;
}

bool ValidateE4M3Scales(const uint8_t *scales, int64_t count)
{
    for (int64_t i = 0; i < count; ++i) {
        uint8_t b = scales[i];
        if (b & 0x80) return false;
        int exp4 = (b >> 3) & 0x0F;
        if (exp4 < 1 || exp4 > 14) return false;
        if (b & 0x07) return false;
    }
    return true;
}

bool ValidateGlobalScale(float scale)
{
    return std::isfinite(scale) && scale > 0.0f;
}

// ── Expert validation ────────────────────────────────────────────────────────

ValidationResult ValidateExpert(const std::string &arrow_path)
{
    ValidationResult r;

    auto result = arrow::io::ReadableFile::Open(arrow_path);
    if (!result.ok()) { r.add_error(); return r; }
    auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
    if (!reader.ok()) { r.add_error(); return r; }
    auto batch = (*reader)->ReadRecordBatch(0);
    if (!batch.ok()) { r.add_error(); return r; }

    auto rb = *batch;
    if (rb->num_columns() != lsxcommon::ExpertSchema::kNumFields) {
        lsxcommon::log::error(absl::StrFormat("  Expert schema: expected %d columns, got %d",
            lsxcommon::ExpertSchema::kNumFields, rb->num_columns()));
        r.add_error(); return r;
    }

    using E = lsxcommon::ExpertSchema;
    int64_t n = rb->num_rows();

    // Loop over every row of the batch (mirrors ValidateDenseNvfp4Schema
    // above): the expert schema does not forbid >1 row per exp_NNN.arrow
    // file, so checking only row 0 would silently PASS a file whose later
    // rows are corrupt.
    auto check_binary = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto bin = std::static_pointer_cast<arrow::LargeBinaryArray>(arr);
        for (int64_t i = 0; i < n; ++i) {
            auto view = bin->GetView(i);
            auto data = reinterpret_cast<const uint8_t*>(view.data());
            auto size = static_cast<int64_t>(view.size());
            if (!ValidateNVFP4Data(data, size)) {
                lsxcommon::log::error(absl::StrFormat("  NVFP4 data invalid in %s at row %d", name, i));
                r.add_error();
            }
        }
    };

    auto check_scales = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto lst = std::static_pointer_cast<arrow::ListArray>(arr);
        auto offsets = lst->raw_value_offsets();
        auto u8 = std::static_pointer_cast<arrow::UInt8Array>(lst->values());
        for (int64_t i = 0; i < n; ++i) {
            int64_t count = offsets[i + 1] - offsets[i];
            const uint8_t *data = u8->raw_values() + offsets[i];
            if (!ValidateE4M3Scales(data, count)) {
                lsxcommon::log::error(absl::StrFormat("  E4M3 scales invalid in %s at row %d", name, i));
                r.add_error();
            }
        }
    };

    auto check_global = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto f32 = std::static_pointer_cast<arrow::FloatArray>(arr);
        for (int64_t i = 0; i < n; ++i) {
            float s = f32->Value(i);
            if (!ValidateGlobalScale(s)) {
                lsxcommon::log::error(absl::StrFormat("  Global scale invalid in %s at row %d: %f", name, i, s));
                r.add_error();
            }
        }
    };

    check_binary(E::kIdxW1Weight, "w1_weight");
    check_binary(E::kIdxW2Weight, "w2_weight");
    check_binary(E::kIdxW3Weight, "w3_weight");
    check_scales(E::kIdxW1Scale, "w1_scale");
    check_scales(E::kIdxW2Scale, "w2_scale");
    check_scales(E::kIdxW3Scale, "w3_scale");
    check_global(E::kIdxW1GlobalScale, "w1_global_scale");
    check_global(E::kIdxW2GlobalScale, "w2_global_scale");
    check_global(E::kIdxW3GlobalScale, "w3_global_scale");

    lsxcommon::log::info(absl::StrFormat("  Expert validated: %d rows", n));
    return r;
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

ValidationResult RunValidator(const std::string &model_path)
{
    namespace fs = std::filesystem;
    ValidationResult r;

    auto index_path = fs::path(model_path) / "logestix" / "model.index.json";
    if (!fs::exists(index_path)) {
        lsxcommon::log::error(absl::StrFormat("model.index.json not found in %s/logestix", model_path));
        r.add_error(); return r;
    }

    auto info = LoadModelIndex(index_path.string());
    bool is_qwen = (info.format == "logestix-qwen");
    bool is_glm = (info.format == "logestix-glm");

    if (info.format != "logestix-v1" && !is_qwen && !is_glm) {
        lsxcommon::log::warn(absl::StrFormat("  Unknown format '%s', continuing anyway", info.format));
        r.add_warning();
    }

    lsxcommon::log::info(absl::StrFormat("Model: %s format=%s layers=%d experts/layer=%d pinned=%d experts=%d",
        info.model_name, info.format, info.num_layers, info.experts_per_layer,
        info.total_pinned, info.total_experts));

    auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";
    if (fs::exists(pinned_dir)) {
        auto pr = ValidatePinnedTensors(pinned_dir.string(), info.total_pinned);
        if (!pr.ok) r.add_error();
        r.warnings += pr.warnings;
    } else {
        lsxcommon::log::warn("  No pinned/ directory found");
        r.add_warning();
    }

    int experts_checked = 0;
    if (is_qwen || is_glm) {
        lsxcommon::log::info(absl::StrFormat("  Format %s: validating DenseNvfp4Schema tensors", info.format));
        auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";
        if (fs::exists(pinned_dir)) {
            for (const auto &entry : fs::directory_iterator(pinned_dir)) {
                auto fn = entry.path().filename().string();
                if (fn.find("nvfp4") != std::string::npos && entry.path().extension() == ".arrow") {
                    auto dr = ValidateDenseNvfp4Schema(entry.path().string());
                    if (!dr.ok) { r.add_error(); r.warnings += dr.warnings; }
                }
            }
        }
    } else {
        for (int layer = 0; layer < info.num_layers; ++layer) {
            auto exp_path = fs::path(model_path) / "logestix"
                / absl::StrCat("experts/layer_", layer)
                / absl::StrCat("exp_", layer, ".arrow");
            if (!fs::exists(exp_path)) {
                lsxcommon::log::warn(absl::StrFormat("  Expert not found: %s", exp_path.string()));
                r.add_warning();
                continue;
            }
            lsxcommon::log::info(absl::StrFormat("Validating expert layer=%d...", layer));
            auto er = ValidateExpert(exp_path.string());
            if (!er.ok) r.add_error();
            r.warnings += er.warnings;
            ++experts_checked;
        }
    }

    lsxcommon::log::info("=== Validation Summary ===");
    lsxcommon::log::info(absl::StrFormat("  Pinned tensors checked: %d", info.total_pinned));
    lsxcommon::log::info(absl::StrFormat("  Experts checked: %d", experts_checked));
    lsxcommon::log::info(absl::StrFormat("  Errors: %d", r.errors));
    lsxcommon::log::info(absl::StrFormat("  Warnings: %d", r.warnings));
    if (r.ok)
        lsxcommon::log::info("  Result: PASS");
    else
        lsxcommon::log::error("  Result: FAIL");

    return r;
}

} // namespace lsxfabric
