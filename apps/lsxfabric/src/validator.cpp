#include "validator.h"
#include <simdjson.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include "lsxcommon/schema.h"
#include <cmath>

namespace lsxfabric {

// ── Model index parser ───────────────────────────────────────────────────────

ModelInfo LoadModelIndex(const std::string &index_path)
{
    ModelInfo info;
    std::ifstream f(index_path);
    if (!f) {
        spdlog::error("Cannot open index: {}", index_path);
        return info;
    }
    std::string json_str((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_str.data(), json_str.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
        spdlog::error("JSON parse error in index");
        return info;
    }

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        spdlog::error("Index root not an object");
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

// ── Pinned tensor validation ─────────────────────────────────────────────────

ValidationResult ValidatePinnedTensors(const std::string &pinned_dir, int64_t expected_count)
{
    namespace fs = std::filesystem;
    ValidationResult r;
    int64_t loaded = 0;

    auto check_batch = [&](const std::string &path) {
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) { r.add_error(); spdlog::error("  Cannot open {}", path); return; }
        auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
        if (!reader.ok()) { r.add_error(); spdlog::error("  Cannot read batch from {}", path); return; }
        auto batch = (*reader)->ReadRecordBatch(0);
        if (!batch.ok()) { r.add_error(); spdlog::error("  Cannot read record from {}", path); return; }
        int ncol = (*batch)->num_columns();
        spdlog::info("  {}: {} columns, {} rows", path, ncol, (*batch)->num_rows());
        loaded += ncol;
    };

    spdlog::info("Validating pinned tensors from {}...", pinned_dir);

    auto global_path = fs::path(pinned_dir) / "global.arrow";
    if (fs::exists(global_path))
        check_batch(global_path.string());

    for (const auto &entry : fs::directory_iterator(pinned_dir)) {
        if (entry.path().extension() != ".arrow") continue;
        if (entry.path().filename() == "global.arrow") continue;
        check_batch(entry.path().string());
    }

    if (expected_count > 0 && loaded != expected_count) {
        spdlog::warn("  Expected {} pinned tensors, loaded {}", expected_count, loaded);
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
        spdlog::error("  Expert schema: expected {} columns, got {}",
            lsxcommon::ExpertSchema::kNumFields, rb->num_columns());
        r.add_error(); return r;
    }

    using E = lsxcommon::ExpertSchema;
    int64_t n = rb->num_rows();

    auto check_binary = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto bin = std::static_pointer_cast<arrow::BinaryArray>(arr);
        if (n > 0) {
            auto view = bin->GetView(0);
            auto data = reinterpret_cast<const uint8_t*>(view.data());
            auto size = static_cast<int64_t>(view.size());
            if (!ValidateNVFP4Data(data, size)) {
                spdlog::error("  NVFP4 data invalid in {}", name);
                r.add_error();
            }
        }
    };

    auto check_scales = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto lst = std::static_pointer_cast<arrow::ListArray>(arr);
        if (n > 0) {
            auto offsets = lst->raw_value_offsets();
            int64_t count = offsets[1] - offsets[0];
            auto u8 = std::static_pointer_cast<arrow::UInt8Array>(lst->values());
            const uint8_t *data = u8->raw_values() + offsets[0];
            if (!ValidateE4M3Scales(data, count)) {
                spdlog::error("  E4M3 scales invalid in {}", name);
                r.add_error();
            }
        }
    };

    auto check_global = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto f32 = std::static_pointer_cast<arrow::FloatArray>(arr);
        if (n > 0) {
            float s = f32->Value(0);
            if (!ValidateGlobalScale(s)) {
                spdlog::error("  Global scale invalid in {}: {}", name, s);
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

    spdlog::info("  Expert validated: {} rows", n);
    return r;
}

// ── Orchestrator ─────────────────────────────────────────────────────────────

ValidationResult RunValidator(const std::string &model_path)
{
    namespace fs = std::filesystem;
    ValidationResult r;

    auto index_path = fs::path(model_path) / "model.index.json";
    if (!fs::exists(index_path)) {
        spdlog::error("model.index.json not found in {}", model_path);
        r.add_error(); return r;
    }

    auto info = LoadModelIndex(index_path.string());
    if (info.format != "logestix-v1") {
        spdlog::warn("  Unknown format '{}', continuing anyway", info.format);
        r.add_warning();
    }

    spdlog::info("Model: {}  layers={} experts/layer={} pinned={} experts={}",
        info.model_name, info.num_layers, info.experts_per_layer,
        info.total_pinned, info.total_experts);

    auto pinned_dir = fs::path(model_path) / "pinned";
    if (fs::exists(pinned_dir)) {
        auto pr = ValidatePinnedTensors(pinned_dir.string(), info.total_pinned);
        if (!pr.ok) r.add_error();
        r.warnings += pr.warnings;
    } else {
        spdlog::warn("  No pinned/ directory found");
        r.add_warning();
    }

    int experts_checked = 0;
    for (int layer = 0; layer < info.num_layers; ++layer) {
        auto exp_path = fs::path(model_path)
            / ("experts/layer_" + std::to_string(layer))
            / "exp_000.arrow";
        if (!fs::exists(exp_path)) {
            spdlog::warn("  Expert not found: {}", exp_path.string());
            r.add_warning();
            continue;
        }
        spdlog::info("Validating expert layer={}...", layer);
        auto er = ValidateExpert(exp_path.string());
        if (!er.ok) r.add_error();
        r.warnings += er.warnings;
        ++experts_checked;
    }

    spdlog::info("=== Validation Summary ===");
    spdlog::info("  Pinned tensors checked: {}", info.total_pinned);
    spdlog::info("  Experts checked: {}", experts_checked);
    spdlog::info("  Errors: {}", r.errors);
    spdlog::info("  Warnings: {}", r.warnings);
    if (r.ok)
        spdlog::info("  Result: PASS");
    else
        spdlog::error("  Result: FAIL");

    return r;
}

} // namespace lsxfabric
