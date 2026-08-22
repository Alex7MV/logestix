#include "lsxcommon/converter.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/diagnostics.h"
#include <arrow/ipc/writer.h>
#include <arrow/io/file.h>
#include <spdlog/spdlog.h>
#include <simdjson.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <regex>
#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <filesystem>

namespace lsxcommon {

namespace {

// ── Safetensors helpers ──────────────────────────────────────────────────────

int64_t dtype_size(const std::string &dtype)
{
    if (dtype == "F8_E8M0") return 1;
    if (dtype == "F8_E4M3") return 1;
    if (dtype == "I8" || dtype == "U8") return 1;
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "F64" || dtype == "I64") return 8;
    return 0;
}

std::string format_size(int64_t bytes)
{
    if (bytes >= 1024LL * 1024 * 1024)
        return absl::StrCat(bytes / (1024 * 1024 * 1024), " GiB");
    if (bytes >= 1024 * 1024)
        return absl::StrCat(bytes / (1024 * 1024), " MiB");
    if (bytes >= 1024)
        return absl::StrCat(bytes / 1024, " KiB");
    return absl::StrCat(bytes, " B");
}

// ── Expert name parsing ──────────────────────────────────────────────────────

struct ExpertKey {
    int layer;
    int expert;
    bool operator==(const ExpertKey &o) const noexcept {
        return layer == o.layer && expert == o.expert;
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey &k) const noexcept {
        return (static_cast<std::size_t>(k.layer) << 32)
             | static_cast<std::size_t>(k.expert);
    }
};

struct ParsedName {
    int layer = -1;
    int expert = -1;
    int weight_id = -1;
    bool is_scale = false;
};

std::optional<ParsedName> parse_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    absl::SimpleAtoi(m[2].str(), &p.expert);
    absl::SimpleAtoi(m[3].str(), &p.weight_id);
    p.is_scale  = (m[4] == "scale");
    return p;
}

// Shared expert tensors: layers.N.ffn.shared_experts.w{1,2,3}.{weight,scale}
std::optional<ParsedName> parse_shared_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.shared_experts\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    p.expert    = 0;
    absl::SimpleAtoi(m[2].str(), &p.weight_id);
    p.is_scale  = (m[3] == "scale");
    return p;
}

} // anonymous namespace

// ── VRAM profile classification (spec 2026-08-21) ───────────────────────────

bool DsIsNvfp4Tensor(const std::string &name)
{
    return parse_expert_name(name).has_value()
        || parse_shared_expert_name(name).has_value();
}

VramTensorClass DsMemClass(const std::string &name)
{
    return parse_expert_name(name).has_value()
        ? VramTensorClass::RoutedExpert
        : VramTensorClass::Resident;
}

// ── Safetensors public API ───────────────────────────────────────────────────

bool is_staged_tensor(const std::string &name)
{
    return name.find(".ffn.experts.") != std::string::npos;
}

static std::string pinned_category_name(const std::string &name)
{
    if (name == "embed.weight")
        return "Embedding";
    if (name == "head.weight")
        return "LM Head";
    if (name.rfind("hc_head_", 0) == 0)
        return "LM Head";
    if (name.find(".attn.") != std::string::npos || name.find(".attn_norm.") != std::string::npos)
        return "Attention";
    if (name.find(".hc_attn") != std::string::npos)
        return "Attention";
    if (name.find(".hc_ffn") != std::string::npos)
        return "HC-FFN";
    if (name.find(".ffn.gate.") != std::string::npos)
        return "MoE Router";
    if (name.find(".ffn.shared_experts.") != std::string::npos)
        return "Shared Experts";
    if (name.find(".ffn_norm.") != std::string::npos)
        return "Layer Norms";
    if (name.rfind("model.norm.", 0) == 0 || name == "norm.weight")
        return "Final Norm";
    if (name.find(".enorm.") != std::string::npos || name.find(".hnorm.") != std::string::npos)
        return "MTP Norms";
    if (name.find(".e_proj.") != std::string::npos || name.find(".h_proj.") != std::string::npos)
        return "MTP Projections";
    if (name.rfind("mtp.", 0) == 0)
        return "MTP Norms";
    return "Other";
}

void print_shard(const SafetensorsModel::Shard &shard)
{
    spdlog::info("--- {} ({}) ---", shard.filename, format_size(shard.file_size));
    for (auto &t : shard.tensors) {
        std::string shape_str = absl::StrCat("[", absl::StrJoin(t.shape, ", "), "]");

        spdlog::info("  {} {:<50} {:<8} {:<24} {}",
            is_staged_tensor(t.name) ? "[S]" : "[P]",
            t.name, t.dtype, shape_str, format_size(t.byte_size()));
    }
}

void print_model(const SafetensorsModel &model)
{
    int64_t total_tensors = 0;
    int64_t total_bytes = 0;
    int64_t pinned_tensors = 0;
    int64_t pinned_bytes = 0;
    int64_t staged_tensors = 0;
    int64_t staged_bytes = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            total_tensors++;
            total_bytes += t.byte_size();
            if (is_staged_tensor(t.name)) {
                staged_tensors++;
                staged_bytes += t.byte_size();
            } else {
                pinned_tensors++;
                pinned_bytes += t.byte_size();
            }
        }
    }

    spdlog::info("=== Model Summary ===");
    spdlog::info("Shards: {}", model.shards.size());
    spdlog::info("Tensors: {}", total_tensors);
    spdlog::info("Total tensor data: {}", format_size(total_bytes));
    spdlog::info("--- Memory classification ---");
    spdlog::info("Pinned (always in VRAM): {} tensors, {}",
        pinned_tensors, format_size(pinned_bytes));
    spdlog::info("Staged (MoE experts):   {} tensors, {}",
        staged_tensors, format_size(staged_bytes));
    spdlog::info("--- Pinned breakdown ---");
    absl::btree_map<std::string, int64_t> cat_tensors, cat_bytes;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            if (is_staged_tensor(t.name)) continue;
            auto cat = pinned_category_name(t.name);
            cat_tensors[cat]++;
            cat_bytes[cat] += t.byte_size();
        }
    }
    for (auto &[cat, cnt] : cat_tensors) {
        spdlog::info("  {:<22} {:>6} tensors, {}",
            cat, cnt, format_size(cat_bytes[cat]));
    }
    // Runtime VRAM estimates (min/optimal/max) are no longer guessed here:
    // they are computed at conversion time and reported via the "vram" block
    // of model.index.json / IModel::VramRequirements() (see vram_estimate).
}

void print_nvfp4_forecast(const SafetensorsModel &model)
{
    struct CatStats {
        int64_t orig_bytes = 0;
        int64_t nvfp4_bytes = 0;
        int count = 0;
    };
    absl::btree_map<std::string, CatStats> cats;

    int64_t total_orig = 0;
    int64_t total_nvfp4 = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto cat = is_staged_tensor(t.name)
                ? std::string("MoE Experts")
                : pinned_category_name(t.name);
            int64_t orig = t.byte_size();
            int64_t nvfp4 = t.numel() / 2;
            cats[cat].orig_bytes += orig;
            cats[cat].nvfp4_bytes += nvfp4;
            cats[cat].count++;
            total_orig += orig;
            total_nvfp4 += nvfp4;
        }
    }

    spdlog::info("=== NVFP4 Quantization Forecast ===");
    for (auto &[cat, s] : cats) {
        double ratio = s.nvfp4_bytes > 0
            ? static_cast<double>(s.orig_bytes) / s.nvfp4_bytes
            : 0.0;
        spdlog::info("  {:<22} {:>6} tensors, {} -> {} ({:.1f}x)",
            cat, s.count,
            format_size(s.orig_bytes),
            format_size(s.nvfp4_bytes),
            ratio);
    }

    double total_ratio = total_nvfp4 > 0
        ? static_cast<double>(total_orig) / total_nvfp4
        : 0.0;
    int64_t cat_count = 0;
    for (auto &[_, s] : cats) cat_count += s.count;
    spdlog::info("  {:<22} {:>6} tensors, {} -> {} ({:.1f}x)",
        "Total", cat_count,
        format_size(total_orig),
        format_size(total_nvfp4),
        total_ratio);
}

// ── Safetensors file parser ──────────────────────────────────────────────────

SafetensorsModel::Shard read_safetensors_file(const std::string &path)
{
    SafetensorsModel::Shard shard;
    shard.filename = std::filesystem::path(path).filename().string();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("Cannot open file: {}", path);
        return shard;
    }

    file.seekg(0, std::ios::end);
    shard.file_size = static_cast<int64_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    if (!file || header_len == 0 || header_len > static_cast<uint64_t>(shard.file_size) - 8) {
        spdlog::error("Invalid safetensors header in {}", path);
        return shard;
    }

    std::string json_str(header_len, '\0');
    file.read(json_str.data(), header_len);
    if (!file) {
        spdlog::error("Failed to read safetensors header from {}", path);
        return shard;
    }

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_str.data(), json_str.size());
    simdjson::ondemand::document doc;
    auto err = parser.iterate(padded).get(doc);
    if (err) {
        spdlog::error("JSON parse error in {}: {}", path, simdjson::error_message(err));
        return shard;
    }

    simdjson::ondemand::object obj;
    err = doc.get_object().get(obj);
    if (err) {
        spdlog::error("JSON root not an object in {}", path);
        return shard;
    }

    for (auto json_field : obj) {
        std::string_view key;
        if (json_field.unescaped_key().get(key) != simdjson::SUCCESS)
            continue;
        if (key == "__metadata__")
            continue;

        simdjson::ondemand::value val;
        if (json_field.value().get(val) != simdjson::SUCCESS)
            continue;

        simdjson::ondemand::object tensor_obj;
        if (val.get_object().get(tensor_obj) != simdjson::SUCCESS)
            continue;

        TensorInfo info;
        info.name = std::string(key);

        std::string_view dtype_sv;
        if (tensor_obj["dtype"].get_string().get(dtype_sv) == simdjson::SUCCESS)
            info.dtype = std::string(dtype_sv);

        simdjson::ondemand::array shape_arr;
        if (tensor_obj["shape"].get_array().get(shape_arr) == simdjson::SUCCESS) {
            for (auto dim_val : shape_arr) {
                int64_t d = 0;
                if (dim_val.get_int64().get(d) == simdjson::SUCCESS)
                    info.shape.push_back(d);
            }
        }

        simdjson::ondemand::array offsets_arr;
        if (tensor_obj["data_offsets"].get_array().get(offsets_arr) == simdjson::SUCCESS) {
            int count = 0;
            for (auto off_val : offsets_arr) {
                int64_t off = 0;
                if (off_val.get_int64().get(off) == simdjson::SUCCESS) {
                    if (count == 0) info.data_begin = off;
                    else if (count == 1) info.data_end = off;
                    ++count;
                }
            }
        }

        shard.tensors.push_back(std::move(info));
    }

    return shard;
}

// ── TensorInfo implementation ─────────────────────────────────────────────────

int64_t TensorInfo::numel() const
{
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

int64_t TensorInfo::byte_size() const
{
    return numel() * dtype_size(dtype);
}

// ── Safetensors scanning ─────────────────────────────────────────────────────

SafetensorsModel scan_directory(const std::string &path)
{
    SafetensorsModel model;

    namespace fs = std::filesystem;
    fs::path dir(path);

    if (!fs::exists(dir)) {
        spdlog::error("Path does not exist: {}", path);
        return model;
    }

    if (fs::is_regular_file(dir)) {
        auto shard = read_safetensors_file(dir.string());
        if (!shard.tensors.empty())
            model.shards.push_back(std::move(shard));
        return model;
    }

    std::vector<std::string> ordered_files;
    fs::path index_path = dir / "model.safetensors.index.json";

    if (fs::exists(index_path)) {
        std::ifstream index_file(index_path);
        if (index_file) {
            std::string index_json((std::istreambuf_iterator<char>(index_file)),
                                    std::istreambuf_iterator<char>());

            simdjson::ondemand::parser parser;
            simdjson::padded_string padded(index_json.data(), index_json.size());
            simdjson::ondemand::document doc;
            if (parser.iterate(padded).get(doc) == simdjson::SUCCESS) {
                simdjson::ondemand::object root;
                if (doc.get_object().get(root) == simdjson::SUCCESS) {
                    simdjson::ondemand::object weight_map;
                    if (root["weight_map"].get_object().get(weight_map) == simdjson::SUCCESS) {
                        for (auto entry : weight_map) {
                            std::string_view shard_name;
                            if (entry.value().get_string().get(shard_name) == simdjson::SUCCESS) {
                                std::string fn(shard_name);
                                if (std::find(ordered_files.begin(), ordered_files.end(), fn) == ordered_files.end())
                                    ordered_files.push_back(fn);
                            }
                        }
                    }
                }
            }
        }
    }

    if (ordered_files.empty()) {
        for (auto &entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors")
                ordered_files.push_back(entry.path().filename().string());
        }
        std::sort(ordered_files.begin(), ordered_files.end());
    }

    for (auto &fn : ordered_files) {
        auto full_path = (dir / fn).string();
        spdlog::info("Reading: {}", fn);
        auto shard = read_safetensors_file(full_path);
        if (!shard.tensors.empty())
            model.shards.push_back(std::move(shard));
    }

    return model;
}

// ── NVFP4 Packing (GPU) ─────────────────────────────────────────────────────

arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw)
{
    PackedExpert pe;
    pe.w1_shape = raw.w1_shape;
    pe.w2_shape = raw.w2_shape;
    pe.w3_shape = raw.w3_shape;

    auto product = [](const std::vector<int64_t> &v) -> int64_t {
        int64_t n = 1;
        for (auto d : v) n *= d;
        return n;
    };

    auto process = [&](const std::vector<uint8_t> &src,
                        const std::string &dtype,
                        const std::vector<int64_t> &shape,
                        const std::vector<uint8_t> &src_scale,
                        const std::vector<int64_t> &scale_shape,
                        std::vector<uint8_t> &packed,
                        std::vector<uint8_t> &scales,
                        float &global_scale) -> bool
    {
        int64_t numel = product(shape);
        if (numel == 0 || src.empty()) return false;

        // Routed/shared experts on disk are already quantized (F8_E4M3 or
        // E2M1 packed 2-elements per byte) with paired F8_E8M0 scales. Pass
        // the bytes through unchanged — re-quantizing would corrupt the values.
        if ((dtype == "I8" || dtype == "U8" || dtype == "F8_E4M3") && !src_scale.empty()) {
            packed       = src;
            scales       = src_scale;
            global_scale = 1.0f;
            return true;
        }

        const uint8_t *e8m0 = src_scale.empty() ? nullptr : src_scale.data();
        return gpu::quantize_tensor_gpu(src.data(), numel, dtype,
                                         e8m0, shape, scale_shape,
                                         packed, scales, global_scale);
    };

    if (!process(raw.w1_raw, raw.w1_dtype, raw.w1_shape,
                  raw.w1_scale_raw, raw.w1_scale_shape,
                  pe.w1_packed, pe.w1_scales, pe.w1_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w1");

    if (!process(raw.w2_raw, raw.w2_dtype, raw.w2_shape,
                  raw.w2_scale_raw, raw.w2_scale_shape,
                  pe.w2_packed, pe.w2_scales, pe.w2_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w2");

    if (!process(raw.w3_raw, raw.w3_dtype, raw.w3_shape,
                  raw.w3_scale_raw, raw.w3_scale_shape,
                  pe.w3_packed, pe.w3_scales, pe.w3_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w3");

    return pe;
}

// ── Save single expert to Arrow IPC file ────────────────────────────────────

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto layer_dir = output_root /
        "experts" / absl::StrCat("layer_", layer_idx);
    fs::create_directories(layer_dir);

    auto out_path = layer_dir /
        absl::StrCat("exp_", expert_idx, ".arrow");

    ARROW_ASSIGN_OR_RAISE(auto batch, ExpertSchema::MakeBatch(expert));
    diag::ValidateBatchBuffers(*batch, "SaveExpertToArrow pre-write");
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, out_path));

    spdlog::info("  wrote layer={:03d} expert={:03d}  {} -> {}",
        layer_idx, expert_idx,
        batch->column(0)->length(),
        out_path.string());

    return arrow::Status::OK();
}

// ── Save all pinned tensors ─────────────────────────────────────────────────

arrow::Status SavePinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_root / "pinned");

    absl::flat_hash_map<std::string, std::vector<TensorRecord>> by_layer;
    std::vector<TensorRecord> global;

    for (auto &t : tensors) {
        static const std::regex layer_re(R"(layers\.(\d+)\..+)");
        std::smatch m;
        if (std::regex_match(t.name, m, layer_re)) {
            by_layer[std::string("layer_") + m[1].str()].push_back(t);
        } else {
            global.push_back(t);
        }
    }

    if (!global.empty()) {
        auto path = output_root / "pinned" / "global.arrow";
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(global));
        diag::ValidateBatchBuffers(*batch, "SavePinnedTensors global");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors -> {}", global.size(), path.string());
    }

    for (auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned" / (layer + ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        diag::ValidateBatchBuffers(*batch, ("SavePinnedTensors " + layer).c_str());
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors for {} -> {}",
            tens.size(), layer, path.string());
    }

    return arrow::Status::OK();
}

// ── Save all shared expert layers ────────────────────────────────────────────

// Packs the NVFP4-compressed shared expert weights (w1/w2/w3) for every layer
// and writes a single pinned/shared_experts.arrow with one record per layer.
arrow::Status SaveSharedExpertsToArrow(
    const absl::flat_hash_map<int, ExpertData> &shared_by_layer,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    if (shared_by_layer.empty()) {
        spdlog::info("  no shared experts found, skipping shared_experts.arrow");
        return arrow::Status::OK();
    }

    int max_layer = -1;
    for (auto &[layer, sd] : shared_by_layer)
        max_layer = std::max(max_layer, layer);
    if (max_layer < 0)
        return arrow::Status::OK();

    std::vector<PackedExpert> shared_layers;
    shared_layers.resize(static_cast<size_t>(max_layer) + 1);

    int packed_ok = 0;
    for (auto &[layer, sd] : shared_by_layer) {
        if (sd.w1_raw.empty() || sd.w2_raw.empty() || sd.w3_raw.empty()) {
            spdlog::warn("  incomplete shared expert layer {} (w1={} w2={} w3={})",
                layer, sd.w1_raw.size(), sd.w2_raw.size(), sd.w3_raw.size());
            continue;
        }
        auto packed_res = PackExpertNVFP4(std::move(sd));
        if (!packed_res.ok()) {
            spdlog::error("  failed to pack shared expert layer {}: {}",
                layer, packed_res.status().ToString());
            continue;
        }
        shared_layers[static_cast<size_t>(layer)] = *packed_res;
        ++packed_ok;
    }

    while (!shared_layers.empty() && shared_layers.back().w1_packed.empty())
        shared_layers.pop_back();
    if (shared_layers.empty())
        return arrow::Status::OK();

    fs::create_directories(output_root / "pinned");
    ARROW_ASSIGN_OR_RAISE(auto batch, SharedExpertSchema::MakeBatch(shared_layers));
    diag::ValidateBatchBuffers(*batch, "SaveSharedExpertsToArrow result");
    auto path = output_root / "pinned" / "shared_experts.arrow";
    ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    spdlog::info("  packed {} shared-expert layers -> {}", packed_ok, path.string());

    return arrow::Status::OK();
}

// ── Targeted shared-expert repack ───────────────────────────────────────────

// Re-packs only the shared-expert weights from the source safetensors shards
// into pinned/shared_experts.arrow. Reads tensors lazily per shard; skips all
// routed experts and pinned tensors. Preserves the paired E8M0 scales so the
// already-quantized NVFP4 values are dequantized with their true magnitude.
arrow::Status RepackSharedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       input_path);

    absl::flat_hash_map<int, ExpertData> shared_by_layer;
    int shard_idx = 0;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        spdlog::info("[{}/{}] scanning {} ...", shard_idx, model.shards.size(),
                     shard.filename);

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf) { spdlog::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        if (sz <= 0) { spdlog::warn("  Zero/negative size, skipping"); continue; }
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { spdlog::warn("  Cannot read, skipping"); continue; }

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto sparsed = parse_shared_expert_name(tensor.name);
            if (!sparsed) continue;

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                spdlog::warn("  Skipping {} (offset out of range)", tensor.name);
                continue;
            }
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            auto &sexp = shared_by_layer[sparsed->layer];
            if (sparsed->is_scale) {
                switch (sparsed->weight_id) {
                case 1:
                    sexp.w1_scale_raw   = std::move(raw);
                    sexp.w1_scale_shape = tensor.shape;
                    break;
                case 2:
                    sexp.w2_scale_raw   = std::move(raw);
                    sexp.w2_scale_shape = tensor.shape;
                    break;
                case 3:
                    sexp.w3_scale_raw   = std::move(raw);
                    sexp.w3_scale_shape = tensor.shape;
                    break;
                }
            } else {
                switch (sparsed->weight_id) {
                case 1:
                    sexp.w1_raw   = std::move(raw);
                    sexp.w1_shape = tensor.shape;
                    sexp.w1_dtype = tensor.dtype;
                    break;
                case 2:
                    sexp.w2_raw   = std::move(raw);
                    sexp.w2_shape = tensor.shape;
                    sexp.w2_dtype = tensor.dtype;
                    break;
                case 3:
                    sexp.w3_raw   = std::move(raw);
                    sexp.w3_shape = tensor.shape;
                    sexp.w3_dtype = tensor.dtype;
                    break;
                }
            }
        }
    }

    spdlog::info("Packing {} shared-expert layers (NVFP4)...",
                 shared_by_layer.size());
    return SaveSharedExpertsToArrow(shared_by_layer, output_root);
}

// ── Targeted routed-expert repack (fp4 passthrough) ─────────────────────────

// Re-writes only the routed expert arrows from the source safetensors shards.
// The source weights are already fp4 E2M1 packed (2 elems/byte) with paired
// F8_E8M0 per-32-element scales, so the bytes are copied through unchanged.
// CPU-only: no CUDA quantization is involved.
arrow::Status RepackRoutedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       input_path);

    struct ExpertKey {
        int layer;
        int expert;
        bool operator==(const ExpertKey &o) const {
            return layer == o.layer && expert == o.expert;
        }
    };
    struct ExpertKeyHash {
        std::size_t operator()(const ExpertKey &k) const {
            return static_cast<std::size_t>(k.layer) * 65536U
                 + static_cast<std::size_t>(k.expert);
        }
    };

    absl::flat_hash_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    int shard_idx = 0;
    int total_written = 0;
    int max_layer = 0;
    int max_expert_per_layer = 0;
    absl::flat_hash_map<int, int> experts_per_layer;

    auto try_flush = [&](ExpertKey key, ExpertData &exp) -> arrow::Status {
        if (exp.w1_raw.empty() || exp.w2_raw.empty() || exp.w3_raw.empty())
            return arrow::Status::OK();
        if (exp.w1_scale_raw.empty() || exp.w2_scale_raw.empty() ||
            exp.w3_scale_raw.empty()) {
            spdlog::warn("  expert layer={} expert={} missing E8M0 scales "
                         "(w1={} w2={} w3={}), skipping",
                         key.layer, key.expert,
                         exp.w1_scale_raw.size(), exp.w2_scale_raw.size(),
                         exp.w3_scale_raw.size());
            return arrow::Status::OK();
        }
        ARROW_ASSIGN_OR_RAISE(auto packed, PackExpertNVFP4(std::move(exp)));
        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
            packed, key.layer, key.expert, output_root));
        ++total_written;
        ++experts_per_layer[key.layer];
        max_expert_per_layer = std::max(max_expert_per_layer, key.expert + 1);
        max_layer = std::max(max_layer, key.layer + 1);
        return arrow::Status::OK();
    };

    for (const auto &shard : model.shards) {
        ++shard_idx;
        spdlog::info("[{}/{}] scanning {} ...", shard_idx, model.shards.size(),
                     shard.filename);

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf) { spdlog::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        if (sz <= 0) { spdlog::warn("  Zero/negative size, skipping"); continue; }
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { spdlog::warn("  Cannot read, skipping"); continue; }

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto parsed = parse_expert_name(tensor.name);
            if (!parsed) continue;

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                spdlog::warn("  Skipping {} (offset out of range)", tensor.name);
                continue;
            }
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            ExpertKey key{parsed->layer, parsed->expert};
            auto &exp = partial[key];

            if (parsed->is_scale) {
                switch (parsed->weight_id) {
                case 1:
                    exp.w1_scale_raw   = std::move(raw);
                    exp.w1_scale_shape = tensor.shape;
                    break;
                case 2:
                    exp.w2_scale_raw   = std::move(raw);
                    exp.w2_scale_shape = tensor.shape;
                    break;
                case 3:
                    exp.w3_scale_raw   = std::move(raw);
                    exp.w3_scale_shape = tensor.shape;
                    break;
                }
            } else {
                switch (parsed->weight_id) {
                case 1:
                    exp.w1_raw   = std::move(raw);
                    exp.w1_shape = tensor.shape;
                    exp.w1_dtype = tensor.dtype;
                    break;
                case 2:
                    exp.w2_raw   = std::move(raw);
                    exp.w2_shape = tensor.shape;
                    exp.w2_dtype = tensor.dtype;
                    break;
                case 3:
                    exp.w3_raw   = std::move(raw);
                    exp.w3_shape = tensor.shape;
                    exp.w3_dtype = tensor.dtype;
                    break;
                }
            }

            if (exp.w1_raw.size() && exp.w2_raw.size() && exp.w3_raw.size() &&
                exp.w1_scale_raw.size() && exp.w2_scale_raw.size() &&
                exp.w3_scale_raw.size()) {
                auto st = try_flush(key, exp);
                if (!st.ok())
                    spdlog::error("  Failed to pack expert layer={} expert={}: {}",
                                  key.layer, key.expert, st.ToString());
                partial.erase(key);
            }
        }
    }

    for (auto &[key, exp] : partial) {
        spdlog::warn("  incomplete expert layer={} expert={} (w1={} w2={} w3={})",
                     key.layer, key.expert,
                     exp.w1_raw.size(), exp.w2_raw.size(), exp.w3_raw.size());
    }

    spdlog::info("Repacked {} routed experts ({} layers, up to {} experts/layer)",
                 total_written, max_layer, max_expert_per_layer);
    return arrow::Status::OK();
}

// ── Model index ─────────────────────────────────────────────────────────────

arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
    const VramProfile &vram,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    std::ofstream f(output_root / "model.index.json");
    if (!f)
        return arrow::Status::IOError("Cannot create model.index.json");

    f << "{\n";
    f << "  \"format\": \"logestix-v1\",\n";
    f << "  \"model\": \"" << model_name << "\",\n";
    f << "  \"num_layers\": " << num_layers << ",\n";
    f << "  \"experts_per_layer\": " << experts_per_layer << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << FormatVramJson(vram) << ",\n";
    f << "  \"files\": {\n";
    f << "    \"pinned\": {\n";
    f << "      \"global\": \"pinned/global.arrow\"\n";
    for (int i = 0; i < num_layers; ++i) {
        f << "      ,\"layer_" << std::setw(3) << std::setfill('0') << i
          << "\": \"pinned/layer_" << std::setw(3) << std::setfill('0') << i << ".arrow\"\n";
    }
    f << "    },\n";
    f << "    \"experts\": {\n";
    for (int i = 0; i < num_layers; ++i) {
        f << "      \"layer_" << std::setw(3) << std::setfill('0') << i
          << "\": \"experts/layer_" << std::setw(3) << std::setfill('0') << i << "/\"";
        if (i < num_layers - 1) f << ",";
        f << "\n";
    }
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    f.close();

    spdlog::info("Wrote model.index.json");
    return arrow::Status::OK();
}

// ── Converter implementation ────────────────────────────────────────────────

struct Converter::Impl {
    std::string input_path;
    std::string output_path;
    bool verbose = false;
};

Converter::Converter()
    : impl_(std::make_unique<Impl>()) {}

Converter::~Converter() = default;
Converter::Converter(Converter &&) noexcept = default;
Converter &Converter::operator=(Converter &&) noexcept = default;

Converter &Converter::SetInputPath(const std::string &path)
{
    impl_->input_path = path;
    return *this;
}

Converter &Converter::SetOutputPath(const std::string &path)
{
    impl_->output_path = path;
    return *this;
}

Converter &Converter::SetVerbose(bool v)
{
    impl_->verbose = v;
    return *this;
}

arrow::Status Converter::Run()
{
    namespace fs = std::filesystem;

    if (impl_->input_path.empty())
        return arrow::Status::Invalid("input path not set");
    if (impl_->output_path.empty())
        return arrow::Status::Invalid("output path not set");

    fs::create_directories(impl_->output_path);

    auto model = lsxcommon::scan_directory(impl_->input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       impl_->input_path);

    // (2) config.json is mandatory; parsed for geometry cross-checks.
    auto cfg_path = fs::path(impl_->input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ",
                                      impl_->input_path);

    int cfg_num_layers = 0;
    int cfg_hidden_size = 0;
    int cfg_vocab_size = 0;
    int cfg_experts_per_layer = 0;
    {
        std::ifstream cf(cfg_path);
        if (!cf)
            return arrow::Status::IOError("cannot open ", cfg_path.string());
        std::string cj((std::istreambuf_iterator<char>(cf)),
                       std::istreambuf_iterator<char>());
        simdjson::padded_string cpj(cj.data(), cj.size());
        simdjson::ondemand::parser cp;
        simdjson::ondemand::document cd;
        if (cp.iterate(cpj).get(cd) != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json parse failed");
        auto root = cd.get_object();
        if (root.error() != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json root not an object");
        auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
            int64_t v;
            if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
            return static_cast<int>(v);
        };
        cfg_num_layers = get_i(root.value(), "num_hidden_layers", 0);
        if (cfg_num_layers <= 0)
            cfg_num_layers = get_i(root.value(), "num_layers", 0);
        cfg_hidden_size = get_i(root.value(), "hidden_size", 0);
        cfg_vocab_size = get_i(root.value(), "vocab_size", 0);
        cfg_experts_per_layer = get_i(root.value(), "num_local_experts", 0);
    }

    // (3) Verify every tensor read is complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(
        impl_->input_path, model));

    auto total_shards = model.shards.size();
    spdlog::info("Found {} shards, {} tensors across {} shards",
        total_shards,
        [&]{
            std::size_t n = 0;
            for (auto &s : model.shards) n += s.tensors.size();
            return n;
        }(),
        total_shards);

    absl::flat_hash_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    std::vector<TensorRecord> pinned_tensors;
    absl::flat_hash_map<int, ExpertData> shared_by_layer;
    int experts_processed = 0;
    int pinned_processed = 0;
    int shard_idx = 0;
    int max_layer = 0;
    int max_experts_per_layer = 0;
    absl::flat_hash_map<int, int> experts_per_layer;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        auto t0 = std::chrono::steady_clock::now();

        spdlog::info("[{}/{}] Scanning {} (pinned={}, experts={})",
            shard_idx, total_shards, shard.filename,
            pinned_processed, partial.size());

        std::ifstream sf(
            (fs::path(impl_->input_path) / shard.filename).string(),
            std::ios::binary | std::ios::ate);
        if (!sf)
            return arrow::Status::IOError("cannot open ", shard.filename);
        int64_t sz = sf.tellg();
        if (sz <= 0)
            return arrow::Status::IOError("zero/negative size: ",
                                          shard.filename);
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf)
            return arrow::Status::IOError("cannot read ", shard.filename);

        spdlog::info("  read {} MB", sz / (1024 * 1024));

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        int experts_in_shard = 0;
        for (const auto &tensor : shard.tensors) {
            auto parsed = parse_expert_name(tensor.name);

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz)
                return arrow::Status::Invalid(
                    shard.filename, " tensor ", tensor.name,
                    " offset ", data_off, "+", tensor.data_begin,
                    " exceeds file size ", sz);

            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed) {
                ExpertKey key{parsed->layer, parsed->expert};
                auto &exp = partial[key];
                max_layer = std::max(max_layer, parsed->layer + 1);

                if (parsed->is_scale) {
                    switch (parsed->weight_id) {
                    case 1:
                        exp.w1_scale_raw   = std::move(raw);
                        exp.w1_scale_shape = tensor.shape;
                        break;
                    case 2:
                        exp.w2_scale_raw   = std::move(raw);
                        exp.w2_scale_shape = tensor.shape;
                        break;
                    case 3:
                        exp.w3_scale_raw   = std::move(raw);
                        exp.w3_scale_shape = tensor.shape;
                        break;
                    }
                } else {
                    switch (parsed->weight_id) {
                    case 1:
                        exp.w1_raw   = std::move(raw);
                        exp.w1_shape = tensor.shape;
                        exp.w1_dtype = tensor.dtype;
                        break;
                    case 2:
                        exp.w2_raw   = std::move(raw);
                        exp.w2_shape = tensor.shape;
                        exp.w2_dtype = tensor.dtype;
                        break;
                    case 3:
                        exp.w3_raw   = std::move(raw);
                        exp.w3_shape = tensor.shape;
                        exp.w3_dtype = tensor.dtype;
                        break;
                    }
                }

                if (!exp.w1_raw.empty() && !exp.w2_raw.empty() && !exp.w3_raw.empty()) {
                    ++experts_in_shard;
                    static bool first_expert = true;
                    if (first_expert) {
                        spdlog::debug("[diag] Converter: processing first expert "
                                      "(CUDA will be initialized here)");
                        diag::LogArrowSystemState("converter-before-first-expert");
                        first_expert = false;
                    }
                    auto status = [&]() -> arrow::Status {
                        ARROW_ASSIGN_OR_RAISE(auto packed, PackExpertNVFP4(std::move(exp)));
                        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                            packed, key.layer, key.expert,
                            fs::path(impl_->output_path)));
                        return arrow::Status::OK();
                    }();
                    if (!status.ok())
                        spdlog::error("  Failed to pack expert layer={} expert={}: {}",
                            key.layer, key.expert, status.ToString());
                    else {
                        ++experts_processed;
                        ++experts_per_layer[parsed->layer];
                    }
                    partial.erase(key);
                }
            } else {
                auto sparsed = parse_shared_expert_name(tensor.name);
                if (sparsed) {
                    auto &sexp = shared_by_layer[sparsed->layer];
                    max_layer = std::max(max_layer, sparsed->layer + 1);
                    if (sparsed->is_scale) {
                        switch (sparsed->weight_id) {
                        case 1:
                            sexp.w1_scale_raw   = std::move(raw);
                            sexp.w1_scale_shape = tensor.shape;
                            break;
                        case 2:
                            sexp.w2_scale_raw   = std::move(raw);
                            sexp.w2_scale_shape = tensor.shape;
                            break;
                        case 3:
                            sexp.w3_scale_raw   = std::move(raw);
                            sexp.w3_scale_shape = tensor.shape;
                            break;
                        }
                    } else {
                        switch (sparsed->weight_id) {
                        case 1:
                            sexp.w1_raw   = std::move(raw);
                            sexp.w1_shape = tensor.shape;
                            sexp.w1_dtype = tensor.dtype;
                            break;
                        case 2:
                            sexp.w2_raw   = std::move(raw);
                            sexp.w2_shape = tensor.shape;
                            sexp.w2_dtype = tensor.dtype;
                            break;
                        case 3:
                            sexp.w3_raw   = std::move(raw);
                            sexp.w3_shape = tensor.shape;
                            sexp.w3_dtype = tensor.dtype;
                            break;
                        }
                    }
                    continue;   // packed separately into shared_experts.arrow
                }

                TensorRecord t;
                t.name  = tensor.name;
                t.dtype = tensor.dtype;
                t.shape = tensor.shape;
                t.data  = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        spdlog::info("  done in {:0.1f}s — {} experts in this shard",
            elapsed / 1000.0, experts_in_shard);
    }

    for (auto &[key, raw] : partial)
        return arrow::Status::Invalid(
            "incomplete expert layer=", key.layer, " expert=", key.expert,
            " (w1=", raw.w1_raw.size(), " w2=", raw.w2_raw.size(),
            " w3=", raw.w3_raw.size(), ")");

    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    // (6) Geometry cross-check: config-declared vs tensor-derived.
    {
        int embed_vocab = 0, embed_hidden = 0;
        for (auto &shard : model.shards) {
            for (auto &t : shard.tensors) {
                // DeepSeek sources name the embedding "embed.weight"
                // (no "model." prefix); accept both spellings.
                if ((t.name == "model.embed_tokens.weight" ||
                     t.name == "embed.weight") && t.shape.size() >= 2) {
                    embed_vocab = static_cast<int>(t.shape[0]);
                    embed_hidden = static_cast<int>(t.shape[1]);
                }
            }
        }
        std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> geom;
        if (cfg_num_layers > 0)
            geom.push_back({"num_layers", {cfg_num_layers, max_layer}});
        if (cfg_hidden_size > 0)
            geom.push_back({"hidden_size", {cfg_hidden_size, embed_hidden}});
        if (cfg_vocab_size > 0)
            geom.push_back({"vocab_size", {cfg_vocab_size, embed_vocab}});
        if (cfg_experts_per_layer > 0)
            geom.push_back({"experts_per_layer",
                            {cfg_experts_per_layer, max_experts_per_layer}});
        if (!geom.empty())
            ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields(geom));
    }

    spdlog::info("Saving {} pinned tensors...", pinned_tensors.size());
    ARROW_RETURN_NOT_OK(SavePinnedTensors(pinned_tensors, impl_->output_path));

    spdlog::info("Packing {} shared-expert layers (NVFP4)...",
        shared_by_layer.size());
    ARROW_RETURN_NOT_OK(SaveSharedExpertsToArrow(shared_by_layer, impl_->output_path));

    // Conversion-time VRAM profile (spec 2026-08-21-vram-profile-design.md).
    VramGeometry vg;
    vg.num_attn_layers = max_layer;
    vg.num_moe_layers = max_layer;
    vg.experts_per_layer = max_experts_per_layer;
    vg.context_length = ReadContextLength(
        (fs::path(impl_->input_path) / "config.json").string());
    // Per-layer KV compression (DeepSeek-V4 CSA): stored tokens = ctx/ratio.
    {
        std::ifstream cf((fs::path(impl_->input_path) / "config.json"));
        if (cf) {
            std::string cj((std::istreambuf_iterator<char>(cf)),
                           std::istreambuf_iterator<char>());
            simdjson::ondemand::parser cp;
            simdjson::padded_string cpj(cj.data(), cj.size());
            simdjson::ondemand::document cd;
            if (cp.iterate(cpj).get(cd) == simdjson::SUCCESS) {
                auto root = cd.get_object();
                if (root.error() == simdjson::SUCCESS) {
                    if (auto arr = root.value()["compress_ratios"].get_array();
                        arr.error() == simdjson::SUCCESS) {
                        for (auto v : arr.value()) {
                            int64_t x;
                            if (v.get_int64().get(x) == simdjson::SUCCESS)
                                vg.kv_compress_ratios.push_back(x);
                        }
                    }
                }
            }
        }
    }
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (vg.kv_dim == 0 && t.shape.size() >= 2 &&
                t.name.find("wkv.weight") != std::string::npos) {
                vg.kv_dim = static_cast<int>(t.shape[0]);
                vg.hidden_dim = static_cast<int>(t.shape[1]);
            }
        }
    }

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});

    VramProfile vram_profile = EstimateVram(
        vram_inventory, {&DsIsNvfp4Tensor, &DsMemClass}, vg);
    spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
        FormatBytesHuman(vram_profile.min_bytes),
        FormatBytesHuman(vram_profile.optimal_bytes),
        FormatBytesHuman(vram_profile.max_bytes),
        vram_profile.context_tokens);

    ARROW_RETURN_NOT_OK(WriteModelIndex(
        fs::path(impl_->input_path).filename().string(),
        max_layer, max_experts_per_layer,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        vram_profile,
        impl_->output_path));

    // (8) Build and verify the binary tokenizer (mandatory).
    auto tokenizer_json = fs::path(impl_->input_path) / "tokenizer.json";
    if (!fs::exists(tokenizer_json))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      impl_->input_path);
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tokenizer_json.string(),
        fs::path(impl_->output_path) / "pinned",
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true}));

    // (9) The converted output must be a loadable, verified model.
    auto vres = integrity::VerifyConvertedModel(impl_->output_path);
    if (!vres.ok())
        return arrow::Status::Invalid(
            "output verification failed: ", vres.status().ToString());

    spdlog::info("Done — {} pinned tensors, {} experts processed, incomplete {}",
        pinned_processed, experts_processed, partial.size());
    return arrow::Status::OK();
}

} // namespace lsxcommon
