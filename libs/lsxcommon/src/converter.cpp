#include "lsxcommon/converter.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
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
#include <unordered_map>
#include <filesystem>
#include <map>

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
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GiB";
    if (bytes >= 1024 * 1024)
        return std::to_string(bytes / (1024 * 1024)) + " MiB";
    if (bytes >= 1024)
        return std::to_string(bytes / 1024) + " KiB";
    return std::to_string(bytes) + " B";
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
};

std::optional<ParsedName> parse_expert_name(const std::string &name)
{
    static const std::regex re(
        R"((?:model\.)?layers\.(\d+)\.ffn\.experts\.(\d+)\.w(\d)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedName p;
    p.layer     = std::stoi(m[1]);
    p.expert    = std::stoi(m[2]);
    p.weight_id = std::stoi(m[3]);
    return p;
}

} // anonymous namespace

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
        std::string shape_str = "[";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i > 0) shape_str += ", ";
            shape_str += std::to_string(t.shape[i]);
        }
        shape_str += "]";

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
    std::map<std::string, int64_t> cat_tensors, cat_bytes;
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

    spdlog::info("--- Runtime memory estimate ---");

    struct LayerExpertInfo {
        int64_t total_expert_bytes = 0;
        int expert_count = 0;
    };
    std::map<int, LayerExpertInfo> moe_layers;
    std::map<int, bool> layers_with_attention;
    int hidden_dim = 0;
    int kv_latent_dim = 0;
    int total_layers = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            int layer_num = -1;
            if (t.name.rfind("layers.", 0) == 0) {
                auto dot = t.name.find('.', 7);
                if (dot != std::string::npos) {
                    layer_num = std::stoi(t.name.substr(7, dot - 7));
                    total_layers = std::max(total_layers, layer_num + 1);
                }
            }

            if (is_staged_tensor(t.name) && layer_num >= 0) {
                moe_layers[layer_num].expert_count++;
                moe_layers[layer_num].total_expert_bytes += t.byte_size();
            }

            if (layer_num >= 0 && t.name.find(".attn.") != std::string::npos && t.name.find(".weight") != std::string::npos) {
                layers_with_attention[layer_num] = true;
                if (t.name.find("wkv.weight") != std::string::npos) {
                    kv_latent_dim = static_cast<int>(t.shape[0]);
                    hidden_dim = static_cast<int>(t.shape[1]);
                }
            }
        }
    }

    int num_moe_layers = static_cast<int>(moe_layers.size());
    int num_attn_layers = static_cast<int>(layers_with_attention.size());
    int experts_per_layer = 0;
    int64_t per_expert_bytes = 0;
    if (!moe_layers.empty()) {
        for (auto &[_, info] : moe_layers) {
            experts_per_layer = std::max(experts_per_layer, info.expert_count);
        }
        int64_t total_expert_bytes = 0;
        for (auto &[_, info] : moe_layers)
            total_expert_bytes += info.total_expert_bytes;
        int64_t total_experts = static_cast<int64_t>(num_moe_layers) * experts_per_layer;
        if (total_experts > 0)
            per_expert_bytes = total_expert_bytes / total_experts;
    }

    if (hidden_dim == 0) hidden_dim = 4096;
    if (kv_latent_dim == 0) kv_latent_dim = 512;
    if (num_attn_layers == 0) num_attn_layers = total_layers;
    if (num_moe_layers == 0) num_moe_layers = total_layers;

    const int ring_buffer_size = 1;
    const int seq_len = 2048;
    const int batch_size = 1;

    int64_t expert_ring_buffer = static_cast<int64_t>(ring_buffer_size) *
                                 per_expert_bytes * num_moe_layers;
    int bytes_per_elem = 2;
    int64_t kv_cache = static_cast<int64_t>(seq_len) * num_attn_layers *
                       kv_latent_dim * 2 * bytes_per_elem;
    int64_t workspace = static_cast<int64_t>(batch_size) * seq_len * hidden_dim *
                        num_attn_layers * bytes_per_elem * 2;

    spdlog::info("  Expert Ring Buffer (ring_size={}): {}",
        ring_buffer_size, format_size(expert_ring_buffer));
    spdlog::info("  KV Cache ({} layers, seq_len={}):  {}",
        num_attn_layers, seq_len, format_size(kv_cache));
    spdlog::info("  Workspace (batch={}, seq_len={}):  {}",
        batch_size, seq_len, format_size(workspace));

    int64_t runtime_total = pinned_bytes + expert_ring_buffer + kv_cache + workspace;
    spdlog::info("  Total runtime (pinned+ring+cache+ws): {}",
        format_size(runtime_total));
    int64_t gpu_min = pinned_bytes + expert_ring_buffer + kv_cache;
    spdlog::info("  Minimum VRAM (pinned+ring+cache): {}",
        format_size(gpu_min));
}

void print_nvfp4_forecast(const SafetensorsModel &model)
{
    struct CatStats {
        int64_t orig_bytes = 0;
        int64_t nvfp4_bytes = 0;
        int count = 0;
    };
    std::map<std::string, CatStats> cats;

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
                        std::vector<uint8_t> &packed,
                        std::vector<uint8_t> &scales,
                        float &global_scale) -> bool
    {
        int64_t numel = product(shape);
        if (numel == 0 || src.empty()) return false;
        return gpu::quantize_tensor_gpu(src.data(), numel, dtype,
                                         packed, scales, global_scale);
    };

    if (!process(raw.w1_raw, raw.w1_dtype, raw.w1_shape,
                  pe.w1_packed, pe.w1_scales, pe.w1_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w1");

    if (!process(raw.w2_raw, raw.w2_dtype, raw.w2_shape,
                  pe.w2_packed, pe.w2_scales, pe.w2_global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for w2");

    if (!process(raw.w3_raw, raw.w3_dtype, raw.w3_shape,
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
        ("layer_" + std::to_string(layer_idx)) / "experts";
    fs::create_directories(layer_dir);

    auto out_path = layer_dir /
        ("exp_" + std::to_string(expert_idx) + ".arrow");

    ARROW_ASSIGN_OR_RAISE(auto batch, ExpertSchema::MakeBatch(expert));
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

    std::unordered_map<std::string, std::vector<TensorRecord>> by_layer;
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
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors -> {}", global.size(), path.string());
    }

    for (auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned" / (layer + ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
        spdlog::info("  wrote {} pinned tensors for {} -> {}",
            tens.size(), layer, path.string());
    }

    return arrow::Status::OK();
}

// ── Model index ─────────────────────────────────────────────────────────────

arrow::Status WriteModelIndex(
    const std::string &model_name,
    int num_layers,
    int experts_per_layer,
    int64_t total_pinned,
    int64_t total_experts,
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
          << "\": \"experts/layer_" << std::setw(3) << std::setfill('0') << i << "/\"\n";
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

    auto total_shards = model.shards.size();
    spdlog::info("Found {} shards, {} tensors across {} shards",
        total_shards,
        [&]{
            std::size_t n = 0;
            for (auto &s : model.shards) n += s.tensors.size();
            return n;
        }(),
        total_shards);

    std::unordered_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    std::vector<TensorRecord> pinned_tensors;
    int experts_processed = 0;
    int pinned_processed = 0;
    int shard_idx = 0;
    int max_layer = 0;
    int max_experts_per_layer = 0;
    std::unordered_map<int, int> experts_per_layer;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        auto t0 = std::chrono::steady_clock::now();

        spdlog::info("[{}/{}] Scanning {} (pinned={}, experts={})",
            shard_idx, total_shards, shard.filename,
            pinned_processed, partial.size());

        std::ifstream sf(
            (fs::path(impl_->input_path) / shard.filename).string(),
            std::ios::binary | std::ios::ate);
        if (!sf) { spdlog::warn("  Cannot open, skipping"); continue; }
        int64_t sz = sf.tellg();
        if (sz <= 0) { spdlog::warn("  Zero or negative file size, skipping"); continue; }
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf) { spdlog::warn("  Cannot read, skipping"); continue; }

        spdlog::info("  read {} MB", sz / (1024 * 1024));

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        int experts_in_shard = 0;
        for (const auto &tensor : shard.tensors) {
            auto parsed = parse_expert_name(tensor.name);

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz) {
                spdlog::warn("  Skipping {} (offset {}+{} exceeds file size {})",
                    tensor.name, data_off, tensor.data_begin, sz);
                continue;
            }

            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed) {
                ExpertKey key{parsed->layer, parsed->expert};
                auto &exp = partial[key];
                max_layer = std::max(max_layer, parsed->layer + 1);
                experts_per_layer[parsed->layer]++;

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

                if (!exp.w1_raw.empty() && !exp.w2_raw.empty() && !exp.w3_raw.empty()) {
                    ++experts_in_shard;
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
                    else
                        ++experts_processed;
                    partial.erase(key);
                }
            } else {
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

    for (auto &[key, raw] : partial) {
        spdlog::warn("  Incomplete expert layer={} expert={} (w1={} w2={} w3={})",
            key.layer, key.expert,
            raw.w1_raw.size(), raw.w2_raw.size(), raw.w3_raw.size());
    }

    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    spdlog::info("Saving {} pinned tensors...", pinned_tensors.size());
    ARROW_RETURN_NOT_OK(SavePinnedTensors(pinned_tensors, impl_->output_path));

    ARROW_RETURN_NOT_OK(WriteModelIndex(
        fs::path(impl_->input_path).filename().string(),
        max_layer, max_experts_per_layer,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        impl_->output_path));

    spdlog::info("Done — {} pinned tensors, {} experts processed, incomplete {}",
        pinned_processed, experts_processed, partial.size());
    return arrow::Status::OK();
}

} // namespace lsxcommon
