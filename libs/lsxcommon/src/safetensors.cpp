#include "lsxcommon/safetensors.h"
#ifdef LOGESTIX_HAS_CUDA
#include "lsxcommon/safetensors_gpu.h"
#endif
#include <spdlog/spdlog.h>
#include <simdjson.h>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <map>
#include <cstring>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace lsxcommon {

namespace {

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

} // anonymous namespace

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

bool is_staged_tensor(const std::string &name)
{
    return name.find(".ffn.experts.") != std::string::npos;
}

std::string pinned_category_name(const std::string &name)
{
    // Order matters: check more specific patterns first
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
    // Collect pinned tensor categories
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

    // Collect architecture info from tensors
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
        // Average per-expert size: total / (num_moe_layers * experts_per_layer)
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

    int bytes_per_elem = 2; // BF16 default for compute
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

namespace {

float read_fp32(const uint8_t *p) { float v; std::memcpy(&v, p, 4); return v; }

float read_bf16(const uint8_t *p) {
    uint16_t u;
    std::memcpy(&u, p, 2);
    uint32_t f32 = static_cast<uint32_t>(u) << 16;
    float v;
    std::memcpy(&v, &f32, 4);
    return v;
}

float read_f8e4m3(uint8_t b) {
    if (!b || b == 0x80) return 0.0f;
    int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0x0F;
    int mant = b & 0x07;
    float v = (1.0f + mant / 8.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f);
    return sign ? -v : v;
}

float read_f8e8m0(uint8_t b) {
    if (!b) return 0.0f;
    return std::pow(2.0f, static_cast<float>(b) - 127.0f);
}

float read_i8(uint8_t b) {
    return static_cast<float>(static_cast<int8_t>(b));
}

uint8_t quantize_nvfp4(float val, float scale) {
    float scaled = val / scale;
    if (scaled == 0.0f) return 0;
    int sign = scaled < 0 ? 1 : 0;
    float abs_val = std::fabs(scaled);
    if (abs_val >= 8.0f) abs_val = 7.999f;
    static const float levels[8] = {
        0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f
    };
    int best = 0;
    float best_diff = std::fabs(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        float d = std::fabs(abs_val - levels[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

static bool cpu_quantize_tensor(const uint8_t *tdata, int64_t n,
                                const std::string &dtype,
                                std::vector<uint8_t> &packed, float &scale)
{
    float max_abs = 0.0f;
    std::vector<float> fvals(n);

    if (dtype == "F32") {
        for (int64_t i = 0; i < n; i++) {
            fvals[i] = read_fp32(tdata + i * 4);
            max_abs = std::max(max_abs, std::fabs(fvals[i]));
        }
    } else if (dtype == "BF16") {
        for (int64_t i = 0; i < n; i++) {
            fvals[i] = read_bf16(tdata + i * 2);
            max_abs = std::max(max_abs, std::fabs(fvals[i]));
        }
    } else if (dtype == "F8_E4M3") {
        for (int64_t i = 0; i < n; i++) {
            fvals[i] = read_f8e4m3(tdata[i]);
            max_abs = std::max(max_abs, std::fabs(fvals[i]));
        }
    } else if (dtype == "F8_E8M0") {
        for (int64_t i = 0; i < n; i++) {
            fvals[i] = read_f8e8m0(tdata[i]);
            max_abs = std::max(max_abs, std::fabs(fvals[i]));
        }
    } else if (dtype == "I8" || dtype == "U8") {
        for (int64_t i = 0; i < n; i++) {
            fvals[i] = read_i8(tdata[i]);
            max_abs = std::max(max_abs, std::fabs(fvals[i]));
        }
    } else {
        return false;
    }

    if (max_abs == 0.0f) max_abs = 1.0f;
    scale = max_abs / 8.0f;

    int64_t npacked = (n + 1) / 2;
    packed.resize(npacked);
    for (int64_t i = 0; i < n; i += 2) {
        uint8_t lo = quantize_nvfp4(fvals[i], scale);
        uint8_t hi = (i + 1 < n) ? quantize_nvfp4(fvals[i + 1], scale) : 0;
        packed[i / 2] = (hi << 4) | lo;
    }
    return true;
}

} // anonymous namespace

void convert_to_nvfp4(const SafetensorsModel &model, const std::string &model_path)
{
    using namespace simdjson;

    for (size_t si = 0; si < model.shards.size(); si++) {
        const auto &shard = model.shards[si];
        std::string in_path = model_path + "/" + shard.filename;
        spdlog::info("Converting: {}...", shard.filename);

        std::ifstream in(in_path, std::ios::binary | std::ios::ate);
        if (!in) { spdlog::error("  Cannot open"); continue; }
        int64_t file_size = static_cast<int64_t>(in.tellg());
        in.seekg(0);
        std::vector<uint8_t> buf(file_size);
        if (!in.read(reinterpret_cast<char*>(buf.data()), file_size)) {
            spdlog::error("  Cannot read"); continue;
        }

        uint64_t header_len;
        std::memcpy(&header_len, buf.data(), 8);
        std::string json_buf(reinterpret_cast<const char*>(buf.data() + 8), header_len);

        ondemand::parser parser;
        ondemand::document doc = parser.iterate(json_buf);
        ondemand::object root = doc.get_object();

        std::string metadata_json;

        struct QTensor {
            std::string name;
            std::string orig_dtype;
            std::vector<int64_t> shape;
            std::vector<uint8_t> packed;
            float scale;
        };
        std::vector<QTensor> qtensors;

        for (auto json_field : root) {
            std::string_view key_sv;
            if (json_field.unescaped_key().get(key_sv) != SUCCESS) break;

            ondemand::value val;
            if (json_field.value().get(val) != SUCCESS) continue;

            if (key_sv == "__metadata__") {
                ondemand::object meta_obj;
                if (val.get_object().get(meta_obj) == SUCCESS) {
                    std::ostringstream ms;
                    ms << meta_obj;
                    metadata_json = ms.str();
                }
                continue;
            }

            ondemand::object tobj;
            if (val.get_object().get(tobj) != SUCCESS) continue;

            std::string_view dtype_view;
            if (tobj["dtype"].get_string().get(dtype_view) != SUCCESS) continue;
            std::string dtype(dtype_view);

            ondemand::array sarr;
            if (tobj["shape"].get_array().get(sarr) != SUCCESS) continue;
            std::vector<int64_t> shape;
            for (auto d : sarr) shape.push_back(static_cast<int64_t>(d));

            ondemand::array oarr;
            if (tobj["data_offsets"].get_array().get(oarr) != SUCCESS) continue;
            int64_t db = 0, de = 0;
            int idx = 0;
            for (auto off_val : oarr) {
                int64_t v = 0;
                if (off_val.get_int64().get(v) == SUCCESS) {
                    if (idx == 0) db = v;
                    else if (idx == 1) de = v;
                }
                idx++;
            }

            int64_t n = 1;
            for (auto d : shape) n *= d;

            const uint8_t *tdata = buf.data() + 8 + header_len + db;

            std::vector<uint8_t> packed;
            float scale;

            bool ok = false;
            if (dtype != "F16") {
#ifdef LOGESTIX_HAS_CUDA
                ok = gpu::quantize_tensor_gpu(tdata, n, dtype, packed, scale);
#endif
                if (!ok)
                    ok = cpu_quantize_tensor(tdata, n, dtype, packed, scale);
            }

            if (!ok) {
                if (dtype == "F16")
                    spdlog::warn("  Skipping tensor {} (F16 not supported)", key_sv);
                else
                    spdlog::warn("  Skipping tensor {} (dtype: {})", key_sv, dtype);
                continue;
            }

            qtensors.push_back({json_escape(std::string(key_sv)), dtype, std::move(shape), std::move(packed), scale});
        }

        std::ostringstream hdr;
        hdr << std::setprecision(9) << "{";
        int64_t data_off = 0;
        for (size_t i = 0; i < qtensors.size(); i++) {
            if (i > 0) hdr << ",";
            hdr << "\"" << qtensors[i].name << "\":{";
            hdr << "\"dtype\":\"NVFP4\",";
            hdr << "\"shape\":[";
            for (size_t d = 0; d < qtensors[i].shape.size(); d++) {
                if (d > 0) hdr << ",";
                hdr << qtensors[i].shape[d];
            }
            hdr << "],";
            int64_t psize = static_cast<int64_t>(qtensors[i].packed.size());
            hdr << "\"data_offsets\":[" << data_off << "," << (data_off + psize) << "],";
            hdr << "\"scale\":" << qtensors[i].scale << ",";
            hdr << "\"original_dtype\":\"" << qtensors[i].orig_dtype << "\"";
            hdr << "}";
            data_off += psize;
        }
        if (!metadata_json.empty()) {
            hdr << ",\"__metadata__\":" << metadata_json;
        }
        hdr << "}";
        std::string hdr_s = hdr.str();
        uint64_t hdr_len = static_cast<uint64_t>(hdr_s.size());

        std::string out_name = "nvfp4_" + shard.filename;
        std::string out_path = model_path + "/" + out_name;
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&hdr_len), 8);
        out.write(hdr_s.data(), hdr_s.size());
        for (auto &qt : qtensors)
            out.write(reinterpret_cast<const char*>(qt.packed.data()), qt.packed.size());

        double pct = 100.0 * static_cast<double>(8 + hdr_s.size() + data_off) / file_size;
        spdlog::info("  {} -> {} ({:.1f}%)", shard.filename, out_name, pct);
    }

    std::string idx_path = model_path + "/model.safetensors.index.json";
    if (std::filesystem::exists(idx_path)) {
        spdlog::info("Converting index file...");
        std::ifstream idx_in(idx_path);
        std::string idx_src((std::istreambuf_iterator<char>(idx_in)),
                             std::istreambuf_iterator<char>());

        ondemand::parser ip;
        ondemand::document idoc = ip.iterate(idx_src);
        ondemand::object iroot = idoc.get_object();

        std::string meta_str;
        auto meta_val = iroot["metadata"];
        ondemand::object meta_obj;
        if (meta_val.get_object().get(meta_obj) == SUCCESS) {
            std::ostringstream ms;
            ms << meta_obj;
            meta_str = ms.str();
        }

        ondemand::object wm;
        if (iroot["weight_map"].get_object().get(wm) != SUCCESS) return;

        std::ostringstream out_idx;
        out_idx << "{\"metadata\":";
        if (meta_str.empty()) out_idx << "{}";
        else out_idx << meta_str;
        out_idx << ",\"weight_map\":{";
        bool first = true;
        for (auto entry : wm) {
            if (!first) out_idx << ",";
            first = false;
            std::string_view k;
            std::string_view v;
            entry.unescaped_key().get(k);
            entry.value().get_string().get(v);
            out_idx << "\"" << json_escape(std::string(k)) << "\":\"nvfp4_" << json_escape(std::string(v)) << "\"";
        }
        out_idx << "}}";

        std::string out_idx_path = model_path + "/nvfp4_model.safetensors.index.json";
        std::ofstream out_idxf(out_idx_path);
        out_idxf << out_idx.str();
        spdlog::info("  index: nvfp4_model.safetensors.index.json");
    }
}

} // namespace lsxcommon
