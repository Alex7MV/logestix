#include "lsxcommon/qwen_convert.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <regex>

namespace lsxcommon::qwen {

// ── Classifier ──────────────────────────────────────────────────────────────

namespace {

std::optional<ParsedTensor> match_layer(
    const std::string &name, const std::regex &re, TensorKind kind)
{
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = kind;
    p.layer = std::stoi(m[1]);
    return p;
}

std::string layer_regex(const std::string &suffix)
{
    return R"((?:model\.)?language_model\.layers\.(\d+)\.)" + suffix;
}

} // namespace

ParsedTensor classify_qwen_tensor(const std::string &name)
{
    using K = TensorKind;

    if (std::regex_match(name, std::regex(R"(model\.language_model\.embed_tokens\.weight$)")))
        return {K::Embed, -1};
    if (name == "lm_head.weight") return {K::LmHead, -1};
    if (std::regex_match(name, std::regex(R"(model\.language_model\.norm\.weight$)")))
        return {K::FinalNorm, -1};

    if (name.rfind("mtp.", 0) == 0) return {K::MTP, -1};
    if (name.rfind("visual.", 0) == 0 || name.rfind("model.visual.", 0) == 0)
        return {K::Visual, -1};

    auto P = [&](K kind, const std::string &suffix) -> std::optional<ParsedTensor> {
        return match_layer(name, std::regex(layer_regex(suffix)), kind);
    };

    if (auto p = P(K::LinearAttnQkv,   R"(linear_attn\.in_proj_qkv\.weight$)"))   return *p;
    if (auto p = P(K::LinearAttnZ,     R"(linear_attn\.in_proj_z\.weight$)"))     return *p;
    if (auto p = P(K::LinearAttnB,     R"(linear_attn\.in_proj_b\.weight$)"))     return *p;
    if (auto p = P(K::LinearAttnA,     R"(linear_attn\.in_proj_a\.weight$)"))     return *p;
    if (auto p = P(K::LinearAttnOut,   R"(linear_attn\.out_proj\.weight$)"))      return *p;
    if (auto p = P(K::LinearAttnConv,  R"(linear_attn\.conv1d\.weight$)"))        return *p;
    if (auto p = P(K::LinearAttnNorm,  R"(linear_attn\.norm\.weight$)"))          return *p;
    if (auto p = P(K::LinearAttnA_log, R"(linear_attn\.A_log$)"))                 return *p;
    if (auto p = P(K::LinearAttnDt,    R"(linear_attn\.dt_bias$)"))               return *p;

    if (auto p = P(K::FullAttnQ,       R"(self_attn\.q_proj\.weight$)"))          return *p;
    if (auto p = P(K::FullAttnK,       R"(self_attn\.k_proj\.weight$)"))          return *p;
    if (auto p = P(K::FullAttnV,       R"(self_attn\.v_proj\.weight$)"))          return *p;
    if (auto p = P(K::FullAttnO,       R"(self_attn\.o_proj\.weight$)"))          return *p;
    if (auto p = P(K::FullAttnQNorm,   R"(self_attn\.q_norm\.weight$)"))          return *p;
    if (auto p = P(K::FullAttnKNorm,   R"(self_attn\.k_norm\.weight$)"))          return *p;

    if (auto p = P(K::MlpGate,         R"(mlp\.gate_proj\.weight$)"))             return *p;
    if (auto p = P(K::MlpUp,           R"(mlp\.up_proj\.weight$)"))               return *p;
    if (auto p = P(K::MlpDown,         R"(mlp\.down_proj\.weight$)"))             return *p;
    if (auto p = P(K::InputNorm,       R"(input_layernorm\.weight$)"))            return *p;
    if (auto p = P(K::PostNorm,        R"(post_attention_layernorm\.weight$)"))   return *p;

    return {K::Unknown, -1};
}

bool qwen_tensor_is_nvfp4(TensorKind kind)
{
    switch (kind) {
    case TensorKind::Embed:
    case TensorKind::LmHead:
    case TensorKind::LinearAttnQkv:
    case TensorKind::LinearAttnZ:
    case TensorKind::LinearAttnB:
    case TensorKind::LinearAttnA:
    case TensorKind::LinearAttnOut:
    case TensorKind::FullAttnQ:
    case TensorKind::FullAttnK:
    case TensorKind::FullAttnV:
    case TensorKind::FullAttnO:
    case TensorKind::MlpGate:
    case TensorKind::MlpUp:
    case TensorKind::MlpDown:
        return true;
    default:
        return false;
    }
}

// ── Topology ────────────────────────────────────────────────────────────────

QwenTopology load_qwen_topology(const std::string &index_path)
{
    QwenTopology top;
    std::ifstream f(index_path);
    if (!f) return top;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return top;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return top;

    auto get_int = [&](const char *k, int dflt) {
        int64_t v;
        simdjson::ondemand::value x;
        if (root[k].get(x) != simdjson::SUCCESS || x.get_int64().get(v) != simdjson::SUCCESS)
            return dflt;
        return static_cast<int>(v);
    };
    auto get_dbl = [&](const char *k, double dflt) {
        if (auto x = root[k].get_double(); x.error() == simdjson::SUCCESS) return x.value();
        return dflt;
    };

    top.num_layers = get_int("num_layers", 0);
    top.hidden_size = get_int("hidden_size", 0);
    top.intermediate_size = get_int("intermediate_size", 0);
    top.vocab_size = get_int("vocab_size", 0);
    top.num_attention_heads = get_int("num_attention_heads", 0);
    top.num_key_value_heads = get_int("num_key_value_heads", 0);
    top.head_dim = get_int("head_dim", 0);
    top.linear_num_key_heads = get_int("linear_num_key_heads", 0);
    top.linear_num_value_heads = get_int("linear_num_value_heads", 0);
    top.linear_key_head_dim = get_int("linear_key_head_dim", 0);
    top.linear_value_head_dim = get_int("linear_value_head_dim", 0);
    top.linear_conv_kernel_dim = get_int("linear_conv_kernel_dim", 0);
    top.rms_norm_eps = get_dbl("rms_norm_eps", 1e-6);
    top.partial_rotary_factor = get_dbl("partial_rotary_factor", 0.25);
    top.rope_theta = get_dbl("rope_theta", 10000000.0);
    top.total_pinned_tensors = get_int("total_pinned_tensors", 0);

    if (auto arr = root["layer_types"].get_array(); arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            std::string_view sv;
            if (v.get_string().get(sv) == simdjson::SUCCESS)
                top.layer_types.emplace_back(sv);
        }
    }
    return top;
}

arrow::Status WriteQwenModelIndex(const std::string &output_root, const QwenTopology &top)
{
    namespace fs = std::filesystem;
    auto dir = fs::path(output_root) / "logestix";
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::string lt = "[";
    for (size_t i = 0; i < top.layer_types.size(); ++i) {
        if (i) lt += ",";
        lt += "\"" + top.layer_types[i] + "\"";
    }
    lt += "]";

    std::string json =
        "{\n"
        "  \"format\": \"logestix-qwen\",\n"
        "  \"model\": \"Qwen3.5-9B\",\n"
        "  \"num_layers\": " + std::to_string(top.num_layers) +
        ",\n  \"layer_types\": " + lt +
        ",\n  \"hidden_size\": " + std::to_string(top.hidden_size) +
        ",\n  \"intermediate_size\": " + std::to_string(top.intermediate_size) +
        ",\n  \"vocab_size\": " + std::to_string(top.vocab_size) +
        ",\n  \"num_attention_heads\": " + std::to_string(top.num_attention_heads) +
        ",\n  \"num_key_value_heads\": " + std::to_string(top.num_key_value_heads) +
        ",\n  \"head_dim\": " + std::to_string(top.head_dim) +
        ",\n  \"linear_num_key_heads\": " + std::to_string(top.linear_num_key_heads) +
        ",\n  \"linear_num_value_heads\": " + std::to_string(top.linear_num_value_heads) +
        ",\n  \"linear_key_head_dim\": " + std::to_string(top.linear_key_head_dim) +
        ",\n  \"linear_value_head_dim\": " + std::to_string(top.linear_value_head_dim) +
        ",\n  \"linear_conv_kernel_dim\": " + std::to_string(top.linear_conv_kernel_dim) +
        ",\n  \"rms_norm_eps\": " + std::to_string(top.rms_norm_eps) +
        ",\n  \"partial_rotary_factor\": " + std::to_string(top.partial_rotary_factor) +
        ",\n  \"rope_theta\": " + std::to_string(top.rope_theta) +
        ",\n  \"total_pinned_tensors\": " + std::to_string(top.total_pinned_tensors) +
        "\n}";

    std::ofstream out(dir / "model.index.json");
    out << json;
    return arrow::Status::OK();
}

// ── Converter ──────────────────────────────────────────────────────────────

namespace {

std::vector<uint8_t> read_tensor_data(
    const std::filesystem::path &input_path,
    const SafetensorsModel::Shard &shard,
    const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((input_path / shard.filename).string(), std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char*>(&hdr_len), 8);
    // ifstream in binary mode: read the 8-byte header length already consumed.
    int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return buf;
}

} // namespace

arrow::Status ConvertQwenModel(const std::string &input_path, const std::string &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_path);

    // Determine layer count from tensor names.
    int max_layer = -1;
    std::map<int, std::string> layer_type;  // layer -> linear_attention | full_attention
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_qwen_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;
            if (p.layer >= 0) {
                max_layer = std::max(max_layer, p.layer);
                if (p.kind == TensorKind::LinearAttnQkv)
                    layer_type[p.layer] = "linear_attention";
                else if (p.kind == TensorKind::FullAttnQ)
                    layer_type[p.layer] = "full_attention";
            }
        }
    if (max_layer < 0)
        return arrow::Status::Invalid("no recognized layer tensors in ", input_path);

    QwenTopology top;
    top.num_layers = max_layer + 1;
    for (int l = 0; l < top.num_layers; ++l) {
        auto it = layer_type.find(l);
        top.layer_types.push_back(it != layer_type.end() ? it->second : "linear_attention");
    }

    // If config.json is present, it is the authoritative geometry source
    // (matches the real artifact). Otherwise fall back to shape inference.
    auto read_cfg = [](const std::string &path, QwenTopology &t) {
        std::ifstream cf(path);
        if (!cf) return false;
        std::string cj((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        simdjson::ondemand::parser cp;
        simdjson::ondemand::document cd;
        simdjson::padded_string cpj(cj.data(), cj.size());
        if (cp.iterate(cpj).get(cd) != simdjson::SUCCESS) return false;

        auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
            int64_t v;
            if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
            return static_cast<int>(v);
        };
        auto get_d = [&](simdjson::ondemand::object &o, const char *k, double dflt) {
            simdjson::ondemand::value x;
            if (o[k].get(x) != simdjson::SUCCESS) return dflt;
            if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) return d.value();
            if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS) return (double)i.value();
            return dflt;
        };
        auto rope_d = [&](simdjson::ondemand::object &o, const char *k, double dflt) {
            return get_d(o, k, dflt);
        };

        if (auto root = cd.get_object(); root.error() == simdjson::SUCCESS) {
            simdjson::ondemand::object top_obj = root.value();
            // Qwen3.5 nests geometry under text_config.
            simdjson::ondemand::object src = top_obj;
            if (auto tc = top_obj["text_config"].get_object(); tc.error() == simdjson::SUCCESS)
                src = tc.value();

            t.hidden_size = get_i(src, "hidden_size", t.hidden_size);
            t.intermediate_size = get_i(src, "intermediate_size", t.intermediate_size);
            t.vocab_size = get_i(src, "vocab_size", t.vocab_size);
            t.num_attention_heads = get_i(src, "num_attention_heads", t.num_attention_heads);
            t.num_key_value_heads = get_i(src, "num_key_value_heads", t.num_key_value_heads);
            t.head_dim = get_i(src, "head_dim", t.head_dim);
            if (t.head_dim == 0 && t.num_attention_heads > 0 && t.hidden_size > 0)
                t.head_dim = t.hidden_size / t.num_attention_heads;
            t.linear_num_key_heads = get_i(src, "linear_num_key_heads", t.linear_num_key_heads);
            t.linear_num_value_heads = get_i(src, "linear_num_value_heads", t.linear_num_value_heads);
            t.linear_key_head_dim = get_i(src, "linear_key_head_dim", t.linear_key_head_dim);
            t.linear_value_head_dim = get_i(src, "linear_value_head_dim", t.linear_value_head_dim);
            t.linear_conv_kernel_dim = get_i(src, "linear_conv_kernel_dim", t.linear_conv_kernel_dim);
            t.rms_norm_eps = get_d(src, "rms_norm_eps", t.rms_norm_eps);

            if (auto ro = src["rope_parameters"].get_object(); ro.error() == simdjson::SUCCESS) {
                simdjson::ondemand::object rope = ro.value();
                t.partial_rotary_factor = rope_d(rope, "partial_rotary_factor", t.partial_rotary_factor);
                t.rope_theta = rope_d(rope, "rope_theta", t.rope_theta);
            }
            return true;
        }
        return false;
    };

    auto cfg_path = fs::path(input_path) / "config.json";
    // Shape-inference fallback first (only used if config.json is absent).
    if (!fs::exists(cfg_path)) {
        // Hidden/intermediate/vocab from tensor shapes.
        for (auto &shard : model.shards) {
            for (auto &t : shard.tensors) {
                auto p = classify_qwen_tensor(t.name);
                if (p.layer < 0 || t.shape.size() < 2) continue;
                int in = static_cast<int>(t.shape[0]);
                int out = static_cast<int>(t.shape[1]);
                switch (p.kind) {
                case TensorKind::Embed:
                case TensorKind::LmHead:
                    top.vocab_size = in;
                    if (!top.hidden_size) top.hidden_size = out;
                    break;
                case TensorKind::MlpGate:
                    top.hidden_size = in;
                    top.intermediate_size = out;
                    break;
                case TensorKind::LinearAttnQkv:
                    // out = k_dim*2 + v_dim; infer linear geometry from siblings
                    top.hidden_size = in;
                    for (auto &s2 : shard.tensors) {
                        auto p2 = classify_qwen_tensor(s2.name);
                        if (p2.kind == TensorKind::LinearAttnA_log) {
                            top.linear_num_value_heads = static_cast<int>(s2.numel());
                        }
                    }
                    break;
                case TensorKind::LinearAttnZ: {
                    if (top.linear_num_value_heads > 0)
                        top.linear_value_head_dim = out / top.linear_num_value_heads;
                    break;
                }
                default:
                    break;
                }
            }
        }
        if (top.linear_num_value_heads > 0 && top.linear_value_head_dim > 0)
            top.linear_num_key_heads = 1;  // placeholder; refine if needed
    } else {
        read_cfg(cfg_path.string(), top);
    }

    // Build pinned outputs per layer.
    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir))
        fs::create_directories(pinned_dir);

    std::map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    std::map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_qwen_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::MTP ||
                p.kind == TensorKind::Visual)
                continue;

            auto data = read_tensor_data(input_path, shard, t);
            if (data.empty()) {
                spdlog::warn("  {} read empty", t.name);
                continue;
            }

            bool nvfp4 = qwen_tensor_is_nvfp4(p.kind);

            if (p.kind == TensorKind::Embed || p.kind == TensorKind::LmHead) {
                if (nvfp4) {
                    DenseNvfp4Record r;
                    r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                    if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype,
                                                  nullptr, t.shape, {},
                                                  r.packed, r.scales, r.global_scale))
                        r.packed.clear();
                    global_nvfp4.push_back(std::move(r));
                } else {
                    TensorRecord r;
                    r.name = t.name; r.dtype = t.dtype; r.shape = t.shape; r.data = std::move(data);
                    global_raw.push_back(std::move(r));
                }
                ++total_pinned;
                continue;
            }
            if (p.kind == TensorKind::FinalNorm) {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape; r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                continue;
            }

            int layer = p.layer;
            if (nvfp4) {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype,
                                              nullptr, t.shape, {},
                                              r.packed, r.scales, r.global_scale))
                    r.packed.clear();
                nvfp4_by_layer[layer].push_back(std::move(r));
            } else {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape; r.data = std::move(data);
                raw_by_layer[layer].push_back(std::move(r));
            }
            ++total_pinned;
        }
    }

    // Write global.arrow with a merged (raw + nvfp4) record layout. To keep the
    // loader simple we store NVFP4 records under DenseNvfp4Schema and raw under
    // PinnedSchema in separate files. global.arrow holds raw; global_nvfp4.arrow
    // holds quantized embed/head. Per-layer: layer_raw_NNN.arrow + layer_NNN.arrow.
    if (!global_raw.empty()) {
        auto batch = PinnedSchema::MakeBatch(global_raw);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / "global.arrow"));
    }
    if (!global_nvfp4.empty()) {
        auto batch = DenseNvfp4Schema::MakeBatch(global_nvfp4);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / "global_nvfp4.arrow"));
        total_pinned += static_cast<int64_t>(global_nvfp4.size());
    }

    for (auto &[layer, recs] : nvfp4_by_layer) {
        auto batch = DenseNvfp4Schema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        auto fname = "layer_" + std::to_string(layer) + "_nvfp4.arrow";
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / fname));
    }
    for (auto &[layer, recs] : raw_by_layer) {
        auto batch = PinnedSchema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        auto fname = "layer_" + std::to_string(layer) + ".arrow";
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / fname));
    }

    top.total_pinned_tensors = static_cast<int>(total_pinned);
    return WriteQwenModelIndex(output_root, top);
}

} // namespace lsxcommon::qwen