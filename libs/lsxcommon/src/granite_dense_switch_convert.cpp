#include "lsxcommon/granite_dense_switch_convert.h"

#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <sstream>

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace granite_switch {

namespace {

using K = GraniteSwitchTensorKind;

std::optional<ParsedGraniteSwitchTensor> match_layer(
    const std::string &name, const std::string &suffix,
    GraniteSwitchTensorKind kind, int slice = -1)
{
    static std::regex re;
    re = std::regex(R"(model\.layers\.(\d+)\.)" + suffix);
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedGraniteSwitchTensor p;
    p.kind = kind;
    p.layer = std::stoi(m[1].str());
    p.slice = slice;
    if (slice >= 0 && m.size() > 2) p.slice = std::stoi(m[2].str());
    return p;
}

} // namespace

ParsedGraniteSwitchTensor classify_granite_switch_tensor(
    const std::string &name)
{
    if (name == "model.embed_tokens.weight") return {K::Embed, -1, -1};
    if (name == "model.norm.weight") return {K::FinalNorm, -1, -1};
    if (name == "model.switch.codebook") return {K::SwitchCodebook, -1, -1};
    if (name == "model.switch.control_to_substitute_lut")
        return {K::SwitchLut, -1, -1};

    if (auto p = match_layer(name, R"(input_layernorm\.weight)", K::InputNorm))
        return *p;
    if (auto p = match_layer(name, R"(post_attention_layernorm\.weight)",
                             K::PostNorm))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.qkv_proj\.base_layer\.weight)",
                             K::AttnQkvBase))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.qkv_proj\.lora_A_slices\.(\d+))",
                             K::AttnQkvLoraA, 0))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.qkv_proj\.lora_B_slices\.(\d+))",
                             K::AttnQkvLoraB, 0))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.o_proj\.base_layer\.weight)",
                             K::AttnOBase))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.o_proj\.lora_A)",
                             K::AttnOLoraA))
        return *p;
    if (auto p = match_layer(name, R"(self_attn\.o_proj\.lora_B)",
                             K::AttnOLoraB))
        return *p;
    if (auto p = match_layer(
            name, R"(shared_mlp\.input_linear\.base_layer\.weight)",
            K::MlpGateUpBase))
        return *p;
    if (auto p = match_layer(
            name, R"(shared_mlp\.input_linear\.lora_A_slices\.(\d+))",
            K::MlpGateUpLoraA, 0))
        return *p;
    if (auto p = match_layer(
            name, R"(shared_mlp\.input_linear\.lora_B_slices\.(\d+))",
            K::MlpGateUpLoraB, 0))
        return *p;
    if (auto p = match_layer(
            name, R"(shared_mlp\.output_linear\.base_layer\.weight)",
            K::MlpDownBase))
        return *p;
    if (auto p = match_layer(name, R"(shared_mlp\.output_linear\.lora_A)",
                             K::MlpDownLoraA))
        return *p;
    if (auto p = match_layer(name, R"(shared_mlp\.output_linear\.lora_B)",
                             K::MlpDownLoraB))
        return *p;

    return {K::Unknown, -1, -1};
}

bool granite_switch_tensor_is_nvfp4(GraniteSwitchTensorKind kind)
{
    switch (kind) {
    case K::Embed:
    case K::AttnQkvBase:
    case K::AttnOBase:
    case K::MlpGateUpBase:
    case K::MlpDownBase:
        return true;
    default:
        return false;
    }
}

GraniteSwitchTopology load_granite_switch_topology(
    const std::string &index_path)
{
    GraniteSwitchTopology top;
    std::ifstream f(index_path);
    if (!f) {
        top.error_msg = "cannot open " + index_path;
        return top;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        top.error_msg = "invalid JSON in " + index_path;
        return top;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        top.error_msg = "index root is not an object";
        return top;
    }

    auto get_int = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_dbl = [&](const char *k, double dflt) {
        double v;
        if (root[k].get_double().get(v) != simdjson::SUCCESS) return dflt;
        return v;
    };
    auto get_bool = [&](const char *k, bool dflt) {
        bool v;
        if (root[k].get_bool().get(v) != simdjson::SUCCESS) return dflt;
        return v;
    };
    auto get_int_array = [&](const char *k, std::vector<int> &out) {
        out.clear();
        simdjson::ondemand::array arr;
        if (root[k].get_array().get(arr) != simdjson::SUCCESS) return;
        for (auto e : arr) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.push_back(static_cast<int>(v));
        }
    };
    auto get_str_array = [&](const char *k, std::vector<std::string> &out) {
        out.clear();
        simdjson::ondemand::array arr;
        if (root[k].get_array().get(arr) != simdjson::SUCCESS) return;
        for (auto e : arr) {
            std::string_view s;
            if (e.get_string().get(s) == simdjson::SUCCESS)
                out.emplace_back(s);
        }
    };

    top.num_layers = get_int("num_layers", 0);
    top.switch_cache_layers = get_int("switch_cache_layers", 2);
    top.hidden_size = get_int("hidden_size", 0);
    top.intermediate_size = get_int("intermediate_size", 0);
    top.vocab_size = get_int("vocab_size", 0);
    top.num_attention_heads = get_int("num_attention_heads", 0);
    top.num_key_value_heads = get_int("num_key_value_heads", 0);
    top.head_dim = get_int("head_dim", 0);
    top.rms_norm_eps = get_dbl("rms_norm_eps", 1e-5);
    top.rope_theta = get_dbl("rope_theta", 50000000.0);
    top.attention_multiplier = get_dbl("attention_multiplier", 0.0078125);
    top.embedding_multiplier = get_dbl("embedding_multiplier", 12.0);
    top.residual_multiplier = get_dbl("residual_multiplier", 0.175);
    top.logits_scaling = get_dbl("logits_scaling", 16.0);
    top.tie_word_embeddings = get_bool("tie_word_embeddings", true);
    top.bos_id = get_int("bos_id", -1);
    top.eos_id = get_int("eos_id", -1);
    top.pad_id = get_int("pad_id", -1);
    top.num_adapters = get_int("num_adapters", 0);
    get_int_array("adapter_ranks", top.adapter_ranks);
    get_int_array("adapter_token_ids", top.adapter_token_ids);
    get_int_array("adapter_substitute_token_ids",
                  top.adapter_substitute_token_ids);
    get_str_array("adapter_names", top.adapter_names);
    top.total_pinned_tensors = get_int("total_pinned_tensors", 0);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        top.error_msg = verr;
        return top;
    }
    return top;
}

arrow::Status WriteGraniteSwitchModelIndex(
    const std::string &output_root, const GraniteSwitchTopology &top)
{
    namespace fs = std::filesystem;
    auto dir = fs::path(output_root) / "logestix";
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    auto int_array = [](const std::vector<int> &v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(v[i]);
        }
        s += "]";
        return s;
    };
    auto str_array = [](const std::vector<std::string> &v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ",";
            s += "\"" + v[i] + "\"";
        }
        s += "]";
        return s;
    };

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"logestix-granite-switch\",\n"
      << "  \"model\": \"Granite-Switch-4.1-30B\",\n"
      << "  \"num_layers\": " << top.num_layers
      << ",\n  \"switch_cache_layers\": " << top.switch_cache_layers
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
      << ",\n  \"rope_theta\": " << top.rope_theta
      << ",\n  \"attention_multiplier\": " << top.attention_multiplier
      << ",\n  \"embedding_multiplier\": " << top.embedding_multiplier
      << ",\n  \"residual_multiplier\": " << top.residual_multiplier
      << ",\n  \"logits_scaling\": " << top.logits_scaling
      << ",\n  \"tie_word_embeddings\": "
      << (top.tie_word_embeddings ? "true" : "false")
      << ",\n  \"num_adapters\": " << top.num_adapters
      << ",\n  \"adapter_names\": " << str_array(top.adapter_names)
      << ",\n  \"adapter_ranks\": " << int_array(top.adapter_ranks)
      << ",\n  \"adapter_token_ids\": " << int_array(top.adapter_token_ids)
      << ",\n  \"adapter_substitute_token_ids\": "
      << int_array(top.adapter_substitute_token_ids)
      << ",\n  \"lora_b_prescaled\": true"
      << ",\n  \"bos_id\": " << top.bos_id
      << ",\n  \"eos_id\": " << top.eos_id
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram)
      << "\n}";

    std::ofstream out(dir / "model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

namespace {

using K2 = GraniteSwitchTensorKind;

struct SwitchConfigReader {
    GraniteSwitchTopology top;
    bool ok = false;
};

bool ReadSwitchConfig(const std::string &path, GraniteSwitchTopology &top)
{
    std::ifstream cf(path);
    if (!cf) return false;
    std::string cj((std::istreambuf_iterator<char>(cf)),
                   std::istreambuf_iterator<char>());
    simdjson::ondemand::parser cp;
    simdjson::padded_string pad(cj.data(), cj.size());
    simdjson::ondemand::document cd;
    if (cp.iterate(pad).get(cd) != simdjson::SUCCESS) return false;
    simdjson::ondemand::object root;
    if (cd.get_object().get(root) != simdjson::SUCCESS) return false;

    auto i32 = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto f64 = [&](const char *k, double dflt) {
        double v;
        if (root[k].get_double().get(v) != simdjson::SUCCESS) return dflt;
        return v;
    };
    auto boolean = [&](const char *k, bool dflt) {
        bool v;
        if (root[k].get_bool().get(v) != simdjson::SUCCESS) return dflt;
        return v;
    };
    auto ints = [&](const char *k, std::vector<int> &out) {
        out.clear();
        simdjson::ondemand::array a;
        if (root[k].get_array().get(a) != simdjson::SUCCESS) return;
        for (auto e : a) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.push_back(static_cast<int>(v));
        }
    };
    auto strs = [&](const char *k, std::vector<std::string> &out) {
        out.clear();
        simdjson::ondemand::array a;
        if (root[k].get_array().get(a) != simdjson::SUCCESS) return;
        for (auto e : a) {
            std::string_view s;
            if (e.get_string().get(s) == simdjson::SUCCESS)
                out.emplace_back(s);
        }
    };

    std::string_view model_type;
    if (root["model_type"].get_string().get(model_type) == simdjson::SUCCESS &&
        model_type != "granite_switch") {
        return false;
    }

    top.hidden_size = i32("hidden_size", 0);
    top.intermediate_size = i32("intermediate_size", 0);
    if (top.intermediate_size == 0)
        top.intermediate_size = i32("shared_intermediate_size", 0);
    top.num_layers = i32("num_hidden_layers", 0);
    top.switch_cache_layers = 2;
    top.vocab_size = i32("vocab_size", 0);
    top.num_attention_heads = i32("num_attention_heads", 0);
    top.num_key_value_heads = i32("num_key_value_heads", 0);
    top.head_dim = i32("projection_head_dim", i32("head_dim", 0));
    if (top.head_dim == 0 && top.num_attention_heads > 0)
        top.head_dim = top.hidden_size / top.num_attention_heads;
    top.rms_norm_eps = f64("rms_norm_eps", 1e-5);
    top.rope_theta = f64("rope_theta", 50000000.0);
    if (auto ro = root["rope_parameters"].get_object();
        ro.error() == simdjson::SUCCESS) {
        simdjson::ondemand::object rope = ro.value();
        double v;
        if (rope["rope_theta"].get_double().get(v) == simdjson::SUCCESS)
            top.rope_theta = v;
    }
    top.attention_multiplier = f64("attention_multiplier", 0.0078125);
    top.embedding_multiplier = f64("embedding_multiplier", 12.0);
    top.residual_multiplier = f64("residual_multiplier", 0.175);
    top.logits_scaling = f64("logits_scaling", 16.0);
    top.tie_word_embeddings = boolean("tie_word_embeddings", true);
    top.bos_id = i32("bos_token_id", -1);
    top.eos_id = i32("eos_token_id", -1);
    top.pad_id = i32("pad_token_id", -1);
    top.num_adapters = i32("num_adapters", 0);
    ints("adapter_ranks", top.adapter_ranks);
    ints("adapter_token_ids", top.adapter_token_ids);
    ints("adapter_substitute_token_ids", top.adapter_substitute_token_ids);
    strs("adapter_names", top.adapter_names);
    return true;
}

std::vector<std::string> ExpectedNames(int layers)
{
    std::vector<std::string> expected;
    expected.push_back("model.embed_tokens.weight");
    expected.push_back("model.norm.weight");
    expected.push_back("model.switch.codebook");
    expected.push_back("model.switch.control_to_substitute_lut");
    for (int l = 0; l < layers; ++l) {
        const std::string b = "model.layers." + std::to_string(l) + ".";
        expected.push_back(b + "input_layernorm.weight");
        expected.push_back(b + "post_attention_layernorm.weight");
        expected.push_back(b + "self_attn.qkv_proj.base_layer.weight");
        for (int s = 0; s < 3; ++s) {
            expected.push_back(b + "self_attn.qkv_proj.lora_A_slices." +
                               std::to_string(s));
            expected.push_back(b + "self_attn.qkv_proj.lora_B_slices." +
                               std::to_string(s));
        }
        expected.push_back(b + "self_attn.o_proj.base_layer.weight");
        expected.push_back(b + "self_attn.o_proj.lora_A");
        expected.push_back(b + "self_attn.o_proj.lora_B");
        expected.push_back(b + "shared_mlp.input_linear.base_layer.weight");
        for (int s = 0; s < 2; ++s) {
            expected.push_back(b + "shared_mlp.input_linear.lora_A_slices." +
                               std::to_string(s));
            expected.push_back(b + "shared_mlp.input_linear.lora_B_slices." +
                               std::to_string(s));
        }
        expected.push_back(b + "shared_mlp.output_linear.base_layer.weight");
        expected.push_back(b + "shared_mlp.output_linear.lora_A");
        expected.push_back(b + "shared_mlp.output_linear.lora_B");
    }
    return expected;
}

std::string ReplaceAll(std::string s, const std::string &from,
                      const std::string &to)
{
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string NvName(const std::string &src)
{
    return ReplaceAll(
        ReplaceAll(
            ReplaceAll(
                ReplaceAll(src, "self_attn.qkv_proj.base_layer.weight",
                           "self_attn.qkv_proj.weight"),
                "self_attn.o_proj.base_layer.weight",
                "self_attn.o_proj.weight"),
            "shared_mlp.input_linear.base_layer.weight",
            "shared_mlp.gate_up_proj.weight"),
        "shared_mlp.output_linear.base_layer.weight",
        "shared_mlp.down_proj.weight");
}

std::vector<uint8_t> SliceBf16(const std::vector<uint8_t> &data, int64_t elem_off,
                               int64_t n)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 2);
    std::memcpy(out.data(), data.data() + elem_off * 2, out.size());
    return out;
}

bool AllZeroBf16(const std::vector<uint8_t> &data)
{
    for (uint8_t b : data)
        if (b != 0) return false;
    return true;
}

} // namespace

arrow::Status ConvertGraniteSwitchModel(const std::string &input_path,
                                        const std::string &output_root)
{
    namespace fs = std::filesystem;
    auto model = scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    GraniteSwitchTopology top;
    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    if (!ReadSwitchConfig(cfg_path.string(), top))
        return arrow::Status::Invalid("failed to parse ", cfg_path.string());

    int max_layer = -1;
    std::vector<std::string> present;
    absl::flat_hash_set<int> present_layers;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_granite_switch_tensor(t.name);
            if (p.kind == K2::Unknown)
                return arrow::Status::Invalid("unclassified tensor: ", t.name);
            present.push_back(t.name);
            if (p.layer >= 0) {
                max_layer = std::max(max_layer, p.layer);
                present_layers.insert(p.layer);
            }
        }
    }
    const int tensor_layers = max_layer + 1;
    if (top.num_layers != tensor_layers + top.switch_cache_layers)
        return arrow::Status::Invalid(
            "config num_hidden_layers (", top.num_layers,
            ") != decoder layers (", tensor_layers, ") + switch cache layers (",
            top.switch_cache_layers, ")");
    top.num_layers = tensor_layers;

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));
    ARROW_RETURN_NOT_OK(integrity::CheckTensorCoverage(
        ExpectedNames(tensor_layers), present, "granite-switch"));
    ARROW_RETURN_NOT_OK(
        integrity::CheckLayerContiguity(tensor_layers, present_layers));

    // Shape / adapter metadata checks.
    if (static_cast<int>(top.adapter_ranks.size()) != top.num_adapters ||
        static_cast<int>(top.adapter_token_ids.size()) != top.num_adapters ||
        static_cast<int>(top.adapter_substitute_token_ids.size()) !=
            top.num_adapters ||
        top.num_adapters <= 0)
        return arrow::Status::Invalid("adapter metadata is incomplete");

    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir)) fs::create_directories(pinned_dir);

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nv_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> adapter_by_layer;
    std::vector<DenseNvfp4Record> global_nv;
    std::vector<TensorRecord> global_raw;

    // Stacked LoRA tensors first, then trim per adapter when the layer is done.
    // LUT verification (control id -> substitute, -1 elsewhere).
    {
        std::vector<uint8_t> lut_raw;
        bool found = false;
        for (auto &shard : model.shards)
            for (auto &t : shard.tensors)
                if (t.name == "model.switch.control_to_substitute_lut") {
                    lut_raw = ReadTensorData(input_path, shard, t);
                    found = true;
                }
        if (!found)
            return arrow::Status::Invalid("switch LUT missing");
        if (lut_raw.size() !=
            static_cast<size_t>(top.vocab_size) * sizeof(int64_t))
            return arrow::Status::Invalid("switch LUT has the wrong size");
        const int64_t *lut = reinterpret_cast<const int64_t *>(lut_raw.data());
        for (int v = 0; v < top.vocab_size; ++v) {
            int expect = -1;
            for (int a = 0; a < top.num_adapters; ++a)
                if (top.adapter_token_ids[a] == v)
                    expect = top.adapter_substitute_token_ids[a];
            if (lut[v] != expect)
                return arrow::Status::Invalid(
                    "switch LUT mismatch at id ", v, ": ", lut[v],
                    " != ", expect);
        }
    }

    int64_t total_pinned = 0;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_granite_switch_tensor(t.name);
            if (p.kind == K2::SwitchLut) continue; // verified, not stored
            auto data = ReadTensorData(input_path, shard, t);
            if (data.empty())
                return arrow::Status::IOError("failed to read ", t.name);

            if (p.kind == K2::Embed) {
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype,
                                              nullptr, t.shape, {}, r.packed,
                                              r.scales, r.global_scale))
                    return arrow::Status::IOError("embed quantize failed");
                global_nv.push_back(std::move(r));
                ++total_pinned;
                continue;
            }
            if (p.kind == K2::FinalNorm || p.kind == K2::SwitchCodebook) {
                global_raw.push_back(
                    {t.name, t.dtype, t.shape, std::move(data)});
                ++total_pinned;
                continue;
            }
            if (p.kind == K2::InputNorm || p.kind == K2::PostNorm) {
                raw_by_layer[p.layer].push_back(
                    {t.name, t.dtype, t.shape, std::move(data)});
                ++total_pinned;
                continue;
            }
            if (p.kind == K2::AttnQkvLoraA || p.kind == K2::AttnQkvLoraB ||
                p.kind == K2::AttnOLoraA || p.kind == K2::AttnOLoraB ||
                p.kind == K2::MlpGateUpLoraA || p.kind == K2::MlpGateUpLoraB ||
                p.kind == K2::MlpDownLoraA || p.kind == K2::MlpDownLoraB) {
                raw_by_layer[p.layer].push_back(
                    {t.name + ".__stack", t.dtype, t.shape, std::move(data)});
                ++total_pinned;
                continue;
            }
            DenseNvfp4Record r;
            r.name = NvName(t.name);
            r.shape = t.shape;
            r.dtype = t.dtype;
            if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype,
                                          nullptr, t.shape, {}, r.packed,
                                          r.scales, r.global_scale))
                return arrow::Status::IOError("quantize failed: ", t.name);
            nv_by_layer[p.layer].push_back(std::move(r));
            ++total_pinned;
        }
    }

    // Trim the stacked LoRA tensors into per-adapter records.
    for (auto &[layer, raw] : raw_by_layer) {
        std::vector<TensorRecord> kept;
        absl::btree_map<std::string, const TensorRecord *> stacked;
        for (auto &r : raw) {
            constexpr size_t kStackSuffixLen = 8; // ".__stack"
            if (r.name.size() > kStackSuffixLen &&
                r.name.compare(r.name.size() - kStackSuffixLen,
                               kStackSuffixLen, ".__stack") == 0)
                stacked.emplace(
                    r.name.substr(0, r.name.size() - kStackSuffixLen), &r);
            else
                kept.push_back(r); // norms
        }
        for (auto &kv : stacked) {
            const TensorRecord &src = *kv.second;
            const bool is_a = kv.first.find("lora_A") != std::string::npos;
            const int R = src.shape[2];
            const int64_t per_adapter = is_a
                ? static_cast<int64_t>(R) * src.shape[3]
                : static_cast<int64_t>(src.shape[2]) * src.shape[3];
            const int dim = is_a ? static_cast<int>(src.shape[3])
                                 : static_cast<int>(src.shape[2]);
            for (int a = 0; a < top.num_adapters; ++a) {
                const int rank = top.adapter_ranks[a];
                if (rank <= 0) continue;
                std::vector<uint8_t> block;
                if (is_a) {
                    block = SliceBf16(src.data, a * per_adapter,
                                      static_cast<int64_t>(rank) * dim);
                    // Bytes beyond the adapter rank must be zero padding.
                    std::vector<uint8_t> tail =
                        SliceBf16(src.data, a * per_adapter +
                                                 static_cast<int64_t>(rank) * dim,
                                  static_cast<int64_t>(R - rank) * dim);
                    if (!AllZeroBf16(tail))
                        return arrow::Status::Invalid(
                            "LoRA A tail is not zero-padded: ", kv.first);
                } else {
                    block.resize(static_cast<size_t>(dim) * rank * 2);
                    for (int n = 0; n < dim; ++n) {
                        std::memcpy(
                            block.data() + static_cast<int64_t>(n) * rank * 2,
                            src.data.data() +
                                (a * per_adapter + static_cast<int64_t>(n) * src.shape[3]) * 2,
                            static_cast<size_t>(rank) * 2);
                    }
                }
                if (AllZeroBf16(block)) continue;
                TensorRecord out;
                out.name = kv.first + ".adapter" + std::to_string(a);
                out.dtype = "BF16";
                out.shape = is_a ? std::vector<int64_t>{rank, dim}
                                 : std::vector<int64_t>{dim, rank};
                out.data = std::move(block);
                adapter_by_layer[layer].push_back(std::move(out));
            }
        }
        raw.swap(kept);
    }

    auto write_raw = [](const fs::path &dir, const std::string &fname,
                        std::vector<TensorRecord> &recs) -> arrow::Status {
        if (recs.empty()) return arrow::Status::OK();
        auto batch = PinnedSchema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        return WriteRecordBatchToFile(*batch, dir / fname);
    };
    auto write_nv = [](const fs::path &dir, const std::string &fname,
                       std::vector<DenseNvfp4Record> &recs) -> arrow::Status {
        if (recs.empty()) return arrow::Status::OK();
        auto batch = DenseNvfp4Schema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        return WriteRecordBatchToFile(*batch, dir / fname);
    };

    ARROW_RETURN_NOT_OK(write_raw(pinned_dir, "global.arrow", global_raw));
    ARROW_RETURN_NOT_OK(write_nv(pinned_dir, "global_nvfp4.arrow", global_nv));
    for (auto &[layer, recs] : nv_by_layer)
        ARROW_RETURN_NOT_OK(write_nv(
            pinned_dir, absl::StrCat("layer_", layer, "_nvfp4.arrow"), recs));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(write_raw(
            pinned_dir, absl::StrCat("layer_", layer, ".arrow"), recs));
    for (auto &[layer, recs] : adapter_by_layer)
        ARROW_RETURN_NOT_OK(write_raw(
            pinned_dir, absl::StrCat("layer_", layer, "_adapters.arrow"), recs));

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = top.num_layers;
        vg.context_length = ReadContextLength(cfg_path.string());
        std::vector<VramTensorInfo> inv;
        for (auto &shard : model.shards)
            for (auto &t : shard.tensors)
                inv.push_back({t.name, t.numel(), t.byte_size()});
        top.vram = EstimateVram(
            inv,
            {+[](const std::string &n) {
                 return granite_switch_tensor_is_nvfp4(
                     classify_granite_switch_tensor(n).kind);
             },
             nullptr},
            vg);
    }

    ARROW_RETURN_NOT_OK(WriteGraniteSwitchModelIndex(output_root, top));

    auto tok_path = fs::path(input_path) / "tokenizer.json";
    if (!fs::exists(tok_path))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    TokenizerSpecialPolicy tok_policy;
    tok_policy.pretokenizer = 6;
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tok_path.string(), pinned_dir,
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true,
                                       .pad_required = true}, tok_policy));
    return arrow::Status::OK();
}

} // namespace granite_switch
} // namespace lsxcommon
