#include "lsxcommon/kimi_convert.h"

#include <simdjson.h>

#include <absl/strings/str_cat.h>

#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

namespace lsxcommon {
namespace kimi {
namespace {

// Strips the multimodal checkpoint wrapper: "language_model.model." or
// "language_model.". Returns the inner name.
std::string strip_language_model(const std::string &name)
{
    const std::string full = "language_model.model.";
    if (name.rfind(full, 0) == 0) return name.substr(full.size());
    const std::string head = "language_model.";
    if (name.rfind(head, 0) == 0) return name.substr(head.size());
    return name;
}

bool parse_int(const std::string &s, int &out)
{
    return absl::SimpleAtoi(s, &out);
}

int weight_id_for(const std::string &proj)
{
    if (proj == "gate_proj") return 1;
    if (proj == "up_proj") return 2;
    if (proj == "down_proj") return 3;
    return 0;
}

// model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale}
std::optional<KimiParsedTensor> parse_routed_expert(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_(packed|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::RoutedExpert;
    parse_int(m[1].str(), p.layer);
    parse_int(m[2].str(), p.expert);
    p.weight_id = weight_id_for(m[3].str());
    p.base = m[3].str() + ".weight_" + m[4].str();
    return p;
}

// model.layers.N.mlp.shared_experts.{gate,up,down}_proj.{weight,weight_packed,
// weight_scale}
std::optional<KimiParsedTensor> parse_shared_expert(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.(weight|weight_packed|weight_scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::SharedExpert;
    parse_int(m[1].str(), p.layer);
    p.expert = 0;
    p.weight_id = weight_id_for(m[2].str());
    p.base = "shared_experts." + m[2].str() + "." + m[3].str();
    return p;
}

// model.layers.N.mlp.gate.{weight,e_score_correction_bias}
std::optional<KimiParsedTensor> parse_router(const std::string &name)
{
    static const std::regex re(R"(layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Router;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// model.0.mlp / layers.0.mlp.{gate,up,down}_proj.{weight,weight_packed,weight_scale}
// Dense MLP only exists on layer 0 (first_k_dense_replace=1).
std::optional<KimiParsedTensor> parse_dense_mlp(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(0)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|weight_packed|weight_scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::DenseMlp;
    p.layer = 0;
    p.weight_id = weight_id_for(m[2].str());
    p.base = m[2].str();
    return p;
}

// model.layers.N.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//                           kv_a_layernorm,kv_b_proj,o_proj}.{weight,bias}
std::optional<KimiParsedTensor> parse_attention(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.self_attn\.(q_a_proj|q_a_layernorm|q_b_proj|kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj|o_proj)\.(weight|bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Attention;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// model.layers.N.(input_layernorm|post_attention_layernorm).weight
std::optional<KimiParsedTensor> parse_layer_norm(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::LayerNorm;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// embed_tokens.weight / lm_head.weight / model.norm.weight (post-strip)
std::optional<KimiParsedTensor> parse_global(const std::string &name)
{
    std::string base;
    if (name == "embed_tokens.weight") base = "embed_tokens";
    else if (name == "lm_head.weight") base = "lm_head";
    else if (name == "norm.weight") base = "model.norm";
    else return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Global;
    p.base = base;
    return p;
}

} // namespace

KimiParsedTensor kimi_classify(const std::string &full_name)
{
    const std::string name = strip_language_model(full_name);
    if (auto p = parse_global(name)) return *p;
    if (auto p = parse_layer_norm(name)) return *p;
    if (auto p = parse_attention(name)) return *p;
    if (auto p = parse_router(name)) return *p;
    if (auto p = parse_shared_expert(name)) return *p;
    if (auto p = parse_routed_expert(name)) return *p;
    if (auto p = parse_dense_mlp(name)) return *p;
    return {};
}

KimiTopology kimi_load_topology(const std::string &index_json_path)
{
    namespace fs = std::filesystem;
    KimiTopology topo;

    std::ifstream f(index_json_path);
    if (!f) {
        topo.error_msg = "cannot open " + index_json_path;
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "simdjson iterate failed for " + index_json_path;
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "root is not an object: " + index_json_path;
        return topo;
    }
    simdjson::ondemand::object weight_map;
    if (root["weight_map"].get_object().get(weight_map) != simdjson::SUCCESS) {
        topo.error_msg = "missing weight_map in " + index_json_path;
        return topo;
    }

    std::set<long long> moe, dense;
    long long max_layer = -1;
    long long max_expert = -1;
    for (simdjson::ondemand::field field : weight_map) {
        std::string_view key_sv;
        if (field.unescaped_key().get(key_sv) != simdjson::SUCCESS) continue;
        std::string key(key_sv);
        auto parsed = kimi_classify(key);
        if (parsed.kind == KimiTensorKind::Router) {
            moe.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        } else if (parsed.kind == KimiTensorKind::RoutedExpert) {
            max_expert = std::max(max_expert,
                                  static_cast<long long>(parsed.expert));
            moe.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        } else if (parsed.kind == KimiTensorKind::DenseMlp) {
            dense.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        }
    }

    if (moe.empty()) {
        topo.error_msg = "no MoE layers found in " + index_json_path;
        return topo;
    }
    topo.num_layers = max_layer + 1;
    topo.experts_per_layer = max_expert + 1;
    topo.moe_layers.assign(moe.begin(), moe.end());
    topo.dense_layers.assign(dense.begin(), dense.end());
    return topo;
}

} // namespace kimi
} // namespace lsxcommon
