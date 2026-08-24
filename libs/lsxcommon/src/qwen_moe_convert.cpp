#include "lsxcommon/qwen_moe_convert.h"

#include <simdjson.h>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::qwen_moe {
namespace {

// simdjson ondemand idioms copied from gemma4_convert.cpp parse_text_config.
auto get_i = [](simdjson::ondemand::object &o, const char *k, int dflt) {
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
};
auto get_d = [](simdjson::ondemand::object &o, const char *k, double dflt) {
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return dflt;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) return d.value();
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
        return static_cast<double>(i.value());
    return dflt;
};

} // namespace

QwenMoeTopology ParseQwenMoeConfig(const std::string &input_dir)
{
    QwenMoeTopology top;
    namespace fs = std::filesystem;
    const fs::path cfg_path = fs::path(input_dir) / "config.json";
    std::error_code ec;
    if (!fs::exists(cfg_path, ec)) {
        top.error_msg = "config.json not found: " + cfg_path.string();
        return top;
    }
    std::ifstream f(cfg_path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        top.error_msg = "config.json parse error";
        return top;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        top.error_msg = "config.json is not an object";
        return top;
    }

    // text_config preferred; fall back to flat root (some fixtures).
    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object();
        tc.error() == simdjson::SUCCESS)
        src = tc.value();

    top.hidden_size = get_i(src, "hidden_size", 0);
    top.num_layers = get_i(src, "num_hidden_layers", 0);
    top.vocab_size = get_i(src, "vocab_size", 0);
    top.num_attention_heads = get_i(src, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(src, "num_key_value_heads", 0);
    top.head_dim = get_i(src, "head_dim", 0);
    top.linear_num_key_heads = get_i(src, "linear_num_key_heads", 0);
    top.linear_num_value_heads = get_i(src, "linear_num_value_heads", 0);
    top.linear_key_head_dim = get_i(src, "linear_key_head_dim", 0);
    top.linear_value_head_dim = get_i(src, "linear_value_head_dim", 0);
    top.linear_conv_kernel_dim = get_i(src, "linear_conv_kernel_dim", 4);
    top.intermediate_size = get_i(src, "intermediate_size", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-6);
    top.partial_rotary_factor = get_d(src, "partial_rotary_factor", 0.25);
    top.rope_theta = get_d(src, "rope_theta", 1e7);

    // MoE fields (etalon: gemma4_convert.cpp num_experts parsing).
    top.num_experts = get_i(src, "num_experts", 0);
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(src, "moe_intermediate_size", 0);
    top.shared_expert_intermediate_size =
        get_i(src, "shared_expert_intermediate_size", 0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);

    if (auto lt = src["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                top.layer_types.emplace_back(sv);
        }
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.layer_types.empty()) {
        top.error_msg = "config.json missing required MoE fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layer_types size ", top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    return top;
}

ParsedMoeTensor classify_qwen_moe_tensor(const std::string &name)
{
    ParsedMoeTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = MoeTensorKind::Mtp;
        return r;
    }
    if (name.find("visual") != std::string::npos) {
        r.kind = MoeTensorKind::Visual;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = MoeTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "model.language_model.";
    if (name.rfind(kPrefix, 0) != 0)
        return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = MoeTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = MoeTensorKind::FinalNorm;
        return r;
    }

    // Extract "layers.<N>." index and the per-layer body.
    std::string body = rest;
    const std::string kLayers = "layers.";
    if (rest.rfind(kLayers, 0) == 0) {
        const auto digits_start = kLayers.size();
        const auto digits_end = rest.find('.', digits_start);
        if (digits_end == std::string::npos)
            return r;
        r.layer =
            std::stoi(rest.substr(digits_start, digits_end - digits_start));
        body = rest.substr(digits_end + 1);
    }

    using K = MoeTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::FullAttnQ},
        {"self_attn.k_proj.weight", K::FullAttnK},
        {"self_attn.v_proj.weight", K::FullAttnV},
        {"self_attn.o_proj.weight", K::FullAttnO},
        {"self_attn.q_norm.weight", K::FullAttnQNorm},
        {"self_attn.k_norm.weight", K::FullAttnKNorm},
        {"linear_attn.in_proj_qkv.weight", K::LinearAttnQkv},
        {"linear_attn.in_proj_z.weight", K::LinearAttnZ},
        {"linear_attn.in_proj_b.weight", K::LinearAttnB},
        {"linear_attn.in_proj_a.weight", K::LinearAttnA},
        {"linear_attn.out_proj.weight", K::LinearAttnOut},
        {"linear_attn.conv1d.weight", K::LinearAttnConv},
        {"linear_attn.norm.weight", K::LinearAttnNorm},
        {"linear_attn.A_log", K::LinearAttnALog},
        {"linear_attn.dt_bias", K::LinearAttnDt},
        {"mlp.gate.weight", K::RouterGate},
        {"mlp.experts.gate_up_proj", K::FusedExpertsGateUp},
        {"mlp.experts.down_proj", K::FusedExpertsDown},
        {"mlp.shared_expert.gate_proj.weight", K::SharedExpertGateProj},
        {"mlp.shared_expert.up_proj.weight", K::SharedExpertUpProj},
        {"mlp.shared_expert.down_proj.weight", K::SharedExpertDownProj},
        {"mlp.shared_expert_gate.weight", K::SharedExpertGate},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool qwen_moe_tensor_is_nvfp4(MoeTensorKind kind)
{
    switch (kind) {
    case MoeTensorKind::Embed:
    case MoeTensorKind::LmHead:
    case MoeTensorKind::FullAttnQ:
    case MoeTensorKind::FullAttnK:
    case MoeTensorKind::FullAttnV:
    case MoeTensorKind::FullAttnO:
    case MoeTensorKind::LinearAttnQkv:
    case MoeTensorKind::LinearAttnZ:
    case MoeTensorKind::LinearAttnB:
    case MoeTensorKind::LinearAttnA:
    case MoeTensorKind::LinearAttnOut:
    case MoeTensorKind::FusedExpertsGateUp:
    case MoeTensorKind::FusedExpertsDown:
    case MoeTensorKind::SharedExpertGateProj:
    case MoeTensorKind::SharedExpertUpProj:
    case MoeTensorKind::SharedExpertDownProj:
        return true;
    default:
        return false;
    }
}

arrow::Status ConvertQwen35MoeModel(const std::string &, const std::string &)
{
    return arrow::Status::NotImplemented("Task 3");
}

} // namespace lsxcommon::qwen_moe
