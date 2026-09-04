// GLM-5.3-Flash converter: tensor classifier (Task 1), config parser +
// FP8->NVFP4 requant host reference (Task 2). Conversion pipeline lands
// in Task 3.
#include "glm_moe_flash_constants.h"
#include "glm_moe_flash_fp8.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <simdjson.h>

namespace lsxcommon::glm_moe_flash {

namespace {

// Matches "<prefix>layers.<N>.<rest>" for both model.layers.N and
// model.language_model.layers.N. Returns true with layer + rest.
bool SplitLayer(const std::string &name, int &layer, std::string &rest)
{
    static const std::array<const char *, 2> kPrefixes = {
        "model.language_model.layers.", "model.layers."};
    for (const char *pre : kPrefixes) {
        const size_t p = name.find(pre);
        if (p == std::string::npos) continue;
        const size_t ds = p + std::strlen(pre);
        size_t i = ds;
        while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
        if (i == ds || i >= name.size() || name[i] != '.') continue;
        layer = std::atoi(name.c_str() + ds);
        rest = name.substr(i + 1);
        return true;
    }
    return false;
}

// Extracts the routed expert index from "mlp.experts.<E>.<suffix>".
bool SplitExpert(const std::string &rest, int &expert, std::string &suffix)
{
    static const char kKey[] = "mlp.experts.";
    const size_t p = rest.find(kKey);
    if (p == std::string::npos) return false;
    const size_t ds = p + sizeof(kKey) - 1;
    size_t i = ds;
    while (i < rest.size() && rest[i] >= '0' && rest[i] <= '9') ++i;
    if (i == ds || i >= rest.size() || rest[i] != '.') return false;
    expert = std::atoi(rest.c_str() + ds);
    suffix = rest.substr(i + 1);
    return true;
}

int ProjId(const std::string &suffix)
{
    if (suffix.rfind("gate_proj", 0) == 0) return 1;
    if (suffix.rfind("down_proj", 0) == 0) return 2;
    if (suffix.rfind("up_proj", 0) == 0) return 3;
    return 0;
}

} // namespace

ParsedFlashTensor classify_flash_tensor(const std::string &name)
{
    ParsedFlashTensor out;
    if (name == "lm_head.weight") {
        out.kind = FlashTensorKind::LmHead;
        return out;
    }
    if (name == "model.language_model.embed_tokens.weight" ||
        name == "model.embed_tokens.weight") {
        out.kind = FlashTensorKind::Embed;
        return out;
    }
    if (name == "model.language_model.norm.weight") {
        out.kind = FlashTensorKind::FinalNorm;
        return out;
    }
    if (name.rfind("model.visual.", 0) == 0 || name.rfind("visual.", 0) == 0) {
        out.kind = FlashTensorKind::Visual;
        return out;
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayer(name, layer, rest)) return out;
    out.layer = layer;

    if (layer == frozen::kMtpLayer && rest != "mlp.experts." + rest) {
        // MTP nextn layer: distinct MTP tensors first; the remaining
        // (attn/mlp/indexer) tensors fall through to normal classification
        // below — the MTP layer is itself a DSA+MoE layer.
        if (rest == "eh_proj.weight" || rest == "enorm.weight" ||
            rest == "hnorm.weight" || rest == "shared_head.norm.weight") {
            out.kind = FlashTensorKind::MtpEhProj;
            return out;
        }
    }

    int expert = -1;
    std::string suffix;
    if (SplitExpert(rest, expert, suffix)) {
        out.expert = expert;
        out.proj_id = ProjId(suffix);
        out.kind = FlashTensorKind::RoutedExpert;
        return out;
    }
    if (rest.rfind("mlp.shared_experts.", 0) == 0) {
        out.proj_id = ProjId(rest);
        out.kind = FlashTensorKind::SharedExpert;
        return out;
    }
    if (rest == "mlp.gate.weight") {
        out.kind = FlashTensorKind::RouterGate;
        return out;
    }
    if (rest == "mlp.gate.e_score_correction_bias") {
        out.kind = FlashTensorKind::RouterBias;
        return out;
    }
    if (rest.rfind("mlp.gate_proj", 0) == 0 ||
        rest.rfind("mlp.up_proj", 0) == 0 ||
        rest.rfind("mlp.down_proj", 0) == 0) {
        out.kind = FlashTensorKind::DenseMlp;
        out.proj_id = ProjId(rest);
        return out;
    }

    if (rest.rfind("self_attn.indexer.", 0) == 0) {
        out.kind = FlashTensorKind::Indexer;
        return out;
    }

    // hc_* tensors carry no ".weight" suffix.
    static const std::array<std::pair<const char *, FlashTensorKind>, 6>
        kHcKinds = {{
            {"hc_attn_base", FlashTensorKind::HcAttnBase},
            {"hc_attn_fn", FlashTensorKind::HcAttnFn},
            {"hc_attn_scale", FlashTensorKind::HcAttnScale},
            {"hc_ffn_base", FlashTensorKind::HcFfnBase},
            {"hc_ffn_fn", FlashTensorKind::HcFfnFn},
            {"hc_ffn_scale", FlashTensorKind::HcFfnScale},
        }};
    for (const auto &kv : kHcKinds) {
        if (rest == kv.first) {
            out.kind = kv.second;
            return out;
        }
    }

    if (rest == "input_layernorm.weight" ||
        rest == "post_attention_layernorm.weight") {
        out.kind = FlashTensorKind::DsaNorms;
        return out;
    }

    if (frozen::IsKdaLayer(layer)) {
        static const std::array<std::pair<const char *, FlashTensorKind>, 14>
            kKdaKinds = {{
                {"self_attn.q_proj.weight", FlashTensorKind::KdaQ},
                {"self_attn.k_proj.weight", FlashTensorKind::KdaK},
                {"self_attn.v_proj.weight", FlashTensorKind::KdaV},
                {"self_attn.o_proj.weight", FlashTensorKind::KdaO},
                {"self_attn.q_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.k_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.v_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.A_log", FlashTensorKind::KdaALog},
                {"self_attn.dt_bias", FlashTensorKind::KdaDtBias},
                {"self_attn.b_proj.weight", FlashTensorKind::KdaB},
                {"self_attn.f_a_proj.weight", FlashTensorKind::KdaFa},
                {"self_attn.f_b_proj.weight", FlashTensorKind::KdaFb},
                {"self_attn.g_a_proj.weight", FlashTensorKind::KdaGa},
                {"self_attn.g_b_proj.weight", FlashTensorKind::KdaGb},
            }};
        for (const auto &kv : kKdaKinds) {
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        }
        if (rest == "self_attn.o_norm.weight") {
            out.kind = FlashTensorKind::KdaONorm;
            return out;
        }
        return out;  // Unknown inside a KDA layer
    }

    if (frozen::IsDsaLayer(layer)) {
        static const std::array<std::pair<const char *, FlashTensorKind>, 5>
            kDsaKinds = {{
                {"self_attn.q_a_proj.weight", FlashTensorKind::DsaQa},
                {"self_attn.q_b_proj.weight", FlashTensorKind::DsaQb},
                {"self_attn.kv_a_proj_with_mqa.weight",
                 FlashTensorKind::DsaKvA},
                {"self_attn.kv_b_proj.weight", FlashTensorKind::DsaKvB},
                {"self_attn.o_proj.weight", FlashTensorKind::DsaOBig},
            }};
        for (const auto &kv : kDsaKinds) {
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        }
        if (rest == "self_attn.q_a_layernorm.weight" ||
            rest == "self_attn.kv_a_layernorm.weight") {
            out.kind = FlashTensorKind::DsaNorms;
            return out;
        }
        return out;
    }

    return out;
}

bool flash_tensor_is_nvfp4(FlashTensorKind kind)
{
    switch (kind) {
    case FlashTensorKind::RoutedExpert:
    case FlashTensorKind::SharedExpert:
    case FlashTensorKind::DsaQa:
    case FlashTensorKind::DsaQb:
    case FlashTensorKind::DsaKvA:
    case FlashTensorKind::DsaOBig:
    case FlashTensorKind::DenseMlp:
        return true;
    default:
        return false;
    }
}

FlashTopology ParseFlashConfig(const std::string &input_dir)
{
    namespace fs = std::filesystem;
    FlashTopology top;
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

    auto get_i = [](simdjson::ondemand::object &o, const char *k,
                    int dflt) -> int {
        int64_t v;
        if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_d = [](simdjson::ondemand::object &o, const char *k,
                    double dflt) -> double {
        simdjson::ondemand::value x;
        if (o[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
            return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };

    // text_config preferred; fall back to flat root.
    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object();
        tc.error() == simdjson::SUCCESS)
        src = tc.value();

    top.hidden = get_i(src, "hidden_size", 0);
    top.layers = get_i(src, "num_hidden_layers", 0);
    top.vocab = get_i(src, "vocab_size", 0);
    top.heads = get_i(src, "num_attention_heads", 0);
    top.qk_head_dim = get_i(src, "qk_head_dim", 0);
    top.q_lora = get_i(src, "q_lora_rank", 0);
    top.kv_lora = get_i(src, "kv_lora_rank", 0);
    top.num_experts = get_i(src, "n_routed_experts", 0);
    top.top_k = get_i(src, "num_experts_per_tok", 0);
    top.moe_intermediate = get_i(src, "moe_intermediate_size", 0);
    top.dense_intermediate = get_i(src, "intermediate_size", 0);
    top.hc_count = get_i(src, "hc_mult", 4);
    top.gate_lowrank = get_i(src, "gate_rank", frozen::kGateLowrank);
    top.indexer_heads = get_i(src, "index_n_heads", 0);
    top.indexer_head_dim = get_i(src, "index_head_dim", 0);
    top.indexer_topk = get_i(src, "index_topk", 0);

    // Short conv kernel size lives in linear_attn_config.
    if (auto lc = src["linear_attn_config"].get_object();
        lc.error() == simdjson::SUCCESS) {
        simdjson::ondemand::object lco = lc.value();
        top.conv_kernel =
            get_i(lco, "short_conv_kernel_size", frozen::kConvKernel);
        if (top.heads == 0)
            top.heads = get_i(lco, "num_heads", 0);
        if (top.qk_head_dim == 0)
            top.qk_head_dim = get_i(lco, "head_dim", 0);
    } else {
        top.conv_kernel = frozen::kConvKernel;
    }

    top.routed_scaling = get_d(src, "routed_scaling_factor", 0.0);
    top.pad_id = get_i(src, "pad_token_id", -1);

    // eos_token_id may be an array or a scalar.
    if (auto arr = src["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t e;
            if (v.get_int64().get(e) == simdjson::SUCCESS)
                top.eos_ids.push_back(static_cast<int>(e));
        }
    } else {
        const int single = get_i(src, "eos_token_id", -1);
        if (single >= 0) top.eos_ids.push_back(single);
    }

    // Structural sanity.
    if (top.hidden <= 0 || top.layers <= 0 || top.vocab <= 0) {
        top.error_msg = "config.json missing core text_config fields";
        return top;
    }
    if (top.num_experts <= 0 || top.top_k <= 0) {
        top.error_msg = "config.json missing MoE fields";
        return top;
    }
    return top;
}

} // namespace lsxcommon::glm_moe_flash
