// GLM-5.3-Flash converter: tensor classifier (Task 1). Conversion
// pipeline lands in Task 3; FP8->NVFP4 requant in Task 2.
#include "glm_moe_flash_constants.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

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

} // namespace lsxcommon::glm_moe_flash
