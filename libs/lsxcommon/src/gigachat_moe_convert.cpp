// GigaChat3.5-432B-A28B-Reasoning converter (Model Ingestion Protocol step 1).
// See gigachat_moe_constants.h for the frozen geometry and the ingestion design
// docs/superpowers/specs/2026-09-20-gigachat35-432b-a28b-reasoning-ingestion-design.md.
#include "lsxcommon/gigachat_moe_convert.h"

#include <cstdlib>
#include <string_view>

namespace lsxcommon::gigachat {
namespace {

bool EndsWith(std::string_view s, std::string_view suf)
{
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Strips a trailing ".weight_scale_inv" or ".weight" to expose the module path.
std::string_view BaseModule(std::string_view name)
{
    for (std::string_view suf : {std::string_view(".weight_scale_inv"),
                                 std::string_view(".weight")})
        if (EndsWith(name, suf))
            return name.substr(0, name.size() - suf.size());
    return name;
}

std::string_view LastComponent(std::string_view module)
{
    const size_t dot = module.rfind('.');
    return dot == std::string_view::npos ? module : module.substr(dot + 1);
}

bool ParseLayerAndRest(std::string_view name, int &layer,
                       std::string_view &rest)
{
    constexpr std::string_view kPrefix = "model.layers.";
    if (name.rfind(kPrefix, 0) != 0) return false;
    std::string_view tail = name.substr(kPrefix.size());
    size_t i = 0;
    while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
    if (i == 0 || i >= tail.size() || tail[i] != '.') return false;
    layer = 0;
    for (size_t j = 0; j < i; ++j) layer = layer * 10 + (tail[j] - '0');
    rest = tail.substr(i + 1);
    // Expert index, when present: experts.<E>.<proj>[.weight]
    return true;
}

GigaChatTensorKind ClassifyModule(std::string_view module, int &expert)
{
    expert = -1;
    const std::string_view last = LastComponent(module);

    // Layer norms (ZeroCenteredGatedNorm): weight / gate projections.
    if (last == "input_layernorm" || last == "post_attention_layernorm" ||
        last == "post_self_attn_layernorm" ||
        last == "post_feedforward_layernorm" ||
        last == "q_a_layernorm" || last == "kv_a_layernorm")
        return GigaChatTensorKind::LayerNormWeight;
    if (last == "gate_up_projection")
        return GigaChatTensorKind::LayerNormGateUp;
    if (last == "gate_down_projection")
        return GigaChatTensorKind::LayerNormGateDown;

    // MLA attention.
    if (module == "self_attn.q_a_proj") return GigaChatTensorKind::QaProj;
    if (module == "self_attn.q_b_proj") return GigaChatTensorKind::QbProj;
    if (module == "self_attn.kv_a_proj_with_mqa") return GigaChatTensorKind::KvaProj;
    if (module == "self_attn.kv_b_proj") return GigaChatTensorKind::KvbProj;
    if (module == "self_attn.o_proj") return GigaChatTensorKind::OProj;
    if (module == "self_attn.gate_proj") return GigaChatTensorKind::AttnGateProj;

    // GatedDeltaNet.
    if (module == "self_attn.in_proj_qkvz") return GigaChatTensorKind::GdnInProjQkvz;
    if (module == "self_attn.out_proj") return GigaChatTensorKind::GdnOutProj;
    if (module == "self_attn.conv1d") return GigaChatTensorKind::GdnConv1d;
    if (module == "self_attn.A_log") return GigaChatTensorKind::GdnALog;
    if (module == "self_attn.dt_bias") return GigaChatTensorKind::GdnDtBias;
    if (module == "self_attn.in_proj_ba") return GigaChatTensorKind::GdnInProjBa;
    if (module == "self_attn.norm") return GigaChatTensorKind::GdnNormWeight;

    // Router (BF16 gate + F32 correction bias).
    if (module == "mlp.gate") return GigaChatTensorKind::RouterWeight;
    if (module == "mlp.gate.e_score_correction_bias")
        return GigaChatTensorKind::RouterBias;

    // Routed experts: mlp.experts.<E>.<proj>
    constexpr std::string_view kExperts = "mlp.experts.";
    if (module.rfind(kExperts, 0) == 0)
    {
        std::string_view tail = module.substr(kExperts.size());
        size_t i = 0;
        while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
        if (i == 0 || i >= tail.size() || tail[i] != '.') return GigaChatTensorKind::Unknown;
        expert = 0;
        for (size_t j = 0; j < i; ++j) expert = expert * 10 + (tail[j] - '0');
        const std::string_view proj = tail.substr(i + 1);
        if (proj == "gate_proj") return GigaChatTensorKind::ExpertGate;
        if (proj == "up_proj") return GigaChatTensorKind::ExpertUp;
        if (proj == "down_proj") return GigaChatTensorKind::ExpertDown;
        return GigaChatTensorKind::Unknown;
    }

    // Shared experts.
    if (module == "mlp.shared_experts.gate_proj") return GigaChatTensorKind::SharedGate;
    if (module == "mlp.shared_experts.up_proj") return GigaChatTensorKind::SharedUp;
    if (module == "mlp.shared_experts.down_proj") return GigaChatTensorKind::SharedDown;

    // Dense MLP.
    if (module == "mlp.gate_proj") return GigaChatTensorKind::MlpGate;
    if (module == "mlp.up_proj") return GigaChatTensorKind::MlpUp;
    if (module == "mlp.down_proj") return GigaChatTensorKind::MlpDown;

    return GigaChatTensorKind::Unknown;
}

} // namespace

ParsedGigaChatTensor classify_gigachat_tensor(const std::string &name)
{
    ParsedGigaChatTensor out;
    const std::string_view n(name);

    if (n == "model.embed_tokens.weight")
    {
        out.kind = GigaChatTensorKind::Embed;
        return out;
    }
    if (n == "lm_head.weight")
    {
        out.kind = GigaChatTensorKind::LmHead;
        return out;
    }
    if (n.rfind("model.norm", 0) == 0)
    {
        const std::string_view module = BaseModule(n);
        if (module == "model.norm")
            out.kind = GigaChatTensorKind::FinalNormWeight;
        else if (module == "model.norm.gate_up_projection")
            out.kind = GigaChatTensorKind::FinalNormGateUp;
        else if (module == "model.norm.gate_down_projection")
            out.kind = GigaChatTensorKind::FinalNormGateDown;
        return out;
    }

    int layer = -1;
    std::string_view rest;
    if (!ParseLayerAndRest(n, layer, rest))
        return out;
    out.layer = layer;

    // MTP layers 40-42: converted + stored, never applied.
    if (layer == 40 || layer == 41 || layer == 42)
    {
        out.kind = GigaChatTensorKind::Mtp;
        return out;
    }

    out.kind = ClassifyModule(BaseModule(rest), out.expert);
    return out;
}

bool gigachat_tensor_is_nvfp4(GigaChatTensorKind kind)
{
    switch (kind)
    {
    case GigaChatTensorKind::QaProj:
    case GigaChatTensorKind::QbProj:
    case GigaChatTensorKind::KvaProj:
    case GigaChatTensorKind::KvbProj:
    case GigaChatTensorKind::OProj:
    case GigaChatTensorKind::AttnGateProj:
    case GigaChatTensorKind::MlpGate:
    case GigaChatTensorKind::MlpUp:
    case GigaChatTensorKind::MlpDown:
    case GigaChatTensorKind::GdnInProjQkvz:
    case GigaChatTensorKind::GdnOutProj:
    case GigaChatTensorKind::SharedGate:
    case GigaChatTensorKind::SharedUp:
    case GigaChatTensorKind::SharedDown:
    case GigaChatTensorKind::ExpertGate:
    case GigaChatTensorKind::ExpertUp:
    case GigaChatTensorKind::ExpertDown:
        return true;
    default:
        return false;
    }
}

bool gigachat_tensor_is_expert(GigaChatTensorKind kind)
{
    return kind == GigaChatTensorKind::ExpertGate ||
           kind == GigaChatTensorKind::ExpertUp ||
           kind == GigaChatTensorKind::ExpertDown;
}

} // namespace lsxcommon::gigachat
