#include "lsxcommon/gemma4_convert.h"

#include <optional>
#include <regex>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>

namespace lsxcommon {
namespace gemma {

namespace {

std::optional<ParsedTensor> match_layer(const std::string &name,
                                        const std::regex &re, TensorKind kind)
{
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = kind;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    return p;
}

std::regex &re_linear(const std::string &suffix)
{
    static absl::flat_hash_map<std::string, std::regex> cache;
    const std::string pat =
        R"((?:model\.)?language_model\.layers\.(\d+)\.)" + suffix;
    auto it = cache.find(suffix);
    if (it == cache.end())
        it = cache.emplace(suffix, std::regex(pat)).first;
    return it->second;
}

} // namespace

ParsedTensor classify_gemma_tensor(const std::string &name)
{
    using K = TensorKind;

    // Multimodal fat (vision tower / vision embed) — skipped at conversion.
    if (name.rfind("model.vision_tower.", 0) == 0 ||
        name.rfind("vision_tower.", 0) == 0 ||
        name.rfind("model.embed_vision.", 0) == 0 ||
        name.rfind("embed_vision.", 0) == 0)
        return {K::Visual, -1};

    if (name == "model.language_model.embed_tokens.weight") return {K::Embed, -1};
    if (name == "model.language_model.norm.weight") return {K::FinalNorm, -1};

    auto P = [&](K kind, const char *suffix) -> std::optional<ParsedTensor> {
        return match_layer(name, re_linear(suffix), kind);
    };

    if (auto p = P(K::ExpertsGateUp,        R"(experts\.gate_up_proj$)"))                return *p;
    if (auto p = P(K::ExpertsDown,          R"(experts\.down_proj$)"))                   return *p;
    if (auto p = P(K::RouterProj,           R"(router\.proj\.weight$)"))                 return *p;
    if (auto p = P(K::RouterScale,          R"(router\.scale$)"))                        return *p;
    if (auto p = P(K::RouterPerExpertScale, R"(router\.per_expert_scale$)"))             return *p;
    if (auto p = P(K::LayerScalar,          R"(layer_scalar$)"))                         return *p;
    if (auto p = P(K::PreFFN2,              R"(pre_feedforward_layernorm_2\.weight$)"))  return *p;
    if (auto p = P(K::PostFFN2,             R"(post_feedforward_layernorm_2\.weight$)")) return *p;
    if (auto p = P(K::PreFFN,               R"(pre_feedforward_layernorm\.weight$)"))    return *p;
    if (auto p = P(K::PostFFN,              R"(post_feedforward_layernorm\.weight$)"))   return *p;
    if (auto p = P(K::AttnQNorm,            R"(self_attn\.q_norm\.weight$)"))            return *p;
    if (auto p = P(K::AttnKNorm,            R"(self_attn\.k_norm\.weight$)"))            return *p;
    if (auto p = P(K::AttnQ,                R"(self_attn\.q_proj\.weight$)"))            return *p;
    if (auto p = P(K::AttnK,                R"(self_attn\.k_proj\.weight$)"))            return *p;
    if (auto p = P(K::AttnV,                R"(self_attn\.v_proj\.weight$)"))            return *p;
    if (auto p = P(K::AttnO,                R"(self_attn\.o_proj\.weight$)"))            return *p;
    if (auto p = P(K::MlpGate,              R"(mlp\.gate_proj\.weight$)"))               return *p;
    if (auto p = P(K::MlpUp,                R"(mlp\.up_proj\.weight$)"))                 return *p;
    if (auto p = P(K::MlpDown,              R"(mlp\.down_proj\.weight$)"))               return *p;
    if (auto p = P(K::InputNorm,            R"(input_layernorm\.weight$)"))              return *p;
    if (auto p = P(K::PostNorm,             R"(post_attention_layernorm\.weight$)"))     return *p;

    return {K::Unknown, -1};
}

bool gemma_tensor_is_nvfp4(TensorKind kind)
{
    return kind == TensorKind::ExpertsGateUp || kind == TensorKind::ExpertsDown;
}

} // namespace gemma
} // namespace lsxcommon
