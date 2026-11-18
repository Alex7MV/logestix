#include "lsxcommon/muse_dense_convert.h"

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>

namespace lsxcommon {
namespace muse_dense {

namespace {

std::optional<ParsedTensor> match_layer(const std::string &name,
                                        const std::regex &re, TensorKind kind)
{
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = kind;
    if (!absl::SimpleAtoi(m[1].str(), &p.layer)) return std::nullopt;
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

bool parse_text_config(const std::string &json, MuseDenseTopology &t)
{
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return false;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return false;

    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object(); tc.error() == simdjson::SUCCESS)
        src = tc.value();

    auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
        int64_t v;
        if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_d = [&](simdjson::ondemand::object &o, const char *k, double dflt) {
        simdjson::ondemand::value x;
        if (o[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
            return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };

    t.hidden_size = get_i(src, "hidden_size", t.hidden_size);
    t.intermediate_size = get_i(src, "intermediate_size", t.intermediate_size);
    t.vocab_size = get_i(src, "vocab_size", t.vocab_size);
    // config.json spells it num_hidden_layers; model.index.json writes
    // num_layers. Accept both so the converted-index round-trip works.
    {
        int64_t v;
        if (src["num_hidden_layers"].get_int64().get(v) == simdjson::SUCCESS)
            t.num_layers = static_cast<int>(v);
        else if (src["num_layers"].get_int64().get(v) == simdjson::SUCCESS)
            t.num_layers = static_cast<int>(v);
    }
    t.total_pinned_tensors =
        get_i(src, "total_pinned_tensors", t.total_pinned_tensors);
    t.skipped_visual_tensors =
        get_i(src, "skipped_visual_tensors", t.skipped_visual_tensors);
    t.num_attention_heads =
        get_i(src, "num_attention_heads", t.num_attention_heads);
    t.num_key_value_heads =
        get_i(src, "num_key_value_heads", t.num_key_value_heads);
    t.head_dim = get_i(src, "head_dim", t.head_dim);
    t.sliding_window = get_i(src, "sliding_window", t.sliding_window);
    t.rms_norm_eps = static_cast<float>(get_d(src, "rms_norm_eps", t.rms_norm_eps));
    t.post_norm_eps =
        static_cast<float>(get_d(src, "post_norm_eps", t.post_norm_eps));
    t.qk_scale_factor =
        static_cast<float>(get_d(src, "qk_scale_factor", t.qk_scale_factor));
    t.output_multiplier =
        static_cast<float>(get_d(src, "output_multiplier", t.output_multiplier));
    t.final_logit_softcapping = static_cast<float>(
        get_d(src, "final_logit_softcapping", t.final_logit_softcapping));

    if (auto ro = src["rope_parameters"].get_object();
        ro.error() == simdjson::SUCCESS) {
        simdjson::ondemand::object rope = ro.value();
        t.rope_theta =
            static_cast<float>(get_d(rope, "rope_theta", t.rope_theta));
    }

    if (auto lt = src["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS) {
        t.layer_types.clear();
        for (auto v : lt.value()) {
            std::string_view sv;
            if (v.get_string().get(sv) == simdjson::SUCCESS)
                t.layer_types.emplace_back(sv);
        }
    }

    // Per-layer RoPE base: 0 disables rotary (NoPE) for that layer. When the
    // array is absent, derive it from the layer types (full -> 0).
    t.layer_rope_theta.clear();
    if (auto rt = src["layer_rope_theta"].get_array();
        rt.error() == simdjson::SUCCESS) {
        for (auto v : rt.value()) {
            double d = 0.0;
            if (auto dd = v.get_double(); dd.error() == simdjson::SUCCESS)
                d = dd.value();
            else if (auto ii = v.get_int64(); ii.error() == simdjson::SUCCESS)
                d = static_cast<double>(ii.value());
            t.layer_rope_theta.push_back(d);
        }
    }
    if (t.layer_rope_theta.empty()) {
        for (size_t l = 0; l < t.layer_types.size(); ++l)
            t.layer_rope_theta.push_back(t.IsFull(static_cast<int>(l))
                                             ? 0.0
                                             : static_cast<double>(t.rope_theta));
    }

    // Special ids. text_config carries bos/eos; pad is null there and comes
    // from generation_config.json (merged below). The converted index stores
    // `eos_ids` / `pad_id` directly.
    {
        int64_t v;
        if (src["bos_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.bos_id = static_cast<int>(v);
        else if (root["bos_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.bos_id = static_cast<int>(v);
        if (src["pad_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.pad_id = static_cast<int>(v);
        else if (root["pad_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.pad_id = static_cast<int>(v);

        t.eos_ids.clear();
        if (auto arr = src["eos_ids"].get_array();
            arr.error() == simdjson::SUCCESS) {
            for (auto x : arr.value()) {
                int64_t e;
                if (x.get_int64().get(e) == simdjson::SUCCESS)
                    t.eos_ids.push_back(static_cast<int>(e));
            }
        }
        if (t.eos_ids.empty()) {
            if (auto arr = src["eos_token_id"].get_array();
                arr.error() == simdjson::SUCCESS) {
                for (auto x : arr.value()) {
                    int64_t e;
                    if (x.get_int64().get(e) == simdjson::SUCCESS)
                        t.eos_ids.push_back(static_cast<int>(e));
                }
            } else if (src["eos_token_id"].get_int64().get(v) ==
                       simdjson::SUCCESS) {
                t.eos_ids.push_back(static_cast<int>(v));
            }
        }
    }
    return true;
}

// HF uses generation_config's eos_token_id for stopping (the turn-end <|eot|>
// is only listed there); it also supplies the pad id the text_config leaves
// null.
void MergeGenerationConfig(const std::string &dir, MuseDenseTopology &top)
{
    std::ifstream gf(std::filesystem::path(dir) / "generation_config.json");
    if (!gf) return;
    std::string gjson((std::istreambuf_iterator<char>(gf)),
                      std::istreambuf_iterator<char>());
    simdjson::ondemand::parser gp;
    simdjson::padded_string gpad(gjson.data(), gjson.size());
    simdjson::ondemand::document gdoc;
    if (gp.iterate(gpad).get(gdoc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object go;
    if (gdoc.get_object().get(go) != simdjson::SUCCESS) return;
    if (auto arr = go["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto x : arr.value()) {
            int64_t e;
            if (x.get_int64().get(e) != simdjson::SUCCESS) continue;
            const int id = static_cast<int>(e);
            if (std::find(top.eos_ids.begin(), top.eos_ids.end(), id) ==
                top.eos_ids.end())
                top.eos_ids.push_back(id);
        }
    } else if (auto e = go["eos_token_id"].get_int64();
               e.error() == simdjson::SUCCESS) {
        const int id = static_cast<int>(e.value());
        if (std::find(top.eos_ids.begin(), top.eos_ids.end(), id) ==
            top.eos_ids.end())
            top.eos_ids.push_back(id);
    }
    if (auto p = go["pad_token_id"].get_int64(); p.error() == simdjson::SUCCESS)
        top.pad_id = static_cast<int>(p.value());
}

} // namespace

ParsedTensor classify_muse_dense_tensor(const std::string &name)
{
    using K = TensorKind;

    // Multimodal ballast (vision tower / adapter / projection) — counted,
    // never applied (this is a text-only runtime).
    if (name.rfind("model.vision_tower.", 0) == 0 ||
        name.rfind("vision_tower.", 0) == 0 ||
        name.rfind("model.vision_adapter.", 0) == 0 ||
        name.rfind("vision_adapter.", 0) == 0 ||
        name.rfind("model.vision_projection.", 0) == 0 ||
        name.rfind("vision_projection.", 0) == 0)
        return {K::Visual, -1};

    if (name == "model.language_model.embed_tokens.weight")
        return {K::Embed, -1};
    if (name == "lm_head.weight") return {K::LmHead, -1};
    if (name == "model.language_model.norm.weight") return {K::FinalNorm, -1};

    auto P = [&](K kind, const char *suffix) -> std::optional<ParsedTensor> {
        return match_layer(name, re_linear(suffix), kind);
    };

    if (auto p = P(K::AttnQ, R"(self_attn\.q_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnK, R"(self_attn\.k_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnV, R"(self_attn\.v_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnO, R"(self_attn\.o_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnGate, R"(self_attn\.gate_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpGate, R"(mlp\.gate_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpUp, R"(mlp\.up_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpDown, R"(mlp\.down_proj\.weight$)")) return *p;
    if (auto p = P(K::InputNorm, R"(input_layernorm\.weight$)")) return *p;
    if (auto p = P(K::PostAttnNorm, R"(post_attention_layernorm\.weight$)"))
        return *p;
    if (auto p = P(K::PreFfnNorm, R"(pre_feedforward_layernorm\.weight$)"))
        return *p;
    if (auto p = P(K::PostFfnNorm, R"(post_feedforward_layernorm\.weight$)"))
        return *p;

    return {K::Unknown, -1};
}

bool muse_dense_tensor_is_nvfp4(TensorKind kind)
{
    switch (kind) {
    case TensorKind::AttnQ:
    case TensorKind::AttnK:
    case TensorKind::AttnV:
    case TensorKind::AttnO:
    case TensorKind::AttnGate:
    case TensorKind::MlpGate:
    case TensorKind::MlpUp:
    case TensorKind::MlpDown:
        return true;
    default:
        return false;
    }
}

// ── Topology ────────────────────────────────────────────────────────────────

MuseDenseTopology load_muse_dense_topology(const std::string &config_path)
{
    MuseDenseTopology top;
    std::ifstream f(config_path);
    if (!f) {
        top.error_msg = "cannot open config file: " + config_path;
        return top;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (!parse_text_config(json, top)) {
        MuseDenseTopology failed;
        failed.error_msg = "failed to parse config JSON: " + config_path;
        return failed;
    }

    MergeGenerationConfig(
        std::filesystem::path(config_path).parent_path().string(), top);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        // Absent vram block is fine at conversion time (it gets written);
        // only a malformed present block is an error.
        if (json.find("\"vram\"") != std::string::npos) {
            MuseDenseTopology failed;
            failed.error_msg = verr;
            return failed;
        }
    }
    return top;
}

} // namespace muse_dense
} // namespace lsxcommon