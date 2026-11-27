#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/config_json.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

namespace lsxcommon {
namespace gemma {

namespace {
void MergeGenerationConfigEos(const std::string &dir, GemmaTopology &top)
{
    lsxcommon::config::MergeGenerationConfigEos(dir, top.eos_ids);
}


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
    if (auto p = P(K::PostFFN1,             R"(post_feedforward_layernorm_1\.weight$)")) return *p;
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

// ── Topology ────────────────────────────────────────────────────────────────

namespace {

bool parse_text_config(const std::string &json, GemmaTopology &t)
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
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };

    t.hidden_size = get_i(src, "hidden_size", t.hidden_size);
    t.intermediate_size = get_i(src, "intermediate_size", t.intermediate_size);
    t.moe_intermediate_size =
        get_i(src, "moe_intermediate_size", t.moe_intermediate_size);
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
    t.num_attention_heads = get_i(src, "num_attention_heads", t.num_attention_heads);
    t.num_key_value_heads = get_i(src, "num_key_value_heads", t.num_key_value_heads);
    t.num_global_key_value_heads =
        get_i(src, "num_global_key_value_heads", t.num_global_key_value_heads);
    t.head_dim = get_i(src, "head_dim", t.head_dim);
    t.global_head_dim = get_i(src, "global_head_dim", t.global_head_dim);
    t.sliding_window = get_i(src, "sliding_window", t.sliding_window);
    t.num_experts = get_i(src, "num_experts", t.num_experts);
    t.top_k_experts = get_i(src, "top_k_experts", t.top_k_experts);
    t.rms_norm_eps = static_cast<float>(
        get_d(src, "rms_norm_eps", t.rms_norm_eps));
    t.final_logit_softcapping = static_cast<float>(
        get_d(src, "final_logit_softcapping", t.final_logit_softcapping));

    // Per-layer-type RoPE parameters (real checkpoint nests them under
    // rope_parameters.{full_attention,sliding_attention}).
    if (auto ro = src["rope_parameters"].get_object(); ro.error() == simdjson::SUCCESS) {
        simdjson::ondemand::object rope = ro.value();
        if (auto fa = rope["full_attention"].get_object();
            fa.error() == simdjson::SUCCESS) {
            simdjson::ondemand::object f = fa.value();
            t.partial_rotary_factor = static_cast<float>(
                get_d(f, "partial_rotary_factor", t.partial_rotary_factor));
            t.rope_theta_full = static_cast<float>(
                get_d(f, "rope_theta", t.rope_theta_full));
        }
        if (auto sa = rope["sliding_attention"].get_object();
            sa.error() == simdjson::SUCCESS) {
            simdjson::ondemand::object s = sa.value();
            t.rope_theta_sliding = static_cast<float>(
                get_d(s, "rope_theta", t.rope_theta_sliding));
        }
    }

    if (auto lt = src["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        t.layer_types.clear();
        for (auto v : lt.value()) {
            std::string_view sv;
            if (v.get_string().get(sv) == simdjson::SUCCESS)
                t.layer_types.emplace_back(sv);
        }
    }

    // Special ids. The top-level config carries the full eos array (text_config
    // keeps only the first element); the converted index stores `eos_ids`.
    // Legacy scalar `eos_id` is accepted so old converted trees still load.
    {
        int64_t v;
        if (root["bos_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.bos_id = static_cast<int>(v);
        else if (src["bos_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.bos_id = static_cast<int>(v);
        if (root["pad_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.pad_id = static_cast<int>(v);
        else if (src["pad_token_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.pad_id = static_cast<int>(v);

        t.eos_ids.clear();
        auto collect = [&](simdjson::ondemand::object &o) {
            if (auto arr = o["eos_token_id"].get_array();
                arr.error() == simdjson::SUCCESS) {
                for (auto x : arr.value()) {
                    int64_t e;
                    if (x.get_int64().get(e) == simdjson::SUCCESS)
                        t.eos_ids.push_back(static_cast<int>(e));
                }
                return !t.eos_ids.empty();
            }
            if (o["eos_token_id"].get_int64().get(v) == simdjson::SUCCESS) {
                t.eos_ids.push_back(static_cast<int>(v));
                return true;
            }
            return false;
        };
        if (auto arr = root["eos_ids"].get_array();
            arr.error() == simdjson::SUCCESS) {
            for (auto x : arr.value()) {
                int64_t e;
                if (x.get_int64().get(e) == simdjson::SUCCESS)
                    t.eos_ids.push_back(static_cast<int>(e));
            }
        }
        if (t.eos_ids.empty() && !collect(root)) collect(src);
        if (t.eos_ids.empty() &&
            root["eos_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.eos_ids.push_back(static_cast<int>(v));
    }
    return true;
}

std::vector<std::string> generate_gemma_expected_names(const GemmaTopology &top)
{
    std::vector<std::string> expected;
    expected.push_back("model.language_model.embed_tokens.weight");
    expected.push_back("model.language_model.norm.weight");

    for (int l = 0; l < top.num_layers; ++l) {
        std::string base =
            absl::StrCat("model.language_model.layers.", l, ".");
        bool full = static_cast<size_t>(l) < top.layer_types.size() &&
                    top.layer_types[static_cast<size_t>(l)] ==
                        "full_attention";
        for (const char *s : {
                 "input_layernorm.weight",
                 "post_attention_layernorm.weight",
                 "pre_feedforward_layernorm.weight",
                 "post_feedforward_layernorm.weight",
                 "pre_feedforward_layernorm_2.weight",
"post_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm_1.weight",
        "layer_scalar",
                 "mlp.gate_proj.weight",
                 "mlp.up_proj.weight",
                 "mlp.down_proj.weight",
                 "experts.gate_up_proj",
                 "experts.down_proj",
                 "router.proj.weight",
                 "router.scale",
                 "router.per_expert_scale",
                 "self_attn.q_proj.weight",
                 "self_attn.k_proj.weight",
                 // Full-attention layers tie V to K (attention_k_eq_v):
                 // no self_attn.v_proj tensor exists in the checkpoint.
                 full ? nullptr : "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight",
                 "self_attn.q_norm.weight",
                 "self_attn.k_norm.weight",
             }) {
            if (s) expected.push_back(base + s);
        }
    }
    return expected;
}

} // namespace

GemmaTopology load_gemma_topology(const std::string &config_path)
{
    GemmaTopology top;
    std::ifstream f(config_path);
    if (!f) {
        top.error_msg = "cannot open config file: " + config_path;
        return top;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (!parse_text_config(json, top)) {
        GemmaTopology failed;
        failed.error_msg = "failed to parse config JSON: " + config_path;
        return failed;
    }

    MergeGenerationConfigEos(std::filesystem::path(config_path).parent_path().string(), top);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        // Absent vram block is fine at conversion time (it gets written);
        // only a malformed present block is an error.
        if (json.find("\"vram\"") != std::string::npos) {
            GemmaTopology failed;
            failed.error_msg = verr;
            return failed;
        }
    }
    return top;
}

arrow::Status WriteGemmaModelIndex(const std::string &output_root,
                                   const GemmaTopology &top)
{
    namespace fs = std::filesystem;
    auto dir = fs::path(output_root);
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"logestix-gemma\",\n"
      << "  \"model\": \"gemma-4-26B-A4B-it\",\n"
      << "  \"mlp_nvfp4\": true,\n      \"attn_nvfp4\": true,\n      \"head_nvfp4\": true,\n"
      << "  \"residual\": \"attn+mlp\",\n"
      << "  \"num_layers\": " << top.num_layers
      << ",\n  \"layer_types\": [";
    for (size_t i = 0; i < top.layer_types.size(); ++i) {
        if (i) j << ",";
        j << "\"" << top.layer_types[i] << "\"";
    }
    j << "]"
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"num_global_key_value_heads\": " << top.num_global_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"global_head_dim\": " << top.global_head_dim
      << ",\n  \"sliding_window\": " << top.sliding_window
      << ",\n  \"num_experts\": " << top.num_experts
      << ",\n  \"top_k_experts\": " << top.top_k_experts
      << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
      << ",\n  \"rope_theta_full\": " << top.rope_theta_full
      << ",\n  \"partial_rotary_factor\": " << top.partial_rotary_factor
      << ",\n  \"rope_theta_sliding\": " << top.rope_theta_sliding
      << ",\n  \"final_logit_softcapping\": " << top.final_logit_softcapping
      << ",\n  \"bos_id\": " << top.bos_id
      << ",\n  \"eos_ids\": [";
    for (size_t i = 0; i < top.eos_ids.size(); ++i) {
        if (i) j << ",";
        j << top.eos_ids[i];
    }
    j << "]"
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram)
      << "\n}";

    std::ofstream out(dir / "model.index.json");
    if (!out) return arrow::Status::IOError("cannot write model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

// ── Converter ───────────────────────────────────────────────────────────────

arrow::Status ConvertGemmaModel(const std::string &input_path,
                                const std::string &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    // Layer count + q-projection row counts from tensor names. Full-attention
    // layers carry the global KV geometry (q rows = heads*global_head_dim);
    // the type decision needs config geometry, so rows are collected first.
    int max_layer = -1;
    absl::btree_map<int, int64_t> q_rows_by_layer;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::Visual)
                continue;
            if (p.layer >= 0) {
                max_layer = std::max(max_layer, p.layer);
                if (p.kind == TensorKind::AttnQ && t.shape.size() == 2)
                    q_rows_by_layer[p.layer] = t.shape[0];
            }
        }
    if (max_layer < 0)
        return arrow::Status::Invalid("no recognized layer tensors in ",
                                      input_path);

    GemmaTopology top;
    top.num_layers = max_layer + 1;

    // Config.json is mandatory and authoritative.
    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    {
        std::ifstream cf(cfg_path, std::ios::binary | std::ios::ate);
        if (!cf) return arrow::Status::IOError("cannot open ", cfg_path.string());
        const auto csz = cf.tellg();
        cf.seekg(0);
        std::string cjson(static_cast<size_t>(csz), '\0');
        cf.read(cjson.data(), csz);
        if (!parse_text_config(cjson, top))
            return arrow::Status::Invalid("failed to parse ",
                                          cfg_path.string());
        MergeGenerationConfigEos(input_path, top);
    }

    int cfg_num_layers = top.num_layers;
    top.num_layers = max_layer + 1;

    // Layer-type decision: a layer is full_attention when its q width matches
    // heads*global_head_dim (verified against the real checkpoint: sliding q
    // [4096,2816] = 16x256, full q [8192,2816] = 16x512). Config's
    // layer_types array, when complete and consistent with shapes, wins.
    {
        std::vector<std::string> by_shape;
        by_shape.reserve(static_cast<size_t>(top.num_layers));
        for (int l = 0; l < top.num_layers; ++l) {
            int64_t rows = q_rows_by_layer.count(l) ? q_rows_by_layer[l] : -1;
            bool full =
                rows >= 0 &&
                rows == static_cast<int64_t>(top.num_attention_heads) *
                                top.global_head_dim;
            by_shape.push_back(full ? "full_attention" : "sliding_attention");
        }
        if (static_cast<int>(top.layer_types.size()) == top.num_layers)
            for (int l = 0; l < top.num_layers; ++l) {
                bool cfg_full = top.layer_types[static_cast<size_t>(l)] ==
                                "full_attention";
                if (cfg_full != (by_shape[static_cast<size_t>(l)] ==
                                 "full_attention"))
                    return arrow::Status::Invalid(
                        "layer_types[", l, "]=\"",
                        top.layer_types[static_cast<size_t>(l)],
                        "\" contradicts the tensor-derived attention shape");
            }
        else
            top.layer_types = std::move(by_shape);
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    auto expected = generate_gemma_expected_names(top);
    std::vector<std::string> present_names;
    absl::flat_hash_set<int> present_layers;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::Visual)
                continue;
            present_names.push_back(t.name);
            if (p.layer >= 0) present_layers.insert(p.layer);
        }
    ARROW_RETURN_NOT_OK(
        integrity::CheckTensorCoverage(expected, present_names, "gemma"));
    ARROW_RETURN_NOT_OK(
        integrity::CheckLayerContiguity(top.num_layers, present_layers));

    // Geometry cross-check: embed [V,H]; experts.gate_up dim0 == num_experts.
    int64_t embed_v = 0, embed_h = 0, gate_up_e = 0;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_tensor(t.name);
            if (p.kind == TensorKind::Embed && t.shape.size() == 2) {
                embed_v = t.shape[0];
                embed_h = t.shape[1];
            }
            if (p.kind == TensorKind::ExpertsGateUp && !t.shape.empty())
                gate_up_e = t.shape[0];
        }
    ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields({
        {"num_layers", {cfg_num_layers, top.num_layers}},
        {"vocab_size", {top.vocab_size, static_cast<int>(embed_v)}},
        {"hidden_size", {top.hidden_size, static_cast<int>(embed_h)}},
        {"num_experts", {top.num_experts, static_cast<int>(gate_up_e)}},
    }));

    // NVFP4 record helpers: dense weights are emitted as a triple of pinned
    // records (<name>.packed E2M1 nibbles | <name>.scales uint8 per 16 elems |
    // <name>.global_scale F32). The loader detects the triple and uploads
    // LsxNvfp4Weight instead of the raw BF16 tensor.
    auto quantize_bf16_to_nvfp4 =
        [](const std::vector<uint8_t> &raw, const std::vector<int64_t> &shape,
           std::vector<uint8_t> &packed, std::vector<uint8_t> &scales,
           float &global_scale) {
            return gpu::quantize_tensor_gpu(
                raw.data(), static_cast<int64_t>(raw.size()) / 2, "BF16",
                nullptr, shape, {}, packed, scales, global_scale);
        };
    auto append_nvfp4_records =
        [](const std::string &base, const std::vector<uint8_t> &packed,
           const std::vector<uint8_t> &scales, float global_scale,
           const std::vector<int64_t> &shape,
           std::vector<TensorRecord> &dst) {
            TensorRecord pk, sc, gs;
            pk.name = base + ".packed";
            pk.dtype = "U8";
            pk.shape = shape;
            pk.data = packed;
            sc.name = base + ".scales";
            sc.dtype = "U8";
            sc.shape = shape;
            sc.data = scales;
            gs.name = base + ".global_scale";
            gs.dtype = "F32";
            gs.shape = {1};
            const auto *p = reinterpret_cast<const uint8_t *>(&global_scale);
            gs.data.assign(p, p + 4);
            dst.push_back(std::move(pk));
            dst.push_back(std::move(sc));
            dst.push_back(std::move(gs));
        };

    auto pinned_dir = fs::path(output_root) / "pinned";
    fs::create_directories(pinned_dir);
    auto logestix_dir = fs::path(output_root);

    std::vector<TensorRecord> global_raw;
    std::vector<lsxcommon::DenseNvfp4Record> global_nvfp4;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    int64_t total_pinned = 0;

    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::Visual)
                continue;

            // Routed experts go to their own per-expert NVFP4 files.
            if (p.kind == TensorKind::ExpertsGateUp ||
                p.kind == TensorKind::ExpertsDown)
                continue;

            auto data = ReadTensorData(input_path, shard, t);
            if (data.empty())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);

            // Dense MLP + attention projections go out as NVFP4 (packed +
            // scales + global_scale records; loader detects the triple).
            // Tied embed/lm_head table: NVFP4 in global_nvfp4.arrow
            // (6-column DenseNvfp4Schema, qwen layout).
            if (p.kind == TensorKind::Embed) {
                lsxcommon::DenseNvfp4Record rec;
                rec.name = t.name;
                rec.shape = t.shape;
                rec.dtype = t.dtype;
                if (!gpu::quantize_tensor_gpu(
                        r.data.data(), static_cast<int64_t>(r.data.size()) / 2,
                        "BF16", nullptr, r.shape, {}, rec.packed, rec.scales,
                        rec.global_scale))
                    return arrow::Status::IOError(
                        "GPU NVFP4 quantization failed for ", t.name);
                global_nvfp4.push_back(std::move(rec));
                ++total_pinned;
                continue;
            }

            const bool attn_proj =
                p.kind == TensorKind::AttnQ || p.kind == TensorKind::AttnK ||
                p.kind == TensorKind::AttnV || p.kind == TensorKind::AttnO;
            if (p.kind == TensorKind::MlpGate || p.kind == TensorKind::MlpUp ||
                p.kind == TensorKind::MlpDown || attn_proj) {
                std::vector<uint8_t> packed, scales;
                float gs = 1.0f;
                if (!quantize_bf16_to_nvfp4(r.data, r.shape, packed, scales,
                                            gs))
                    return arrow::Status::IOError(
                        "GPU NVFP4 quantization failed for ", t.name);
                append_nvfp4_records(t.name, packed, scales, gs, r.shape,
                                     raw_by_layer[p.layer]);
                total_pinned += 3;  // packed + scales + global_scale records
                // Second plane: the residual w - dequant(base), decoded as one
                // more GEMV with acc=true (~8 bits for 2x the bytes, no new
                // codec). Attn + dense MLP only: an offline recipe ablation
                // (2026-10-24) cleared every gemma-4-26b FSI/CODE flip with
                // these at 8 bits and the routed experts at 4, and the experts
                // are what the decode streams, so upgrading them would cost
                // t/s for nothing measured.
                std::vector<uint8_t> rp, rs;
                float rgs = 1.0f;
                if (!gpu::quantize_residual_tensor_gpu(
                        r.data.data(), static_cast<int64_t>(r.data.size()) / 2,
                        "BF16", r.shape, packed, scales, gs, rp, rs, rgs))
                    return arrow::Status::IOError(
                        "GPU NVFP4 residual quantization failed for ", t.name);
                append_nvfp4_records(t.name + ".res", rp, rs, rgs, r.shape,
                                     raw_by_layer[p.layer]);
                total_pinned += 3;
                continue;
            }

            if (p.layer < 0) {
                global_raw.push_back(std::move(r));
            } else {
                raw_by_layer[p.layer].push_back(std::move(r));
            }
            ++total_pinned;
        }

    // Global + per-layer raw arrow files (GLM-style layout).
    if (!global_raw.empty()) {
        auto batch = PinnedSchema::MakeBatch(global_raw);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(
            WriteRecordBatchToFile(*batch, pinned_dir / "global.arrow"));
    }
    if (!global_nvfp4.empty()) {
        auto batch = lsxcommon::DenseNvfp4Schema::MakeBatch(global_nvfp4);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(
            WriteRecordBatchToFile(*batch, pinned_dir / "global_nvfp4.arrow"));
    }
    for (auto &[layer, recs] : raw_by_layer) {
        auto batch = PinnedSchema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(
            *batch, pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    }

    // Routed experts: slice the fused tensors per expert and quantize.
    //   w1 = gate half of experts.gate_up_proj[e]   rows [0, MI)
    //   w3 = up   half of experts.gate_up_proj[e]   rows [MI, 2*MI)
    //   w2 = experts.down_proj[e]
    for (int L = 0; L < top.num_layers; ++L) {
        std::vector<uint8_t> gu, dn;
        std::array<int64_t, 3> gu_shape{}, dn_shape{};
        for (auto &shard : model.shards)
            for (auto &t : shard.tensors) {
                auto p = classify_gemma_tensor(t.name);
                if (p.layer != L) continue;
                if (p.kind == TensorKind::ExpertsGateUp &&
                    t.shape.size() == 3) {
                    gu = ReadTensorData(input_path, shard, t);
                    gu_shape = {t.shape[0], t.shape[1], t.shape[2]};
                }
                if (p.kind == TensorKind::ExpertsDown && t.shape.size() == 3) {
                    dn = ReadTensorData(input_path, shard, t);
                    dn_shape = {t.shape[0], t.shape[1], t.shape[2]};
                }
            }
        if (gu.empty() || dn.empty())
            return arrow::Status::Invalid(
                "missing routed-expert tensors in layer ", L);

        const int E = static_cast<int>(gu_shape[0]);
        const int twoMI = static_cast<int>(gu_shape[1]);
        const int H = static_cast<int>(gu_shape[2]);
        const int MI = twoMI / 2;
        const int DMI = static_cast<int>(dn_shape[2]); // down inner dim
        if (DMI != MI)
            return arrow::Status::Invalid(
                "moe_intermediate mismatch between experts.gate_up_proj and "
                "experts.down_proj in layer ", L);

        for (int e = 0; e < E; ++e) {
            ExpertData raw;
            raw.w1_dtype = raw.w2_dtype = raw.w3_dtype = "BF16";

            size_t row_bytes = static_cast<size_t>(H) * 2;
            size_t half_bytes = static_cast<size_t>(MI) * row_bytes;
            size_t gu_base = static_cast<size_t>(e) * twoMI * row_bytes;
            raw.w1_raw.assign(gu.begin() + static_cast<long>(gu_base),
                              gu.begin() +
                                  static_cast<long>(gu_base + half_bytes));
            raw.w1_shape = {MI, H};
            raw.w3_raw.assign(gu.begin() + static_cast<long>(gu_base + half_bytes),
                              gu.begin() +
                                  static_cast<long>(gu_base + 2 * half_bytes));
            raw.w3_shape = {MI, H};

            size_t dn_row_bytes = static_cast<size_t>(MI) * 2;
            size_t dn_base = static_cast<size_t>(e) * H * dn_row_bytes;
            raw.w2_raw.assign(dn.begin() + static_cast<long>(dn_base),
                              dn.begin() +
                                  static_cast<long>(dn_base + H * dn_row_bytes));
            raw.w2_shape = {H, MI};

            auto packed = PackExpertNVFP4(raw);
            if (!packed.ok())
                return packed.status();
            ARROW_RETURN_NOT_OK(SaveExpertToArrow(*packed, L, e, logestix_dir));
        }
        spdlog::info("  gemma layer {} wrote {} NVFP4 experts", L, E);
    }

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Conversion-time VRAM profile. Experts are RoutedExpert-class.
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim > 0
                        ? top.head_dim * top.num_key_value_heads
                        : 0;
        vg.num_moe_layers = top.num_layers;
        vg.experts_per_layer = top.num_experts;
        vg.context_length =
            ReadContextLength(cfg_path.string());

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});

        top.vram = EstimateVram(vram_inventory,
            {+[](const std::string &n) {
                 return gemma_tensor_is_nvfp4(
                     classify_gemma_tensor(n).kind);
             },
             +[](const std::string &n) -> VramTensorClass {
                 auto k = classify_gemma_tensor(n).kind;
                 return (k == TensorKind::ExpertsGateUp ||
                         k == TensorKind::ExpertsDown)
                            ? VramTensorClass::RoutedExpert
                            : VramTensorClass::Resident;
             },
             // attn + dense MLP carry the residual plane (experts do not).
             +[](const std::string &n) {
                 auto k = classify_gemma_tensor(n).kind;
                 return k == TensorKind::AttnQ || k == TensorKind::AttnK ||
                        k == TensorKind::AttnV || k == TensorKind::AttnO ||
                        k == TensorKind::MlpGate || k == TensorKind::MlpUp ||
                        k == TensorKind::MlpDown;
             }},
            vg);
        spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
                     FormatBytesHuman(top.vram.min_bytes),
                     FormatBytesHuman(top.vram.optimal_bytes),
                     FormatBytesHuman(top.vram.max_bytes),
                     top.vram.context_tokens);
    }

    ARROW_RETURN_NOT_OK(WriteGemmaModelIndex(output_root, top));

    // Tokenizer bridge (mandatory): real checkpoint resolves specials via
    // tokenizer_config.json strings ("<bos>"/"<eos>"/"<pad>").
    auto tok_path = fs::path(input_path) / "tokenizer.json";
    if (!fs::exists(tok_path))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    // Gemma-4 is raw-text BPE (tokenizer.json vocab/merges verbatim,
    // U+2581 space normalization, byte fallback), written as
    // tokenizer_kind 11.
    TokenizerSpecialPolicy tok_policy;
    tok_policy.pretokenizer = 9;
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tok_path.string(), pinned_dir,
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true}, tok_policy));

    return arrow::Status::OK();
}

} // namespace gemma
} // namespace lsxcommon
