#include "lsxcommon/gemma_dense_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>

namespace lsxcommon {
namespace gemma_dense {

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

bool parse_text_config(const std::string &json, GemmaDenseTopology &t)
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
    t.intermediate_size =
        get_i(src, "intermediate_size", t.intermediate_size);
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
    t.num_attention_heads =
        get_i(src, "num_attention_heads", t.num_attention_heads);
    t.num_key_value_heads =
        get_i(src, "num_key_value_heads", t.num_key_value_heads);
    t.num_global_key_value_heads = get_i(
        src, "num_global_key_value_heads", t.num_global_key_value_heads);
    t.head_dim = get_i(src, "head_dim", t.head_dim);
    t.global_head_dim = get_i(src, "global_head_dim", t.global_head_dim);
    t.sliding_window = get_i(src, "sliding_window", t.sliding_window);
    t.rms_norm_eps =
        static_cast<float>(get_d(src, "rms_norm_eps", t.rms_norm_eps));
    t.final_logit_softcapping = static_cast<float>(
        get_d(src, "final_logit_softcapping", t.final_logit_softcapping));

    // Per-layer-type RoPE parameters (the real checkpoint nests them under
    // rope_parameters.{full_attention,sliding_attention}).
    if (auto ro = src["rope_parameters"].get_object();
        ro.error() == simdjson::SUCCESS) {
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

    if (auto lt = src["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS) {
        t.layer_types.clear();
        for (auto v : lt.value()) {
            std::string_view sv;
            if (v.get_string().get(sv) == simdjson::SUCCESS)
                t.layer_types.emplace_back(sv);
        }
    }

    // Special ids. The top-level config carries the full eos array (text_config
    // keeps only the first element); the converted index stores `eos_ids`.
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

std::vector<std::string> generate_expected_names(const GemmaDenseTopology &top)
{
    std::vector<std::string> expected;
    expected.push_back("model.language_model.embed_tokens.weight");
    expected.push_back("model.language_model.norm.weight");

    for (int l = 0; l < top.num_layers; ++l) {
        const std::string base =
            absl::StrCat("model.language_model.layers.", l, ".");
        const bool full = top.IsFull(l);
        for (const char *s : {
                 "input_layernorm.weight",
                 "post_attention_layernorm.weight",
                 "pre_feedforward_layernorm.weight",
                 "post_feedforward_layernorm.weight",
                 "layer_scalar",
                 "mlp.gate_proj.weight",
                 "mlp.up_proj.weight",
                 "mlp.down_proj.weight",
                 "self_attn.q_proj.weight",
                 "self_attn.k_proj.weight",
                 // Full-attention layers tie V to K (attention_k_eq_v): the
                 // checkpoint ships no self_attn.v_proj for them.
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

std::vector<uint8_t> read_tensor_data(
    const std::filesystem::path &input_path,
    const SafetensorsModel::Shard &shard, const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((input_path / shard.filename).string(), std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    const int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

// HF uses generation_config's eos_token_id for stopping; add any ids beyond
// the config array (the real checkpoint ends a turn on 106 and a tool turn on
// 50).
void MergeGenerationConfigEos(const std::string &dir, GemmaDenseTopology &top)
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
    }
}

// STRICT coverage: this family has no second branch to accommodate, so a
// tensor that is neither expected nor known ballast is a hard error. Silently
// skipping it is how a MoE checkpoint gets fed to a dense loader (or the
// reverse) and produces a model that "answers, but not the model's answers".
arrow::Status CheckNoUnexpectedTensors(
    const std::vector<std::string> &expected,
    const std::vector<std::string> &present)
{
    const absl::flat_hash_set<std::string> want(expected.begin(),
                                                expected.end());
    std::vector<std::string> extra;
    for (const auto &n : present)
        if (!want.count(n)) extra.push_back(n);
    if (extra.empty()) return arrow::Status::OK();
    std::sort(extra.begin(), extra.end());
    return arrow::Status::Invalid(
        "unexpected tensor(s) for the dense Gemma-4 layout: ",
        absl::StrJoin(extra, ", "));
}

} // namespace

ParsedTensor classify_gemma_dense_tensor(const std::string &name)
{
    using K = TensorKind;

    // Multimodal fat (vision tower / vision embed) — stored and counted, never
    // applied (this is a text-only runtime).
    if (name.rfind("model.vision_tower.", 0) == 0 ||
        name.rfind("vision_tower.", 0) == 0 ||
        name.rfind("model.embed_vision.", 0) == 0 ||
        name.rfind("embed_vision.", 0) == 0)
        return {K::Visual, -1};

    if (name == "model.language_model.embed_tokens.weight")
        return {K::Embed, -1};
    if (name == "model.language_model.norm.weight") return {K::FinalNorm, -1};

    auto P = [&](K kind, const char *suffix) -> std::optional<ParsedTensor> {
        return match_layer(name, re_linear(suffix), kind);
    };

    if (auto p = P(K::LayerScalar, R"(layer_scalar$)")) return *p;
    if (auto p = P(K::AttnQNorm, R"(self_attn\.q_norm\.weight$)")) return *p;
    if (auto p = P(K::AttnKNorm, R"(self_attn\.k_norm\.weight$)")) return *p;
    if (auto p = P(K::AttnQ, R"(self_attn\.q_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnK, R"(self_attn\.k_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnV, R"(self_attn\.v_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnO, R"(self_attn\.o_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpGate, R"(mlp\.gate_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpUp, R"(mlp\.up_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpDown, R"(mlp\.down_proj\.weight$)")) return *p;
    if (auto p = P(K::InputNorm, R"(input_layernorm\.weight$)")) return *p;
    if (auto p = P(K::PostNorm, R"(post_attention_layernorm\.weight$)"))
        return *p;
    if (auto p = P(K::PreFFN, R"(pre_feedforward_layernorm\.weight$)")) return *p;
    if (auto p = P(K::PostFFN, R"(post_feedforward_layernorm\.weight$)"))
        return *p;

    return {K::Unknown, -1};
}

bool gemma_dense_tensor_is_nvfp4(TensorKind kind)
{
    switch (kind) {
    case TensorKind::AttnQ:
    case TensorKind::AttnK:
    case TensorKind::AttnV:
    case TensorKind::AttnO:
    case TensorKind::MlpGate:
    case TensorKind::MlpUp:
    case TensorKind::MlpDown:
    case TensorKind::Embed:
        return true;
    default:
        return false;
    }
}

// ── Topology ────────────────────────────────────────────────────────────────

GemmaDenseTopology load_gemma_dense_topology(const std::string &config_path)
{
    GemmaDenseTopology top;
    std::ifstream f(config_path);
    if (!f) {
        top.error_msg = "cannot open config file: " + config_path;
        return top;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (!parse_text_config(json, top)) {
        GemmaDenseTopology failed;
        failed.error_msg = "failed to parse config JSON: " + config_path;
        return failed;
    }

    MergeGenerationConfigEos(
        std::filesystem::path(config_path).parent_path().string(), top);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        // Absent vram block is fine at conversion time (it gets written);
        // only a malformed present block is an error.
        if (json.find("\"vram\"") != std::string::npos) {
            GemmaDenseTopology failed;
            failed.error_msg = verr;
            return failed;
        }
    }
    return top;
}

arrow::Status WriteGemmaDenseModelIndex(const std::string &output_root,
                                       const GemmaDenseTopology &top)
{
    namespace fs = std::filesystem;
    const auto dir = fs::path(output_root) / "logestix";
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"" << kFormatTag << "\",\n"
      << "  \"model\": \"gemma-4-31B-it\",\n"
      << "  \"mlp_nvfp4\": true,\n"
      << "  \"attn_nvfp4\": true,\n"
      << "  \"head_nvfp4\": true,\n"
      << "  \"num_layers\": " << top.num_layers << ",\n  \"layer_types\": [";
    for (size_t i = 0; i < top.layer_types.size(); ++i) {
        if (i) j << ",";
        j << "\"" << top.layer_types[i] << "\"";
    }
    j << "]"
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"num_global_key_value_heads\": " << top.num_global_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"global_head_dim\": " << top.global_head_dim
      << ",\n  \"sliding_window\": " << top.sliding_window
      << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
      << ",\n  \"rope_theta_full\": " << top.rope_theta_full
      << ",\n  \"partial_rotary_factor\": " << top.partial_rotary_factor
      << ",\n  \"rope_theta_sliding\": " << top.rope_theta_sliding
      << ",\n  \"final_logit_softcapping\": " << top.final_logit_softcapping
      << ",\n  \"bos_id\": " << top.bos_id << ",\n  \"eos_ids\": [";
    for (size_t i = 0; i < top.eos_ids.size(); ++i) {
        if (i) j << ",";
        j << top.eos_ids[i];
    }
    j << "]"
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram) << "\n}";

    std::ofstream out(dir / "model.index.json");
    if (!out) return arrow::Status::IOError("cannot write model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

// ── Converter ───────────────────────────────────────────────────────────────

arrow::Status ConvertGemmaDenseModel(const std::string &input_path,
                                     const std::string &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    // Layer count + q-projection row counts from tensor names. Full-attention
    // layers carry the global KV geometry (q rows = heads*global_head_dim), so
    // the type decision needs config geometry: rows are collected first.
    int max_layer = -1;
    absl::btree_map<int, int64_t> q_rows_by_layer;
    int64_t skipped_visual = 0;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;
            if (p.kind == TensorKind::Visual) {
                ++skipped_visual;
                continue;
            }
            if (p.layer >= 0) {
                max_layer = std::max(max_layer, p.layer);
                if (p.kind == TensorKind::AttnQ && t.shape.size() == 2)
                    q_rows_by_layer[p.layer] = t.shape[0];
            }
        }
    if (max_layer < 0)
        return arrow::Status::Invalid("no recognized layer tensors in ",
                                      input_path);

    // config.json is mandatory and authoritative.
    const auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    GemmaDenseTopology top;
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

    const int cfg_num_layers = top.num_layers;
    top.num_layers = max_layer + 1;

    // Layer-type decision: a layer is full_attention when its q width matches
    // heads*global_head_dim (real checkpoint: sliding q [8192,5376] =
    // 32x256, full q [16384,5376] = 32x512). The config's layer_types array,
    // when complete and consistent with the shapes, wins; a contradiction is an
    // error, never a silent re-derivation.
    {
        std::vector<std::string> by_shape;
        by_shape.reserve(static_cast<size_t>(top.num_layers));
        for (int l = 0; l < top.num_layers; ++l) {
            const int64_t rows =
                q_rows_by_layer.count(l) ? q_rows_by_layer[l] : -1;
            const bool full =
                rows >= 0 &&
                rows == static_cast<int64_t>(top.num_attention_heads) *
                            top.global_head_dim;
            by_shape.push_back(full ? "full_attention" : "sliding_attention");
        }
        if (static_cast<int>(top.layer_types.size()) == top.num_layers) {
            for (int l = 0; l < top.num_layers; ++l) {
                const bool cfg_full =
                    top.layer_types[static_cast<size_t>(l)] == "full_attention";
                if (cfg_full !=
                    (by_shape[static_cast<size_t>(l)] == "full_attention"))
                    return arrow::Status::Invalid(
                        "layer_types[", l, "]=\"",
                        top.layer_types[static_cast<size_t>(l)],
                        "\" contradicts the tensor-derived attention shape");
            }
        } else {
            top.layer_types = std::move(by_shape);
        }
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    const auto expected = generate_expected_names(top);
    std::vector<std::string> present_names;
    absl::flat_hash_set<int> present_layers;
    std::vector<std::string> unclassified;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown) {
                unclassified.push_back(t.name);
                continue;
            }
            if (p.kind == TensorKind::Visual) continue;
            present_names.push_back(t.name);
            if (p.layer >= 0) present_layers.insert(p.layer);
        }
    if (!unclassified.empty()) {
        std::sort(unclassified.begin(), unclassified.end());
        return arrow::Status::Invalid(
            "unclassified tensor(s) — the dense Gemma-4 converter refuses to "
            "drop them: ",
            absl::StrJoin(unclassified, ", "));
    }
    ARROW_RETURN_NOT_OK(
        integrity::CheckTensorCoverage(expected, present_names, "gemma_dense"));
    ARROW_RETURN_NOT_OK(CheckNoUnexpectedTensors(expected, present_names));
    ARROW_RETURN_NOT_OK(
        integrity::CheckLayerContiguity(top.num_layers, present_layers));

    // Geometry cross-check: embed [V,H].
    int64_t embed_v = 0, embed_h = 0;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_dense_tensor(t.name);
            if (p.kind == TensorKind::Embed && t.shape.size() == 2) {
                embed_v = t.shape[0];
                embed_h = t.shape[1];
            }
        }
    ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields({
        {"num_layers", {cfg_num_layers, top.num_layers}},
        {"vocab_size", {top.vocab_size, static_cast<int>(embed_v)}},
        {"hidden_size", {top.hidden_size, static_cast<int>(embed_h)}},
    }));

    // Dense weights are emitted as a triple of pinned records
    // (<name>.packed E2M1 nibbles | <name>.scales E4M3 per 16 | <name>.
    // global_scale F32); the loader detects the triple and uploads an
    // LsxNvfp4Weight instead of the raw BF16 tensor.
    auto append_nvfp4_records = [](const std::string &base,
                                   const std::vector<uint8_t> &packed,
                                   const std::vector<uint8_t> &scales,
                                   float global_scale,
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

    const auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    fs::create_directories(pinned_dir);
    const auto logestix_dir = fs::path(output_root) / "logestix";

    std::vector<TensorRecord> global_raw;
    std::vector<lsxcommon::DenseNvfp4Record> global_nvfp4;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    int64_t total_pinned = 0;

    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_gemma_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::Visual)
                continue;

            auto data = read_tensor_data(input_path, shard, t);
            if (data.empty())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);

            // Tied embed/lm_head table: NVFP4 in global_nvfp4.arrow
            // (6-column DenseNvfp4Schema, the repo's shared layout).
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

            if (gemma_dense_tensor_is_nvfp4(p.kind)) {
                std::vector<uint8_t> packed, scales;
                float gs = 1.0f;
                if (!gpu::quantize_tensor_gpu(
                        r.data.data(), static_cast<int64_t>(r.data.size()) / 2,
                        "BF16", nullptr, r.shape, {}, packed, scales, gs))
                    return arrow::Status::IOError(
                        "GPU NVFP4 quantization failed for ", t.name);
                append_nvfp4_records(t.name, packed, scales, gs, r.shape,
                                     raw_by_layer[p.layer]);
                total_pinned += 3;  // packed + scales + global_scale records
                continue;
            }

            if (p.layer < 0) {
                global_raw.push_back(std::move(r));
            } else {
                raw_by_layer[p.layer].push_back(std::move(r));
            }
            ++total_pinned;
        }

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
    spdlog::info("gemma-4 dense: {} layers converted, {} vision tensors stored "
                 "and counted, {} pinned records",
                 top.num_layers, skipped_visual, total_pinned);

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Conversion-time VRAM profile. Everything this family stores is resident
    // (there are no routed experts); the vision ballast is NOT uploaded, so it
    // is excluded from the resident inventory.
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = top.num_layers;
        vg.num_moe_layers = 0;
        vg.experts_per_layer = 0;
        vg.context_length = ReadContextLength(cfg_path.string());

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors) {
                if (classify_gemma_dense_tensor(t.name).kind ==
                    TensorKind::Visual)
                    continue;
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
            }

        top.vram = EstimateVram(
            vram_inventory,
            {+[](const std::string &n) {
                 return gemma_dense_tensor_is_nvfp4(
                     classify_gemma_dense_tensor(n).kind);
             },
             +[](const std::string &) { return VramTensorClass::Resident; }},
            vg);
        spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
                     FormatBytesHuman(top.vram.min_bytes),
                     FormatBytesHuman(top.vram.optimal_bytes),
                     FormatBytesHuman(top.vram.max_bytes),
                     top.vram.context_tokens);
    }

    ARROW_RETURN_NOT_OK(WriteGemmaDenseModelIndex(output_root, top));

    // Tokenizer bridge (mandatory): the real checkpoint resolves specials via
    // tokenizer_config.json strings ("<bos>"/"<eos>"/"<pad>").
    const auto tok_path = fs::path(input_path) / "tokenizer.json";
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

} // namespace gemma_dense
} // namespace lsxcommon
