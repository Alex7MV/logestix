#include "lsxcommon/muse_dense_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <simdjson.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
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
        else if (src["pad_id"].get_int64().get(v) == simdjson::SUCCESS)
            t.pad_id = static_cast<int>(v);   // converted model.index.json

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

// ── Index writer ────────────────────────────────────────────────────────────

arrow::Status WriteMuseDenseModelIndex(const std::string &output_root,
                                       const MuseDenseTopology &top)
{
    namespace fs = std::filesystem;
    const auto dir = fs::path(output_root) / "logestix";
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"" << frozen::kFormatTag << "\",\n"
      << "  \"model\": \"" << frozen::kModelType << "\",\n"
      << "  \"mlp_nvfp4\": true,\n"
      << "  \"attn_nvfp4\": true,\n"
      << "  \"head_fp16\": true,\n"
      << "  \"num_layers\": " << top.num_layers << ",\n  \"layer_types\": [";
    for (size_t i = 0; i < top.layer_types.size(); ++i) {
        if (i) j << ",";
        j << "\"" << top.layer_types[i] << "\"";
    }
    j << "]"
      << ",\n  \"layer_rope_theta\": [";
    for (size_t i = 0; i < top.layer_rope_theta.size(); ++i) {
        if (i) j << ",";
        j << top.layer_rope_theta[i];
    }
    j << "]"
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"sliding_window\": " << top.sliding_window
      << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
      << ",\n  \"post_norm_eps\": " << top.post_norm_eps
      << ",\n  \"rope_theta\": " << top.rope_theta
      << ",\n  \"qk_scale_factor\": " << top.qk_scale_factor
      << ",\n  \"output_multiplier\": " << top.output_multiplier
      << ",\n  \"final_logit_softcapping\": " << top.final_logit_softcapping
      << ",\n  \"bos_id\": " << top.bos_id << ",\n  \"eos_ids\": [";
    for (size_t i = 0; i < top.eos_ids.size(); ++i) {
        if (i) j << ",";
        j << top.eos_ids[i];
    }
    j << "]"
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n  \"skipped_visual_tensors\": " << top.skipped_visual_tensors
      << ",\n" << FormatVramJson(top.vram) << "\n}";

    std::ofstream out(dir / "model.index.json");
    if (!out) return arrow::Status::IOError("cannot write model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

// ── Converter ───────────────────────────────────────────────────────────────

namespace {

std::vector<std::string> generate_expected_names(const MuseDenseTopology &top)
{
    std::vector<std::string> expected;
    expected.push_back("model.language_model.embed_tokens.weight");
    expected.push_back("lm_head.weight");
    expected.push_back("model.language_model.norm.weight");

    for (int l = 0; l < top.num_layers; ++l) {
        const std::string base =
            absl::StrCat("model.language_model.layers.", l, ".");
        // EVERY layer carries all five attention projections (the gate is
        // present on sliding AND full layers) and all four centered norms.
        for (const char *s : {
                 "input_layernorm.weight",
                 "post_attention_layernorm.weight",
                 "pre_feedforward_layernorm.weight",
                 "post_feedforward_layernorm.weight",
                 "mlp.gate_proj.weight",
                 "mlp.up_proj.weight",
                 "mlp.down_proj.weight",
                 "self_attn.q_proj.weight",
                 "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight",
                 "self_attn.gate_proj.weight",
             }) {
            expected.push_back(base + s);
        }
    }
    return expected;
}

// STRICT coverage: a tensor that is neither expected nor known ballast is a
// hard error. Silently skipping it is how a wrong checkpoint produces a model
// that "answers, but not the model's answers".
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
        "unexpected tensor(s) for the Muse-Glimmer layout: ",
        absl::StrJoin(extra, ", "));
}

// Frozen-passport gate (protocol step 2): the source config must match the
// compile-time geometry unless the synthetic-test override is set.
arrow::Status CheckFrozenGeometry(const MuseDenseTopology &top)
{
    namespace fz = frozen;
    std::vector<std::string> bad;
    auto eq = [&](const char *name, long long want, long long got) {
        if (want != got)
            bad.push_back(absl::StrCat(name, ": frozen ", want, " != config ",
                                       got));
    };
    eq("num_layers", fz::kNumLayers, top.num_layers);
    eq("hidden_size", fz::kHiddenSize, top.hidden_size);
    eq("intermediate_size", fz::kIntermediateSize, top.intermediate_size);
    eq("vocab_size", fz::kVocabSize, top.vocab_size);
    eq("num_attention_heads", fz::kNumHeads, top.num_attention_heads);
    eq("num_key_value_heads", fz::kNumKVHeads, top.num_key_value_heads);
    eq("head_dim", fz::kHeadDim, top.head_dim);
    eq("sliding_window", fz::kSlidingWindow, top.sliding_window);
    if (static_cast<int>(top.layer_types.size()) == top.num_layers) {
        for (int l = 0; l < top.num_layers; ++l) {
            if (top.IsFull(l) != fz::IsFullLayer(l))
                bad.push_back(absl::StrCat("layer_types[", l,
                                           "] contradicts the frozen 3:1 "
                                           "sliding/full pattern"));
        }
    }
    if (bad.empty()) return arrow::Status::OK();
    return arrow::Status::Invalid("Muse-Glimmer frozen geometry mismatch: ",
                                  absl::StrJoin(bad, "; "));
}

} // namespace

arrow::Status ConvertMuseDenseModel(const std::string &input_path,
                                    const std::string &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    int max_layer = -1;
    int64_t skipped_visual = 0;
    std::vector<std::string> unclassified;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_muse_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown) {
                unclassified.push_back(t.name);
                continue;
            }
            if (p.kind == TensorKind::Visual) {
                ++skipped_visual;
                continue;
            }
            if (p.layer >= 0) max_layer = std::max(max_layer, p.layer);
        }
    if (!unclassified.empty()) {
        std::sort(unclassified.begin(), unclassified.end());
        return arrow::Status::Invalid(
            "unclassified tensor(s) — the Muse-Glimmer converter refuses to "
            "drop them: ",
            absl::StrJoin(unclassified, ", "));
    }
    if (max_layer < 0)
        return arrow::Status::Invalid("no recognized layer tensors in ",
                                      input_path);

    // config.json is mandatory and authoritative.
    const auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    MuseDenseTopology top;
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
        MergeGenerationConfig(input_path, top);
    }

    const int cfg_num_layers = top.num_layers;
    top.num_layers = max_layer + 1;
    if (static_cast<int>(top.layer_types.size()) != top.num_layers)
        return arrow::Status::Invalid(
            "config layer_types size ", top.layer_types.size(),
            " does not cover the ", top.num_layers, " tensor layers");
    if (top.layer_rope_theta.empty()) {
        for (int l = 0; l < top.num_layers; ++l)
            top.layer_rope_theta.push_back(top.IsFull(l) ? 0.0
                                                         : top.rope_theta);
    }

    if (!SyntheticGeometryAllowed()) {
        ARROW_RETURN_NOT_OK(CheckFrozenGeometry(top));
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    const auto expected = generate_expected_names(top);
    std::vector<std::string> present_names;
    absl::flat_hash_set<int> present_layers;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_muse_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown ||
                p.kind == TensorKind::Visual)
                continue;
            present_names.push_back(t.name);
            if (p.layer >= 0) present_layers.insert(p.layer);
        }
    ARROW_RETURN_NOT_OK(
        integrity::CheckTensorCoverage(expected, present_names, "muse_dense"));
    ARROW_RETURN_NOT_OK(CheckNoUnexpectedTensors(expected, present_names));
    ARROW_RETURN_NOT_OK(
        integrity::CheckLayerContiguity(top.num_layers, present_layers));

    // Geometry cross-check: embed [V,H].
    int64_t embed_v = 0, embed_h = 0;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_muse_dense_tensor(t.name);
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

    // Dense projections are emitted as a triple of pinned records
    // (<name>.packed E2M1 nibbles | <name>.scales E4M3 per 16 | <name>.
    // global_scale F32); the loader detects the triple and uploads an
    // LsxNvfp4Weight. embed/lm_head/norms stay raw BF16 (the loader converts
    // to FP16 at upload — norms and the untied head are never quantized).
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

    std::vector<TensorRecord> global_raw;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    int64_t total_pinned = 0;

    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_muse_dense_tensor(t.name);
            if (p.kind == TensorKind::Unknown || p.kind == TensorKind::Visual)
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

            if (muse_dense_tensor_is_nvfp4(p.kind)) {
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
    for (auto &[layer, recs] : raw_by_layer) {
        auto batch = PinnedSchema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(
            *batch, pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    }
    spdlog::info("muse-glimmer: {} layers converted, {} vision tensors counted "
                 "(never applied), {} pinned records",
                 top.num_layers, skipped_visual, total_pinned);

    top.total_pinned_tensors = static_cast<int>(total_pinned);
    top.skipped_visual_tensors = static_cast<int>(skipped_visual);

    // Conversion-time VRAM profile. Everything this family stores is resident
    // (no routed experts); the vision ballast is NOT uploaded. embed/lm_head
    // stay FP16 (2 B/elt), the projections are NVFP4.
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
                if (classify_muse_dense_tensor(t.name).kind ==
                    TensorKind::Visual)
                    continue;
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
            }

        top.vram = EstimateVram(
            vram_inventory,
            {+[](const std::string &n) {
                 return muse_dense_tensor_is_nvfp4(
                     classify_muse_dense_tensor(n).kind);
             },
             +[](const std::string &) { return VramTensorClass::Resident; }},
            vg);
        spdlog::info("VRAM profile: min={} optimal={} max={} (ctx={})",
                     FormatBytesHuman(top.vram.min_bytes),
                     FormatBytesHuman(top.vram.optimal_bytes),
                     FormatBytesHuman(top.vram.max_bytes),
                     top.vram.context_tokens);
    }

    ARROW_RETURN_NOT_OK(WriteMuseDenseModelIndex(output_root, top));

    // Tokenizer bridge (mandatory): o200k/Llama-3 byte-level BPE, kind 3.
    // The ATEM markers are added tokens (special:true) above the BPE table;
    // the encoder matches them literally before BPE.
    const auto tok_path = fs::path(input_path) / "tokenizer.json";
    if (!fs::exists(tok_path))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    TokenizerSpecialPolicy tok_policy;
    tok_policy.pretokenizer = 1;
    tok_policy.bos_id = top.bos_id;
    tok_policy.eos_id = top.eos_ids.empty() ? INT32_MIN : top.eos_ids.front();
    tok_policy.pad_id = top.pad_id;
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tok_path.string(), pinned_dir,
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true,
                                       .pad_required = true}, tok_policy));

    return arrow::Status::OK();
}

} // namespace muse_dense
} // namespace lsxcommon