#include "lsxcommon/beam_moe_convert.h"

#include "beam_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "tensor_index_parse.h"

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

namespace lsxcommon::beam {
namespace {

using frozen::kType;

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
auto get_s = [](simdjson::ondemand::object &o, const char *k,
                std::string dflt) {
    std::string_view s;
    if (o[k].get_string().get(s) != simdjson::SUCCESS) return dflt;
    return std::string(s);
};
auto get_b = [](simdjson::ondemand::object &o, const char *k, bool dflt) {
    bool v;
    if (o[k].get_bool().get(v) != simdjson::SUCCESS) return dflt;
    return v;
};

// Cross-checks the parsed config against the frozen passport. Every branch is a
// silently-wrong-token risk: a wrong layer table decodes garbage, a wrong
// window truncates context, a wrong rope/NoPE set rotates the wrong layers.
bool CheckFrozenGeometry(const BeamTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != kHidden) return bad("hidden_size", top.hidden_size, kHidden);
    if (top.num_layers != kLayers) return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKvHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKvHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   kMoeIntermediate);
    if (top.dense_intermediate_size != kDenseIntermediate)
        return bad("intermediate_size", top.dense_intermediate_size,
                   kDenseIntermediate);
    if (top.sliding_window != kWindow)
        return bad("sliding_window", top.sliding_window, kWindow);
    if (top.n_group != kNGroups) return bad("n_group", top.n_group, kNGroups);
    if (top.topk_group != kTopkGroup)
        return bad("topk_group", top.topk_group, kTopkGroup);

    if (static_cast<int>(top.layer_types.size()) != kLayers)
        return bad("layer_types.size()",
                   static_cast<int>(top.layer_types.size()), kLayers);
    for (int l = 0; l < kLayers; ++l) {
        const bool want_full = IsFullAttn(l);
        const bool got_full =
            top.layer_types[static_cast<size_t>(l)] == "full_attention";
        if (want_full != got_full)
            return bad("layer_types[l] full/sliding", l, want_full ? 1 : 0);
    }
    if (static_cast<int>(top.mlp_layer_types.size()) != kLayers)
        return bad("mlp_layer_types.size()",
                   static_cast<int>(top.mlp_layer_types.size()), kLayers);
    for (int l = 0; l < kLayers; ++l) {
        const bool want_dense = IsDenseLayer(l);
        const bool got_dense =
            top.mlp_layer_types[static_cast<size_t>(l)] == "dense";
        if (want_dense != got_dense)
            return bad("mlp_layer_types[l] dense/sparse", l,
                       want_dense ? 1 : 0);
    }
    return true;
}

void ReadGenerationIds(const std::filesystem::path &path, int &bos, int &eos,
                       int &pad, std::vector<int> *all_eos)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty()) return;
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad_json(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad_json).get(doc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return;
    const int b = get_i(root, "bos_token_id", -1);
    // eos_token_id may be an int or a list; the first id is the tokenizer stop.
    int e = -1;
    if (auto arr = root["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t id = 0;
            if (v.get_int64().get(id) == simdjson::SUCCESS) {
                if (e < 0) e = static_cast<int>(id);
                if (all_eos) all_eos->push_back(static_cast<int>(id));
            }
        }
    } else {
        e = get_i(root, "eos_token_id", -1);
    }
    const int p = get_i(root, "pad_token_id", -1);
    if (b >= 0) bos = b;
    if (e >= 0) eos = e;
    if (p >= 0) pad = p;
}

} // namespace

BeamTopology ParseBeamConfig(const std::string &input_dir)
{
    BeamTopology top;
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

    const std::string model_type = get_s(root, "model_type", "");
    if (model_type != "beam")
        top.error_msg = "model_type must be beam (got " + model_type + ")";

    top.hidden_size = get_i(root, "hidden_size", 0);
    top.num_layers = get_i(root, "num_hidden_layers", 0);
    top.vocab_size = get_i(root, "vocab_size", 0);
    top.num_attention_heads = get_i(root, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(root, "num_key_value_heads", 0);
    top.head_dim = get_i(root, "head_dim", 0);
    top.dense_intermediate_size = get_i(root, "intermediate_size", 0);
    top.num_experts = get_i(root, "num_experts", get_i(root, "num_local_experts", 0));
    top.num_experts_per_tok = get_i(root, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(root, "moe_intermediate_size", 0);
    top.n_group = get_i(root, "n_group", 0);
    top.topk_group = get_i(root, "topk_group", 0);
    top.routed_scaling_factor = get_d(root, "routed_scaling_factor", 0.0);
    top.final_logit_softcapping = get_d(root, "final_logit_softcapping", 0.0);
    top.sliding_window = get_i(root, "sliding_window", 0);
    top.rms_norm_eps = get_d(root, "rms_norm_eps", 0.0);
    top.bos_token_id = get_i(root, "bos_token_id", -1);
    top.pad_token_id = get_i(root, "pad_token_id", -1);

    // rope_parameters is keyed by layer type: sliding_attention carries the
    // rotary config, full_attention is null (NoPE). This is the single most
    // important semantic in the checkpoint.
    {
        simdjson::ondemand::object rp;
        if (root["rope_parameters"].get_object().get(rp) !=
            simdjson::SUCCESS) {
            top.error_msg = "rope_parameters must be a per-layer-type object";
        } else {
            simdjson::ondemand::object sliding;
            if (rp["sliding_attention"].get_object().get(sliding) !=
                simdjson::SUCCESS) {
                top.error_msg =
                    "rope_parameters.sliding_attention must be an object";
            } else {
                top.rope_theta = get_d(sliding, "rope_theta", 0.0);
                const std::string rt = get_s(sliding, "rope_type", "default");
                if (rt != "default")
                    top.error_msg = "sliding rope_type must be default (got " +
                                    rt + ")";
            }
            simdjson::ondemand::value full;
            if (rp["full_attention"].get(full) != simdjson::SUCCESS) {
                top.error_msg =
                    "rope_parameters.full_attention must be present (null)";
            } else if (full.type() != simdjson::ondemand::json_type::null) {
                top.error_msg =
                    "rope_parameters.full_attention must be null (full layers "
                    "are NoPE + SSMax)";
            }
        }
    }

    // eos_token_id is a LIST in this checkpoint.
    if (auto arr = root["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t id = 0;
            if (v.get_int64().get(id) == simdjson::SUCCESS)
                top.eos_ids.push_back(static_cast<int>(id));
        }
    } else {
        const int e = get_i(root, "eos_token_id", -1);
        if (e >= 0) top.eos_ids.push_back(e);
    }
    if (!top.eos_ids.empty()) top.eos_token_id = top.eos_ids.front();
    if (top.eos_ids.size() > 1) top.eos_token_id_alt = top.eos_ids[1];

    // Layer tables: the config carries them explicitly.
    if (auto arr = root["layer_types"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto e : arr.value()) {
            std::string_view s;
            if (e.get_string().get(s) != simdjson::SUCCESS) continue;
            top.layer_types.emplace_back(s);
        }
    }
    if (auto arr = root["mlp_layer_types"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto e : arr.value()) {
            std::string_view s;
            if (e.get_string().get(s) != simdjson::SUCCESS) continue;
            top.mlp_layer_types.emplace_back(s);
        }
    }

    // Semantics that would silently change the model if they differed.
    const std::string act = get_s(root, "hidden_act", "silu");
    const bool attention_bias = get_b(root, "attention_bias", false);
    const bool tie = get_b(root, "tie_word_embeddings", false);
    const bool norm_topk = get_b(root, "norm_topk_prob", true);
    if (!top.error_msg.empty()) {
    } else if (act != "silu") {
        top.error_msg = "hidden_act must be silu (got " + act + ")";
    } else if (attention_bias) {
        top.error_msg = "attention_bias must be false";
    } else if (tie) {
        top.error_msg = "tie_word_embeddings must be false";
    } else if (!norm_topk) {
        top.error_msg = "norm_topk_prob must be true (weights are renormalized)";
    } else if (top.routed_scaling_factor != 2.5) {
        top.error_msg = absl::StrFormat("routed_scaling_factor %.6f != 2.5",
                                        top.routed_scaling_factor);
    } else if (top.final_logit_softcapping != 30.0) {
        top.error_msg = absl::StrFormat("final_logit_softcapping %.6f != 30",
                                        top.final_logit_softcapping);
    } else if (!SyntheticGeometryAllowed()) {
        // Frozen geometry that the kernels cannot derive at runtime. The full
        // passport cross-check is CheckFrozenGeometry below; these three are
        // checked here too because the converter test exercises Parse alone.
        if (top.n_group != frozen::kNGroups)
            top.error_msg = absl::StrFormat("n_group %d != %d", top.n_group,
                                            frozen::kNGroups);
        else if (top.topk_group != frozen::kTopkGroup)
            top.error_msg = absl::StrFormat("topk_group %d != %d",
                                            top.topk_group,
                                            frozen::kTopkGroup);
        else if (top.sliding_window != frozen::kWindow)
            top.error_msg = absl::StrFormat("sliding_window %d != %d",
                                            top.sliding_window, frozen::kWindow);
    }

    if (top.bos_token_id < 0 || top.eos_token_id < 0 || top.pad_token_id < 0) {
        std::vector<int> all_eos;
        ReadGenerationIds(fs::path(input_dir) / "generation_config.json",
                          top.bos_token_id, top.eos_token_id,
                          top.pad_token_id, &all_eos);
        // config.json carries eos_token_id: null, so the list lives in
        // generation_config.json. Both ids are real stops (`<|return|>` and
        // `<|call|>`), and dropping the second one makes the model repeat a
        // tool call forever.
        if (top.eos_ids.empty() && !all_eos.empty())
            top.eos_ids = all_eos;
    }
    if (top.eos_ids.empty() && top.eos_token_id >= 0)
        top.eos_ids.push_back(top.eos_token_id);
    if (top.eos_ids.size() > 1) top.eos_token_id_alt = top.eos_ids[1];

    if (!top.error_msg.empty()) return top;
    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.vocab_size <= 0 ||
        top.num_attention_heads <= 0 || top.num_key_value_heads <= 0 ||
        top.head_dim <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.dense_intermediate_size <= 0 || top.sliding_window <= 1 ||
        top.n_group <= 0 || top.topk_group <= 0) {
        top.error_msg = "config.json missing required beam fields";
        return top;
    }
    if (top.num_experts_per_tok > top.num_experts) {
        top.error_msg = "num_experts_per_tok > num_experts";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layer_types size ", top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    for (size_t i = 0; i < top.layer_types.size(); ++i) {
        if (top.layer_types[i] != "full_attention" &&
            top.layer_types[i] != "sliding_attention") {
            top.error_msg = absl::StrCat("layer_types[", i,
                                         "] is neither full_attention nor "
                                         "sliding_attention");
            return top;
        }
    }
    if (static_cast<int>(top.mlp_layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("mlp_layer_types size ",
                                     top.mlp_layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    for (size_t i = 0; i < top.mlp_layer_types.size(); ++i) {
        if (top.mlp_layer_types[i] != "dense" &&
            top.mlp_layer_types[i] != "sparse") {
            top.error_msg = absl::StrCat("mlp_layer_types[", i,
                                         "] is neither dense nor sparse");
            return top;
        }
    }
    return top;
}

ParsedBeamTensor classify_beam_tensor(const std::string &name)
{
    ParsedBeamTensor r;
    if (name == "lm_head.weight") {
        r.kind = BeamTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "model.";
    if (name.rfind(kPrefix, 0) != 0) return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = BeamTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = BeamTensorKind::FinalNorm;
        return r;
    }

    std::string body = rest;
    const std::string kLayers = "layers.";
    if (rest.rfind(kLayers, 0) == 0) {
        const auto digits_start = kLayers.size();
        const auto digits_end = rest.find('.', digits_start);
        if (digits_end == std::string::npos) return r;
        int layer = 0;
        if (!detail::ParseTensorIndex(
                std::string_view(rest).substr(digits_start,
                                              digits_end - digits_start),
                layer))
            return r;
        r.layer = layer;
        body = rest.substr(digits_end + 1);
    } else {
        return r;
    }

    // Routed experts carry an expert index in the path.
    const std::string kExpert = "mlp.experts.";
    if (body.rfind(kExpert, 0) == 0) {
        const auto digits_start = kExpert.size();
        const auto digits_end = body.find('.', digits_start);
        if (digits_end == std::string::npos) return r;
        int expert = 0;
        if (!detail::ParseTensorIndex(
                std::string_view(body).substr(digits_start,
                                              digits_end - digits_start),
                expert))
            return r;
        r.expert = expert;
        const std::string leaf = body.substr(digits_end + 1);
        if (leaf == "gate_proj.weight") r.kind = BeamTensorKind::ExpertGate;
        else if (leaf == "down_proj.weight") r.kind = BeamTensorKind::ExpertDown;
        else if (leaf == "up_proj.weight") r.kind = BeamTensorKind::ExpertUp;
        return r;
    }

    using K = BeamTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostAttnNorm},
        {"pre_feedforward_layernorm.weight", K::PreFfnNorm},
        {"post_feedforward_layernorm.weight", K::PostFfnNorm},
        {"self_attn.q_proj.weight", K::AttnQ},
        {"self_attn.k_proj.weight", K::AttnK},
        {"self_attn.v_proj.weight", K::AttnV},
        {"self_attn.o_proj.weight", K::AttnO},
        {"self_attn.gate_proj.weight", K::AttnGate},
        {"self_attn.gate_bias", K::AttnGateBias},
        {"self_attn.q_norm.weight", K::AttnQNorm},
        {"self_attn.k_norm.weight", K::AttnKNorm},
        {"mlp.gate_proj.weight", K::DenseMlpGate},
        {"mlp.up_proj.weight", K::DenseMlpUp},
        {"mlp.down_proj.weight", K::DenseMlpDown},
        {"mlp.gate.weight", K::RouterGate},
        {"mlp.gate.e_score_correction_bias", K::RouterBias},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool beam_tensor_is_nvfp4(BeamTensorKind kind)
{
    switch (kind) {
    case BeamTensorKind::AttnQ:
    case BeamTensorKind::AttnK:
    case BeamTensorKind::AttnV:
    case BeamTensorKind::AttnO:
    case BeamTensorKind::AttnGate:
    case BeamTensorKind::DenseMlpGate:
    case BeamTensorKind::DenseMlpUp:
    case BeamTensorKind::DenseMlpDown:
    case BeamTensorKind::ExpertGate:
    case BeamTensorKind::ExpertDown:
    case BeamTensorKind::ExpertUp:
        return true;
    default:
        return false;
    }
}

bool beam_tensor_is_routed_expert(BeamTensorKind kind)
{
    return kind == BeamTensorKind::ExpertGate ||
           kind == BeamTensorKind::ExpertDown ||
           kind == BeamTensorKind::ExpertUp;
}

arrow::Status ConvertBeamModel(const std::string &input_dir,
                               const std::string &output_root)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    BeamTopology top = ParseBeamConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("beam config error: ", top.error_msg);
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, why))
            return arrow::Status::Invalid("beam frozen geometry mismatch: ", why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    absl::flat_hash_set<int> dense_layers;
    for (size_t l = 0; l < top.mlp_layer_types.size(); ++l)
        if (top.mlp_layer_types[l] == "dense")
            dense_layers.insert(static_cast<int>(l));
    auto is_dense = [&](int layer) { return dense_layers.contains(layer); };

    // Classify the inventory up front: unknown names are hard errors, routed
    // experts are indexed by name for the expert-major second pass.
    struct ExpertLoc {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    absl::btree_map<std::string, ExpertLoc> expert_index;
    absl::btree_map<int, int> layer_kind_count;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_beam_tensor(t.name);
            if (p.kind == BeamTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in beam checkpoint: ", t.name);
            if (beam_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.expert < 0)
                    return arrow::Status::Invalid("expert tensor without index: ",
                                                  t.name);
                if (is_dense(p.layer))
                    return arrow::Status::Invalid(
                        "expert tensor on a dense-MLP layer: ", t.name);
                if (p.expert >= top.num_experts)
                    return arrow::Status::Invalid("expert index out of range: ",
                                                  t.name);
                expert_index.emplace(t.name, ExpertLoc{&shard, &t});
            }
            if (p.layer >= 0) ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");
    const int moe_layers =
        top.num_layers - static_cast<int>(dense_layers.size());
    if (static_cast<int>(expert_index.size()) !=
        moe_layers * top.num_experts * 3)
        return arrow::Status::Invalid(
            "routed-expert inventory size ", expert_index.size(),
            " != moe_layers * experts * 3");

    // (2) Source reads must be complete before any output is written.
    {
        SafetensorsModel resident_only;
        for (const auto &shard : model.shards) {
            SafetensorsModel::Shard s;
            s.filename = shard.filename;
            s.file_size = shard.file_size;
            for (const auto &t : shard.tensors) {
                auto p = classify_beam_tensor(t.name);
                if (!beam_tensor_is_routed_expert(p.kind))
                    s.tensors.push_back(t);
            }
            if (!s.tensors.empty()) resident_only.shards.push_back(std::move(s));
        }
        ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir,
                                                            resident_only));
    }

    const fs::path base = fs::path(output_root);
    const fs::path pinned_dir = base / "pinned";
    for (const fs::path &d : {base, pinned_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_beam_tensor(t.name);
            if (beam_tensor_is_routed_expert(p.kind)) continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            switch (p.kind) {
            case BeamTensorKind::Embed:
            case BeamTensorKind::LmHead:
            case BeamTensorKind::FinalNorm: {
                // FP16 on the device: the head multiplies every logit, so a
                // 4-bit row error moves every logit.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case BeamTensorKind::InputNorm:
            case BeamTensorKind::PostAttnNorm:
            case BeamTensorKind::PreFfnNorm:
            case BeamTensorKind::PostFfnNorm:
            case BeamTensorKind::AttnQNorm:
            case BeamTensorKind::AttnKNorm:
            case BeamTensorKind::AttnGateBias:
            case BeamTensorKind::RouterGate:
            case BeamTensorKind::RouterBias: {
                // Norms, the per-head q/k norms, the gate bias, and the router
                // stay high precision: the router moves the top-k SELECTION.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case BeamTensorKind::AttnQ:
            case BeamTensorKind::AttnK:
            case BeamTensorKind::AttnV:
            case BeamTensorKind::AttnO:
            case BeamTensorKind::AttnGate:
            case BeamTensorKind::DenseMlpGate:
            case BeamTensorKind::DenseMlpUp:
            case BeamTensorKind::DenseMlpDown: {
                if ((p.kind == BeamTensorKind::DenseMlpGate ||
                     p.kind == BeamTensorKind::DenseMlpUp ||
                     p.kind == BeamTensorKind::DenseMlpDown) &&
                    !is_dense(p.layer))
                    return arrow::Status::Invalid(
                        "dense-MLP tensor outside the dense prefix: ", t.name);
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!QuantizeSlice(data.data(), t.numel(), t.dtype, r))
                    return arrow::Status::IOError(
                        "GPU quantization failed for ", t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case BeamTensorKind::ExpertGate:
            case BeamTensorKind::ExpertDown:
            case BeamTensorKind::ExpertUp:
                break;  // handled by the expert pass
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    // (3) Routed experts: expert-major second pass. Reads gate/down/up
    // (BF16, <= 26 MB each) straight from their shards, quantizes, and writes
    // one ExpertSchema file per (layer, expert).
    const int H = top.hidden_size;
    const int I = top.moe_intermediate_size;
    int experts_done = 0;
    const int experts_total = moe_layers * top.num_experts;
    int last_pct = -1;
    // Operational fast path: re-convert only the dense side when the expert
    // store is already on disk.
    const bool skip_experts = std::getenv("LSX_BEAM_SKIP_EXPERTS") != nullptr;
    if (skip_experts)
        lsxcommon::log::info(
            "beam: LSX_BEAM_SKIP_EXPERTS set — reusing the existing expert "
            "store");
    for (int l = 0; !skip_experts && l < top.num_layers; ++l) {
        if (is_dense(l)) continue;
        for (int e = 0; e < top.num_experts; ++e) {
            {
                const int pct = experts_total > 0
                                    ? (100 * experts_done) / experts_total
                                    : 100;
                if (pct != last_pct && pct % 5 == 0) {
                    lsxcommon::log::info(absl::StrFormat(
                        "beam experts: %d/%d (%d%%) layer %d", experts_done,
                        experts_total, pct, l));
                    last_pct = pct;
                }
                ++experts_done;
            }
            // Repo expert layout: w1 = gate, w2 = down, w3 = up.
            const std::string stem =
                absl::StrCat("model.layers.", l, ".mlp.experts.", e, ".");
            const char *kLeaf[3] = {"gate_proj.weight", "down_proj.weight",
                                    "up_proj.weight"};
            DenseNvfp4Record recs[3];
            for (int k = 0; k < 3; ++k) {
                const std::string name = stem + kLeaf[k];
                auto it = expert_index.find(name);
                if (it == expert_index.end())
                    return arrow::Status::Invalid("missing expert tensor: ", name);
                auto data = ReadTensorData(input_dir, *it->second.shard,
                                           *it->second.info);
                if (data.empty() || static_cast<int64_t>(data.size()) !=
                                        it->second.info->byte_size())
                    return arrow::Status::IOError("failed to read expert tensor: ",
                                                  name);
                const auto &shape = it->second.info->shape;
                if (shape.size() != 2)
                    return arrow::Status::Invalid("expert tensor must be 2-D: ",
                                                  name);
                if (k == 1) {
                    if (shape[0] != H || shape[1] != I)
                        return arrow::Status::Invalid("down shape != [H,I]: ", name);
                } else if (shape[0] != I || shape[1] != H) {
                    return arrow::Status::Invalid("gate/up shape != [I,H]: ", name);
                }
                recs[k].name = name;
                recs[k].shape = shape;
                recs[k].dtype = it->second.info->dtype;
                if (!QuantizeSlice(data.data(), it->second.info->numel(),
                                   recs[k].dtype, recs[k]))
                    return arrow::Status::IOError("quantize failed: ", name);
            }
            PackedExpert pe;
            pe.w1_packed = std::move(recs[0].packed);
            pe.w1_scales = std::move(recs[0].scales);
            pe.w1_shape = recs[0].shape;
            pe.w1_global_scale = recs[0].global_scale;
            pe.w2_packed = std::move(recs[1].packed);
            pe.w2_scales = std::move(recs[1].scales);
            pe.w2_shape = recs[1].shape;
            pe.w2_global_scale = recs[1].global_scale;
            pe.w3_packed = std::move(recs[2].packed);
            pe.w3_scales = std::move(recs[2].scales);
            pe.w3_shape = recs[2].shape;
            pe.w3_global_scale = recs[2].global_scale;
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
        }
    }

    // (4) Dense-side artifacts.
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(
            WriteBatch(PinnedSchema::MakeBatch(global_raw),
                       pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // (5) Tokenizer: the reference tokenizer emits NO BOS (bos_token is null;
    // the Harmony template spells <|start|> out), and the pre-tokenizer is the
    // Qwen2/3 single-run pattern (tokenizer_kind 6).
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        tok_policy.pretokenizer = 4;
        if (top.eos_token_id >= 0) tok_policy.eos_id = top.eos_token_id;
        if (top.pad_token_id >= 0) tok_policy.pad_id = top.pad_token_id;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true,
                                           .pad_required = true},
            tok_policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // (6) Conversion-time VRAM profile. The routed experts are STREAMED from
    // the host store (277 GB at NVFP4 does not fit the card), so they are
    // classified RoutedExpert: min_bytes is the pinned dense graph only.
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.num_moe_layers = moe_layers;
    vg.experts_per_layer = top.num_experts;
    vg.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            inv.push_back({t.name, t.numel(), t.byte_size()});
    const VramProfile vram = EstimateVram(
        inv,
        {+[](const std::string &n) {
             return beam_tensor_is_nvfp4(classify_beam_tensor(n).kind);
         },
         +[](const std::string &n) {
             return beam_tensor_is_routed_expert(
                        classify_beam_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         nullptr},
        vg);

    // (7) Structural manifest (deterministic byte content).
    {
        std::ostringstream j;
        j << "{\n  \"layout\": 2,\n"
          << "  \"model\": \"" << frozen::kModelName << "\",\n"
          << "  \"model_type\": \"" << frozen::kType << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"mlp_layer_types\": [";
        for (size_t i = 0; i < top.mlp_layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.mlp_layer_types[i] << "\"";
        }
        j << "],\n  \"full_attention_layers\": [";
        {
            bool first = true;
            for (size_t i = 0; i < top.layer_types.size(); ++i) {
                if (top.layer_types[i] != "full_attention") continue;
                if (!first) j << ",";
                first = false;
                j << i;
            }
        }
        j << "],\n  \"no_rope_layers\": [";
        {
            bool first = true;
            for (size_t i = 0; i < top.layer_types.size(); ++i) {
                if (top.layer_types[i] != "full_attention") continue;
                if (!first) j << ",";
                first = false;
                j << i;
            }
        }
        j << "],\n  \"first_moe_layer\": " << dense_layers.size()
          << ",\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"sliding_window\": " << top.sliding_window
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"n_group\": " << top.n_group
          << ",\n  \"topk_group\": " << top.topk_group
          << ",\n  \"routed_scaling_factor\": " << top.routed_scaling_factor
          << ",\n  \"final_logit_softcapping\": "
          << top.final_logit_softcapping
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate_size
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"eos_token_ids\": [";
        for (size_t i = 0; i < top.eos_ids.size(); ++i) {
            if (i) j << ",";
            j << top.eos_ids[i];
        }
        j << "]"
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"expert_store\": \"streamed\""
          << ",\n  \"skipped_visual_tensors\": " << top.skipped_ballast_tensors
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << experts_total << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // (8) Final verification: every produced arrow file must re-open cleanly.
    {
        std::error_code ec;
        for (fs::directory_iterator it(pinned_dir, ec), end;
             !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file()) continue;
            auto res = ReadRecordBatchFromFile(it->path());
            if (!res.ok())
                return arrow::Status::IOError("unreadable output arrow: ",
                                              it->path().string());
        }
        if (ec) return arrow::Status::IOError("cannot list ",
                                              pinned_dir.string());
    }

    int full_layers = 0;
    for (const auto &t : top.layer_types)
        if (t == "full_attention") ++full_layers;
    lsxcommon::log::info(absl::StrFormat(
        "beam conversion: %d layers (%d MoE, %d full-attn), %d experts/layer, "
        "%d pinned tensors",
        top.num_layers, moe_layers, full_layers, top.num_experts,
        top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::beam
