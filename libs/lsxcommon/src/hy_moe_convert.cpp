#include "lsxcommon/hy_moe_convert.h"

#include "hy_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "tensor_index_parse.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

namespace lsxcommon::hy {
namespace {

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

// Cross-checks the parsed config against the caller's frozen passport. Returns
// false and fills `why` on mismatch (synthetic fixtures bypass this upstream).
bool CheckFrozenGeometry(const HyPassportGeometry &g, const HyTopology &top,
                         std::string &why)
{
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != g.hidden) return bad("hidden_size", top.hidden_size, g.hidden);
    if (top.num_layers != g.layers) return bad("num_hidden_layers", top.num_layers, g.layers);
    if (top.num_attention_heads != g.heads)
        return bad("num_attention_heads", top.num_attention_heads, g.heads);
    if (top.num_key_value_heads != g.kv_heads)
        return bad("num_key_value_heads", top.num_key_value_heads, g.kv_heads);
    if (top.head_dim != g.head_dim) return bad("head_dim", top.head_dim, g.head_dim);
    if (top.rotary_dim != g.rotary_dim) return bad("rotary_dim", top.rotary_dim, g.rotary_dim);
    if (top.vocab_size != g.vocab) return bad("vocab_size", top.vocab_size, g.vocab);
    if (top.num_experts != g.num_experts)
        return bad("num_experts", top.num_experts, g.num_experts);
    if (top.num_experts_per_tok != g.top_k)
        return bad("num_experts_per_tok", top.num_experts_per_tok, g.top_k);
    if (top.moe_intermediate_size != g.moe_intermediate)
        return bad("expert_hidden_dim", top.moe_intermediate_size,
                   g.moe_intermediate);
    if (top.shared_intermediate_size != g.shared_intermediate)
        return bad("shared_intermediate_size", top.shared_intermediate_size,
                   g.shared_intermediate);
    if (top.dense_intermediate_size != g.dense_intermediate)
        return bad("dense_intermediate_size", top.dense_intermediate_size,
                   g.dense_intermediate);
    for (int i = 0; i < top.num_layers; ++i) {
        const size_t ui = static_cast<size_t>(i);
        const char *want_mlp = (i >= g.first_moe_layer) ? "sparse" : "dense";
        if (top.mlp_types[ui] != want_mlp)
            return bad("mlp_layer_types[i]", i, 0);
    }
    return true;
}

// The speech encoder of Hy-SimulMT is ballast: stored + counted, never applied.
bool IsAudioBallast(const std::string &name)
{
    return name.rfind("audio_tower.", 0) == 0;
}

HyPassportGeometry HyMemoryPassportGeometry()
{
    using namespace frozen;
    return HyPassportGeometry{
        .layers = kLayers,
        .hidden = kHidden,
        .heads = kHeads,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .rotary_dim = kRotaryDim,
        .vocab = kVocab,
        .dense_intermediate = kDenseIntermediate,
        .moe_intermediate = kMoeIntermediate,
        .shared_intermediate = kSharedIntermediate,
        .num_experts = kNumExperts,
        .top_k = kTopK,
        .first_moe_layer = kFirstMoeLayer,
    };
}

// Quantizes a contiguous BF16/FP32 slice to NVFP4. Returns false on GPU failure.
// Appends one `eos_token_id` value, which the checkpoints encode either as a
// single int or as an array (Hy-SimulMT: [120025, 120008]).
void AppendEosIds(simdjson::ondemand::value val, std::vector<int> &out)
{
    int64_t x = 0;
    if (val.get_int64().get(x) == simdjson::SUCCESS) {
        out.push_back(static_cast<int>(x));
        return;
    }
    auto arr = val.get_array();
    if (arr.error() != simdjson::SUCCESS) return;
    for (auto e : arr.value()) {
        int64_t v = 0;
        if (e.get_int64().get(v) == simdjson::SUCCESS)
            out.push_back(static_cast<int>(v));
    }
}

void ReadEosIdsFromObject(simdjson::ondemand::object &root,
                          std::vector<int> &out)
{
    auto v = root["eos_token_id"];
    if (v.error() != simdjson::SUCCESS) return;
    AppendEosIds(v.value(), out);
}

void ReadEosIdsFile(const std::filesystem::path &path, std::vector<int> &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty()) return;
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return;
    ReadEosIdsFromObject(root, out);
}

void ReadGenerationIds(const std::filesystem::path &path, int &bos, int &eos,
                       int &pad)
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
    const int e = get_i(root, "eos_token_id", -1);
    const int p = get_i(root, "pad_token_id", -1);
    if (b >= 0) bos = b;
    if (e >= 0) eos = e;
    if (p >= 0) pad = p;
}

} // namespace

HyTopology ParseHyConfig(const std::string &input_dir)
{
    HyTopology top;
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

    // hy_v3 keeps everything flat at the root (no text_config).
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
    top.dense_intermediate_size = get_i(src, "intermediate_size", 0);
    top.num_experts = get_i(src, "num_experts", get_i(src, "num_local_experts", 0));
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-5);
    top.router_scaling_factor = get_d(src, "router_scaling_factor", 1.0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);
    top.bos_token_id = get_i(src, "bos_token_id", -1);
    top.pad_token_id = get_i(src, "pad_token_id", -1);

    // RoPE: the checkpoint nests it in rope_parameters (new transformers
    // layout); fall back to the legacy flat rope_theta.
    top.rope_theta = 11158840.0;
    if (auto rp = src["rope_parameters"].get_object();
        rp.error() == simdjson::SUCCESS) {
        auto r = rp.value();
        top.rope_theta = get_d(r, "rope_theta", top.rope_theta);
    } else {
        top.rope_theta = get_d(src, "rope_theta", top.rope_theta);
    }

    // Rotary width: full head unless partial_rotary_factor narrows it.
    top.rotary_dim = get_i(src, "rotary_dim", -1);
    if (top.rotary_dim < 0) {
        const double prf = get_d(src, "partial_rotary_factor", 1.0);
        top.rotary_dim = static_cast<int>(top.head_dim * prf);
    }

    // Expert width: the shipped config carries BOTH `moe_intermediate_size`
    // (1536, a vestigial default) and `expert_hidden_dim` (768, the real tensor
    // width). The tensors win; only fall back when expert_hidden_dim is absent.
    top.moe_intermediate_size = get_i(src, "expert_hidden_dim", -1);
    if (top.moe_intermediate_size <= 0)
        top.moe_intermediate_size = get_i(src, "moe_intermediate_size", 0);
    const int num_shared = get_i(src, "num_shared_experts", 1);
    top.shared_intermediate_size =
        num_shared > 0 ? top.moe_intermediate_size * num_shared : 0;

    // Layer dispatch: mlp_layer_types (strings) preferred; first_k_dense_replace
    // is the fallback and the cross-check.
    if (auto arr = src["mlp_layer_types"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto e : arr.value()) {
            std::string_view s;
            if (e.get_string().get(s) != simdjson::SUCCESS) continue;
            top.mlp_types.emplace_back(s);
        }
    }
    if (top.mlp_types.empty() && top.num_layers > 0) {
        const int first_moe = get_i(src, "first_k_dense_replace", 0);
        for (int i = 0; i < top.num_layers; ++i)
            top.mlp_types.emplace_back(i < first_moe ? "dense" : "sparse");
    }

    // Token ids: root first, then generation_config.json.
    if (top.bos_token_id < 0 || top.eos_token_id < 0 || top.pad_token_id < 0)
        ReadGenerationIds(fs::path(input_dir) / "generation_config.json",
                          top.bos_token_id, top.eos_token_id,
                          top.pad_token_id);

    // Stop ids: the config's primary eos first, then everything the
    // generation_config declares (which may be an array).
    top.eos_ids.clear();
    if (top.eos_token_id >= 0) top.eos_ids.push_back(top.eos_token_id);
    {
        // Fresh parse (ondemand is single-pass): the config may encode the
        // stop id as an int or an array, and the generation_config adds more.
        ReadEosIdsFile(cfg_path, top.eos_ids);
        ReadEosIdsFile(fs::path(input_dir) / "generation_config.json",
                       top.eos_ids);
        std::vector<int> uniq;
        for (int id : top.eos_ids)
            if (std::find(uniq.begin(), uniq.end(), id) == uniq.end())
                uniq.push_back(id);
        top.eos_ids = std::move(uniq);
        if (top.eos_token_id < 0 && !top.eos_ids.empty())
            top.eos_token_id = top.eos_ids.front();
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.vocab_size <= 0 ||
        top.num_attention_heads <= 0 || top.num_key_value_heads <= 0 ||
        top.head_dim <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.dense_intermediate_size <= 0) {
        top.error_msg = "config.json missing required hy_v3 fields";
        return top;
    }
    if (static_cast<int>(top.mlp_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("mlp_layer_types size ", top.mlp_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    for (size_t i = 0; i < top.mlp_types.size(); ++i) {
        if (top.mlp_types[i] != "dense" && top.mlp_types[i] != "sparse") {
            top.error_msg = absl::StrCat("mlp_layer_types[", i,
                                         "] is neither dense nor sparse");
            return top;
        }
    }
    return top;
}

ParsedHyTensor classify_hy_tensor(const std::string &name)
{
    ParsedHyTensor r;
    if (name == "lm_head.weight") {
        r.kind = HyTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "model.";
    if (name.rfind(kPrefix, 0) != 0) return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = HyTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = HyTensorKind::FinalNorm;
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
        if (leaf == "gate_proj.weight") r.kind = HyTensorKind::ExpertGate;
        else if (leaf == "down_proj.weight") r.kind = HyTensorKind::ExpertDown;
        else if (leaf == "up_proj.weight") r.kind = HyTensorKind::ExpertUp;
        return r;
    }

    using K = HyTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::AttnQ},
        {"self_attn.k_proj.weight", K::AttnK},
        {"self_attn.v_proj.weight", K::AttnV},
        {"self_attn.o_proj.weight", K::AttnO},
        {"self_attn.q_norm.weight", K::AttnQNorm},
        {"self_attn.k_norm.weight", K::AttnKNorm},
        {"mlp.gate_proj.weight", K::DenseMlpGate},
        {"mlp.up_proj.weight", K::DenseMlpUp},
        {"mlp.down_proj.weight", K::DenseMlpDown},
        {"mlp.router.gate.weight", K::RouterGate},
        {"mlp.expert_bias", K::RouterBias},
        {"mlp.shared_mlp.gate_proj.weight", K::SharedGateProj},
        {"mlp.shared_mlp.up_proj.weight", K::SharedUpProj},
        {"mlp.shared_mlp.down_proj.weight", K::SharedDownProj},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool hy_tensor_is_nvfp4(HyTensorKind kind)
{
    switch (kind) {
    case HyTensorKind::AttnQ:
    case HyTensorKind::AttnK:
    case HyTensorKind::AttnV:
    case HyTensorKind::AttnO:
    case HyTensorKind::DenseMlpGate:
    case HyTensorKind::DenseMlpUp:
    case HyTensorKind::DenseMlpDown:
    case HyTensorKind::ExpertGate:
    case HyTensorKind::ExpertDown:
    case HyTensorKind::ExpertUp:
    case HyTensorKind::SharedGateProj:
    case HyTensorKind::SharedUpProj:
    case HyTensorKind::SharedDownProj:
        return true;
    default:
        return false;
    }
}

bool hy_tensor_is_routed_expert(HyTensorKind kind)
{
    return kind == HyTensorKind::ExpertGate ||
           kind == HyTensorKind::ExpertDown ||
           kind == HyTensorKind::ExpertUp;
}

arrow::Status ConvertHyMoeModelWithSpec(const std::string &input_dir,
                                        const std::string &output_root,
                                        const HyConvertSpec &spec)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    HyTopology top = ParseHyConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid(spec.model_name, " config error: ",
                                      top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(spec.geometry, top, why))
            return arrow::Status::Invalid(spec.model_name,
                                          " frozen geometry mismatch: ", why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Classify the inventory up front: unknown names are hard errors, routed
    // experts are indexed by name for the expert-major second pass (they are
    // ~96% of the bytes and must never be held in host RAM as a whole).
    struct ExpertLoc {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    absl::btree_map<std::string, ExpertLoc> expert_index;
    absl::btree_map<int, int> layer_kind_count;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (spec.allow_audio_ballast && IsAudioBallast(t.name)) {
                ++top.skipped_ballast_tensors;
                continue;
            }
            auto p = classify_hy_tensor(t.name);
            if (p.kind == HyTensorKind::Unknown)
                return arrow::Status::Invalid("unrecognized tensor in ",
                                              spec.model_name, " checkpoint: ",
                                              t.name);
            if (hy_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.expert < 0)
                    return arrow::Status::Invalid("expert tensor without index: ",
                                                  t.name);
                if (top.mlp_types[static_cast<size_t>(p.layer)] != "sparse")
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

    // (2) Source reads must be complete before any output is written. Skip the
    // routed-expert payloads here: they are re-read (and verified) by the
    // expert pass below.
    {
        SafetensorsModel resident_only;
        for (const auto &shard : model.shards) {
            SafetensorsModel::Shard s;
            s.filename = shard.filename;
            s.file_size = shard.file_size;
            for (const auto &t : shard.tensors) {
                if (spec.allow_audio_ballast && IsAudioBallast(t.name))
                    continue;
                auto p = classify_hy_tensor(t.name);
                if (!hy_tensor_is_routed_expert(p.kind))
                    s.tensors.push_back(t);
            }
            if (!s.tensors.empty()) resident_only.shards.push_back(std::move(s));
        }
        ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir,
                                                            resident_only));
    }

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    for (const fs::path &d : {base, pinned_dir, router_dir, shared_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> shared_nvfp4_by_layer;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;

    auto add_nvfp4 = [&](const TensorInfo &t, std::vector<uint8_t> &data,
                         std::vector<DenseNvfp4Record> &dst) -> arrow::Status {
        DenseNvfp4Record r;
        r.name = t.name;
        r.shape = t.shape;
        r.dtype = t.dtype;
        if (!QuantizeSlice(data.data(), t.numel(), t.dtype, r))
            return arrow::Status::IOError("GPU quantization failed for ", t.name);
        dst.push_back(std::move(r));
        ++total_pinned;
        return arrow::Status::OK();
    };

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            if (spec.allow_audio_ballast && IsAudioBallast(t.name)) continue;
            auto p = classify_hy_tensor(t.name);
            if (hy_tensor_is_routed_expert(p.kind)) continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            switch (p.kind) {
            case HyTensorKind::Embed:
            case HyTensorKind::LmHead:
            case HyTensorKind::FinalNorm: {
                // FP16 on the device: the head multiplies every logit, so a
                // 4-bit row error moves every logit (phi-4 / gemma-4 rule).
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case HyTensorKind::InputNorm:
            case HyTensorKind::PostNorm:
            case HyTensorKind::AttnQNorm:
            case HyTensorKind::AttnKNorm:
            case HyTensorKind::RouterGate:
            case HyTensorKind::RouterBias: {
                // Norms and the router stay high precision on purpose: the
                // routing bias moves the top-k SELECTION, so a 4-bit router
                // swaps the expert set.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                if (p.kind == HyTensorKind::RouterGate ||
                    p.kind == HyTensorKind::RouterBias)
                    router_by_layer[p.layer].push_back(std::move(r));
                else
                    raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case HyTensorKind::SharedGateProj:
            case HyTensorKind::SharedUpProj:
            case HyTensorKind::SharedDownProj:
                ARROW_RETURN_NOT_OK(
                    add_nvfp4(t, data, shared_nvfp4_by_layer[p.layer]));
                break;
            case HyTensorKind::AttnQ:
            case HyTensorKind::AttnK:
            case HyTensorKind::AttnV:
            case HyTensorKind::AttnO:
            case HyTensorKind::DenseMlpGate:
            case HyTensorKind::DenseMlpUp:
            case HyTensorKind::DenseMlpDown:
                ARROW_RETURN_NOT_OK(add_nvfp4(t, data, nvfp4_by_layer[p.layer]));
                break;
            case HyTensorKind::ExpertGate:
            case HyTensorKind::ExpertDown:
            case HyTensorKind::ExpertUp:
                break;  // handled by the expert pass
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    // (3) Routed experts: expert-major second pass. Reads gate/down/up (BF16,
    // <= 3 MB each) straight from their shards, quantizes, and writes one
    // ExpertSchema file per (layer, expert). No host-resident expert pool.
    const int H = top.hidden_size;
    const int I = top.moe_intermediate_size;
    int experts_done = 0;
    const int experts_total =
        (top.num_layers - spec.geometry.first_moe_layer) * top.num_experts;
    int last_pct = -1;
    // Operational fast path: re-convert only the dense side when the expert
    // store is already on disk. The written index is byte-identical (every
    // field is topology-derived).
    const bool skip_experts = std::getenv("LSX_HY_SKIP_EXPERTS") != nullptr;
    if (skip_experts)
        lsxcommon::log::info(absl::StrCat(
            spec.model_name,
            ": LSX_HY_SKIP_EXPERTS set — reusing the existing expert store"));
    for (int l = 0; !skip_experts && l < top.num_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] != "sparse") continue;
        for (int e = 0; e < top.num_experts; ++e) {
            {
                const int pct = experts_total > 0
                                    ? (100 * experts_done) / experts_total
                                    : 100;
                if (pct != last_pct && pct % 5 == 0) {
                    lsxcommon::log::info(absl::StrFormat(
                        "%s experts: %d/%d (%d%%) layer %d", spec.model_name,
                        experts_done, experts_total, pct, l));
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
                if (data.empty() ||
                    static_cast<int64_t>(data.size()) != it->second.info->byte_size())
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
    for (auto &[layer, recs] : router_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            router_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : shared_nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // (5) Tokenizer: mandatory for real checkpoints; synthetic fixtures may
    // omit it (recorded in the index). The chat template spells the BOS marker
    // out as text, so the reference tokenizer must NOT auto-prepend BOS. The
    // pre-tokenizer is the digit-run / CJK / 6-alternative GPT-2 family
    // (tokenizer_kind 0, the default policy).
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        tok_policy.pretokenizer = 0;
        // This vocabulary has no single-byte token for CR or for the 13
        // UTF-8-illegal lead bytes; the reference tokenizer drops "\r".
        tok_policy.drop_unmapped_bytes = 1;
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

    // (6) Conversion-time VRAM profile. Every routed expert is resident on the
    // device (this family's expert set is ~16 GiB at NVFP4), so they are
    // classified Resident, not RoutedExpert.
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.num_moe_layers = top.num_layers - spec.geometry.first_moe_layer;
    vg.experts_per_layer = top.num_experts;
    vg.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            if (spec.allow_audio_ballast && IsAudioBallast(t.name)) continue;
            inv.push_back({t.name, t.numel(), t.byte_size()});
        }
    const VramProfile vram = EstimateVram(
        inv,
        {+[](const std::string &n) {
             return hy_tensor_is_nvfp4(classify_hy_tensor(n).kind);
         },
         +[](const std::string &) -> VramTensorClass {
             return VramTensorClass::Resident;
         },
         nullptr},
        vg);

    // (7) Structural manifest (deterministic byte content).
    {
        std::ostringstream j;
        j << "{\n  \"format\": \"" << spec.format_tag << "\",\n"
          << "  \"model\": \"" << spec.model_name << "\",\n"
          << "  \"model_type\": \"" << spec.model_type << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n  \"mlp_layer_types\": [";
        for (size_t i = 0; i < top.mlp_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.mlp_types[i] << "\"";
        }
        j << "],\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"rotary_dim\": " << top.rotary_dim
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate_size
          << ",\n  \"shared_intermediate_size\": "
          << top.shared_intermediate_size
          << ",\n  \"router_scaling_factor\": " << top.router_scaling_factor
          << ",\n  \"first_moe_layer\": " << spec.geometry.first_moe_layer
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
          << ",\n  \"skipped_visual_tensors\": " << top.skipped_ballast_tensors
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << top.num_experts * (top.num_layers -
                                                             spec.geometry.first_moe_layer)
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // (8) Final verification: every produced arrow file must re-open cleanly.
    for (const fs::path &dir : {pinned_dir, router_dir, shared_dir}) {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
             it.increment(ec)) {
            if (!it->is_regular_file()) continue;
            auto res = ReadRecordBatchFromFile(it->path());
            if (!res.ok())
                return arrow::Status::IOError("unreadable output arrow: ",
                                              it->path().string());
        }
        if (ec) return arrow::Status::IOError("cannot list ", dir.string());
    }

    lsxcommon::log::info(absl::StrFormat(
        "%s conversion: %d layers (%d MoE), %d experts/layer, "
        "%d pinned tensors, %d ballast tensors",
        spec.model_name, top.num_layers,
        top.num_layers - spec.geometry.first_moe_layer, top.num_experts,
        top.total_pinned_tensors, top.skipped_ballast_tensors));
    return arrow::Status::OK();
}

arrow::Status ConvertHyMoeModel(const std::string &input_dir,
                                const std::string &output_root)
{
    HyConvertSpec spec;
    spec.geometry = HyMemoryPassportGeometry();
    return ConvertHyMoeModelWithSpec(input_dir, output_root, spec);
}

} // namespace lsxcommon::hy
