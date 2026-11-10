#include "lsxcommon/nemotron_moe_ultra_convert.h"

#include "nemotron_moe_ultra_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "tensor_index_parse.h"

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace lsxcommon::nemotron_ultra {
namespace {

namespace fs = std::filesystem;

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
auto get_b = [](simdjson::ondemand::object &o, const char *k, bool dflt) {
    bool v;
    if (o[k].get_bool().get(v) != simdjson::SUCCESS) return dflt;
    return v;
};

// Legacy NemotronH layer tags map onto the modeling block types.
std::string normalize_block_type(std::string_view tag)
{
    if (tag == "linear_attention" || tag == "mamba") return "mamba";
    if (tag == "full_attention" || tag == "attention") return "attention";
    if (tag == "moe" || tag == "mlp") return "moe";
    return std::string(tag);
}

bool check_frozen_geometry(const UltraTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.num_layers != kLayers) return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.hidden_size != kHidden) return bad("hidden_size", top.hidden_size, kHidden);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.mamba_num_heads != kMambaHeads)
        return bad("mamba_num_heads", top.mamba_num_heads, kMambaHeads);
    if (top.mamba_head_dim != kMambaHeadDim)
        return bad("mamba_head_dim", top.mamba_head_dim, kMambaHeadDim);
    if (top.ssm_state_size != kSsmStateSize)
        return bad("ssm_state_size", top.ssm_state_size, kSsmStateSize);
    if (top.n_groups != kSsmGroups) return bad("n_groups", top.n_groups, kSsmGroups);
    if (top.conv_kernel != kMambaConvKernel)
        return bad("conv_kernel", top.conv_kernel, kMambaConvKernel);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKVHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.num_experts != kNumRouted)
        return bad("n_routed_experts", top.num_experts, kNumRouted);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.moe_latent_size != kMoeLatent)
        return bad("moe_latent_size", top.moe_latent_size, kMoeLatent);
    if (top.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   top.shared_expert_intermediate_size, kSharedIntermediate);
    if (top.n_group != kRouterGroups) return bad("n_group", top.n_group, kRouterGroups);
    if (top.topk_group != kRouterTopkGroup)
        return bad("topk_group", top.topk_group, kRouterTopkGroup);

    int mamba = 0, moe = 0, attn = 0;
    for (const auto &t : top.layer_types) {
        if (t == "mamba") ++mamba;
        else if (t == "moe") ++moe;
        else if (t == "attention") ++attn;
        else {
            why = "unknown layer type: " + t;
            return false;
        }
    }
    if (mamba != kMambaLayers || moe != kMoeLayers || attn != kAttnLayers) {
        why = absl::StrFormat("layer mix got mamba=%d moe=%d attn=%d, want %d/%d/%d",
                              mamba, moe, attn, kMambaLayers, kMoeLayers, kAttnLayers);
        return false;
    }
    return true;
}

bool is_global_kind(UltraTensorKind k)
{
    return k == UltraTensorKind::Embed || k == UltraTensorKind::LmHead ||
           k == UltraTensorKind::FinalNorm;
}

bool is_expert_kind(UltraTensorKind k)
{
    return k == UltraTensorKind::ExpertUpProj || k == UltraTensorKind::ExpertDownProj;
}

// Where a tensor lives: shard index + tensor index inside that shard. Stored per
// (layer, expert) up front so the per-layer expert pass never re-scans 225
// shards (that would be ~2.5 billion classifications on the real checkpoint).
struct TensorLoc {
    int shard = -1;
    int tensor = -1;
    bool ok() const { return shard >= 0 && tensor >= 0; }
};

struct ExpertLoc {
    TensorLoc up, down;
};

// Which layers already carry a converted tree, derived from the files on disk
// rather than from the index: a killed conversion then resumes from what
// actually exists instead of from what the index claims.
std::vector<int> ConvertedLayers(const fs::path &base, int num_layers)
{
    std::vector<int> out;
    std::error_code ec;
    for (int l = 0; l < num_layers; ++l) {
        const std::string stem = absl::StrCat("layer_", l);
        if (fs::exists(base / "pinned" / absl::StrCat(stem, ".arrow"), ec) ||
            fs::exists(base / "pinned" / absl::StrCat(stem, "_nvfp4.arrow"), ec)) {
            out.push_back(l);
        }
    }
    return out;
}

arrow::Status QuantizeInto(const fs::path &input_dir, const SafetensorsModel &model,
                           const TensorLoc &loc, DenseNvfp4Record &rec,
                           const std::string &name_for_errors)
{
    const auto &shard = model.shards[loc.shard];
    const auto &t = shard.tensors[loc.tensor];
    auto data = ReadTensorData(input_dir, shard, t);
    if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
        return arrow::Status::IOError("failed to read tensor data: ", name_for_errors);
    rec.name = t.name;
    rec.shape = t.shape;
    rec.dtype = t.dtype;
    if (!QuantizeSliceBf16(data.data(), t.numel(), rec))
        return arrow::Status::IOError("GPU quantization failed for ", name_for_errors);
    return arrow::Status::OK();
}

} // namespace

ParsedUltraTensor classify_ultra_tensor(const std::string &name)
{
    using K = UltraTensorKind;
    ParsedUltraTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = K::Mtp;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = K::LmHead;
        return r;
    }
    if (name == "backbone.embeddings.weight") {
        r.kind = K::Embed;
        return r;
    }
    if (name == "backbone.norm_f.weight") {
        r.kind = K::FinalNorm;
        return r;
    }
    const std::string kPre = "backbone.layers.";
    if (name.rfind(kPre, 0) != 0) return r;
    const std::string rest = name.substr(kPre.size());
    const auto dend = rest.find('.');
    if (dend == std::string::npos) return r;
    int layer = 0;
    if (!detail::ParseTensorIndex(std::string_view(rest).substr(0, dend), layer))
        return r;  // malformed index -> Unknown, never throw
    r.layer = layer;
    std::string body = rest.substr(dend + 1);
    if (body == "norm.weight") {
        r.kind = K::LayerNorm;
        return r;
    }
    const std::string kMixer = "mixer.";
    if (body.rfind(kMixer, 0) != 0) return r;
    const std::string m = body.substr(kMixer.size());

    // Routed experts: mixer.experts.<E>.{up,down}_proj.weight
    const std::string kExperts = "experts.";
    if (m.rfind(kExperts, 0) == 0) {
        const std::string e = m.substr(kExperts.size());
        const auto ed = e.find('.');
        if (ed == std::string::npos) return r;
        int expert = 0;
        if (!detail::ParseTensorIndex(std::string_view(e).substr(0, ed), expert))
            return r;
        r.expert = expert;
        const std::string eb = e.substr(ed + 1);
        if (eb == "up_proj.weight") r.kind = K::ExpertUpProj;
        else if (eb == "down_proj.weight") r.kind = K::ExpertDownProj;
        return r;
    }

    static const std::pair<const char *, K> kMap[] = {
        {"in_proj.weight", K::MambaInProj},
        {"out_proj.weight", K::MambaOutProj},
        {"conv1d.weight", K::MambaConv},
        {"conv1d.bias", K::MambaConvBias},
        {"A_log", K::MambaALog},
        {"D", K::MambaD},
        {"dt_bias", K::MambaDtBias},
        {"norm.weight", K::MambaNorm},
        {"q_proj.weight", K::AttnQ},
        {"k_proj.weight", K::AttnK},
        {"v_proj.weight", K::AttnV},
        {"o_proj.weight", K::AttnO},
        {"gate.weight", K::RouterGate},
        {"gate.e_score_correction_bias", K::RouterBias},
        {"fc1_latent_proj.weight", K::MoeLatentUp},
        {"fc2_latent_proj.weight", K::MoeLatentDown},
        {"shared_experts.up_proj.weight", K::SharedUpProj},
        {"shared_experts.down_proj.weight", K::SharedDownProj},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (m == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool ultra_tensor_is_nvfp4(UltraTensorKind kind)
{
    using K = UltraTensorKind;
    switch (kind) {
    case K::MambaInProj:
    case K::MambaOutProj:
    case K::AttnQ:
    case K::AttnK:
    case K::AttnV:
    case K::AttnO:
    case K::MoeLatentUp:
    case K::MoeLatentDown:
    case K::SharedUpProj:
    case K::SharedDownProj:
    case K::ExpertUpProj:
    case K::ExpertDownProj:
        return true;
    default:
        return false;
    }
}

UltraTopology ParseUltraConfig(const std::string &input_dir)
{
    UltraTopology top;
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

    top.hidden_size = get_i(root, "hidden_size", 0);
    top.vocab_size = get_i(root, "vocab_size", 0);
    top.num_layers = get_i(root, "num_hidden_layers", 0);
    top.mamba_num_heads = get_i(root, "mamba_num_heads", 0);
    top.mamba_head_dim = get_i(root, "mamba_head_dim", 0);
    top.ssm_state_size = get_i(root, "ssm_state_size", 0);
    top.n_groups = get_i(root, "n_groups", 0);
    top.conv_kernel = get_i(root, "conv_kernel", 0);
    top.num_attention_heads = get_i(root, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(root, "num_key_value_heads", 0);
    top.head_dim = get_i(root, "head_dim", 0);
    top.rope_theta = get_d(root, "rope_theta", 10000.0);
    top.rms_norm_eps = get_d(root, "layer_norm_epsilon", 1e-5);
    top.num_experts = get_i(root, "n_routed_experts", 0);
    top.num_experts_per_tok = get_i(root, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(root, "moe_intermediate_size", 0);
    top.shared_expert_intermediate_size =
        get_i(root, "moe_shared_expert_intermediate_size", 0);
    top.moe_latent_size = get_i(root, "moe_latent_size", 0);
    top.n_group = get_i(root, "n_group", 1);
    top.topk_group = get_i(root, "topk_group", 1);
    top.routed_scaling_factor =
        static_cast<float>(get_d(root, "routed_scaling_factor", 1.0));
    top.norm_topk_prob = get_b(root, "norm_topk_prob", false);
    top.bos_token_id = get_i(root, "bos_token_id", -1);
    top.pad_token_id = get_i(root, "pad_token_id", -1);
    // eos_token_id: config.json declares a bare int here, generation_config.json
    // the real stop set. Read the config's spelling for the scalar and prefer the
    // generation config's list when it exists (below).
    top.eos_token_id = get_i(root, "eos_token_id", -1);
    if (top.eos_token_id >= 0) top.eos_token_ids.push_back(top.eos_token_id);

    if (auto lt = root["layers_block_type"].get_array();
        lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                top.layer_types.push_back(normalize_block_type(sv));
        }
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.layer_types.empty()) {
        top.error_msg = "config.json missing required NemotronH fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layers_block_type size ",
                                     top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    // Grouped routing geometry: the reference splits the scores into n_group
    // equal groups and keeps topk_group of them. A count that does not divide the
    // expert set is a config the reference itself cannot run, so it is a parse
    // error here instead of a divide-by-zero in the router.
    if (top.n_group <= 0 || top.num_experts % top.n_group != 0) {
        top.error_msg = absl::StrFormat(
            "n_group %d does not divide n_routed_experts %d (expert grouping "
            "requires equal groups)",
            top.n_group, top.num_experts);
        return top;
    }
    if (top.topk_group <= 0 || top.topk_group > top.n_group) {
        top.error_msg = absl::StrFormat("topk_group %d out of range for n_group %d",
                                        top.topk_group, top.n_group);
        return top;
    }
    if (top.num_experts_per_tok > top.num_experts) {
        top.error_msg = "num_experts_per_tok > n_routed_experts";
        return top;
    }
    for (const auto &t : top.layer_types) {
        if (t != "mamba" && t != "moe" && t != "attention") {
            top.error_msg = "unknown layer type in layers_block_type: " + t;
            return top;
        }
    }

    // generation_config.json carries the authoritative stop set. Reading it
    // matters: this checkpoint stops on 2 AND 11, and a reader that keeps only 2
    // generates past the model's own turn end.
    {
        const fs::path gen_path = fs::path(input_dir) / "generation_config.json";
        if (fs::exists(gen_path, ec)) {
            std::ifstream gf(gen_path, std::ios::binary);
            std::stringstream gs;
            gs << gf.rdbuf();
            std::string gjson = gs.str();
            simdjson::ondemand::parser gp;
            simdjson::padded_string gpad(gjson.data(), gjson.size());
            simdjson::ondemand::document gdoc;
            simdjson::ondemand::object groot;
            if (gp.iterate(gpad).get(gdoc) == simdjson::SUCCESS &&
                gdoc.get_object().get(groot) == simdjson::SUCCESS) {
                std::vector<int> stops;
                simdjson::ondemand::value eos;
                if (groot["eos_token_id"].get(eos) == simdjson::SUCCESS) {
                    if (auto arr = eos.get_array();
                        arr.error() == simdjson::SUCCESS) {
                        for (auto e : arr.value()) {
                            int64_t v;
                            if (e.get_int64().get(v) == simdjson::SUCCESS)
                                stops.push_back(static_cast<int>(v));
                        }
                    } else {
                        int64_t v;
                        if (eos.get_int64().get(v) == simdjson::SUCCESS)
                            stops.push_back(static_cast<int>(v));
                    }
                }
                if (!stops.empty()) {
                    top.eos_token_ids = std::move(stops);
                    top.eos_token_id = top.eos_token_ids.front();
                }
            }
        }
    }
    return top;
}

arrow::Status ConvertNemotronUltraModel(const std::string &input_dir,
                                        const std::string &output_root,
                                        const ConversionWindow &window)
{
    UltraTopology top = ParseUltraConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Ultra config error: ", top.error_msg);

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid("Ultra frozen geometry mismatch: ", why);
    }

    const int first = window.first_layer;
    const int last = window.last_layer < 0 ? top.num_layers - 1 : window.last_layer;
    if (first < 0 || last >= top.num_layers || first > last)
        return arrow::Status::Invalid("conversion window [", first, ",", last,
                                      "] out of range for ", top.num_layers,
                                      " layers");

    const SafetensorsModel model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    // Inventory: classify up front. MTP is a counted cut; unknown is fatal; the
    // location of every tensor is recorded so nothing re-scans the shards later.
    int mtp_cut = 0;
    absl::btree_map<int, int> layer_kind_count;
    absl::btree_map<int, std::array<bool, 2>> latent_present;  // layer -> {fc1, fc2}
    absl::btree_map<std::pair<int, int>, ExpertLoc, std::less<>> expert_at;
    absl::btree_map<int, std::vector<TensorLoc>> dense_of_layer;
    std::vector<TensorLoc> globals;
    for (std::size_t si = 0; si < model.shards.size(); ++si) {
        const auto &shard = model.shards[si];
        for (std::size_t ti = 0; ti < shard.tensors.size(); ++ti) {
            const auto &t = shard.tensors[ti];
            auto p = classify_ultra_tensor(t.name);
            if (p.kind == UltraTensorKind::Mtp) {
                ++mtp_cut;
                continue;
            }
            if (p.kind == UltraTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unclassified tensor in nemotron_h checkpoint: ", t.name);
            const TensorLoc loc{static_cast<int>(si), static_cast<int>(ti)};
            if (p.layer >= 0) ++layer_kind_count[p.layer];
            if (is_global_kind(p.kind)) {
                globals.push_back(loc);
                continue;
            }
            if (is_expert_kind(p.kind)) {
                auto &e = expert_at[{p.layer, p.expert}];
                (p.kind == UltraTensorKind::ExpertUpProj ? e.up : e.down) = loc;
                continue;
            }
            dense_of_layer[p.layer].push_back(loc);
            if (p.kind == UltraTensorKind::MoeLatentUp)
                latent_present[p.layer][0] = true;
            if (p.kind == UltraTensorKind::MoeLatentDown)
                latent_present[p.layer][1] = true;
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    // The latent bridge is required exactly when the config declares a latent
    // width: a tree converted without it would silently drop the routed path.
    if (top.moe_latent_size > 0) {
        for (int l = 0; l < top.num_layers; ++l) {
            if (top.layer_types[l] != "moe") continue;
            const auto it = latent_present.find(l);
            const bool up = it != latent_present.end() && it->second[0];
            const bool down = it != latent_present.end() && it->second[1];
            if (!up)
                return arrow::Status::Invalid(
                    "layer ", l, ": config declares moe_latent_size=",
                    top.moe_latent_size, " but tensor backbone.layers.", l,
                    ".mixer.fc1_latent_proj.weight is missing");
            if (!down)
                return arrow::Status::Invalid(
                    "layer ", l, ": config declares moe_latent_size=",
                    top.moe_latent_size, " but tensor backbone.layers.", l,
                    ".mixer.fc2_latent_proj.weight is missing");
        }
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    for (const fs::path &d : {base, pinned_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    int64_t total_pinned = 0;

    // ── Globals (small, written once) ──────────────────────────────────────
    {
        std::vector<TensorRecord> raw;
        for (const TensorLoc &loc : globals) {
            const auto &shard = model.shards[loc.shard];
            const auto &t = shard.tensors[loc.tensor];
            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ", t.name);
            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);
            raw.push_back(std::move(r));
        }
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(raw),
                                      pinned_dir / "global.arrow"));
        total_pinned += static_cast<int64_t>(raw.size());
    }

    // ── Dense pass: this window's layers only, one layer at a time ─────────
    for (int l = first; l <= last; ++l) {
        auto it = dense_of_layer.find(l);
        if (it == dense_of_layer.end()) continue;
        std::vector<TensorRecord> raw;
        std::vector<DenseNvfp4Record> nvfp4;
        for (const TensorLoc &loc : it->second) {
            const auto &shard = model.shards[loc.shard];
            const auto &t = shard.tensors[loc.tensor];
            auto p = classify_ultra_tensor(t.name);
            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ", t.name);
            if (ultra_tensor_is_nvfp4(p.kind)) {
                DenseNvfp4Record rec;
                rec.name = t.name;
                rec.shape = t.shape;
                rec.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), rec))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4.push_back(std::move(rec));
            } else {
                TensorRecord rec;
                rec.name = t.name;
                rec.dtype = t.dtype;
                rec.shape = t.shape;
                rec.data = std::move(data);
                raw.push_back(std::move(rec));
            }
            ++total_pinned;
        }
        if (!nvfp4.empty()) {
            ARROW_RETURN_NOT_OK(WriteBatch(
                DenseNvfp4Schema::MakeBatch(nvfp4),
                pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow")));
        }
        if (!raw.empty()) {
            ARROW_RETURN_NOT_OK(WriteBatch(
                PinnedSchema::MakeBatch(raw),
                pinned_dir / absl::StrCat("layer_", l, ".arrow")));
        }
    }

    // ── Routed experts: one layer at a time (the whole pool is ~290 GB) ────
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int LAT = top.moe_latent_size > 0 ? top.moe_latent_size : top.hidden_size;
    int64_t peak_expert_records = 0;
    int total_experts = 0;
    for (int l = first; l <= last; ++l) {
        if (top.layer_types[l] != "moe") continue;
        absl::btree_map<int, DenseNvfp4Record> up_map, down_map;
        for (int e = 0; e < E; ++e) {
            const auto it = expert_at.find({l, e});
            if (it == expert_at.end() || !it->second.up.ok() || !it->second.down.ok())
                return arrow::Status::Invalid("missing routed expert: layer ", l,
                                              " expert ", e);
            DenseNvfp4Record up, down;
            ARROW_RETURN_NOT_OK(QuantizeInto(input_dir, model, it->second.up, up,
                                             absl::StrCat("layer ", l, " expert ", e,
                                                          " up_proj")));
            ARROW_RETURN_NOT_OK(QuantizeInto(input_dir, model, it->second.down, down,
                                             absl::StrCat("layer ", l, " expert ", e,
                                                          " down_proj")));
            // Experts are stored at the LATENT width, not at hidden.
            if (up.shape.size() != 2 || up.shape[0] != I || up.shape[1] != LAT)
                return arrow::Status::Invalid("bad expert up shape: layer ", l,
                                              " expert ", e);
            if (down.shape.size() != 2 || down.shape[0] != LAT || down.shape[1] != I)
                return arrow::Status::Invalid("bad expert down shape: layer ", l,
                                              " expert ", e);
            up_map.insert_or_assign(e, std::move(up));
            down_map.insert_or_assign(e, std::move(down));
        }
        peak_expert_records =
            std::max<int64_t>(peak_expert_records, static_cast<int64_t>(up_map.size()));
        for (int e = 0; e < E; ++e) {
            PackedExpert pe;
            // Non-gated relu2 MLP: w1 unused, w3 = up, w2 = down.
            pe.w3_packed = up_map[e].packed;
            pe.w3_scales = up_map[e].scales;
            pe.w3_shape = up_map[e].shape;
            pe.w3_global_scale = up_map[e].global_scale;
            pe.w2_packed = down_map[e].packed;
            pe.w2_scales = down_map[e].scales;
            pe.w2_shape = down_map[e].shape;
            pe.w2_global_scale = down_map[e].global_scale;
            pe.w1_shape = {0, 0};
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
        }
        total_experts += E;
        // Both maps die here: the next layer starts from zero host residency.
    }

    // ── VRAM estimate (routed experts are the max term) ────────────────────
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = frozen::kAttnLayers;
        vg.num_moe_layers = frozen::kMoeLayers;
        vg.experts_per_layer = top.num_experts;
        vg.context_length =
            ReadContextLength((fs::path(input_dir) / "config.json").string());
        std::vector<VramTensorInfo> inv;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                inv.push_back({t.name, t.numel(), t.byte_size()});
        top.vram = EstimateVram(
            inv,
            {+[](const std::string &n) {
                 return ultra_tensor_is_nvfp4(classify_ultra_tensor(n).kind);
             },
             +[](const std::string &n) -> VramTensorClass {
                 auto k = classify_ultra_tensor(n).kind;
                 return (k == UltraTensorKind::ExpertUpProj ||
                         k == UltraTensorKind::ExpertDownProj)
                            ? VramTensorClass::RoutedExpert
                            : VramTensorClass::Resident;
             }},
            vg);
    }

    // ── Tokenizer: mandatory for a real checkpoint, optional for a fixture ──
    bool tokenizer_ok = false;
    {
        std::error_code ec;
        const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
        if (fs::exists(tok_path, ec)) {
            TokenizerSpecialPolicy tok_policy;
            tok_policy.pretokenizer = 7;
            ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
                tok_path.string(), pinned_dir,
                integrity::TokenizerSpecialIds{.bos_required = false,
                                               .eos_required = true},
                tok_policy));
            tokenizer_ok = true;
        } else if (!SyntheticGeometryAllowed()) {
            return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
        }
    }

    // ── Index ──────────────────────────────────────────────────────────────
    // The converted-layer list is the union of what is on disk, so a resuming
    // window extends it instead of forgetting the layers an earlier run wrote.
    const std::vector<int> converted = ConvertedLayers(base, top.num_layers);
    const bool partial = static_cast<int>(converted.size()) != top.num_layers;
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"logestix-nemotron-ultra-nvfp4\",\n"
          << "  \"model\": \"NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n"
          << "  \"layer_types\": [";
        for (std::size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"attn_layer_idx\": [";
        bool first_idx = true;
        for (int i = 0; i < top.num_layers; ++i) {
            if (top.layer_types[i] != "attention") continue;
            if (!first_idx) j << ",";
            j << i;
            first_idx = false;
        }
        j << "],\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"mamba_num_heads\": " << top.mamba_num_heads
          << ",\n  \"mamba_head_dim\": " << top.mamba_head_dim
          << ",\n  \"ssm_state_size\": " << top.ssm_state_size
          << ",\n  \"n_groups\": " << top.n_groups
          << ",\n  \"conv_kernel\": " << top.conv_kernel
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"shared_expert_intermediate_size\": "
          << top.shared_expert_intermediate_size
          << ",\n  \"moe_latent_size\": " << top.moe_latent_size
          << ",\n  \"n_group\": " << top.n_group
          << ",\n  \"topk_group\": " << top.topk_group
          << ",\n  \"routed_scaling_factor\": " << top.routed_scaling_factor
          << ",\n  \"norm_topk_prob\": " << (top.norm_topk_prob ? "true" : "false")
          << ",\n  \"experts_no_gate\": true"
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"eos_token_id\": [";
        for (std::size_t i = 0; i < top.eos_token_ids.size(); ++i) {
            if (i) j << ",";
            j << top.eos_token_ids[i];
        }
        j << "]"
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"mtp_tensors\": " << mtp_cut
          << ",\n  \"partial\": " << (partial ? "true" : "false")
          << ",\n  \"layers_converted\": [";
        for (std::size_t i = 0; i < converted.size(); ++i) {
            if (i) j << ",";
            j << converted[i];
        }
        j << "],\n  \"peak_expert_records\": " << peak_expert_records
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\",\n  \"vram\": " << FormatVramJson(top.vram)
          << ",\n  \"deferred\": {\"mtp\": \"cut\", \"mtp_tensors\": " << mtp_cut
          << "}"
          << ",\n  \"total_pinned_tensors\": " << total_pinned
          << ",\n  \"total_experts\": " << total_experts << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Output verification: every produced arrow file re-opens cleanly.
    std::error_code dir_ec;
    for (fs::recursive_directory_iterator it(base, dir_ec), end;
         !dir_ec && it != end; it.increment(dir_ec)) {
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".arrow") continue;
        auto res = ReadRecordBatchFromFile(it->path());
        if (!res.ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                         it->path().string());
    }
    if (dir_ec) return arrow::Status::IOError("cannot list ", base.string());

    lsxcommon::log::info(absl::StrFormat(
        "Nemotron-Ultra conversion: window [%d,%d] of %d layers (%d mamba/%d moe/"
        "%d attn), %d experts, %d MTP tensors cut, %lld pinned tensors, peak "
        "expert records %lld",
        first, last, top.num_layers,
        static_cast<int>(std::count(top.layer_types.begin(), top.layer_types.end(),
                                    std::string("mamba"))),
        static_cast<int>(std::count(top.layer_types.begin(), top.layer_types.end(),
                                    std::string("moe"))),
        static_cast<int>(std::count(top.layer_types.begin(), top.layer_types.end(),
                                    std::string("attention"))),
        total_experts, mtp_cut, static_cast<long long>(total_pinned),
        static_cast<long long>(peak_expert_records)));
    return arrow::Status::OK();
}

} // namespace lsxcommon::nemotron_ultra
