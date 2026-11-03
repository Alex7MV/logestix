#include "lsxcommon/nemotron_moe_convert.h"
#include "lsxcommon/synthetic_geometry.h"
#include "nemotron_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "tensor_index_parse.h"

#include <simdjson.h>
#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>

namespace lsxcommon::nemotron_moe {
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

// Legacy NemotronH layer tags map onto the modeling block types.
std::string normalize_block_type(std::string_view tag)
{
    if (tag == "linear_attention" || tag == "mamba") return "mamba";
    if (tag == "full_attention" || tag == "attention") return "attention";
    if (tag == "moe" || tag == "mlp") return "moe";
    return std::string(tag);
}

bool check_frozen_geometry(const NemotronTopology &top, std::string &why)
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
    if (top.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   top.shared_expert_intermediate_size, kSharedIntermediate);

    int mamba = 0, moe = 0, attn = 0;
    for (const auto &t : top.layer_types) {
        if (t == "mamba") ++mamba;
        else if (t == "moe") ++moe;
        else if (t == "attention") ++attn;
        else { why = "unknown layer type: " + t; return false; }
    }
    if (mamba != kMambaLayers || moe != kMoeLayers || attn != kAttnLayers) {
        why = absl::StrFormat("layer mix got mamba=%d moe=%d attn=%d, want %d/%d/%d",
                              mamba, moe, attn, kMambaLayers, kMoeLayers, kAttnLayers);
        return false;
    }
    return true;
}

} // namespace

ParsedNemotronTensor classify_nemotron_tensor(const std::string &name)
{
    using K = NemotronTensorKind;
    ParsedNemotronTensor r;
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
    if (!detail::ParseTensorIndex(std::string_view(rest).substr(0, dend),
                                  layer))
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
        if (!detail::ParseTensorIndex(std::string_view(e).substr(0, ed),
                                      expert))
            return r;  // malformed index -> Unknown, never throw
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

bool nemotron_tensor_is_nvfp4(NemotronTensorKind kind)
{
    using K = NemotronTensorKind;
    switch (kind) {
    case K::MambaInProj:
    case K::MambaOutProj:
    case K::AttnQ:
    case K::AttnK:
    case K::AttnV:
    case K::AttnO:
    case K::SharedUpProj:
    case K::SharedDownProj:
    case K::ExpertUpProj:
    case K::ExpertDownProj:
        return true;
    default:
        return false;
    }
}

NemotronTopology ParseNemotronConfig(const std::string &input_dir)
{
    NemotronTopology top;
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
    top.eos_token_id = get_i(root, "eos_token_id", -1);
    top.bos_token_id = get_i(root, "bos_token_id", -1);
    top.pad_token_id = get_i(root, "pad_token_id", -1);

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
    return top;
}

arrow::Status ConvertNemotronModel(const std::string &input_dir,
                                   const std::string &output_root)
{
    namespace fs = std::filesystem;

    NemotronTopology top = ParseNemotronConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Nemotron config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > n_routed_experts");

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid("Nemotron frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Inventory: classify up front. MTP is a counted cut; unknown is fatal.
    int mtp_cut = 0;
    absl::btree_map<int, int> layer_kind_count;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_nemotron_tensor(t.name);
            if (p.kind == NemotronTensorKind::Mtp) {
                ++mtp_cut;
                continue;
            }
            if (p.kind == NemotronTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in nemotron_h checkpoint: ", t.name);
            if (p.layer >= 0) ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    const fs::path base = fs::path(output_root) / "logestix";
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
            auto p = classify_nemotron_tensor(t.name);
            if (p.kind == NemotronTensorKind::Mtp) continue;  // counted cut
            // Routed experts are converted by the dedicated per-expert pass.
            if (p.kind == NemotronTensorKind::ExpertUpProj ||
                p.kind == NemotronTensorKind::ExpertDownProj)
                continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            if (nemotron_tensor_is_nvfp4(p.kind)) {
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
            } else {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                switch (p.kind) {
                case NemotronTensorKind::Embed:
                case NemotronTensorKind::LmHead:
                case NemotronTensorKind::FinalNorm:
                    global_raw.push_back(std::move(r));
                    break;
                default:
                    raw_by_layer[p.layer].push_back(std::move(r));
                    break;
                }
                ++total_pinned;
            }
        }
    }

    // Routed experts: per-expert up/down pair (no gate projection).
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    using ExpertKey = std::pair<int, int>;  // (layer, expert)
    absl::btree_map<ExpertKey, DenseNvfp4Record> up_map, down_map;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_nemotron_tensor(t.name);
            if (p.kind != NemotronTensorKind::ExpertUpProj &&
                p.kind != NemotronTensorKind::ExpertDownProj)
                continue;
            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);
            DenseNvfp4Record r;
            r.name = t.name;
            r.shape = t.shape;
            r.dtype = t.dtype;
            if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                return arrow::Status::IOError("GPU quantization failed for ",
                                              t.name);
            if (p.kind == NemotronTensorKind::ExpertUpProj) {
                if (t.shape.size() != 2 || t.shape[0] != I || t.shape[1] != H)
                    return arrow::Status::Invalid("bad expert up shape: ",
                                                  t.name);
                up_map.insert_or_assign({p.layer, p.expert}, std::move(r));
            } else {
                if (t.shape.size() != 2 || t.shape[0] != H || t.shape[1] != I)
                    return arrow::Status::Invalid("bad expert down shape: ",
                                                  t.name);
                down_map.insert_or_assign({p.layer, p.expert}, std::move(r));
            }
        }
    }
    for (int l = 0; l < top.num_layers; ++l) {
        if (top.layer_types[l] != "moe") continue;
        for (int e = 0; e < E; ++e) {
            auto up = up_map.find({l, e});
            auto dn = down_map.find({l, e});
            if (up == up_map.end() || dn == down_map.end())
                return arrow::Status::Invalid(
                    "missing routed expert: layer ", l, " expert ", e);
            PackedExpert pe;
            // NemotronH experts are non-gated MLPs: w1 is unused, w3 = up,
            // w2 = down (repo PackedExpert slot convention).
            pe.w3_packed = up->second.packed;
            pe.w3_scales = up->second.scales;
            pe.w3_shape = up->second.shape;
            pe.w3_global_scale = up->second.global_scale;
            pe.w2_packed = dn->second.packed;
            pe.w2_scales = dn->second.scales;
            pe.w2_shape = dn->second.shape;
            pe.w2_global_scale = dn->second.global_scale;
            pe.w1_shape = {0, 0};
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
        }
    }
    int total_experts = 0;
    for (int l = 0; l < top.num_layers; ++l)
        if (top.layer_types[l] == "moe") total_experts += E;

    if (!global_raw.empty()) {
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    }
    for (auto &[layer, recs] : nvfp4_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    }
    for (auto &[layer, recs] : raw_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    }

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Tokenizer: mandatory for real checkpoints; synthetic fixtures may omit.
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 7;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true}, tok_policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // Conversion-time VRAM profile (routed experts are the max term).
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = frozen::kAttnLayers;
        vg.num_moe_layers = frozen::kMoeLayers;
        vg.experts_per_layer = top.num_experts;
        vg.context_length = ReadContextLength(
            (fs::path(input_dir) / "config.json").string());
        std::vector<VramTensorInfo> inv;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                inv.push_back({t.name, t.numel(), t.byte_size()});
        top.vram = EstimateVram(
            inv,
            {+[](const std::string &n) {
                 return nemotron_tensor_is_nvfp4(
                     classify_nemotron_tensor(n).kind);
             },
             +[](const std::string &n) -> VramTensorClass {
                 auto k = classify_nemotron_tensor(n).kind;
                 return (k == NemotronTensorKind::ExpertUpProj ||
                         k == NemotronTensorKind::ExpertDownProj)
                            ? VramTensorClass::RoutedExpert
                            : VramTensorClass::Resident;
             }},
            vg);
    }

    // Structural manifest (deterministic).
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"logestix-nemotron-nvfp4\",\n"
          << "  \"model\": \"Nemotron-3.5-Lightning-30B-A3B\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n"
          << "  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"attn_layer_idx\": [";
        bool first = true;
        for (int i = 0; i < top.num_layers; ++i) {
            if (top.layer_types[i] != "attention") continue;
            if (!first) j << ",";
            j << i;
            first = false;
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
          << ",\n  \"router_scaling_factor\": " << frozen::kRoutedScale
          << ",\n  \"experts_no_gate\": true"
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"vram\": {\n"
          << "    \"min_bytes\": " << top.vram.min_bytes
          << ",\n    \"optimal_bytes\": " << top.vram.optimal_bytes
          << ",\n    \"max_bytes\": " << top.vram.max_bytes
          << ",\n    \"context_tokens\": " << top.vram.context_tokens << "\n  }"
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"deferred\": {\"mtp\": \"cut\", \"mtp_tensors\": " << mtp_cut
          << "}"
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << total_experts
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Output verification: every produced arrow file re-opens cleanly.
    std::error_code ec;
    for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".arrow") continue;
        auto res = ReadRecordBatchFromFile(it->path());
        if (!res.ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                          it->path().string());
    }
    if (ec)
        return arrow::Status::IOError("cannot list ", base.string());

    lsxcommon::log::info(absl::StrFormat(
        "Nemotron conversion: %d layers (%d mamba/%d moe/%d attn), %d experts, "
        "%d MTP tensors cut, %d pinned tensors",
        top.num_layers, frozen::kMambaLayers, frozen::kMoeLayers,
        frozen::kAttnLayers, top.num_experts, mtp_cut, top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::nemotron_moe
