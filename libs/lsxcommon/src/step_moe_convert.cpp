#include "lsxcommon/step_moe_convert.h"

#include "step_moe_constants.h"
#include "step_moe_dequant.h"

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

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

namespace lsxcommon::step {
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

bool CheckFrozenGeometry(const StepTopology &top, std::string &why)
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
        return bad("num_attention_groups", top.num_key_value_heads, kKvHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("moe_num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("moe_top_k", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.shared_intermediate_size != kSharedIntermediate)
        return bad("share_expert_dim", top.shared_intermediate_size,
                   kSharedIntermediate);
    if (top.dense_intermediate_size != kDenseIntermediate)
        return bad("intermediate_size", top.dense_intermediate_size, kDenseIntermediate);
    if (top.first_moe_layer != kFirstMoeLayer)
        return bad("first moe layer", top.first_moe_layer, kFirstMoeLayer);
    if (top.num_mtp_layers != kMtpLayers)
        return bad("num_nextn_predict_layers", top.num_mtp_layers, kMtpLayers);
    if (top.sliding_window != kSlidingWindow)
        return bad("sliding_window", top.sliding_window, kSlidingWindow);
    for (int i = 0; i < top.num_layers; ++i) {
        const char *want = IsFullAttention(i) ? "full_attention"
                                              : "sliding_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
        const char *want_mlp = IsMoeLayer(i) ? "sparse" : "dense";
        if (top.mlp_types[static_cast<size_t>(i)] != want_mlp)
            return bad("mlp dispatch[i]", i, 0);
    }
    return true;
}

bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &r)
{
    return gpu::quantize_tensor_gpu(data, numel, dtype, nullptr, r.shape, {},
                                    r.packed, r.scales, r.global_scale);
}

arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

arrow::Status WriteExpertFile(const std::filesystem::path &base_dir, int layer,
                              int expert, PackedExpert &&pe)
{
    namespace fs = std::filesystem;
    fs::path dir = base_dir / "experts" / absl::StrCat("layer_", layer);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch,
                                  dir / absl::StrCat("exp_", expert, ".arrow"));
}

} // namespace

ParsedStepTensor classify_step_tensor(const std::string &name)
{
    ParsedStepTensor r;
    if (name == "lm_head.weight") {
        r.kind = StepTensorKind::LmHead;
        return r;
    }
    if (name.rfind("vision_model.", 0) == 0 ||
        name.rfind("vit_large_projector.", 0) == 0 ||
        name.rfind("vit.", 0) == 0 || name.rfind("model.visual.", 0) == 0 ||
        name.rfind("model.vit", 0) == 0) {
        r.kind = StepTensorKind::VisionBallast;
        return r;
    }
    const std::string kPrefix = "model.";
    if (name.rfind(kPrefix, 0) != 0) return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = StepTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = StepTensorKind::FinalNorm;
        return r;
    }

    std::string body = rest;
    const std::string kLayers = "layers.";
    if (rest.rfind(kLayers, 0) != 0) return r;
    const auto digits_start = kLayers.size();
    const auto digits_end = rest.find('.', digits_start);
    if (digits_end == std::string::npos) return r;
    int layer = 0;
    if (!detail::ParseTensorIndex(
            std::string_view(rest).substr(digits_start, digits_end - digits_start),
            layer))
        return r;
    r.layer = layer;
    body = rest.substr(digits_end + 1);

    // The three MTP layers (92..94) are stored-not-applied ballast. Two
    // independent triggers: the frozen layer bound (production) and the
    // MTP-only tensor names (so a synthetic fixture with a smaller layer count
    // hits the same branch).
    if (layer >= frozen::kLayers ||
        body.rfind("transformer.shared_head.", 0) == 0 ||
        body == "enorm.weight" || body == "hnorm.weight" ||
        body == "eh_proj.weight") {
        r.kind = StepTensorKind::MtpBallast;
        return r;
    }

    using K = StepTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::AttnQ},
        {"self_attn.k_proj.weight", K::AttnK},
        {"self_attn.v_proj.weight", K::AttnV},
        {"self_attn.o_proj.weight", K::AttnO},
        {"self_attn.g_proj.weight", K::AttnG},
        {"self_attn.q_norm.weight", K::AttnQNorm},
        {"self_attn.k_norm.weight", K::AttnKNorm},
        {"self_attn.sparse_indexer_q.weight", K::IndexQ},
        {"self_attn.sparse_indexer_k.weight", K::IndexK},
        {"self_attn.sparse_indexer_z.weight", K::IndexZ},
        {"self_attn.sparse_indexer_w.weight", K::IndexW},
        {"self_attn.sparse_indexer_q_norm.weight", K::IndexQNorm},
        {"self_attn.sparse_indexer_k_norm.weight", K::IndexKNorm},
        {"self_attn.sparse_indexer_k_norm.bias", K::IndexKNormBias},
        {"self_attn.ssmax_s", K::IndexSsmax},
        {"mlp.gate_proj.weight", K::DenseMlpGate},
        {"mlp.up_proj.weight", K::DenseMlpUp},
        {"mlp.down_proj.weight", K::DenseMlpDown},
        {"moe.gate.weight", K::RouterGate},
        {"moe.router_bias", K::RouterBias},
        {"moe.gate_proj.weight", K::ExpertGate},
        {"moe.up_proj.weight", K::ExpertUp},
        {"moe.down_proj.weight", K::ExpertDown},
        {"moe.gate_proj.weight_scale_inv", K::ExpertGateScale},
        {"moe.up_proj.weight_scale_inv", K::ExpertUpScale},
        {"moe.down_proj.weight_scale_inv", K::ExpertDownScale},
        {"share_expert.gate_proj.weight", K::SharedGateProj},
        {"share_expert.up_proj.weight", K::SharedUpProj},
        {"share_expert.down_proj.weight", K::SharedDownProj},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool step_tensor_is_nvfp4(StepTensorKind kind)
{
    switch (kind) {
    case StepTensorKind::AttnQ:
    case StepTensorKind::AttnK:
    case StepTensorKind::AttnV:
    case StepTensorKind::AttnO:
    case StepTensorKind::AttnG:
    case StepTensorKind::DenseMlpGate:
    case StepTensorKind::DenseMlpUp:
    case StepTensorKind::DenseMlpDown:
    case StepTensorKind::ExpertGate:
    case StepTensorKind::ExpertDown:
    case StepTensorKind::ExpertUp:
    case StepTensorKind::SharedGateProj:
    case StepTensorKind::SharedUpProj:
    case StepTensorKind::SharedDownProj:
        return true;
    default:
        return false;
    }
}

bool step_tensor_is_routed_expert(StepTensorKind kind)
{
    return kind == StepTensorKind::ExpertGate ||
           kind == StepTensorKind::ExpertDown ||
           kind == StepTensorKind::ExpertUp ||
           kind == StepTensorKind::ExpertGateScale ||
           kind == StepTensorKind::ExpertDownScale ||
           kind == StepTensorKind::ExpertUpScale;
}

StepTopology ParseStepConfig(const std::string &input_dir)
{
    StepTopology top;
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

    // step5_vl keeps the text backbone under text_config.
    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object(); tc.error() == simdjson::SUCCESS)
        src = tc.value();

    top.hidden_size = get_i(src, "hidden_size", 0);
    top.num_layers = get_i(src, "num_hidden_layers", 0);
    top.vocab_size = get_i(src, "vocab_size", 0);
    top.num_attention_heads = get_i(src, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(
        src, "num_attention_groups", get_i(src, "num_key_value_heads", 0));
    top.head_dim = get_i(src, "head_dim", 0);
    top.dense_intermediate_size = get_i(src, "intermediate_size", 0);
    top.moe_intermediate_size = get_i(src, "moe_intermediate_size", 0);
    top.shared_intermediate_size = get_i(src, "share_expert_dim", 0);
    top.num_experts = get_i(src, "moe_num_experts", get_i(src, "num_experts", 0));
    top.num_experts_per_tok =
        get_i(src, "moe_top_k", get_i(src, "num_experts_per_tok", 0));
    top.num_mtp_layers = get_i(src, "num_nextn_predict_layers", 0);
    top.sliding_window = get_i(src, "sliding_window", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-5);
    top.router_scaling_factor = get_d(src, "moe_router_scaling_factor", 3.0);
    top.bos_token_id = get_i(src, "bos_token_id", -1);
    top.pad_token_id = get_i(src, "pad_token_id", -1);

    {
        simdjson::ondemand::value eos;
        if (src["eos_token_id"].get(eos) == simdjson::SUCCESS) {
            if (auto one = eos.get_int64(); one.error() == simdjson::SUCCESS) {
                top.eos_token_id = static_cast<int>(one.value());
            } else if (auto arr = eos.get_array(); arr.error() == simdjson::SUCCESS) {
                int idx = 0;
                for (auto v : arr.value()) {
                    int64_t id;
                    if (v.get_int64().get(id) != simdjson::SUCCESS) continue;
                    if (idx == 0) top.eos_token_id = static_cast<int>(id);
                    else if (idx == 1) top.eos_token_id2 = static_cast<int>(id);
                    ++idx;
                }
            }
        }
    }

    // Layer types: `layer_types` carries one entry per main + MTP layer.
    if (auto lt = src["layer_types"].get_array(); lt.error() == simdjson::SUCCESS)
        for (auto e : lt.value()) {
            std::string_view s;
            if (e.get_string().get(s) == simdjson::SUCCESS)
                top.layer_types.emplace_back(s);
        }
    if (top.layer_types.size() < static_cast<size_t>(top.num_layers))
        top.layer_types.clear();

    // MoE dispatch: `moe_layer_list` is the authoritative layer set.
    absl::btree_set<int> moe_layers;
    if (auto ml = src["moe_layer_list"].get_array(); ml.error() == simdjson::SUCCESS)
        for (auto e : ml.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                moe_layers.insert(static_cast<int>(v));
        }
    // Fallback / cross-check: `moe_layers_enum` (a comma-separated string).
    if (moe_layers.empty()) {
        std::string_view s;
        if (src["moe_layers_enum"].get_string().get(s) == simdjson::SUCCESS) {
            std::stringstream es{std::string(s)};
            std::string item;
            while (std::getline(es, item, ','))
                if (!item.empty()) moe_layers.insert(std::atoi(item.c_str()));
        }
    }
    if (top.num_layers > 0) {
        top.first_moe_layer = -1;
        for (int l = 0; l < top.num_layers; ++l) {
            const bool is_moe = moe_layers.contains(l);
            if (is_moe && top.first_moe_layer < 0) top.first_moe_layer = l;
            top.mlp_types.emplace_back(is_moe ? "sparse" : "dense");
        }
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.vocab_size <= 0 ||
        top.num_attention_heads <= 0 || top.num_key_value_heads <= 0 ||
        top.head_dim <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.dense_intermediate_size <= 0 ||
        static_cast<int>(top.mlp_types.size()) != top.num_layers ||
        static_cast<int>(top.layer_types.size()) < top.num_layers) {
        top.error_msg = "config.json missing required step5 fields";
        return top;
    }
    top.layer_types.resize(static_cast<size_t>(top.num_layers));
    for (const auto &t : top.layer_types)
        if (t != "full_attention" && t != "sliding_attention") {
            top.error_msg = "unsupported layer type: " + t;
            return top;
        }
    return top;
}

arrow::Status ConvertStepMoeModel(const std::string &input_dir,
                                  const std::string &output_root)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    StepTopology top = ParseStepConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Step-5 config error: ", top.error_msg);
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, why))
            return arrow::Status::Invalid("Step-5 frozen geometry mismatch: ", why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    // `vit.safetensors` is a duplicate copy of `model-vit-00001.safetensors`
    // (identical 667 tensor names) and both are pure ballast — reading it twice
    // would double the "skipped_visual_tensors" count.
    {
        SafetensorsModel deduped;
        absl::btree_set<std::string> seen_files;
        for (auto &shard : model.shards) {
            if (shard.filename == "vit.safetensors") continue;
            if (!seen_files.insert(shard.filename).second) continue;
            deduped.shards.push_back(std::move(shard));
        }
        model = std::move(deduped);
    }
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    // (2) Classify the inventory up front. Unknown names are hard errors; the
    // routed experts are indexed per (layer, projection) because they are ~96%
    // of the bytes and must never be held in host RAM as a whole.
    struct ExpertLoc {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    // key: "L|gate"/"L|down"/"L|up"/"L|gate_s" / ...
    absl::btree_map<std::string, ExpertLoc> expert_index;
    absl::btree_set<std::string> ballast_names;
    int layer_kind_count[1024] = {};
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_step_tensor(t.name);
            if (p.kind == StepTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in Step-5 checkpoint: ", t.name);
            if (p.kind == StepTensorKind::VisionBallast) {
                ballast_names.insert(t.name);
                continue;
            }
            if (p.kind == StepTensorKind::MtpBallast) {
                ballast_names.insert(t.name);
                continue;
            }
            if (step_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.layer >= top.num_layers)
                    return arrow::Status::Invalid("expert tensor without a valid layer: ",
                                                  t.name);
                if (top.mlp_types[static_cast<size_t>(p.layer)] != "sparse")
                    return arrow::Status::Invalid(
                        "expert tensor on a dense-MLP layer: ", t.name);
                const char *suffix = "?";
                switch (p.kind) {
                case StepTensorKind::ExpertGate: suffix = "gate"; break;
                case StepTensorKind::ExpertDown: suffix = "down"; break;
                case StepTensorKind::ExpertUp: suffix = "up"; break;
                case StepTensorKind::ExpertGateScale: suffix = "gate_s"; break;
                case StepTensorKind::ExpertDownScale: suffix = "down_s"; break;
                case StepTensorKind::ExpertUpScale: suffix = "up_s"; break;
                default: break;
                }
                expert_index.emplace(absl::StrCat(p.layer, "|", suffix),
                                     ExpertLoc{&shard, &t});
            }
            if (p.layer >= 0 && p.layer < top.num_layers) ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (layer_kind_count[l] == 0)
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    const fs::path indexer_dir = base / "indexer";
    for (const fs::path &d : {base, pinned_dir, router_dir, shared_dir, indexer_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> indexer_by_layer;
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
    auto add_raw = [&](const TensorInfo &t, std::vector<uint8_t> &&data,
                       std::vector<TensorRecord> &dst) {
        TensorRecord r;
        r.name = t.name;
        r.dtype = t.dtype;
        r.shape = t.shape;
        r.data = std::move(data);
        dst.push_back(std::move(r));
        ++total_pinned;
    };

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_step_tensor(t.name);
            if (step_tensor_is_routed_expert(p.kind)) continue;
            if (p.kind == StepTensorKind::VisionBallast ||
                p.kind == StepTensorKind::MtpBallast)
                continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ", t.name);

            switch (p.kind) {
            case StepTensorKind::Embed:
            case StepTensorKind::LmHead:
            case StepTensorKind::FinalNorm:
                // Embed/head stay BF16 on the device: the head multiplies every
                // logit, so a 4-bit row error moves every logit.
                add_raw(t, std::move(data), global_raw);
                break;
            case StepTensorKind::InputNorm:
            case StepTensorKind::PostNorm:
            case StepTensorKind::AttnQNorm:
            case StepTensorKind::AttnKNorm:
                // Norms keep F32 (norm_dtype=float32 in the checkpoint).
                add_raw(t, std::move(data), raw_by_layer[p.layer]);
                break;
            case StepTensorKind::RouterGate:
            case StepTensorKind::RouterBias:
                // The routing bias moves the top-k SELECTION, so a 4-bit
                // router swaps the expert set.
                add_raw(t, std::move(data), router_by_layer[p.layer]);
                break;
            case StepTensorKind::IndexQ:
            case StepTensorKind::IndexK:
            case StepTensorKind::IndexZ:
            case StepTensorKind::IndexW:
            case StepTensorKind::IndexQNorm:
            case StepTensorKind::IndexKNorm:
            case StepTensorKind::IndexKNormBias:
            case StepTensorKind::IndexSsmax:
                // DSA indexer: stored for the >4096-token stage; it never
                // affects attention values at the supported context.
                add_raw(t, std::move(data), indexer_by_layer[p.layer]);
                ++top.indexer_tensors;
                break;
            case StepTensorKind::SharedGateProj:
            case StepTensorKind::SharedUpProj:
            case StepTensorKind::SharedDownProj:
                ARROW_RETURN_NOT_OK(
                    add_nvfp4(t, data, shared_nvfp4_by_layer[p.layer]));
                break;
            case StepTensorKind::AttnQ:
            case StepTensorKind::AttnK:
            case StepTensorKind::AttnV:
            case StepTensorKind::AttnO:
            case StepTensorKind::AttnG:
            case StepTensorKind::DenseMlpGate:
            case StepTensorKind::DenseMlpUp:
            case StepTensorKind::DenseMlpDown:
                ARROW_RETURN_NOT_OK(add_nvfp4(t, data, nvfp4_by_layer[p.layer]));
                break;
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    top.skipped_ballast_tensors = static_cast<int>(ballast_names.size());
    {
        int mtp = 0;
        for (const auto &n : ballast_names) {
            auto p = classify_step_tensor(n);
            if (p.kind == StepTensorKind::MtpBallast) ++mtp;
        }
        top.skipped_mtp_tensors = mtp;
        top.skipped_ballast_tensors -= mtp;
    }

    // (3) Routed experts: layer-major, projection-major second pass. Each
    // projection is FUSED [E, N, K] on disk; dequantize the whole stack on the
    // GPU, then NVFP4-quantize one expert slice at a time and write one
    // ExpertSchema file per (layer, expert). No host-resident expert pool.
    const int H = top.hidden_size;
    const int I = top.moe_intermediate_size;
    const int E = top.num_experts;
    int experts_done = 0;
    int experts_total = 0;
    for (int l = 0; l < top.num_layers; ++l)
        if (top.mlp_types[static_cast<size_t>(l)] == "sparse") ++experts_total;
    const int moe_layers_total = experts_total;
    experts_total *= E;
    int last_pct = -1;
    const bool skip_experts = std::getenv("LSX_STEP_SKIP_EXPERTS") != nullptr;
    if (skip_experts)
        lsxcommon::log::info(
            "Step-5: LSX_STEP_SKIP_EXPERTS set — reusing the existing expert store");

    struct ProjSpec {
        const char *suffix;
        const char *scale_suffix;
        int rows;   // per expert
        int cols;
        int k;      // repo expert slot: 0 = gate(w1), 1 = down(w2), 2 = up(w3)
    };
    const ProjSpec kProj[3] = {
        {"|gate", "|gate_s", I, H, 0},
        {"|down", "|down_s", H, I, 1},
        {"|up", "|up_s", I, H, 2},
    };

    for (int l = 0; !skip_experts && l < top.num_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] != "sparse") continue;
        std::vector<std::vector<uint8_t>> bf16_stack(3);
        std::vector<std::vector<int64_t>> shapes(3);
        for (int pi = 0; pi < 3; ++pi) {
            const auto &ps = kProj[pi];
            auto wit = expert_index.find(absl::StrCat(l, ps.suffix));
            auto sit = expert_index.find(absl::StrCat(l, ps.scale_suffix));
            if (wit == expert_index.end())
                return arrow::Status::Invalid(
                    absl::StrCat("missing expert tensor for layer ", l, " ",
                                 ps.suffix));
            const TensorInfo &wt = *wit->second.info;
            const int srow = (ps.rows + 127) / 128;
            const int scol = (ps.cols + 127) / 128;
            if (wt.shape.size() != 3 || wt.shape[0] != E ||
                wt.shape[1] != ps.rows || wt.shape[2] != ps.cols)
                return arrow::Status::Invalid(
                    absl::StrCat("unexpected expert weight shape on layer ", l,
                                 " ", ps.suffix));
            auto wdata = ReadTensorData(input_dir, *wit->second.shard, wt);
            if (wdata.empty())
                return arrow::Status::IOError("failed to read expert stack layer ",
                                              l, ps.suffix);
            if (sit != expert_index.end()) {
                // FP8 block-scaled stack: dequantize on the GPU.
                const TensorInfo &st = *sit->second.info;
                if (wt.dtype != "F8_E4M3" || st.dtype != "F32" ||
                    st.shape.size() != 3 || st.shape[0] != E ||
                    st.shape[1] != srow || st.shape[2] != scol)
                    return arrow::Status::Invalid(absl::StrCat(
                        "unexpected expert weight/scale shape on layer ", l,
                        " ", ps.suffix));
                auto sdata = ReadTensorData(input_dir, *sit->second.shard, st);
                if (sdata.empty())
                    return arrow::Status::IOError(
                        "failed to read expert scales layer ", l, ps.scale_suffix);
                if (!StepFp8StackedDequantToBf16Host(
                        wdata.data(),
                        reinterpret_cast<const float *>(sdata.data()), E,
                        ps.rows, ps.cols, bf16_stack[static_cast<size_t>(pi)]))
                    return arrow::Status::IOError(
                        absl::StrCat("FP8 dequant failed on layer ", l,
                                     ps.suffix));
            } else {
                // `modules_to_not_convert` keeps some layers' experts in BF16
                // (no weight_scale_inv tensor): use the stack verbatim.
                if (wt.dtype != "BF16")
                    return arrow::Status::Invalid(absl::StrCat(
                        "expert layer ", l, " ", ps.suffix,
                        " has neither an FP8 scale grid nor BF16 weights"));
                bf16_stack[static_cast<size_t>(pi)] = std::move(wdata);
            }
            shapes[static_cast<size_t>(pi)] = {ps.rows, ps.cols};
        }

        for (int e = 0; e < E; ++e) {
            {
                const int pct = experts_total > 0 ? (100 * experts_done) / experts_total
                                                  : 100;
                if (pct != last_pct && pct % 5 == 0) {
                    lsxcommon::log::info(absl::StrFormat(
                        "Step-5 experts: %d/%d (%d%%) layer %d", experts_done,
                        experts_total, pct, l));
                    last_pct = pct;
                }
                ++experts_done;
            }
            DenseNvfp4Record recs[3];
            for (int pi = 0; pi < 3; ++pi) {
                const auto &ps = kProj[pi];
                const int64_t numel = static_cast<int64_t>(ps.rows) * ps.cols;
                const uint8_t *src =
                    bf16_stack[static_cast<size_t>(pi)].data() +
                    static_cast<size_t>(e) * static_cast<size_t>(numel) * 2;
                recs[pi].name = absl::StrCat("model.layers.", l, ".moe.",
                                             pi == 0 ? "gate_proj"
                                                     : (pi == 1 ? "down_proj"
                                                                : "up_proj"),
                                             ".weight");
                recs[pi].shape = shapes[static_cast<size_t>(pi)];
                recs[pi].dtype = "BF16";
                if (!QuantizeSlice(src, numel, "BF16", recs[pi]))
                    return arrow::Status::IOError(
                        absl::StrCat("quantize failed: layer ", l, " expert ", e));
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
            ARROW_RETURN_NOT_OK(WriteExpertFile(base, l, e, std::move(pe)));
        }
    }

    // (4) Dense-side artifacts.
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                       pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(recs),
                                       pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : router_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(recs),
                                       router_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : indexer_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            indexer_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : shared_nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // (5) Tokenizer. The chat template spells the BOS marker out as text, so
    // the reference tokenizer must NOT auto-prepend it. Step-5's
    // tokenizer.json is the DeepSeek three-stage Sequence — \p{N}{1,3}
    // Isolated, CJK Isolated, then the six-alternative main pattern — so the
    // engine's WordStream (pretokenizer 0, tokenizer_kind 0) is the matching
    // split. The o200k case-run pattern (kind 4) is Flint's, not Step's, and
    // mis-splits mixed-case identifiers and letter/digit boundaries.
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        tok_policy.pretokenizer = 0;
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

    // (6) Conversion-time VRAM profile.
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.num_moe_layers = moe_layers_total;
    vg.experts_per_layer = top.num_experts;
    vg.context_length = frozen::kMaxContextTokens;
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            inv.push_back({t.name, t.numel(), t.byte_size()});
    const VramProfile vram = EstimateVram(
        inv,
        {+[](const std::string &n) {
             return step_tensor_is_nvfp4(classify_step_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return step_tensor_is_routed_expert(classify_step_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         nullptr},
        vg);
    top.vram = vram;

    // (7) Structural manifest (deterministic byte content).
    {
        std::ostringstream j;
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n"
          << "  \"model\": \"" << frozen::kModelName << "\",\n"
          << "  \"model_type\": \"" << frozen::kType << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n  \"mlp_layer_types\": [";
        for (size_t i = 0; i < top.mlp_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.mlp_types[i] << "\"";
        }
        j << "],\n  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"rotary_span_sliding\": " << frozen::kRotarySpanSliding
          << ",\n  \"rotary_span_full\": " << frozen::kRotarySpanFull
          << ",\n  \"sliding_window\": " << top.sliding_window
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"router_scaling_factor\": " << top.router_scaling_factor
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate_size
          << ",\n  \"shared_intermediate_size\": " << top.shared_intermediate_size
          << ",\n  \"first_moe_layer\": " << top.first_moe_layer
          << ",\n  \"bf16_expert_layers\": [88, 89, 90]"
          << ",\n  \"num_mtp_layers_stripped\": " << top.num_mtp_layers
          << ",\n  \"max_context_tokens\": " << frozen::kMaxContextTokens
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"eos_token_id2\": " << top.eos_token_id2
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"indexer\": {\"proxy_dim\": " << frozen::kIndexProxyDim
          << ", \"heads\": " << frozen::kIndexHeads
          << ", \"topk\": " << frozen::kIndexTopk
          << ", \"region_size\": " << frozen::kIndexRegion
          << ", \"rope_dim\": " << frozen::kIndexRopeDim
          << ", \"applied\": false}"
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"skipped_visual_tensors\": " << top.skipped_ballast_tensors
          << ",\n  \"skipped_mtp_tensors\": " << top.skipped_mtp_tensors
          << ",\n  \"indexer_tensors\": " << top.indexer_tensors
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << moe_layers_total * top.num_experts
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // (8) Final verification: every produced arrow file must re-open cleanly.
    for (const fs::path &dir : {pinned_dir, router_dir, shared_dir, indexer_dir}) {
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
        "Step-5 conversion: %d layers (%d MoE), %d experts/layer, %d pinned "
        "tensors, %d visual + %d MTP ballast tensors stripped",
        top.num_layers, top.num_layers - top.first_moe_layer, top.num_experts,
        top.total_pinned_tensors, top.skipped_ballast_tensors,
        top.skipped_mtp_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::step
