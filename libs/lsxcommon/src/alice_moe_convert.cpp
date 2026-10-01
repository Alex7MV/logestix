#include "lsxcommon/alice_moe_convert.h"

#include "alice_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <simdjson.h>
#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <utility>

namespace lsxcommon::alice {
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

bool check_frozen_geometry(const AliceTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != kHidden) return bad("hidden_size", top.hidden_size, kHidden);
    if (top.num_layers != kLayers) return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.num_attention_heads != kHeads) return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKVHeads) return bad("num_key_value_heads", top.num_key_value_heads, kKVHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.linear_num_key_heads != kLinearKeyHeads) return bad("linear_num_key_heads", top.linear_num_key_heads, kLinearKeyHeads);
    if (top.linear_num_value_heads != kLinearValueHeads) return bad("linear_num_value_heads", top.linear_num_value_heads, kLinearValueHeads);
    if (top.linear_key_head_dim != kLinearKeyHeadDim) return bad("linear_key_head_dim", top.linear_key_head_dim, kLinearKeyHeadDim);
    if (top.linear_value_head_dim != kLinearValueHeadDim) return bad("linear_value_head_dim", top.linear_value_head_dim, kLinearValueHeadDim);
    if (top.linear_conv_kernel_dim != kLinearConvKernel) return bad("linear_conv_kernel_dim", top.linear_conv_kernel_dim, kLinearConvKernel);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts) return bad("num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK) return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate) return bad("moe_intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.shared_expert_intermediate_size != kSharedIntermediate) return bad("shared_expert_intermediate_size", top.shared_expert_intermediate_size, kSharedIntermediate);
    if (top.block_attn_res_block_size != kBlockAttnRes) return bad("block_attn_res_block_size", top.block_attn_res_block_size, kBlockAttnRes);
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want =
            (i % kFullAttnInterval == kFullAttnInterval - 1)
                ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
    }
    return true;
}

std::vector<uint8_t> read_tensor_data(
    const std::filesystem::path &input_path,
    const SafetensorsModel::Shard &shard,
    const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((input_path / shard.filename).string(), std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

bool quantize_slice(const uint8_t *data, int64_t numel, DenseNvfp4Record &r)
{
    return gpu::quantize_tensor_gpu(data, numel, "BF16", nullptr, r.shape, {},
                                    r.packed, r.scales, r.global_scale);
}

arrow::Status write_batch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

arrow::Status write_expert_file(const std::filesystem::path &dir, int expert,
                                PackedExpert &&pe)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch,
                                  dir / absl::StrCat("exp_", expert, ".arrow"));
}

bool ends_with(const std::string &s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parses "model.layers.<N>." and returns the body after it.
bool parse_layer_body(const std::string &name, int &layer, std::string &body)
{
    const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) return false;
    const size_t digits_start = prefix.size();
    const size_t dot = name.find('.', digits_start);
    if (dot == std::string::npos) return false;
    int v = 0;
    for (size_t i = digits_start; i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
        v = v * 10 + (name[i] - '0');
    }
    layer = v;
    body = name.substr(dot + 1);
    return true;
}

} // namespace

ParsedAliceTensor classify_alice_tensor(const std::string &name)
{
    ParsedAliceTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = AliceTensorKind::Mtp;
        return r;
    }
    if (name == "model.embed_tokens.weight") {
        r.kind = AliceTensorKind::Embed;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = AliceTensorKind::LmHead;
        return r;
    }
    if (name == "model.norm.weight") {
        r.kind = AliceTensorKind::FinalNorm;
        return r;
    }
    if (name == "model.attnres_final.res_proj.weight") {
        r.kind = AliceTensorKind::AttnResFinalProj;
        return r;
    }
    if (name == "model.attnres_final.res_norm_weight") {
        r.kind = AliceTensorKind::AttnResFinalNorm;
        return r;
    }

    int layer = -1;
    std::string body;
    if (!parse_layer_body(name, layer, body)) return r;
    r.layer = layer;

    using K = AliceTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"mlp_res_proj.weight", K::MlpResProj},
        {"mlp_res_norm_weight", K::MlpResNorm},
        {"attn_res_proj.weight", K::AttnResProj},
        {"attn_res_norm_weight", K::AttnResNorm},
        {"linear_attn.q_proj.weight", K::KdaQ},
        {"linear_attn.k_proj.weight", K::KdaK},
        {"linear_attn.v_proj.weight", K::KdaV},
        {"linear_attn.o_proj.weight", K::KdaO},
        {"linear_attn.f_a_proj.weight", K::KdaFA},
        {"linear_attn.f_b_proj.weight", K::KdaFB},
        {"linear_attn.g_a_proj.weight", K::KdaGA},
        {"linear_attn.g_b_proj.weight", K::KdaGB},
        {"linear_attn.b_proj.weight", K::KdaB},
        {"linear_attn.q_conv1d.weight", K::KdaQConv},
        {"linear_attn.k_conv1d.weight", K::KdaKConv},
        {"linear_attn.v_conv1d.weight", K::KdaVConv},
        {"linear_attn.a_log_bias", K::KdaALog},
        {"linear_attn.dt_bias", K::KdaDt},
        {"linear_attn.o_norm.weight", K::KdaONorm},
        {"self_attn.q_proj.weight", K::FaQ},
        {"self_attn.k_proj.weight", K::FaK},
        {"self_attn.v_proj.weight", K::FaV},
        {"self_attn.o_proj.weight", K::FaO},
        {"self_attn.q_norm.weight", K::FaQNorm},
        {"self_attn.k_norm.weight", K::FaKNorm},
        {"mlp.gate.weight", K::RouterGate},
        {"mlp.gate.e_score_correction_bias", K::RouterBias},
        {"mlp.shared_expert.gate_proj.weight", K::SharedGateProj},
        {"mlp.shared_expert.up_proj.weight", K::SharedUpProj},
        {"mlp.shared_expert.down_proj.weight", K::SharedDownProj},
        {"mlp.shared_expert_gate.weight", K::SharedGate},
        {"mlp.experts.gate_up_proj", K::FusedExpertsGateUp},
        {"mlp.experts.down_proj", K::FusedExpertsDown},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool alice_tensor_is_nvfp4(AliceTensorKind kind)
{
    switch (kind) {
    case AliceTensorKind::KdaQ:
    case AliceTensorKind::KdaK:
    case AliceTensorKind::KdaV:
    case AliceTensorKind::KdaO:
    case AliceTensorKind::KdaFB:
    case AliceTensorKind::KdaGB:
    case AliceTensorKind::FaQ:
    case AliceTensorKind::FaK:
    case AliceTensorKind::FaV:
    case AliceTensorKind::FaO:
    case AliceTensorKind::SharedGateProj:
    case AliceTensorKind::SharedUpProj:
    case AliceTensorKind::SharedDownProj:
        return true;
    default:
        return false;
    }
}

AliceTopology ParseAliceConfig(const std::string &input_dir)
{
    AliceTopology top;
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
    top.num_layers = get_i(root, "num_hidden_layers", 0);
    top.vocab_size = get_i(root, "vocab_size", 0);
    top.num_attention_heads = get_i(root, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(root, "num_key_value_heads", 0);
    top.head_dim = get_i(root, "head_dim", 0);
    top.linear_num_key_heads = get_i(root, "linear_num_key_heads", 0);
    top.linear_num_value_heads = get_i(root, "linear_num_value_heads", 0);
    top.linear_key_head_dim = get_i(root, "linear_key_head_dim", 0);
    top.linear_value_head_dim = get_i(root, "linear_value_head_dim", 0);
    top.linear_conv_kernel_dim = get_i(root, "linear_conv_kernel_dim", 4);
    top.num_experts = get_i(root, "num_experts", 0);
    top.num_experts_per_tok = get_i(root, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(root, "moe_intermediate_size", 0);
    top.shared_expert_intermediate_size =
        get_i(root, "shared_expert_intermediate_size", 0);
    top.block_attn_res_block_size = get_i(root, "block_attn_res_block_size", 4);
    top.mtp_num_hidden_layers = get_i(root, "mtp_num_hidden_layers", 0);
    top.rms_norm_eps = get_d(root, "rms_norm_eps", 1e-6);
    top.partial_rotary_factor = get_d(root, "partial_rotary_factor", 0.25);
    top.rope_theta = get_d(root, "rope_theta", 1e6);
    top.bos_token_id = get_i(root, "bos_token_id", 1);
    top.eos_token_id = get_i(root, "eos_token_id", 2);
    top.pad_token_id = get_i(root, "pad_token_id", -1);

    if (auto lt = root["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                top.layer_types.emplace_back(sv);
        }
    }
    if (top.layer_types.empty() && top.num_layers > 0) {
        for (int i = 0; i < top.num_layers; ++i)
            top.layer_types.emplace_back(
                (i % 4 == 3) ? "full_attention" : "linear_attention");
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.layer_types.empty()) {
        top.error_msg = "config.json missing required fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layer_types size ", top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    return top;
}

arrow::Status ConvertAliceModel(const std::string &input_dir,
                                const std::string &output_root)
{
    namespace fs = std::filesystem;

    AliceTopology top = ParseAliceConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Alice config error: ", top.error_msg);

    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid("Alice frozen geometry mismatch: ", why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_alice_tensor(t.name);
            if (p.kind == AliceTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in alice checkpoint: ", t.name);
        }
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path experts_dir = base / "experts";
    const fs::path mtp_dir = base / "mtp";
    for (const fs::path &d : {base, pinned_dir, experts_dir, mtp_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<TensorRecord> global_raw;
    std::vector<DenseNvfp4Record> mtp_nvfp4;
    std::vector<TensorRecord> mtp_raw;

    int64_t total_pinned = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_alice_tensor(t.name);
            auto data = read_tensor_data(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            if (p.kind == AliceTensorKind::Mtp) {
                if (ends_with(t.name, "mlp.experts.gate_up_proj") ||
                    ends_with(t.name, "mlp.experts.down_proj"))
                    continue;  // dedicated expert pass
                const bool raw = t.name.find("norm") != std::string::npos ||
                                 ends_with(t.name, "mlp.gate.weight") ||
                                 ends_with(t.name, "e_score_correction_bias") ||
                                 ends_with(t.name, "shared_expert_gate.weight");
                if (raw) {
                    TensorRecord r;
                    r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                    r.data = std::move(data);
                    mtp_raw.push_back(std::move(r));
                } else {
                    DenseNvfp4Record r;
                    r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                    if (!quantize_slice(data.data(), t.numel(), r))
                        return arrow::Status::IOError(
                            "GPU quantization failed for ", t.name);
                    mtp_nvfp4.push_back(std::move(r));
                }
                ++total_pinned;
                continue;
            }

            switch (p.kind) {
            case AliceTensorKind::Embed:
            case AliceTensorKind::LmHead:
            case AliceTensorKind::FinalNorm:
            case AliceTensorKind::AttnResFinalProj:
            case AliceTensorKind::AttnResFinalNorm: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case AliceTensorKind::InputNorm:
            case AliceTensorKind::PostNorm:
            case AliceTensorKind::MlpResProj:
            case AliceTensorKind::MlpResNorm:
            case AliceTensorKind::AttnResProj:
            case AliceTensorKind::AttnResNorm:
            case AliceTensorKind::KdaFA:
            case AliceTensorKind::KdaGA:
            case AliceTensorKind::KdaB:
            case AliceTensorKind::KdaQConv:
            case AliceTensorKind::KdaKConv:
            case AliceTensorKind::KdaVConv:
            case AliceTensorKind::KdaALog:
            case AliceTensorKind::KdaDt:
            case AliceTensorKind::KdaONorm:
            case AliceTensorKind::FaQNorm:
            case AliceTensorKind::FaKNorm:
            case AliceTensorKind::RouterGate:
            case AliceTensorKind::RouterBias:
            case AliceTensorKind::SharedGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case AliceTensorKind::KdaQ:
            case AliceTensorKind::KdaK:
            case AliceTensorKind::KdaV:
            case AliceTensorKind::KdaO:
            case AliceTensorKind::KdaFB:
            case AliceTensorKind::KdaGB:
            case AliceTensorKind::FaQ:
            case AliceTensorKind::FaK:
            case AliceTensorKind::FaV:
            case AliceTensorKind::FaO:
            case AliceTensorKind::SharedGateProj:
            case AliceTensorKind::SharedUpProj:
            case AliceTensorKind::SharedDownProj: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!quantize_slice(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case AliceTensorKind::FusedExpertsGateUp:
            case AliceTensorKind::FusedExpertsDown:
                continue;  // dedicated expert pass
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    // Fused-expert split. gate_up_proj[e] is [2I,H]: gate = rows [0,I), up =
    // rows [I,2I). down_proj[e] is [H,I]. MTP experts use a synthetic layer key
    // equal to num_layers and are written under mtp/experts.
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    const int64_t row_bytes = static_cast<int64_t>(H) * 2;  // BF16 row
    const int kMtpLayerKey = top.num_layers;
    using ExpertKey = std::pair<int, int>;
    absl::btree_map<ExpertKey, std::pair<DenseNvfp4Record, DenseNvfp4Record>>
        gate_up_w1_w3;
    absl::btree_map<ExpertKey, DenseNvfp4Record> down_w2;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            const bool gu = ends_with(t.name, "mlp.experts.gate_up_proj");
            const bool dn = ends_with(t.name, "mlp.experts.down_proj");
            if (!gu && !dn) continue;
            const bool is_mtp = t.name.rfind("mtp.", 0) == 0;
            const int layer = is_mtp ? kMtpLayerKey
                                     : classify_alice_tensor(t.name).layer;
            auto data = read_tensor_data(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);
            if (t.shape.size() != 3 || t.shape[0] != E)
                return arrow::Status::Invalid("bad fused-expert shape: ",
                                              t.name);
            for (int e = 0; e < E; ++e) {
                if (gu) {
                    if (t.shape[1] != 2 * I)
                        return arrow::Status::Invalid(
                            "gate_up_proj dim1 != 2*moe_intermediate: ", t.name);
                    const int64_t g_off =
                        static_cast<int64_t>(e) * 2 * I * row_bytes;
                    DenseNvfp4Record w1r, w3r;
                    w1r.name = absl::StrCat(t.name, ".w1.e", e);
                    w1r.shape = {I, H}; w1r.dtype = t.dtype;
                    w3r.name = absl::StrCat(t.name, ".w3.e", e);
                    w3r.shape = {I, H}; w3r.dtype = t.dtype;
                    if (!quantize_slice(data.data() + g_off,
                                        static_cast<int64_t>(I) * H, w1r))
                        return arrow::Status::IOError("quantize failed w1 ",
                                                      t.name, " e", e);
                    if (!quantize_slice(data.data() + g_off + I * row_bytes,
                                        static_cast<int64_t>(I) * H, w3r))
                        return arrow::Status::IOError("quantize failed w3 ",
                                                      t.name, " e", e);
                    gate_up_w1_w3.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(layer, e),
                        std::forward_as_tuple(std::move(w1r), std::move(w3r)));
                } else {
                    if (t.shape[1] != H || t.shape[2] != I)
                        return arrow::Status::Invalid(
                            "down_proj shape != [E,H,I]: ", t.name);
                    const int64_t d_off = static_cast<int64_t>(e) * H * I * 2;
                    DenseNvfp4Record w2r;
                    w2r.name = absl::StrCat(t.name, ".w2.e", e);
                    w2r.shape = {H, I}; w2r.dtype = t.dtype;
                    if (!quantize_slice(data.data() + d_off,
                                        static_cast<int64_t>(H) * I, w2r))
                        return arrow::Status::IOError("quantize failed w2 ",
                                                      t.name, " e", e);
                    down_w2.insert_or_assign(std::make_pair(layer, e),
                                             std::move(w2r));
                }
            }
        }
    }

    auto write_layer_experts = [&](int l, const fs::path &dir) -> arrow::Status {
        for (int e = 0; e < E; ++e) {
            auto gu = gate_up_w1_w3.find({l, e});
            auto dn = down_w2.find({l, e});
            if (gu == gate_up_w1_w3.end() || dn == down_w2.end())
                return arrow::Status::Invalid("missing fused-expert half: layer ",
                                              l, " expert ", e);
            PackedExpert pe;
            pe.w1_packed = gu->second.first.packed;
            pe.w1_scales = gu->second.first.scales;
            pe.w1_shape = gu->second.first.shape;
            pe.w1_global_scale = gu->second.first.global_scale;
            pe.w3_packed = gu->second.second.packed;
            pe.w3_scales = gu->second.second.scales;
            pe.w3_shape = gu->second.second.shape;
            pe.w3_global_scale = gu->second.second.global_scale;
            pe.w2_packed = dn->second.packed;
            pe.w2_scales = dn->second.scales;
            pe.w2_shape = dn->second.shape;
            pe.w2_global_scale = dn->second.global_scale;
            ARROW_RETURN_NOT_OK(write_expert_file(dir, e, std::move(pe)));
        }
        return arrow::Status::OK();
    };
    for (int l = 0; l < top.num_layers; ++l)
        ARROW_RETURN_NOT_OK(write_layer_experts(
            l, experts_dir / absl::StrCat("layer_", l)));
    if (top.mtp_num_hidden_layers > 0)
        ARROW_RETURN_NOT_OK(write_layer_experts(kMtpLayerKey,
                                                mtp_dir / "experts"));

    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(write_batch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(write_batch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    if (!mtp_nvfp4.empty())
        ARROW_RETURN_NOT_OK(write_batch(DenseNvfp4Schema::MakeBatch(mtp_nvfp4),
                                        mtp_dir / "nvfp4.arrow"));
    if (!mtp_raw.empty())
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(mtp_raw),
                                        mtp_dir / "raw.arrow"));

    total_pinned += static_cast<int64_t>(E) * 3;  // expert tensors
    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Tokenizer: mandatory for real checkpoints; synthetic fixtures omit it.
    const fs::path spm_path = fs::path(input_dir) / "tokenizer.model";
    bool tokenizer_ok = false;
    if (fs::exists(spm_path)) {
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizerSentencePiece(
            spm_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = true,
                                           .eos_required = true,
                                           .pad_required = false}));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.model not found in ", input_dir);
    }

    // VRAM profile.
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.num_moe_layers = top.num_layers;
    vg.experts_per_layer = top.num_experts;
    vg.context_length = ReadContextLength(
        (fs::path(input_dir) / "config.json").string());

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
    VramProfile vram = EstimateVram(
        vram_inventory,
        {+[](const std::string &n) {
             return alice_tensor_is_nvfp4(classify_alice_tensor(n).kind);
         },
         nullptr},
        vg);

    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"logestix-alice-nvfp4\",\n"
          << "  \"model\": \"AliceAI-Foundation-80B-A3B-Base\",\n"
          << "  \"num_layers\": " << top.num_layers
          << ",\n  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "]"
          << ",\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"linear_num_key_heads\": " << top.linear_num_key_heads
          << ",\n  \"linear_num_value_heads\": " << top.linear_num_value_heads
          << ",\n  \"linear_key_head_dim\": " << top.linear_key_head_dim
          << ",\n  \"linear_value_head_dim\": " << top.linear_value_head_dim
          << ",\n  \"linear_conv_kernel_dim\": " << top.linear_conv_kernel_dim
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"partial_rotary_factor\": " << top.partial_rotary_factor
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"shared_expert_intermediate_size\": "
          << top.shared_expert_intermediate_size
          << ",\n  \"block_attn_res_block_size\": "
          << top.block_attn_res_block_size
          << ",\n  \"mtp_num_hidden_layers\": " << top.mtp_num_hidden_layers
          << ",\n  \"bos_id\": " << top.bos_token_id
          << ",\n  \"eos_id\": " << top.eos_token_id
          << ",\n  \"pad_id\": " << top.pad_token_id
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "spm" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n" << FormatVramJson(vram)
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Every produced arrow file must re-open cleanly.
    for (const fs::path &dir : {pinned_dir, experts_dir, mtp_dir}) {
        std::error_code ec;
        for (fs::recursive_directory_iterator it(dir, ec), end;
             !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file()) continue;
            auto res = ReadRecordBatchFromFile(it->path());
            if (!res.ok())
                return arrow::Status::IOError("unreadable output arrow: ",
                                              it->path().string());
        }
        if (ec)
            return arrow::Status::IOError("cannot list ", dir.string());
    }

    lsxcommon::log::info(absl::StrFormat(
        "Alice conversion: %d layers x %d experts, %d pinned tensors",
        top.num_layers, top.num_experts, top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::alice
