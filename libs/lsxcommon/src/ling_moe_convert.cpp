// Ling-3.1-flash (BailingMoeV3) converter: config parse, tensor classifier,
// FP8 block-128 -> NVFP4 requant and the conversion pipeline.
#include "ling_moe_constants.h"

#include "lsxcommon/ling_moe_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "tensor_index_parse.h"

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <arrow/api.h>
#include <simdjson.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace lsxcommon::ling {

namespace {

using frozen::kType;

int ProveInt(simdjson::ondemand::object &o, const char *k, int dflt)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

double ProveDouble(simdjson::ondemand::object &o, const char *k, double dflt)
{
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return dflt;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
        return d.value();
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
        return static_cast<double>(i.value());
    return dflt;
}

std::string ProveString(simdjson::ondemand::object &o, const char *k,
                        std::string dflt)
{
    std::string_view s;
    if (o[k].get_string().get(s) != simdjson::SUCCESS) return dflt;
    return std::string(s);
}

bool ProveBool(simdjson::ondemand::object &o, const char *k, bool dflt)
{
    bool v;
    if (o[k].get_bool().get(v) != simdjson::SUCCESS) return dflt;
    return v;
}

int ProjId(const std::string &suffix)
{
    if (suffix.rfind("gate_proj", 0) == 0) return 1;
    if (suffix.rfind("down_proj", 0) == 0) return 2;
    if (suffix.rfind("up_proj", 0) == 0) return 3;
    return 0;
}

bool SplitLayer(const std::string &name, int &layer, std::string &rest)
{
    static const char kPrefix[] = "model.layers.";
    if (name.rfind(kPrefix, 0) != 0) return false;
    const size_t ds = sizeof(kPrefix) - 1;
    const size_t de = name.find('.', ds);
    if (de == std::string::npos) return false;
    if (!detail::ParseTensorIndex(
            std::string_view(name).substr(ds, de - ds), layer))
        return false;
    rest = name.substr(de + 1);
    return true;
}

bool SplitExpert(const std::string &rest, int &expert, std::string &suffix)
{
    static const char kKey[] = "mlp.experts.";
    if (rest.rfind(kKey, 0) != 0) return false;
    const size_t ds = sizeof(kKey) - 1;
    const size_t de = rest.find('.', ds);
    if (de == std::string::npos) return false;
    if (!detail::ParseTensorIndex(
            std::string_view(rest).substr(ds, de - ds), expert))
        return false;
    suffix = rest.substr(de + 1);
    return true;
}

} // namespace

ParsedLingTensor classify_ling_tensor(const std::string &name)
{
    ParsedLingTensor out;
    if (name == "lm_head.weight") {
        out.kind = LingTensorKind::LmHead;
        return out;
    }
    if (name == "model.word_embeddings.weight") {
        out.kind = LingTensorKind::Embed;
        return out;
    }
    if (name == "model.norm.weight") {
        out.kind = LingTensorKind::FinalNorm;
        return out;
    }
    if (name.rfind("model.visual.", 0) == 0 || name.rfind("visual.", 0) == 0) {
        out.kind = LingTensorKind::Visual;
        return out;
    }
    if (name.size() > 10 &&
        name.compare(name.size() - 10, 10, "_scale_inv") == 0) {
        return out;  // FP8 scale records are consumed with their weight
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayer(name, layer, rest)) return out;
    out.layer = layer;
    if (layer >= frozen::kLayers) {
        out.kind = LingTensorKind::Mtp;
        return out;
    }

    if (rest == "input_layernorm.weight") {
        out.kind = LingTensorKind::InputNorm;
        return out;
    }
    if (rest == "post_attention_layernorm.weight") {
        out.kind = LingTensorKind::PostNorm;
        return out;
    }
    if (rest == "mlp.gate.weight") {
        out.kind = LingTensorKind::RouterGate;
        return out;
    }
    if (rest == "mlp.gate.expert_bias") {
        out.kind = LingTensorKind::RouterBias;
        return out;
    }

    int expert = -1;
    std::string suffix;
    if (SplitExpert(rest, expert, suffix)) {
        out.expert = expert;
        out.proj_id = ProjId(suffix);
        switch (out.proj_id) {
        case 1: out.kind = LingTensorKind::RoutedGate; break;
        case 2: out.kind = LingTensorKind::RoutedDown; break;
        case 3: out.kind = LingTensorKind::RoutedUp; break;
        default: break;
        }
        return out;
    }
    if (rest.rfind("mlp.shared_experts.", 0) == 0) {
        out.proj_id =
            ProjId(rest.substr(std::strlen("mlp.shared_experts.")));
        switch (out.proj_id) {
        case 1: out.kind = LingTensorKind::SharedGate; break;
        case 2: out.kind = LingTensorKind::SharedDown; break;
        case 3: out.kind = LingTensorKind::SharedUp; break;
        default: break;
        }
        return out;
    }
    if (rest.rfind("mlp.gate_proj", 0) == 0 ||
        rest.rfind("mlp.up_proj", 0) == 0 ||
        rest.rfind("mlp.down_proj", 0) == 0) {
        out.proj_id = ProjId(rest.substr(std::strlen("mlp.")));
        switch (out.proj_id) {
        case 1: out.kind = LingTensorKind::DenseMlpGate; break;
        case 2: out.kind = LingTensorKind::DenseMlpDown; break;
        case 3: out.kind = LingTensorKind::DenseMlpUp; break;
        default: break;
        }
        return out;
    }

    if (frozen::IsMlaLayer(layer)) {
        static const std::array<std::pair<const char *, LingTensorKind>, 10>
            kMla = {{
                {"attention.q_a_proj.weight", LingTensorKind::MlaQa},
                {"attention.q_b_proj.weight", LingTensorKind::MlaQb},
                {"attention.kv_a_proj_with_mqa.weight",
                 LingTensorKind::MlaKvA},
                {"attention.kv_b_proj.weight", LingTensorKind::MlaKvB},
                {"attention.dense.weight", LingTensorKind::MlaDense},
                {"attention.q_a_layernorm.weight", LingTensorKind::MlaNorm},
                {"attention.kv_a_layernorm.weight", LingTensorKind::MlaNorm},
                {"attention.g_proj.weight", LingTensorKind::MlaG},
                {"input_layernorm.weight", LingTensorKind::InputNorm},
                {"post_attention_layernorm.weight",
                 LingTensorKind::PostNorm},
            }};
        for (const auto &kv : kMla)
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        return out;
    }

    if (frozen::IsKdaLayer(layer)) {
        static const std::array<std::pair<const char *, LingTensorKind>, 13>
            kKda = {{
                {"attention.q_proj.weight", LingTensorKind::KdaQ},
                {"attention.k_proj.weight", LingTensorKind::KdaK},
                {"attention.v_proj.weight", LingTensorKind::KdaV},
                {"attention.o_proj.weight", LingTensorKind::KdaO},
                {"attention.f_proj.weight", LingTensorKind::KdaF},
                {"attention.g_proj.weight", LingTensorKind::KdaG},
                {"attention.q_conv1d.weight", LingTensorKind::KdaConv},
                {"attention.k_conv1d.weight", LingTensorKind::KdaConv},
                {"attention.v_conv1d.weight", LingTensorKind::KdaConv},
                {"attention.A_log", LingTensorKind::KdaALog},
                {"attention.dt_bias", LingTensorKind::KdaDtBias},
                {"attention.b_proj.weight", LingTensorKind::KdaB},
                {"attention.o_norm.weight", LingTensorKind::KdaONorm},
            }};
        for (const auto &kv : kKda)
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        return out;
    }
    return out;
}

bool ling_tensor_is_nvfp4(LingTensorKind kind)
{
    switch (kind) {
    case LingTensorKind::KdaQ:
    case LingTensorKind::KdaK:
    case LingTensorKind::KdaV:
    case LingTensorKind::KdaO:
    case LingTensorKind::KdaF:
    case LingTensorKind::KdaG:
    case LingTensorKind::MlaQa:
    case LingTensorKind::MlaQb:
    case LingTensorKind::MlaKvA:
    case LingTensorKind::MlaKvB:
    case LingTensorKind::MlaDense:
    case LingTensorKind::DenseMlpGate:
    case LingTensorKind::DenseMlpUp:
    case LingTensorKind::DenseMlpDown:
    case LingTensorKind::SharedGate:
    case LingTensorKind::SharedUp:
    case LingTensorKind::SharedDown:
    case LingTensorKind::RoutedGate:
    case LingTensorKind::RoutedDown:
    case LingTensorKind::RoutedUp:
        return true;
    default:
        return false;
    }
}

bool ling_tensor_is_routed_expert(LingTensorKind kind)
{
    return kind == LingTensorKind::RoutedGate ||
           kind == LingTensorKind::RoutedDown ||
           kind == LingTensorKind::RoutedUp;
}

LingTopology ParseLingConfig(const std::string &config_json_path)
{
    LingTopology top;
    std::error_code ec;
    if (!fs::exists(config_json_path, ec)) {
        top.error_msg = "config.json not found: " + config_json_path;
        return top;
    }
    std::ifstream f(config_json_path, std::ios::binary);
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

    top.hidden = ProveInt(root, "hidden_size", 0);
    top.layers = ProveInt(root, "num_hidden_layers", 0);
    top.vocab = ProveInt(root, "vocab_size", 0);
    top.heads = ProveInt(root, "num_attention_heads", 0);
    top.kv_heads = ProveInt(root, "num_key_value_heads", 0);
    top.head_dim = ProveInt(root, "head_dim", 0);
    top.qk_head_dim = ProveInt(root, "qk_head_dim", 0);
    top.qk_nope = ProveInt(root, "qk_nope_head_dim", 0);
    top.qk_rope = ProveInt(root, "qk_rope_head_dim", 0);
    top.v_head = ProveInt(root, "v_head_dim", 0);
    top.q_lora = ProveInt(root, "q_lora_rank", 0);
    top.kv_lora = ProveInt(root, "kv_lora_rank", 0);
    top.conv_kernel = ProveInt(root, "short_conv_kernel_size", 0);
    top.num_experts = ProveInt(root, "num_experts", 0);
    top.top_k = ProveInt(root, "num_experts_per_tok", 0);
    top.n_group = ProveInt(root, "n_group", 0);
    top.topk_group = ProveInt(root, "topk_group", 0);
    top.moe_intermediate = ProveInt(root, "moe_intermediate_size", 0);
    top.shared_intermediate =
        ProveInt(root, "moe_shared_expert_intermediate_size", 0);
    top.dense_intermediate = ProveInt(root, "intermediate_size", 0);
    top.first_k_dense = ProveInt(root, "first_k_dense_replace", 0);
    top.layer_group_size = ProveInt(root, "layer_group_size", 0);
    top.rms_eps = ProveDouble(root, "rms_norm_eps", 0.0);
    top.rope_theta = ProveDouble(root, "rope_theta", 0.0);
    top.routed_scale = ProveDouble(root, "routed_scaling_factor", 0.0);
    top.pad_id = ProveInt(root, "pad_token_id", -1);
    auto read_farray = [&](const char *key, std::vector<float> &out) {
        if (auto arr = root[key].get_array(); arr.error() == simdjson::SUCCESS) {
            for (auto v : arr.value()) {
                double d = 0.0;
                if (v.get_double().get(d) != simdjson::SUCCESS) {
                    int64_t i = 0;
                    if (v.get_int64().get(i) == simdjson::SUCCESS)
                        d = static_cast<double>(i);
                }
                out.push_back(static_cast<float>(d));
            }
        }
    };
    read_farray("expert_swiglu_limit_list", top.expert_swiglu_limit);
    read_farray("share_expert_swiglu_limit_list", top.shared_swiglu_limit);

    if (auto arr = root["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t id = 0;
            if (v.get_int64().get(id) == simdjson::SUCCESS && top.eos_id < 0)
                top.eos_id = static_cast<int>(id);
        }
    } else {
        top.eos_id = ProveInt(root, "eos_token_id", -1);
    }

    const std::string act = ProveString(root, "hidden_act", "silu");
    const std::string scoring = ProveString(root, "scoring_func", "sigmoid");
    const std::string topk_method = ProveString(root, "topk_method", "");
    const std::string granularity =
        ProveString(root, "gated_attention_proj_granularity_type", "");
    const bool norm_topk = ProveBool(root, "norm_topk_prob", false);
    const bool tie = ProveBool(root, "tie_word_embeddings", true);
    const bool use_bias = ProveBool(root, "use_bias", true);
    const bool qkv_bias = ProveBool(root, "use_qkv_bias", true);
    const bool kda_lora = ProveBool(root, "no_kda_lora", false);
    const bool safe_gate = ProveBool(root, "kda_safe_gate", false);
    const bool linear_silu = ProveBool(root, "linear_silu", false);
    const bool enable_bias = ProveBool(root, "moe_router_enable_expert_bias",
                                       false);
    const bool interleave = ProveBool(root, "rope_interleave", false);
    const double partial = ProveDouble(root, "partial_rotary_factor", 0.0);

    if (act != "silu")
        top.error_msg = "hidden_act must be silu (got " + act + ")";
    else if (scoring != "sigmoid")
        top.error_msg = "scoring_func must be sigmoid (got " + scoring + ")";
    else if (topk_method != "noaux_tc")
        top.error_msg = "topk_method must be noaux_tc (got " + topk_method + ")";
    else if (granularity != "head_wise")
        top.error_msg =
            "gated_attention_proj_granularity_type must be head_wise (got " +
            granularity + ")";
    else if (!norm_topk)
        top.error_msg = "norm_topk_prob must be true";
    else if (tie)
        top.error_msg = "tie_word_embeddings must be false";
    else if (use_bias || qkv_bias)
        top.error_msg = "use_bias/use_qkv_bias must be false";
    else if (!kda_lora)
        top.error_msg = "no_kda_lora must be true (direct f/g projections)";
    else if (!safe_gate)
        top.error_msg = "kda_safe_gate must be true";
    else if (!linear_silu)
        top.error_msg = "linear_silu must be true";
    else if (!enable_bias)
        top.error_msg = "moe_router_enable_expert_bias must be true";
    else if (!interleave)
        top.error_msg = "rope_interleave must be true";
    else if (partial != 0.5)
        top.error_msg = "partial_rotary_factor must be 0.5";
    else {
        simdjson::ondemand::value rs;
        if (root["rope_scaling"].get(rs) == simdjson::SUCCESS) {
            bool is_null = false;
            if (rs.is_null().get(is_null) != simdjson::SUCCESS || !is_null)
                top.error_msg = "rope_scaling must be null";
        }
    }

    if (!top.error_msg.empty()) return top;
    if (top.hidden <= 0 || top.layers <= 0 || top.vocab <= 0 ||
        top.heads <= 0 || top.kv_heads <= 0 || top.head_dim <= 0 ||
        top.qk_head_dim <= 0 || top.qk_nope <= 0 || top.qk_rope <= 0 ||
        top.v_head <= 0 ||
        top.q_lora <= 0 || top.kv_lora <= 0 || top.conv_kernel <= 0 ||
        top.num_experts <= 0 || top.top_k <= 0 || top.n_group <= 0 ||
        top.topk_group <= 0 || top.moe_intermediate <= 0 ||
        top.shared_intermediate <= 0 || top.dense_intermediate <= 0 ||
        top.first_k_dense <= 0 || top.layer_group_size <= 0) {
        top.error_msg = "config.json missing required Ling fields";
        return top;
    }
    if (top.kv_heads != top.heads) {
        top.error_msg = "num_key_value_heads must equal num_attention_heads";
        return top;
    }
    if (top.qk_nope + top.qk_rope != top.qk_head_dim) {
        top.error_msg = "qk_nope + qk_rope != qk_head_dim";
        return top;
    }
    if (top.top_k > top.num_experts ||
        top.num_experts % top.n_group != 0 ||
        top.topk_group > top.n_group) {
        top.error_msg = "invalid group-limited top-k geometry";
        return top;
    }
    if (top.layers % top.layer_group_size != 0) {
        top.error_msg = "layer_group_size must divide num_hidden_layers";
        return top;
    }
    if (top.first_k_dense > top.layers) {
        top.error_msg = "first_k_dense_replace > num_hidden_layers";
        return top;
    }

    for (int l = 0; l < top.layers; ++l) {
        const bool mla = (l + 1) % top.layer_group_size == 0;
        (mla ? top.mla_layers : top.kda_layers).push_back(l);
        (l < top.first_k_dense ? top.dense_layers : top.moe_layers)
            .push_back(l);
    }
    return top;
}

} // namespace lsxcommon::ling

namespace lsxcommon::ling {

namespace {

std::vector<uint8_t> ReadTensorData(const std::string &input_dir,
                                    const SafetensorsModel::Shard &shard,
                                    const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((fs::path(input_dir) / shard.filename).string(),
                     std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    const int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    if (!sf || static_cast<int64_t>(sf.gcount()) != info.byte_size())
        return {};
    return buf;
}

arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const fs::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

arrow::Status WriteExpertFile(const fs::path &base_dir, int layer, int expert,
                              PackedExpert &&pe)
{
    fs::path dir = base_dir / "experts" / absl::StrCat("layer_", layer);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(
        *batch,
        dir / absl::StrCat("exp_", expert, ".arrow"));
}

struct TensorLoc {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

arrow::Status QuantizeBf16(const std::vector<uint8_t> &data, int64_t numel,
                           const std::vector<int64_t> &shape,
                           DenseNvfp4Record &r)
{
    if (!gpu::quantize_tensor_gpu(data.data(), numel, "BF16", nullptr, shape,
                                  {}, r.packed, r.scales, r.global_scale))
        return arrow::Status::IOError("GPU quantization failed for ", r.name);
    r.dtype = "BF16";
    return arrow::Status::OK();
}

// FP8 (block-128 scale_inv) -> BF16 -> NVFP4.
arrow::Status QuantizeFp8(const std::vector<uint8_t> &fp8,
                          const std::vector<int64_t> &shape,
                          const std::vector<uint8_t> &scale,
                          const std::vector<int64_t> &scale_shape,
                          DenseNvfp4Record &r)
{
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(
        LingDequantFp8Block128ToBf16Gpu(fp8, shape, scale, scale_shape, bf16));
    return QuantizeBf16(bf16, shape[0] * shape[1], shape, r);
}

} // namespace

arrow::Status ConvertLingModel(const std::string &input_dir,
                               const std::string &output_root)
{
    LingTopology top = ParseLingConfig(
        (fs::path(input_dir) / "config.json").string());
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("ling config error: ", top.error_msg);

    const bool synth = SyntheticGeometryAllowed();
    if (!synth) {
        using namespace frozen;
        auto bad = [](const char *field, int got, int want) {
            return arrow::Status::Invalid("ling frozen geometry mismatch: ",
                                          field, " got ", got, " want ", want);
        };
        if (top.hidden != kHidden) return bad("hidden_size", top.hidden, kHidden);
        if (top.layers != kLayers) return bad("num_hidden_layers", top.layers, kLayers);
        if (top.vocab != kVocab) return bad("vocab_size", top.vocab, kVocab);
        if (top.heads != kHeads) return bad("num_attention_heads", top.heads, kHeads);
        if (top.kv_heads != kKvHeads) return bad("num_key_value_heads", top.kv_heads, kKvHeads);
        if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
        if (top.qk_head_dim != kQkHeadDim) return bad("qk_head_dim", top.qk_head_dim, kQkHeadDim);
        if (top.qk_nope != kQkNope) return bad("qk_nope_head_dim", top.qk_nope, kQkNope);
        if (top.qk_rope != kQkRope) return bad("qk_rope_head_dim", top.qk_rope, kQkRope);
        if (top.v_head != kVHead) return bad("v_head_dim", top.v_head, kVHead);
        if (top.q_lora != kQLora) return bad("q_lora_rank", top.q_lora, kQLora);
        if (top.kv_lora != kKvLora) return bad("kv_lora_rank", top.kv_lora, kKvLora);
        if (top.conv_kernel != kConvKernel) return bad("short_conv_kernel_size", top.conv_kernel, kConvKernel);
        if (top.num_experts != kNumExperts) return bad("num_experts", top.num_experts, kNumExperts);
        if (top.top_k != kTopK) return bad("num_experts_per_tok", top.top_k, kTopK);
        if (top.n_group != kGroupCount) return bad("n_group", top.n_group, kGroupCount);
        if (top.topk_group != kTopkGroup) return bad("topk_group", top.topk_group, kTopkGroup);
        if (top.moe_intermediate != kMoeIntermediate) return bad("moe_intermediate_size", top.moe_intermediate, kMoeIntermediate);
        if (top.shared_intermediate != kSharedIntermediate) return bad("moe_shared_expert_intermediate_size", top.shared_intermediate, kSharedIntermediate);
        if (top.dense_intermediate != kDenseIntermediate) return bad("intermediate_size", top.dense_intermediate, kDenseIntermediate);
        if (top.first_k_dense != kFirstKDense) return bad("first_k_dense_replace", top.first_k_dense, kFirstKDense);
        if (top.layer_group_size != kLayerGroupSize) return bad("layer_group_size", top.layer_group_size, kLayerGroupSize);
        if (static_cast<int>(top.mla_layers.size()) != kMlaLayerCount)
            return arrow::Status::Invalid("ling MLA layer table mismatch");
        for (int i = 0; i < kMlaLayerCount; ++i)
            if (top.mla_layers[static_cast<size_t>(i)] != kMlaLayerIds[i])
                return arrow::Status::Invalid(
                    "ling MLA layer table mismatch at index ", i);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    absl::flat_hash_set<int> dense_layers(top.dense_layers.begin(),
                                          top.dense_layers.end());
    auto is_dense = [&](int l) { return dense_layers.contains(l); };

    absl::btree_map<std::string, TensorLoc> tensors;
    absl::btree_map<std::string, TensorLoc> expert_tensors;
    absl::btree_map<std::string, TensorLoc> scale_tensors;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            if (t.name.size() > 17 &&
                t.name.compare(t.name.size() - 17, 17, ".weight_scale_inv") == 0) {
                scale_tensors.emplace(t.name, TensorLoc{&shard, &t});
                continue;  // consumed with its FP8 weight
            }
            auto p = classify_ling_tensor(t.name);
            if (p.layer >= top.layers || p.kind == LingTensorKind::Mtp ||
                p.kind == LingTensorKind::Visual) {
                tensors.emplace(t.name, TensorLoc{&shard, &t});
                continue;
            }
            if (p.kind == LingTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in ling checkpoint: ", t.name);
            if (ling_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.expert < 0 || p.expert >= top.num_experts)
                    return arrow::Status::Invalid("bad expert tensor: ", t.name);
                if (is_dense(p.layer))
                    return arrow::Status::Invalid(
                        "expert tensor on a dense-MLP layer: ", t.name);
                expert_tensors.emplace(t.name, TensorLoc{&shard, &t});
            } else {
                tensors.emplace(t.name, TensorLoc{&shard, &t});
            }
        }
    }

    const int H = top.hidden;
    const int I = top.moe_intermediate;
    const int moe_layers = static_cast<int>(top.moe_layers.size());
    const int64_t experts_total =
        static_cast<int64_t>(moe_layers) * top.num_experts;
    if (static_cast<int64_t>(expert_tensors.size()) != experts_total * 3)
        return arrow::Status::Invalid(
            "routed-expert inventory size ", expert_tensors.size(), " != ",
            experts_total * 3);

    {
        SafetensorsModel resident_only;
        for (const auto &shard : model.shards) {
            SafetensorsModel::Shard s;
            s.filename = shard.filename;
            s.file_size = shard.file_size;
            for (const auto &t : shard.tensors) {
                if (t.name.size() > 17 &&
                    t.name.compare(t.name.size() - 17, 17,
                                   ".weight_scale_inv") == 0)
                    continue;
                if (expert_tensors.contains(t.name)) continue;
                s.tensors.push_back(t);
            }
            if (!s.tensors.empty()) resident_only.shards.push_back(std::move(s));
        }
        ARROW_RETURN_NOT_OK(
            integrity::VerifyAllTensorReads(input_dir, resident_only));
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
    int deferred_mtp = 0;

    auto lookup_scale = [&](const std::string &weight_name,
                            const TensorInfo *&info,
                            const SafetensorsModel::Shard *&shard)
        -> arrow::Status {
        const std::string scale_name = weight_name + "_scale_inv";
        auto it = scale_tensors.find(scale_name);
        if (it == scale_tensors.end())
            return arrow::Status::Invalid("missing weight_scale_inv for ",
                                          weight_name);
        shard = it->second.shard;
        info = it->second.info;
        return arrow::Status::OK();
    };

    auto add_nvfp4 = [&](const std::string &name, const TensorInfo &t,
                         const std::vector<uint8_t> &data,
                         std::vector<DenseNvfp4Record> &dst) -> arrow::Status {
        DenseNvfp4Record r;
        r.name = name;
        r.shape = t.shape;
        r.dtype = t.dtype;
        if (t.dtype == "F8_E4M3") {
            const TensorInfo *si = nullptr;
            const SafetensorsModel::Shard *ss = nullptr;
            ARROW_RETURN_NOT_OK(lookup_scale(name, si, ss));
            auto sdata = ReadTensorData(input_dir, *ss, *si);
            if (sdata.size() != static_cast<size_t>(si->byte_size()))
                return arrow::Status::IOError("failed to read scale: ", name);
            ARROW_RETURN_NOT_OK(QuantizeFp8(data, t.shape, sdata, si->shape, r));
        } else {
            ARROW_RETURN_NOT_OK(QuantizeBf16(data, t.numel(), t.shape, r));
        }
        dst.push_back(std::move(r));
        ++total_pinned;
        return arrow::Status::OK();
    };

    for (const auto &[name, loc] : tensors) {
        const auto &t = *loc.info;
        auto p = classify_ling_tensor(name);
        if (p.layer >= top.layers || p.kind == LingTensorKind::Mtp ||
            p.kind == LingTensorKind::Visual) {
            ++deferred_mtp;
            continue;
        }
        auto data = ReadTensorData(input_dir, *loc.shard, t);
        if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        switch (p.kind) {
        case LingTensorKind::Embed:
        case LingTensorKind::LmHead:
        case LingTensorKind::FinalNorm: {
            TensorRecord r;
            r.name = name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);
            global_raw.push_back(std::move(r));
            ++total_pinned;
            break;
        }
        case LingTensorKind::InputNorm:
        case LingTensorKind::PostNorm:
        case LingTensorKind::KdaConv:
        case LingTensorKind::KdaALog:
        case LingTensorKind::KdaDtBias:
        case LingTensorKind::KdaB:
        case LingTensorKind::KdaONorm:
        case LingTensorKind::MlaNorm:
        case LingTensorKind::MlaG: {
            TensorRecord r;
            r.name = name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);
            raw_by_layer[p.layer].push_back(std::move(r));
            ++total_pinned;
            break;
        }
        case LingTensorKind::RouterGate:
        case LingTensorKind::RouterBias: {
            TensorRecord r;
            r.name = name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);
            router_by_layer[p.layer].push_back(std::move(r));
            ++total_pinned;
            break;
        }
        case LingTensorKind::SharedGate:
        case LingTensorKind::SharedUp:
        case LingTensorKind::SharedDown:
            ARROW_RETURN_NOT_OK(
                add_nvfp4(name, t, data, shared_nvfp4_by_layer[p.layer]));
            break;
        case LingTensorKind::KdaQ:
        case LingTensorKind::KdaK:
        case LingTensorKind::KdaV:
        case LingTensorKind::KdaO:
        case LingTensorKind::KdaF:
        case LingTensorKind::KdaG:
        case LingTensorKind::MlaQa:
        case LingTensorKind::MlaQb:
        case LingTensorKind::MlaKvA:
        case LingTensorKind::MlaKvB:
        case LingTensorKind::MlaDense:
        case LingTensorKind::DenseMlpGate:
        case LingTensorKind::DenseMlpUp:
        case LingTensorKind::DenseMlpDown:
            ARROW_RETURN_NOT_OK(
                add_nvfp4(name, t, data, nvfp4_by_layer[p.layer]));
            break;
        default:
            return arrow::Status::Invalid("unhandled tensor kind: ", name);
        }
    }

    if (static_cast<int64_t>(global_raw.size()) != 3)
        return arrow::Status::Invalid(
            "global tensors missing (embed/lm_head/final norm)");

    // Routed experts: expert-major second pass (w1 = gate, w2 = down,
    // w3 = up), reading straight from the shards so the 535 GB source is
    // never resident.
    int experts_done = 0;
    int last_pct = -1;
    const bool skip_experts = std::getenv("LSX_LING_SKIP_EXPERTS") != nullptr;
    if (skip_experts)
        lsxcommon::log::info(
            "ling: LSX_LING_SKIP_EXPERTS set — reusing the existing expert "
            "store");
    for (int li = 0; !skip_experts && li < moe_layers; ++li) {
        const int l = top.moe_layers[static_cast<size_t>(li)];
        for (int e = 0; e < top.num_experts; ++e) {
            const int pct = experts_total > 0
                                ? static_cast<int>(100LL * experts_done /
                                                   experts_total)
                                : 100;
            if (pct != last_pct && pct % 5 == 0) {
                lsxcommon::log::info(absl::StrFormat(
                    "ling experts: %d/%d (%d%%) layer %d", experts_done,
                    experts_total, pct, l));
                last_pct = pct;
            }
            ++experts_done;
            const std::string stem =
                absl::StrCat("model.layers.", l, ".mlp.experts.", e, ".");
            const char *kLeaf[3] = {"gate_proj.weight", "down_proj.weight",
                                    "up_proj.weight"};
            DenseNvfp4Record recs[3];
            for (int k = 0; k < 3; ++k) {
                const std::string name = stem + kLeaf[k];
                auto it = expert_tensors.find(name);
                if (it == expert_tensors.end())
                    return arrow::Status::Invalid("missing expert tensor: ",
                                                  name);
                auto data = ReadTensorData(input_dir, *it->second.shard,
                                           *it->second.info);
                if (data.empty() ||
                    static_cast<int64_t>(data.size()) !=
                        it->second.info->byte_size())
                    return arrow::Status::IOError(
                        "failed to read expert tensor: ", name);
                const auto &shape = it->second.info->shape;
                if (shape.size() != 2)
                    return arrow::Status::Invalid(
                        "expert tensor must be 2-D: ", name);
                if (k == 1) {
                    if (shape[0] != H || shape[1] != I)
                        return arrow::Status::Invalid(
                            "down shape != [H,I]: ", name);
                } else if (shape[0] != I || shape[1] != H) {
                    return arrow::Status::Invalid(
                        "gate/up shape != [I,H]: ", name);
                }
                recs[k].name = name;
                recs[k].shape = shape;
                recs[k].dtype = "BF16";
                const TensorInfo *si = nullptr;
                const SafetensorsModel::Shard *ss = nullptr;
                ARROW_RETURN_NOT_OK(lookup_scale(name, si, ss));
                auto sdata = ReadTensorData(input_dir, *ss, *si);
                if (sdata.size() != static_cast<size_t>(si->byte_size()))
                    return arrow::Status::IOError("failed to read scale: ",
                                                  name);
                ARROW_RETURN_NOT_OK(
                    QuantizeFp8(data, shape, sdata, si->shape, recs[k]));
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

    ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
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

    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        tok_policy.pretokenizer = 3;  // Ling letter-run pattern (kind 5)
        if (top.eos_id >= 0) tok_policy.eos_id = top.eos_id;
        if (top.pad_id >= 0) tok_policy.pad_id = top.pad_id;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true,
                                           .pad_required = true},
            tok_policy));
        tokenizer_ok = true;
    } else if (!synth) {
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_dir);
    }

    VramGeometry vg;
    vg.hidden_dim = top.hidden;
    vg.kv_dim = top.heads * (top.qk_nope + top.qk_rope + top.v_head) / 2;
    vg.num_attn_layers = static_cast<int>(top.mla_layers.size());
    vg.num_moe_layers = moe_layers;
    vg.experts_per_layer = top.num_experts;
    vg.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            if (t.name.size() > 17 &&
                t.name.compare(t.name.size() - 17, 17, ".weight_scale_inv") ==
                    0)
                continue;
            inv.push_back({t.name, t.numel(), t.byte_size()});
        }
    const VramProfile vram = EstimateVram(
        inv,
        {+[](const std::string &n) {
             return ling_tensor_is_nvfp4(classify_ling_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return ling_tensor_is_routed_expert(
                        classify_ling_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         +[](const std::string &) { return false; }},
        vg);

    {
        std::ostringstream j;
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n"
          << "  \"model\": \"" << frozen::kModelName << "\",\n"
          << "  \"model_type\": \"" << kType << "\",\n"
          << "  \"num_layers\": " << top.layers << ",\n  \"mla_layers\": [";
        for (size_t i = 0; i < top.mla_layers.size(); ++i) {
            if (i) j << ",";
            j << top.mla_layers[i];
        }
        j << "],\n  \"kda_layers\": [";
        for (size_t i = 0; i < top.kda_layers.size(); ++i) {
            if (i) j << ",";
            j << top.kda_layers[i];
        }
        j << "],\n  \"moe_layers\": [";
        for (size_t i = 0; i < top.moe_layers.size(); ++i) {
            if (i) j << ",";
            j << top.moe_layers[i];
        }
        j << "],\n  \"dense_layers\": [";
        for (size_t i = 0; i < top.dense_layers.size(); ++i) {
            if (i) j << ",";
            j << top.dense_layers[i];
        }
        j << "],\n  \"expert_swiglu_limit_list\": [";
        for (size_t i = 0; i < top.expert_swiglu_limit.size(); ++i) {
            if (i) j << ",";
            j << top.expert_swiglu_limit[i];
        }
        j << "],\n  \"share_expert_swiglu_limit_list\": [";
        for (size_t i = 0; i < top.shared_swiglu_limit.size(); ++i) {
            if (i) j << ",";
            j << top.shared_swiglu_limit[i];
        }
        j << "],\n  \"hidden_size\": " << top.hidden
          << ",\n  \"vocab_size\": " << top.vocab
          << ",\n  \"num_attention_heads\": " << top.heads
          << ",\n  \"num_key_value_heads\": " << top.kv_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"qk_head_dim\": " << top.qk_head_dim
          << ",\n  \"qk_nope_head_dim\": " << top.qk_nope
          << ",\n  \"qk_rope_head_dim\": " << top.qk_rope
          << ",\n  \"v_head_dim\": " << top.v_head
          << ",\n  \"q_lora_rank\": " << top.q_lora
          << ",\n  \"kv_lora_rank\": " << top.kv_lora
          << ",\n  \"short_conv_kernel_size\": " << top.conv_kernel
          << ",\n  \"rms_norm_eps\": " << top.rms_eps
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.top_k
          << ",\n  \"n_group\": " << top.n_group
          << ",\n  \"topk_group\": " << top.topk_group
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate
          << ",\n  \"shared_intermediate_size\": " << top.shared_intermediate
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate
          << ",\n  \"first_moe_layer\": " << top.first_k_dense
          << ",\n  \"routed_scaling\": " << top.routed_scale
          << ",\n  \"eos_token_id\": " << top.eos_id
          << ",\n  \"pad_token_id\": " << top.pad_id
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"deferred\": {\"mtp\": " << deferred_mtp << "}"
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << total_pinned
          << ",\n  \"total_experts\": " << experts_total << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

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
        "ling conversion: %d layers (%d MoE, %d MLA, %d KDA), %d experts/layer, "
        "%d pinned tensors, %d deferred MTP tensors",
        top.layers, moe_layers, static_cast<int>(top.mla_layers.size()),
        static_cast<int>(top.kda_layers.size()), top.num_experts,
        total_pinned, deferred_mtp));
    return arrow::Status::OK();
}

} // namespace lsxcommon::ling
