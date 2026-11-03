#include "lsxcommon/qwen_moe_convert.h"
#include "lsxcommon/synthetic_geometry.h"
#include "qwen_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
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

namespace lsxcommon::qwen_moe {
namespace {

// simdjson ondemand idioms copied from gemma_moe_convert.cpp parse_text_config.
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

} // namespace

namespace {

// Synthetic fixtures (tests) may use tiny geometry; the real pipeline hard
// gates against the frozen constants.

// Cross-checks parsed config against the frozen passport. Returns false and
// fills `why` on mismatch.
bool check_frozen_geometry(const QwenMoeTopology &top, std::string &why)
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
    if (top.linear_num_key_heads != kLinearNumKeyHeads) return bad("linear_num_key_heads", top.linear_num_key_heads, kLinearNumKeyHeads);
    if (top.linear_num_value_heads != kLinearNumValueHeads) return bad("linear_num_value_heads", top.linear_num_value_heads, kLinearNumValueHeads);
    if (top.linear_key_head_dim != kLinearKeyHeadDim) return bad("linear_key_head_dim", top.linear_key_head_dim, kLinearKeyHeadDim);
    if (top.linear_value_head_dim != kLinearValueHeadDim) return bad("linear_value_head_dim", top.linear_value_head_dim, kLinearValueHeadDim);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts) return bad("num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK) return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate) return bad("moe_intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.shared_expert_intermediate_size != kSharedIntermediate) return bad("shared_expert_intermediate_size", top.shared_expert_intermediate_size, kSharedIntermediate);
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want = (i % kFullAttnInterval == kFullAttnInterval - 1)
                                     ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
    }
    return true;
}

// Quantizes a contiguous BF16 slice to NVFP4. Returns false on GPU failure.
bool quantize_slice(const uint8_t *data, int64_t numel,
                    DenseNvfp4Record &r)
{
    return gpu::quantize_tensor_gpu(data, numel, "BF16", nullptr, r.shape,
                                    {}, r.packed, r.scales, r.global_scale);
}

arrow::Status write_batch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

// Writes one expert's NVFP4 payload triple to experts/layer_<l>/exp_<e>.arrow.
arrow::Status write_expert_file(const std::filesystem::path &base_dir,
                                int layer, int expert, PackedExpert &&pe)
{
    namespace fs = std::filesystem;
    fs::path dir = base_dir / "experts" /
                   absl::StrCat("layer_", layer);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch,
                                  dir / absl::StrCat("exp_", expert, ".arrow"));
}

} // namespace

QwenMoeTopology ParseQwenMoeConfig(const std::string &input_dir)
{
    QwenMoeTopology top;
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

    // text_config preferred; fall back to flat root (some fixtures).
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
    top.linear_num_key_heads = get_i(src, "linear_num_key_heads", 0);
    top.linear_num_value_heads = get_i(src, "linear_num_value_heads", 0);
    top.linear_key_head_dim = get_i(src, "linear_key_head_dim", 0);
    top.linear_value_head_dim = get_i(src, "linear_value_head_dim", 0);
    top.linear_conv_kernel_dim = get_i(src, "linear_conv_kernel_dim", 4);
    top.intermediate_size = get_i(src, "intermediate_size", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-6);
    top.partial_rotary_factor = get_d(src, "partial_rotary_factor", 0.25);
    top.rope_theta = get_d(src, "rope_theta", 1e7);

    // MoE fields (etalon: gemma_moe_convert.cpp num_experts parsing).
    top.num_experts = get_i(src, "num_experts", 0);
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(src, "moe_intermediate_size", 0);
    top.shared_expert_intermediate_size =
        get_i(src, "shared_expert_intermediate_size", 0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);
    top.bos_token_id = get_i(src, "bos_token_id", -1);

    if (auto lt = src["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                top.layer_types.emplace_back(sv);
        }
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.layer_types.empty()) {
        top.error_msg = "config.json missing required MoE fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layer_types size ", top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    return top;
}

ParsedMoeTensor classify_qwen_moe_tensor(const std::string &name)
{
    ParsedMoeTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = MoeTensorKind::Mtp;
        return r;
    }
    if (name.find("visual") != std::string::npos) {
        r.kind = MoeTensorKind::Visual;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = MoeTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "model.language_model.";
    if (name.rfind(kPrefix, 0) != 0)
        return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = MoeTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = MoeTensorKind::FinalNorm;
        return r;
    }

    // Extract "layers.<N>." index and the per-layer body.
    std::string body = rest;
    const std::string kLayers = "layers.";
    if (rest.rfind(kLayers, 0) == 0) {
        const auto digits_start = kLayers.size();
        const auto digits_end = rest.find('.', digits_start);
        if (digits_end == std::string::npos)
            return r;
        int layer = 0;
        if (!detail::ParseTensorIndex(
                std::string_view(rest).substr(digits_start,
                                              digits_end - digits_start),
                layer))
            return r;  // malformed index -> Unknown, never throw
        r.layer = layer;
        body = rest.substr(digits_end + 1);
    }

    using K = MoeTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::FullAttnQ},
        {"self_attn.k_proj.weight", K::FullAttnK},
        {"self_attn.v_proj.weight", K::FullAttnV},
        {"self_attn.o_proj.weight", K::FullAttnO},
        {"self_attn.q_norm.weight", K::FullAttnQNorm},
        {"self_attn.k_norm.weight", K::FullAttnKNorm},
        {"linear_attn.in_proj_qkv.weight", K::LinearAttnQkv},
        {"linear_attn.in_proj_z.weight", K::LinearAttnZ},
        {"linear_attn.in_proj_b.weight", K::LinearAttnB},
        {"linear_attn.in_proj_a.weight", K::LinearAttnA},
        {"linear_attn.out_proj.weight", K::LinearAttnOut},
        {"linear_attn.conv1d.weight", K::LinearAttnConv},
        {"linear_attn.norm.weight", K::LinearAttnNorm},
        {"linear_attn.A_log", K::LinearAttnALog},
        {"linear_attn.dt_bias", K::LinearAttnDt},
        {"mlp.gate.weight", K::RouterGate},
        {"mlp.experts.gate_up_proj", K::FusedExpertsGateUp},
        {"mlp.experts.down_proj", K::FusedExpertsDown},
        {"mlp.shared_expert.gate_proj.weight", K::SharedExpertGateProj},
        {"mlp.shared_expert.up_proj.weight", K::SharedExpertUpProj},
        {"mlp.shared_expert.down_proj.weight", K::SharedExpertDownProj},
        {"mlp.shared_expert_gate.weight", K::SharedExpertGate},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool qwen_moe_tensor_is_nvfp4(MoeTensorKind kind)
{
    switch (kind) {
    case MoeTensorKind::Embed:
    case MoeTensorKind::LmHead:
    case MoeTensorKind::FullAttnQ:
    case MoeTensorKind::FullAttnK:
    case MoeTensorKind::FullAttnV:
    case MoeTensorKind::FullAttnO:
    case MoeTensorKind::LinearAttnQkv:
    case MoeTensorKind::LinearAttnZ:
    case MoeTensorKind::LinearAttnB:
    case MoeTensorKind::LinearAttnA:
    case MoeTensorKind::LinearAttnOut:
    case MoeTensorKind::FusedExpertsGateUp:
    case MoeTensorKind::FusedExpertsDown:
    case MoeTensorKind::SharedExpertGateProj:
    case MoeTensorKind::SharedExpertUpProj:
    case MoeTensorKind::SharedExpertDownProj:
        return true;
    default:
        return false;
    }
}

arrow::Status ConvertQwenMoeModelCommon(const std::string &input_dir,
                                        const std::string &output_root,
                                        const QwenMoeConvertSpec &spec)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    QwenMoeTopology top = ParseQwenMoeConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Qwen-MoE config error: ", top.error_msg);

    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!spec.check_frozen || !spec.check_frozen(top, why))
            return arrow::Status::Invalid("Qwen-MoE frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    // Classify the inventory up front: unknown names are hard errors, visual
    // tensors are counted skips, everything else must be consumable.
    int visual_skipped = 0;
    absl::btree_map<int, int> layer_kind_count;  // layer -> classified tensors
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_qwen_moe_tensor(t.name);
            if (p.kind == MoeTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == MoeTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in qwen3_5_moe checkpoint: ", t.name);
            if (p.layer >= 0)
                ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    // (2) Source reads must be complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    // Unified vault convention: <output_root>/logestix (same as the dense
    // qwen/glm/kimi converters; legacy logestix-qwen35moe readers keep a
    // loader-side fallback for pre-unification artifacts).
    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    const fs::path mtp_dir = base / "mtp";
    for (const fs::path &d : {base, pinned_dir, router_dir, shared_dir, mtp_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    // Buckets (etalon: ConvertQwenModel bucket layout).
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> shared_nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> shared_gate_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;
    std::vector<DenseNvfp4Record> mtp_nvfp4;
    std::vector<TensorRecord> mtp_raw;

    int64_t total_pinned = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_qwen_moe_tensor(t.name);
            if (p.kind == MoeTensorKind::Visual)
                continue;  // counted skip

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            // MTP subtree: converted and stored, never applied at inference
            // (HF parity — modeling_qwen3_5_moe.py ignores ^mtp.* on load).
            // Norms stay high precision; everything else is NVFP4.
            if (p.kind == MoeTensorKind::Mtp) {
                const bool keep_raw = t.name.find("norm") != std::string::npos;
                if (keep_raw) {
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
                continue;
            }

            switch (p.kind) {
            case MoeTensorKind::Embed:
            case MoeTensorKind::LmHead: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!quantize_slice(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                global_nvfp4.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::FinalNorm:
            case MoeTensorKind::InputNorm:
            case MoeTensorKind::PostNorm:
            case MoeTensorKind::FullAttnQNorm:
            case MoeTensorKind::FullAttnKNorm:
            case MoeTensorKind::LinearAttnConv:
            case MoeTensorKind::LinearAttnNorm:
            case MoeTensorKind::LinearAttnALog:
            case MoeTensorKind::LinearAttnDt: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                if (p.kind == MoeTensorKind::FinalNorm)
                    global_raw.push_back(std::move(r));
                else
                    raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::FullAttnQ:
            case MoeTensorKind::FullAttnK:
            case MoeTensorKind::FullAttnV:
            case MoeTensorKind::FullAttnO:
            case MoeTensorKind::LinearAttnQkv:
            case MoeTensorKind::LinearAttnZ:
            case MoeTensorKind::LinearAttnB:
            case MoeTensorKind::LinearAttnA:
            case MoeTensorKind::LinearAttnOut: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!quantize_slice(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::RouterGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                router_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::SharedExpertGateProj:
            case MoeTensorKind::SharedExpertUpProj:
            case MoeTensorKind::SharedExpertDownProj: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!quantize_slice(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                shared_nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::SharedExpertGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                shared_gate_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MoeTensorKind::FusedExpertsGateUp:
            case MoeTensorKind::FusedExpertsDown:
                continue;  // handled by the dedicated fused-expert pass below
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ",
                                              t.name);
            }
        }
    }

    // Fused-expert split. HF semantics (modeling_qwen3_5_moe.py):
    // gate_up_proj[e] is [2I, H]; gate = rows [0,I), up = rows [I,2I).
    // down_proj[e] is [H, I]; expert e's block is contiguous.
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    const int64_t row_bytes = static_cast<int64_t>(H) * 2;      // BF16 row
    using ExpertKey = std::pair<int, int>;                      // (layer, eid)
    absl::btree_map<ExpertKey, std::pair<DenseNvfp4Record, DenseNvfp4Record>>
        gate_up_w1_w3;
    absl::btree_map<ExpertKey, DenseNvfp4Record> down_w2;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_qwen_moe_tensor(t.name);
            if (p.kind != MoeTensorKind::FusedExpertsGateUp &&
                p.kind != MoeTensorKind::FusedExpertsDown)
                continue;
            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);
            if (t.shape.size() != 3 || t.shape[0] != E)
                return arrow::Status::Invalid("bad fused-expert shape: ",
                                              t.name);

            for (int e = 0; e < E; ++e) {
                if (p.kind == MoeTensorKind::FusedExpertsGateUp) {
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
                    // down_proj partner is quantized below into a side map.
                    gate_up_w1_w3.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(p.layer, e),
                        std::forward_as_tuple(std::move(w1r), std::move(w3r)));
                } else {
                    if (t.shape[1] != H || t.shape[2] != I)
                        return arrow::Status::Invalid(
                            "down_proj shape != [E,H,I]: ", t.name);
                    const int64_t d_off =
                        static_cast<int64_t>(e) * H * I * 2;
                    DenseNvfp4Record w2r;
                    w2r.name = absl::StrCat(t.name, ".w2.e", e);
                    w2r.shape = {H, I}; w2r.dtype = t.dtype;
                    if (!quantize_slice(data.data() + d_off,
                                        static_cast<int64_t>(H) * I, w2r))
                        return arrow::Status::IOError("quantize failed w2 ",
                                                      t.name, " e", e);
                    down_w2.insert_or_assign(std::make_pair(p.layer, e),
                                             std::move(w2r));
                }
            }
        }
    }

    // Join gate_up halves with down halves and write per-expert files.
    for (int l = 0; l < top.num_layers; ++l) {
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
            ARROW_RETURN_NOT_OK(write_expert_file(base, l, e, std::move(pe)));
        }
    }

    // Globals + dense layers (naming identical to the qwen loader layout).
    if (!global_raw.empty()) {
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    }
    if (!global_nvfp4.empty()) {
        ARROW_RETURN_NOT_OK(
            write_batch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                        pinned_dir / "global_nvfp4.arrow"));
    }
    for (auto &[layer, recs] : nvfp4_by_layer) {
        ARROW_RETURN_NOT_OK(write_batch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    }
    for (auto &[layer, recs] : raw_by_layer) {
        ARROW_RETURN_NOT_OK(write_batch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    }
    for (auto &[layer, recs] : router_by_layer) {
        ARROW_RETURN_NOT_OK(write_batch(
            PinnedSchema::MakeBatch(recs),
            router_dir / absl::StrCat("layer_", layer, ".arrow")));
    }
    for (auto &[layer, recs] : shared_nvfp4_by_layer) {
        ARROW_RETURN_NOT_OK(write_batch(
            DenseNvfp4Schema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    }
    for (auto &[layer, recs] : shared_gate_by_layer) {
        ARROW_RETURN_NOT_OK(write_batch(
            PinnedSchema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_gate.arrow")));
    }
    if (!mtp_nvfp4.empty()) {
        ARROW_RETURN_NOT_OK(
            write_batch(DenseNvfp4Schema::MakeBatch(mtp_nvfp4),
                        mtp_dir / "mtp_nvfp4.arrow"));
    }
    if (!mtp_raw.empty()) {
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(mtp_raw),
                                        mtp_dir / "mtp_raw.arrow"));
    }

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    // Tokenizer: mandatory for real checkpoints; synthetic fixtures may omit
    // it (recorded in the index).
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 5;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true}, tok_policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // Structural manifest (index). Deterministic byte content.
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"" << spec.format_tag << "\",\n"
          << "  \"model\": \"" << spec.model_name << "\",\n"
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
          << ",\n  \"eos_token_id\": " << top.eos_token_id;
        if (spec.record_bos && top.bos_token_id >= 0)
            j << ",\n  \"bos_token_id\": " << top.bos_token_id;
        j << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Final verification: every produced arrow file must re-open cleanly.
    for (const fs::path &dir :
         {pinned_dir, router_dir, shared_dir, mtp_dir}) {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
             it.increment(ec)) {
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
        "Qwen-MoE conversion: %d layers x %d experts, %d visual tensors "
        "skipped, %d pinned tensors",
        top.num_layers, top.num_experts, visual_skipped,
        top.total_pinned_tensors));
    return arrow::Status::OK();
}

arrow::Status ConvertQwen35MoeModel(const std::string &input_dir,
                                    const std::string &output_root)
{
    constexpr QwenMoeConvertSpec kSpec = {"logestix-qwen35moe",
                                          "Qwen3.5-35B-A3B",
                                          &check_frozen_geometry,
                                          /*record_bos=*/false};
    return ConvertQwenMoeModelCommon(input_dir, output_root, kSpec);
}

} // namespace lsxcommon::qwen_moe
