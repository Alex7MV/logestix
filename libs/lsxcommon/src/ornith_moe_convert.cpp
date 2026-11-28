#include "lsxcommon/ornith_moe_convert.h"
#include "lsxcommon/synthetic_geometry.h"
#include "ornith_moe_constants.h"

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

namespace lsxcommon::ornith {
namespace {

// simdjson ondemand idioms copied from qwen_moe_convert.cpp parse_text_config.
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

// Synthetic fixtures (tests) may use tiny geometry; the real pipeline hard
// gates against the frozen constants.

// Converts a plain RMSNorm weight from the MLX absolute form (multiplier `w`)
// to the Qwen/HF offset form used by the runtime (`out = rmsnorm(x) * (1 + w)`)
// and stores it as F32 to preserve the small offset. RMSNormGated
// (linear_attn.norm) is NOT offset in either exporter and must not pass here.
std::vector<uint8_t> OffsetNormToF32(const uint8_t *data, int64_t n,
                                     const std::string &dtype)
{
    std::vector<uint8_t> out(static_cast<size_t>(n) * 4);
    for (int64_t i = 0; i < n; ++i) {
        float f;
        if (dtype == "F32") {
            std::memcpy(&f, data + i * 4, 4);
        } else {
            uint16_t bits = static_cast<uint16_t>(data[i * 2]) |
                            (static_cast<uint16_t>(data[i * 2 + 1]) << 8);
            uint32_t u = static_cast<uint32_t>(bits) << 16;
            std::memcpy(&f, &u, 4);
        }
        f -= 1.0f;
        std::memcpy(out.data() + static_cast<size_t>(i) * 4, &f, 4);
    }
    return out;
}

// Quantizes a contiguous BF16 slice to NVFP4. Returns false on GPU failure.
// Writes one expert's NVFP4 payload triple to experts/layer_<l>/exp_<e>.arrow.
} // namespace

// ── Frozen-geometry gate ──────────────────────────────────────────────────

bool CheckOrnithFrozenGeometry(const OrnithMoeTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != kHidden)
        return bad("hidden_size", top.hidden_size, kHidden);
    if (top.num_layers != kLayers)
        return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKVHeads);
    if (top.head_dim != kHeadDim)
        return bad("head_dim", top.head_dim, kHeadDim);
    if (top.linear_num_key_heads != kLinearNumKeyHeads)
        return bad("linear_num_key_heads", top.linear_num_key_heads,
                   kLinearNumKeyHeads);
    if (top.linear_num_value_heads != kLinearNumValueHeads)
        return bad("linear_num_value_heads", top.linear_num_value_heads,
                   kLinearNumValueHeads);
    if (top.linear_key_head_dim != kLinearKeyHeadDim)
        return bad("linear_key_head_dim", top.linear_key_head_dim,
                   kLinearKeyHeadDim);
    if (top.linear_value_head_dim != kLinearValueHeadDim)
        return bad("linear_value_head_dim", top.linear_value_head_dim,
                   kLinearValueHeadDim);
    if (top.vocab_size != kVocab)
        return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("num_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   kMoeIntermediate);
    if (top.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   top.shared_expert_intermediate_size, kSharedIntermediate);
    if (top.bos_token_id != kBosTokenId)
        return bad("bos_token_id", top.bos_token_id, kBosTokenId);
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want = (i % kFullAttnInterval == kFullAttnInterval - 1)
                                     ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
    }
    return true;
}

// ── Config parsing ────────────────────────────────────────────────────────

OrnithMoeTopology ParseOrnithMoeConfig(const std::string &input_dir)
{
    OrnithMoeTopology top;
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

// ── Tensor classification (MLX and HF layouts) ─────────────────────────────

ParsedOrnithTensor classify_ornith_moe_tensor(const std::string &name)
{
    ParsedOrnithTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = OrnithMoeTensorKind::Mtp;
        return r;
    }
    if (name.find("visual") != std::string::npos) {
        r.kind = OrnithMoeTensorKind::Visual;
        return r;
    }
    // The language-model prefix selects the source layout; the two differ in
    // the expert tensors they carry and in the plain-norm storage form.
    std::string rest;
    if (name == "lm_head.weight") {
        r.kind = OrnithMoeTensorKind::LmHead;
        r.layout = OrnithMoeSourceLayout::HuggingFace;
        return r;
    }
    if (name == "language_model.lm_head.weight") {
        r.kind = OrnithMoeTensorKind::LmHead;
        r.layout = OrnithMoeSourceLayout::Mlx;
        return r;
    }
    constexpr std::string_view kMlxPrefix = "language_model.model.";
    constexpr std::string_view kHfPrefix = "model.language_model.";
    if (name.rfind(kMlxPrefix, 0) == 0) {
        rest = name.substr(kMlxPrefix.size());
        r.layout = OrnithMoeSourceLayout::Mlx;
    } else if (name.rfind(kHfPrefix, 0) == 0) {
        rest = name.substr(kHfPrefix.size());
        r.layout = OrnithMoeSourceLayout::HuggingFace;
    } else {
        return r;
    }
    if (rest == "embed_tokens.weight") {
        r.kind = OrnithMoeTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = OrnithMoeTensorKind::FinalNorm;
        return r;
    }

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

    using K = OrnithMoeTensorKind;
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
        {"mlp.switch_mlp.gate_proj.weight", K::SwitchGateProj},
        {"mlp.switch_mlp.up_proj.weight", K::SwitchUpProj},
        {"mlp.switch_mlp.down_proj.weight", K::SwitchDownProj},
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

bool ornith_plain_norm_is_absolute(OrnithMoeSourceLayout layout)
{
    // Only the MLX export stores the plain RMSNorm weights as absolute
    // multipliers; transformers exports use the runtime offset form.
    return layout == OrnithMoeSourceLayout::Mlx;
}

bool ornith_moe_tensor_is_nvfp4(OrnithMoeTensorKind kind)
{
    switch (kind) {
    case OrnithMoeTensorKind::Embed:
    case OrnithMoeTensorKind::LmHead:
    case OrnithMoeTensorKind::FullAttnQ:
    case OrnithMoeTensorKind::FullAttnK:
    case OrnithMoeTensorKind::FullAttnV:
    case OrnithMoeTensorKind::FullAttnO:
    case OrnithMoeTensorKind::LinearAttnQkv:
    case OrnithMoeTensorKind::LinearAttnZ:
    case OrnithMoeTensorKind::LinearAttnB:
    case OrnithMoeTensorKind::LinearAttnA:
    case OrnithMoeTensorKind::LinearAttnOut:
    case OrnithMoeTensorKind::SwitchGateProj:
    case OrnithMoeTensorKind::SwitchUpProj:
    case OrnithMoeTensorKind::SwitchDownProj:
    case OrnithMoeTensorKind::FusedExpertsGateUp:
    case OrnithMoeTensorKind::FusedExpertsDown:
    case OrnithMoeTensorKind::SharedExpertGateProj:
    case OrnithMoeTensorKind::SharedExpertUpProj:
    case OrnithMoeTensorKind::SharedExpertDownProj:
        return true;
    default:
        return false;
    }
}

// ── Conversion ────────────────────────────────────────────────────────────

arrow::Status ConvertOrnithMoeModel(const std::string &input_dir,
                                    const std::string &output_root)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    OrnithMoeTopology top = ParseOrnithMoeConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Ornith-MoE config error: ", top.error_msg);

    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckOrnithFrozenGeometry(top, why))
            return arrow::Status::Invalid("Ornith-MoE frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    int visual_skipped = 0;
    absl::btree_map<int, int> layer_kind_count;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_ornith_moe_tensor(t.name);
            if (p.kind == OrnithMoeTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == OrnithMoeTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in ornith checkpoint: ", t.name);
            if (p.layer >= 0)
                ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    // (2) Source reads must be complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    const fs::path base = fs::path(output_root);
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    // No mtp/: v1 ships no ballast (design §1.4), so the artifact's mask names
    // only planes the load path can reach.
    for (const fs::path &d : {base, pinned_dir, router_dir, shared_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> shared_nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> shared_gate_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int mtp_tensors = 0;
    int64_t total_pinned = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_ornith_moe_tensor(t.name);
            if (p.kind == OrnithMoeTensorKind::Visual)
                continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            if (p.kind == OrnithMoeTensorKind::Mtp) {
                // MTP subtree: never applied at inference (HF parity — the
                // reference implementation ignores ^mtp.*) and therefore not
                // stored. COUNTED, because a silent drop is unauditable.
                ++mtp_tensors;
                continue;
            }

            switch (p.kind) {
            case OrnithMoeTensorKind::Embed:
            case OrnithMoeTensorKind::LmHead: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                global_nvfp4.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::FinalNorm:
            case OrnithMoeTensorKind::InputNorm:
            case OrnithMoeTensorKind::PostNorm:
            case OrnithMoeTensorKind::FullAttnQNorm:
            case OrnithMoeTensorKind::FullAttnKNorm: {
                // MLX absolute -> runtime offset form (see OffsetNormToF32);
                // the HF export already stores the offset form and passes
                // through unchanged.
                TensorRecord r;
                r.name = t.name;
                r.shape = t.shape;
                if (ornith_plain_norm_is_absolute(p.layout)) {
                    r.dtype = "F32";
                    r.data = OffsetNormToF32(data.data(), t.numel(), t.dtype);
                } else {
                    r.dtype = t.dtype;
                    r.data = std::move(data);
                }
                if (p.kind == OrnithMoeTensorKind::FinalNorm)
                    global_raw.push_back(std::move(r));
                else
                    raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::LinearAttnNorm:
            case OrnithMoeTensorKind::LinearAttnALog:
            case OrnithMoeTensorKind::LinearAttnDt: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::LinearAttnConv: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype;
                // MLX [C, K, 1] -> HF [C, 1, K]; identical flat order
                // (c*K+k). The HF export is already [C, 1, K].
                if (t.shape.size() == 3 && t.shape[2] == 1)
                    r.shape = {t.shape[0], 1, t.shape[1]};
                else
                    r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::FullAttnQ:
            case OrnithMoeTensorKind::FullAttnK:
            case OrnithMoeTensorKind::FullAttnV:
            case OrnithMoeTensorKind::FullAttnO:
            case OrnithMoeTensorKind::LinearAttnQkv:
            case OrnithMoeTensorKind::LinearAttnZ:
            case OrnithMoeTensorKind::LinearAttnB:
            case OrnithMoeTensorKind::LinearAttnA:
            case OrnithMoeTensorKind::LinearAttnOut: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::RouterGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                router_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::SharedExpertGateProj:
            case OrnithMoeTensorKind::SharedExpertUpProj:
            case OrnithMoeTensorKind::SharedExpertDownProj: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                shared_nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::SharedExpertGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                shared_gate_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case OrnithMoeTensorKind::SwitchGateProj:
            case OrnithMoeTensorKind::SwitchUpProj:
            case OrnithMoeTensorKind::SwitchDownProj:
            case OrnithMoeTensorKind::FusedExpertsGateUp:
            case OrnithMoeTensorKind::FusedExpertsDown:
                continue;  // handled by the expert passes below
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ",
                                              t.name);
            }
        }
    }

    // Expert split — two source layouts, one destination:
    //   MLX stacked  switch_mlp.gate_proj [E,I,H] -> w1, up_proj -> w3,
    //                down_proj [E,H,I] -> w2;
    //   HF fused     experts.gate_up_proj [E,2I,H]: gate = rows [0,I),
    //                up = rows [I,2I); down_proj [E,H,I] -> w2.
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    using ExpertKey = std::pair<int, int>;  // (layer, eid)
    absl::btree_map<ExpertKey, DenseNvfp4Record> w1_map, w2_map, w3_map;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_ornith_moe_tensor(t.name);
            const bool is_w1 = p.kind == OrnithMoeTensorKind::SwitchGateProj;
            const bool is_w3 = p.kind == OrnithMoeTensorKind::SwitchUpProj;
            const bool is_w2 = p.kind == OrnithMoeTensorKind::SwitchDownProj;
            const bool fused_up =
                p.kind == OrnithMoeTensorKind::FusedExpertsGateUp;
            const bool fused_down =
                p.kind == OrnithMoeTensorKind::FusedExpertsDown;
            if (!is_w1 && !is_w2 && !is_w3 && !fused_up && !fused_down)
                continue;
            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);
            if (t.shape.size() != 3 || t.shape[0] != E)
                return arrow::Status::Invalid("bad expert shape: ", t.name);
            if (fused_up && t.shape[1] != 2 * I)
                return arrow::Status::Invalid(
                    "gate_up_proj dim1 != 2*moe_intermediate: ", t.name);
            if (fused_down && (t.shape[1] != H || t.shape[2] != I))
                return arrow::Status::Invalid(
                    "experts.down_proj shape != [E,H,I]: ", t.name);
            // Stacked rows: [E,I,H] for gate/up, [E,H,I] for down. Fused rows:
            // [E,2I,H] split at I.
            const int64_t rows = is_w2 || fused_down ? H : (fused_up ? 2 * I : I);
            const int64_t cols = is_w2 || fused_down ? I : H;
            if (!fused_up && !fused_down &&
                (t.shape[1] != rows || t.shape[2] != cols))
                return arrow::Status::Invalid("expert shape mismatch: ",
                                              t.name);
            const int64_t row_bytes = cols * 2;   // BF16
            const int64_t block = rows * row_bytes;
            for (int e = 0; e < E; ++e) {
                const uint8_t *base =
                    data.data() + static_cast<int64_t>(e) * block;
                if (fused_up) {
                    // gate = rows [0,I) -> w1, up = rows [I,2I) -> w3.
                    DenseNvfp4Record w1r, w3r;
                    w1r.name = absl::StrCat(t.name, ".w1.e", e);
                    w1r.shape = {I, H}; w1r.dtype = t.dtype;
                    w3r.name = absl::StrCat(t.name, ".w3.e", e);
                    w3r.shape = {I, H}; w3r.dtype = t.dtype;
                    if (!QuantizeSliceBf16(base, static_cast<int64_t>(I) * H, w1r))
                        return arrow::Status::IOError("quantize failed w1 ",
                                                      t.name, " e", e);
                    if (!QuantizeSliceBf16(base + I * row_bytes,
                                        static_cast<int64_t>(I) * H, w3r))
                        return arrow::Status::IOError("quantize failed w3 ",
                                                      t.name, " e", e);
                    w1_map[{p.layer, e}] = std::move(w1r);
                    w3_map[{p.layer, e}] = std::move(w3r);
                    continue;
                }
                DenseNvfp4Record r;
                r.name = fused_down ? absl::StrCat(t.name, ".w2.e", e)
                                    : absl::StrCat(t.name, ".e", e);
                r.shape = {rows, cols}; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(base, rows * cols, r))
                    return arrow::Status::IOError("quantize failed ", t.name,
                                                  " e", e);
                auto &map = is_w1 ? w1_map : (is_w3 ? w3_map : w2_map);
                map[{p.layer, e}] = std::move(r);
            }
        }
    }

    for (int l = 0; l < top.num_layers; ++l) {
        for (int e = 0; e < E; ++e) {
            auto a = w1_map.find({l, e});
            auto b = w3_map.find({l, e});
            auto c = w2_map.find({l, e});
            if (a == w1_map.end() || b == w3_map.end() || c == w2_map.end())
                return arrow::Status::Invalid("missing switch_mlp half: layer ",
                                              l, " expert ", e);
            PackedExpert pe;
            pe.w1_packed = a->second.packed;
            pe.w1_scales = a->second.scales;
            pe.w1_shape = a->second.shape;
            pe.w1_global_scale = a->second.global_scale;
            pe.w3_packed = b->second.packed;
            pe.w3_scales = b->second.scales;
            pe.w3_shape = b->second.shape;
            pe.w3_global_scale = b->second.global_scale;
            pe.w2_packed = c->second.packed;
            pe.w2_scales = c->second.scales;
            pe.w2_shape = c->second.shape;
            pe.w2_global_scale = c->second.global_scale;
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
        }
    }

    if (!global_raw.empty()) {
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    }
    if (!global_nvfp4.empty()) {
        ARROW_RETURN_NOT_OK(
            WriteBatch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                        pinned_dir / "global_nvfp4.arrow"));
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
    for (auto &[layer, recs] : router_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            router_dir / absl::StrCat("layer_", layer, ".arrow")));
    }
    for (auto &[layer, recs] : shared_nvfp4_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    }
    for (auto &[layer, recs] : shared_gate_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_gate.arrow")));
    }
    top.total_pinned_tensors = static_cast<int>(total_pinned);

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

    {
        std::ostringstream j;
        j << "{\n"
          << "  \"layout\": 2,\n"
          << "  \"model\": \"Ornith-1.5-35B-A3B\",\n"
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
        if (top.bos_token_id >= 0)
            j << ",\n  \"bos_token_id\": " << top.bos_token_id;
        j << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          << ",\n  \"dropped_mtp_tensors\": " << mtp_tensors
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << "\n}";
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
        if (ec)
            return arrow::Status::IOError("cannot list ", dir.string());
    }

    lsxcommon::log::info(absl::StrFormat(
        "Ornith-MoE conversion: %d layers x %d experts, %d visual tensors "
        "skipped, %d MTP tensors dropped (no ballast), %d pinned tensors",
        top.num_layers, top.num_experts, visual_skipped, mtp_tensors,
        top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::ornith
