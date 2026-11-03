// Kimi-K3 (KimiLinear) converter.
//
// Layout: <output>/logestix/{pinned,streamed,experts}/ + model.index.json.
//   * pinned/       attention + layer-0 MLP + embed/lm_head + router + all
//                   norms/gates: the session-resident graph.
//   * streamed/     the Stable LatentMoE plane (latent down/up + the two shared
//                   experts) per layer: mmap views copied to the device once
//                   per layer per step, because pinning it needs ~34 GiB.
//   * experts/      one Arrow file per routed expert (neutral ExpertSchema).
//
// Routed experts ship as MXFP4 (E2M1 nibbles packed 2/byte, E8M0 exponent
// scale per 32 columns, no global scale). The NVFP4 contour decodes
// pow2-E4M3 per-16 scales multiplied by a per-tensor global scale, so the
// repack is EXACT (nibbles verbatim, each E8M0 byte duplicated into the two
// per-16 halves of its 32-block, exponent bias folded into global_scale):
// see lsxcommon::nvfp4::E8m0ToE4m3Pow2Pair and PackExpertNVFP4. A tensor whose
// E8M0 byte spread exceeds the exact window falls back to GPU dequant +
// requantize.
//
// Verified against the released tree: every layer but the first carries
// `block_sparse_moe.experts.{0..895}.w{1,2,3}`; the vision tower and the
// projector are stripped and counted, never applied.
#include "lsxcommon/kimi_moe_k3_convert.h"

#include "kimi_moe_k3_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "nvfp4_scale.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <simdjson.h>
#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

namespace lsxcommon::kimi_k3 {
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

bool ends_with(const std::string &s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parses "model.layers.<N>." (after the `language_model.` scope) and returns
// the body after it.
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

// "block_sparse_moe.experts.<E>.<rest>" -> expert + rest.
bool parse_expert_body(const std::string &body, int &expert, std::string &rest)
{
    const std::string prefix = "block_sparse_moe.experts.";
    if (body.rfind(prefix, 0) != 0) return false;
    const size_t ds = prefix.size();
    const size_t dot = body.find('.', ds);
    if (dot == std::string::npos) return false;
    int v = 0;
    for (size_t i = ds; i < dot; ++i) {
        if (body[i] < '0' || body[i] > '9') return false;
        v = v * 10 + (body[i] - '0');
    }
    expert = v;
    rest = body.substr(dot + 1);
    return true;
}

bool tensor_is_f32(const TensorInfo &t) { return t.dtype == "F32"; }

bool check_frozen_geometry(const KimiK3ConvertTopology &top, std::string &why)
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
    if (top.vocab_size != kVocab)
        return bad("vocab_size", top.vocab_size, kVocab);
    if (top.dense_layers != kDenseLayers)
        return bad("first_k_dense_replace", top.dense_layers, kDenseLayers);
    if (top.attn_res_block != kAttnResBlock)
        return bad("attn_res_block_size", top.attn_res_block, kAttnResBlock);
    if (top.mla_heads != kMlaHeads)
        return bad("num_attention_heads", top.mla_heads, kMlaHeads);
    if (top.q_lora != kQLoraRank)
        return bad("q_lora_rank", top.q_lora, kQLoraRank);
    if (top.kv_lora != kKvLoraRank)
        return bad("kv_lora_rank", top.kv_lora, kKvLoraRank);
    if (top.qk_nope != kQkNopeHeadDim)
        return bad("qk_nope_head_dim", top.qk_nope, kQkNopeHeadDim);
    if (top.qk_rope != kQkRopeHeadDim)
        return bad("qk_rope_head_dim", top.qk_rope, kQkRopeHeadDim);
    if (top.v_head != kVHeadDim)
        return bad("v_head_dim", top.v_head, kVHeadDim);
    if (top.kda_heads != kKdaHeads)
        return bad("linear_attn.num_heads", top.kda_heads, kKdaHeads);
    if (top.kda_head_dim != kKdaHeadDim)
        return bad("linear_attn.head_dim", top.kda_head_dim, kKdaHeadDim);
    if (top.conv_kernel != kKdaConvKernel)
        return bad("short_conv_kernel_size", top.conv_kernel, kKdaConvKernel);
    if (top.num_experts != kExperts)
        return bad("num_experts", top.num_experts, kExperts);
    if (top.top_k != kTopK)
        return bad("num_experts_per_token", top.top_k, kTopK);
    if (top.moe_intermediate != kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate,
                   kMoeIntermediate);
    if (top.latent_dim != kLatentDim)
        return bad("routed_expert_hidden_size", top.latent_dim, kLatentDim);
    if (top.shared_experts != kSharedExperts)
        return bad("num_shared_experts", top.shared_experts, kSharedExperts);
    if (top.shared_intermediate != kSharedIntermediate)
        return bad("shared intermediate", top.shared_intermediate,
                   kSharedIntermediate);
    const int mla = static_cast<int>(top.mla_layers.size());
    if (mla != kMlaLayerCount)
        return bad("mla layer count", mla, kMlaLayerCount);
    for (int l : frozen::kMlaLayers)
        if (std::find(top.mla_layers.begin(), top.mla_layers.end(), l) ==
            top.mla_layers.end())
            return bad("mla layer index", l, l);
    return true;
}

}  // namespace

ParsedKimiK3Tensor ClassifyKimiK3Tensor(const std::string &full_name)
{
    ParsedKimiK3Tensor r;
    std::string name = full_name;
    const std::string scope = "language_model.";
    if (name.rfind(scope, 0) == 0) name = name.substr(scope.size());

    if (name.rfind("vision_tower.", 0) == 0) {
        r.kind = KimiK3TensorKind::VisionTower;
        return r;
    }
    if (name.rfind("mm_projector.", 0) == 0) {
        r.kind = KimiK3TensorKind::Projector;
        return r;
    }
    if (name == "model.embed_tokens.weight") {
        r.kind = KimiK3TensorKind::Embed;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = KimiK3TensorKind::LmHead;
        return r;
    }
    if (name == "model.norm.weight") {
        r.kind = KimiK3TensorKind::FinalNorm;
        return r;
    }
    if (name == "model.output_attn_res_proj.weight") {
        r.kind = KimiK3TensorKind::OutputAttnResProj;
        return r;
    }
    if (name == "model.output_attn_res_norm.weight") {
        r.kind = KimiK3TensorKind::OutputAttnResNorm;
        return r;
    }

    int layer = -1;
    std::string body;
    if (!parse_layer_body(name, layer, body)) return r;
    r.layer = layer;

    // Routed experts (per-expert files in the released layout).
    int expert = -1;
    std::string rest;
    if (parse_expert_body(body, expert, rest)) {
        r.expert = expert;
        using K = KimiK3TensorKind;
        static const std::pair<const char *, K> kExpertMap[] = {
            {"w1.weight_packed", K::ExpertW1Packed},
            {"w1.weight_scale", K::ExpertW1Scale},
            {"w2.weight_packed", K::ExpertW2Packed},
            {"w2.weight_scale", K::ExpertW2Scale},
            {"w3.weight_packed", K::ExpertW3Packed},
            {"w3.weight_scale", K::ExpertW3Scale},
        };
        for (const auto &[suffix, kind] : kExpertMap)
            if (rest == suffix) {
                r.kind = kind;
                return r;
            }
        return r;  // unknown expert tensor -> Unknown (converter errors)
    }

    using K = KimiK3TensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attention_res_proj.weight", K::AttnResProj},
        {"self_attention_res_norm.weight", K::AttnResNorm},
        {"mlp_res_proj.weight", K::MlpResProj},
        {"mlp_res_norm.weight", K::MlpResNorm},
        {"self_attn.q_a_proj.weight", K::MlaQA},
        {"self_attn.q_a_layernorm.weight", K::MlaQANorm},
        {"self_attn.q_b_proj.weight", K::MlaQB},
        {"self_attn.kv_a_proj_with_mqa.weight", K::MlaKvA},
        {"self_attn.kv_a_layernorm.weight", K::MlaKvANorm},
        {"self_attn.kv_b_proj.weight", K::MlaKvB},
        {"self_attn.o_proj.weight", K::MlaO},
        {"self_attn.g_proj.weight", K::MlaG},
        {"self_attn.q_proj.weight", K::KdaQ},
        {"self_attn.k_proj.weight", K::KdaK},
        {"self_attn.v_proj.weight", K::KdaV},
        {"self_attn.f_a_proj.weight", K::KdaFA},
        {"self_attn.f_b_proj.weight", K::KdaFB},
        {"self_attn.b_proj.weight", K::KdaB},
        {"self_attn.q_conv1d.weight", K::KdaQConv},
        {"self_attn.k_conv1d.weight", K::KdaKConv},
        {"self_attn.v_conv1d.weight", K::KdaVConv},
        {"self_attn.A_log", K::KdaALog},
        {"self_attn.dt_bias", K::KdaDt},
        {"self_attn.o_norm.weight", K::KdaONorm},
        {"mlp.gate_proj.weight", K::DenseGate},
        {"mlp.up_proj.weight", K::DenseUp},
        {"mlp.down_proj.weight", K::DenseDown},
        {"block_sparse_moe.gate.weight", K::RouterGate},
        {"block_sparse_moe.gate.e_score_correction_bias", K::RouterBias},
        {"block_sparse_moe.routed_expert_down_proj.weight", K::LatentDown},
        {"block_sparse_moe.routed_expert_up_proj.weight", K::LatentUp},
        {"block_sparse_moe.routed_expert_norm.weight", K::LatentNorm},
        {"block_sparse_moe.shared_experts.gate_proj.weight", K::SharedGate},
        {"block_sparse_moe.shared_experts.up_proj.weight", K::SharedUp},
        {"block_sparse_moe.shared_experts.down_proj.weight", K::SharedDown},
    };
    for (const auto &[suffix, kind] : kMap)
        if (body == suffix) {
            r.kind = kind;
            return r;
        }

    // o_proj / g_proj are shared names between the two attention kinds; the
    // config layer table decides. The classifier cannot see the config, so it
    // reports the KDA kind here and ConvertKimiK3Model re-maps it: the
    // projection shapes are identical, and the loader keys off the layer table.
    if (body == "self_attn.o_proj.weight") {
        r.kind = KimiK3TensorKind::KdaO;
        return r;
    }
    if (body == "self_attn.g_proj.weight") {
        r.kind = KimiK3TensorKind::KdaG;
        return r;
    }
    return r;
}

bool KimiK3TensorIsPinnedNvfp4(KimiK3TensorKind kind)
{
    switch (kind) {
    case KimiK3TensorKind::Embed:
    case KimiK3TensorKind::LmHead:
    case KimiK3TensorKind::MlaQA:
    case KimiK3TensorKind::MlaQB:
    case KimiK3TensorKind::MlaKvA:
    case KimiK3TensorKind::MlaKvB:
    case KimiK3TensorKind::MlaO:
    case KimiK3TensorKind::MlaG:
    case KimiK3TensorKind::KdaQ:
    case KimiK3TensorKind::KdaK:
    case KimiK3TensorKind::KdaV:
    case KimiK3TensorKind::KdaO:
    case KimiK3TensorKind::KdaG:
    case KimiK3TensorKind::DenseGate:
    case KimiK3TensorKind::DenseUp:
    case KimiK3TensorKind::DenseDown:
        return true;
    default:
        return false;
    }
}

bool KimiK3TensorIsStreamed(KimiK3TensorKind kind)
{
    switch (kind) {
    case KimiK3TensorKind::LatentDown:
    case KimiK3TensorKind::LatentUp:
    case KimiK3TensorKind::SharedGate:
    case KimiK3TensorKind::SharedUp:
    case KimiK3TensorKind::SharedDown:
        return true;
    default:
        return false;
    }
}

KimiK3ConvertTopology ParseKimiK3Config(const std::string &input_dir)
{
    KimiK3ConvertTopology top;
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

    // Text tower lives under text_config in the multimodal wrapper.
    simdjson::ondemand::object tc;
    if (root["text_config"].get_object().get(tc) != simdjson::SUCCESS) {
        top.error_msg = "config.json missing text_config";
        return top;
    }

    top.hidden_size = get_i(tc, "hidden_size", 0);
    top.num_layers = get_i(tc, "num_hidden_layers", 0);
    top.vocab_size = get_i(tc, "vocab_size", 0);
    top.dense_layers = get_i(tc, "first_k_dense_replace", 1);
    top.attn_res_block = get_i(tc, "attn_res_block_size", 0);
    top.dense_intermediate = get_i(tc, "intermediate_size", 0);
    top.mla_heads = get_i(tc, "num_attention_heads", 0);
    top.kv_heads = get_i(tc, "num_key_value_heads", top.mla_heads);
    top.q_lora = get_i(tc, "q_lora_rank", 0);
    top.kv_lora = get_i(tc, "kv_lora_rank", 0);
    top.qk_nope = get_i(tc, "qk_nope_head_dim", 0);
    top.qk_rope = get_i(tc, "qk_rope_head_dim", 0);
    top.v_head = get_i(tc, "v_head_dim", 0);
    top.mla_nope = get_b(tc, "mla_use_nope", true);
    top.mla_output_gate = get_b(tc, "mla_use_output_gate", true);
    top.num_experts = get_i(tc, "num_experts", 0);
    top.top_k = get_i(tc, "num_experts_per_token", 0);
    top.moe_intermediate = get_i(tc, "moe_intermediate_size", 0);
    top.latent_dim = get_i(tc, "routed_expert_hidden_size", 0);
    top.latent_norm = get_b(tc, "latent_moe_use_norm", true);
    top.shared_experts = get_i(tc, "num_shared_experts", 0);
    top.moe_renormalize = get_b(tc, "moe_renormalize", true);
    top.expert_group = get_i(tc, "num_expert_group", 1);
    top.topk_group = get_i(tc, "topk_group", 1);
    top.route_scale = get_d(tc, "routed_scaling_factor", 1.0);
    top.rms_norm_eps = get_d(tc, "rms_norm_eps", 1e-5);
    top.situ_beta = get_d(tc, "activation_situ_beta", 1.0);
    top.situ_linear_beta = get_d(tc, "activation_situ_linear_beta", 0.0);
    // ondemand is forward-only: the top-level ids were read in a separate
    // pass BEFORE text_config was opened (see the pass at the end).

    simdjson::ondemand::object lac;
    if (tc["linear_attn_config"].get_object().get(lac) != simdjson::SUCCESS) {
        top.error_msg = "config.json missing linear_attn_config";
        return top;
    }
    top.kda_heads = get_i(lac, "num_heads", 0);
    top.kda_head_dim = get_i(lac, "head_dim", 0);
    top.conv_kernel = get_i(lac, "short_conv_kernel_size", 4);
    top.kda_full_rank_gate = get_b(lac, "use_full_rank_gate", true);
    top.kda_gate_lower_bound = get_d(lac, "gate_lower_bound", -5.0);

    // 1-based full-attention list in the checkpoint -> 0-based tables.
    std::vector<int> full_1based;
    for (auto v : lac["full_attn_layers"].get_array()) {
        int64_t x = 0;
        if (v.get_int64().get(x) == simdjson::SUCCESS)
            full_1based.push_back(static_cast<int>(x));
    }
    if (full_1based.empty()) {
        top.error_msg = "linear_attn_config.full_attn_layers is empty";
        return top;
    }
    for (int l = 0; l < top.num_layers; ++l) {
        const bool full = std::find(full_1based.begin(), full_1based.end(),
                                    l + 1) != full_1based.end();
        (full ? top.mla_layers : top.kda_layers).push_back(l);
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.top_k <= 0 || top.moe_intermediate <= 0 || top.latent_dim <= 0 ||
        top.mla_heads <= 0 || top.kda_heads <= 0) {
        top.error_msg = "config.json missing required fields";
        return top;
    }
    if (top.shared_intermediate == 0)
        top.shared_intermediate = top.moe_intermediate * top.shared_experts;

    // Second pass for the wrapper-level ids (forward-only ondemand cursor).
    {
        simdjson::ondemand::parser p2;
        simdjson::padded_string pp(json.data(), json.size());
        simdjson::ondemand::document d2;
        if (p2.iterate(pp).get(d2) == simdjson::SUCCESS) {
            simdjson::ondemand::object r2;
            if (d2.get_object().get(r2) == simdjson::SUCCESS) {
                const int outer_bos = get_i(r2, "bos_token_id", -1);
                const int outer_eos = get_i(r2, "eos_token_id", -1);
                const int outer_pad = get_i(r2, "pad_token_id", -1);
                top.bos_id = outer_bos;
                top.eos_id = outer_eos;
                top.pad_id = outer_pad;
            }
        }
    }
    return top;
}

arrow::Status ConvertKimiK3Model(const std::string &input_dir,
                                 const std::string &output_root)
{
    KimiK3ConvertTopology top = ParseKimiK3Config(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Kimi-K3 config error: ", top.error_msg);

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid("Kimi-K3 frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Classify everything first: a leftover tensor is a hard error inside the
    // main pass (an unknown name must never be skipped silently). The vision
    // tower / projector are the only intentional omissions and are counted.
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            auto p = ClassifyKimiK3Tensor(t.name);
            if (p.kind == KimiK3TensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in Kimi-K3 checkpoint: ", t.name);
        }

    // NOTE: integrity::VerifyAllTensorReads is deliberately not used here — it
    // would read the whole 1.5 TB tree a second time. Every read below is
    // length-checked against the safetensors header instead.

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path streamed_dir = base / "streamed";
    const fs::path experts_dir = base / "experts";
    for (const fs::path &d : {base, pinned_dir, streamed_dir, experts_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> pinned_nvfp4;
    absl::btree_map<int, std::vector<TensorRecord>> pinned_raw;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> streamed_nvfp4;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int64_t pinned_count = 0;
    int64_t streamed_count = 0;
    int64_t experts_written = 0;

    // Per-layer expert buckets: kind -> {layer, expert, TensorInfo*}.
    struct ExpertPart {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *packed = nullptr;
        const TensorInfo *scale = nullptr;
    };
    // (layer, expert, weight 1..3) -> parts
    absl::btree_map<std::tuple<int, int, int>, ExpertPart> expert_parts;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = ClassifyKimiK3Tensor(t.name);

            if (p.kind == KimiK3TensorKind::VisionTower) {
                ++top.skipped_visual_tensors;
                continue;
            }
            if (p.kind == KimiK3TensorKind::Projector) {
                ++top.skipped_projector_tensors;
                continue;
            }

            // Expert tensors: indexed now, written in a per-expert pass so the
            // converter never holds a whole layer's expert bytes.
            if (p.expert >= 0) {
                int w = 0;
                bool is_scale = false;
                switch (p.kind) {
                case KimiK3TensorKind::ExpertW1Packed: w = 1; break;
                case KimiK3TensorKind::ExpertW1Scale: w = 1; is_scale = true; break;
                case KimiK3TensorKind::ExpertW2Packed: w = 2; break;
                case KimiK3TensorKind::ExpertW2Scale: w = 2; is_scale = true; break;
                case KimiK3TensorKind::ExpertW3Packed: w = 3; break;
                case KimiK3TensorKind::ExpertW3Scale: w = 3; is_scale = true; break;
                default:
                    return arrow::Status::Invalid("unhandled expert tensor: ",
                                                  t.name);
                }
                if (p.layer < top.dense_layers)
                    return arrow::Status::Invalid(
                        "expert tensor on a dense layer: ", t.name);
                auto &slot = expert_parts[{p.layer, p.expert, w}];
                slot.shard = &shard;
                (is_scale ? slot.scale : slot.packed) = &t;
                continue;
            }

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            // o_proj/g_proj are named identically on both attention kinds; the
            // layer table (not the name) says which. Shapes are the same, so
            // the only thing that matters is that neither is misrouted into the
            // streamed plane — they are pinned either way.
            if (p.kind == KimiK3TensorKind::Embed ||
                p.kind == KimiK3TensorKind::LmHead ||
                KimiK3TensorIsPinnedNvfp4(p.kind)) {
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!QuantizeSlice(data.data(), t.numel(),
                                   tensor_is_f32(t) ? "F32" : "BF16", r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                if (p.layer < 0)
                    global_nvfp4.push_back(std::move(r));
                else
                    pinned_nvfp4[p.layer].push_back(std::move(r));
                ++pinned_count;
                continue;
            }
            if (KimiK3TensorIsStreamed(p.kind)) {
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!QuantizeSlice(data.data(), t.numel(),
                                   tensor_is_f32(t) ? "F32" : "BF16", r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                streamed_nvfp4[p.layer].push_back(std::move(r));
                ++streamed_count;
                continue;
            }

            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(data);
            if (p.layer < 0)
                global_raw.push_back(std::move(r));
            else
                pinned_raw[p.layer].push_back(std::move(r));
            ++pinned_count;
        }
    }

    // ── Routed-expert pass: one Arrow file per (layer, expert) ──
    for (int l = top.dense_layers; l < top.num_layers; ++l) {
        const fs::path dir = experts_dir / absl::StrCat("layer_", l);
        std::error_code ec;
        fs::create_directories(dir, ec);
        for (int e = 0; e < top.num_experts; ++e) {
            ExpertData raw;
            for (int w = 1; w <= 3; ++w) {
                auto it = expert_parts.find({l, e, w});
                if (it == expert_parts.end() || !it->second.packed ||
                    !it->second.scale)
                    return arrow::Status::Invalid(
                        "incomplete expert: layer ", l, " expert ", e,
                        " weight ", w);
                const auto &part = it->second;
                auto packed_bytes =
                    ReadTensorData(input_dir, *part.shard, *part.packed);
                auto scale_bytes =
                    ReadTensorData(input_dir, *part.shard, *part.scale);
                if (packed_bytes.empty() || scale_bytes.empty())
                    return arrow::Status::IOError("failed to read expert ",
                                                  part.packed->name);
                // Logical [rows, cols]: packed is [rows, cols/2], scale is
                // [rows, cols/32].
                const int64_t rows = part.packed->shape[0];
                const int64_t cols = part.packed->shape[1] * 2;
                if (part.scale->shape[0] != rows ||
                    part.scale->shape[1] * 32 != cols)
                    return arrow::Status::Invalid(
                        "expert scale shape mismatch: ", part.scale->name);
                std::vector<int64_t> shape{rows, cols};
                std::vector<int64_t> scale_shape{part.scale->shape[0],
                                                 part.scale->shape[1]};
                switch (w) {
                case 1:
                    raw.w1_raw = std::move(packed_bytes);
                    raw.w1_scale_raw = std::move(scale_bytes);
                    raw.w1_shape = shape;
                    raw.w1_scale_shape = scale_shape;
                    raw.w1_dtype = part.packed->dtype;
                    break;
                case 2:
                    raw.w2_raw = std::move(packed_bytes);
                    raw.w2_scale_raw = std::move(scale_bytes);
                    raw.w2_shape = shape;
                    raw.w2_scale_shape = scale_shape;
                    raw.w2_dtype = part.packed->dtype;
                    break;
                default:
                    raw.w3_raw = std::move(packed_bytes);
                    raw.w3_scale_raw = std::move(scale_bytes);
                    raw.w3_shape = shape;
                    raw.w3_scale_shape = scale_shape;
                    raw.w3_dtype = part.packed->dtype;
                    break;
                }
            }
            auto pe = PackExpertNVFP4(raw);
            if (!pe.ok())
                return arrow::Status::ExecutionError(
                    "expert NVFP4 repack failed: layer ", l, " expert ", e,
                    ": ", pe.status().ToString());
            auto batch = ExpertSchema::MakeBatch(*pe);
            if (!batch.ok()) return batch.status();
            ARROW_RETURN_NOT_OK(
                WriteRecordBatchToFile(*batch, dir / absl::StrCat("exp_", e,
                                                                 ".arrow")));
            ++experts_written;
        }
    }

    // ── Write the pinned / streamed / global batches ──
    if (!global_nvfp4.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                                        pinned_dir / "global_nvfp4.arrow"));
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : pinned_nvfp4)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : pinned_raw)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : streamed_nvfp4)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            streamed_dir / absl::StrCat("layer_", layer, ".arrow")));

    top.total_pinned_tensors = static_cast<int>(pinned_count);

    // ── Tokenizer (mandatory for a real checkpoint; synthetic fixtures omit) ──
    const fs::path tk_path = fs::path(input_dir) / "tiktoken.model";
    bool tokenizer_ok = false;
    if (fs::exists(tk_path)) {
        TikTokenSpecialPolicy policy;
        // The reference tokenizer never prepends [BOS] (encoding_k3.py builds
        // XTML segments only); the stop id is <|end_of_msg|> = config eos.
        policy.bos_id = -1;
        policy.eos_id = top.eos_id;
        policy.pad_id = top.pad_id;
        ARROW_RETURN_NOT_OK(BuildTokenizerArrowFilesTikToken(
            tk_path.string(),
            (fs::path(input_dir) / "tokenizer_config.json").string(),
            pinned_dir, policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tiktoken.model not found in ", input_dir);
    }

    // ── VRAM profile ──
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.kv_lora + top.qk_rope;
    vg.num_attn_layers = static_cast<int>(top.mla_layers.size());
    vg.num_moe_layers = top.num_layers - top.dense_layers;
    vg.experts_per_layer = top.num_experts;
    vg.context_length = ReadContextLength(
        (fs::path(input_dir) / "config.json").string());

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
    auto nvfp4_policy = +[](const std::string &name) {
        const auto p = ClassifyKimiK3Tensor(name);
        if (p.expert >= 0) return true;
        return KimiK3TensorIsPinnedNvfp4(p.kind) ||
               KimiK3TensorIsStreamed(p.kind);
    };
    // The streamed latent/shared plane is NOT device-resident: it is mmap'd
    // and copied per layer per step, so it belongs in the streamed class
    // alongside the routed experts (mirroring the loader's residency split).
    auto mem_class = +[](const std::string &name) {
        const auto p = ClassifyKimiK3Tensor(name);
        return (p.expert >= 0 || KimiK3TensorIsStreamed(p.kind))
                   ? VramTensorClass::RoutedExpert
                   : VramTensorClass::Resident;
    };
    VramProfile vram = EstimateVram(
        vram_inventory,
        {.is_nvfp4 = nvfp4_policy, .mem_class = mem_class, .has_residual = nullptr},
        vg);

    // ── model.index.json ──
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"" << frozen::kFormatTag << "\",\n"
          << "  \"model\": \"Kimi-K3\",\n"
          << "  \"model_type\": \"" << frozen::kModelType << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n"
          << "  \"num_dense_layers\": " << top.dense_layers << ",\n"
          << "  \"hidden_size\": " << top.hidden_size << ",\n"
          << "  \"intermediate_size\": " << top.dense_intermediate << ",\n"
          << "  \"vocab_size\": " << top.vocab_size << ",\n"
          << "  \"attn_res_block_size\": " << top.attn_res_block << ",\n"
          << "  \"mla_layers\": [";
        for (size_t i = 0; i < top.mla_layers.size(); ++i)
            j << (i ? "," : "") << top.mla_layers[i];
        j << "],\n  \"kda_layers\": [";
        for (size_t i = 0; i < top.kda_layers.size(); ++i)
            j << (i ? "," : "") << top.kda_layers[i];
        j << "],\n"
          << "  \"mla_heads\": " << top.mla_heads << ",\n"
          << "  \"mla_kv_heads\": " << top.kv_heads << ",\n"
          << "  \"q_lora_rank\": " << top.q_lora << ",\n"
          << "  \"kv_lora_rank\": " << top.kv_lora << ",\n"
          << "  \"qk_nope_head_dim\": " << top.qk_nope << ",\n"
          << "  \"qk_rope_head_dim\": " << top.qk_rope << ",\n"
          << "  \"v_head_dim\": " << top.v_head << ",\n"
          << "  \"mla_use_nope\": " << (top.mla_nope ? "true" : "false")
          << ",\n"
          << "  \"mla_use_output_gate\": "
          << (top.mla_output_gate ? "true" : "false") << ",\n"
          << "  \"kda_heads\": " << top.kda_heads << ",\n"
          << "  \"kda_head_dim\": " << top.kda_head_dim << ",\n"
          << "  \"kda_conv_kernel\": " << top.conv_kernel << ",\n"
          << "  \"kda_full_rank_gate\": "
          << (top.kda_full_rank_gate ? "true" : "false") << ",\n"
          << "  \"kda_gate_lower_bound\": " << top.kda_gate_lower_bound
          << ",\n"
          << "  \"num_experts\": " << top.num_experts << ",\n"
          << "  \"num_experts_per_tok\": " << top.top_k << ",\n"
          << "  \"experts_per_layer\": " << top.num_experts << ",\n"
          << "  \"moe_intermediate_size\": " << top.moe_intermediate << ",\n"
          << "  \"routed_expert_hidden_size\": " << top.latent_dim << ",\n"
          << "  \"latent_moe_use_norm\": "
          << (top.latent_norm ? "true" : "false") << ",\n"
          << "  \"num_shared_experts\": " << top.shared_experts << ",\n"
          << "  \"shared_intermediate_size\": " << top.shared_intermediate
          << ",\n"
          << "  \"moe_renormalize\": "
          << (top.moe_renormalize ? "true" : "false") << ",\n"
          << "  \"num_expert_group\": " << top.expert_group << ",\n"
          << "  \"topk_group\": " << top.topk_group << ",\n"
          << "  \"routed_scaling_factor\": " << top.route_scale << ",\n"
          << "  \"rms_norm_eps\": " << top.rms_norm_eps << ",\n"
          << "  \"activation_situ_beta\": " << top.situ_beta << ",\n"
          << "  \"activation_situ_linear_beta\": " << top.situ_linear_beta
          << ",\n"
          << "  \"bos_id\": " << top.bos_id << ",\n"
          << "  \"eos_id\": " << top.eos_id << ",\n"
          << "  \"pad_id\": " << top.pad_id << ",\n"
          << "  \"tokenizer_kind\": " << frozen::kTokenizerKind << ",\n"
          << "  \"tokenizer\": \"" << (tokenizer_ok ? "tiktoken" : "fixture")
          << "\",\n"
          << "  \"streamed_dense_plane\": true,\n"
          << "  \"skipped_visual_tensors\": " << top.skipped_visual_tensors
          << ",\n"
          << "  \"skipped_projector_tensors\": "
          << top.skipped_projector_tensors << ",\n"
          << "  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n"
          << "  \"total_streamed_tensors\": " << streamed_count << ",\n"
          << "  \"total_experts\": " << experts_written << ",\n"
          << FormatVramJson(vram) << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    lsxcommon::log::info(absl::StrFormat(
        "Kimi-K3 conversion: %d layers (%zu MLA / %zu KDA), %d experts x %d "
        "layers, %lld pinned, %lld streamed, stripped %d vision + %d "
        "projector tensors",
        top.num_layers, top.mla_layers.size(), top.kda_layers.size(),
        top.num_experts, top.num_layers - top.dense_layers,
        static_cast<long long>(pinned_count),
        static_cast<long long>(streamed_count), top.skipped_visual_tensors,
        top.skipped_projector_tensors));
    return arrow::Status::OK();
}

}  // namespace lsxcommon::kimi_k3
