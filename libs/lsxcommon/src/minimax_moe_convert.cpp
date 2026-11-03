#include "lsxcommon/minimax_moe_convert.h"

#include "minimax_moe_constants.h"

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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

namespace lsxcommon::minimax {
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

// Cross-checks the parsed config against the frozen passport. Returns false
// and fills `why` on mismatch (synthetic fixtures bypass this upstream).
bool CheckFrozenGeometry(const MinimaxTopology &top, std::string &why)
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
    if (top.rotary_dim != kRotaryDim) return bad("rotary_dim", top.rotary_dim, kRotaryDim);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("num_local_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.shared_intermediate_size != kSharedIntermediate)
        return bad("shared_intermediate_size", top.shared_intermediate_size,
                   kSharedIntermediate);
    if (top.dense_intermediate_size != kDenseIntermediate)
        return bad("dense_intermediate_size", top.dense_intermediate_size,
                   kDenseIntermediate);
    if (top.index_n_heads != kIndexNHeads)
        return bad("sparse_num_index_heads", top.index_n_heads, kIndexNHeads);
    if (top.index_head_dim != kIndexHeadDim)
        return bad("sparse_index_dim", top.index_head_dim, kIndexHeadDim);
    if (top.index_block_size != kIndexBlockSize)
        return bad("sparse_block_size", top.index_block_size, kIndexBlockSize);
    if (top.index_topk_blocks != kIndexTopkBlocks)
        return bad("sparse_topk_blocks", top.index_topk_blocks, kIndexTopkBlocks);
    if (top.index_local_blocks != kIndexLocalBlocks)
        return bad("sparse_local_block", top.index_local_blocks, kIndexLocalBlocks);
    // MLP dispatch is shared by every family member (three dense-MLP layers).
    // The ATTENTION dispatch is NOT: M3 runs full attention on layers 0-2,
    // M3.1-Flash runs MSA on every layer. It is validated as a monotone
    // prefix in ParseMinimaxConfig; the variant discriminator (and each
    // model's frozen passport) pins which member it is.
    for (int i = 0; i < top.num_layers; ++i) {
        const size_t ui = static_cast<size_t>(i);
        const char *want_mlp = IsMoeLayer(i) ? "sparse" : "dense";
        if (top.mlp_types[ui] != want_mlp)
            return bad("mlp_layer_types[i]", i, 0);
    }
    return true;
}

// Quantizes a contiguous BF16/FP32/F8_E4M3 slice to NVFP4. An E8M0 scale
// tile (`e8m0` non-null) makes the shared quantizer dequantize the payload
// with `float * 2^(u-127)` per tile first (MXFP8 -> NVFP4). Returns false on
// GPU failure.
// Reads a light JSON file's "text_config"-less top level for token ids.
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

MinimaxTopology ParseMinimaxConfig(const std::string &input_dir)
{
    MinimaxTopology top;
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
    top.rotary_dim = get_i(src, "rotary_dim", -1);
    top.dense_intermediate_size = get_i(src, "dense_intermediate_size", 0);
    top.moe_intermediate_size = get_i(src, "intermediate_size", 0);
    top.shared_intermediate_size = get_i(src, "shared_intermediate_size", 0);
    top.num_experts = get_i(src, "num_local_experts", 0);
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-6);
    top.rope_theta = get_d(src, "rope_theta", 5000000.0);
    top.routed_scaling_factor = get_d(src, "routed_scaling_factor", 2.0);
    top.swiglu_alpha = get_d(src, "swiglu_alpha", 1.702);
    top.swiglu_limit = get_d(src, "swiglu_limit", 7.0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);
    top.bos_token_id = get_i(src, "bos_token_id", -1);
    top.pad_token_id = get_i(src, "pad_token_id", -1);
    // rotary_dim falls back to head_dim * partial_rotary_factor.
    if (top.rotary_dim < 0) {
        const double prf = get_d(src, "partial_rotary_factor", 1.0);
        top.rotary_dim = static_cast<int>(top.head_dim * prf);
    }

    // MiniMax Sparse Attention block (legacy flat names).
    if (auto sac = src["sparse_attention_config"].get_object();
        sac.error() == simdjson::SUCCESS) {
        auto s = sac.value();
        top.index_n_heads = get_i(s, "sparse_num_index_heads", 0);
        top.index_head_dim = get_i(s, "sparse_index_dim", 0);
        top.index_block_size = get_i(s, "sparse_block_size", 0);
        top.index_topk_blocks = get_i(s, "sparse_topk_blocks", 0);
        top.index_local_blocks = get_i(s, "sparse_local_block", 1);
        if (auto freq = s["sparse_attention_freq"].get_array();
            freq.error() == simdjson::SUCCESS) {
            for (auto e : freq.value()) {
                int64_t v = 0;
                if (e.get_int64().get(v) != simdjson::SUCCESS) continue;
                top.layer_types.push_back(v ? "sparse_attention"
                                            : "dense_attention");
            }
        }
    }
    if (auto moe = src["moe_layer_freq"].get_array();
        moe.error() == simdjson::SUCCESS) {
        for (auto e : moe.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) != simdjson::SUCCESS) continue;
            top.mlp_types.push_back(v ? "sparse" : "dense");
        }
    }

    // Token ids: text_config first, then generation_config.json (this
    // checkpoint puts them there, NOT in text_config).
    if (top.bos_token_id < 0 || top.eos_token_id < 0 || top.pad_token_id < 0)
        ReadGenerationIds(fs::path(input_dir) / "generation_config.json",
                          top.bos_token_id, top.eos_token_id,
                          top.pad_token_id);

    if (top.hidden_size <= 0 || top.num_layers <= 0 ||
        top.num_experts <= 0 || top.num_experts_per_tok <= 0 ||
        top.moe_intermediate_size <= 0 || top.head_dim <= 0 ||
        top.index_n_heads <= 0 || top.index_block_size <= 0) {
        top.error_msg = "config.json missing required MiniMax-M3 fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("sparse_attention_freq size ",
                                     top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    if (static_cast<int>(top.mlp_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("moe_layer_freq size ",
                                     top.mlp_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }

    // Dispatch arrays are a monotone prefix: dense layers first, then the
    // sparse/MoE kind. Record the boundaries and reject an interleaved pattern
    // (a single `layer_types` assumption silently decodes garbage).
    auto first_of = [](const std::vector<std::string> &v,
                       const char *dense) -> int {
        int first = static_cast<int>(v.size());
        bool seen_sparse = false;
        for (size_t i = 0; i < v.size(); ++i) {
            const bool is_dense = v[i] == dense;
            if (!is_dense && !seen_sparse) {
                first = static_cast<int>(i);
                seen_sparse = true;
            } else if (is_dense && seen_sparse) {
                return -1;  // dense after sparse: not a prefix pattern
            }
        }
        return first;
    };
    top.first_moe_layer = first_of(top.mlp_types, "dense");
    top.first_sparse_layer = first_of(top.layer_types, "dense_attention");
    if (top.first_moe_layer < 0 || top.first_sparse_layer < 0) {
        top.error_msg = "layer dispatch arrays are not a dense-prefix pattern";
        return top;
    }
    return top;
}

MinimaxVariant ClassifyMinimaxVariant(const MinimaxTopology &top)
{
    if (top.num_layers <= 0 ||
        static_cast<int>(top.layer_types.size()) != top.num_layers ||
        static_cast<int>(top.mlp_types.size()) != top.num_layers)
        return MinimaxVariant::kUnknown;
    const double theta = top.rope_theta;
    // M3: a dense-attention prefix (layers 0-2 in the real checkpoint),
    // theta 5e6.
    if (top.first_sparse_layer > 0 &&
        std::abs(theta - 5000000.0) <= 5000000.0 * 1e-6)
        return MinimaxVariant::kM3;
    // M3.1-Flash: MSA on every layer, theta 1e7.
    if (top.first_sparse_layer == 0 &&
        std::abs(theta - 10000000.0) <= 10000000.0 * 1e-6)
        return MinimaxVariant::kM3Flash;
    return MinimaxVariant::kUnknown;
}

ParsedMinimaxTensor classify_minimax_tensor(const std::string &name)
{
    ParsedMinimaxTensor r;
    if (name.rfind("vision_tower.", 0) == 0 ||
        name.rfind("multi_modal_projector.", 0) == 0 ||
        name.rfind("patch_merge_mlp.", 0) == 0 ||
        name.rfind("mtp.", 0) == 0) {
        r.kind = MinimaxTensorKind::Visual;
        return r;
    }
    if (name == "language_model.lm_head.weight") {
        r.kind = MinimaxTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "language_model.model.";
    if (name.rfind(kPrefix, 0) != 0) return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = MinimaxTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = MinimaxTensorKind::FinalNorm;
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
    const std::string kExpert = "block_sparse_moe.experts.";
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
        auto set = [&r](MinimaxTensorKind k, MinimaxSourceEncoding e) {
            r.kind = k;
            r.encoding = e;
        };
        using E = MinimaxSourceEncoding;
        if (leaf == "w1.weight") set(MinimaxTensorKind::ExpertW1, E::kRaw);
        else if (leaf == "w2.weight") set(MinimaxTensorKind::ExpertW2, E::kRaw);
        else if (leaf == "w3.weight") set(MinimaxTensorKind::ExpertW3, E::kRaw);
        else if (leaf == "w1.weight_packed")
            set(MinimaxTensorKind::ExpertW1, E::kExpertPacked);
        else if (leaf == "w2.weight_packed")
            set(MinimaxTensorKind::ExpertW2, E::kExpertPacked);
        else if (leaf == "w3.weight_packed")
            set(MinimaxTensorKind::ExpertW3, E::kExpertPacked);
        else if (leaf == "w1.weight_scale")
            set(MinimaxTensorKind::ExpertW1, E::kExpertScale);
        else if (leaf == "w2.weight_scale")
            set(MinimaxTensorKind::ExpertW2, E::kExpertScale);
        else if (leaf == "w3.weight_scale")
            set(MinimaxTensorKind::ExpertW3, E::kExpertScale);
        else if (leaf == "w1.weight_global_scale")
            set(MinimaxTensorKind::ExpertW1, E::kExpertGlobalScale);
        else if (leaf == "w2.weight_global_scale")
            set(MinimaxTensorKind::ExpertW2, E::kExpertGlobalScale);
        else if (leaf == "w3.weight_global_scale")
            set(MinimaxTensorKind::ExpertW3, E::kExpertGlobalScale);
        else if (leaf == "w1.input_global_scale")
            set(MinimaxTensorKind::ExpertW1, E::kExpertInputGlobalScale);
        else if (leaf == "w2.input_global_scale")
            set(MinimaxTensorKind::ExpertW2, E::kExpertInputGlobalScale);
        else if (leaf == "w3.input_global_scale")
            set(MinimaxTensorKind::ExpertW3, E::kExpertInputGlobalScale);
        return r;
    }

    using K = MinimaxTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::AttnQ},
        {"self_attn.k_proj.weight", K::AttnK},
        {"self_attn.v_proj.weight", K::AttnV},
        {"self_attn.o_proj.weight", K::AttnO},
        {"self_attn.q_norm.weight", K::AttnQNorm},
        {"self_attn.k_norm.weight", K::AttnKNorm},
        {"self_attn.index_q_proj.weight", K::IndexQ},
        {"self_attn.index_k_proj.weight", K::IndexK},
        {"self_attn.index_q_norm.weight", K::IndexQNorm},
        {"self_attn.index_k_norm.weight", K::IndexKNorm},
        {"mlp.gate_proj.weight", K::DenseMlpGate},
        {"mlp.up_proj.weight", K::DenseMlpUp},
        {"mlp.down_proj.weight", K::DenseMlpDown},
        {"block_sparse_moe.gate.weight", K::RouterGate},
        {"block_sparse_moe.e_score_correction_bias", K::RouterBias},
        {"block_sparse_moe.shared_experts.gate_proj.weight", K::SharedGateProj},
        {"block_sparse_moe.shared_experts.up_proj.weight", K::SharedUpProj},
        {"block_sparse_moe.shared_experts.down_proj.weight", K::SharedDownProj},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    // MXFP8 companion: `<base>.weight` + `<base>.weight_scale_inv` (per-32
    // E8M0 exponents). Classified as the base kind so the inventory/verifier
    // sees one logical tensor; the read loop consumes the pair together.
    static const std::string kScaleInv = ".weight_scale_inv";
    if (body.size() > kScaleInv.size() &&
        body.compare(body.size() - kScaleInv.size(), kScaleInv.size(),
                     kScaleInv) == 0) {
        const std::string base = body.substr(0, body.size() - kScaleInv.size()) +
                                 ".weight";
        for (const auto &[suffix, kind] : kMap) {
            if (base == suffix) {
                r.kind = kind;
                r.encoding = MinimaxSourceEncoding::kMxfp8ScaleInv;
                return r;
            }
        }
    }
    return r;
}

bool minimax_tensor_is_nvfp4(MinimaxTensorKind kind)
{
    switch (kind) {
    case MinimaxTensorKind::AttnQ:
    case MinimaxTensorKind::AttnK:
    case MinimaxTensorKind::AttnV:
    case MinimaxTensorKind::AttnO:
    case MinimaxTensorKind::IndexQ:
    case MinimaxTensorKind::IndexK:
    case MinimaxTensorKind::DenseMlpGate:
    case MinimaxTensorKind::DenseMlpUp:
    case MinimaxTensorKind::DenseMlpDown:
    case MinimaxTensorKind::ExpertW1:
    case MinimaxTensorKind::ExpertW2:
    case MinimaxTensorKind::ExpertW3:
    case MinimaxTensorKind::SharedGateProj:
    case MinimaxTensorKind::SharedUpProj:
    case MinimaxTensorKind::SharedDownProj:
        return true;
    default:
        return false;
    }
}

bool minimax_tensor_is_routed_expert(MinimaxTensorKind kind)
{
    return kind == MinimaxTensorKind::ExpertW1 ||
           kind == MinimaxTensorKind::ExpertW2 ||
           kind == MinimaxTensorKind::ExpertW3;
}

arrow::Status ConvertMinimaxMoeModel(const std::string &input_dir,
                                     const std::string &output_root)
{
    return ConvertMinimaxMoeModel(input_dir, output_root,
                                  MinimaxConvertVariant{});
}

arrow::Status ConvertMinimaxMoeModel(const std::string &input_dir,
                                     const std::string &output_root,
                                     const MinimaxConvertVariant &variant)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    MinimaxTopology top = ParseMinimaxConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("MiniMax config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_local_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, why))
            return arrow::Status::Invalid("MiniMax frozen geometry mismatch: ",
                                          why);
        const MinimaxVariant found = ClassifyMinimaxVariant(top);
        if (found != variant.variant)
            return arrow::Status::Invalid(
                "checkpoint does not match the requested model '",
                variant.model_type, "' (theta ", top.rope_theta,
                ", first sparse layer ", top.first_sparse_layer,
                ") — use the matching converter entry point");
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Classify the inventory up front: unknown names are hard errors, vision
    // ballast is counted, routed experts are indexed by name for the
    // expert-major second pass (they are ~98% of the bytes and must never be
    // held in host RAM as a whole).
    struct ExpertLoc {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    absl::btree_map<std::string, ExpertLoc> expert_index;
    // MXFP8 companions keyed by their BASE tensor name (the read loop consumes
    // the pair together).
    absl::btree_map<std::string, ExpertLoc> scale_inv_index;
    absl::btree_map<int, int> layer_kind_count;
    int visual_skipped = 0;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_minimax_tensor(t.name);
            if (p.kind == MinimaxTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == MinimaxTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in MiniMax checkpoint: ", t.name);
            if (p.encoding == MinimaxSourceEncoding::kMxfp8ScaleInv) {
                const std::string base_name =
                    t.name.substr(0, t.name.size() -
                                         std::string(".weight_scale_inv").size()) +
                    ".weight";
                if (!scale_inv_index.emplace(base_name,
                                             ExpertLoc{&shard, &t}).second)
                    return arrow::Status::Invalid(
                        "duplicate weight_scale_inv companion: ", t.name);
            }
            if (minimax_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.expert < 0)
                    return arrow::Status::Invalid("expert tensor without index: ",
                                                  t.name);
                if (top.mlp_types[static_cast<size_t>(p.layer)] != "sparse")
                    return arrow::Status::Invalid(
                        "expert tensor on a dense-MLP layer: ", t.name);
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
    // expert pass below, and a full pass would read the 800 GB tree twice.
    {
        SafetensorsModel resident_only;
        for (const auto &shard : model.shards) {
            SafetensorsModel::Shard s;
            s.filename = shard.filename;
            s.file_size = shard.file_size;
            for (const auto &t : shard.tensors) {
                auto p = classify_minimax_tensor(t.name);
                if (!minimax_tensor_is_routed_expert(p.kind))
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
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;

    auto add_nvfp4 = [&](const TensorInfo &t, std::vector<uint8_t> &data,
                         std::vector<DenseNvfp4Record> &dst) -> arrow::Status {
        DenseNvfp4Record r;
        r.name = t.name;
        r.shape = t.shape;
        r.dtype = t.dtype;
        // MXFP8 source: the payload is F8_E4M3 and the per-32 E8M0 exponents
        // ride in the `<base>.weight_scale_inv` companion. Pass them to the
        // shared quantizer (dtype 2 = E8M0); a BF16 source keeps the old path.
        const uint8_t *e8m0 = nullptr;
        std::vector<int64_t> e8m0_shape;
        std::vector<uint8_t> scale_bytes;
        auto it = scale_inv_index.find(t.name);
        if (it != scale_inv_index.end()) {
            if (t.dtype != "F8_E4M3")
                return arrow::Status::Invalid(
                    "weight_scale_inv companion on a non-F8_E4M3 tensor: ",
                    t.name);
            scale_bytes = ReadTensorData(input_dir, *it->second.shard,
                                         *it->second.info);
            if (scale_bytes.empty() ||
                static_cast<int64_t>(scale_bytes.size()) !=
                    it->second.info->byte_size())
                return arrow::Status::IOError("failed to read MXFP8 scales: ",
                                              it->second.info->name);
            const auto &ss = it->second.info->shape;
            if (t.shape.size() != 2 || ss.size() != 2 ||
                ss[0] != t.shape[0] || ss[1] * 32 != t.shape[1])
                return arrow::Status::Invalid(
                    "MXFP8 scale grid is not [rows, cols/32]: ",
                    it->second.info->name);
            e8m0 = scale_bytes.data();
            e8m0_shape = ss;
        }
        if (!QuantizeSlice(data.data(), t.numel(), t.dtype, e8m0, t.shape,
                           e8m0_shape, r))
            return arrow::Status::IOError("GPU quantization failed for ", t.name);
        dst.push_back(std::move(r));
        ++total_pinned;
        return arrow::Status::OK();
    };

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_minimax_tensor(t.name);
            if (p.kind == MinimaxTensorKind::Visual ||
                p.encoding == MinimaxSourceEncoding::kMxfp8ScaleInv ||
                p.encoding == MinimaxSourceEncoding::kExpertInputGlobalScale ||
                minimax_tensor_is_routed_expert(p.kind))
                continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            switch (p.kind) {
            case MinimaxTensorKind::Embed:
            case MinimaxTensorKind::LmHead: {
                // FP16 on the device: the head multiplies every logit, so a
                // 9% row error moves a logit by ~3 at this model's logit scale
                // (~38) while the first-token margin is 0.4. Same rule as
                // phi-4 / gemma-4: no standard 4-bit recipe quantizes the
                // embedding and the head.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::FinalNorm: {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::InputNorm:
            case MinimaxTensorKind::PostNorm:
            case MinimaxTensorKind::AttnQNorm:
            case MinimaxTensorKind::AttnKNorm:
            case MinimaxTensorKind::IndexQNorm:
            case MinimaxTensorKind::IndexKNorm: {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::RouterGate:
            case MinimaxTensorKind::RouterBias: {
                // High precision on purpose: the routing bias moves the
                // top-k SELECTION, so a 4-bit router swaps the expert set.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                router_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::SharedGateProj:
            case MinimaxTensorKind::SharedUpProj:
            case MinimaxTensorKind::SharedDownProj:
                ARROW_RETURN_NOT_OK(
                    add_nvfp4(t, data, shared_nvfp4_by_layer[p.layer]));
                break;
            case MinimaxTensorKind::AttnQ:
            case MinimaxTensorKind::AttnK:
            case MinimaxTensorKind::AttnV:
            case MinimaxTensorKind::AttnO:
            case MinimaxTensorKind::IndexQ:
            case MinimaxTensorKind::IndexK:
            case MinimaxTensorKind::DenseMlpGate:
            case MinimaxTensorKind::DenseMlpUp:
            case MinimaxTensorKind::DenseMlpDown:
                ARROW_RETURN_NOT_OK(add_nvfp4(t, data, nvfp4_by_layer[p.layer]));
                break;
            case MinimaxTensorKind::ExpertW1:
            case MinimaxTensorKind::ExpertW2:
            case MinimaxTensorKind::ExpertW3:
                break;  // handled by the expert pass
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    // (3) Routed experts: expert-major second pass. Reads w1/w2/w3 (BF16,
    // <= 38 MB each) straight from their shards, quantizes, and writes one
    // ExpertSchema file per (layer, expert). No host-resident expert pool.
    const int H = top.hidden_size;
    const int I = top.moe_intermediate_size;
    int experts_done = 0;
    const int experts_total =
        (top.num_layers - top.first_moe_layer) * top.num_experts;
    int last_pct = -1;
    // Operational fast path: re-convert only the dense side (globals, router,
    // shared experts, attention/dense-MLP layers) when the 232 GB expert store
    // is already on disk. The written index is byte-identical (every field is
    // topology-derived), so the tree stays a single consistent artifact.
    const bool skip_experts = std::getenv("LSX_MINIMAX_SKIP_EXPERTS") != nullptr;
    if (skip_experts)
        lsxcommon::log::info(
            "MiniMax: LSX_MINIMAX_SKIP_EXPERTS set — reusing the existing "
            "expert store");
    for (int l = 0; !skip_experts && l < top.num_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] != "sparse") continue;
        for (int e = 0; e < top.num_experts; ++e) {
            // Progress for a multi-hour 800 GB pass: one line per 2%.
            {
                const int pct = experts_total > 0
                                    ? (100 * experts_done) / experts_total
                                    : 100;
                if (pct != last_pct && pct % 2 == 0) {
                    lsxcommon::log::info(absl::StrFormat(
                        "MiniMax experts: %d/%d (%d%%) layer %d",
                        experts_done, experts_total, pct, l));
                    last_pct = pct;
                }
                ++experts_done;
            }
            const std::string stem = absl::StrCat(
                "language_model.model.layers.", l,
                ".block_sparse_moe.experts.", e, ".");
            const char *kProjection[3] = {"w1", "w2", "w3"};
            DenseNvfp4Record recs[3];
            for (int k = 0; k < 3; ++k) {
                const std::string base = stem + kProjection[k];
                const std::vector<int64_t> logical_shape =
                    (k == 1) ? std::vector<int64_t>{H, I}
                             : std::vector<int64_t>{I, H};

                auto read_ref = [&](const std::string &full,
                                    std::vector<uint8_t> &out,
                                    const TensorInfo *&info) -> arrow::Status {
                    auto it = expert_index.find(full);
                    if (it == expert_index.end())
                        return arrow::Status::Invalid(
                            "missing expert tensor: ", full);
                    info = it->second.info;
                    out = ReadTensorData(input_dir, *it->second.shard,
                                         *it->second.info);
                    if (out.empty() ||
                        static_cast<int64_t>(out.size()) != info->byte_size())
                        return arrow::Status::IOError(
                            "failed to read expert tensor: ", full);
                    return arrow::Status::OK();
                };

                // Already-quantized ModelOpt NVFP4 expert: verbatim passthrough
                // (packed E2M1 nibbles + per-16 E4M3 scales), with the
                // compressed-tensors DIVISOR global scale stored as its
                // reciprocal (w = nibble * block_scale / weight_global_scale,
                // the repo contour multiplies by the stored global scale).
                if (expert_index.contains(base + ".weight_packed")) {
                    std::vector<uint8_t> packed, scales, gs_raw;
                    const TensorInfo *pinfo = nullptr;
                    const TensorInfo *sinfo = nullptr;
                    const TensorInfo *ginfo = nullptr;
                    ARROW_RETURN_NOT_OK(
                        read_ref(base + ".weight_packed", packed, pinfo));
                    ARROW_RETURN_NOT_OK(
                        read_ref(base + ".weight_scale", scales, sinfo));
                    ARROW_RETURN_NOT_OK(read_ref(base + ".weight_global_scale",
                                                 gs_raw, ginfo));
                    if (pinfo->shape.size() != 2 ||
                        pinfo->shape[0] != logical_shape[0] ||
                        pinfo->shape[1] * 2 != logical_shape[1])
                        return arrow::Status::Invalid(
                            "NVFP4 packed shape is not [rows, cols/2]: ",
                            pinfo->name);
                    if (sinfo->shape.size() != 2 ||
                        sinfo->shape[0] != logical_shape[0] ||
                        sinfo->shape[1] * 16 != logical_shape[1])
                        return arrow::Status::Invalid(
                            "NVFP4 scale shape is not [rows, cols/16]: ",
                            sinfo->name);
                    if (sinfo->dtype != "F8_E4M3")
                        return arrow::Status::Invalid(
                            "NVFP4 block scale is not F8_E4M3: ", sinfo->name);
                    if (gs_raw.size() != 4)
                        return arrow::Status::Invalid(
                            "bad NVFP4 weight_global_scale: ", ginfo->name);
                    float gs = 0.0f;
                    std::memcpy(&gs, gs_raw.data(), 4);
                    if (!(gs > 0.0f))
                        return arrow::Status::Invalid(
                            "non-positive NVFP4 weight_global_scale: ",
                            ginfo->name);
                    recs[k].name = base + ".weight";
                    recs[k].shape = logical_shape;
                    recs[k].dtype = "F8_E4M3";
                    recs[k].packed = std::move(packed);
                    recs[k].scales = std::move(scales);
                    recs[k].global_scale = 1.0f / gs;
                    continue;
                }

                const std::string name = base + ".weight";
                std::vector<uint8_t> data;
                const TensorInfo *info = nullptr;
                ARROW_RETURN_NOT_OK(read_ref(name, data, info));
                const auto &shape = info->shape;
                if (shape.size() != 2)
                    return arrow::Status::Invalid("expert tensor must be 2-D: ",
                                                  name);
                if (shape != logical_shape)
                    return arrow::Status::Invalid(
                        "expert tensor has the wrong shape: ", name);
                recs[k].name = name;
                recs[k].shape = shape;
                recs[k].dtype = info->dtype;
                if (!QuantizeSlice(data.data(), info->numel(), recs[k].dtype,
                                   recs[k]))
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
    if (!global_nvfp4.empty())
        ARROW_RETURN_NOT_OK(
            WriteBatch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                       pinned_dir / "global_nvfp4.arrow"));
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
    top.skipped_visual_tensors = visual_skipped;

    // (5) Tokenizer: mandatory for real checkpoints; synthetic fixtures may
    // omit it (recorded in the index). The chat template spells `]~b]` out as
    // text, so the reference tokenizer must NOT auto-prepend BOS.
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        // The checkpoint's tokenizer.json uses the Llama-3/o200k pattern
        // (case-run alternatives, no CJK split), not the legacy GPT-2 3-split.
        tok_policy.pretokenizer = 1;
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
    vg.num_moe_layers = top.num_layers - top.first_moe_layer;
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
             return minimax_tensor_is_nvfp4(classify_minimax_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return minimax_tensor_is_routed_expert(
                        classify_minimax_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         nullptr},
        vg);

    // (7) Structural manifest (deterministic byte content).
    {
        std::ostringstream j;
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n"
          << "  \"model\": \"" << variant.model << "\",\n"
          << "  \"model_type\": \"" << variant.model_type << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"mlp_layer_types\": [";
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
          << ",\n  \"routed_scaling_factor\": " << top.routed_scaling_factor
          << ",\n  \"swiglu_alpha\": " << top.swiglu_alpha
          << ",\n  \"swiglu_limit\": " << top.swiglu_limit
          << ",\n  \"index_n_heads\": " << top.index_n_heads
          << ",\n  \"index_head_dim\": " << top.index_head_dim
          << ",\n  \"index_block_size\": " << top.index_block_size
          << ",\n  \"index_topk_blocks\": " << top.index_topk_blocks
          << ",\n  \"index_local_blocks\": " << top.index_local_blocks
          << ",\n  \"first_moe_layer\": " << top.first_moe_layer
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << top.num_experts * (top.num_layers -
                                                             top.first_moe_layer)
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
        "MiniMax conversion (%s): %d layers (%d MoE), %d experts/layer, "
        "%d visual tensors skipped, %d pinned tensors",
        variant.model, top.num_layers,
        top.num_layers - top.first_moe_layer, top.num_experts,
        visual_skipped, top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::minimax
