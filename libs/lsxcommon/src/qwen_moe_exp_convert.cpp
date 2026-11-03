#include "lsxcommon/qwen_moe_exp_convert.h"
#include "lsxcommon/synthetic_geometry.h"
#include "qwen_moe_exp_constants.h"

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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::qwen_moe_exp {
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

// Synthetic fixtures (tests) may use tiny geometry; the real pipeline hard
// gates against the frozen constants.

// Cross-checks parsed config against the frozen passport. Returns false and
// fills `why` on mismatch.
bool check_frozen_geometry(const Qwen4ExpTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int64_t got, int64_t want) {
        why = absl::StrFormat("%s: got %lld, want %lld", field,
                              static_cast<long long>(got),
                              static_cast<long long>(want));
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
    if (top.hc_count != kHcCount) return bad("hc_count", top.hc_count, kHcCount);
    if (top.hc_lowrank != kHcLowrank) return bad("hc_lowrank", top.hc_lowrank, kHcLowrank);
    if (top.indexer_n_heads != kIndexerNHeads) return bad("indexer_n_heads", top.indexer_n_heads, kIndexerNHeads);
    if (top.indexer_kv_heads != kIndexerKVHeads) return bad("indexer_kv_heads", top.indexer_kv_heads, kIndexerKVHeads);
    if (top.indexer_head_dim != kIndexerHeadDim) return bad("indexer_head_dim", top.indexer_head_dim, kIndexerHeadDim);
    if (top.indexer_budget != kIndexerBudget) return bad("indexer_budget", top.indexer_budget, kIndexerBudget);
    if (top.indexer_compress_ratio != kIndexerCompress) return bad("indexer_compress_ratio", top.indexer_compress_ratio, kIndexerCompress);
    if (top.ngram_size != kPleNgramSize) return bad("ngram_size", top.ngram_size, kPleNgramSize);
    if (top.heads_per_ngram != kPleHeadsPerNgram) return bad("heads_per_ngram", top.heads_per_ngram, kPleHeadsPerNgram);
    if (top.ple_embed_dim != kPleEmbedDim) return bad("ple_embed_dim", top.ple_embed_dim, kPleEmbedDim);
    if (top.split_ngram_parts != kPleSplitParts) return bad("split_ngram_parts", top.split_ngram_parts, kPleSplitParts);
    for (int i = 0; i < top.num_layers; ++i) {
        const std::string want = (i % kFullAttnInterval == kFullAttnInterval - 1)
                                     ? "full_attention" : "linear_attention";
        if (top.layer_types[static_cast<size_t>(i)] != want)
            return bad("layer_types[i]", i, 0);
    }
    if (top.ple_layer_ids.size() != 1 || top.ple_layer_ids[0] != kPleLayerIdx + 1)
        return bad("ple_layer_ids", top.ple_layer_ids.empty() ? -1 : top.ple_layer_ids[0],
                   kPleLayerIdx + 1);
    return true;
}

// Quantizes a contiguous BF16 slice to NVFP4. Returns false on GPU failure.
// Writes one expert's NVFP4 payload triple to experts/layer_<l>/exp_<e>.arrow.
// Parses "shard_S" trailing index from a PLE n-gram shard name body. Returns
// -1 when the body does not match. The body keeps the "ple." prefix, e.g.
// "ple.ple_embedding.ngram_embedding.shard_0.weight".
int parse_ple_shard_index(const std::string &body)
{
    static const std::string kHead =
        "ple.ple_embedding.ngram_embedding.shard_";
    if (body.rfind(kHead, 0) != 0) return -1;
    const size_t start = kHead.size();
    const size_t end = body.find('.', start);
    if (end == std::string::npos) return -1;
    if (body.substr(end) != ".weight") return -1;
    int shard = 0;
    if (!detail::ParseTensorIndex(
            std::string_view(body).substr(start, end - start), shard))
        return -1;  // malformed / implausible index -> not a PLE shard
    return shard;
}

bool is_ple_meta_body(const std::string &body)
{
    return body == "ple.ple_embedding.layer_multipliers" ||
           body == "ple.ple_embedding.ngram_heads_offsets" ||
           body == "ple.ple_embedding.ngram_heads_vocab_sizes";
}

} // namespace

Qwen4ExpTopology ParseQwen4ExpConfig(const std::string &input_dir)
{
    Qwen4ExpTopology top;
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
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-6);
    top.partial_rotary_factor = get_d(src, "partial_rotary_factor", 0.25);
    top.rope_theta = get_d(src, "rope_theta", 1e7);

    // MoE fields.
    top.num_experts = get_i(src, "num_experts", 0);
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.moe_intermediate_size = get_i(src, "moe_intermediate_size", 0);
    top.shared_expert_intermediate_size =
        get_i(src, "shared_expert_intermediate_size", 0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);

    // Hyper-connections.
    top.hc_count = get_i(src, "hc_count", 4);
    top.hc_lowrank = get_i(src, "hc_lowrank", 320);

    // QSA indexer.
    top.indexer_n_heads = get_i(src, "indexer_n_heads", 4);
    top.indexer_kv_heads = get_i(src, "indexer_kv_heads", 1);
    top.indexer_head_dim = get_i(src, "indexer_head_dim", 128);
    top.indexer_budget = get_i(src, "indexer_budget", 2048);
    top.indexer_compress_ratio = get_i(src, "indexer_compress_ratio", 4);

    // PLE.
    top.ngram_size = get_i(src, "ngram_size", 3);
    top.heads_per_ngram = get_i(src, "heads_per_ngram", 8);
    top.ple_embed_dim = get_i(src, "ple_embed_dim", 0);
    top.ple_conv_kernel_size = get_i(src, "ple_conv_kernel_size", 4);
    top.split_ngram_parts = get_i(src, "split_ngram_parts", 128);
    if (auto v = src["ngram_vocab_size_base"].get_int64();
        v.error() == simdjson::SUCCESS)
        top.ngram_vocab_size_base = v.value();
    if (auto v = src["output_gate_type"].get_string();
        v.error() == simdjson::SUCCESS)
        top.output_gate_type = std::string(v.value());

    if (auto lt = src["layer_types"].get_array(); lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                top.layer_types.emplace_back(sv);
        }
    }
    if (auto pl = src["ple_layer_ids"].get_array(); pl.error() == simdjson::SUCCESS) {
        for (auto e : pl.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                top.ple_layer_ids.push_back(static_cast<int>(v));
        }
    }

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.num_experts_per_tok <= 0 || top.moe_intermediate_size <= 0 ||
        top.layer_types.empty()) {
        top.error_msg = "config.json missing required qwen4_exp fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("layer_types size ", top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    return top;
}

ParsedExpTensor classify_qwen4_exp_tensor(const std::string &name)
{
    ParsedExpTensor r;
    if (name.rfind("mtp.", 0) == 0) {
        r.kind = ExpTensorKind::Mtp;
        return r;
    }
    if (name.find("visual") != std::string::npos) {
        r.kind = ExpTensorKind::Visual;
        return r;
    }
    if (name == "lm_head.weight") {
        r.kind = ExpTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "model.language_model.";
    if (name.rfind(kPrefix, 0) != 0)
        return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = ExpTensorKind::Embed;
        return r;
    }

    // Global hyper-connection mixer (no layer prefix).
    static const std::pair<const char *, ExpTensorKind> kGlobalMap[] = {
        {"hyper_connection_mixer.hc_norm.weight", ExpTensorKind::GlobalMixerNorm},
        {"hyper_connection_mixer.input_mix_weight_down.weight", ExpTensorKind::GlobalMixerDown},
        {"hyper_connection_mixer.input_mix_weight_up.weight", ExpTensorKind::GlobalMixerUp},
    };
    for (const auto &[suffix, kind] : kGlobalMap) {
        if (rest == suffix) {
            r.kind = kind;
            return r;
        }
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

    using K = ExpTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"self_attn.q_proj.weight", K::QsaQ},
        {"self_attn.k_proj.weight", K::QsaK},
        {"self_attn.v_proj.weight", K::QsaV},
        {"self_attn.o_proj.weight", K::QsaO},
        {"self_attn.q_norm.weight", K::QsaQNorm},
        {"self_attn.k_norm.weight", K::QsaKNorm},
        {"self_attn.indexer.index_qk_proj.weight", K::IndexerQkProj},
        {"self_attn.indexer.q_layernorm.weight", K::IndexerQNorm},
        {"self_attn.indexer.k_layernorm.weight", K::IndexerKNorm},
        {"linear_attn.in_proj_qkv.weight", K::GdnQkv},
        {"linear_attn.in_proj_z.weight", K::GdnZ},
        {"linear_attn.in_proj_b.weight", K::GdnB},
        {"linear_attn.in_proj_a.weight", K::GdnA},
        {"linear_attn.out_proj.weight", K::GdnOut},
        {"linear_attn.conv1d.weight", K::GdnConv},
        {"linear_attn.norm.weight", K::GdnNorm},
        {"linear_attn.A_log", K::GdnALog},
        {"linear_attn.dt_bias", K::GdnDt},
        {"attn_hyper_connection.hc_norm.weight", K::AttnHcNorm},
        {"attn_hyper_connection.input_mix_weight_down.weight", K::AttnHcDown},
        {"attn_hyper_connection.input_mix_weight_up.weight", K::AttnHcUp},
        {"attn_hyper_connection.block_inject_weight.weight", K::AttnHcBlockInject},
        {"mlp_hyper_connection.hc_norm.weight", K::MlpHcNorm},
        {"mlp_hyper_connection.input_mix_weight_down.weight", K::MlpHcDown},
        {"mlp_hyper_connection.input_mix_weight_up.weight", K::MlpHcUp},
        {"mlp_hyper_connection.block_inject_weight.weight", K::MlpHcBlockInject},
        {"mlp.gate.weight", K::RouterGate},
        {"mlp.experts.gate_up_proj", K::FusedExpertsGateUp},
        {"mlp.experts.down_proj", K::FusedExpertsDown},
        {"mlp.shared_expert.gate_proj.weight", K::SharedGateProj},
        {"mlp.shared_expert.up_proj.weight", K::SharedUpProj},
        {"mlp.shared_expert.down_proj.weight", K::SharedDownProj},
        {"mlp.shared_expert_gate.weight", K::SharedGate},
        {"ple.key_proj.weight", K::PleKeyProj},
        {"ple.value_proj.weight", K::PleValueProj},
        {"ple.norm_key.weight", K::PleNormKey},
        {"ple.norm_query.weight", K::PleNormQuery},
        {"ple.norm_conv.weight", K::PleNormConv},
        {"ple.conv1d.weight", K::PleConv},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }

    // PLE n-gram shards + meta buffers.
    const int shard = parse_ple_shard_index(body);
    if (shard >= 0) {
        r.kind = K::PleNgramShard;
        r.shard = shard;
        return r;
    }
    if (is_ple_meta_body(body)) {
        r.kind = K::PleMeta;
        return r;
    }
    return r;
}

bool qwen4_exp_tensor_is_nvfp4(ExpTensorKind kind)
{
    switch (kind) {
    case ExpTensorKind::Embed:
    case ExpTensorKind::LmHead:
    case ExpTensorKind::GlobalMixerDown:
    case ExpTensorKind::GlobalMixerUp:
    case ExpTensorKind::GdnQkv:
    case ExpTensorKind::GdnZ:
    case ExpTensorKind::GdnB:
    case ExpTensorKind::GdnA:
    case ExpTensorKind::GdnOut:
    case ExpTensorKind::QsaQ:
    case ExpTensorKind::QsaK:
    case ExpTensorKind::QsaV:
    case ExpTensorKind::QsaO:
    case ExpTensorKind::IndexerQkProj:
    case ExpTensorKind::AttnHcDown:
    case ExpTensorKind::AttnHcUp:
    case ExpTensorKind::AttnHcBlockInject:
    case ExpTensorKind::MlpHcDown:
    case ExpTensorKind::MlpHcUp:
    case ExpTensorKind::MlpHcBlockInject:
    case ExpTensorKind::FusedExpertsGateUp:
    case ExpTensorKind::FusedExpertsDown:
    case ExpTensorKind::SharedGateProj:
    case ExpTensorKind::SharedUpProj:
    case ExpTensorKind::SharedDownProj:
    case ExpTensorKind::PleKeyProj:
    case ExpTensorKind::PleValueProj:
        return true;
    default:
        return false;
    }
}

void Splitmix64Multipliers(int64_t seed, int ple_layer_index,
                           int64_t unigram_vocab, int ngram_size,
                           int64_t out[3])
{
    constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ull;
    constexpr uint64_t kM1 = 0xBF58476D1CE4E5B9ull;
    constexpr uint64_t kM2 = 0x94D049BB133111EBull;
    constexpr int64_t kPrime1 = 10007;
    auto splitmix64 = [](uint64_t v) -> uint64_t {
        v += kGamma;
        v ^= v >> 30;
        v *= kM1;
        v ^= v >> 27;
        v *= kM2;
        v ^= v >> 31;
        return v;
    };
    const int64_t max_long = std::numeric_limits<int64_t>::max();
    const int64_t multiplier_max = max_long / (unigram_vocab > 0 ? unigram_vocab : 1);
    const int64_t half_bound = std::max<int64_t>(1, multiplier_max / 2);
    const int64_t base = seed + kPrime1 * ple_layer_index;
    for (int i = 0; i < ngram_size && i < 3; ++i) {
        const uint64_t v = static_cast<uint64_t>(base) +
                           kGamma * static_cast<uint64_t>(i + 1);
        out[i] = static_cast<int64_t>(
            2 * (splitmix64(v) % static_cast<uint64_t>(half_bound)) + 1);
    }
}

int64_t FindNthPrimeAfter(int64_t start, int64_t count)
{
    auto is_prime = [](int64_t v) {
        if (v < 2) return false;
        if (v % 2 == 0) return v == 2;
        for (int64_t d = 3; d * d <= v; d += 2)
            if (v % d == 0) return false;
        return true;
    };
    int64_t p = start;
    for (int64_t i = 0; i < count; ++i) {
        ++p;
        while (!is_prime(p)) ++p;
    }
    return p;
}

arrow::Status ConvertQwen38FlashNextModel(const std::string &input_dir,
                                          const std::string &output_root)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    Qwen4ExpTopology top = ParseQwen4ExpConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Qwen4-exp config error: ", top.error_msg);

    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");

    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!check_frozen_geometry(top, why))
            return arrow::Status::Invalid(
                "Qwen3.8-Flash-Next frozen geometry mismatch: ", why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    // Classify the inventory up front: unknown names are hard errors, visual
    // tensors are counted skips, everything else must be consumable.
    int visual_skipped = 0;
    int ple_shard_count = 0;
    absl::btree_map<int, int> layer_kind_count;  // layer -> classified tensors
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_qwen4_exp_tensor(t.name);
            if (p.kind == ExpTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == ExpTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in qwen4_exp checkpoint: ", t.name);
            if (p.kind == ExpTensorKind::PleNgramShard)
                ++ple_shard_count;
            if (p.layer >= 0)
                ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");
    if (ple_shard_count > 0 &&
        ple_shard_count != top.split_ngram_parts)
        return arrow::Status::Invalid("PLE n-gram shard count ", ple_shard_count,
                                      " != split_ngram_parts ",
                                      top.split_ngram_parts);

    // (2) Source reads must be complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    // Unified vault convention: <output_root>/logestix.
    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    const fs::path mtp_dir = base / "mtp";
    const fs::path ple_dir = base / "ple";
    for (const fs::path &d :
         {base, pinned_dir, router_dir, shared_dir, mtp_dir, ple_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    // Buckets (etalon: ConvertQwen35MoeModel bucket layout).
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> shared_nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> shared_gate_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> ple_nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> ple_raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> ple_shard_by_idx;
    std::vector<TensorRecord> ple_meta_records;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;
    std::vector<DenseNvfp4Record> mtp_nvfp4;
    std::vector<TensorRecord> mtp_raw;

    int64_t total_pinned = 0;

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_qwen4_exp_tensor(t.name);
            if (p.kind == ExpTensorKind::Visual)
                continue;  // counted skip

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() ||
                static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            // MTP subtree: converted and stored, never applied at inference
            // (HF parity — modeling_qwen4_exp.py ignores ^mtp.* on load).
            // Norms stay high precision; everything else is NVFP4.
            if (p.kind == ExpTensorKind::Mtp) {
                const bool keep_raw = t.name.find("norm") != std::string::npos;
                if (keep_raw) {
                    TensorRecord r;
                    r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                    r.data = std::move(data);
                    mtp_raw.push_back(std::move(r));
                } else {
                    DenseNvfp4Record r;
                    r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                    if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                        return arrow::Status::IOError(
                            "GPU quantization failed for ", t.name);
                    mtp_nvfp4.push_back(std::move(r));
                }
                continue;
            }

            // PLE n-gram shards: raw BF16 passthrough (mmap-only at load).
            if (p.kind == ExpTensorKind::PleNgramShard) {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                ple_shard_by_idx[p.shard].push_back(std::move(r));
                continue;
            }
            // PLE meta buffers: raw I64 passthrough.
            if (p.kind == ExpTensorKind::PleMeta) {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                ple_meta_records.push_back(std::move(r));
                continue;
            }

            switch (p.kind) {
            case ExpTensorKind::LmHead: {
                // LM head stays in BF16: NVFP4 logit noise (~0.5) exceeds the
                // typical top-1/top-2 logit margin of this model.
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::Embed: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                global_nvfp4.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::GlobalMixerDown:
            case ExpTensorKind::GlobalMixerUp: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                global_nvfp4.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::GlobalMixerNorm:
            case ExpTensorKind::FinalNorm: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::GdnConv:
            case ExpTensorKind::GdnNorm:
            case ExpTensorKind::GdnALog:
            case ExpTensorKind::GdnDt:
            case ExpTensorKind::QsaQNorm:
            case ExpTensorKind::QsaKNorm:
            case ExpTensorKind::IndexerQNorm:
            case ExpTensorKind::IndexerKNorm:
            case ExpTensorKind::AttnHcNorm:
            case ExpTensorKind::MlpHcNorm: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::GdnQkv:
            case ExpTensorKind::GdnZ:
            case ExpTensorKind::GdnB:
            case ExpTensorKind::GdnA:
            case ExpTensorKind::GdnOut:
            case ExpTensorKind::QsaQ:
            case ExpTensorKind::QsaK:
            case ExpTensorKind::QsaV:
            case ExpTensorKind::QsaO:
            case ExpTensorKind::IndexerQkProj:
            case ExpTensorKind::AttnHcDown:
            case ExpTensorKind::AttnHcUp:
            case ExpTensorKind::AttnHcBlockInject:
            case ExpTensorKind::MlpHcDown:
            case ExpTensorKind::MlpHcUp:
            case ExpTensorKind::MlpHcBlockInject: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::RouterGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                router_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::SharedGateProj:
            case ExpTensorKind::SharedUpProj:
            case ExpTensorKind::SharedDownProj: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                shared_nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::SharedGate: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                shared_gate_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::PleKeyProj:
            case ExpTensorKind::PleValueProj: {
                DenseNvfp4Record r;
                r.name = t.name; r.shape = t.shape; r.dtype = t.dtype;
                if (!QuantizeSliceBf16(data.data(), t.numel(), r))
                    return arrow::Status::IOError("GPU quantization failed for ",
                                                  t.name);
                ple_nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::PleNormKey:
            case ExpTensorKind::PleNormQuery:
            case ExpTensorKind::PleNormConv:
            case ExpTensorKind::PleConv: {
                TensorRecord r;
                r.name = t.name; r.dtype = t.dtype; r.shape = t.shape;
                r.data = std::move(data);
                ple_raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case ExpTensorKind::FusedExpertsGateUp:
            case ExpTensorKind::FusedExpertsDown:
                continue;  // handled by the dedicated fused-expert pass below
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ",
                                              t.name);
            }
        }
    }

    // Fused-expert split. HF semantics (modeling_qwen4_exp.py,
    // Qwen4ExpTextExperts.forward): gate_up_proj[e] is [2I, H];
    // gate = rows [0,I), up = rows [I,2I). down_proj[e] is [H, I].
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    const int64_t row_bytes = static_cast<int64_t>(H) * 2;  // BF16 row
    using ExpertKey = std::pair<int, int>;                  // (layer, eid)
    absl::btree_map<ExpertKey, std::pair<DenseNvfp4Record, DenseNvfp4Record>>
        gate_up_w1_w3;
    absl::btree_map<ExpertKey, DenseNvfp4Record> down_w2;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_qwen4_exp_tensor(t.name);
            if (p.kind != ExpTensorKind::FusedExpertsGateUp &&
                p.kind != ExpTensorKind::FusedExpertsDown)
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
                if (p.kind == ExpTensorKind::FusedExpertsGateUp) {
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
                    if (!QuantizeSliceBf16(data.data() + g_off,
                                        static_cast<int64_t>(I) * H, w1r))
                        return arrow::Status::IOError("quantize failed w1 ",
                                                      t.name, " e", e);
                    if (!QuantizeSliceBf16(data.data() + g_off + I * row_bytes,
                                        static_cast<int64_t>(I) * H, w3r))
                        return arrow::Status::IOError("quantize failed w3 ",
                                                      t.name, " e", e);
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
                    if (!QuantizeSliceBf16(data.data() + d_off,
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
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
        }
    }

    // PLE meta cross-check: recompute multipliers + prime sizes and compare
    // against the stored checkpoint buffers (bit-exact requirement). Synthetic
    // fixtures without PLE skip this block.
    const bool has_ple = !ple_meta_records.empty();
    if (has_ple) {
        int64_t mult[3];
        Splitmix64Multipliers(frozen::kPleSeed, 0, top.vocab_size,
                              top.ngram_size, mult);
        const int ple_heads = (top.ngram_size - 1) * top.heads_per_ngram;
        bool mult_ok = false, offs_ok = false, sizes_ok = false;
        for (const auto &rec : ple_meta_records) {
            const size_t want64 =
                rec.data.size() >= 8 ? rec.data.size() / 8 : 0;
            if (rec.name.ends_with("layer_multipliers")) {
                mult_ok = rec.data.size() == sizeof(int64_t) * 3;
                if (mult_ok) {
                    int64_t got[3];
                    std::memcpy(got, rec.data.data(), sizeof(got));
                    for (int i = 0; i < 3; ++i)
                        mult_ok = mult_ok && got[i] == mult[i];
                }
            } else if (rec.name.ends_with("ngram_heads_vocab_sizes")) {
                sizes_ok = static_cast<int>(want64) == ple_heads;
                for (size_t i = 0; sizes_ok && i < want64; ++i) {
                    int64_t got;
                    std::memcpy(&got, rec.data.data() + i * 8, 8);
                    sizes_ok = got == FindNthPrimeAfter(
                                           top.ngram_vocab_size_base - 1,
                                           static_cast<int64_t>(i) + 1);
                }
            } else if (rec.name.ends_with("ngram_heads_offsets")) {
                offs_ok = static_cast<int>(want64) == ple_heads;
                int64_t expect = 0;
                for (size_t i = 0; offs_ok && i < want64; ++i) {
                    int64_t got;
                    std::memcpy(&got, rec.data.data() + i * 8, 8);
                    offs_ok = got == expect;
                    expect += FindNthPrimeAfter(top.ngram_vocab_size_base - 1,
                                                static_cast<int64_t>(i) + 1);
                }
            }
        }
        if (!mult_ok)
            return arrow::Status::Invalid(
                "PLE layer_multipliers mismatch vs splitmix64(seed=",
                frozen::kPleSeed, ")");
        if (!sizes_ok || !offs_ok)
            return arrow::Status::Invalid(
                "PLE ngram prime table mismatch vs recomputation");
    }

    // Globals + dense layers (naming identical to the qwen loader layout).
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
    for (auto &[layer, recs] : ple_nvfp4_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            ple_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    }
    for (auto &[layer, recs] : ple_raw_by_layer) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            ple_dir / absl::StrCat("layer_", layer, ".arrow")));
    }
    for (auto &[shard_idx, recs] : ple_shard_by_idx) {
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            ple_dir / absl::StrCat("shard_", shard_idx, ".arrow")));
    }
    if (!ple_meta_records.empty()) {
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(ple_meta_records),
                                        ple_dir / "meta.arrow"));
    }
    if (!mtp_nvfp4.empty()) {
        ARROW_RETURN_NOT_OK(
            WriteBatch(DenseNvfp4Schema::MakeBatch(mtp_nvfp4),
                        mtp_dir / "mtp_nvfp4.arrow"));
    }
    if (!mtp_raw.empty()) {
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(mtp_raw),
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
          << "  \"format\": \"logestix-qwen38flashnext\",\n"
          << "  \"model\": \"Qwen3.8-Flash-Next\",\n"
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
          << ",\n  \"hc_count\": " << top.hc_count
          << ",\n  \"hc_lowrank\": " << top.hc_lowrank
          << ",\n  \"indexer_n_heads\": " << top.indexer_n_heads
          << ",\n  \"indexer_kv_heads\": " << top.indexer_kv_heads
          << ",\n  \"indexer_head_dim\": " << top.indexer_head_dim
          << ",\n  \"indexer_budget\": " << top.indexer_budget
          << ",\n  \"indexer_compress_ratio\": " << top.indexer_compress_ratio
          << ",\n  \"ngram_size\": " << top.ngram_size
          << ",\n  \"heads_per_ngram\": " << top.heads_per_ngram
          << ",\n  \"ple_embed_dim\": " << top.ple_embed_dim
          << ",\n  \"ple_conv_kernel_size\": " << top.ple_conv_kernel_size
          << ",\n  \"split_ngram_parts\": " << top.split_ngram_parts
          << ",\n  \"ngram_vocab_size_base\": " << top.ngram_vocab_size_base
          << ",\n  \"ple_layer_ids\": ["
          << (top.ple_layer_ids.empty() ? -1 : top.ple_layer_ids[0]) << "]"
          << ",\n  \"output_gate_type\": \"" << top.output_gate_type << "\""
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          << ",\n  \"ple_shards\": " << ple_shard_count
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Final verification: every produced arrow file must re-open cleanly.
    for (const fs::path &dir :
         {pinned_dir, router_dir, shared_dir, mtp_dir, ple_dir}) {
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
        "Qwen4-exp conversion: %d layers x %d experts, %d visual tensors "
        "skipped, %d PLE shards, %d pinned tensors",
        top.num_layers, top.num_experts, visual_skipped, ple_shard_count,
        top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::qwen_moe_exp
