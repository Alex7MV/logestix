#include "lsxcommon/hy_moe_flash_convert.h"

#include "hy_moe_flash_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "lsxcommon/vram_estimate.h"

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace lsxcommon {
namespace hy4 {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::hy4;

// ── Name classification ────────────────────────────────────────────────────
namespace {

// Parses the leading `model.layers.<n>.` prefix; returns -1 when absent.
int LayerOf(std::string_view name, std::string_view *rest)
{
    constexpr std::string_view kPrefix = "model.layers.";
    if (name.substr(0, kPrefix.size()) != kPrefix) return -1;
    std::string_view tail = name.substr(kPrefix.size());
    const size_t dot = tail.find('.');
    if (dot == std::string_view::npos || dot == 0 || dot > 3) return -1;
    int layer = 0;
    for (size_t i = 0; i < dot; ++i) {
        const char c = tail[i];
        if (c < '0' || c > '9') return -1;
        layer = layer * 10 + (c - '0');
    }
    *rest = tail.substr(dot + 1);
    return layer;
}

bool EndsWith(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Layer-local tensors that stay at source precision. Norms and biases are never
// quantized (repo invariant); the router gate moves the top-k SELECTION and the
// iHC scalars are fp32 in the reference, so they ride along.
bool IsRawLayerLeaf(std::string_view leaf)
{
    static const char *const kRaw[] = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "hc_attn_layer.hc_pre.hc_fn",
        "hc_attn_layer.hc_pre.hc_base",
        "hc_attn_layer.hc_pre.hc_scale",
        "hc_mlp_layer.hc_pre.hc_fn",
        "hc_mlp_layer.hc_pre.hc_base",
        "hc_mlp_layer.hc_pre.hc_scale",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.post_proj_norm.weight",
        "self_attn.learnable_sink_param",
        "mlp.post_fc2_norm.weight",
        "mlp.combine_output_norm.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        // Indexer scalars the reference keeps in F32.
        "self_attn.compressed_indexer.key_value_projection.weight",
        "self_attn.compressed_indexer.key_gate_projection.weight",
        "self_attn.compressed_indexer.token_key_norm.weight",
        "self_attn.compressed_indexer.key_norm.weight",
        "self_attn.compressed_indexer.intra_group_ape",
    };
    for (const char *raw : kRaw)
        if (leaf == raw) return true;
    return false;
}

// Layer-local 2-D linear matrices (NVFP4).
bool IsDenseLayerLeaf(std::string_view leaf)
{
    static const char *const kDense[] = {
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.linear_gate.weight",
        "self_attn.compressed_indexer.query_projection.weight",
        "self_attn.compressed_indexer.token_key_projection.weight",
        "self_attn.compressed_indexer.head_weight_projection.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "mlp.routed_expert_down_proj.weight",
        "mlp.routed_expert_up_proj.weight",
    };
    for (const char *d : kDense)
        if (leaf == d) return true;
    return false;
}

} // namespace

ParsedTensor ParseHy4Tensor(std::string_view name)
{
    ParsedTensor r;
    // Global tensors.
    if (name == "model.embed_tokens.weight" || name == "lm_head.weight" ||
        name == "model.norm.weight" || name.rfind("model.hc_head.", 0) == 0) {
        r.kind = TensorKind::kGlobalRaw;
        return r;
    }
    // Ballast, counted.
    if (name.rfind("vit.", 0) == 0) {
        r.kind = TensorKind::kCutVision;
        return r;
    }
    if (name.rfind("model.dfly_layers.", 0) == 0) {
        r.kind = TensorKind::kCutDfly;
        return r;
    }
    std::string_view leaf;
    const int layer = LayerOf(name, &leaf);
    if (layer < 0) return r;  // kUnknown

    if (leaf.rfind("engram_layer.", 0) == 0) {
        r.kind = TensorKind::kCutEngram;
        r.layer = layer;
        return r;
    }
    if (leaf == "mlp.experts.gate_up_proj") {
        r.kind = TensorKind::kExpertGateUp;
        r.layer = layer;
        return r;
    }
    if (leaf == "mlp.experts.down_proj") {
        r.kind = TensorKind::kExpertDown;
        r.layer = layer;
        return r;
    }
    if (IsRawLayerLeaf(leaf)) {
        r.kind = TensorKind::kLayerRaw;
        r.layer = layer;
        return r;
    }
    if (IsDenseLayerLeaf(leaf)) {
        r.kind = TensorKind::kDenseNvfp4;
        r.layer = layer;
        return r;
    }
    return r;  // kUnknown — the converter refuses the tree
}

// ── Config ─────────────────────────────────────────────────────────────────
namespace {

struct Config {
    int layers = 0;
    int hidden = 0;
    int vocab = 0;
    int experts = 0;
    int topk = 0;
    int moe_inter = 0;
    int latent = 0;
    int shared_inter = 0;
    int dense_inter = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int heads_dsa = 0;
    int heads_swa = 0;
    int window = 0;
    int hc_mult = 0;
    int index_heads = 0;
    int index_dim = 0;
    int index_topk = 0;
    int64_t context = 0;
    int bos = -1, eos = -1, pad = -1;
    std::vector<std::string> layer_types;
    std::vector<std::string> mlp_types;
    std::vector<int> engram_layers;
};

template <typename Obj>
int GetInt(Obj &o, const char *key, int dflt)
{
    int64_t v = 0;
    if (o[key].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

template <typename Obj>
std::vector<std::string> GetStrArray(Obj &o, const char *key)
{
    std::vector<std::string> out;
    simdjson::ondemand::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return out;
    for (auto e : arr) {
        std::string_view s;
        if (e.get_string().get(s) == simdjson::SUCCESS) out.emplace_back(s);
    }
    return out;
}

template <typename Obj>
std::vector<int> GetIntArray(Obj &o, const char *key)
{
    std::vector<int> out;
    simdjson::ondemand::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return out;
    for (auto e : arr) {
        int64_t v = 0;
        if (e.get_int64().get(v) == simdjson::SUCCESS)
            out.push_back(static_cast<int>(v));
    }
    return out;
}

arrow::Result<Config> ParseConfig(const fs::path &cfg_path)
{
    std::ifstream f(cfg_path);
    if (!f) return arrow::Status::IOError("cannot open ", cfg_path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
        return arrow::Status::Invalid("config.json parse failed");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("config.json root is not an object");

    // The text backbone lives under text_config in the multimodal wrapper and at
    // the top level in the text-only export. Both are accepted; the frozen gate
    // decides whether the values are ours.
    simdjson::ondemand::object text_obj;
    const bool has_text =
        root["text_config"].get_object().get(text_obj) == simdjson::SUCCESS;
    simdjson::ondemand::object &t = has_text ? text_obj : root;

    Config c;
    c.layers = GetInt(t, "num_hidden_layers", 0);
    c.hidden = GetInt(t, "hidden_size", 0);
    c.vocab = GetInt(t, "vocab_size", 0);
    c.experts = GetInt(t, "n_routed_experts", 0);
    c.topk = GetInt(t, "num_experts_per_tok", 0);
    c.moe_inter = GetInt(t, "moe_intermediate_size", 0);
    c.latent = GetInt(t, "routed_expert_hidden_size", c.moe_inter);
    c.shared_inter = c.moe_inter * std::max(GetInt(t, "n_shared_experts", 1), 1);
    c.dense_inter = GetInt(t, "intermediate_size", 0);
    c.q_lora = GetInt(t, "q_lora_rank", 0);
    c.kv_lora = GetInt(t, "kv_lora_rank", 0);
    c.qk_nope = GetInt(t, "qk_nope_head_dim", 0);
    c.qk_rope = GetInt(t, "qk_rope_head_dim", 0);
    c.v_head = GetInt(t, "v_head_dim", 0);
    c.heads_dsa = GetInt(t, "num_attention_heads", 0);
    c.heads_swa = GetInt(t, "swa_num_attention_heads", c.heads_dsa);
    c.window = GetInt(t, "sliding_window", 0);
    c.hc_mult = GetInt(t, "hc_mult", 0);
    c.index_heads = GetInt(t, "index_n_heads", 0);
    c.index_dim = GetInt(t, "index_head_dim", 0);
    c.index_topk = GetInt(t, "index_topk", 0);
    c.bos = GetInt(t, "bos_token_id", -1);
    c.eos = GetInt(t, "eos_token_id", -1);
    c.pad = GetInt(t, "pad_token_id", -1);
    c.layer_types = GetStrArray(t, "layer_types");
    c.mlp_types = GetStrArray(t, "mlp_layer_types");
    c.engram_layers = GetIntArray(t, "engram_layer_ids");
    c.context = lsxcommon::ReadContextLength(cfg_path.string());
    return c;
}

// Cross-checks the parsed config against the frozen passport. Without the
// synthetic-geometry bypass every mismatch is a hard error: a 42-layer artifact
// loaded as if it were 41 decodes garbage rather than failing.
arrow::Status CheckFrozenGeometry(const Config &c)
{
    if (SyntheticGeometryAllowed()) return arrow::Status::OK();
    auto bad = [](const char *field, long long got, long long want) {
        return arrow::Status::Invalid(absl::StrFormat(
            "%s: source %lld, frozen passport %lld", field, got, want));
    };
    if (c.layers != frozen::kLayers) return bad("num_hidden_layers", c.layers, frozen::kLayers);
    if (c.hidden != frozen::kHidden) return bad("hidden_size", c.hidden, frozen::kHidden);
    if (c.vocab != frozen::kVocab) return bad("vocab_size", c.vocab, frozen::kVocab);
    if (c.experts != frozen::kNumRouted) return bad("n_routed_experts", c.experts, frozen::kNumRouted);
    if (c.topk != frozen::kTopK) return bad("num_experts_per_tok", c.topk, frozen::kTopK);
    if (c.moe_inter != frozen::kMoeIntermediate)
        return bad("moe_intermediate_size", c.moe_inter, frozen::kMoeIntermediate);
    if (c.latent != frozen::kLatent) return bad("routed_expert_hidden_size", c.latent, frozen::kLatent);
    if (c.shared_inter != frozen::kSharedIntermediate)
        return bad("shared intermediate", c.shared_inter, frozen::kSharedIntermediate);
    if (c.dense_inter != frozen::kDenseIntermediate)
        return bad("intermediate_size", c.dense_inter, frozen::kDenseIntermediate);
    if (c.q_lora != frozen::kQLora) return bad("q_lora_rank", c.q_lora, frozen::kQLora);
    if (c.kv_lora != frozen::kKvLora) return bad("kv_lora_rank", c.kv_lora, frozen::kKvLora);
    if (c.qk_nope != frozen::kQkNope) return bad("qk_nope_head_dim", c.qk_nope, frozen::kQkNope);
    if (c.qk_rope != frozen::kQkRope) return bad("qk_rope_head_dim", c.qk_rope, frozen::kQkRope);
    if (c.v_head != frozen::kVHead) return bad("v_head_dim", c.v_head, frozen::kVHead);
    if (c.heads_dsa != frozen::kHeadsDsa) return bad("num_attention_heads", c.heads_dsa, frozen::kHeadsDsa);
    if (c.heads_swa != frozen::kHeadsSwa)
        return bad("swa_num_attention_heads", c.heads_swa, frozen::kHeadsSwa);
    if (c.window != frozen::kSwaWindow) return bad("sliding_window", c.window, frozen::kSwaWindow);
    if (c.hc_mult != frozen::kHcMult) return bad("hc_mult", c.hc_mult, frozen::kHcMult);
    if (c.index_heads != frozen::kIndexNHeads)
        return bad("index_n_heads", c.index_heads, frozen::kIndexNHeads);
    if (c.index_dim != frozen::kIndexHeadDim)
        return bad("index_head_dim", c.index_dim, frozen::kIndexHeadDim);
    if (c.index_topk != frozen::kIndexTopk)
        return bad("index_topk", c.index_topk, frozen::kIndexTopk);
    if (static_cast<int>(c.layer_types.size()) != c.layers)
        return arrow::Status::Invalid(absl::StrFormat(
            "layer_types has %zu entries, num_hidden_layers is %d",
            c.layer_types.size(), c.layers));
    for (int l = 0; l < c.layers; ++l) {
        const bool dsa = c.layer_types[static_cast<size_t>(l)] ==
                         "deepseek_sparse_attention";
        if (dsa != frozen::IsDsa(l))
            return arrow::Status::Invalid(absl::StrFormat(
                "layer_types[%d] = %s disagrees with the frozen hybrid "
                "pattern (IsDsa=%d)", l,
                c.layer_types[static_cast<size_t>(l)].c_str(),
                frozen::IsDsa(l) ? 1 : 0));
    }
    return arrow::Status::OK();
}

} // namespace

arrow::Status ReadHy4LayerShapes(const std::string &input_path,
                                 std::vector<LayerShape> &layers,
                                 std::string &why)
{
    layers.clear();
    why.clear();
    const fs::path cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path)) {
        why = "config.json not found: " + cfg_path.string();
        return arrow::Status::Invalid(why);
    }
    auto cfg = ParseConfig(cfg_path);
    if (!cfg.ok()) {
        why = cfg.status().ToString();
        return cfg.status();
    }
    const Config c = *cfg;
    if (c.layers <= 0 || c.hidden <= 0) {
        why = "config.json declares no layers";
        return arrow::Status::Invalid(why);
    }
    layers.resize(static_cast<size_t>(c.layers));
    for (int l = 0; l < c.layers; ++l) {
        LayerShape s;
        const std::string &type = l < static_cast<int>(c.layer_types.size())
                                      ? c.layer_types[static_cast<size_t>(l)]
                                      : std::string("sliding_attention");
        s.is_dsa = type == "deepseek_sparse_attention";
        const std::string &mlp = l < static_cast<int>(c.mlp_types.size())
                                     ? c.mlp_types[static_cast<size_t>(l)]
                                     : std::string("sparse");
        s.is_moe = mlp == "sparse";
        // The config's `engram_layer_ids` is 1-BASED: it lists [2, 12, 22, 32]
        // while the tensors (and the model card) put the four engram layers at
        // 1, 11, 21, 31. Trusting the list verbatim marks the wrong layers, so
        // the passport is the source of truth here and the converter
        // additionally verifies it against the tensors it actually sees.
        s.has_engram = frozen::IsEngramCut(l);
        s.heads = s.is_dsa ? c.heads_dsa : c.heads_swa;
        layers[static_cast<size_t>(l)] = s;
    }
    return arrow::Status::OK();
}

namespace {

// ── Index writer ───────────────────────────────────────────────────────────

std::string JoinInts(const std::vector<int> &v)
{
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(v[i]);
    }
    return out;
}

std::string CardNameFor(const std::string &input_path)
{
    const fs::path p(input_path);
    std::string name = p.filename().string();
    if (name.empty() || name == "." || name == "..") return frozen::kModelName;
    return name;
}

arrow::Status WriteIndex(const Config &c, const std::vector<LayerShape> &shapes,
                         int64_t total_pinned, int64_t total_experts,
                         const VramProfile &vram, const DeferredCounts &cuts,
                         const std::string &card_name, const fs::path &log_root)
{
    std::ofstream f(log_root / "model.index.json");
    if (!f) return arrow::Status::IOError("cannot create model.index.json");

    f << "{\n";
    f << "  \"format\": \"" << frozen::kFormatTag << "\",\n";
    f << "  \"model\": \"" << card_name << "\",\n";
    f << "  \"model_type\": \"" << frozen::kType << "\",\n";
    f << "  \"num_layers\": " << c.layers << ",\n";
    f << "  \"experts_per_layer\": " << c.experts << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  \"hidden_size\": " << c.hidden << ",\n";
    f << "  \"vocab_size\": " << c.vocab << ",\n";
    f << "  \"num_experts_per_tok\": " << c.topk << ",\n";
    f << "  \"moe_intermediate_size\": " << c.moe_inter << ",\n";
    f << "  \"latent_size\": " << c.latent << ",\n";
    f << "  \"shared_intermediate_size\": " << c.shared_inter << ",\n";
    f << "  \"dense_intermediate_size\": " << c.dense_inter << ",\n";
    f << "  \"num_attention_heads\": " << c.heads_dsa << ",\n";
    f << "  \"swa_num_attention_heads\": " << c.heads_swa << ",\n";
    f << "  \"q_lora_rank\": " << c.q_lora << ",\n";
    f << "  \"kv_lora_rank\": " << c.kv_lora << ",\n";
    f << "  \"qk_nope_head_dim\": " << c.qk_nope << ",\n";
    f << "  \"qk_rope_head_dim\": " << c.qk_rope << ",\n";
    f << "  \"v_head_dim\": " << c.v_head << ",\n";
    f << "  \"sliding_window\": " << c.window << ",\n";
    f << "  \"hc_mult\": " << c.hc_mult << ",\n";
    f << "  \"index_n_heads\": " << c.index_heads << ",\n";
    f << "  \"index_head_dim\": " << c.index_dim << ",\n";
    f << "  \"index_topk\": " << c.index_topk << ",\n";
    f << "  \"bos_token_id\": " << c.bos << ",\n";
    f << "  \"eos_token_id\": " << c.eos << ",\n";
    f << "  \"pad_token_id\": " << c.pad << ",\n";
    f << "  " << FormatVramJson(vram) << ",\n";

    f << "  \"layers\": [\n";
    for (int l = 0; l < c.layers; ++l) {
        const LayerShape &s = shapes[static_cast<size_t>(l)];
        f << "    {\"is_dsa\": " << (s.is_dsa ? "true" : "false")
          << ", \"heads\": " << s.heads
          << ", \"is_moe\": " << (s.is_moe ? "true" : "false")
          << ", \"has_engram\": " << (s.has_engram ? "true" : "false") << "}";
        if (l + 1 < c.layers) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    f << "  \"deferred\": {\"engram\": " << cuts.engram
      << ", \"vision\": " << cuts.vision << ", \"dfly\": " << cuts.dfly
      << "},\n";

    f << "  \"files\": {\n    \"pinned\": {\n      \"global\": \"pinned/global.arrow\"";
    for (int i = 0; i < c.layers; ++i)
        f << ",\n      \"layer_" << absl::StrFormat("%03d", i)
          << "\": \"pinned/layer_" << absl::StrFormat("%03d", i) << ".arrow\""
          << ",\n      \"layer_" << absl::StrFormat("%03d", i)
          << "_nvfp4\": \"pinned/layer_" << absl::StrFormat("%03d", i)
          << "_nvfp4.arrow\"";
    f << "\n    },\n    \"experts\": {\n";
    bool first = true;
    for (int i = 0; i < c.layers; ++i) {
        const LayerShape &s = shapes[static_cast<size_t>(i)];
        if (!s.is_moe) continue;
        if (!first) f << ",\n";
        first = false;
        f << "      \"layer_" << absl::StrFormat("%03d", i)
          << "\": \"experts/layer_" << absl::StrFormat("%03d", i) << "/\"";
    }
    f << "\n    }\n  }\n}\n";
    return arrow::Status::OK();
}

arrow::Status SaveDense(const std::vector<TensorRecord> &globals,
                        const absl::btree_map<int, std::vector<TensorRecord>> &raw,
                        const absl::btree_map<int, std::vector<DenseNvfp4Record>> &nvfp4,
                        const fs::path &log_root)
{
    fs::create_directories(log_root / "pinned");
    if (!globals.empty()) {
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(globals));
        ARROW_RETURN_NOT_OK(
            WriteBatch(batch, log_root / "pinned" / "global.arrow"));
    }
    for (const auto &[layer, recs] : raw) {
        if (recs.empty()) continue;
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(recs));
        ARROW_RETURN_NOT_OK(WriteBatch(
            batch, log_root / "pinned" /
                       (absl::StrCat("layer_", absl::StrFormat("%03d", layer)) +
                        ".arrow")));
    }
    for (const auto &[layer, recs] : nvfp4) {
        if (recs.empty()) continue;
        ARROW_ASSIGN_OR_RAISE(auto batch, DenseNvfp4Schema::MakeBatch(recs));
        ARROW_RETURN_NOT_OK(WriteBatch(
            batch, log_root / "pinned" /
                       (absl::StrCat("layer_", absl::StrFormat("%03d", layer)) +
                        "_nvfp4.arrow")));
    }
    return arrow::Status::OK();
}

} // namespace

arrow::Status ConvertHy4FlashModel(const std::string &input_path,
                                   const std::string &output_path)
{
    if (input_path.empty() || output_path.empty())
        return arrow::Status::Invalid("input/output path required");

    const fs::path log_root = fs::path(output_path) / "logestix";
    const fs::path cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);

    ARROW_ASSIGN_OR_RAISE(Config cfg, ParseConfig(cfg_path));
    if (cfg.layers <= 0)
        return arrow::Status::Invalid("config declares no layers");
    ARROW_RETURN_NOT_OK(CheckFrozenGeometry(cfg));

    std::vector<LayerShape> shapes;
    std::string why;
    ARROW_RETURN_NOT_OK(ReadHy4LayerShapes(input_path, shapes, why));

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards in ", input_path);

    // (1) Classify the whole inventory before writing anything, so a source
    // drift fails here rather than after 90 GB of output. The iteration order is
    // the shard order (not a hash map) so the artifact is byte-deterministic.
    absl::flat_hash_map<std::string, std::pair<const TensorInfo *,
                                              const SafetensorsModel::Shard *>>
        by_name;
    std::vector<std::pair<std::string, std::pair<const TensorInfo *,
                                                 const SafetensorsModel::Shard *>>>
        ordered;
    std::vector<VramTensorInfo> inventory;
    DeferredCounts cuts;
    int64_t dense_tensors = 0, raw_tensors = 0, expert_tensors = 0;
    // Engram layers as the TENSORS spell them (0-based); the frozen set must
    // agree, or the artifact would carry a cut the passport does not describe.
    std::vector<int> engram_layers_seen;

    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            inventory.push_back({t.name, t.numel(), t.byte_size()});
            const ParsedTensor p = ParseHy4Tensor(t.name);
            switch (p.kind) {
            case TensorKind::kUnknown:
                return arrow::Status::Invalid(
                    "unclassified tensor (not global/layer/expert/cut): ",
                    t.name);
            case TensorKind::kCutVision: ++cuts.vision; continue;
            case TensorKind::kCutDfly: ++cuts.dfly; continue;
            case TensorKind::kCutEngram:
                ++cuts.engram;
                if (p.layer >= 0 &&
                    std::find(engram_layers_seen.begin(),
                              engram_layers_seen.end(),
                              p.layer) == engram_layers_seen.end())
                    engram_layers_seen.push_back(p.layer);
                continue;
            case TensorKind::kExpertGateUp:
            case TensorKind::kExpertDown: ++expert_tensors; break;
            case TensorKind::kGlobalRaw:
            case TensorKind::kLayerRaw: ++raw_tensors; break;
            case TensorKind::kDenseNvfp4: ++dense_tensors; break;
            }
            const auto entry = std::make_pair(&t, &shard);
            by_name.emplace(t.name, entry);
            ordered.emplace_back(t.name, entry);
        }
    }

    // Two packed 3-D tensors (gate_up + down) per MoE layer, each holding all
    // `experts_per_layer` experts.
    const int64_t expected_expert_tensors =
        2 * static_cast<int64_t>(frozen::MoeLayerCount());
    if (!SyntheticGeometryAllowed() &&
        expert_tensors != expected_expert_tensors)
        return arrow::Status::Invalid(absl::StrFormat(
            "routed-expert tensor count: %lld, frozen passport expects %lld",
            expert_tensors, expected_expert_tensors));

    std::sort(engram_layers_seen.begin(), engram_layers_seen.end());
    if (!SyntheticGeometryAllowed()) {
        std::vector<int> frozen_engram;
        for (int l = 0; l < frozen::kLayers; ++l)
            if (frozen::IsEngramCut(l)) frozen_engram.push_back(l);
        if (engram_layers_seen != frozen_engram)
            return arrow::Status::Invalid(absl::StrFormat(
                "Engram tensors are on layers [%s], the passport says [%s]",
                JoinInts(engram_layers_seen), JoinInts(frozen_engram)));
    }

    lsxcommon::log::info(absl::StrFormat(
        "[hy4-convert] source: %d layers, %d experts/layer, %lld dense, "
        "%lld raw, %lld packed expert tensors; cut (engram=%lld vision=%lld "
        "dfly=%lld)", cfg.layers, cfg.experts, dense_tensors, raw_tensors,
        expert_tensors, cuts.engram, cuts.vision, cuts.dfly));

    // (2) The artifact tree.
    fs::create_directories(log_root / "pinned");

    std::vector<TensorRecord> globals;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    int64_t total_pinned = 0;

    for (const auto &[name, where] : ordered) {
        const ParsedTensor p = ParseHy4Tensor(name);
        if (IsRoutedExpert(p.kind)) continue;  // second pass, per layer
        const auto data = ReadTensorData(input_path, *where.second, *where.first);
        if (data.empty() ||
            static_cast<int64_t>(data.size()) != where.first->byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        if (p.kind == TensorKind::kDenseNvfp4) {
            // kv_b is the one projection the engine stores SPLIT: MLA decode
            // needs the per-head nope rows contiguous (a transposed projection
            // qk = kv_b_nope^T q, 6.3M MACs) and the value rows contiguous (one
            // GEMV against the softmax-weighted latent sum). Interleaved by head
            // in the source ([h*448 + nope|value]), so the rows are gathered here
            // rather than at load time.
            const bool is_kv_b =
                name.size() > 16 &&
                name.compare(name.size() - 16, 16, "kv_b_proj.weight") == 0;
            if (is_kv_b && where.first->shape.size() == 2) {
                const int64_t rows = where.first->shape[0];
                const int64_t cols = where.first->shape[1];
                const int64_t per_head = cfg.qk_nope + cfg.v_head;
                if (per_head <= 0 || rows % per_head != 0)
                    return arrow::Status::Invalid(
                        "kv_b rows are not a whole number of heads: ", name);
                const int64_t heads = rows / per_head;
                const int64_t esz = TensorDtypeBytes(where.first->dtype);
                const uint8_t *src = data.data();
                std::vector<uint8_t> nope(
                    static_cast<size_t>(heads * cfg.qk_nope * cols * esz));
                std::vector<uint8_t> value(
                    static_cast<size_t>(heads * cfg.v_head * cols * esz));
                for (int64_t h = 0; h < heads; ++h) {
                    std::memcpy(
                        nope.data() + h * cfg.qk_nope * cols * esz,
                        src + (h * per_head) * cols * esz,
                        static_cast<size_t>(cfg.qk_nope * cols * esz));
                    std::memcpy(
                        value.data() + h * cfg.v_head * cols * esz,
                        src + (h * per_head + cfg.qk_nope) * cols * esz,
                        static_cast<size_t>(cfg.v_head * cols * esz));
                }
                DenseNvfp4Record rec;
                rec.name = name;
                rec.shape = {heads * cfg.qk_nope, cols};
                rec.dtype = where.first->dtype;
                if (!QuantizeSlice(nope.data(), rec.shape[0] * rec.shape[1],
                                   rec.dtype, rec))
                    return arrow::Status::IOError(
                        "GPU quantization failed for ", name);
                DenseNvfp4Record val;
                val.name = name + ".value";
                val.shape = {heads * cfg.v_head, cols};
                val.dtype = where.first->dtype;
                if (!QuantizeSlice(value.data(), val.shape[0] * val.shape[1],
                                   val.dtype, val))
                    return arrow::Status::IOError(
                        "GPU quantization failed for ", name, ".value");
                nvfp4_by_layer[p.layer].push_back(std::move(rec));
                nvfp4_by_layer[p.layer].push_back(std::move(val));
                total_pinned += 2;
                continue;
            }
            DenseNvfp4Record rec;
            rec.name = name;
            rec.shape = where.first->shape;
            rec.dtype = where.first->dtype;
            if (!QuantizeSlice(data.data(), where.first->numel(),
                               where.first->dtype, rec))
                return arrow::Status::IOError("GPU quantization failed for ",
                                              name);
            nvfp4_by_layer[p.layer].push_back(std::move(rec));
            ++total_pinned;
            continue;
        }
        TensorRecord rec;
        rec.name = name;
        rec.dtype = where.first->dtype;
        rec.shape = where.first->shape;
        rec.data = std::move(data);
        if (p.kind == TensorKind::kGlobalRaw) {
            globals.push_back(std::move(rec));
        } else {
            raw_by_layer[p.layer].push_back(std::move(rec));
        }
        ++total_pinned;
    }

    ARROW_RETURN_NOT_OK(SaveDense(globals, raw_by_layer, nvfp4_by_layer, log_root));

    // (3) Routed experts. The checkpoint stores each layer's experts as TWO
    // 3-D tensors ([E, 2*I, latent] gate_up and [E, latent, I] down), so one
    // read per layer feeds all 512 per-expert records.
    int64_t total_experts = 0;
    const int64_t gate_up_stride =
        static_cast<int64_t>(2) * cfg.moe_inter * cfg.latent;
    const int64_t down_stride = static_cast<int64_t>(cfg.latent) * cfg.moe_inter;
    for (int l = 0; l < cfg.layers; ++l) {
        if (!shapes[static_cast<size_t>(l)].is_moe) continue;
        const std::string up_name = absl::StrCat(
            "model.layers.", l, ".mlp.experts.gate_up_proj");
        const std::string dn_name = absl::StrCat(
            "model.layers.", l, ".mlp.experts.down_proj");
        auto it_up = by_name.find(up_name);
        auto it_dn = by_name.find(dn_name);
        if (it_up == by_name.end() || it_dn == by_name.end())
            return arrow::Status::Invalid(
                "MoE layer ", l, " is missing its packed expert tensors");
        const TensorInfo &up_info = *it_up->second.first;
        const TensorInfo &dn_info = *it_dn->second.first;
        if (up_info.shape.size() != 3 || dn_info.shape.size() != 3)
            return arrow::Status::Invalid("expert tensors must be 3-D (layer ",
                                          l, ")");
        if (up_info.shape[0] != cfg.experts || dn_info.shape[0] != cfg.experts)
            return arrow::Status::Invalid("expert count mismatch at layer ", l);
        if (up_info.shape[1] != 2 * cfg.moe_inter ||
            up_info.shape[2] != cfg.latent)
            return arrow::Status::Invalid("gate_up shape is not [E, 2I, latent] "
                                          "at layer ", l);
        if (dn_info.shape[1] != cfg.latent || dn_info.shape[2] != cfg.moe_inter)
            return arrow::Status::Invalid("down shape is not [E, latent, I] at "
                                          "layer ", l);

        auto up_data = ReadTensorData(input_path, *it_up->second.second, up_info);
        auto dn_data = ReadTensorData(input_path, *it_dn->second.second, dn_info);
        if (up_data.empty() || dn_data.empty())
            return arrow::Status::IOError("failed to read expert tensors at "
                                          "layer ", l);
        if (static_cast<int64_t>(up_data.size()) !=
                gate_up_stride * cfg.experts * 2 ||
            static_cast<int64_t>(dn_data.size()) !=
                down_stride * cfg.experts * 2)
            return arrow::Status::Invalid("expert tensor byte size mismatch at "
                                          "layer ", l);

        const int64_t half = gate_up_stride / 2;  // one projection [I, latent]
        for (int e = 0; e < cfg.experts; ++e) {
            DenseNvfp4Record w1, w2, w3;
            w1.name = absl::StrCat("w1.");
            w1.shape = {cfg.moe_inter, cfg.latent};
            w1.dtype = up_info.dtype;
            w2.name = absl::StrCat("w2.");
            w2.shape = {cfg.latent, cfg.moe_inter};
            w2.dtype = dn_info.dtype;
            w3.name = absl::StrCat("w3.");
            w3.shape = {cfg.moe_inter, cfg.latent};
            w3.dtype = up_info.dtype;
            const uint8_t *gu = up_data.data() + e * gate_up_stride * 2;
            const uint8_t *dn = dn_data.data() + e * down_stride * 2;
            if (!QuantizeSliceBf16(gu, half, w1) ||
                !QuantizeSliceBf16(gu + half * 2, half, w3) ||
                !QuantizeSliceBf16(dn, down_stride, w2))
                return arrow::Status::IOError(
                    "GPU expert quantization failed at layer ", l, " expert ",
                    e);
            PackedExpert pe;
            pe.w1_packed = std::move(w1.packed);
            pe.w1_scales = std::move(w1.scales);
            pe.w1_shape = w1.shape;
            pe.w1_global_scale = w1.global_scale;
            pe.w2_packed = std::move(w2.packed);
            pe.w2_scales = std::move(w2.scales);
            pe.w2_shape = w2.shape;
            pe.w2_global_scale = w2.global_scale;
            pe.w3_packed = std::move(w3.packed);
            pe.w3_scales = std::move(w3.scales);
            pe.w3_shape = w3.shape;
            pe.w3_global_scale = w3.global_scale;
            ARROW_RETURN_NOT_OK(WriteExpertArrow(ExpertArrowPath(log_root, l, e), pe));
            ++total_experts;
        }
        lsxcommon::log::info(absl::StrFormat(
            "[hy4-convert] layer %d/%d experts packed (%lld total)", l + 1,
            cfg.layers, total_experts));
    }

    // (4) Tokenizer (byte-level BPE, Hunyuan pre-tokenizer family).
    const fs::path tok_json = fs::path(input_path) / "tokenizer.json";
    if (fs::exists(tok_json)) {
        TokenizerSpecialPolicy policy;
        // The chat template spells every marker out itself and the reference
        // tokenizer prepends nothing, so no automatic BOS.
        policy.bos_id = -1;
        policy.pretokenizer = 11;  // HunyuanWordStream -> tokenizer_kind 13
        if (cfg.eos >= 0) policy.eos_id = cfg.eos;
        if (cfg.pad >= 0) policy.pad_id = cfg.pad;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_json.string(), log_root / "pinned",
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true,
                                           .pad_required = true},
            policy));
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    }

    // (5) Conversion-time VRAM profile. Routed experts are streamed, so they are
    // classified as such; everything else is resident.
    VramGeometry geom;
    geom.hidden_dim = cfg.hidden;
    geom.kv_dim = cfg.kv_lora + cfg.qk_rope;
    geom.num_attn_layers = cfg.layers;
    geom.num_moe_layers = frozen::MoeLayerCount();
    geom.experts_per_layer = cfg.experts;
    geom.context_length = cfg.context;
    const VramProfile vram = EstimateVram(
        inventory,
        {+[](const std::string &n) {
             return IsDenseNvfp4(ParseHy4Tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return IsRoutedExpert(ParseHy4Tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         +[](const std::string &) { return false; }},
        geom);

    ARROW_RETURN_NOT_OK(WriteIndex(cfg, shapes, total_pinned, total_experts,
                                   vram, cuts, CardNameFor(input_path),
                                   log_root));

    lsxcommon::log::info(absl::StrFormat(
        "[hy4-convert] done: %lld pinned tensors, %lld experts, deferred "
        "(engram=%lld vision=%lld dfly=%lld)", total_pinned, total_experts,
        cuts.engram, cuts.vision, cuts.dfly));
    return arrow::Status::OK();
}

} // namespace hy4
} // namespace lsxcommon
