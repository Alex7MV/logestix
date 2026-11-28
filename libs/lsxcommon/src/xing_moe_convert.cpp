// Xing4.0-29B-A4B classifier + topology (Model Ingestion Protocol step 1).
// Names are parsed straight out of the source weight_map; no safetensors bytes
// are touched here, so the same code serves the converter and the loader's
// geometry cross-check.

#include "lsxcommon/xing_moe_convert.h"

#include "xing_moe_constants.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/match.h>
#include <absl/strings/strip.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace lsxcommon {
namespace xing {
namespace {

namespace fs = std::filesystem;

int WeightIdFor(const std::string &proj)
{
    if (proj == "gate_proj") return 1;
    if (proj == "up_proj") return 2;
    if (proj == "down_proj") return 3;
    return 0;
}

// Splits "model.layers.<N>.<rest>" into its parts. Returns false for any name
// that is not a decoder-layer tensor (globals, MTP handled by the caller).
bool SplitLayerName(const std::string &name, int &layer, std::string &rest)
{
    const std::string kPrefix = "model.layers.";
    if (!absl::StartsWith(name, kPrefix)) return false;
    const size_t dot = name.find('.', kPrefix.size());
    if (dot == std::string::npos) return false;
    if (!absl::SimpleAtoi(name.substr(kPrefix.size(), dot - kPrefix.size()),
                          &layer))
        return false;
    rest = name.substr(dot + 1);
    return true;
}

std::optional<XingParsedTensor> ParseMoeExpert(const std::string &rest)
{
    const std::string kP = "mlp.experts.";
    if (!absl::StartsWith(rest, kP)) return std::nullopt;
    const size_t dot = rest.find('.', kP.size());
    if (dot == std::string::npos) return std::nullopt;
    XingParsedTensor p;
    if (!absl::SimpleAtoi(rest.substr(kP.size(), dot - kP.size()), &p.expert))
        return std::nullopt;
    const std::string tail = rest.substr(dot + 1);
    if (!absl::EndsWith(tail, ".weight")) return std::nullopt;
    const std::string proj = tail.substr(0, tail.size() - 7);
    if (WeightIdFor(proj) == 0) return std::nullopt;
    p.kind = XingTensorKind::RoutedExpert;
    p.weight_id = WeightIdFor(proj);
    p.base = absl::StrCat("mlp.experts.", p.expert, ".", proj);
    return p;
}

std::optional<XingParsedTensor> ParseSharedExpert(const std::string &rest)
{
    const std::string kP = "mlp.shared_experts.";
    if (!absl::StartsWith(rest, kP)) return std::nullopt;
    const std::string tail = rest.substr(kP.size());
    if (!absl::EndsWith(tail, ".weight")) return std::nullopt;
    const std::string proj = tail.substr(0, tail.size() - 7);
    if (WeightIdFor(proj) == 0) return std::nullopt;
    XingParsedTensor p;
    p.kind = XingTensorKind::SharedExpert;
    p.weight_id = WeightIdFor(proj);
    p.base = absl::StrCat("mlp.shared_experts.", proj);
    return p;
}

std::optional<XingParsedTensor> ParseRouter(const std::string &rest)
{
    if (rest == "mlp.gate.weight")
    {
        XingParsedTensor p;
        p.kind = XingTensorKind::Router;
        p.base = "mlp.gate.weight";
        return p;
    }
    if (rest == "mlp.gate.e_score_correction_bias")
    {
        XingParsedTensor p;
        p.kind = XingTensorKind::Router;
        p.base = rest;
        return p;
    }
    return std::nullopt;
}

std::optional<XingParsedTensor> ParseDenseMlp(const std::string &rest)
{
    static const std::vector<std::string> kProjs = {"gate_proj", "up_proj",
                                                    "down_proj"};
    for (const auto &proj : kProjs)
    {
        const std::string full = absl::StrCat("mlp.", proj, ".weight");
        if (rest == full)
        {
            XingParsedTensor p;
            p.kind = XingTensorKind::DenseMlp;
            p.weight_id = WeightIdFor(proj);
            p.base = proj;
            return p;
        }
    }
    return std::nullopt;
}

std::optional<XingParsedTensor> ParseAttention(const std::string &rest)
{
    // `base` follows the family convention (see kimi_moe_convert.h): for a
    // projection it is the bare projection id, so the loader can key on
    // {q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj}.
    static const std::vector<std::string> kProjs = {
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"};
    for (const auto &proj : kProjs)
    {
        const std::string full = absl::StrCat("self_attn.", proj, ".weight");
        if (rest == full)
        {
            XingParsedTensor p;
            p.kind = XingTensorKind::Attention;
            p.base = proj;
            return p;
        }
    }
    return std::nullopt;
}

// attn_hc / ffn_hc .hc_fn / .hc_base / .hc_scale. The mHC input norm is
// deliberately weight-free upstream, so a "...input_norm.weight" name is a
// different architecture and must NOT be classified here.
std::optional<XingParsedTensor> ParseHyperConnection(const std::string &rest)
{
    static const std::vector<std::string> kBlocks = {"attn_hc", "ffn_hc"};
    static const std::vector<std::string> kParts = {".hc_fn", ".hc_base",
                                                    ".hc_scale"};
    for (const auto &block : kBlocks)
    {
        for (const auto &part : kParts)
        {
            const std::string full = absl::StrCat(block, part);
            if (rest == full)
            {
                XingParsedTensor p;
                p.kind = XingTensorKind::HyperConnection;
                p.base = full;
                return p;
            }
        }
    }
    return std::nullopt;
}

std::optional<XingParsedTensor> ParseLayerNorm(const std::string &rest)
{
    static const std::vector<std::string> kNorms = {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight"};
    for (const auto &full : kNorms)
    {
        if (rest == full)
        {
            XingParsedTensor p;
            p.kind = XingTensorKind::LayerNorm;
            p.base = full;
            return p;
        }
    }
    return std::nullopt;
}

// The stored MTP block is a FULL decoder layer plus the speculative head:
// model.layers.40 carries its own MLA projections, a 64-expert MoE and the
// router, next to the MTP-only tensors (embed_tokens/enorm/hnorm/eh_proj/
// shared_head). Classifying by tensor identity would therefore scatter it
// across the resident graph, so the block is keyed by layer INDEX: every
// tensor at or above num_hidden_layers belongs to MTP.
bool IsMtpLayer(int layer)
{
    return layer >= lsxcommon::xing::frozen::kLayers;
}

} // namespace

XingParsedTensor xing_classify(const std::string &name)
{
    XingParsedTensor out;
    if (name == "model.embed_tokens.weight" || name == "lm_head.weight" ||
        name == "model.norm.weight")
    {
        out.kind = XingTensorKind::Global;
        out.base = name;
        return out;
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayerName(name, layer, rest)) return out;

    if (IsMtpLayer(layer))
    {
        out.kind = XingTensorKind::Mtp;
        out.layer = layer;
        out.base = rest;
        return out;
    }

    if (auto p = ParseMoeExpert(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseSharedExpert(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseRouter(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseDenseMlp(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseAttention(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseHyperConnection(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseLayerNorm(rest))
    {
        p->layer = layer;
        return *p;
    }
    return out;
}

XingTopology xing_load_topology(const std::string &index_json_path)
{
    namespace frozen = lsxcommon::xing::frozen;
    XingTopology top;

    std::ifstream f(index_json_path);
    if (!f)
    {
        top.error_msg = "cannot open " + index_json_path;
        return top;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    auto scan_lists = [&](const char *key, std::vector<long long> &into) {
        const std::string k = absl::StrCat("\"", key, "\":");
        size_t pos = text.find(k);
        if (pos == std::string::npos) return;
        pos = text.find('[', pos);
        if (pos == std::string::npos) return;
        const size_t end = text.find(']', pos);
        if (end == std::string::npos) return;
        for (const auto &part :
             absl::StrSplit(text.substr(pos + 1, end - pos - 1), ','))
        {
            const std::string t{std::string(absl::StripAsciiWhitespace(part))};
            if (!t.empty()) into.push_back(std::strtoll(t.c_str(), nullptr, 10));
        }
    };

    // Converted index (round-trip through the loader) or HF source index
    // (weight_map scan): both expose the layer lists under the same keys.
    if (text.find("\"moe_layers\"") != std::string::npos)
    {
        auto scalar = [&](const char *key, long long &into) {
            const size_t k = text.find(absl::StrCat("\"", key, "\""));
            if (k == std::string::npos) return;
            const size_t colon = text.find(':', k);
            if (colon == std::string::npos) return;
            const std::string tail{
                std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 24)))};
            into = std::strtoll(tail.c_str(), nullptr, 10);
        };
        scalar("num_layers", top.num_layers);
        scalar("experts_per_layer", top.experts_per_layer);
        scalar("hc_mult", top.hc_mult);
        scan_lists("moe_layers", top.moe_layers);
        scan_lists("dense_layers", top.dense_layers);
        scan_lists("mtp_layers", top.mtp_layers);
        // layout is an integer; the old string tag is gone (see index_layout.h).
        {
            const size_t fp = text.find("\"layout\"");
            if (fp != std::string::npos) {
                const size_t c = text.find(':', fp);
                if (c != std::string::npos)
                    top.layout = std::atoll(text.c_str() + c + 1);
            }
        }
        std::sort(top.moe_layers.begin(), top.moe_layers.end());
        std::sort(top.dense_layers.begin(), top.dense_layers.end());
        std::sort(top.mtp_layers.begin(), top.mtp_layers.end());
        return top;
    }

    // HF source: walk the weight_map keys.
    std::set<long long> moe, dense, mtp;
    long long experts = 0;
    size_t pos = 0;
    while ((pos = text.find("\"model.layers.", pos)) != std::string::npos)
    {
        const size_t key_start = pos + 1;
        const size_t key_end = text.find('"', key_start + 14);
        if (key_end == std::string::npos) break;
        const std::string key = text.substr(key_start, key_end - key_start);
        pos = key_end;
        const auto t = xing_classify(key);
        switch (t.kind)
        {
        case XingTensorKind::RoutedExpert:
            moe.insert(t.layer);
            experts = std::max(experts, static_cast<long long>(t.expert) + 1);
            break;
        case XingTensorKind::DenseMlp:
            dense.insert(t.layer);
            break;
        case XingTensorKind::Mtp:
            mtp.insert(t.layer);
            break;
        default:
            break;
        }
    }
    if (moe.empty() && dense.empty())
    {
        top.error_msg = "no decoder layers found in " + index_json_path;
        return top;
    }

    top.moe_layers.assign(moe.begin(), moe.end());
    top.dense_layers.assign(dense.begin(), dense.end());
    top.mtp_layers.assign(mtp.begin(), mtp.end());
    top.experts_per_layer = experts;
    top.num_layers = static_cast<long long>(top.moe_layers.size() +
                                            top.dense_layers.size());

    // Frozen-geometry gate: a real checkpoint must match the passport, a
    // synthetic fixture is allowed to shrink (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY).
    const bool relaxed = lsxcommon::SyntheticGeometryAllowed();
    if (!relaxed && (top.num_layers != frozen::kLayers ||
                     top.experts_per_layer != frozen::kNumRouted))
    {
        top.error_msg = absl::StrCat(
            "frozen geometry mismatch: layers=", top.num_layers,
            " (expected ", frozen::kLayers, "), experts=", top.experts_per_layer,
            " (expected ", frozen::kNumRouted, ")");
        return top;
    }
    return top;
}

namespace {

// ── config.json ────────────────────────────────────────────────────────────
struct XingConfig {
    long long hidden = 0;
    long long layers = 0;
    long long vocab = 0;
    long long heads = 0;
    long long kv_heads = 0;
    long long q_lora = 0;
    long long kv_lora = 0;
    long long qk_nope = 0;
    long long qk_rope = 0;
    long long v_head = 0;
    long long first_k_dense = 0;
    long long moe_intermediate = 0;
    long long dense_intermediate = 0;
    long long num_experts = 0;
    long long top_k = 0;
    long long n_shared = 0;
    double routed_scale = 1.0;
    bool norm_topk_prob = false;
    bool sigmoid_score = false;
    bool noaux_tc = false;
    long long hc_mult = 0;
    long long hc_sinkhorn_iters = 0;
    double hc_eps = 0.0;
    double clamp_min = 0.0;
    double clamp_max = 0.0;
    bool rope_interleave = false;
    long long bos_id = -1;
    long long eos_id = -1;
    long long context_length = 0;
    std::string error_msg;
};

long long JsonInt(const std::string &text, const std::string &key,
                  long long fallback)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return fallback;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return fallback;
    return std::strtoll(
        std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 32)))
            .c_str(),
        nullptr, 10);
}

double JsonDouble(const std::string &text, const std::string &key,
                  double fallback)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return fallback;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return fallback;
    return std::strtod(
        std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 32)))
            .c_str(),
        nullptr);
}

bool JsonBool(const std::string &text, const std::string &key, bool fallback)
{
    const size_t k = text.find(absl::StrCat("\"", key, "\""));
    if (k == std::string::npos) return fallback;
    const size_t colon = text.find(':', k);
    if (colon == std::string::npos) return fallback;
    const std::string tail{
        std::string(absl::StripAsciiWhitespace(text.substr(colon + 1, 16)))};
    if (tail.rfind("true", 0) == 0) return true;
    if (tail.rfind("false", 0) == 0) return false;
    return fallback;
}

XingConfig ParseConfig(const std::string &text)
{
    XingConfig c;
    c.hidden = JsonInt(text, "hidden_size", 0);
    c.layers = JsonInt(text, "num_hidden_layers", 0);
    c.vocab = JsonInt(text, "vocab_size", 0);
    c.heads = JsonInt(text, "num_attention_heads", 0);
    c.kv_heads = JsonInt(text, "num_key_value_heads", 0);
    c.q_lora = JsonInt(text, "q_lora_rank", 0);
    c.kv_lora = JsonInt(text, "kv_lora_rank", 0);
    c.qk_nope = JsonInt(text, "qk_nope_head_dim", 0);
    c.qk_rope = JsonInt(text, "qk_rope_head_dim", 0);
    c.v_head = JsonInt(text, "v_head_dim", 0);
    c.first_k_dense = JsonInt(text, "first_k_dense_replace", 0);
    c.moe_intermediate = JsonInt(text, "moe_intermediate_size", 0);
    c.dense_intermediate = JsonInt(text, "intermediate_size", 0);
    c.num_experts = JsonInt(text, "n_routed_experts", 0);
    c.top_k = JsonInt(text, "num_experts_per_tok", 0);
    c.n_shared = JsonInt(text, "n_shared_experts", 0);
    c.routed_scale = JsonDouble(text, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = JsonBool(text, "norm_topk_prob", false);
    c.sigmoid_score = text.find("\"scoring_func\": \"sigmoid\"") != std::string::npos;
    c.noaux_tc = text.find("\"topk_method\": \"noaux_tc\"") != std::string::npos;
    c.hc_mult = JsonInt(text, "hc_mult", 0);
    c.hc_sinkhorn_iters = JsonInt(text, "hc_sinkhorn_iters", 0);
    c.hc_eps = JsonDouble(text, "hc_eps", 0.0);
    c.clamp_min = JsonDouble(text, "mhc_h_res_clamp_min", 0.0);
    c.clamp_max = JsonDouble(text, "mhc_h_res_clamp_max", 0.0);
    // Absent means the Xing4_0Config dataclass default (True): config.json
    // only spells the key out when it deviates, so defaulting to false would
    // reject the real checkpoint as a non-frozen architecture.
    c.rope_interleave = JsonBool(text, "rope_interleave", true);
    c.bos_id = JsonInt(text, "bos_token_id", -1);
    c.eos_id = JsonInt(text, "eos_token_id", -1);
    c.context_length = JsonInt(text, "max_position_embeddings", 0);
    if (c.hidden <= 0 || c.layers <= 0 || c.vocab <= 0)
        c.error_msg = "config.json missing required geometry";
    return c;
}

struct TensorRef {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

// One routed-expert projection as read from the source shards (Xing ships
// plain BF16, so there is no paired source scale tensor).
struct RawProjection {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

struct ExpertParts {
    RawProjection gate, up, down;
};

std::string LayerFile(int layer, const char *suffix)
{
    // Non-padded, matching the experts/layer_N/ directory naming.
    return absl::StrCat("layer_", layer, suffix);
}

arrow::Status WriteBatchOr(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &b,
    const std::filesystem::path &path)
{
    if (!b.ok()) return b.status();
    return WriteRecordBatchToFile(*b, path);
}

// BF16/FP32 -> NVFP4 for one dense projection.
arrow::Status QuantizeDense(const std::string &input_dir, const TensorRef &w,
                            DenseNvfp4Record &out)
{
    const auto raw = ReadTensorData(input_dir, *w.shard, *w.info);
    if (raw.empty() ||
        static_cast<int64_t>(raw.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ", w.info->name);
    out.name = w.info->name;
    out.shape = w.info->shape;
    out.dtype = w.info->dtype;
    if (!gpu::quantize_tensor_gpu(raw.data(), w.info->numel(), w.info->dtype,
                                  nullptr, w.info->shape, {}, out.packed,
                                  out.scales, out.global_scale))
        return arrow::Status::IOError("GPU quantization failed for ", w.info->name);
    return arrow::Status::OK();
}

arrow::Status RawRecord(const std::string &input_dir, const TensorRef &w,
                        TensorRecord &out)
{
    const auto raw = ReadTensorData(input_dir, *w.shard, *w.info);
    if (raw.empty() ||
        static_cast<int64_t>(raw.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ", w.info->name);
    out.name = w.info->name;
    out.dtype = w.info->dtype;
    out.shape = w.info->shape;
    out.data = raw;
    return arrow::Status::OK();
}

} // namespace

arrow::Status ConvertXingModel(const std::string &input_dir,
                               const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    if (input_dir.empty() || output_root.empty())
        return arrow::Status::Invalid("input/output path not set");

    const auto cfg_path = fs::path(input_dir) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_dir);
    std::ifstream cf(cfg_path);
    if (!cf) return arrow::Status::IOError("cannot open ", cfg_path.string());
    std::string cfg_text((std::istreambuf_iterator<char>(cf)),
                         std::istreambuf_iterator<char>());
    const XingConfig cfg = ParseConfig(cfg_text);
    if (!cfg.error_msg.empty())
        return arrow::Status::Invalid("Xing config error: ", cfg.error_msg);

    if (!SyntheticGeometryAllowed())
    {
        // Name every mismatching field: a single "geometry mismatch" string
        // cost a real conversion run before this diagnostic existed.
        std::vector<std::string> bad;
        const auto chk = [&bad](const char *name, long long got, long long want) {
            if (got != want) bad.push_back(absl::StrFormat("%s=%lld!=%lld", name, got, want));
        };
        chk("hidden", cfg.hidden, frozen::kHidden);
        chk("layers", cfg.layers, frozen::kLayers);
        chk("vocab", cfg.vocab, frozen::kVocab);
        chk("heads", cfg.heads, frozen::kHeads);
        chk("kv_heads", cfg.kv_heads, frozen::kKvHeads);
        chk("q_lora", cfg.q_lora, frozen::kQLoraRank);
        chk("kv_lora", cfg.kv_lora, frozen::kKvLoraRank);
        chk("qk_nope", cfg.qk_nope, frozen::kQkNopeHeadDim);
        chk("qk_rope", cfg.qk_rope, frozen::kQkRopeHeadDim);
        chk("v_head", cfg.v_head, frozen::kVHeadDim);
        chk("first_k_dense", cfg.first_k_dense, frozen::kFirstKDense);
        chk("moe_intermediate", cfg.moe_intermediate, frozen::kMoeIntermediate);
        chk("dense_intermediate", cfg.dense_intermediate, frozen::kDenseIntermediate);
        chk("num_experts", cfg.num_experts, frozen::kNumRouted);
        chk("top_k", cfg.top_k, frozen::kTopK);
        chk("n_shared", cfg.n_shared, frozen::kSharedExperts);
        chk("hc_mult", cfg.hc_mult, frozen::kHcMult);
        chk("hc_sinkhorn_iters", cfg.hc_sinkhorn_iters, frozen::kHcSinkhornIters);
        // llround: 1e-6f * 1e6 truncates to 0 in float, which reported a false
        // mismatch on the real checkpoint.
        chk("hc_eps", std::llround(cfg.hc_eps * 1e6),
            std::llround(static_cast<double>(frozen::kHcEps) * 1e6));
        if (cfg.clamp_min != frozen::kHcClampMin || cfg.clamp_max != frozen::kHcClampMax)
            bad.push_back(absl::StrFormat("clamp=[%g,%g]!=[%g,%g]", cfg.clamp_min,
                                          cfg.clamp_max, frozen::kHcClampMin,
                                          frozen::kHcClampMax));
        if (cfg.rope_interleave != frozen::kRopeInterleave)
            bad.push_back(absl::StrFormat("rope_interleave=%d", cfg.rope_interleave));
        if (cfg.norm_topk_prob != frozen::kNormTopkProb)
            bad.push_back("norm_topk_prob");
        if (cfg.sigmoid_score != frozen::kSigmoidScore)
            bad.push_back("scoring_func");
        if (cfg.noaux_tc != frozen::kNoauxTc) bad.push_back("topk_method");
        if (cfg.routed_scale != frozen::kRoutedScale)
            bad.push_back(absl::StrFormat("routed_scale=%g", cfg.routed_scale));
        if (!bad.empty())
            return arrow::Status::Invalid(
                "Xing frozen geometry mismatch: ",
                absl::StrJoin(bad, ", "),
                " (set LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 only for synthetic tests)");
    }

    auto model = scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    absl::flat_hash_map<std::string, TensorRef> refs;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            refs.insert_or_assign(t.name, TensorRef{&shard, &t});

    // Every decoder tensor must classify; Unknown is stripped ballast.
    absl::flat_hash_map<int, int> layer_tensor_count;
    int mtp_tensor_count = 0;
    std::vector<std::string> skipped;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
        {
            const auto p = xing_classify(t.name);
            if (p.kind == XingTensorKind::Unknown)
            {
                skipped.push_back(t.name);
                continue;
            }
            if (p.kind == XingTensorKind::Mtp)
            {
                ++mtp_tensor_count;
                continue;
            }
            if (p.layer >= 0) ++layer_tensor_count[p.layer];
        }
    // Observed layer set drives the sweep (a synthetic fixture is smaller than
    // the frozen 40); a real checkpoint must cover exactly 0..layers-1.
    int observed_layers = 0;
    for (const auto &[l, n] : layer_tensor_count)
    {
        if (n == 0)
            return arrow::Status::Invalid("layer ", l, " has no tensors");
        observed_layers = std::max(observed_layers, l + 1);
    }
    if (observed_layers == 0)
        return arrow::Status::Invalid("no decoder layers in ", input_dir);
    if (!SyntheticGeometryAllowed() && observed_layers != cfg.layers)
        return arrow::Status::Invalid(
            "checkpoint has ", observed_layers, " decoder layers, expected ",
            cfg.layers);

    const fs::path base = output_root;
    const fs::path pinned_dir = base / "pinned";
    // No mtp/: v1 ships no ballast (design §1.4). The MTP layer is counted in
    // the index (`mtp_tensors`) and not written, so every plane in the tree is
    // one the load path can reach.
    for (const fs::path &d : {base, pinned_dir})
    {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::flat_hash_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::flat_hash_map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    // layer -> {record base -> global_scale} for the loader's dequant path.
    absl::flat_hash_map<int, std::vector<std::pair<std::string, float>>> layer_scales;

    // The mHC block and the norms/router bias stay BF16/F32; every linear
    // matrix (MLA, dense MLP, shared expert, embed/lm_head, router) goes NVFP4.
    absl::flat_hash_map<int64_t, absl::flat_hash_map<int, ExpertParts>> experts;

    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            const auto p = xing_classify(t.name);
            const TensorRef ref{&shard, &t};
            switch (p.kind)
            {
            case XingTensorKind::Unknown:
                break;
            case XingTensorKind::RoutedExpert:
            {
                ExpertParts &e = experts[p.layer][p.expert];
                switch (p.weight_id)
                {
                case 1:
                    e.gate = {t.name, t.dtype, t.shape, ReadTensorData(input_dir, shard, t)};
                    break;
                case 2:
                    e.up = {t.name, t.dtype, t.shape, ReadTensorData(input_dir, shard, t)};
                    break;
                case 3:
                    e.down = {t.name, t.dtype, t.shape, ReadTensorData(input_dir, shard, t)};
                    break;
                default:
                    break;
                }
                break;
            }
            case XingTensorKind::SharedExpert:
            {
                DenseNvfp4Record rec;
                ARROW_RETURN_NOT_OK(QuantizeDense(input_dir, ref, rec));
                if (p.layer >= 0)
                {
                    nvfp4_by_layer[p.layer].push_back(std::move(rec));
                    layer_scales[p.layer].push_back(
                        {rec.name, rec.global_scale});
                }
                break;
            }
            case XingTensorKind::Router:
            {
                // e_score_correction_bias is kept in FP32 exactly (the source
                // lists it in _keep_in_fp32_modules_strict); the router weight
                // is a real linear map and is quantized like any other.
                // BOTH router tensors stay high precision. The weight is a
                // [64, 3584] map whose ARGMAX decides which experts run: NVFP4
                // rounding there does not blur an output, it silently swaps the
                // expert set and produces fluent garbage. Every other family
                // keeps its router in BF16 for the same reason.
                TensorRecord r;
                ARROW_RETURN_NOT_OK(RawRecord(input_dir, ref, r));
                raw_by_layer[p.layer].push_back(std::move(r));
                break;
            }
            case XingTensorKind::DenseMlp:
            case XingTensorKind::Attention:
            {
                DenseNvfp4Record rec;
                ARROW_RETURN_NOT_OK(QuantizeDense(input_dir, ref, rec));
                nvfp4_by_layer[p.layer].push_back(std::move(rec));
                layer_scales[p.layer].push_back({rec.name, rec.global_scale});
                break;
            }
            case XingTensorKind::HyperConnection:
            case XingTensorKind::LayerNorm:
            {
                TensorRecord r;
                ARROW_RETURN_NOT_OK(RawRecord(input_dir, ref, r));
                raw_by_layer[p.layer].push_back(std::move(r));
                break;
            }
            case XingTensorKind::Global:
            {
                if (t.name == "model.norm.weight")
                {
                    TensorRecord r;
                    ARROW_RETURN_NOT_OK(RawRecord(input_dir, ref, r));
                    // The final norm rides in the raw pinned batch.
                    raw_by_layer[-1].push_back(std::move(r));
                }
                else
                {
                    DenseNvfp4Record rec;
                    ARROW_RETURN_NOT_OK(QuantizeDense(input_dir, ref, rec));
                    global_nvfp4.push_back(std::move(rec));
                }
                break;
            }
            case XingTensorKind::Mtp:
                // Never applied at inference, so never stored. Already counted
                // by the classification pass above (the index reports it).
                break;
            }
        }
    }

    // Routed experts: one NVFP4 record triple per (layer, expert).
    int total_experts = 0;
    for (auto &[layer, per_expert] : experts)
    {
        for (auto &[expert, e] : per_expert)
        {
            if (e.gate.data.empty() || e.up.data.empty() || e.down.data.empty())
                return arrow::Status::Invalid(
                    "incomplete expert ", layer, "/", expert);
            ExpertData ed;
            ed.w1_raw = e.gate.data; ed.w1_dtype = e.gate.dtype; ed.w1_shape = e.gate.shape;
            ed.w3_raw = e.up.data; ed.w3_dtype = e.up.dtype; ed.w3_shape = e.up.shape;
            ed.w2_raw = e.down.data; ed.w2_dtype = e.down.dtype; ed.w2_shape = e.down.shape;
            auto packed = PackExpertNVFP4(ed);
            if (!packed.ok()) return packed.status();
            const fs::path dir = base / "experts" / absl::StrCat("layer_", layer);
            std::error_code ec;
            fs::create_directories(dir, ec);
            ARROW_RETURN_NOT_OK(
                WriteBatchOr(ExpertSchema::MakeBatch(*packed),
                             dir / absl::StrCat("exp_", expert, ".arrow")));
            ++total_experts;
        }
    }

    if (!global_nvfp4.empty())
        ARROW_RETURN_NOT_OK(WriteBatchOr(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                                         pinned_dir / "global_nvfp4.arrow"));
    // The final norm is the only global raw tensor: keep it in its own file so
    // the loader never has to merge two schemas under the same name.
    auto final_norm = raw_by_layer.find(-1);
    if (final_norm != raw_by_layer.end() && !final_norm->second.empty())
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(final_norm->second),
                                         pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : raw_by_layer)
    {
        if (layer < 0) continue;
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(recs),
                                         pinned_dir / LayerFile(layer, ".arrow")));
    }
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(DenseNvfp4Schema::MakeBatch(recs),
                                         pinned_dir / LayerFile(layer, "_nvfp4.arrow")));

    // Tokenizer: SentencePiece-BPE (tokenizer.model). Synthetic fixtures omit
    // it; a real checkpoint must carry it.
    const fs::path spm_path = fs::path(input_dir) / "tokenizer.model";
    if (fs::exists(spm_path))
    {
        // Xing spells its specials <_start>/<_end>/<_pad> (tokenizer_config.json
        // added_tokens_decoder ids 1/2/3) — the LLaMA-style defaults the shared
        // sp-bpe builder used to assume would resolve to -1 and fail the
        // integrity gate.
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizerSentencePiece(
            spm_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = true,
                                           .eos_required = true,
                                           .pad_required = true},
            SpSpecialTokens{"<_start>", "<_end>", "<_pad>", "<unk>"}));
    }
    else if (!SyntheticGeometryAllowed())
    {
        return arrow::Status::Invalid("tokenizer.model not found in ", input_dir);
    }

    // Index.
    std::vector<long long> dense_layers, moe_layers;
    for (int l = 0; l < observed_layers; ++l)
    {
        const auto it = raw_by_layer.find(l);
        bool has_router = false;
        if (it != raw_by_layer.end())
            has_router = absl::c_any_of(
                it->second,
                [](const TensorRecord &r) {
                    return absl::EndsWith(r.name, ".e_score_correction_bias");
                });
        (has_router ? moe_layers : dense_layers).push_back(l);
    }
    int total_pinned =
        static_cast<int>(global_nvfp4.size()) +
        static_cast<int>(raw_by_layer[-1].size());
    for (const auto &[l, v] : raw_by_layer)
        if (l >= 0) total_pinned += static_cast<int>(v.size());
    for (const auto &[l, v] : nvfp4_by_layer) total_pinned += static_cast<int>(v.size());

    std::ofstream f(base / "model.index.json");
    if (!f) return arrow::Status::IOError("cannot write model.index.json");
    f << "{\n";
    f << "  \"layout\": 2,\n";
    f << "  \"model\": \"Xing4.0-29B-A4B\",\n";
    // The index reports what was actually written (observed), not what the
    // config asked for: a synthetic fixture is legitimately smaller, and the
    // loader's frozen gate compares against the passport, not against this.
    long long observed_experts = 0;
    for (const auto &[l, per_expert] : experts)
        observed_experts = std::max<long long>(
            observed_experts, static_cast<long long>(per_expert.size()));
    f << "  \"num_layers\": " << observed_layers << ",\n";
    f << "  \"experts_per_layer\": " << observed_experts << ",\n";
    f << "  \"hc_mult\": " << cfg.hc_mult << ",\n";
    f << "  \"hc_sinkhorn_iters\": " << cfg.hc_sinkhorn_iters << ",\n";
    f << "  \"context_length\": " << cfg.context_length << ",\n";
    f << "  \"nvfp4\": {\n";
    {
        std::vector<int> ordered;
        ordered.reserve(layer_scales.size());
        for (const auto &[l, v] : layer_scales) ordered.push_back(l);
        std::sort(ordered.begin(), ordered.end());
        for (size_t i = 0; i < ordered.size(); ++i)
        {
            f << (i ? ",\n" : "");
            f << "    \"" << ordered[i] << "\": {";
            const auto &bases = layer_scales[ordered[i]];
            for (size_t j = 0; j < bases.size(); ++j)
                f << (j ? ", " : "")
                  << "\"" << bases[j].first << "\": "
                  << absl::StrFormat("%.9g", bases[j].second);
            f << "}";
        }
    }
    f << "\n  },\n";
    f << "  \"moe_layers\": [";
    for (size_t i = 0; i < moe_layers.size(); ++i)
        f << (i ? "," : "") << moe_layers[i];
    f << "],\n";
    f << "  \"dense_layers\": [";
    for (size_t i = 0; i < dense_layers.size(); ++i)
        f << (i ? "," : "") << dense_layers[i];
    f << "],\n";
    f << "  \"mtp_layers\": [";
    {
        std::vector<int> mtp;
        for (int l : frozen::kMtpLayers) mtp.push_back(l);
        for (size_t i = 0; i < mtp.size(); ++i)
            f << (i ? "," : "") << mtp[i];
    }
    f << "],\n";
    f << "  \"mtp_tensors\": " << mtp_tensor_count << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  \"skipped_tensors\": [";
    for (size_t i = 0; i < skipped.size(); ++i)
        f << (i ? ", " : "") << "\"" << skipped[i] << "\"";
    f << "]\n";
    f << "}\n";
    f.close();

    return arrow::Status::OK();
}

} // namespace xing
} // namespace lsxcommon