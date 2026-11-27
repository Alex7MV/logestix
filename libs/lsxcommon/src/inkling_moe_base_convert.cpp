// Inkling (base) converter (Model Ingestion Protocol stages 1-2).
//
// Source: the pure-BF16 `thinkingmachines/Inkling` release (architecture
// `inkling_mm_model`). Everything here is standard NVFP4 quantization through
// the neutral helpers; the things that are easy to get wrong are pinned:
//
//   1. THREE fused gate/up slabs, ALL ROW-INTERLEAVED (gate=even rows, up=odd
//      rows): routed `mlp.experts.w13_weight`, `shared_w13_weight` and the
//      dense `w13_dn.weight`. transformers' conversion_mapping.py
//      "inkling_mm_model" applies Interleave(dim=1) to the routed/shared slabs
//      and Interleave(dim=0)+Chunk to the dense one; each de-interleaves
//      [g0,u0,g1,u1,...] into [gate;up] before the forward chunks the output.
//      Splitting any of them as HALVES keeps every byte and every plane size
//      while pairing the wrong rows — invisible to every check except the
//      model's own output. The convert test pins all three. (The routed slab
//      was originally mis-read as HALVES; the release op on the real
//      checkpoint proved it interleaved, 2026-11-18.)
//   2. The router weight has E+S rows (256 routed + 2 shared); the selection
//      bias covers only the E routed rows. Both stay high precision: NVFP4
//      rounding on the argmax surface silently swaps the expert set.
//   3. The vision (hmlp) / audio / MTP towers are ballast: counted in the
//      index, never applied. An unrecognized tensor is a hard error.
//   4. embed/unembed stay raw BF16 (uploaded as FP16 tables): they carry the
//      residual stream at full width and 4-bit rounding there displaces every
//      token vector.
#include "lsxcommon/inkling_moe_base_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/synthetic_geometry.h"
#include "inkling_moe_base_constants.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::inkling::base {
namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::inkling::base::frozen;

constexpr char kLlmScope[] = "model.llm.";
constexpr char kVisualScope[] = "model.visual.";
constexpr char kAudioScope[] = "model.audio.";
constexpr char kMtpScope[] = "model.mtp.";

bool ParseLayerIndex(std::string_view s, int &out)
{
    if (s.empty()) return false;
    int v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    out = v;
    return true;
}

std::string ReadWholeFile(const fs::path &p)
{
    std::ifstream f(p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

namespace dom = simdjson::dom;

template <typename T>
void GetInt(dom::object &o, const char *key, T &out)
{
    int64_t v = 0;
    if (o[key].get_int64().get(v) == simdjson::SUCCESS)
        out = static_cast<T>(v);
}

void GetDouble(dom::object &o, const char *key, double &out)
{
    dom::element e;
    if (o[key].get(e) != simdjson::SUCCESS) return;
    if (auto d = e.get_double(); d.error() == simdjson::SUCCESS) {
        out = d.value();
        return;
    }
    int64_t i = 0;
    if (e.get_int64().get(i) == simdjson::SUCCESS)
        out = static_cast<double>(i);
}

void GetIntArray(dom::object &o, const char *key, std::vector<int> &out)
{
    dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return;
    out.clear();
    for (auto v : arr) {
        int64_t x = 0;
        if (v.get_int64().get(x) == simdjson::SUCCESS)
            out.push_back(static_cast<int>(x));
    }
}

bool IsGlobalKind(TensorKind k)
{
    switch (k) {
    case TensorKind::Embed:
    case TensorKind::Unembed:
    case TensorKind::EmbedNorm:
    case TensorKind::FinalNorm:
        return true;
    default:
        return false;
    }
}

bool IsExpertKind(TensorKind k)
{
    return k == TensorKind::ExpertW13 || k == TensorKind::ExpertW2;
}

// Frozen passport cross-check; relaxed under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
bool CheckFrozenGeometry(const ConvertTopology &top, std::string &why)
{
    auto bad = [&why](const char *field, long long got, long long want) {
        why = absl::StrFormat("%s: got %lld, want %lld", field, got, want);
        return false;
    };
    if (top.hidden_size != frozen::kHidden)
        return bad("hidden_size", top.hidden_size, frozen::kHidden);
    if (top.num_layers != frozen::kLayers)
        return bad("num_hidden_layers", top.num_layers, frozen::kLayers);
    if (top.vocab_size != frozen::kVocab)
        return bad("vocab_size", top.vocab_size, frozen::kVocab);
    if (top.unpadded_vocab_size != frozen::kBaseVocab)
        return bad("unpadded_vocab_size", top.unpadded_vocab_size,
                   frozen::kBaseVocab);
    if (top.num_attention_heads != frozen::kHeads)
        return bad("num_attention_heads", top.num_attention_heads,
                   frozen::kHeads);
    if (top.num_key_value_heads != frozen::kKvHeadsFull)
        return bad("num_key_value_heads", top.num_key_value_heads,
                   frozen::kKvHeadsFull);
    if (top.swa_num_key_value_heads != frozen::kKvHeadsSliding)
        return bad("swa_num_key_value_heads", top.swa_num_key_value_heads,
                   frozen::kKvHeadsSliding);
    if (top.head_dim != frozen::kHeadDim)
        return bad("head_dim", top.head_dim, frozen::kHeadDim);
    if (top.d_rel != frozen::kDRel)
        return bad("d_rel", top.d_rel, frozen::kDRel);
    if (top.rel_extent != frozen::kRelExtent)
        return bad("rel_extent", top.rel_extent, frozen::kRelExtent);
    if (top.sliding_window_size != frozen::kSlidingWindow)
        return bad("sliding_window_size", top.sliding_window_size,
                   frozen::kSlidingWindow);
    if (top.conv_kernel != frozen::kConvKernel)
        return bad("sconv_kernel_size", top.conv_kernel, frozen::kConvKernel);
    if (top.dense_mlp_layers != frozen::kDenseMlpLayers)
        return bad("dense_mlp_idx", top.dense_mlp_layers,
                   frozen::kDenseMlpLayers);
    if (top.dense_intermediate_size != frozen::kDenseIntermediate)
        return bad("dense_intermediate_size", top.dense_intermediate_size,
                   frozen::kDenseIntermediate);
    if (top.num_experts != frozen::kExperts)
        return bad("n_routed_experts", top.num_experts, frozen::kExperts);
    if (top.num_experts_per_tok != frozen::kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok,
                   frozen::kTopK);
    if (top.moe_intermediate_size != frozen::kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   frozen::kMoeIntermediate);
    if (top.num_shared_experts != frozen::kSharedExperts)
        return bad("n_shared_experts", top.num_shared_experts,
                   frozen::kSharedExperts);
    if (top.full_layers.size() != static_cast<size_t>(frozen::kFullLayerCount)) {
        why = absl::StrFormat("full-attention layers: got %zu, want %d",
                              top.full_layers.size(), frozen::kFullLayerCount);
        return false;
    }
    for (size_t i = 0; i < top.full_layers.size(); ++i)
        if (top.full_layers[i] != frozen::kFullLayers[i]) {
            why = absl::StrFormat("full-attention layer %zu: got %d, want %d", i,
                                  top.full_layers[i], frozen::kFullLayers[i]);
            return false;
        }
    return true;
}

struct SourceTensor {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

// Copies `rows` rows of `row_bytes` from `src`, taking every second row
// starting at `parity` (0 = even rows, 1 = odd rows). This is the
// de-interleave of the shared/dense fused slabs.
void TakeParityRows(const uint8_t *src, size_t rows, size_t row_bytes,
                    size_t parity, std::vector<uint8_t> &dst)
{
    dst.resize(rows * row_bytes);
    for (size_t r = 0; r < rows; ++r)
        std::memcpy(dst.data() + r * row_bytes,
                    src + (2 * r + parity) * row_bytes, row_bytes);
}

}  // namespace

ConvertTopology ParseConfig(const std::string &input_dir)
{
    ConvertTopology top;
    const fs::path cfg_path = fs::path(input_dir) / "config.json";
    if (!fs::exists(cfg_path)) {
        top.error_msg = "config.json not found in " + input_dir;
        return top;
    }
    const std::string json = ReadWholeFile(cfg_path);
    dom::parser parser;
    simdjson::padded_string pad(json);
    dom::element doc;
    if (parser.parse(pad).get(doc) != simdjson::SUCCESS) {
        top.error_msg = "config.json parse error";
        return top;
    }
    dom::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        top.error_msg = "config.json root is not an object";
        return top;
    }
    std::string_view sview;
    if (root["model_type"].get_string().get(sview) == simdjson::SUCCESS)
        top.model_type = std::string(sview);

    dom::object text;
    if (root["text_config"].get_object().get(text) != simdjson::SUCCESS) {
        top.error_msg = "config.json has no text_config";
        return top;
    }

    GetInt(text, "hidden_size", top.hidden_size);
    GetInt(text, "num_hidden_layers", top.num_layers);
    GetInt(text, "vocab_size", top.vocab_size);
    GetInt(text, "unpadded_vocab_size", top.unpadded_vocab_size);
    GetInt(text, "num_attention_heads", top.num_attention_heads);
    GetInt(text, "num_key_value_heads", top.num_key_value_heads);
    GetInt(text, "swa_num_key_value_heads", top.swa_num_key_value_heads);
    GetInt(text, "head_dim", top.head_dim);
    GetInt(text, "d_rel", top.d_rel);
    GetInt(text, "rel_extent", top.rel_extent);
    GetInt(text, "sliding_window_size", top.sliding_window_size);
    GetInt(text, "sconv_kernel_size", top.conv_kernel);
    GetInt(text, "dense_mlp_idx", top.dense_mlp_layers);
    GetInt(text, "dense_intermediate_size", top.dense_intermediate_size);
    GetInt(text, "n_routed_experts", top.num_experts);
    GetInt(text, "num_experts_per_tok", top.num_experts_per_tok);
    // The checkpoint calls the MoE width `intermediate_size` (HF maps it onto
    // `moe_intermediate_size`); accept either spelling.
    GetInt(text, "moe_intermediate_size", top.moe_intermediate_size);
    if (top.moe_intermediate_size == 0)
        GetInt(text, "intermediate_size", top.moe_intermediate_size);
    GetInt(text, "n_shared_experts", top.num_shared_experts);
    GetDouble(text, "route_scale", top.route_scale);
    GetDouble(text, "rms_norm_eps", top.rms_norm_eps);
    GetInt(text, "log_scaling_n_floor", top.log_scaling_n_floor);
    GetDouble(text, "log_scaling_alpha", top.log_scaling_alpha);
    GetDouble(text, "logits_mup_width_multiplier", top.logits_mup_div);

    std::vector<int> local_layers;
    GetIntArray(text, "local_layer_ids", local_layers);

    int64_t eos = 0;
    if (root["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    else if (text["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    // No BOS and no pad in this checkpoint.
    top.bos_id = -1;
    top.pad_id = -1;
    top.stop_id_alt = frozen::kStopIdAlt;

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.moe_intermediate_size <= 0 || top.head_dim <= 0 ||
        top.swa_num_key_value_heads <= 0) {
        top.error_msg = "config.json is missing required geometry";
        return top;
    }

    // ── layer split ────────────────────────────────────────────────────────
    top.full_layers.clear();
    top.sliding_layers.clear();
    if (!local_layers.empty()) {
        std::vector<bool> local(static_cast<size_t>(top.num_layers), false);
        for (int l : local_layers)
            if (l >= 0 && l < top.num_layers) local[static_cast<size_t>(l)] = true;
        for (int l = 0; l < top.num_layers; ++l) {
            if (local[static_cast<size_t>(l)])
                top.sliding_layers.push_back(l);
            else
                top.full_layers.push_back(l);
        }
    } else {
        // HF default: full attention where (layer+1) % 6 == 0.
        for (int l = 0; l < top.num_layers; ++l) {
            if ((l + 1) % 6 == 0)
                top.full_layers.push_back(l);
            else
                top.sliding_layers.push_back(l);
        }
    }
    return top;
}

ParsedTensor ClassifyTensor(const std::string &name)
{
    ParsedTensor p;
    const std::string_view n(name);
    if (n.substr(0, sizeof(kVisualScope) - 1) == kVisualScope) {
        p.kind = TensorKind::Vision;
        return p;
    }
    if (n.substr(0, sizeof(kAudioScope) - 1) == kAudioScope) {
        p.kind = TensorKind::Audio;
        return p;
    }
    if (n.substr(0, sizeof(kMtpScope) - 1) == kMtpScope) {
        p.kind = TensorKind::Mtp;
        return p;
    }
    if (n.substr(0, sizeof(kLlmScope) - 1) != kLlmScope) return p;
    const std::string_view rest = n.substr(sizeof(kLlmScope) - 1);

    if (rest == "embed.weight") { p.kind = TensorKind::Embed; return p; }
    if (rest == "unembed.weight") { p.kind = TensorKind::Unembed; return p; }
    if (rest == "embed_norm.weight") { p.kind = TensorKind::EmbedNorm; return p; }
    if (rest == "norm.weight") { p.kind = TensorKind::FinalNorm; return p; }

    const std::string_view lp = "layers.";
    if (rest.substr(0, lp.size()) != lp) return p;
    std::string_view tail = rest.substr(lp.size());
    const size_t dot = tail.find('.');
    if (dot == std::string_view::npos) return p;
    int layer = -1;
    if (!ParseLayerIndex(tail.substr(0, dot), layer)) return p;
    tail = tail.substr(dot + 1);
    p.layer = layer;

    struct Entry {
        std::string_view name;
        TensorKind kind;
    };
    static constexpr Entry kTable[] = {
        {"attn.wq_du.weight", TensorKind::Wq},
        {"attn.wk_dv.weight", TensorKind::Wk},
        {"attn.wv_dv.weight", TensorKind::Wv},
        {"attn.wo_ud.weight", TensorKind::Wo},
        {"attn.wr_du.weight", TensorKind::Wr},
        {"attn.q_norm.weight", TensorKind::QNorm},
        {"attn.k_norm.weight", TensorKind::KNorm},
        {"attn.rel_logits_proj.proj", TensorKind::RelProj},
        {"attn.k_sconv.weight", TensorKind::KSconv},
        {"attn.v_sconv.weight", TensorKind::VSconv},
        {"attn_sconv.weight", TensorKind::AttnSconv},
        {"mlp_sconv.weight", TensorKind::MlpSconv},
        {"attn_norm.weight", TensorKind::AttnNorm},
        {"mlp_norm.weight", TensorKind::MlpNorm},
        {"mlp.w13_dn.weight", TensorKind::DenseW13},
        {"mlp.w2_md.weight", TensorKind::DenseW2},
        {"mlp.global_scale", TensorKind::DenseScale},
        {"mlp.experts.w13_weight", TensorKind::ExpertW13},
        {"mlp.experts.w2_weight", TensorKind::ExpertW2},
        {"mlp.shared_experts.shared_w13_weight", TensorKind::SharedW13},
        {"mlp.shared_experts.shared_w2_weight", TensorKind::SharedW2},
        {"mlp.gate.weight", TensorKind::RouterGate},
        {"mlp.gate.bias", TensorKind::RouterBias},
        {"mlp.gate.global_scale", TensorKind::RouterScale},
    };
    for (const auto &e : kTable)
        if (tail == e.name) {
            p.kind = e.kind;
            return p;
        }
    return p;
}

bool TensorIsNvfp4(TensorKind kind)
{
    switch (kind) {
    case TensorKind::Wq:
    case TensorKind::Wk:
    case TensorKind::Wv:
    case TensorKind::Wo:
    case TensorKind::Wr:
    case TensorKind::DenseW13:
    case TensorKind::DenseW2:
    case TensorKind::ExpertW13:
    case TensorKind::ExpertW2:
    case TensorKind::SharedW13:
    case TensorKind::SharedW2:
        return true;
    default:
        return false;
    }
}

arrow::Status ConvertInklingBaseModel(const std::string &input_dir,
                                      const std::string &output_root)
{
    ConvertTopology top = ParseConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Inkling config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > n_routed_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, why))
            return arrow::Status::Invalid("Inkling frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // ── inventory ──────────────────────────────────────────────────────────
    absl::btree_map<std::string, SourceTensor> weights;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = ClassifyTensor(t.name);
            switch (p.kind) {
            case TensorKind::Vision:
                ++top.skipped_visual_tensors;
                continue;
            case TensorKind::Audio:
                ++top.skipped_audio_tensors;
                continue;
            case TensorKind::Mtp:
                ++top.skipped_mtp_tensors;
                continue;
            case TensorKind::Unknown:
                return arrow::Status::Invalid(
                    "unrecognized tensor in the inkling checkpoint: ", t.name);
            default:
                break;
            }
            if (p.layer >= top.num_layers && p.layer >= 0)
                return arrow::Status::Invalid("layer index out of range: ",
                                              t.name);
            SourceTensor st;
            st.shard = &shard;
            st.info = &t;
            weights[t.name] = std::move(st);
        }
    }

    const int L = top.num_layers;
    const int H = top.hidden_size;
    const int E = top.num_experts;
    const int S = top.num_shared_experts;
    const int I = top.moe_intermediate_size;
    const int DI = top.dense_intermediate_size;

    // The source is pure BF16: a quantized expert slab means the wrong
    // checkpoint (or a future mixed-precision release this converter has not
    // been audited for).
    for (int l = top.dense_mlp_layers; l < L; ++l) {
        const std::string key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w13_weight", l);
        auto it = weights.find(key);
        if (it == weights.end())
            return arrow::Status::Invalid("missing routed experts for layer ", l);
        if (it->second.info->dtype != "BF16")
            return arrow::Status::Invalid(absl::StrFormat(
                "layer %d routed experts have dtype %s; this converter expects "
                "the unquantized BF16 release",
                l, it->second.info->dtype));
    }

    const fs::path base = fs::path(output_root);
    const fs::path pinned_dir = base / "pinned";
    const fs::path experts_dir = base / "experts";
    fs::create_directories(pinned_dir);
    fs::create_directories(experts_dir);

    std::vector<TensorRecord> global_raw;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    int pinned_count = 0;

    auto push_raw = [&](const std::string &name, const SourceTensor &st,
                        int layer) {
        auto data = ReadTensorData(input_dir, *st.shard, *st.info);
        TensorRecord r;
        r.name = name;
        r.dtype = st.info->dtype;
        r.shape = st.info->shape;
        r.data = std::move(data);
        if (layer < 0)
            global_raw.push_back(std::move(r));
        else
            raw_by_layer[layer].push_back(std::move(r));
        ++pinned_count;
        return true;
    };

    auto quantize = [&](const std::string &out_name, const uint8_t *data,
                        int64_t numel, const std::string &dtype,
                        std::vector<int64_t> shape, int layer) -> arrow::Status {
        DenseNvfp4Record r;
        r.name = out_name;
        r.shape = std::move(shape);
        r.dtype = "NVFP4";
        if (!QuantizeSlice(data, numel, dtype, r))
            return arrow::Status::IOError("NVFP4 quantization failed: ",
                                          out_name);
        nvfp4_by_layer[layer].push_back(std::move(r));
        ++pinned_count;
        return arrow::Status::OK();
    };

    // ── dense / raw tensors ────────────────────────────────────────────────
    for (const auto &[name, st] : weights) {
        const auto p = ClassifyTensor(name);
        if (IsExpertKind(p.kind)) continue;  // handled per (layer, expert) below
        if (p.kind == TensorKind::SharedW13 || p.kind == TensorKind::SharedW2)
            continue;  // handled per shared expert below
        auto data = ReadTensorData(input_dir, *st.shard, *st.info);
        if (data.empty() ||
            static_cast<int64_t>(data.size()) != st.info->byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        if (!TensorIsNvfp4(p.kind)) {
            TensorRecord r;
            r.name = name;
            r.dtype = st.info->dtype;
            r.shape = st.info->shape;
            r.data = std::move(data);
            if (IsGlobalKind(p.kind))
                global_raw.push_back(std::move(r));
            else
                raw_by_layer[p.layer].push_back(std::move(r));
            ++pinned_count;
            continue;
        }

        if (p.kind == TensorKind::DenseW13) {
            // ROW-INTERLEAVED [2*DI, H]: even rows gate, odd rows up
            // (Interleave(dim=0)+Chunk in the release's mapping).
            const int64_t rows = st.info->shape[0];
            if (rows != 2 * DI || st.info->shape[1] != H)
                return arrow::Status::Invalid("dense w13 shape: ", name);
            const size_t row_bytes = static_cast<size_t>(H) * 2;  // BF16
            std::vector<uint8_t> gate, up;
            TakeParityRows(data.data(), static_cast<size_t>(DI), row_bytes, 0,
                           gate);
            TakeParityRows(data.data(), static_cast<size_t>(DI), row_bytes, 1,
                           up);
            const std::string prefix =
                absl::StrFormat("model.llm.layers.%d.mlp.", p.layer);
            ARROW_RETURN_NOT_OK(quantize(prefix + "gate_proj.weight",
                                         gate.data(), int64_t(DI) * H, "BF16",
                                         {DI, H}, p.layer));
            ARROW_RETURN_NOT_OK(quantize(prefix + "up_proj.weight", up.data(),
                                         int64_t(DI) * H, "BF16", {DI, H},
                                         p.layer));
            continue;
        }
        if (p.kind == TensorKind::DenseW2) {
            const std::string out_name =
                absl::StrFormat("model.llm.layers.%d.mlp.down_proj.weight",
                                p.layer);
            ARROW_RETURN_NOT_OK(quantize(out_name, data.data(),
                                         st.info->numel(), st.info->dtype,
                                         st.info->shape, p.layer));
            continue;
        }
        ARROW_RETURN_NOT_OK(quantize(name, data.data(), st.info->numel(),
                                     st.info->dtype, st.info->shape, p.layer));
    }

    // ── shared experts: ROW-INTERLEAVED w13, per-expert records ────────────
    for (int l = top.dense_mlp_layers; l < L; ++l) {
        const std::string w13_key = absl::StrFormat(
            "model.llm.layers.%d.mlp.shared_experts.shared_w13_weight", l);
        const std::string w2_key = absl::StrFormat(
            "model.llm.layers.%d.mlp.shared_experts.shared_w2_weight", l);
        if (weights.find(w13_key) == weights.end() ||
            weights.find(w2_key) == weights.end())
            return arrow::Status::Invalid("missing shared experts for layer ", l);
        const auto &w13 = weights.at(w13_key);
        const auto &w2 = weights.at(w2_key);
        auto w13_data = ReadTensorData(input_dir, *w13.shard, *w13.info);
        auto w2_data = ReadTensorData(input_dir, *w2.shard, *w2.info);
        if (w13_data.empty() || w2_data.empty())
            return arrow::Status::IOError("failed to read shared experts: ", l);
        if (w13.info->shape != std::vector<int64_t>{S, 2 * I, H} ||
            w2.info->shape != std::vector<int64_t>{S, H, I})
            return arrow::Status::Invalid("shared expert shape on layer ", l);
        const size_t row_bytes = static_cast<size_t>(H) * 2;
        const size_t w13_expert = static_cast<size_t>(2 * I) * row_bytes;
        const size_t w2_expert = static_cast<size_t>(H) * I * 2;
        for (int s = 0; s < S; ++s) {
            const uint8_t *w13p = w13_data.data() + w13_expert * s;
            std::vector<uint8_t> gate, up;
            TakeParityRows(w13p, static_cast<size_t>(I), row_bytes, 0, gate);
            TakeParityRows(w13p, static_cast<size_t>(I), row_bytes, 1, up);
            const std::string prefix = absl::StrFormat(
                "model.llm.layers.%d.mlp.shared_experts.%d.", l, s);
            ARROW_RETURN_NOT_OK(quantize(prefix + "gate_proj.weight",
                                         gate.data(), int64_t(I) * H, "BF16",
                                         {I, H}, l));
            ARROW_RETURN_NOT_OK(quantize(prefix + "up_proj.weight", up.data(),
                                         int64_t(I) * H, "BF16", {I, H}, l));
            ARROW_RETURN_NOT_OK(quantize(
                prefix + "down_proj.weight", w2_data.data() + w2_expert * s,
                int64_t(H) * I, "BF16", {H, I}, l));
        }
    }

    // ── routed experts: ROW-INTERLEAVED w13, one Arrow file per (layer,
    //    expert). The release conversion_mapping applies Interleave(dim=1)
    //    to de-interleave [g0,u0,g1,u1,...] into [gate;up] before the forward
    //    chunks the output — the SAME layout as shared_w13 / dense w13_dn.
    //    Verified against the release op on the real checkpoint (2026-11-18):
    //    the HALVES split mixed gate/up rows and made every MoE layer diverge.
    int experts_written = 0;
    for (int l = top.dense_mlp_layers; l < L; ++l) {
        const std::string w13_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w13_weight", l);
        const std::string w2_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w2_weight", l);
        const auto &w13 = weights.at(w13_key);
        const auto &w2 = weights.at(w2_key);
        auto w13_data = ReadTensorData(input_dir, *w13.shard, *w13.info);
        auto w2_data = ReadTensorData(input_dir, *w2.shard, *w2.info);
        if (w13_data.empty() || w2_data.empty())
            return arrow::Status::IOError("failed to read experts for layer ", l);
        if (w13.info->shape != std::vector<int64_t>{E, 2 * I, H} ||
            w2.info->shape != std::vector<int64_t>{E, H, I})
            return arrow::Status::Invalid(
                "routed expert tensor shape does not match the geometry on "
                "layer ",
                l);
        const size_t elem = 2;  // BF16
        const size_t w1_plane = static_cast<size_t>(I) * H * elem;
        const size_t w13_expert = 2 * w1_plane;
        const size_t w2_expert = static_cast<size_t>(H) * I * elem;
        for (int e = 0; e < E; ++e) {
            const uint8_t *w13p = w13_data.data() + w13_expert * e;
            ExpertData ed;
            // ROW-INTERLEAVED: source row 2r is gate row r, row 2r+1 is up
            // row r (release Interleave(dim=1) de-interleaves to [gate;up]).
            std::vector<uint8_t> gate, up;
            TakeParityRows(w13p, static_cast<size_t>(I), w1_plane / I, 0, gate);
            TakeParityRows(w13p, static_cast<size_t>(I), w1_plane / I, 1, up);
            ed.w1_raw = std::move(gate);
            ed.w3_raw = std::move(up);
            ed.w2_raw.assign(w2_data.data() + w2_expert * e,
                             w2_data.data() + w2_expert * (e + 1));
            ed.w1_dtype = ed.w2_dtype = ed.w3_dtype = "BF16";
            ed.w1_shape = {I, H};
            ed.w3_shape = {I, H};
            ed.w2_shape = {H, I};
            auto packed = PackExpertNVFP4(ed);
            if (!packed.ok()) return packed.status();
            ARROW_RETURN_NOT_OK(
                WriteExpertArrow(ExpertArrowPath(base, l, e), *packed));
            ++experts_written;
        }
        lsxcommon::log::info(absl::StrFormat(
            "Inkling layer %d: %d routed experts written", l, E));
    }

    // ── write pinned batches ───────────────────────────────────────────────
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                       pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));

    top.total_pinned_tensors = pinned_count;

    // ── tokenizer (mandatory for a real checkpoint) ────────────────────────
    const fs::path tk_path = fs::path(input_dir) / "tiktoken" / "tokenizer.model";
    bool tokenizer_ok = false;
    if (fs::exists(tk_path)) {
        TikTokenSpecialPolicy policy;
        // The reference tokenizer never prepends BOS; <|endoftext|> is the
        // alternate stop id.
        policy.bos_id = -1;
        policy.eos_id = top.eos_id;
        policy.pad_id = top.pad_id;
        ARROW_RETURN_NOT_OK(BuildTokenizerArrowFilesTikToken(
            tk_path.string(),
            (fs::path(input_dir) / "tokenizer_config.json").string(), pinned_dir,
            policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tiktoken/tokenizer.model not found in ",
                                      input_dir);
    }

    // ── VRAM profile ───────────────────────────────────────────────────────
    std::vector<VramTensorInfo> vram_tensors;
    VramTensorPolicy policy;
    policy.is_nvfp4 = [](const std::string &name) {
        return TensorIsNvfp4(ClassifyTensor(name).kind);
    };
    policy.mem_class = [](const std::string &name) {
        switch (ClassifyTensor(name).kind) {
        case TensorKind::ExpertW13:
        case TensorKind::ExpertW2:
            return VramTensorClass::RoutedExpert;
        default:
            return VramTensorClass::Resident;
        }
    };
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            auto p = ClassifyTensor(t.name);
            if (p.kind == TensorKind::Vision || p.kind == TensorKind::Audio ||
                p.kind == TensorKind::Mtp)
                continue;
            VramTensorInfo ti;
            ti.name = t.name;
            ti.numel = t.numel();
            ti.raw_bytes = t.byte_size();
            vram_tensors.push_back(std::move(ti));
        }
    VramGeometry geom;
    geom.hidden_dim = top.hidden_size;
    // The sliding layers dominate the KV footprint (55 of 66, 16 kv heads).
    geom.kv_dim = top.swa_num_key_value_heads * top.head_dim;
    geom.num_attn_layers = top.num_layers;
    geom.num_moe_layers = top.num_layers - top.dense_mlp_layers;
    geom.experts_per_layer = top.num_experts;
    geom.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    geom.nvfp4_scale_block = frozen::kQuantBlock;
    const VramProfile vram = EstimateVram(vram_tensors, policy, geom);

    // ── index ──────────────────────────────────────────────────────────────
    {
        std::ostringstream j;
        j.precision(17);
        auto int_array = [&j](const char *key, const std::vector<int> &v,
                              bool trailing) {
            j << "  \"" << key << "\": [";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) j << ", ";
                j << v[i];
            }
            j << "]" << (trailing ? ",\n" : "\n");
        };
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n";
        // The CARD name: gpu::detect_model_type normalizes it (spaces -> '-',
        // lower case) into the LsxModelFactory key, so it must normalize to
        // exactly InklingMoeBaseModel::kType ("inkling").
        j << "  \"model\": \"Inkling\",\n";
        j << "  \"source_model_type\": \"" << top.model_type << "\",\n";
        j << "  \"num_layers\": " << top.num_layers << ",\n";
        j << "  \"hidden_size\": " << top.hidden_size << ",\n";
        j << "  \"vocab_size\": " << top.vocab_size << ",\n";
        j << "  \"unpadded_vocab_size\": " << top.unpadded_vocab_size << ",\n";
        j << "  \"num_attention_heads\": " << top.num_attention_heads << ",\n";
        j << "  \"num_key_value_heads\": " << top.num_key_value_heads << ",\n";
        j << "  \"swa_num_key_value_heads\": " << top.swa_num_key_value_heads
          << ",\n";
        j << "  \"head_dim\": " << top.head_dim << ",\n";
        // q/k are per-head RMS-normed: the reference scales by 1/head_dim.
        j << "  \"attn_scale_denom\": " << static_cast<double>(top.head_dim)
          << ",\n";
        j << "  \"d_rel\": " << top.d_rel << ",\n";
        j << "  \"rel_extent\": " << top.rel_extent << ",\n";
        j << "  \"sliding_window_size\": " << top.sliding_window_size << ",\n";
        j << "  \"conv_kernel\": " << top.conv_kernel << ",\n";
        j << "  \"dense_mlp_layers\": " << top.dense_mlp_layers << ",\n";
        j << "  \"dense_intermediate_size\": " << top.dense_intermediate_size
          << ",\n";
        j << "  \"num_experts\": " << top.num_experts << ",\n";
        j << "  \"num_experts_per_tok\": " << top.num_experts_per_tok << ",\n";
        j << "  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n";
        j << "  \"num_shared_experts\": " << top.num_shared_experts << ",\n";
        j << "  \"routed_scaling_factor\": " << top.route_scale << ",\n";
        j << "  \"rms_norm_eps\": " << top.rms_norm_eps << ",\n";
        j << "  \"log_scaling_n_floor\": " << top.log_scaling_n_floor << ",\n";
        j << "  \"log_scaling_alpha\": " << top.log_scaling_alpha << ",\n";
        j << "  \"logits_mup_div\": " << top.logits_mup_div << ",\n";
        j << "  \"bos_id\": " << top.bos_id << ",\n";
        j << "  \"eos_id\": " << top.eos_id << ",\n";
        j << "  \"stop_id_alt\": " << top.stop_id_alt << ",\n";
        j << "  \"pad_id\": " << top.pad_id << ",\n";
        j << "  \"tokenizer_kind\": " << frozen::kTokenizerKind << ",\n";
        j << "  \"tokenizer\": \"" << (tokenizer_ok ? "tiktoken" : "fixture")
          << "\",\n";
        j << "  \"skipped_visual_tensors\": " << top.skipped_visual_tensors
          << ",\n";
        j << "  \"skipped_audio_tensors\": " << top.skipped_audio_tensors
          << ",\n";
        j << "  \"skipped_mtp_tensors\": " << top.skipped_mtp_tensors << ",\n";
        j << "  \"total_pinned_tensors\": " << top.total_pinned_tensors << ",\n";
        j << "  \"total_experts\": " << experts_written << ",\n";
        int_array("full_layers", top.full_layers, true);
        int_array("sliding_layers", top.sliding_layers, true);
        j << FormatVramJson(vram) << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    lsxcommon::log::info(absl::StrFormat(
        "Inkling conversion: %d layers (%zu full / %zu sliding), %d dense-MLP "
        "prefix layers, %d experts x %d MoE layers, %d pinned tensors, %d "
        "expert files, stripped %d vision / %d audio / %d mtp tensors, "
        "tiktoken %s",
        top.num_layers, top.full_layers.size(), top.sliding_layers.size(),
        top.dense_mlp_layers, top.num_experts,
        top.num_layers - top.dense_mlp_layers, pinned_count, experts_written,
        top.skipped_visual_tensors, top.skipped_audio_tensors,
        top.skipped_mtp_tensors, tokenizer_ok ? "built" : "absent"));
    return arrow::Status::OK();
}

}  // namespace lsxcommon::inkling::base