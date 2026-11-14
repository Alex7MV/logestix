// Inkling-2-Small converter (Model Ingestion Protocol stages 1-2).
//
// Source: modelopt release `thinkingmachines/Inkling-2-Small`. Three things
// about it are easy to get wrong and are pinned here:
//
//   1. MIXED EXPERT CODEC. `hf_quant_config.json` puts the routed experts of
//      layers 0..39 in NVFP4 (E2M1 nibbles, per-16 E4M3FN scales, per-expert
//      `amax` with `outer_scale_denominator` 2688) and layers 40..47 in MXFP8
//      (E4M3 weights, ue8m0 scale per 32). A converter that assumes one codec
//      decodes a sixth of the stack as garbage. The split is taken from the
//      OBSERVED dtype of each layer's `w13_weight` and cross-checked against
//      `hf_quant_config.json`.
//   2. The NVFP4 half is passed through verbatim. The block scales are full
//      E4M3FN and the runtime scale decode is mantissa-aware
//      (`nvfp4_codec.h::E4M3ScaleToFloat`), and the nibble order (element 2i in
//      the low nibble) is the repo's own. The per-expert global scale follows
//      from the checkpoint's own invariant, verified numerically on the real
//      shards: 6 * max(block_scale_e) * (amax_e / 2688) == amax_e.
//   3. `attn_res.attn_queries.L` is indexed by `layer-1` (layer 0 has no
//      attention mixture) while `mlp_queries.L` is indexed by layer. The
//      artifact renames both to the layer they belong to.
#include "lsxcommon/inkling_moe_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/synthetic_geometry.h"
#include "inkling_moe_constants.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::inkling {
namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::inkling::frozen;

constexpr char kLlmScope[] = "model.llm.";
constexpr char kVisualScope[] = "model.visual.";

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

// ── simdjson::dom helpers (random access; ondemand is forward-only and the
// config files are read out of order) ─────────────────────────────────────
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

}  // namespace

InklingConvertTopology ParseInklingConfig(const std::string &input_dir)
{
    InklingConvertTopology top;
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
    GetInt(text, "head_dim", top.head_dim);
    GetInt(text, "d_rel", top.d_rel);
    GetInt(text, "rel_extent", top.rel_extent);
    GetInt(text, "sliding_window_size", top.sliding_window_size);
    GetInt(text, "sconv_kernel_size", top.conv_kernel);
    GetInt(text, "n_layers_per_attn_res_block", top.attn_res_block);
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
    GetDouble(text, "qk_l2_clip_threshold", top.qk_l2_clip);
    GetDouble(text, "rel_l2_clip_threshold", top.rel_l2_clip);
    GetInt(text, "log_scaling_n_floor", top.log_scaling_n_floor);
    GetDouble(text, "log_scaling_alpha", top.log_scaling_alpha);
    GetDouble(text, "mlp_gate_softcap", top.mlp_gate_softcap);
    GetDouble(text, "mlp_up_softcap", top.mlp_up_softcap);
    GetDouble(text, "final_logit_softcapping", top.final_logit_softcap);

    std::vector<int> local_layers;
    GetIntArray(text, "local_layer_ids", local_layers);

    int64_t eos = 0;
    if (root["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    else if (text["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    // No BOS and no pad in this checkpoint (bos_token_id: null).
    top.bos_id = -1;
    top.pad_id = -1;
    top.stop_id_alt = frozen::kStopIdAlt;
    // The reference scales by head_dim**-0.5 (Inkling-1 used 1/head_dim).
    top.attn_scale_denom = static_cast<double>(top.head_dim);

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.moe_intermediate_size <= 0 || top.head_dim <= 0) {
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
            if ((l + 1) % frozen::kAttnResBlock == 0)
                top.full_layers.push_back(l);
            else
                top.sliding_layers.push_back(l);
        }
    }

    // ── expert codec split (from hf_quant_config.json when present) ────────
    top.expert_nvfp4.assign(static_cast<size_t>(top.num_layers), true);
    const fs::path q_path = fs::path(input_dir) / "hf_quant_config.json";
    if (fs::exists(q_path)) {
        const std::string qjson = ReadWholeFile(q_path);
        dom::parser qparser;
        simdjson::padded_string qpad(qjson);
        dom::element qdoc;
        if (qparser.parse(qpad).get(qdoc) != simdjson::SUCCESS) {
            top.error_msg = "hf_quant_config.json parse error";
            return top;
        }
        dom::array groups;
        if (qdoc["model"]["parameter_groups"].get_array().get(groups) !=
            simdjson::SUCCESS) {
            top.error_msg = "hf_quant_config.json has no model.parameter_groups";
            return top;
        }
        for (auto g : groups) {
            dom::object go;
            if (g.get_object().get(go) != simdjson::SUCCESS) continue;
            std::string_view compute;
            if (go["compute_dtype"].get_string().get(compute) !=
                simdjson::SUCCESS)
                continue;
            if (compute != "nvfp4" && compute != "mxfp8") continue;
            const bool nvfp4 = (compute == "nvfp4");
            dom::array tensors;
            if (go["tensors"].get_array().get(tensors) != simdjson::SUCCESS)
                continue;
            for (auto t : tensors) {
                std::string_view name;
                if (t.get_string().get(name) != simdjson::SUCCESS) continue;
                // model.llm.layers.<L>.mlp.experts.w13_weight
                const std::string_view key = "model.llm.layers.";
                if (name.substr(0, key.size()) != key) continue;
                std::string_view tail = name.substr(key.size());
                const size_t dot = tail.find('.');
                if (dot == std::string_view::npos) continue;
                int layer = -1;
                if (!ParseLayerIndex(tail.substr(0, dot), layer)) continue;
                if (layer < 0 || layer >= top.num_layers) continue;
                top.expert_nvfp4[static_cast<size_t>(layer)] = nvfp4;
            }
        }
    }
    return top;
}

ParsedInklingTensor ClassifyInklingTensor(const std::string &name)
{
    ParsedInklingTensor p;
    const std::string_view n(name);
    if (n.substr(0, sizeof(kVisualScope) - 1) == kVisualScope) {
        p.kind = InklingTensorKind::Vision;
        return p;
    }
    if (n.substr(0, sizeof(kLlmScope) - 1) != kLlmScope) return p;
    const std::string_view rest = n.substr(sizeof(kLlmScope) - 1);

    auto match = [](std::string_view s, std::string_view lit) {
        return s == lit;
    };

    if (match(rest, "embed.weight")) { p.kind = InklingTensorKind::Embed; return p; }
    if (match(rest, "unembed.weight")) { p.kind = InklingTensorKind::Unembed; return p; }
    if (match(rest, "embed_norm.weight")) { p.kind = InklingTensorKind::EmbedNorm; return p; }
    if (match(rest, "norm.weight")) { p.kind = InklingTensorKind::FinalNorm; return p; }
    if (match(rest, "attn_res.output_query")) {
        p.kind = InklingTensorKind::OutputQuery;
        return p;
    }
    {
        const std::string_view aq = "attn_res.attn_queries.";
        const std::string_view mq = "attn_res.mlp_queries.";
        if (rest.substr(0, aq.size()) == aq) {
            int idx = -1;
            if (ParseLayerIndex(rest.substr(aq.size()), idx)) {
                p.kind = InklingTensorKind::AttnQuery;
                // Source indexes attn queries by layer-1: layer 0 has none.
                p.layer = idx + 1;
            }
            return p;
        }
        if (rest.substr(0, mq.size()) == mq) {
            int idx = -1;
            if (ParseLayerIndex(rest.substr(mq.size()), idx)) {
                p.kind = InklingTensorKind::MlpQuery;
                p.layer = idx;
            }
            return p;
        }
    }

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
        InklingTensorKind kind;
    };
    static constexpr Entry kTable[] = {
        {"attn.attn_sink_proj.weight", InklingTensorKind::AttnSink},
        {"attn.gate_attn.weight", InklingTensorKind::AttnGate},
        {"attn.k_sconv.weight", InklingTensorKind::KSConv},
        {"attn.k_sconv.static_w", InklingTensorKind::KSConvStatic},
        {"attn.k_sconv.weight_proj", InklingTensorKind::KSConvProj},
        {"attn.v_sconv.weight", InklingTensorKind::VSConv},
        {"attn.v_sconv.static_w", InklingTensorKind::VSConvStatic},
        {"attn.v_sconv.weight_proj", InklingTensorKind::VSConvProj},
        {"attn.q_l2_clip.scale", InklingTensorKind::QClipScale},
        {"attn.rel_l2_clip.scale", InklingTensorKind::RelClipScale},
        {"attn.rel_logits_proj.proj", InklingTensorKind::RelProj},
        {"attn.wk_dv.weight", InklingTensorKind::Wk},
        {"attn.wv_dv.weight", InklingTensorKind::Wv},
        {"attn.wo_ud.weight", InklingTensorKind::Wo},
        {"attn.wq_du.weight", InklingTensorKind::Wq},
        {"attn.wr_du.weight", InklingTensorKind::Wr},
        {"attn_norm.weight", InklingTensorKind::AttnNorm},
        {"attn_sconv.weight", InklingTensorKind::AttnSConv},
        {"k_input_norm.weight", InklingTensorKind::KInputNorm},
        {"q_input_norm.weight", InklingTensorKind::QInputNorm},
        {"v_input_norm.weight", InklingTensorKind::VInputNorm},
        {"mlp.experts.w13_weight", InklingTensorKind::ExpertW13},
        {"mlp.experts.w13_weight_scale", InklingTensorKind::ExpertW13Scale},
        {"mlp.experts.w13_weight_amax", InklingTensorKind::ExpertW13Amax},
        {"mlp.experts.w13_weight_outer_scale_denominator",
         InklingTensorKind::ExpertW13Denom},
        {"mlp.experts.w2_weight", InklingTensorKind::ExpertW2},
        {"mlp.experts.w2_weight_scale", InklingTensorKind::ExpertW2Scale},
        {"mlp.experts.w2_weight_amax", InklingTensorKind::ExpertW2Amax},
        {"mlp.experts.w2_weight_outer_scale_denominator",
         InklingTensorKind::ExpertW2Denom},
        {"mlp.gate.bias", InklingTensorKind::RouterBias},
        {"mlp.gate.weight", InklingTensorKind::RouterGate},
        {"mlp_norm.weight", InklingTensorKind::MlpNorm},
        {"mlp_sconv.weight", InklingTensorKind::MlpSConv},
    };
    for (const auto &e : kTable)
        if (tail == e.name) {
            p.kind = e.kind;
            return p;
        }
    return p;
}

bool InklingTensorIsPinnedNvfp4(InklingTensorKind)
{
    // NOTHING on the dense side is NVFP4, by the checkpoint's own choice.
    //
    // `hf_quant_config.json` puts ONLY the routed experts in a low-precision
    // codec (NVFP4 for layers 0-39, MXFP8 for 40-47); every attention
    // projection, every gate/up/down of a dense MLP, both norms per layer, the
    // router and the embedding tables are `bfloat16`, and the AttnRes queries
    // and router bias are `float32`. Requantizing those BF16 weights costs a
    // MEASURED 9.5% relative L2 on the attention projections, and on this model
    // that is not free: with them at 4 bits the engine produced a repetition
    // loop (`: The capital of France is The capital of France is ...`) where the
    // release answers; with them at 16 bits it converts correctly-sized logits.
    //
    // So the artifact carries the release's precision: BF16 dense weights (wide
    // to FP16 on device, which is lossless for their magnitude range) and the
    // routed experts exactly as shipped (byte-verbatim NVFP4).
    return false;
}

namespace {

// Globals live in pinned/global*.arrow; the AttnRes query banks are layer
// indexed but are not per-layer records.
bool IsGlobalKind(InklingTensorKind k)
{
    switch (k) {
    case InklingTensorKind::Embed:
    case InklingTensorKind::Unembed:
    case InklingTensorKind::EmbedNorm:
    case InklingTensorKind::FinalNorm:
    case InklingTensorKind::OutputQuery:
    case InklingTensorKind::AttnQuery:
    case InklingTensorKind::MlpQuery:
        return true;
    default:
        return false;
    }
}

bool IsExpertKind(InklingTensorKind k)
{
    switch (k) {
    case InklingTensorKind::ExpertW13:
    case InklingTensorKind::ExpertW13Scale:
    case InklingTensorKind::ExpertW13Amax:
    case InklingTensorKind::ExpertW13Denom:
    case InklingTensorKind::ExpertW2:
    case InklingTensorKind::ExpertW2Scale:
    case InklingTensorKind::ExpertW2Amax:
    case InklingTensorKind::ExpertW2Denom:
        return true;
    default:
        return false;
    }
}

// Frozen passport cross-check; relaxed under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
bool CheckFrozenGeometry(const InklingConvertTopology &top, std::string &why)
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
        return bad("num_attention_heads", top.num_attention_heads, frozen::kHeads);
    if (top.num_key_value_heads != frozen::kKvHeads)
        return bad("num_key_value_heads", top.num_key_value_heads,
                   frozen::kKvHeads);
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
    if (top.attn_res_block != frozen::kAttnResBlock)
        return bad("n_layers_per_attn_res_block", top.attn_res_block,
                   frozen::kAttnResBlock);
    if (top.num_experts != frozen::kExperts)
        return bad("n_routed_experts", top.num_experts, frozen::kExperts);
    if (top.num_experts_per_tok != frozen::kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, frozen::kTopK);
    if (top.moe_intermediate_size != frozen::kMoeIntermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   frozen::kMoeIntermediate);
    if (top.num_shared_experts != frozen::kSharedExperts)
        return bad("n_shared_experts", top.num_shared_experts,
                   frozen::kSharedExperts);
    if (top.full_layers.size() !=
        static_cast<size_t>(frozen::kFullLayerCount)) {
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
    int nvfp4_layers = 0;
    for (bool b : top.expert_nvfp4) nvfp4_layers += b ? 1 : 0;
    if (nvfp4_layers != frozen::kNovfp4ExpertLayerCount)
        return bad("nvfp4 expert layers", nvfp4_layers,
                   frozen::kNovfp4ExpertLayerCount);
    return true;
}

struct SourceTensor {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

std::vector<float> ToF32(const std::vector<uint8_t> &data, const std::string &dt)
{
    std::vector<float> out;
    if (dt == "F32") {
        out.resize(data.size() / 4);
        std::memcpy(out.data(), data.data(), out.size() * 4);
    } else if (dt == "BF16") {
        out.resize(data.size() / 2);
        const uint16_t *h = reinterpret_cast<const uint16_t *>(data.data());
        for (size_t i = 0; i < out.size(); ++i) {
            const uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            out[i] = f;
        }
    }
    return out;
}

}  // namespace

arrow::Status ConvertInklingModel(const std::string &input_dir,
                                  const std::string &output_root)
{
    InklingConvertTopology top = ParseInklingConfig(input_dir);
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
    std::map<std::string, SourceTensor> weights;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = ClassifyInklingTensor(t.name);
            switch (p.kind) {
            case InklingTensorKind::Vision:
                ++top.skipped_visual_tensors;
                continue;
            case InklingTensorKind::Unknown:
                return arrow::Status::Invalid(
                    "unrecognized tensor in the inkling2 checkpoint: ", t.name);
            default:
                break;
            }
            if ((p.kind == InklingTensorKind::AttnQuery ||
                 p.kind == InklingTensorKind::MlpQuery) &&
                (p.layer < 0 || p.layer >= top.num_layers))
                return arrow::Status::Invalid("attn_res query layer out of range: ",
                                              t.name);
            if (p.layer >= top.num_layers && p.layer >= 0)
                return arrow::Status::Invalid("layer index out of range: ",
                                              t.name);
            SourceTensor st;
            st.shard = &shard;
            st.info = &t;
            weights[t.name] = std::move(st);
        }
    }

    // Cross-check the expert codec split against the observed weight dtypes.
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string key = absl::StrFormat(
            "model.llm.layers.%d.mlp.experts.w13_weight", l);
        auto it = weights.find(key);
        if (it == weights.end())
            return arrow::Status::Invalid("missing routed experts for layer ", l);
        const std::string &dt = it->second.info->dtype;
        const bool nvfp4 = (dt == "U8");
        if (!nvfp4 && dt != "F8_E4M3")
            return arrow::Status::Invalid(absl::StrFormat(
                "layer %d routed experts have unsupported dtype %s", l, dt));
        if (top.expert_nvfp4[static_cast<size_t>(l)] != nvfp4)
            return arrow::Status::Invalid(absl::StrFormat(
                "layer %d expert codec mismatch: hf_quant_config says %s, "
                "tensor dtype is %s",
                l, top.expert_nvfp4[static_cast<size_t>(l)] ? "nvfp4" : "mxfp8",
                dt.c_str()));
    }

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path experts_dir = base / "experts";
    fs::create_directories(pinned_dir);
    fs::create_directories(experts_dir);

    std::vector<TensorRecord> global_raw;
    std::vector<DenseNvfp4Record> global_nvfp4;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    int pinned_count = 0;

    // ── dense / raw tensors ────────────────────────────────────────────────
    for (const auto &[name, st] : weights) {
        if (st.info == nullptr) continue;
        const auto p = ClassifyInklingTensor(name);
        if (IsExpertKind(p.kind)) continue;
        auto data = ReadTensorData(input_dir, *st.shard, *st.info);
        if (data.empty() ||
            static_cast<int64_t>(data.size()) != st.info->byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        // The source indexes the attention mixtures by `layer-1` (layer 0 has
        // no attention mixture) while the MLP mixtures are indexed by layer.
        // The artifact stores both under their owning layer so the loader
        // cannot repeat the off-by-one.
        std::string out_name = name;
        if (p.kind == InklingTensorKind::AttnQuery)
            out_name = absl::StrFormat("model.llm.attn_res.attn_queries.%d",
                                       p.layer);

        if (!InklingTensorIsPinnedNvfp4(p.kind)) {
            TensorRecord r;
            r.name = out_name;
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

        DenseNvfp4Record r;
        r.name = out_name;
        r.shape = st.info->shape;
        r.dtype = "NVFP4";
        const int64_t numel = st.info->numel();
        if (!QuantizeSlice(data.data(), numel, st.info->dtype, r))
            return arrow::Status::IOError("NVFP4 quantization failed: ", name);
        if (IsGlobalKind(p.kind))
            global_nvfp4.push_back(std::move(r));
        else
            nvfp4_by_layer[p.layer].push_back(std::move(r));
        ++pinned_count;
    }

    // ── routed experts: one Arrow file per (layer, expert) ─────────────────
    const int E = top.num_experts;
    const int I = top.moe_intermediate_size;
    const int H = top.hidden_size;
    int experts_written = 0;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string w13_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w13_weight", l);
        const std::string w2_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w2_weight", l);
        const auto &w13 = weights.at(w13_key);
        const auto &w2 = weights.at(w2_key);
        const bool nvfp4 = top.expert_nvfp4[static_cast<size_t>(l)];

        const std::string s13_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w13_weight_scale", l);
        const std::string s2_key =
            absl::StrFormat("model.llm.layers.%d.mlp.experts.w2_weight_scale", l);
        if (weights.find(s13_key) == weights.end() ||
            weights.find(s2_key) == weights.end())
            return arrow::Status::Invalid("missing expert scales for layer ", l);
        const auto &s13 = weights.at(s13_key);
        const auto &s2 = weights.at(s2_key);

        auto w13_data = ReadTensorData(input_dir, *w13.shard, *w13.info);
        auto w2_data = ReadTensorData(input_dir, *w2.shard, *w2.info);
        auto s13_data = ReadTensorData(input_dir, *s13.shard, *s13.info);
        auto s2_data = ReadTensorData(input_dir, *s2.shard, *s2.info);
        if (w13_data.empty() || w2_data.empty() || s13_data.empty() ||
            s2_data.empty())
            return arrow::Status::IOError("failed to read experts for layer ", l);

        // Per-expert global scales for the NVFP4 source.
        std::vector<float> amax13, amax2;
        float denom13 = 2688.0f, denom2 = 2688.0f;
        if (nvfp4) {
            const std::string a13_key = absl::StrFormat(
                "model.llm.layers.%d.mlp.experts.w13_weight_amax", l);
            const std::string a2_key = absl::StrFormat(
                "model.llm.layers.%d.mlp.experts.w2_weight_amax", l);
            const std::string d13_key = absl::StrFormat(
                "model.llm.layers.%d.mlp.experts.w13_weight_outer_scale_"
                "denominator",
                l);
            const std::string d2_key = absl::StrFormat(
                "model.llm.layers.%d.mlp.experts.w2_weight_outer_scale_"
                "denominator",
                l);
            if (weights.find(a13_key) == weights.end() ||
                weights.find(a2_key) == weights.end() ||
                weights.find(d13_key) == weights.end() ||
                weights.find(d2_key) == weights.end())
                return arrow::Status::Invalid(
                    "NVFP4 expert layer is missing amax/denominator: ", l);
            const auto &a13 = weights.at(a13_key);
            const auto &a2 = weights.at(a2_key);
            const auto &d13 = weights.at(d13_key);
            const auto &d2 = weights.at(d2_key);
            amax13 = ToF32(ReadTensorData(input_dir, *a13.shard, *a13.info),
                           a13.info->dtype);
            amax2 = ToF32(ReadTensorData(input_dir, *a2.shard, *a2.info),
                          a2.info->dtype);
            auto dv13 = ToF32(ReadTensorData(input_dir, *d13.shard, *d13.info),
                              d13.info->dtype);
            auto dv2 = ToF32(ReadTensorData(input_dir, *d2.shard, *d2.info),
                             d2.info->dtype);
            if (amax13.size() != static_cast<size_t>(E) ||
                amax2.size() != static_cast<size_t>(E) || dv13.empty() ||
                dv2.empty())
                return arrow::Status::Invalid("expert amax/denominator shape: ",
                                              l);
            denom13 = dv13[0];
            denom2 = dv2[0];
            if (!(denom13 > 0.0f) || !(denom2 > 0.0f))
                return arrow::Status::Invalid("expert outer scale denominator: ", l);
        }

        // Per-tensor strides.
        const size_t w13_row = nvfp4 ? static_cast<size_t>(H / 2)
                                     : static_cast<size_t>(H);
        const size_t w13_scale_row =
            nvfp4 ? static_cast<size_t>(H / 16) : static_cast<size_t>(H / 32);
        const size_t w2_row = nvfp4 ? static_cast<size_t>(I / 2)
                                    : static_cast<size_t>(I);
        const size_t w2_scale_row =
            nvfp4 ? static_cast<size_t>(I / 16) : static_cast<size_t>(I / 32);
        const size_t w13_expert = static_cast<size_t>(2 * I) * w13_row;
        const size_t w13_scale_expert = static_cast<size_t>(2 * I) * w13_scale_row;
        const size_t w2_expert = static_cast<size_t>(H) * w2_row;
        const size_t w2_scale_expert = static_cast<size_t>(H) * w2_scale_row;
        if (w13_data.size() != w13_expert * static_cast<size_t>(E) ||
            w2_data.size() != w2_expert * static_cast<size_t>(E) ||
            s13_data.size() != w13_scale_expert * static_cast<size_t>(E) ||
            s2_data.size() != w2_scale_expert * static_cast<size_t>(E))
            return arrow::Status::Invalid(
                "routed expert tensor size does not match the geometry on layer ",
                l);

        const fs::path dir = experts_dir / absl::StrCat("layer_", l);
        fs::create_directories(dir);
        for (int e = 0; e < E; ++e) {
            PackedExpert pe;
            if (nvfp4) {
                const uint8_t *w13p = w13_data.data() + w13_expert * e;
                const uint8_t *w2p = w2_data.data() + w2_expert * e;
                const uint8_t *s13p = s13_data.data() + w13_scale_expert * e;
                const uint8_t *s2p = s2_data.data() + w2_scale_expert * e;
                const size_t gate_rows = static_cast<size_t>(I);
                pe.w1_packed.assign(w13p, w13p + gate_rows * w13_row);
                pe.w3_packed.assign(w13p + gate_rows * w13_row,
                                    w13p + 2 * gate_rows * w13_row);
                pe.w1_scales.assign(s13p, s13p + gate_rows * w13_scale_row);
                pe.w3_scales.assign(s13p + gate_rows * w13_scale_row,
                                    s13p + 2 * gate_rows * w13_scale_row);
                pe.w2_packed.assign(w2p, w2p + w2_expert);
                pe.w2_scales.assign(s2p, s2p + w2_scale_expert);
                pe.w1_global_scale = amax13[static_cast<size_t>(e)] / denom13;
                pe.w3_global_scale = pe.w1_global_scale;
                pe.w2_global_scale = amax2[static_cast<size_t>(e)] / denom2;
            } else {
                const std::vector<int64_t> gate_shape = {I, H};
                const std::vector<int64_t> down_shape = {H, I};
                const std::vector<int64_t> gate_scale_shape = {I, H / 32};
                const std::vector<int64_t> down_scale_shape = {H, I / 32};
                const uint8_t *w13p = w13_data.data() + w13_expert * e;
                const uint8_t *w2p = w2_data.data() + w2_expert * e;
                const uint8_t *s13p = s13_data.data() + w13_scale_expert * e;
                const uint8_t *s2p = s2_data.data() + w2_scale_expert * e;
                const size_t gate_bytes = static_cast<size_t>(I) * H;
                DenseNvfp4Record g, u, d;
                if (!QuantizeSlice(w13p, static_cast<int64_t>(I) * H, "F8_E4M3",
                                   s13p, gate_shape, gate_scale_shape, g))
                    return arrow::Status::IOError(
                        "MXFP8 gate requantization failed: layer ", l, " expert ",
                        e);
                if (!QuantizeSlice(w13p + gate_bytes,
                                   static_cast<int64_t>(I) * H, "F8_E4M3",
                                   s13p + static_cast<size_t>(I) * (H / 32),
                                   gate_shape, gate_scale_shape, u))
                    return arrow::Status::IOError(
                        "MXFP8 up requantization failed: layer ", l, " expert ", e);
                if (!QuantizeSlice(w2p, static_cast<int64_t>(H) * I, "F8_E4M3",
                                   s2p, down_shape, down_scale_shape, d))
                    return arrow::Status::IOError(
                        "MXFP8 down requantization failed: layer ", l, " expert ",
                        e);
                pe.w1_packed = std::move(g.packed);
                pe.w1_scales = std::move(g.scales);
                pe.w1_global_scale = g.global_scale;
                pe.w3_packed = std::move(u.packed);
                pe.w3_scales = std::move(u.scales);
                pe.w3_global_scale = u.global_scale;
                pe.w2_packed = std::move(d.packed);
                pe.w2_scales = std::move(d.scales);
                pe.w2_global_scale = d.global_scale;
            }
            pe.w1_shape = {I, H};
            pe.w2_shape = {H, I};
            pe.w3_shape = {I, H};
            ARROW_RETURN_NOT_OK(
                WriteExpertArrow(ExpertArrowPath(base, l, e), pe));
            ++experts_written;
        }
    }

    // ── write pinned batches ──────────────────────────────────────────────
    if (!global_nvfp4.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                                       pinned_dir / "global_nvfp4.arrow"));
    if (!global_raw.empty())
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

    // ── VRAM profile ──────────────────────────────────────────────────────
    std::vector<VramTensorInfo> vram_tensors;
    VramTensorPolicy policy;
    policy.is_nvfp4 = [](const std::string &name) {
        return InklingTensorIsPinnedNvfp4(ClassifyInklingTensor(name).kind);
    };
    // Every routed-expert tensor (weights AND their amax/denominator/scale
    // companions) belongs to the routed class; leaving the companions resident
    // inflates the estimate by the whole per-16 scale plane (~30 GiB here).
    policy.mem_class = [](const std::string &name) {
        switch (ClassifyInklingTensor(name).kind) {
        case InklingTensorKind::ExpertW13:
        case InklingTensorKind::ExpertW13Scale:
        case InklingTensorKind::ExpertW13Amax:
        case InklingTensorKind::ExpertW13Denom:
        case InklingTensorKind::ExpertW2:
        case InklingTensorKind::ExpertW2Scale:
        case InklingTensorKind::ExpertW2Amax:
        case InklingTensorKind::ExpertW2Denom:
            return VramTensorClass::RoutedExpert;
        default:
            return VramTensorClass::Resident;
        }
    };
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            auto p = ClassifyInklingTensor(t.name);
            if (p.kind == InklingTensorKind::Vision) continue;
            VramTensorInfo ti;
            ti.name = t.name;
            ti.numel = t.numel();
            ti.raw_bytes = t.byte_size();
            vram_tensors.push_back(std::move(ti));
        }
    VramGeometry geom;
    geom.hidden_dim = top.hidden_size;
    geom.kv_dim = top.num_key_value_heads * top.head_dim;
    geom.num_attn_layers = top.num_layers;
    geom.num_moe_layers = top.num_layers;
    geom.experts_per_layer = top.num_experts;
    geom.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    geom.nvfp4_scale_block = frozen::kQuantBlock;
    const VramProfile vram = EstimateVram(vram_tensors, policy, geom);

    // ── index ─────────────────────────────────────────────────────────────
    {
        // 17 significant digits must be set BEFORE the numbers are formatted
        // into the stream (setting it at the end leaves them at the default 6,
        // which turned the clip threshold 22.627416997969522 into 22.6274).
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
        auto bool_array = [&j](const char *key, const std::vector<bool> &v,
                               bool trailing) {
            j << "  \"" << key << "\": [";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) j << ", ";
                j << (v[i] ? "true" : "false");
            }
            j << "]" << (trailing ? ",\n" : "\n");
        };
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n";
        j << "  \"source_model_type\": \"" << top.model_type << "\",\n";
        j << "  \"num_layers\": " << top.num_layers << ",\n";
        j << "  \"hidden_size\": " << top.hidden_size << ",\n";
        j << "  \"vocab_size\": " << top.vocab_size << ",\n";
        j << "  \"unpadded_vocab_size\": " << top.unpadded_vocab_size << ",\n";
        j << "  \"num_attention_heads\": " << top.num_attention_heads << ",\n";
        j << "  \"num_key_value_heads\": " << top.num_key_value_heads << ",\n";
        j << "  \"head_dim\": " << top.head_dim << ",\n";
        j << "  \"attn_scale_denom\": " << top.attn_scale_denom << ",\n";
        j << "  \"d_rel\": " << top.d_rel << ",\n";
        j << "  \"rel_extent\": " << top.rel_extent << ",\n";
        j << "  \"sliding_window_size\": " << top.sliding_window_size << ",\n";
        j << "  \"conv_kernel\": " << top.conv_kernel << ",\n";
        j << "  \"attn_res_block\": " << top.attn_res_block << ",\n";
        j << "  \"num_experts\": " << top.num_experts << ",\n";
        j << "  \"num_experts_per_tok\": " << top.num_experts_per_tok << ",\n";
        j << "  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n";
        j << "  \"num_shared_experts\": " << top.num_shared_experts << ",\n";
        j << "  \"moe_renormalize\": " << (top.moe_renormalize ? "true" : "false")
          << ",\n";
        j << "  \"routed_scaling_factor\": " << top.route_scale << ",\n";
        j << "  \"rms_norm_eps\": " << top.rms_norm_eps << ",\n";
        j << "  \"qk_l2_clip\": " << top.qk_l2_clip << ",\n";
        j << "  \"rel_l2_clip\": " << top.rel_l2_clip << ",\n";
        j << "  \"log_scaling_n_floor\": " << top.log_scaling_n_floor << ",\n";
        j << "  \"log_scaling_alpha\": " << top.log_scaling_alpha << ",\n";
        j << "  \"mlp_gate_softcap\": " << top.mlp_gate_softcap << ",\n";
        j << "  \"mlp_up_softcap\": " << top.mlp_up_softcap << ",\n";
        j << "  \"final_logit_softcap\": " << top.final_logit_softcap << ",\n";
        j << "  \"bos_id\": " << top.bos_id << ",\n";
        j << "  \"eos_id\": " << top.eos_id << ",\n";
        j << "  \"stop_id_alt\": " << top.stop_id_alt << ",\n";
        j << "  \"pad_id\": " << top.pad_id << ",\n";
        j << "  \"tokenizer_kind\": " << frozen::kTokenizerKind << ",\n";
        j << "  \"tokenizer\": \"" << (tokenizer_ok ? "tiktoken" : "fixture")
          << "\",\n";
        j << "  \"skipped_visual_tensors\": " << top.skipped_visual_tensors
          << ",\n";
        j << "  \"skipped_mtp_tensors\": 0,\n";
        j << "  \"total_pinned_tensors\": " << top.total_pinned_tensors << ",\n";
        j << "  \"total_experts\": " << experts_written << ",\n";
        int_array("full_layers", top.full_layers, true);
        int_array("sliding_layers", top.sliding_layers, true);
        bool_array("expert_nvfp4", top.expert_nvfp4, true);
        j << FormatVramJson(vram) << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    lsxcommon::log::info(absl::StrFormat(
        "Inkling conversion: %d layers (%zu full / %zu sliding), %d experts x "
        "%d layers, %lld pinned tensors, %d expert files, stripped %d vision "
        "tensors, tiktoken %s",
        top.num_layers, top.full_layers.size(), top.sliding_layers.size(),
        top.num_experts, top.num_layers, static_cast<long long>(pinned_count),
        experts_written, top.skipped_visual_tensors,
        tokenizer_ok ? "built" : "absent"));
    int nvfp4_layers = 0;
    for (bool b : top.expert_nvfp4) nvfp4_layers += b ? 1 : 0;
    lsxcommon::log::info(absl::StrFormat(
        "Inkling expert codec: %d layers NVFP4 (verbatim), %d layers MXFP8 -> "
        "NVFP4 requantized",
        nvfp4_layers, top.num_layers - nvfp4_layers));
    return arrow::Status::OK();
}

}  // namespace lsxcommon::inkling
