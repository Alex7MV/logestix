// Inkling-2-Mini converter (Model Ingestion Protocol stages 1-2).
//
// Source: modelopt release `thinkingmachines/Inkling-2-Mini` (inkling2_loop).
// The traps pinned here are the ones measured on Inkling-2-Small plus the
// loop-specific ones:
//
//   1. MIXED EXPERT CODEC. `hf_quant_config.json` puts the routed experts of
//      layers 0..19 in NVFP4 (E2M1 nibbles, per-16 E4M3FN scales, per-expert
//      `amax` with `outer_scale_denominator` 2688) and layers 20..23 in MXFP8
//      (E4M3 weights, ue8m0 scale per 32). The split is taken from the
//      OBSERVED dtype of each layer's `w13_weight` and cross-checked against
//      `hf_quant_config.json`.
//   2. The NVFP4 half passes through verbatim; the per-expert global scale
//      follows the checkpoint's own invariant, verified numerically on the
//      real shards: 6 * max(block_scale_e) * (amax_e / 2688) == amax_e.
//   3. `attn_res.attn_queries.L` is indexed by `layer-1` (layer 0 has no
//      attention mixture) while `mlp_queries.L` is indexed by layer.
//   4. `mlp.experts.w13_weight` and `w13_weight_scale` are stored
//      ROW-INTERLEAVED ([gate_0, up_0, gate_1, up_1, ...]) and must be
//      de-interleaved before the gate/up split — the release's own conversion
//      does exactly that (`Inkling2LoopDequantizeExperts` + `Interleave(dim=1)`
//      in transformers 5.24's conversion_mapping.py, entry "inkling2_loop").
//      Splitting the slab in halves keeps every byte and every magnitude but
//      destroys the model's knowledge (measured on Small: perplexity 12,881
//      instead of 30.4). `test_inkling_moe_loop_convert` pins the row pairing
//      for BOTH codecs, weights AND scales.
//   5. The two cross-pass query tensors merge into one [loop_layers, 4,
//      hidden] FP32 record, order [q, k, v, mlp] per loop layer:
//      q=queries[2i], mlp=queries[2i+1], k=kv_queries[2i], v=kv_queries[2i+1]
//      (the release's Inkling2LoopMergeMessageQueries, verified on the real
//      tensors with a byte-exact inverse round-trip).
//   6. The loop controllers (four MessageGates per loop layer) are dense
//      side tensors: down [rank, 2*hidden] BF16, forget/gain [hidden, rank].
//      They exist ONLY on layers [loop_start, loop_end).
#include "lsxcommon/inkling_moe_loop_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/synthetic_geometry.h"
#include "inkling_moe_loop_constants.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::inkling {

InklingLoopConvertSpec MiniConvertSpec()
{
    using namespace lsxcommon::inkling::loop::frozen;
    InklingLoopConvertSpec spec;
    spec.model_name = "Inkling-2-Mini";
    spec.varloop = false;
    spec.max_passes = kLoopPasses;
    spec.frozen.hidden = kHidden;
    spec.frozen.layers = kLayers;
    spec.frozen.vocab = kVocab;
    spec.frozen.base_vocab = kBaseVocab;
    spec.frozen.heads = kHeads;
    spec.frozen.kv_heads = kKvHeads;
    spec.frozen.head_dim = kHeadDim;
    spec.frozen.d_rel = kDRel;
    spec.frozen.rel_extent = kRelExtent;
    spec.frozen.sliding_window = kSlidingWindow;
    spec.frozen.conv_kernel = kConvKernel;
    spec.frozen.attn_res_block = kAttnResBlock;
    spec.frozen.experts = kExperts;
    spec.frozen.top_k = kTopK;
    spec.frozen.moe_intermediate = kMoeIntermediate;
    spec.frozen.shared_experts = kSharedExperts;
    spec.frozen.loop_start = kLoopStart;
    spec.frozen.loop_end = kLoopEnd;
    spec.frozen.loop_passes = kLoopPasses;
    spec.frozen.loop_gate_rank = kLoopGateRank;
    spec.frozen.log_scaling_floor = kLogScaleFloor;
    spec.frozen.nvfp4_expert_layers = kNovfp4ExpertLayerCount;
    spec.frozen.full_layers.assign(kFullLayers.begin(), kFullLayers.end());
    return spec;
}

namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::inkling::loop::frozen;

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

bool IsGlobalKind(InklingLoopTensorKind k)
{
    switch (k) {
    case InklingLoopTensorKind::Embed:
    case InklingLoopTensorKind::Unembed:
    case InklingLoopTensorKind::EmbedNorm:
    case InklingLoopTensorKind::FinalNorm:
    case InklingLoopTensorKind::OutputQuery:
    case InklingLoopTensorKind::AttnQuery:
    case InklingLoopTensorKind::MlpQuery:
    case InklingLoopTensorKind::CrossPassQueries:
    case InklingLoopTensorKind::CrossPassKvQueries:
        return true;
    default:
        return false;
    }
}

bool IsExpertKind(InklingLoopTensorKind k)
{
    switch (k) {
    case InklingLoopTensorKind::ExpertW13:
    case InklingLoopTensorKind::ExpertW13Scale:
    case InklingLoopTensorKind::ExpertW13Amax:
    case InklingLoopTensorKind::ExpertW13Denom:
    case InklingLoopTensorKind::ExpertW2:
    case InklingLoopTensorKind::ExpertW2Scale:
    case InklingLoopTensorKind::ExpertW2Amax:
    case InklingLoopTensorKind::ExpertW2Denom:
        return true;
    default:
        return false;
    }
}

// Frozen passport cross-check; relaxed under LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY.
// The numbers come from the SPEC (the model row's own passport) and never from
// a shared header: converting a tree under one model's identity must not
// validate it against another model's geometry.
bool CheckFrozenGeometry(const InklingLoopConvertTopology &top,
                         const InklingLoopFrozenGeometry &want,
                         const InklingLoopConvertSpec &spec, std::string &why)
{
    auto bad = [&why](const char *field, long long got, long long want) {
        why = absl::StrFormat("%s: got %lld, want %lld", field, got, want);
        return false;
    };
    if (top.hidden_size != want.hidden)
        return bad("hidden_size", top.hidden_size, want.hidden);
    if (top.num_layers != want.layers)
        return bad("num_hidden_layers", top.num_layers, want.layers);
    if (top.vocab_size != want.vocab)
        return bad("vocab_size", top.vocab_size, want.vocab);
    if (top.unpadded_vocab_size != want.base_vocab)
        return bad("unpadded_vocab_size", top.unpadded_vocab_size, want.base_vocab);
    if (top.num_attention_heads != want.heads)
        return bad("num_attention_heads", top.num_attention_heads, want.heads);
    if (top.num_key_value_heads != want.kv_heads)
        return bad("num_key_value_heads", top.num_key_value_heads, want.kv_heads);
    if (top.head_dim != want.head_dim)
        return bad("head_dim", top.head_dim, want.head_dim);
    if (top.d_rel != want.d_rel) return bad("d_rel", top.d_rel, want.d_rel);
    if (top.rel_extent != want.rel_extent)
        return bad("rel_extent", top.rel_extent, want.rel_extent);
    if (top.sliding_window_size != want.sliding_window)
        return bad("sliding_window_size", top.sliding_window_size,
                   want.sliding_window);
    if (top.conv_kernel != want.conv_kernel)
        return bad("sconv_kernel_size", top.conv_kernel, want.conv_kernel);
    if (top.attn_res_block != want.attn_res_block)
        return bad("n_layers_per_attn_res_block", top.attn_res_block,
                   want.attn_res_block);
    if (top.num_experts != want.experts)
        return bad("n_routed_experts", top.num_experts, want.experts);
    if (top.num_experts_per_tok != want.top_k)
        return bad("num_experts_per_tok", top.num_experts_per_tok, want.top_k);
    if (top.moe_intermediate_size != want.moe_intermediate)
        return bad("moe_intermediate_size", top.moe_intermediate_size,
                   want.moe_intermediate);
    if (top.num_shared_experts != want.shared_experts)
        return bad("n_shared_experts", top.num_shared_experts,
                   want.shared_experts);
    if (top.loop_start != want.loop_start)
        return bad("loop_config.core_start", top.loop_start, want.loop_start);
    if (top.loop_end != want.loop_end)
        return bad("loop_config.core_end", top.loop_end, want.loop_end);
    // The DECLARED count is the checkpoint's own, so it is checked exactly in
    // both pipelines; a varloop tree additionally carries a ceiling.
    if (top.loop_passes != want.loop_passes)
        return bad("loop_config.num_passes", top.loop_passes, want.loop_passes);
    if (spec.varloop && (top.loop_passes < 1 ||
                         top.loop_passes > spec.max_passes)) {
        why = absl::StrFormat(
            "loop_config.num_passes %d is outside the varloop range 1..%d",
            top.loop_passes, spec.max_passes);
        return false;
    }
    if (top.loop_gate_rank != want.loop_gate_rank)
        return bad("loop_config.gate_rank", top.loop_gate_rank,
                   want.loop_gate_rank);
    if (top.log_scaling_n_floor != want.log_scaling_floor)
        return bad("log_scaling_n_floor", top.log_scaling_n_floor,
                   want.log_scaling_floor);
    if (top.full_layers.size() != want.full_layers.size()) {
        why = absl::StrFormat("full-attention layers: got %zu, want %zu",
                              top.full_layers.size(), want.full_layers.size());
        return false;
    }
    for (size_t i = 0; i < top.full_layers.size(); ++i)
        if (top.full_layers[i] != want.full_layers[i]) {
            why = absl::StrFormat("full-attention layer %zu: got %d, want %d", i,
                                  top.full_layers[i], want.full_layers[i]);
            return false;
        }
    int nvfp4_layers = 0;
    for (bool b : top.expert_nvfp4) nvfp4_layers += b ? 1 : 0;
    if (nvfp4_layers != want.nvfp4_expert_layers)
        return bad("nvfp4 expert layers", nvfp4_layers,
                   want.nvfp4_expert_layers);
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

InklingLoopConvertTopology ParseInklingLoopConfig(
    const std::string &input_dir, const InklingLoopConvertSpec &spec)
{
    InklingLoopConvertTopology top;
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

    // The loop core (loop_config in the checkpoint's own spelling).
    dom::object loop;
    if (text["loop_config"].get_object().get(loop) == simdjson::SUCCESS) {
        GetInt(loop, "core_start", top.loop_start);
        GetInt(loop, "core_end", top.loop_end);
        GetInt(loop, "num_passes", top.loop_passes);
        GetInt(loop, "gate_rank", top.loop_gate_rank);
        bool varloop = false;
        if (loop["is_varloop"].get_bool().get(varloop) != simdjson::SUCCESS)
            varloop = false;
        // The declared capability IS the identity of the two rows: the same
        // architecture, one trained at a fixed count and one trained across a
        // range. Accepting the mismatch would publish a tree whose pass count
        // (and, for the varloop row, whose whole point) is wrong — and the
        // resulting model still answers fluently, so nothing downstream
        // catches it.
        if (varloop != spec.varloop) {
            top.error_msg = absl::StrFormat(
                "loop_config.is_varloop=%s does not match this pipeline (this "
                "row converts is_varloop=%s checkpoints)",
                varloop ? "true" : "false", spec.varloop ? "true" : "false");
            return top;
        }
        top.varloop = varloop;
        top.loop_passes_max = spec.varloop ? spec.max_passes : top.loop_passes;
        if (top.loop_passes_max < top.loop_passes || top.loop_passes_max < 1) {
            top.error_msg = absl::StrFormat(
                "loop_passes_max %d cannot be below the declared "
                "loop_config.num_passes %d",
                top.loop_passes_max, top.loop_passes);
            return top;
        }
    } else {
        top.error_msg = "config.json text_config has no loop_config";
        return top;
    }

    std::vector<int> local_layers;
    GetIntArray(text, "local_layer_ids", local_layers);

    int64_t eos = 0;
    if (root["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    else if (text["eos_token_id"].get_int64().get(eos) == simdjson::SUCCESS)
        top.eos_id = static_cast<int>(eos);
    // No BOS in this checkpoint; generation_config.json names <|endoftext|>
    // (199999) as the pad id.
    top.bos_id = -1;
    top.pad_id = frozen::kPadId;
    top.stop_id_alt = frozen::kStopIdAlt;
    // The reference scales by head_dim**-0.5 (Inkling-1 used 1/head_dim).
    top.attn_scale_denom = static_cast<double>(top.head_dim);

    if (top.hidden_size <= 0 || top.num_layers <= 0 || top.num_experts <= 0 ||
        top.moe_intermediate_size <= 0 || top.head_dim <= 0) {
        top.error_msg = "config.json is missing required geometry";
        return top;
    }
    // The reference config validator's own rules.
    if (top.loop_passes < 1 || top.loop_start <= 0 ||
        top.loop_start >= top.loop_end || top.loop_end > top.num_layers ||
        top.loop_start % top.attn_res_block != 0 ||
        top.loop_end % top.attn_res_block != 0) {
        top.error_msg = "loop_config bounds are invalid (must be block "
                        "multiples inside the stack, at least one pass)";
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

ParsedInklingLoopTensor ClassifyInklingLoopTensor(const std::string &name)
{
    ParsedInklingLoopTensor p;
    const std::string_view n(name);
    if (n.substr(0, sizeof(kVisualScope) - 1) == kVisualScope) {
        p.kind = InklingLoopTensorKind::Vision;
        return p;
    }
    if (n.substr(0, sizeof(kLlmScope) - 1) != kLlmScope) return p;
    const std::string_view rest = n.substr(sizeof(kLlmScope) - 1);

    auto match = [](std::string_view s, std::string_view lit) {
        return s == lit;
    };

    if (match(rest, "embed.weight")) { p.kind = InklingLoopTensorKind::Embed; return p; }
    if (match(rest, "unembed.weight")) { p.kind = InklingLoopTensorKind::Unembed; return p; }
    if (match(rest, "embed_norm.weight")) { p.kind = InklingLoopTensorKind::EmbedNorm; return p; }
    if (match(rest, "norm.weight")) { p.kind = InklingLoopTensorKind::FinalNorm; return p; }
    if (match(rest, "attn_res.output_query")) {
        p.kind = InklingLoopTensorKind::OutputQuery;
        return p;
    }
    if (match(rest, "middle_cycle_cross_pass_queries")) {
        p.kind = InklingLoopTensorKind::CrossPassQueries;
        return p;
    }
    if (match(rest, "middle_cycle_cross_pass_kv_queries")) {
        p.kind = InklingLoopTensorKind::CrossPassKvQueries;
        return p;
    }
    {
        const std::string_view aq = "attn_res.attn_queries.";
        const std::string_view mq = "attn_res.mlp_queries.";
        if (rest.substr(0, aq.size()) == aq) {
            int idx = -1;
            if (ParseLayerIndex(rest.substr(aq.size()), idx)) {
                p.kind = InklingLoopTensorKind::AttnQuery;
                // Source indexes attn queries by layer-1: layer 0 has none.
                p.layer = idx + 1;
            }
            return p;
        }
        if (rest.substr(0, mq.size()) == mq) {
            int idx = -1;
            if (ParseLayerIndex(rest.substr(mq.size()), idx)) {
                p.kind = InklingLoopTensorKind::MlpQuery;
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

    // Loop controllers: controllers.<m>.{down,forget,gain}_weight.
    const std::string_view cp = "controllers.";
    if (tail.substr(0, cp.size()) == cp) {
        std::string_view ct = tail.substr(cp.size());
        const size_t cdot = ct.find('.');
        if (cdot == std::string_view::npos) return p;
        const std::string_view msg = ct.substr(0, cdot);
        const std::string_view field = ct.substr(cdot + 1);
        if (msg == "attn") p.message = 0;
        else if (msg == "k") p.message = 1;
        else if (msg == "v") p.message = 2;
        else if (msg == "mlp") p.message = 3;
        else return p;
        if (field == "down_weight") p.kind = InklingLoopTensorKind::ControllerDown;
        else if (field == "forget_weight") p.kind = InklingLoopTensorKind::ControllerForget;
        else if (field == "gain_weight") p.kind = InklingLoopTensorKind::ControllerGain;
        return p;
    }

    struct Entry {
        std::string_view name;
        InklingLoopTensorKind kind;
    };
    static constexpr Entry kTable[] = {
        {"attn.attn_sink_proj.weight", InklingLoopTensorKind::AttnSink},
        {"attn.gate_attn.weight", InklingLoopTensorKind::AttnGate},
        {"attn.k_sconv.weight", InklingLoopTensorKind::KSConv},
        {"attn.k_sconv.static_w", InklingLoopTensorKind::KSConvStatic},
        {"attn.k_sconv.weight_proj", InklingLoopTensorKind::KSConvProj},
        {"attn.v_sconv.weight", InklingLoopTensorKind::VSConv},
        {"attn.v_sconv.static_w", InklingLoopTensorKind::VSConvStatic},
        {"attn.v_sconv.weight_proj", InklingLoopTensorKind::VSConvProj},
        {"attn.q_l2_clip.scale", InklingLoopTensorKind::QClipScale},
        {"attn.rel_l2_clip.scale", InklingLoopTensorKind::RelClipScale},
        {"attn.rel_logits_proj.proj", InklingLoopTensorKind::RelProj},
        {"attn.wk_dv.weight", InklingLoopTensorKind::Wk},
        {"attn.wv_dv.weight", InklingLoopTensorKind::Wv},
        {"attn.wo_ud.weight", InklingLoopTensorKind::Wo},
        {"attn.wq_du.weight", InklingLoopTensorKind::Wq},
        {"attn.wr_du.weight", InklingLoopTensorKind::Wr},
        {"attn_norm.weight", InklingLoopTensorKind::AttnNorm},
        {"attn_sconv.weight", InklingLoopTensorKind::AttnSConv},
        {"k_input_norm.weight", InklingLoopTensorKind::KInputNorm},
        {"q_input_norm.weight", InklingLoopTensorKind::QInputNorm},
        {"v_input_norm.weight", InklingLoopTensorKind::VInputNorm},
        {"mlp.experts.w13_weight", InklingLoopTensorKind::ExpertW13},
        {"mlp.experts.w13_weight_scale", InklingLoopTensorKind::ExpertW13Scale},
        {"mlp.experts.w13_weight_amax", InklingLoopTensorKind::ExpertW13Amax},
        {"mlp.experts.w13_weight_outer_scale_denominator",
         InklingLoopTensorKind::ExpertW13Denom},
        {"mlp.experts.w2_weight", InklingLoopTensorKind::ExpertW2},
        {"mlp.experts.w2_weight_scale", InklingLoopTensorKind::ExpertW2Scale},
        {"mlp.experts.w2_weight_amax", InklingLoopTensorKind::ExpertW2Amax},
        {"mlp.experts.w2_weight_outer_scale_denominator",
         InklingLoopTensorKind::ExpertW2Denom},
        {"mlp.gate.bias", InklingLoopTensorKind::RouterBias},
        {"mlp.gate.weight", InklingLoopTensorKind::RouterGate},
        {"mlp_norm.weight", InklingLoopTensorKind::MlpNorm},
        {"mlp_sconv.weight", InklingLoopTensorKind::MlpSConv},
    };
    for (const auto &e : kTable)
        if (tail == e.name) {
            p.kind = e.kind;
            return p;
        }
    return p;
}

arrow::Status ConvertInklingLoopModel(const std::string &input_dir,
                                      const std::string &output_root)
{
    return ConvertInklingLoopModelWithSpec(input_dir, output_root,
                                           MiniConvertSpec());
}

InklingLoopConvertTopology ParseInklingLoopConfig(const std::string &input_dir)
{
    return ParseInklingLoopConfig(input_dir, MiniConvertSpec());
}

arrow::Status ConvertInklingLoopModelWithSpec(const std::string &input_dir,
                                              const std::string &output_root,
                                              const InklingLoopConvertSpec &spec)
{
    InklingLoopConvertTopology top = ParseInklingLoopConfig(input_dir, spec);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("Inkling loop config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > n_routed_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, spec.frozen, spec, why))
            return arrow::Status::Invalid("Inkling loop frozen geometry mismatch: ",
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
            auto p = ClassifyInklingLoopTensor(t.name);
            switch (p.kind) {
            case InklingLoopTensorKind::Vision:
                ++top.skipped_visual_tensors;
                continue;
            case InklingLoopTensorKind::Unknown:
                return arrow::Status::Invalid(
                    "unrecognized tensor in the inkling2_loop checkpoint: ", t.name);
            default:
                break;
            }
            if ((p.kind == InklingLoopTensorKind::AttnQuery ||
                 p.kind == InklingLoopTensorKind::MlpQuery) &&
                (p.layer < 0 || p.layer >= top.num_layers))
                return arrow::Status::Invalid("attn_res query layer out of range: ",
                                              t.name);
            if (p.layer >= top.num_layers && p.layer >= 0)
                return arrow::Status::Invalid("layer index out of range: ",
                                              t.name);
            if ((p.kind == InklingLoopTensorKind::ControllerDown ||
                 p.kind == InklingLoopTensorKind::ControllerForget ||
                 p.kind == InklingLoopTensorKind::ControllerGain) &&
                (p.layer < top.loop_start || p.layer >= top.loop_end))
                return arrow::Status::Invalid(
                    "loop controller outside the loop core: ", t.name);
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

    const fs::path base = fs::path(output_root);
    const fs::path pinned_dir = base / "pinned";
    const fs::path experts_dir = base / "experts";
    fs::create_directories(pinned_dir);
    fs::create_directories(experts_dir);

    std::vector<TensorRecord> global_raw;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    int pinned_count = 0;

    // ── dense / raw tensors (NOTHING on the dense side is quantized) ───────
    for (const auto &[name, st] : weights) {
        if (st.info == nullptr) continue;
        const auto p = ClassifyInklingLoopTensor(name);
        if (IsExpertKind(p.kind)) continue;
        // The cross-pass queries are merged below, not copied under their
        // source names.
        if (p.kind == InklingLoopTensorKind::CrossPassQueries ||
            p.kind == InklingLoopTensorKind::CrossPassKvQueries)
            continue;
        auto data = ReadTensorData(input_dir, *st.shard, *st.info);
        if (data.empty() ||
            static_cast<int64_t>(data.size()) != st.info->byte_size())
            return arrow::Status::IOError("failed to read tensor data: ", name);

        // The source indexes the attention mixtures by `layer-1` (layer 0 has
        // no attention mixture) while the MLP mixtures are indexed by layer.
        // The artifact stores both under their owning layer so the loader
        // cannot repeat the off-by-one.
        std::string out_name = name;
        if (p.kind == InklingLoopTensorKind::AttnQuery)
            out_name = absl::StrFormat("model.llm.attn_res.attn_queries.%d",
                                       p.layer);

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
    }

    // ── cross-pass message queries: merge [q,mlp] + [k,v] -> [q,k,v,mlp] ───
    {
        const std::string qk = "model.llm.middle_cycle_cross_pass_queries";
        const std::string kvk = "model.llm.middle_cycle_cross_pass_kv_queries";
        if (weights.find(qk) == weights.end() || weights.find(kvk) == weights.end())
            return arrow::Status::Invalid(
                "missing middle_cycle_cross_pass query tensors");
        const auto &q = weights.at(qk);
        const auto &kv = weights.at(kvk);
        const int loop_layers = top.loop_end - top.loop_start;
        const int64_t want = static_cast<int64_t>(2 * loop_layers) * top.hidden_size;
        if (q.info->numel() != want || kv.info->numel() != want ||
            q.info->dtype != "F32" || kv.info->dtype != "F32")
            return arrow::Status::Invalid(
                "cross-pass query tensors must be F32 [2*loop_layers, hidden]");
        auto qd = ReadTensorData(input_dir, *q.shard, *q.info);
        auto kvd = ReadTensorData(input_dir, *kv.shard, *kv.info);
        if (qd.empty() || kvd.empty())
            return arrow::Status::IOError("failed to read cross-pass queries");
        const size_t H = static_cast<size_t>(top.hidden_size);
        TensorRecord r;
        r.name = "model.llm.message_queries";
        r.dtype = "F32";
        r.shape = {loop_layers, 4, top.hidden_size};
        r.data.resize(static_cast<size_t>(loop_layers) * 4 * H * 4);
        const uint8_t *qs = qd.data();
        const uint8_t *kvs = kvd.data();
        uint8_t *dst = r.data.data();
        const size_t row_bytes = H * 4;
        for (int i = 0; i < loop_layers; ++i) {
            // [q, k, v, mlp]: q=queries[2i], k=kv[2i], v=kv[2i+1],
            // mlp=queries[2i+1] (Inkling2LoopMergeMessageQueries).
            uint8_t *out_row = dst + static_cast<size_t>(i) * 4 * row_bytes;
            std::memcpy(out_row, qs + static_cast<size_t>(2 * i) * row_bytes,
                        row_bytes);
            std::memcpy(out_row + row_bytes,
                        kvs + static_cast<size_t>(2 * i) * row_bytes, row_bytes);
            std::memcpy(out_row + 2 * row_bytes,
                        kvs + static_cast<size_t>(2 * i + 1) * row_bytes,
                        row_bytes);
            std::memcpy(out_row + 3 * row_bytes,
                        qs + static_cast<size_t>(2 * i + 1) * row_bytes,
                        row_bytes);
        }
        global_raw.push_back(std::move(r));
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
        // ROW-INTERLEAVED fused slab: source rows alternate gate, up. The
        // release's conversion de-interleaves weights AND scales before the
        // gate/up split; splitting in halves keeps every byte but destroys the
        // model's knowledge (see the header comment).
        auto deinterleave = [](const uint8_t *src, std::size_t rows,
                               std::size_t row_bytes, std::vector<uint8_t> &even,
                               std::vector<uint8_t> &odd_rows) {
            even.resize(rows * row_bytes);
            odd_rows.resize(rows * row_bytes);
            for (std::size_t r = 0; r < rows; ++r) {
                std::memcpy(even.data() + r * row_bytes, src + 2 * r * row_bytes,
                            row_bytes);
                std::memcpy(odd_rows.data() + r * row_bytes,
                            src + (2 * r + 1) * row_bytes, row_bytes);
            }
        };
        for (int e = 0; e < E; ++e) {
            PackedExpert pe;
            if (nvfp4) {
                const uint8_t *w13p = w13_data.data() + w13_expert * e;
                const uint8_t *w2p = w2_data.data() + w2_expert * e;
                const uint8_t *s13p = s13_data.data() + w13_scale_expert * e;
                const uint8_t *s2p = s2_data.data() + w2_scale_expert * e;
                deinterleave(w13p, static_cast<std::size_t>(I), w13_row,
                             pe.w1_packed, pe.w3_packed);
                deinterleave(s13p, static_cast<std::size_t>(I), w13_scale_row,
                             pe.w1_scales, pe.w3_scales);
                pe.w2_packed.assign(w2p, w2p + w2_expert);
                pe.w2_scales.assign(s2p, s2p + w2_scale_expert);
                pe.w1_global_scale = amax13[static_cast<size_t>(e)] / denom13;
                pe.w3_global_scale = pe.w1_global_scale;
                pe.w2_global_scale = amax2[static_cast<size_t>(e)] / denom2;
            } else {
                // MXFP8 source: E4M3 bytes + ue8m0 per-32 scales pass through
                // VERBATIM (requantizing to 4 bits costs a measured 16-17%
                // relative L2 on Small's affected layers). Each ue8m0 byte is
                // DUPLICATED (per-32 -> per-16) so the record satisfies the
                // neutral ExpertSchema's per-16 scale count.
                const size_t s13_row = static_cast<size_t>(H / 32);
                const uint8_t *w13p = w13_data.data() + w13_expert * e;
                const uint8_t *w2p = w2_data.data() + w2_expert * e;
                const uint8_t *s13p = s13_data.data() + w13_scale_expert * e;
                const uint8_t *s2p = s2_data.data() + w2_scale_expert * e;
                auto dup_scales = [](const uint8_t *src, int64_t rows,
                                     int64_t row_scales,
                                     std::vector<uint8_t> &dst) {
                    dst.resize(static_cast<size_t>(rows * row_scales * 2));
                    for (int64_t r = 0; r < rows; ++r)
                        for (int64_t i = 0; i < row_scales; ++i) {
                            const uint8_t b = src[r * row_scales + i];
                            dst[static_cast<size_t>(2 * (r * row_scales + i))] = b;
                            dst[static_cast<size_t>(2 * (r * row_scales + i) + 1)] = b;
                        }
                };
                deinterleave(w13p, static_cast<std::size_t>(I),
                             static_cast<std::size_t>(H), pe.w1_packed,
                             pe.w3_packed);
                {
                    std::vector<uint8_t> s_even, s_odd;
                    deinterleave(s13p, static_cast<std::size_t>(I), s13_row, s_even,
                                 s_odd);
                    dup_scales(s_even.data(), I, s13_row, pe.w1_scales);
                    dup_scales(s_odd.data(), I, s13_row, pe.w3_scales);
                }
                pe.w2_packed.assign(w2p, w2p + w2_expert);
                dup_scales(s2p, H, static_cast<size_t>(I / 32), pe.w2_scales);
                pe.w1_global_scale = 1.0f;
                pe.w3_global_scale = 1.0f;
                pe.w2_global_scale = 1.0f;
                // The passthrough keeps E4M3 BYTES, so the record must declare
                // mxfp8: without this the verifier assumed NVFP4 from the
                // nine-column shape and the artifact's mask named a codec its
                // bytes do not carry (Task 9c / D1).
                pe.dtype = "F8_E4M3";
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
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                       pinned_dir / "global.arrow"));
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
        // alternate stop id and the generation-config pad id.
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
    policy.is_nvfp4 = [](const std::string &) { return false; };
    policy.mem_class = [](const std::string &name) {
        switch (ClassifyInklingLoopTensor(name).kind) {
        case InklingLoopTensorKind::ExpertW13:
        case InklingLoopTensorKind::ExpertW13Scale:
        case InklingLoopTensorKind::ExpertW13Amax:
        case InklingLoopTensorKind::ExpertW13Denom:
        case InklingLoopTensorKind::ExpertW2:
        case InklingLoopTensorKind::ExpertW2Scale:
        case InklingLoopTensorKind::ExpertW2Amax:
        case InklingLoopTensorKind::ExpertW2Denom:
            return VramTensorClass::RoutedExpert;
        default:
            return VramTensorClass::Resident;
        }
    };
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            auto p = ClassifyInklingLoopTensor(t.name);
            if (p.kind == InklingLoopTensorKind::Vision) continue;
            VramTensorInfo ti;
            ti.name = t.name;
            ti.numel = t.numel();
            ti.raw_bytes = t.byte_size();
            vram_tensors.push_back(std::move(ti));
        }
    VramGeometry geom;
    geom.hidden_dim = top.hidden_size;
    geom.kv_dim = top.num_key_value_heads * top.head_dim;
    // Every pass owns its own KV slots: the slab sizes for the MAXIMUM count
    // this row can run (72 for a varloop tree, 48 for the fixed-pass one), and a
    // smaller active count addresses a prefix of it.
    geom.num_attn_layers = top.num_layers + (top.loop_passes_max - 1) *
                                                (top.loop_end - top.loop_start);
    geom.num_moe_layers = top.num_layers;
    geom.experts_per_layer = top.num_experts;
    geom.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    geom.nvfp4_scale_block = frozen::kQuantBlock;
    const VramProfile vram = EstimateVram(vram_tensors, policy, geom);

    // ── index ─────────────────────────────────────────────────────────────
    {
        // 17 significant digits BEFORE the numbers are formatted (the clip
        // threshold 22.627416997969522 must not round to 22.6274).
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
        j << "{\n  \"layout\": 2,\n";
        // The CARD name: gpu::detect_model_type normalizes it into the factory
        // key, so it must normalize to exactly this row's model_type.
        j << "  \"model\": \"" << spec.model_name << "\",\n";
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
        j << "  \"loop_start\": " << top.loop_start << ",\n";
        j << "  \"loop_end\": " << top.loop_end << ",\n";
        j << "  \"loop_passes\": " << top.loop_passes << ",\n";
        // The pass ceiling and the capability flag. A fixed-pass tree records
        // its own count as the ceiling, so the loader's slab arithmetic has one
        // shape for both rows.
        j << "  \"loop_passes_max\": " << top.loop_passes_max << ",\n";
        j << "  \"varloop\": " << (top.varloop ? "true" : "false") << ",\n";
        j << "  \"loop_gate_rank\": " << top.loop_gate_rank << ",\n";
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
        {
            std::vector<bool> fp8(top.expert_nvfp4.size(), false);
            for (size_t i = 0; i < fp8.size(); ++i) fp8[i] = !top.expert_nvfp4[i];
            bool_array("expert_fp8", fp8, true);
        }
        j << FormatVramJson(vram) << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    lsxcommon::log::info(absl::StrFormat(
        "Inkling loop conversion: %d layers (%zu full / %zu sliding), loop "
        "[%d,%d) x %d passes, %d experts x %d layers, %lld pinned tensors, "
        "%d expert files, stripped %d vision tensors, tiktoken %s",
        top.num_layers, top.full_layers.size(), top.sliding_layers.size(),
        top.loop_start, top.loop_end, top.loop_passes, top.num_experts,
        top.num_layers, static_cast<long long>(pinned_count), experts_written,
        top.skipped_visual_tensors, tokenizer_ok ? "built" : "absent"));
    return arrow::Status::OK();
}

}  // namespace lsxcommon::inkling