// GigaChat4-37B-A4B converter (Model Ingestion Protocol stage 1/2).
// See gigachat_moe_37b_constants.h for the frozen geometry and the design doc
// docs/superpowers/specs/2026-11-26-gigachat4-37b-a4b-ingest-design.md.
//
// Mixed-precision source: FP8 E4M3 block-128 tensors (attention/expert/GDN
// projections, indexer wq_b) are dequantized on the GPU and re-quantized to the
// neutral NVFP4 contour; the BF16 linears (attention gate, shared experts,
// dense MLP, indexer wk/weights_proj) are quantized to NVFP4 directly; norms,
// routers, biases and the GDN scalars stay at source precision.
#include "lsxcommon/gigachat_moe_37b_convert.h"

#include "gigachat_moe_37b_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/gigachat_moe_convert.h"  // DequantFp8Block128ToBf16Gpu
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "lsxcommon/vram_estimate.h"

#include <simdjson.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>

namespace lsxcommon::gigachat4 {
namespace {

bool EndsWith(std::string_view s, std::string_view suf)
{
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Strips a trailing ".weight_scale_inv" or ".weight" to expose the module path.
std::string_view BaseModule(std::string_view name)
{
    for (std::string_view suf : {std::string_view(".weight_scale_inv"),
                                 std::string_view(".weight")})
        if (EndsWith(name, suf))
            return name.substr(0, name.size() - suf.size());
    return name;
}

std::string_view LastComponent(std::string_view module)
{
    const size_t dot = module.rfind('.');
    return dot == std::string_view::npos ? module : module.substr(dot + 1);
}

bool ParseLayerAndRest(std::string_view name, int &layer,
                       std::string_view &rest)
{
    constexpr std::string_view kPrefix = "model.layers.";
    if (name.rfind(kPrefix, 0) != 0) return false;
    std::string_view tail = name.substr(kPrefix.size());
    size_t i = 0;
    while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
    if (i == 0 || i >= tail.size() || tail[i] != '.') return false;
    layer = 0;
    for (size_t j = 0; j < i; ++j) layer = layer * 10 + (tail[j] - '0');
    rest = tail.substr(i + 1);
    return true;
}

Gc4TensorKind ClassifyModule(std::string_view module, int &expert)
{
    expert = -1;
    const std::string_view last = LastComponent(module);

    // Layer norms (GatedNorm): weight / gate projections.
    if (last == "input_layernorm" || last == "post_attention_layernorm" ||
        last == "post_self_attn_layernorm" ||
        last == "post_feedforward_layernorm" ||
        last == "q_a_layernorm" || last == "kv_a_layernorm")
        return Gc4TensorKind::LayerNormWeight;
    if (last == "gate_up_projection")
        return Gc4TensorKind::LayerNormGateUp;
    if (last == "gate_down_projection")
        return Gc4TensorKind::LayerNormGateDown;

    // MLA attention.
    if (module == "self_attn.q_a_proj") return Gc4TensorKind::QaProj;
    if (module == "self_attn.q_b_proj") return Gc4TensorKind::QbProj;
    if (module == "self_attn.kv_a_proj_with_mqa") return Gc4TensorKind::KvaProj;
    if (module == "self_attn.kv_b_proj") return Gc4TensorKind::KvbProj;
    if (module == "self_attn.o_proj") return Gc4TensorKind::OProj;
    if (module == "self_attn.gate_proj") return Gc4TensorKind::AttnGateProj;

    // DSA lightning indexer.
    if (module == "self_attn.indexer.wq_b") return Gc4TensorKind::IndexWqB;
    if (module == "self_attn.indexer.wk") return Gc4TensorKind::IndexWk;
    if (module == "self_attn.indexer.weights_proj")
        return Gc4TensorKind::IndexWeightsProj;
    if (module == "self_attn.indexer.k_norm")
        return Gc4TensorKind::IndexKNorm;

    // GatedDeltaNet.
    if (module == "self_attn.in_proj_qkvz") return Gc4TensorKind::GdnInProjQkvz;
    if (module == "self_attn.out_proj") return Gc4TensorKind::GdnOutProj;
    if (module == "self_attn.conv1d") return Gc4TensorKind::GdnConv1d;
    if (module == "self_attn.A_log") return Gc4TensorKind::GdnALog;
    if (module == "self_attn.dt_bias") return Gc4TensorKind::GdnDtBias;
    if (module == "self_attn.in_proj_ba") return Gc4TensorKind::GdnInProjBa;
    if (module == "self_attn.norm") return Gc4TensorKind::GdnNormWeight;

    // Router (BF16 gate + BF16 correction bias).
    if (module == "mlp.gate") return Gc4TensorKind::RouterWeight;
    if (module == "mlp.gate.e_score_correction_bias")
        return Gc4TensorKind::RouterBias;

    // Routed experts: mlp.experts.<E>.<proj>
    constexpr std::string_view kExperts = "mlp.experts.";
    if (module.rfind(kExperts, 0) == 0)
    {
        std::string_view tail = module.substr(kExperts.size());
        size_t i = 0;
        while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
        if (i == 0 || i >= tail.size() || tail[i] != '.')
            return Gc4TensorKind::Unknown;
        expert = 0;
        for (size_t j = 0; j < i; ++j) expert = expert * 10 + (tail[j] - '0');
        const std::string_view proj = tail.substr(i + 1);
        if (proj == "gate_proj") return Gc4TensorKind::ExpertGate;
        if (proj == "up_proj") return Gc4TensorKind::ExpertUp;
        if (proj == "down_proj") return Gc4TensorKind::ExpertDown;
        return Gc4TensorKind::Unknown;
    }

    // Shared experts.
    if (module == "mlp.shared_experts.gate_proj") return Gc4TensorKind::SharedGate;
    if (module == "mlp.shared_experts.up_proj") return Gc4TensorKind::SharedUp;
    if (module == "mlp.shared_experts.down_proj") return Gc4TensorKind::SharedDown;

    // Dense MLP (layer 0).
    if (module == "mlp.gate_proj") return Gc4TensorKind::MlpGate;
    if (module == "mlp.up_proj") return Gc4TensorKind::MlpUp;
    if (module == "mlp.down_proj") return Gc4TensorKind::MlpDown;

    return Gc4TensorKind::Unknown;
}

} // namespace

ParsedGc4Tensor classify_gigachat4_tensor(const std::string &name)
{
    return classify_gigachat4_tensor(name, frozen::kLayers);
}

ParsedGc4Tensor classify_gigachat4_tensor(const std::string &name,
                                         int num_layers)
{
    ParsedGc4Tensor out;
    const std::string_view n(name);

    if (n == "model.embed_tokens.weight")
    {
        out.kind = Gc4TensorKind::Embed;
        return out;
    }
    if (n == "lm_head.weight")
    {
        out.kind = Gc4TensorKind::LmHead;
        return out;
    }
    if (n.rfind("model.norm", 0) == 0)
    {
        const std::string_view module = BaseModule(n);
        if (module == "model.norm")
            out.kind = Gc4TensorKind::FinalNormWeight;
        else if (module == "model.norm.gate_up_projection")
            out.kind = Gc4TensorKind::FinalNormGateUp;
        else if (module == "model.norm.gate_down_projection")
            out.kind = Gc4TensorKind::FinalNormGateDown;
        return out;
    }

    int layer = -1;
    std::string_view rest;
    if (!ParseLayerAndRest(n, layer, rest))
        return out;
    out.layer = layer;

    // MTP prediction layers (>= num_hidden_layers): converted + stored, never
    // applied. GigaChat4 has exactly one (kMtpLayer).
    if (num_layers > 0 && layer >= num_layers)
    {
        out.kind = Gc4TensorKind::Mtp;
        return out;
    }

    out.kind = ClassifyModule(BaseModule(rest), out.expert);
    return out;
}

bool gigachat4_tensor_is_nvfp4(Gc4TensorKind kind)
{
    switch (kind)
    {
    case Gc4TensorKind::QaProj:
    case Gc4TensorKind::QbProj:
    case Gc4TensorKind::KvaProj:
    case Gc4TensorKind::KvbProj:
    case Gc4TensorKind::OProj:
    case Gc4TensorKind::AttnGateProj:
    case Gc4TensorKind::IndexWqB:
    case Gc4TensorKind::MlpGate:
    case Gc4TensorKind::MlpUp:
    case Gc4TensorKind::MlpDown:
    case Gc4TensorKind::GdnInProjQkvz:
    case Gc4TensorKind::GdnOutProj:
    case Gc4TensorKind::SharedGate:
    case Gc4TensorKind::SharedUp:
    case Gc4TensorKind::SharedDown:
    case Gc4TensorKind::ExpertGate:
    case Gc4TensorKind::ExpertUp:
    case Gc4TensorKind::ExpertDown:
        return true;
    default:
        return false;
    }
}

bool gigachat4_tensor_is_expert(Gc4TensorKind kind)
{
    return kind == Gc4TensorKind::ExpertGate ||
           kind == Gc4TensorKind::ExpertUp ||
           kind == Gc4TensorKind::ExpertDown;
}

Gc4Topology ParseGigaChat4Config(const std::string &config_json_text)
{
    return ParseGigaChat4Config(config_json_text, Gc4DefaultIngestSpec());
}

Gc4IngestSpec Gc4DefaultIngestSpec()
{
    Gc4IngestSpec s;
    s.format = frozen::kFormat;
    s.model_name = frozen::kModelName;
    s.frozen = Gc4FrozenGeometry{
        /*hidden=*/frozen::kHidden,
        /*layers=*/frozen::kLayers,
        /*vocab=*/frozen::kVocab,
        /*heads=*/frozen::kHeads,
        /*kv_heads=*/frozen::kKVHeads,
        /*num_experts=*/frozen::kNumRouted,
        /*top_k=*/frozen::kTopK,
        /*moe_intermediate=*/frozen::kMoeIntermediate,
        /*dense_intermediate=*/frozen::kDenseIntermediate,
        /*first_k_dense=*/frozen::kFirstKDense,
        /*q_lora=*/frozen::kQLoraRank,
        /*kv_lora=*/frozen::kKvLoraRank,
        /*qk_nope=*/frozen::kQkNopeHeadDim,
        /*qk_rope=*/frozen::kQkRopeHeadDim,
        /*v_head=*/frozen::kVHeadDim,
        /*index_heads=*/frozen::kIndexHeads,
        /*index_head_dim=*/frozen::kIndexHeadDim,
        /*index_topk=*/frozen::kIndexTopk,
        /*rms_eps=*/static_cast<double>(frozen::kRmsEps),
        /*routed_scale=*/static_cast<double>(frozen::kRoutedScale),
        /*swiglu_limit=*/static_cast<double>(frozen::kSwigluLimit),
        /*logit_softcap=*/static_cast<double>(frozen::kLogitSoftcap),
        /*bos_id=*/frozen::kBosId,
        /*eos_id=*/frozen::kEosId,
        /*pad_id=*/frozen::kPadId};
    s.full_attention_layers.assign(frozen::kFullAttnLayers,
                                   frozen::kFullAttnLayers + frozen::kNumFullAttn);
    s.attn_scale = frozen::kAttnScale;
    return s;
}

Gc4Topology ParseGigaChat4Config(const std::string &config_json_text,
                                 const Gc4IngestSpec &spec)
{
    Gc4Topology t;
    // The member's own geometry is the parse default; config.json is
    // authoritative wherever the field is present.
    t.hidden = spec.frozen.hidden;
    t.layers = spec.frozen.layers;
    t.vocab = spec.frozen.vocab;
    t.heads = spec.frozen.heads;
    t.kv_heads = spec.frozen.kv_heads;
    t.num_experts = spec.frozen.num_experts;
    t.top_k = spec.frozen.top_k;
    t.moe_intermediate = spec.frozen.moe_intermediate;
    t.dense_intermediate = spec.frozen.dense_intermediate;
    t.first_k_dense = spec.frozen.first_k_dense;
    t.q_lora = spec.frozen.q_lora;
    t.kv_lora = spec.frozen.kv_lora;
    t.qk_nope = spec.frozen.qk_nope;
    t.qk_rope = spec.frozen.qk_rope;
    t.v_head = spec.frozen.v_head;
    t.index_heads = spec.frozen.index_heads;
    t.index_head_dim = spec.frozen.index_head_dim;
    t.index_topk = spec.frozen.index_topk;
    t.rms_eps = spec.frozen.rms_eps;
    t.routed_scale = spec.frozen.routed_scale;
    t.swiglu_limit = spec.frozen.swiglu_limit;
    t.logit_softcap = spec.frozen.logit_softcap;
    t.bos_id = spec.frozen.bos_id;
    t.eos_id = spec.frozen.eos_id;
    t.pad_id = spec.frozen.pad_id;

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(config_json_text.data(), config_json_text.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
    {
        t.error_msg = "cannot parse config.json";
        return t;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
    {
        t.error_msg = "config.json root is not an object";
        return t;
    }

    auto as_num = [](simdjson::ondemand::value v, double &out) -> bool {
        if (v.get_double().get(out) == simdjson::SUCCESS) return true;
        int64_t i = 0;
        if (v.get_int64().get(i) == simdjson::SUCCESS)
        {
            out = static_cast<double>(i);
            return true;
        }
        return false;
    };

    bool have_hidden = false, have_layers = false, have_vocab = false;
    for (auto field : root)
    {
        std::string_view key;
        if (field.unescaped_key().get(key) != simdjson::SUCCESS) continue;
        if (key == "full_attention_layers")
        {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto elem : arr)
                {
                    int64_t v = 0;
                    if (elem.get_int64().get(v) == simdjson::SUCCESS)
                        t.full_attention_layers.push_back(static_cast<int>(v));
                }
            continue;
        }
        double d = 0.0;
        if (!as_num(field.value(), d)) continue;
        const int di = static_cast<int>(d);
        if (key == "hidden_size") { t.hidden = di; have_hidden = true; }
        else if (key == "num_hidden_layers") { t.layers = di; have_layers = true; }
        else if (key == "vocab_size") { t.vocab = di; have_vocab = true; }
        else if (key == "num_attention_heads") t.heads = di;
        else if (key == "num_key_value_heads") t.kv_heads = di;
        else if (key == "n_routed_experts") t.num_experts = di;
        else if (key == "num_experts_per_tok") t.top_k = di;
        else if (key == "moe_intermediate_size") t.moe_intermediate = di;
        else if (key == "intermediate_size") t.dense_intermediate = di;
        else if (key == "first_k_dense_replace") t.first_k_dense = di;
        else if (key == "q_lora_rank") t.q_lora = di;
        else if (key == "kv_lora_rank") t.kv_lora = di;
        else if (key == "qk_nope_head_dim") t.qk_nope = di;
        else if (key == "qk_rope_head_dim") t.qk_rope = di;
        else if (key == "v_head_dim") t.v_head = di;
        else if (key == "index_n_heads") t.index_heads = di;
        else if (key == "index_head_dim") t.index_head_dim = di;
        else if (key == "index_topk") t.index_topk = di;
        else if (key == "rms_norm_eps") t.rms_eps = d;
        else if (key == "routed_scaling_factor") t.routed_scale = d;
        else if (key == "swiglu_limit") t.swiglu_limit = d;
        else if (key == "final_logit_softcapping") t.logit_softcap = d;
        else if (key == "bos_token_id") t.bos_id = di;
        else if (key == "eos_token_id") t.eos_id = di;
        else if (key == "pad_token_id") t.pad_id = di;
    }

    if (!have_hidden || !have_layers || !have_vocab)
    {
        t.error_msg = "config.json missing required geometry "
                      "(hidden_size/num_hidden_layers/vocab_size)";
        return t;
    }
    if (t.full_attention_layers.empty())
        t.error_msg = "config.json has no full_attention_layers";
    return t;
}

namespace {

struct TensorRef {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

// FP8 E4M3 + per-128x128 scale tiles -> BF16 -> NVFP4.
arrow::Status QuantizeFp8(const std::string &input_dir, const TensorRef &w,
                          const TensorRef &s, DenseNvfp4Record &out)
{
    auto fp8 = ReadTensorData(input_dir, *w.shard, *w.info);
    auto sc = ReadTensorData(input_dir, *s.shard, *s.info);
    if (fp8.empty() ||
        static_cast<int64_t>(fp8.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      w.info->name);
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(gigachat::DequantFp8Block128ToBf16Gpu(
        fp8, w.info->shape, sc, s.info->shape, bf16));
    out.name = w.info->name;
    out.shape = w.info->shape;
    out.dtype = "BF16";
    if (!gpu::quantize_tensor_gpu(bf16.data(), w.info->numel(), "BF16", nullptr,
                                  w.info->shape, {}, out.packed, out.scales,
                                  out.global_scale))
        return arrow::Status::IOError("GPU quantization failed for ",
                                      w.info->name);
    return arrow::Status::OK();
}

// A BF16 linear matrix (no scale tiles in the checkpoint) -> NVFP4.
arrow::Status QuantizeBf16(const std::string &input_dir, const TensorRef &w,
                           DenseNvfp4Record &out)
{
    if (w.info->dtype != "BF16")
        return arrow::Status::Invalid(
            "gigachat4: linear tensor is neither FP8 nor BF16: ", w.info->name,
            " (dtype ", w.info->dtype, ")");
    auto raw = ReadTensorData(input_dir, *w.shard, *w.info);
    if (raw.empty() || static_cast<int64_t>(raw.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      w.info->name);
    out.name = w.info->name;
    out.shape = w.info->shape;
    out.dtype = "BF16";
    if (!gpu::quantize_tensor_gpu(raw.data(), w.info->numel(), "BF16", nullptr,
                                  w.info->shape, {}, out.packed, out.scales,
                                  out.global_scale))
        return arrow::Status::IOError("GPU quantization failed for ",
                                      w.info->name);
    return arrow::Status::OK();
}

std::string LayerFile(int layer, const char *suffix)
{
    return absl::StrFormat("layer_%03d%s", layer, suffix);
}

arrow::Status WriteBatchOr(const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &b,
                           const std::filesystem::path &path)
{
    if (!b.ok()) return b.status();
    return WriteRecordBatchToFile(*b, path);
}

// EstimateVram takes plain function pointers (NVCC-safe header), so the depth
// the classifier must use travels through this file-scope slot instead of a
// lambda capture. A non-zero value is the depth of the member BEING converted;
// zero falls back to this TU's own passport (the one-argument overload).
int g_classify_depth = 0;

struct ScopedClassifyDepth {
    explicit ScopedClassifyDepth(int depth) : saved(g_classify_depth)
    {
        g_classify_depth = depth;
    }
    ~ScopedClassifyDepth() { g_classify_depth = saved; }
    ScopedClassifyDepth(const ScopedClassifyDepth &) = delete;
    ScopedClassifyDepth &operator=(const ScopedClassifyDepth &) = delete;
    int saved;
};

ParsedGc4Tensor ClassifyAtDepth(const std::string &name)
{
    return g_classify_depth > 0 ? classify_gigachat4_tensor(name, g_classify_depth)
                                : classify_gigachat4_tensor(name);
}

bool VramIsNvfp4(const std::string &name)
{
    return gigachat4_tensor_is_nvfp4(ClassifyAtDepth(name).kind);
}

VramTensorClass VramMemClass(const std::string &name)
{
    return gigachat4_tensor_is_expert(ClassifyAtDepth(name).kind)
               ? VramTensorClass::RoutedExpert
               : VramTensorClass::Resident;
}

} // namespace

arrow::Status ConvertGigaChat4Model(const std::string &input_dir,
                                    const std::string &output_root)
{
    return ConvertGigaChat4ModelWithSpec(input_dir, output_root,
                                         Gc4DefaultIngestSpec());
}

arrow::Status ConvertGigaChat4ModelWithSpec(const std::string &input_dir,
                                            const std::string &output_root,
                                            const Gc4IngestSpec &spec)
{
    namespace fs = std::filesystem;
    const Gc4FrozenGeometry &frz = spec.frozen;

    std::ifstream cf(fs::path(input_dir) / "config.json");
    if (!cf) return arrow::Status::Invalid("config.json not found in ", input_dir);
    std::stringstream css;
    css << cf.rdbuf();
    const Gc4Topology top = ParseGigaChat4Config(css.str(), spec);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("GigaChat4 config error: ", top.error_msg);

    if (!SyntheticGeometryAllowed())
    {
        const bool ok = top.hidden == frz.hidden &&
                        top.layers == frz.layers &&
                        top.vocab == frz.vocab &&
                        top.heads == frz.heads &&
                        top.kv_heads == frz.kv_heads &&
                        top.num_experts == frz.num_experts &&
                        top.top_k == frz.top_k &&
                        top.moe_intermediate == frz.moe_intermediate &&
                        top.dense_intermediate == frz.dense_intermediate &&
                        top.first_k_dense == frz.first_k_dense &&
                        top.q_lora == frz.q_lora &&
                        top.kv_lora == frz.kv_lora &&
                        top.qk_nope == frz.qk_nope &&
                        top.qk_rope == frz.qk_rope &&
                        top.v_head == frz.v_head;
        if (!ok)
            return arrow::Status::Invalid(
                "GigaChat4 frozen geometry mismatch (set "
                "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 only for synthetic tests)");
        if (!spec.full_attention_layers.empty())
        {
            if (top.full_attention_layers.size() !=
                spec.full_attention_layers.size())
                return arrow::Status::Invalid(
                    "GigaChat4 frozen full_attention_layers count mismatch");
            for (size_t i = 0; i < spec.full_attention_layers.size(); ++i)
                if (top.full_attention_layers[i] != spec.full_attention_layers[i])
                    return arrow::Status::Invalid(
                        "GigaChat4 frozen full_attention_layers mismatch at "
                        "index ",
                        static_cast<int>(i));
        }
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    absl::flat_hash_map<std::string, TensorRef> refs;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            refs.insert_or_assign(t.name, TensorRef{&shard, &t});

    int mtp_count = 0;
    std::vector<int> layer_tensor_count(top.layers, 0);
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
        {
            auto p = classify_gigachat4_tensor(t.name, top.layers);
            if (p.kind == Gc4TensorKind::Mtp)
            {
                ++mtp_count;
                continue;
            }
            if (p.kind == Gc4TensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in gigachat4 checkpoint: ", t.name);
            if (p.layer >= 0 && p.layer < top.layers) ++layer_tensor_count[p.layer];
        }
    for (int l = 0; l < top.layers; ++l)
        if (layer_tensor_count[l] == 0)
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path mtp_dir = base / "mtp";
    for (const fs::path &d : {base, pinned_dir, mtp_dir})
    {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::flat_hash_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::flat_hash_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::flat_hash_map<int, std::vector<TensorRecord>> mtp_by_layer;
    std::vector<TensorRecord> global_raw;
    int64_t total_pinned = 0;

    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            auto p = classify_gigachat4_tensor(t.name, top.layers);
            if (p.kind == Gc4TensorKind::Mtp)
            {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = ReadTensorData(input_dir, shard, t);
                mtp_by_layer[p.layer].push_back(std::move(r));
                continue;
            }
            constexpr std::string_view kScale = ".weight_scale_inv";
            if (EndsWith(t.name, kScale))
                continue;  // consumed together with its `.weight`
            if (gigachat4_tensor_is_expert(p.kind)) continue;  // expert pass

            if (gigachat4_tensor_is_nvfp4(p.kind))
            {
                DenseNvfp4Record r;
                auto it = refs.find(t.name + "_scale_inv");
                if (it != refs.end())
                    ARROW_RETURN_NOT_OK(
                        QuantizeFp8(input_dir, refs[t.name], it->second, r));
                else
                    ARROW_RETURN_NOT_OK(
                        QuantizeBf16(input_dir, refs[t.name], r));
                nvfp4_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
            }
            else
            {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = ReadTensorData(input_dir, shard, t);
                if (r.data.empty() ||
                    static_cast<int64_t>(r.data.size()) != t.byte_size())
                    return arrow::Status::IOError(
                        "failed to read tensor data: ", t.name);
                switch (p.kind)
                {
                case Gc4TensorKind::Embed:
                case Gc4TensorKind::LmHead:
                case Gc4TensorKind::FinalNormWeight:
                case Gc4TensorKind::FinalNormGateUp:
                case Gc4TensorKind::FinalNormGateDown:
                    global_raw.push_back(std::move(r));
                    break;
                default:
                    raw_by_layer[p.layer].push_back(std::move(r));
                    break;
                }
                ++total_pinned;
            }
        }
    }

    // Routed experts: 3 projections per expert, flushed LAYER BY LAYER.
    // The member's expert set is the dominant payload (386.9 GB on the 708B
    // row); keeping it resident until the end of the pass is an OOM, not a
    // slowdown, so one layer (256 experts) is the residency bound.
    struct ExpertRec {
        DenseNvfp4Record gate, up, down;
        bool has_gate = false, has_up = false, has_down = false;
    };
    struct ExpertTensorRef {
        const SafetensorsModel::Shard *shard;
        const TensorInfo *info;
        Gc4TensorKind kind;
        int expert;
    };
    std::vector<std::vector<ExpertTensorRef>> experts_by_layer(
        static_cast<size_t>(top.layers));
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
        {
            auto p = classify_gigachat4_tensor(t.name, top.layers);
            if (!gigachat4_tensor_is_expert(p.kind)) continue;
            constexpr std::string_view kScale = ".weight_scale_inv";
            if (EndsWith(t.name, kScale))
                continue;  // consumed together with its `.weight`
            if (p.layer < 0 || p.layer >= top.layers)
                return arrow::Status::Invalid("expert outside the layer range: ",
                                              t.name);
            experts_by_layer[static_cast<size_t>(p.layer)].push_back(
                ExpertTensorRef{&shard, &t, p.kind, p.expert});
        }
    for (int l = 0; l < top.layers; ++l)
        experts_by_layer[static_cast<size_t>(l)].shrink_to_fit();

    int total_experts = 0;
    for (int layer = 0; layer < top.layers; ++layer)
    {
        const auto &layer_refs = experts_by_layer[static_cast<size_t>(layer)];
        if (layer_refs.empty()) continue;
        absl::flat_hash_map<int, ExpertRec> experts;
        for (const ExpertTensorRef &ref : layer_refs)
        {
            DenseNvfp4Record r;
            auto it = refs.find(ref.info->name + "_scale_inv");
            if (it != refs.end())
                ARROW_RETURN_NOT_OK(QuantizeFp8(
                    input_dir, TensorRef{ref.shard, ref.info}, it->second, r));
            else
                ARROW_RETURN_NOT_OK(QuantizeBf16(
                    input_dir, TensorRef{ref.shard, ref.info}, r));
            auto &e = experts[ref.expert];
            if (ref.kind == Gc4TensorKind::ExpertGate)
            {
                e.gate = std::move(r);
                e.has_gate = true;
            }
            else if (ref.kind == Gc4TensorKind::ExpertUp)
            {
                e.up = std::move(r);
                e.has_up = true;
            }
            else
            {
                e.down = std::move(r);
                e.has_down = true;
            }
        }
        const fs::path dir = base / "experts" / absl::StrCat("layer_", layer);
        std::error_code ec;
        fs::create_directories(dir, ec);
        for (auto &[expert, e] : experts)
        {
            if (!e.has_gate || !e.has_up || !e.has_down)
                return arrow::Status::Invalid("incomplete expert: layer ",
                                              layer, " expert ", expert);
            PackedExpert pe;
            // Repo PackedExpert slot convention: w1 = gate, w3 = up, w2 = down.
            pe.w1_packed = e.gate.packed;
            pe.w1_scales = e.gate.scales;
            pe.w1_shape = e.gate.shape;
            pe.w1_global_scale = e.gate.global_scale;
            pe.w3_packed = e.up.packed;
            pe.w3_scales = e.up.scales;
            pe.w3_shape = e.up.shape;
            pe.w3_global_scale = e.up.global_scale;
            pe.w2_packed = e.down.packed;
            pe.w2_scales = e.down.scales;
            pe.w2_shape = e.down.shape;
            pe.w2_global_scale = e.down.global_scale;
            // Shared ExpertStore path convention (unpadded, like every family).
            ARROW_RETURN_NOT_OK(WriteBatchOr(
                ExpertSchema::MakeBatch(pe),
                dir / absl::StrCat("exp_", expert, ".arrow")));
            ++total_experts;
        }
        // Release this layer before the next one is read.
        absl::flat_hash_map<int, ExpertRec>().swap(experts);
        experts_by_layer[static_cast<size_t>(layer)].clear();
        experts_by_layer[static_cast<size_t>(layer)].shrink_to_fit();
    }


    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(global_raw),
                                         pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / LayerFile(layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(recs),
                                         pinned_dir / LayerFile(layer, ".arrow")));
    for (auto &[layer, recs] : mtp_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(recs),
                                         mtp_dir / LayerFile(layer, ".arrow")));

    // Model Ingestion Protocol stage 4 (tokenizer bridge). The checkpoint is a
    // byte-level BPE with the Llama-3 case-run pre-tokenizer, i.e. engine
    // tokenizer_kind 3. The builder resolves `<s>`/`</s>` from
    // tokenizer_config.json; the config's bos/eos/pad are the authoritative
    // override.
    bool tokenizer_ok = false;
    {
        const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
        if (fs::exists(tok_path))
        {
            TokenizerSpecialPolicy tok_policy;
            tok_policy.pretokenizer = 1;  // Llama3WordStream -> kind 3
            if (top.eos_id >= 0) tok_policy.eos_id = top.eos_id;
            if (top.pad_id >= 0) tok_policy.pad_id = top.pad_id;
            ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
                tok_path.string(), pinned_dir,
                integrity::TokenizerSpecialIds{.bos_required = true,
                                               .eos_required = true,
                                               .pad_required = true},
                tok_policy));
            tokenizer_ok = true;
        }
        else if (!SyntheticGeometryAllowed())
        {
            return arrow::Status::Invalid("tokenizer.json not found in ",
                                          input_dir);
        }
    }

    // Conversion-time VRAM profile.
    VramGeometry vg;
    vg.hidden_dim = top.hidden;
    vg.kv_dim = top.kv_lora + top.qk_rope;
    vg.num_attn_layers = static_cast<int>(top.full_attention_layers.size());
    vg.num_moe_layers = top.layers - top.first_k_dense;
    vg.experts_per_layer = top.num_experts;
    vg.context_length =
        ReadContextLength((fs::path(input_dir) / "config.json").string());
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            inv.push_back({t.name, t.numel(), t.byte_size()});
    // The depth argument matters: with the one-argument overload every layer at
    // or above ANOTHER member's depth would classify as MTP ballast and the
    // expert planes would vanish from the estimate the governor consumes.
    const ScopedClassifyDepth depth_scope(top.layers);
    const VramProfile vram = EstimateVram(
        inv, {&VramIsNvfp4, &VramMemClass, nullptr}, vg);

    std::vector<int> full_attn = top.full_attention_layers;
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"" << spec.format << "\",\n"
          << "  \"model\": \"" << spec.model_name << "\",\n"
          << "  \"num_layers\": " << top.layers << ",\n"
          << "  \"layer_types\": [";
        for (int i = 0; i < top.layers; ++i)
        {
            if (i) j << ",";
            const bool full =
                std::find(full_attn.begin(), full_attn.end(), i) !=
                full_attn.end();
            j << "\"" << (full ? "full_attention" : "linear_attention") << "\"";
        }
        j << "],\n  \"full_attention_layers\": [";
        for (size_t i = 0; i < full_attn.size(); ++i)
        {
            if (i) j << ",";
            j << full_attn[i];
        }
        j << "],\n  \"hidden_size\": " << top.hidden
          << ",\n  \"vocab_size\": " << top.vocab
          << ",\n  \"num_attention_heads\": " << top.heads
          << ",\n  \"num_key_value_heads\": " << top.kv_heads
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.top_k
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate
          << ",\n  \"first_k_dense_replace\": " << top.first_k_dense
          << ",\n  \"q_lora_rank\": " << top.q_lora
          << ",\n  \"kv_lora_rank\": " << top.kv_lora
          << ",\n  \"qk_nope_head_dim\": " << top.qk_nope
          << ",\n  \"qk_rope_head_dim\": " << top.qk_rope
          << ",\n  \"v_head_dim\": " << top.v_head
          << ",\n  \"rms_norm_eps\": " << top.rms_eps
          << ",\n  \"router_scaling_factor\": " << top.routed_scale
          << ",\n  \"swiglu_limit\": " << top.swiglu_limit
          << ",\n  \"logit_softcap\": " << top.logit_softcap
          << ",\n  \"attn_scale\": " << spec.attn_scale
          << ",\n  \"index_heads\": " << top.index_heads
          << ",\n  \"index_head_dim\": " << top.index_head_dim
          << ",\n  \"index_topk\": " << top.index_topk
          << ",\n  \"index_rope\": " << top.qk_rope
          << ",\n  \"bos_token_id\": " << top.bos_id
          << ",\n  \"eos_token_id\": " << top.eos_id
          << ",\n  \"pad_token_id\": " << top.pad_id
          << ",\n  \"vram\": {\n"
          << "    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"tokenizer\": \""
          << (tokenizer_ok ? "arrow-bpe" : "fixture") << "\""
          << ",\n  \"deferred\": {\"mtp\": \"stored\", \"mtp_layer\": "
          << top.layers << ", \"mtp_tensors\": " << mtp_count
          << ", \"dflash2\": \"separate 5-layer drafter directory, not "
             "ingested\"}"
          << ",\n  \"total_pinned_tensors\": " << total_pinned
          << ",\n  \"total_experts\": " << total_experts << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // Output verification: every produced arrow file re-opens cleanly.
    std::error_code ec;
    for (fs::recursive_directory_iterator it(base, ec), end; !ec && it != end;
         it.increment(ec))
    {
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".arrow") continue;
        auto res = ReadRecordBatchFromFile(it->path());
        if (!res.ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                          it->path().string());
    }
    if (ec) return arrow::Status::IOError("cannot list ", base.string());

    lsxcommon::log::info(absl::StrFormat(
        "GigaChat4 conversion: %d layers, %d experts, %d MTP tensors stored, "
        "%d pinned tensors",
        top.layers, total_experts, mtp_count, total_pinned));
    return arrow::Status::OK();
}

} // namespace lsxcommon::gigachat4
