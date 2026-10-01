// MiMo-V2.6-Flash-RL converter (Model Ingestion Protocol steps 1-2).
// Geometry: config.json is authoritative; the frozen passport
// (mimo_moe_constants.h) is cross-checked unless SyntheticGeometryAllowed().
// Dense linears are FP8 E4M3 + F32 weight_scale_inv block-128; routed experts
// are MXFP4 (U8 nibbles + U8 F8_E8M0 per-32 `.weight_scale`); norms/router/
// o_proj/sink bias stay high precision. Multimodal ballast is stripped; MTP is
// stored and counted, never applied.
#include "lsxcommon/mimo_moe_convert.h"
#include "lsxcommon/synthetic_geometry.h"

#include "mimo_moe_constants.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"

#include <simdjson.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string_view>

namespace lsxcommon {
namespace mimo {
namespace {

bool EndsWith(std::string_view s, std::string_view suf)
{
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool StartsWith(std::string_view s, std::string_view pre)
{
    return s.rfind(pre, 0) == 0;
}

// Strips a trailing quant scale/weight suffix to expose the module path.
std::string_view BaseModule(std::string_view name)
{
    for (std::string_view suf : {std::string_view(".weight_scale_inv"),
                                 std::string_view(".weight_scale"),
                                 std::string_view(".weight")})
        if (EndsWith(name, suf)) return name.substr(0, name.size() - suf.size());
    return name;
}

bool IsScaleTensor(std::string_view name)
{
    return EndsWith(name, ".weight_scale_inv") || EndsWith(name, ".weight_scale");
}

bool ParseLayerAndRest(std::string_view name, int &layer, std::string_view &rest)
{
    constexpr std::string_view kPrefix = "model.layers.";
    if (!StartsWith(name, kPrefix)) return false;
    std::string_view tail = name.substr(kPrefix.size());
    size_t i = 0;
    while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
    if (i == 0 || i >= tail.size() || tail[i] != '.') return false;
    layer = 0;
    for (size_t j = 0; j < i; ++j) layer = layer * 10 + (tail[j] - '0');
    rest = tail.substr(i + 1);
    return true;
}

} // namespace

ParsedMimoTensor classify_mimo_tensor(const std::string &name)
{
    ParsedMimoTensor out;
    const std::string_view n(name);

    if (StartsWith(n, "visual.") || StartsWith(n, "audio_encoder.") ||
        StartsWith(n, "speech_embeddings.") || StartsWith(n, "audio_tokenizer"))
    {
        out.kind = MiMoTensorKind::Multimodal;
        return out;
    }
    if (StartsWith(n, "model.mtp."))
    {
        out.kind = MiMoTensorKind::Mtp;
        out.layer = 0;
        return out;
    }
    if (n == "model.embed_tokens.weight") { out.kind = MiMoTensorKind::Embed; return out; }
    if (n == "lm_head.weight") { out.kind = MiMoTensorKind::LmHead; return out; }
    if (n == "model.norm.weight") { out.kind = MiMoTensorKind::FinalNorm; return out; }

    int layer = -1;
    std::string_view rest;
    if (!ParseLayerAndRest(n, layer, rest)) return out;
    out.layer = layer;

    const std::string_view base = BaseModule(rest);
    using K = MiMoTensorKind;
    if (base == "input_layernorm") { out.kind = K::InputNorm; return out; }
    if (base == "post_attention_layernorm") { out.kind = K::PostNorm; return out; }
    if (base == "self_attn.qkv_proj") { out.kind = K::QkvProj; return out; }
    if (base == "self_attn.o_proj") { out.kind = K::OProj; return out; }
    if (base == "self_attn.attention_sink_bias") { out.kind = K::SinkBias; return out; }
    if (base == "mlp.gate") { out.kind = K::RouterWeight; return out; }
    if (base == "mlp.gate.e_score_correction_bias") { out.kind = K::RouterBias; return out; }
    if (base == "mlp.gate_proj") { out.kind = K::DenseGate; return out; }
    if (base == "mlp.up_proj") { out.kind = K::DenseUp; return out; }
    if (base == "mlp.down_proj") { out.kind = K::DenseDown; return out; }

    // Routed experts: mlp.experts.<E>.<proj>.
    constexpr std::string_view kExperts = "mlp.experts.";
    if (StartsWith(base, kExperts))
    {
        std::string_view tail = base.substr(kExperts.size());
        size_t i = 0;
        while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
        if (i == 0 || i >= tail.size() || tail[i] != '.') return out;
        int expert = 0;
        for (size_t j = 0; j < i; ++j) expert = expert * 10 + (tail[j] - '0');
        const std::string_view proj = tail.substr(i + 1);
        out.expert = expert;
        if (proj == "gate_proj") out.kind = K::ExpertGate;
        else if (proj == "up_proj") out.kind = K::ExpertUp;
        else if (proj == "down_proj") out.kind = K::ExpertDown;
        return out;
    }
    return out;
}

bool mimo_tensor_is_nvfp4(MiMoTensorKind kind)
{
    switch (kind)
    {
    case MiMoTensorKind::QkvProj:
    case MiMoTensorKind::DenseGate:
    case MiMoTensorKind::DenseUp:
    case MiMoTensorKind::DenseDown:
    case MiMoTensorKind::ExpertGate:
    case MiMoTensorKind::ExpertUp:
    case MiMoTensorKind::ExpertDown:
        return true;
    default:
        return false;
    }
}

bool mimo_tensor_is_expert(MiMoTensorKind kind)
{
    return kind == MiMoTensorKind::ExpertGate ||
           kind == MiMoTensorKind::ExpertUp ||
           kind == MiMoTensorKind::ExpertDown;
}

MiMoTopology ParseMiMoConfig(const std::string &config_json_text)
{
    MiMoTopology t;
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
    double prf = -1.0;
    for (auto field : root)
    {
        std::string_view key;
        if (field.unescaped_key().get(key) != simdjson::SUCCESS) continue;
        if (key == "hybrid_layer_pattern" || key == "moe_layer_freq")
        {
            simdjson::ondemand::array arr;
            std::vector<int> vals;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto elem : arr)
                {
                    int64_t v = 0;
                    if (elem.get_int64().get(v) == simdjson::SUCCESS)
                        vals.push_back(static_cast<int>(v));
                }
            if (key == "hybrid_layer_pattern")
                for (size_t i = 0; i < vals.size(); ++i)
                    if (vals[i] == 0) t.full_attn_layers.push_back(static_cast<int>(i));
            if (key == "moe_layer_freq")
                for (size_t i = 0; i < vals.size(); ++i)
                    if (vals[i] != 0) t.moe_layers.push_back(static_cast<int>(i));
            continue;
        }
        if (key == "rope_parameters" || key == "quantization_config") continue;
        double d = 0.0;
        if (!as_num(field.value(), d)) continue;
        const int di = static_cast<int>(d);
        if (key == "hidden_size") { t.hidden = di; have_hidden = true; }
        else if (key == "num_hidden_layers") { t.layers = di; have_layers = true; }
        else if (key == "vocab_size") { t.vocab = di; have_vocab = true; }
        else if (key == "intermediate_size") t.dense_intermediate = di;
        else if (key == "moe_intermediate_size") t.moe_intermediate = di;
        else if (key == "n_routed_experts") t.num_experts = di;
        else if (key == "num_experts_per_tok") t.top_k = di;
        else if (key == "num_attention_heads") t.heads = di;
        else if (key == "num_key_value_heads") t.kv_heads_ga = di;
        else if (key == "swa_num_key_value_heads") t.kv_heads_swa = di;
        else if (key == "head_dim") t.head_dim = di;
        else if (key == "v_head_dim") t.v_head_dim = di;
        else if (key == "sliding_window" || key == "sliding_window_size") t.sliding_window = di;
        else if (key == "layernorm_epsilon") t.rms_eps = d;
        else if (key == "rope_theta") t.rope_theta_ga = d;
        else if (key == "swa_rope_theta") t.rope_theta_swa = d;
        else if (key == "attention_value_scale") t.value_scale = d;
        else if (key == "partial_rotary_factor") prf = d;
    }
    if (prf > 0.0) t.rope_dim = static_cast<int>(t.head_dim * prf);
    if (t.full_attn_layers.empty())
        for (int i = 0; i < t.layers; ++i) t.full_attn_layers.push_back(i);
    if (t.moe_layers.empty())
        for (int i = 1; i < t.layers; ++i) t.moe_layers.push_back(i);
    if (!have_hidden || !have_layers || !have_vocab)
        t.error_msg = "config.json missing required geometry";
    return t;
}

namespace {

std::vector<uint8_t> ReadTensorData(const std::string &input_dir,
                                    const SafetensorsModel::Shard &shard,
                                    const TensorInfo &info)
{
    std::ifstream sf(
        (std::filesystem::path(input_dir) / shard.filename).string(),
        std::ios::binary);
    if (!sf) return {};
    uint64_t hdr_len = 0;
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    sf.seekg(data_off + info.data_begin, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

struct TensorRef {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

std::string LayerFile(int layer, const char *suffix)
{
    return absl::StrFormat("layer_%03d%s", layer, suffix);
}

arrow::Status WriteBatchOr(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &b,
    const std::filesystem::path &path)
{
    if (!b.ok()) return b.status();
    return WriteRecordBatchToFile(*b, path);
}

// Dequantize one FP8 dense tensor and requantize it to NVFP4.
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
    ARROW_RETURN_NOT_OK(DequantFp8Block128ToBf16Gpu(
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

// FP8 -> dequantized BF16 raw record (the dense high-precision path: MiMo's
// FP8 dense weights are too sensitive to an 8-bit -> 4-bit requantization to
// survive 48 layers, so they stay BF16).
arrow::Status DequantFp8Record(const std::string &input_dir, const TensorRef &w,
                               const TensorRef &s, TensorRecord &out)
{
    auto fp8 = ReadTensorData(input_dir, *w.shard, *w.info);
    auto sc = ReadTensorData(input_dir, *s.shard, *s.info);
    if (fp8.empty() ||
        static_cast<int64_t>(fp8.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      w.info->name);
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(DequantFp8Block128ToBf16Gpu(
        fp8, w.info->shape, sc, s.info->shape, bf16));
    out.name = w.info->name;
    out.dtype = "BF16";
    out.shape = w.info->shape;
    out.data = std::move(bf16);
    return arrow::Status::OK();
}

struct RawProjection {
    std::vector<uint8_t> data;
    std::vector<int64_t> shape;
    std::vector<uint8_t> scale;
    std::vector<int64_t> scale_shape;
};

struct ExpertParts {
    RawProjection gate, up, down;
    bool has_gate = false, has_up = false, has_down = false;
};

// One projection -> slot, via dequant + requantize (the E8M0 repack fallback).
arrow::Status FallbackProjection(const RawProjection &p,
                                 std::vector<uint8_t> &out_packed,
                                 std::vector<uint8_t> &out_scales,
                                 std::vector<int64_t> &out_shape, float &out_gs)
{
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(
        DequantMxfp4ToBf16Gpu(p.data, p.shape, p.scale, p.scale_shape, bf16));
    std::vector<int64_t> logical_shape = {p.shape[0], p.shape[1] * 2};
    if (!gpu::quantize_tensor_gpu(bf16.data(), p.shape[0] * p.shape[1] * 2,
                                  "BF16", nullptr, logical_shape, {}, out_packed,
                                  out_scales, out_gs))
        return arrow::Status::IOError("fallback quantization failed");
    out_shape = p.shape;  // packed shape (ExpertStore sizes by bytes)
    return arrow::Status::OK();
}

// Pack a full expert (gate=w1, up=w3, down=w2). The neutral PackExpertNVFP4
// performs an exact power-of-two E8M0 -> NVFP4 repack; when the per-tensor
// scale spread cannot be folded it fails, and every projection falls back to
// dequant + requantize.
arrow::Status PackExpertParts(const ExpertParts &e, PackedExpert &pe)
{
    ExpertData ed;
    ed.w1_raw = e.gate.data; ed.w1_dtype = "U8"; ed.w1_shape = e.gate.shape;
    ed.w1_scale_raw = e.gate.scale; ed.w1_scale_shape = e.gate.scale_shape;
    ed.w3_raw = e.up.data; ed.w3_dtype = "U8"; ed.w3_shape = e.up.shape;
    ed.w3_scale_raw = e.up.scale; ed.w3_scale_shape = e.up.scale_shape;
    ed.w2_raw = e.down.data; ed.w2_dtype = "U8"; ed.w2_shape = e.down.shape;
    ed.w2_scale_raw = e.down.scale; ed.w2_scale_shape = e.down.scale_shape;
    auto packed = PackExpertNVFP4(ed);
    if (packed.ok())
    {
        pe = std::move(*packed);
        return arrow::Status::OK();
    }
    ARROW_RETURN_NOT_OK(FallbackProjection(e.gate, pe.w1_packed, pe.w1_scales,
                                           pe.w1_shape, pe.w1_global_scale));
    ARROW_RETURN_NOT_OK(FallbackProjection(e.up, pe.w3_packed, pe.w3_scales,
                                           pe.w3_shape, pe.w3_global_scale));
    ARROW_RETURN_NOT_OK(FallbackProjection(e.down, pe.w2_packed, pe.w2_scales,
                                           pe.w2_shape, pe.w2_global_scale));
    return arrow::Status::OK();
}

} // namespace

arrow::Status ConvertMiMoModel(const std::string &input_dir,
                               const std::string &output_root)
{
    namespace fs = std::filesystem;

    std::ifstream cf(fs::path(input_dir) / "config.json");
    if (!cf) return arrow::Status::Invalid("config.json not found in ", input_dir);
    std::stringstream css;
    css << cf.rdbuf();
    const MiMoTopology top = ParseMiMoConfig(css.str());
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("MiMo config error: ", top.error_msg);

    if (!SyntheticGeometryAllowed())
    {
        const bool ok =
            top.hidden == frozen::kHidden && top.layers == frozen::kLayers &&
            top.vocab == frozen::kVocab && top.heads == frozen::kHeads &&
            top.kv_heads_ga == frozen::kKvHeadsGa &&
            top.kv_heads_swa == frozen::kKvHeadsSwa &&
            top.head_dim == frozen::kQkHeadDim &&
            top.v_head_dim == frozen::kVHeadDim &&
            top.num_experts == frozen::kNumRouted && top.top_k == frozen::kTopK &&
            top.moe_intermediate == frozen::kMoeIntermediate &&
            top.dense_intermediate == frozen::kDenseIntermediate;
        if (!ok)
            return arrow::Status::Invalid(
                "MiMo frozen geometry mismatch (set "
                "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 only for synthetic tests)");
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ", input_dir);

    absl::flat_hash_map<std::string, TensorRef> refs;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) refs.insert_or_assign(t.name, TensorRef{&shard, &t});

    // Validate: every non-multimodal tensor is recognized.
    int mtp_count = 0;
    std::vector<int> layer_tensor_count(top.layers, 0);
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
        {
            auto p = classify_mimo_tensor(t.name);
            if (p.kind == MiMoTensorKind::Multimodal) continue;
            if (p.kind == MiMoTensorKind::Mtp) { ++mtp_count; continue; }
            if (p.kind == MiMoTensorKind::Unknown)
                return arrow::Status::Invalid("unrecognized tensor in MiMo checkpoint: ",
                                              t.name);
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

    absl::flat_hash_map<int64_t, ExpertParts> experts;

    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            auto p = classify_mimo_tensor(t.name);
            if (p.kind == MiMoTensorKind::Multimodal) continue;
            if (p.kind == MiMoTensorKind::Mtp)
            {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = ReadTensorData(input_dir, shard, t);
                mtp_by_layer[p.layer].push_back(std::move(r));
                continue;
            }
            if (IsScaleTensor(t.name)) continue;  // consumed with its `.weight`
            if (mimo_tensor_is_expert(p.kind)) continue;  // expert pass

            if (p.kind == MiMoTensorKind::QkvProj ||
                p.kind == MiMoTensorKind::DenseGate ||
                p.kind == MiMoTensorKind::DenseUp ||
                p.kind == MiMoTensorKind::DenseDown)
            {
                auto it = refs.find(t.name + "_scale_inv");
                if (it == refs.end())
                    return arrow::Status::Invalid("missing weight_scale_inv for ",
                                                  t.name);
                TensorRecord r;
                ARROW_RETURN_NOT_OK(
                    DequantFp8Record(input_dir, refs[t.name], it->second, r));
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                continue;
            }

            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = ReadTensorData(input_dir, shard, t);
            if (r.data.empty() ||
                static_cast<int64_t>(r.data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ", t.name);
            switch (p.kind)
            {
            case MiMoTensorKind::Embed:
            case MiMoTensorKind::LmHead:
            case MiMoTensorKind::FinalNorm:
                global_raw.push_back(std::move(r));
                break;
            default:
                raw_by_layer[p.layer].push_back(std::move(r));
                break;
            }
            ++total_pinned;
        }
    }

    // Expert pass: pair each `.weight` (MXFP4 U8) with its `.weight_scale` (U8
    // E8M0 per-32).
    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            auto p = classify_mimo_tensor(t.name);
            if (!mimo_tensor_is_expert(p.kind)) continue;
            if (IsScaleTensor(t.name)) continue;
            auto it = refs.find(t.name.substr(0, t.name.size() - 7) + ".weight_scale");
            if (it == refs.end())
                return arrow::Status::Invalid("missing .weight_scale for ", t.name);
            RawProjection rp;
            rp.data = ReadTensorData(input_dir, shard, t);
            rp.shape = t.shape;
            rp.scale = ReadTensorData(input_dir, *it->second.shard, *it->second.info);
            rp.scale_shape = it->second.info->shape;
            if (rp.data.empty() || rp.scale.empty())
                return arrow::Status::IOError("failed to read expert data: ", t.name);
            const int64_t key = static_cast<int64_t>(p.layer) * 100000 + p.expert;
            auto &e = experts[key];
            if (p.kind == MiMoTensorKind::ExpertGate) { e.gate = std::move(rp); e.has_gate = true; }
            else if (p.kind == MiMoTensorKind::ExpertUp) { e.up = std::move(rp); e.has_up = true; }
            else { e.down = std::move(rp); e.has_down = true; }
        }
    }

    int total_experts = 0;
    for (auto &[key, e] : experts)
    {
        const int layer = static_cast<int>(key / 100000);
        const int expert = static_cast<int>(key % 100000);
        if (!e.has_gate || !e.has_up || !e.has_down)
            return arrow::Status::Invalid("incomplete expert: layer ", layer, " expert ",
                                          expert);
        PackedExpert pe;
        ARROW_RETURN_NOT_OK(PackExpertParts(e, pe));
        const fs::path dir = base / "experts" / absl::StrCat("layer_", layer);
        std::error_code ec;
        fs::create_directories(dir, ec);
        ARROW_RETURN_NOT_OK(WriteBatchOr(ExpertSchema::MakeBatch(pe),
                                         dir / absl::StrCat("exp_", expert, ".arrow")));
        ++total_experts;
    }

    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(
            WriteBatchOr(PinnedSchema::MakeBatch(global_raw), pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(DenseNvfp4Schema::MakeBatch(recs),
                                         pinned_dir / LayerFile(layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(recs),
                                         pinned_dir / LayerFile(layer, ".arrow")));
    for (auto &[layer, recs] : mtp_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(recs),
                                         mtp_dir / LayerFile(layer, ".arrow")));

    // Conversion-time VRAM profile.
    VramGeometry vg;
    vg.hidden_dim = top.hidden;
    vg.kv_dim = top.head_dim * std::max(top.kv_heads_ga, top.kv_heads_swa);
    vg.num_attn_layers = top.layers;
    vg.num_moe_layers = static_cast<int>(top.moe_layers.size());
    vg.experts_per_layer = top.num_experts;
    vg.context_length = ReadContextLength((fs::path(input_dir) / "config.json").string());
    std::vector<VramTensorInfo> inv;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) inv.push_back({t.name, t.numel(), t.byte_size()});
    const VramProfile vram = EstimateVram(
        inv,
        {+[](const std::string &n) {
             return mimo_tensor_is_nvfp4(classify_mimo_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return mimo_tensor_is_expert(classify_mimo_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         }},
        vg);

    // Structural manifest (deterministic).
    {
        std::ostringstream j;
        j << "{\n  \"format\": \"logestix-mimo-nvfp4\",\n"
          << "  \"model\": \"MiMo-V2.6-Flash-RL\",\n"
          << "  \"num_layers\": " << top.layers << ",\n  \"layer_types\": [";
        for (int i = 0; i < top.layers; ++i)
        {
            if (i) j << ",";
            const bool full = std::find(top.full_attn_layers.begin(),
                                        top.full_attn_layers.end(),
                                        i) != top.full_attn_layers.end();
            j << "\"" << (full ? "full_attention" : "sliding_attention") << "\"";
        }
        j << "],\n  \"full_attention_layers\": [";
        for (size_t i = 0; i < top.full_attn_layers.size(); ++i)
        {
            if (i) j << ",";
            j << top.full_attn_layers[i];
        }
        j << "],\n  \"hidden_size\": " << top.hidden
          << ",\n  \"vocab_size\": " << top.vocab
          << ",\n  \"num_attention_heads\": " << top.heads
          << ",\n  \"num_key_value_heads\": " << top.kv_heads_ga
          << ",\n  \"num_key_value_heads_swa\": " << top.kv_heads_swa
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"v_head_dim\": " << top.v_head_dim
          << ",\n  \"rope_dim\": " << top.rope_dim
          << ",\n  \"sliding_window\": " << top.sliding_window
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.top_k
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate
          << ",\n  \"rms_norm_eps\": " << top.rms_eps
          << ",\n  \"rope_theta\": " << top.rope_theta_ga
          << ",\n  \"swa_rope_theta\": " << top.rope_theta_swa
          << ",\n  \"attention_value_scale\": " << top.value_scale
          << ",\n  \"scoring_func\": \"sigmoid\", \"topk_method\": \"noaux_tc\""
          << ",\n  \"norm_topk_prob\": true"
          << ",\n  \"eos_token_id\": " << frozen::kEosId
          << ",\n  \"pad_token_id\": " << frozen::kPadId
          << ",\n  \"bos_token_id\": " << frozen::kBosId
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"deferred\": {\"mtp\": \"stored\", \"mtp_layers\": [0,1,2]"
             ", \"mtp_tensors\": "
          << mtp_count << "}"
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
        if (!it->is_regular_file() || it->path().extension() != ".arrow") continue;
        if (!ReadRecordBatchFromFile(it->path()).ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                          it->path().string());
    }
    if (ec) return arrow::Status::IOError("cannot list ", base.string());

    // Tokenizer bridge.
    auto tok_path = fs::path(input_dir) / "tokenizer.json";
    if (fs::exists(tok_path))
    {
        auto st = BuildTokenizerArrowFiles(tok_path.string(), pinned_dir);
        if (!st.ok()) return st;
    }

    lsxcommon::log::info(absl::StrFormat(
        "MiMo conversion: %d layers, %d experts, %d MTP tensors stored, "
        "%d pinned tensors",
        top.layers, total_experts, mtp_count, total_pinned));
    return arrow::Status::OK();
}

} // namespace mimo
} // namespace lsxcommon
