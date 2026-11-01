// GigaChat3.5-432B-A28B-Reasoning converter (Model Ingestion Protocol step 1).
// See gigachat_moe_constants.h for the frozen geometry and the ingestion design
// docs/superpowers/specs/2026-09-20-gigachat35-432b-a28b-reasoning-ingestion-design.md.
#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/synthetic_geometry.h"

#include "gigachat_moe_constants.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
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

namespace lsxcommon::gigachat {
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
    // Expert index, when present: experts.<E>.<proj>[.weight]
    return true;
}

GigaChatTensorKind ClassifyModule(std::string_view module, int &expert)
{
    expert = -1;
    const std::string_view last = LastComponent(module);

    // Layer norms (ZeroCenteredGatedNorm): weight / gate projections.
    if (last == "input_layernorm" || last == "post_attention_layernorm" ||
        last == "post_self_attn_layernorm" ||
        last == "post_feedforward_layernorm" ||
        last == "q_a_layernorm" || last == "kv_a_layernorm")
        return GigaChatTensorKind::LayerNormWeight;
    if (last == "gate_up_projection")
        return GigaChatTensorKind::LayerNormGateUp;
    if (last == "gate_down_projection")
        return GigaChatTensorKind::LayerNormGateDown;

    // MLA attention.
    if (module == "self_attn.q_a_proj") return GigaChatTensorKind::QaProj;
    if (module == "self_attn.q_b_proj") return GigaChatTensorKind::QbProj;
    if (module == "self_attn.kv_a_proj_with_mqa") return GigaChatTensorKind::KvaProj;
    if (module == "self_attn.kv_b_proj") return GigaChatTensorKind::KvbProj;
    if (module == "self_attn.o_proj") return GigaChatTensorKind::OProj;
    if (module == "self_attn.gate_proj") return GigaChatTensorKind::AttnGateProj;

    // GatedDeltaNet.
    if (module == "self_attn.in_proj_qkvz") return GigaChatTensorKind::GdnInProjQkvz;
    if (module == "self_attn.out_proj") return GigaChatTensorKind::GdnOutProj;
    if (module == "self_attn.conv1d") return GigaChatTensorKind::GdnConv1d;
    if (module == "self_attn.A_log") return GigaChatTensorKind::GdnALog;
    if (module == "self_attn.dt_bias") return GigaChatTensorKind::GdnDtBias;
    if (module == "self_attn.in_proj_ba") return GigaChatTensorKind::GdnInProjBa;
    if (module == "self_attn.norm") return GigaChatTensorKind::GdnNormWeight;

    // Router (BF16 gate + F32 correction bias).
    if (module == "mlp.gate") return GigaChatTensorKind::RouterWeight;
    if (module == "mlp.gate.e_score_correction_bias")
        return GigaChatTensorKind::RouterBias;

    // Routed experts: mlp.experts.<E>.<proj>
    constexpr std::string_view kExperts = "mlp.experts.";
    if (module.rfind(kExperts, 0) == 0)
    {
        std::string_view tail = module.substr(kExperts.size());
        size_t i = 0;
        while (i < tail.size() && tail[i] >= '0' && tail[i] <= '9') ++i;
        if (i == 0 || i >= tail.size() || tail[i] != '.') return GigaChatTensorKind::Unknown;
        expert = 0;
        for (size_t j = 0; j < i; ++j) expert = expert * 10 + (tail[j] - '0');
        const std::string_view proj = tail.substr(i + 1);
        if (proj == "gate_proj") return GigaChatTensorKind::ExpertGate;
        if (proj == "up_proj") return GigaChatTensorKind::ExpertUp;
        if (proj == "down_proj") return GigaChatTensorKind::ExpertDown;
        return GigaChatTensorKind::Unknown;
    }

    // Shared experts.
    if (module == "mlp.shared_experts.gate_proj") return GigaChatTensorKind::SharedGate;
    if (module == "mlp.shared_experts.up_proj") return GigaChatTensorKind::SharedUp;
    if (module == "mlp.shared_experts.down_proj") return GigaChatTensorKind::SharedDown;

    // Dense MLP.
    if (module == "mlp.gate_proj") return GigaChatTensorKind::MlpGate;
    if (module == "mlp.up_proj") return GigaChatTensorKind::MlpUp;
    if (module == "mlp.down_proj") return GigaChatTensorKind::MlpDown;

    return GigaChatTensorKind::Unknown;
}

} // namespace

ParsedGigaChatTensor classify_gigachat_tensor(const std::string &name)
{
    ParsedGigaChatTensor out;
    const std::string_view n(name);

    if (n == "model.embed_tokens.weight")
    {
        out.kind = GigaChatTensorKind::Embed;
        return out;
    }
    if (n == "lm_head.weight")
    {
        out.kind = GigaChatTensorKind::LmHead;
        return out;
    }
    if (n.rfind("model.norm", 0) == 0)
    {
        const std::string_view module = BaseModule(n);
        if (module == "model.norm")
            out.kind = GigaChatTensorKind::FinalNormWeight;
        else if (module == "model.norm.gate_up_projection")
            out.kind = GigaChatTensorKind::FinalNormGateUp;
        else if (module == "model.norm.gate_down_projection")
            out.kind = GigaChatTensorKind::FinalNormGateDown;
        return out;
    }

    int layer = -1;
    std::string_view rest;
    if (!ParseLayerAndRest(n, layer, rest))
        return out;
    out.layer = layer;

    // MTP layers 40-42: converted + stored, never applied.
    if (layer == 40 || layer == 41 || layer == 42)
    {
        out.kind = GigaChatTensorKind::Mtp;
        return out;
    }

    out.kind = ClassifyModule(BaseModule(rest), out.expert);
    return out;
}

bool gigachat_tensor_is_nvfp4(GigaChatTensorKind kind)
{
    switch (kind)
    {
    case GigaChatTensorKind::QaProj:
    case GigaChatTensorKind::QbProj:
    case GigaChatTensorKind::KvaProj:
    case GigaChatTensorKind::KvbProj:
    case GigaChatTensorKind::OProj:
    case GigaChatTensorKind::AttnGateProj:
    case GigaChatTensorKind::MlpGate:
    case GigaChatTensorKind::MlpUp:
    case GigaChatTensorKind::MlpDown:
    case GigaChatTensorKind::GdnInProjQkvz:
    case GigaChatTensorKind::GdnOutProj:
    case GigaChatTensorKind::SharedGate:
    case GigaChatTensorKind::SharedUp:
    case GigaChatTensorKind::SharedDown:
    case GigaChatTensorKind::ExpertGate:
    case GigaChatTensorKind::ExpertUp:
    case GigaChatTensorKind::ExpertDown:
        return true;
    default:
        return false;
    }
}

bool gigachat_tensor_is_expert(GigaChatTensorKind kind)
{
    return kind == GigaChatTensorKind::ExpertGate ||
           kind == GigaChatTensorKind::ExpertUp ||
           kind == GigaChatTensorKind::ExpertDown;
}

GigaChatTopology ParseGigaChatConfig(const std::string &config_json_text)
{
    GigaChatTopology t;
    // Spec defaults (config.json is authoritative where present).
    t.heads = 64;
    t.kv_heads = 64;
    t.num_experts = 256;
    t.top_k = 8;
    t.moe_intermediate = 2048;
    t.first_k_dense = 3;
    t.rms_eps = 1e-6;
    t.routed_scale = 2.5;
    t.swiglu_limit = 10.0;
    t.bos_id = 1;
    t.eos_id = 2;
    t.pad_id = 2;

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
        else if (key == "first_k_dense_replace") t.first_k_dense = di;
        else if (key == "rms_norm_eps") t.rms_eps = d;
        else if (key == "routed_scaling_factor") t.routed_scale = d;
        else if (key == "swiglu_limit") t.swiglu_limit = d;
        else if (key == "bos_token_id") t.bos_id = di;
        else if (key == "eos_token_id") t.eos_id = di;
        else if (key == "pad_token_id") t.pad_id = di;
    }

    if (!have_hidden || !have_layers || !have_vocab)
    {
        t.error_msg = "config.json missing required geometry "
                      "(hidden_size/num_hidden_layers/vocab_size)";
    }
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
    ARROW_RETURN_NOT_OK(DequantFp8Block128ToBf16Gpu(fp8, w.info->shape, sc,
                                                    s.info->shape, bf16));
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

} // namespace

arrow::Status ConvertGigaChatModel(const std::string &input_dir,
                                   const std::string &output_root)
{
    namespace fs = std::filesystem;

    std::ifstream cf(fs::path(input_dir) / "config.json");
    if (!cf) return arrow::Status::Invalid("config.json not found in ", input_dir);
    std::stringstream css;
    css << cf.rdbuf();
    const GigaChatTopology top = ParseGigaChatConfig(css.str());
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("GigaChat config error: ", top.error_msg);

    if (!SyntheticGeometryAllowed())
    {
        const bool ok = top.hidden == frozen::kHidden &&
                        top.layers == frozen::kLayers &&
                        top.vocab == frozen::kVocab &&
                        top.heads == frozen::kHeads &&
                        top.kv_heads == frozen::kKVHeads &&
                        top.num_experts == frozen::kNumRouted &&
                        top.top_k == frozen::kTopK &&
                        top.moe_intermediate == frozen::kMoeIntermediate &&
                        top.first_k_dense == frozen::kFirstKDense;
        if (!ok)
            return arrow::Status::Invalid(
                "GigaChat frozen geometry mismatch (set "
                "LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 only for synthetic tests)");
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
            auto p = classify_gigachat_tensor(t.name);
            if (p.kind == GigaChatTensorKind::Mtp)
            {
                ++mtp_count;
                continue;
            }
            if (p.kind == GigaChatTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in gigachat checkpoint: ", t.name);
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

    struct ExpertRec {
        DenseNvfp4Record gate, up, down;
        bool has_gate = false, has_up = false, has_down = false;
    };
    absl::flat_hash_map<int64_t, ExpertRec> experts;

    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            auto p = classify_gigachat_tensor(t.name);
            if (p.kind == GigaChatTensorKind::Mtp)
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
            if (t.name.size() >= kScale.size() &&
                std::string_view(t.name).substr(t.name.size() -
                                                kScale.size()) == kScale)
                continue;  // consumed together with its `.weight`
            if (gigachat_tensor_is_expert(p.kind)) continue;  // expert pass

            if (gigachat_tensor_is_nvfp4(p.kind))
            {
                auto it = refs.find(t.name + "_scale_inv");
                if (it == refs.end())
                    return arrow::Status::Invalid(
                        "missing weight_scale_inv for ", t.name);
                DenseNvfp4Record r;
                ARROW_RETURN_NOT_OK(
                    QuantizeFp8(input_dir, refs[t.name], it->second, r));
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
                case GigaChatTensorKind::Embed:
                case GigaChatTensorKind::LmHead:
                case GigaChatTensorKind::FinalNormWeight:
                case GigaChatTensorKind::FinalNormGateUp:
                case GigaChatTensorKind::FinalNormGateDown:
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

    // Routed experts: 3 projections per expert.
    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            auto p = classify_gigachat_tensor(t.name);
            if (!gigachat_tensor_is_expert(p.kind)) continue;
            constexpr std::string_view kScale = ".weight_scale_inv";
            if (t.name.size() >= kScale.size() &&
                std::string_view(t.name).substr(t.name.size() -
                                                kScale.size()) == kScale)
                continue;  // consumed together with its `.weight`
            auto it = refs.find(t.name + "_scale_inv");
            if (it == refs.end())
                return arrow::Status::Invalid("missing weight_scale_inv for ",
                                              t.name);
            DenseNvfp4Record r;
            ARROW_RETURN_NOT_OK(
                QuantizeFp8(input_dir, refs[t.name], it->second, r));
            const int64_t key =
                static_cast<int64_t>(p.layer) * 100000 + p.expert;
            auto &e = experts[key];
            if (p.kind == GigaChatTensorKind::ExpertGate)
            {
                e.gate = std::move(r);
                e.has_gate = true;
            }
            else if (p.kind == GigaChatTensorKind::ExpertUp)
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
    }

    int total_experts = 0;
    for (auto &[key, e] : experts)
    {
        const int layer = static_cast<int>(key / 100000);
        const int expert = static_cast<int>(key % 100000);
        if (!e.has_gate || !e.has_up || !e.has_down)
            return arrow::Status::Invalid("incomplete expert: layer ", layer,
                                          " expert ", expert);
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
        const fs::path dir = base / "experts" / absl::StrCat("layer_", layer);
        std::error_code ec;
        fs::create_directories(dir, ec);
        ARROW_RETURN_NOT_OK(WriteBatchOr(
            ExpertSchema::MakeBatch(pe),
            dir / absl::StrCat("exp_", expert, ".arrow")));
        ++total_experts;
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

    // Model Ingestion Protocol stage 4 (tokenizer bridge). GigaChat's
    // checkpoint is a byte-level BPE with the Llama-3/o200k case-run
    // pre-tokenizer (contraction suffix, \p{N}{1,3}, CJK inside the letter
    // runs), i.e. engine tokenizer_kind 3 — NOT the legacy kind-0 split, which
    // mis-splits mixed-case identifiers. The builder resolves `<s>`/`</s>`
    // from tokenizer_config.json; the config's bos/eos/pad are passed through
    // as the authoritative override.
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
    vg.kv_dim = frozen::kKvLoraRank + frozen::kQkRopeHeadDim;
    vg.num_attn_layers = static_cast<int>(top.full_attention_layers.size());
    vg.num_moe_layers = top.layers - top.first_k_dense;
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
             return gigachat_tensor_is_nvfp4(classify_gigachat_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return gigachat_tensor_is_expert(classify_gigachat_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         }},
        vg);

    // Structural manifest (deterministic).
    std::vector<int> full_attn = top.full_attention_layers;
    {
        std::ostringstream j;
        j << "{\n"
          << "  \"format\": \"logestix-gigachat-nvfp4\",\n"
          << "  \"model\": \"GigaChat3.5-432B-A28B-Reasoning\",\n"
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
          << ",\n  \"first_k_dense_replace\": " << top.first_k_dense
          << ",\n  \"rms_norm_eps\": " << top.rms_eps
          << ",\n  \"router_scaling_factor\": " << top.routed_scale
          << ",\n  \"swiglu_limit\": " << top.swiglu_limit
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
          << ",\n  \"deferred\": {\"mtp\": \"stored\", \"mtp_layers\": [40,41,42]"
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
        if (!it->is_regular_file()) continue;
        if (it->path().extension() != ".arrow") continue;
        auto res = ReadRecordBatchFromFile(it->path());
        if (!res.ok())
            return arrow::Status::IOError("unreadable output arrow: ",
                                          it->path().string());
    }
    if (ec) return arrow::Status::IOError("cannot list ", base.string());

    lsxcommon::log::info(absl::StrFormat(
        "GigaChat conversion: %d layers, %d experts, %d MTP tensors stored, "
        "%d pinned tensors",
        top.layers, total_experts, mtp_count, total_pinned));
    return arrow::Status::OK();
}

} // namespace lsxcommon::gigachat
