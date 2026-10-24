// GLM-5.3-Flash converter: tensor classifier (Task 1), config parser +
// FP8->NVFP4 requant host reference (Task 2), conversion pipeline (Task 3).
#include "glm_moe_flash_constants.h"
#include "lsxcommon/glm_moe_flash_convert.h"
#include "lsxcommon/synthetic_geometry.h"
#include "glm_moe_flash_fp8.h"
#include "tensor_index_parse.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <arrow/api.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <simdjson.h>

#include "lsxcommon/converter.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/tokenizer.h"

namespace fs = std::filesystem;

namespace lsxcommon::glm_moe_flash {

namespace {

// Matches "<prefix>layers.<N>.<rest>" for both model.layers.N and
// model.language_model.layers.N. Returns true with layer + rest.
bool SplitLayer(const std::string &name, int &layer, std::string &rest)
{
    static const std::array<const char *, 2> kPrefixes = {
        "model.language_model.layers.", "model.layers."};
    for (const char *pre : kPrefixes) {
        const size_t p = name.find(pre);
        if (p == std::string::npos) continue;
        const size_t ds = p + std::strlen(pre);
        size_t i = ds;
        while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
        if (i == ds || i >= name.size() || name[i] != '.') continue;
        // Bounded parse: an over-range digit run must reject the name, not
        // invoke atoi's undefined overflow behavior.
        if (!detail::ParseTensorIndex(std::string_view(name).substr(ds, i - ds),
                                      layer))
            continue;
        rest = name.substr(i + 1);
        return true;
    }
    return false;
}

// Extracts the routed expert index from "mlp.experts.<E>.<suffix>".
bool SplitExpert(const std::string &rest, int &expert, std::string &suffix)
{
    static const char kKey[] = "mlp.experts.";
    const size_t p = rest.find(kKey);
    if (p == std::string::npos) return false;
    const size_t ds = p + sizeof(kKey) - 1;
    size_t i = ds;
    while (i < rest.size() && rest[i] >= '0' && rest[i] <= '9') ++i;
    if (i == ds || i >= rest.size() || rest[i] != '.') return false;
    // Bounded parse: see SplitLayer.
    if (!detail::ParseTensorIndex(std::string_view(rest).substr(ds, i - ds),
                                  expert))
        return false;
    suffix = rest.substr(i + 1);
    return true;
}

int ProjId(const std::string &suffix)
{
    if (suffix.rfind("gate_proj", 0) == 0) return 1;
    if (suffix.rfind("down_proj", 0) == 0) return 2;
    if (suffix.rfind("up_proj", 0) == 0) return 3;
    return 0;
}

} // namespace

ParsedFlashTensor classify_flash_tensor(const std::string &name)
{
    ParsedFlashTensor out;
    if (name == "lm_head.weight") {
        out.kind = FlashTensorKind::LmHead;
        return out;
    }
    if (name == "model.language_model.embed_tokens.weight" ||
        name == "model.embed_tokens.weight") {
        out.kind = FlashTensorKind::Embed;
        return out;
    }
    if (name == "model.language_model.norm.weight") {
        out.kind = FlashTensorKind::FinalNorm;
        return out;
    }
    if (name.rfind("model.visual.", 0) == 0 || name.rfind("visual.", 0) == 0) {
        out.kind = FlashTensorKind::Visual;
        return out;
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayer(name, layer, rest)) return out;
    out.layer = layer;

    if (layer == frozen::kMtpLayer && rest != "mlp.experts." + rest) {
        // MTP nextn layer: distinct MTP tensors first; the remaining
        // (attn/mlp/indexer) tensors fall through to normal classification
        // below — the MTP layer is itself a DSA+MoE layer.
        if (rest == "eh_proj.weight" || rest == "enorm.weight" ||
            rest == "hnorm.weight" || rest == "shared_head.norm.weight") {
            out.kind = FlashTensorKind::MtpEhProj;
            return out;
        }
    }

    int expert = -1;
    std::string suffix;
    if (SplitExpert(rest, expert, suffix)) {
        out.expert = expert;
        out.proj_id = ProjId(suffix);
        out.kind = FlashTensorKind::RoutedExpert;
        return out;
    }
    if (rest.rfind("mlp.shared_experts.", 0) == 0) {
        out.proj_id = ProjId(rest);
        out.kind = FlashTensorKind::SharedExpert;
        return out;
    }
    if (rest == "mlp.gate.weight") {
        out.kind = FlashTensorKind::RouterGate;
        return out;
    }
    if (rest == "mlp.gate.e_score_correction_bias") {
        out.kind = FlashTensorKind::RouterBias;
        return out;
    }
    if (rest.rfind("mlp.gate_proj", 0) == 0 ||
        rest.rfind("mlp.up_proj", 0) == 0 ||
        rest.rfind("mlp.down_proj", 0) == 0) {
        out.kind = FlashTensorKind::DenseMlp;
        out.proj_id = ProjId(rest);
        return out;
    }

    if (rest.rfind("self_attn.indexer.", 0) == 0) {
        out.kind = FlashTensorKind::Indexer;
        return out;
    }

    // hc_* tensors carry no ".weight" suffix.
    static const std::array<std::pair<const char *, FlashTensorKind>, 6>
        kHcKinds = {{
            {"hc_attn_base", FlashTensorKind::HcAttnBase},
            {"hc_attn_fn", FlashTensorKind::HcAttnFn},
            {"hc_attn_scale", FlashTensorKind::HcAttnScale},
            {"hc_ffn_base", FlashTensorKind::HcFfnBase},
            {"hc_ffn_fn", FlashTensorKind::HcFfnFn},
            {"hc_ffn_scale", FlashTensorKind::HcFfnScale},
        }};
    for (const auto &kv : kHcKinds) {
        if (rest == kv.first) {
            out.kind = kv.second;
            return out;
        }
    }

    if (rest == "input_layernorm.weight" ||
        rest == "post_attention_layernorm.weight") {
        out.kind = FlashTensorKind::DsaNorms;
        return out;
    }

    if (frozen::IsKdaLayer(layer)) {
        static const std::array<std::pair<const char *, FlashTensorKind>, 14>
            kKdaKinds = {{
                {"self_attn.q_proj.weight", FlashTensorKind::KdaQ},
                {"self_attn.k_proj.weight", FlashTensorKind::KdaK},
                {"self_attn.v_proj.weight", FlashTensorKind::KdaV},
                {"self_attn.o_proj.weight", FlashTensorKind::KdaO},
                {"self_attn.q_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.k_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.v_conv1d.weight", FlashTensorKind::KdaConv},
                {"self_attn.A_log", FlashTensorKind::KdaALog},
                {"self_attn.dt_bias", FlashTensorKind::KdaDtBias},
                {"self_attn.b_proj.weight", FlashTensorKind::KdaB},
                {"self_attn.f_a_proj.weight", FlashTensorKind::KdaFa},
                {"self_attn.f_b_proj.weight", FlashTensorKind::KdaFb},
                {"self_attn.g_a_proj.weight", FlashTensorKind::KdaGa},
                {"self_attn.g_b_proj.weight", FlashTensorKind::KdaGb},
            }};
        for (const auto &kv : kKdaKinds) {
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        }
        if (rest == "self_attn.o_norm.weight") {
            out.kind = FlashTensorKind::KdaONorm;
            return out;
        }
        return out;  // Unknown inside a KDA layer
    }

    if (frozen::IsDsaLayer(layer) || layer == frozen::kMtpLayer) {
        static const std::array<std::pair<const char *, FlashTensorKind>, 5>
            kDsaKinds = {{
                {"self_attn.q_a_proj.weight", FlashTensorKind::DsaQa},
                {"self_attn.q_b_proj.weight", FlashTensorKind::DsaQb},
                {"self_attn.kv_a_proj_with_mqa.weight",
                 FlashTensorKind::DsaKvA},
                {"self_attn.kv_b_proj.weight", FlashTensorKind::DsaKvB},
                {"self_attn.o_proj.weight", FlashTensorKind::DsaOBig},
            }};
        for (const auto &kv : kDsaKinds) {
            if (rest == kv.first) {
                out.kind = kv.second;
                return out;
            }
        }
        if (rest == "self_attn.q_a_layernorm.weight" ||
            rest == "self_attn.kv_a_layernorm.weight") {
            out.kind = FlashTensorKind::DsaNorms;
            return out;
        }
        return out;
        return out;
    }

    return out;
}

bool flash_tensor_is_nvfp4(FlashTensorKind kind)
{
    switch (kind) {
    case FlashTensorKind::RoutedExpert:
    case FlashTensorKind::SharedExpert:
    case FlashTensorKind::DsaQa:
    case FlashTensorKind::DsaQb:
    case FlashTensorKind::DsaKvA:
    case FlashTensorKind::DsaOBig:
    case FlashTensorKind::DenseMlp:
    // KDA attention projections: [4096,8192]/[8192,4096] fp16 pairs
    // dominated the device-dense footprint (~9 GiB). Quantize at
    // conversion time (fp8 source -> bf16 -> NVFP4, same pipeline as the
    // DSA projections) so the loader can upload NVFP4 directly.
    case FlashTensorKind::KdaQ:
    case FlashTensorKind::KdaK:
    case FlashTensorKind::KdaV:
    case FlashTensorKind::KdaO:
        return true;
    default:
        return false;
    }
}

FlashTopology ParseFlashConfig(const std::string &input_dir)
{
    namespace fs = std::filesystem;
    FlashTopology top;
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

    auto get_i = [](simdjson::ondemand::object &o, const char *k,
                    int dflt) -> int {
        int64_t v;
        if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_d = [](simdjson::ondemand::object &o, const char *k,
                    double dflt) -> double {
        simdjson::ondemand::value x;
        if (o[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
            return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };

    // text_config preferred; fall back to flat root.
    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object();
        tc.error() == simdjson::SUCCESS)
        src = tc.value();

    top.hidden = get_i(src, "hidden_size", 0);
    top.layers = get_i(src, "num_hidden_layers", 0);
    top.vocab = get_i(src, "vocab_size", 0);
    top.heads = get_i(src, "num_attention_heads", 0);
    top.qk_head_dim = get_i(src, "qk_head_dim", 0);
    top.q_lora = get_i(src, "q_lora_rank", 0);
    top.kv_lora = get_i(src, "kv_lora_rank", 0);
    top.num_experts = get_i(src, "n_routed_experts", 0);
    top.top_k = get_i(src, "num_experts_per_tok", 0);
    top.moe_intermediate = get_i(src, "moe_intermediate_size", 0);
    top.dense_intermediate = get_i(src, "intermediate_size", 0);
    top.hc_count = get_i(src, "hc_mult", 4);
    top.gate_lowrank = get_i(src, "gate_rank", frozen::kGateLowrank);
    top.indexer_heads = get_i(src, "index_n_heads", 0);
    top.indexer_head_dim = get_i(src, "index_head_dim", 0);
    top.indexer_topk = get_i(src, "index_topk", 0);

    // Short conv kernel size lives in linear_attn_config.
    if (auto lc = src["linear_attn_config"].get_object();
        lc.error() == simdjson::SUCCESS) {
        simdjson::ondemand::object lco = lc.value();
        top.conv_kernel =
            get_i(lco, "short_conv_kernel_size", frozen::kConvKernel);
        if (top.heads == 0)
            top.heads = get_i(lco, "num_heads", 0);
        if (top.qk_head_dim == 0)
            top.qk_head_dim = get_i(lco, "head_dim", 0);
    } else {
        top.conv_kernel = frozen::kConvKernel;
    }

    top.routed_scaling = get_d(src, "routed_scaling_factor", 0.0);
    top.pad_id = get_i(src, "pad_token_id", -1);

    // eos_token_id may be an array or a scalar.
    if (auto arr = src["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto v : arr.value()) {
            int64_t e;
            if (v.get_int64().get(e) == simdjson::SUCCESS)
                top.eos_ids.push_back(static_cast<int>(e));
        }
    } else {
        const int single = get_i(src, "eos_token_id", -1);
        if (single >= 0) top.eos_ids.push_back(single);
    }

    // Structural sanity.
    if (top.hidden <= 0 || top.layers <= 0 || top.vocab <= 0) {
        top.error_msg = "config.json missing core text_config fields";
        return top;
    }
    if (top.num_experts <= 0 || top.top_k <= 0) {
        top.error_msg = "config.json missing MoE fields";
        return top;
    }
    return top;
}

// ── Conversion pipeline (Task 3) ─────────────────────────────────────────

namespace {

std::vector<uint8_t> read_tensor_data(const std::string &input_dir,
                                      const SafetensorsModel::Shard &shard,
                                      const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((fs::path(input_dir) / shard.filename).string(),
                     std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    sf.seekg(data_off + info.data_begin, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

arrow::Status write_batch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const fs::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

// NVFP4-quantizes a raw FP8 + F32-tile-scale tensor: GPU dequant to BF16
// then quantize_tensor_gpu("BF16").
arrow::Status quantize_fp8_nvfp4(const std::vector<uint8_t> &fp8,
                                 const std::vector<int64_t> &shape,
                                 const std::vector<uint8_t> &scale_f32,
                                 const std::vector<int64_t> &scale_shape,
                                 const std::string &name,
                                 DenseNvfp4Record &r)
{
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(
        DequantFp8TilesToBf16Gpu(fp8, shape, scale_f32, scale_shape, bf16));
    r.name = name;
    r.shape = shape;
    r.dtype = "BF16";
    if (!gpu::quantize_tensor_gpu(bf16.data(),
                                  static_cast<int64_t>(shape[0]) * shape[1],
                                  "BF16", nullptr, shape, {}, r.packed,
                                  r.scales, r.global_scale))
        return arrow::Status::IOError("GPU quantization failed for ", name);
    return arrow::Status::OK();
}

// Writes one expert's NVFP4 payload triple to experts/layer_<l>/exp_<e>.arrow.
arrow::Status write_expert_file(const fs::path &base_dir, int layer,
                                int expert, PackedExpert &&pe)
{
    const std::string l3 = absl::StrCat("layer_",
                                        absl::Dec(layer, absl::kZeroPad3));
    fs::path dir = base_dir / "experts" / l3;
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(
        *batch,
        dir / absl::StrCat("exp_", absl::Dec(expert, absl::kZeroPad3),
                           ".arrow"));
}

arrow::Status FillPackedExpert(int proj_id, DenseNvfp4Record r,
                               PackedExpert &pe)
{
    switch (proj_id) {
    case 1:
        pe.w1_packed = std::move(r.packed);
        pe.w1_scales = std::move(r.scales);
        pe.w1_shape = std::move(r.shape);
        pe.w1_global_scale = r.global_scale;
        return arrow::Status::OK();
    case 2:
        pe.w2_packed = std::move(r.packed);
        pe.w2_scales = std::move(r.scales);
        pe.w2_shape = std::move(r.shape);
        pe.w2_global_scale = r.global_scale;
        return arrow::Status::OK();
    case 3:
        pe.w3_packed = std::move(r.packed);
        pe.w3_scales = std::move(r.scales);
        pe.w3_shape = std::move(r.shape);
        pe.w3_global_scale = r.global_scale;
        return arrow::Status::OK();
    default:
        return arrow::Status::Invalid("bad expert projection id");
    }
}

// Synthetic fixtures (tests) may use tiny geometry; the real pipeline hard
// gates against the frozen constants.

} // namespace

namespace {

// VRAM policy for this family, derived from the converter's own classifier so
// the estimate cannot disagree with what the pipeline actually writes:
// `flash_tensor_is_nvfp4` decides the packed size, RoutedExpert decides the
// placement class.
VramTensorPolicy GlmFlashVramPolicy()
{
    VramTensorPolicy p;
    p.is_nvfp4 = [](const std::string &n) {
        return flash_tensor_is_nvfp4(classify_flash_tensor(n).kind);
    };
    p.mem_class = [](const std::string &n) {
        return classify_flash_tensor(n).kind == FlashTensorKind::RoutedExpert
                   ? VramTensorClass::RoutedExpert
                   : VramTensorClass::Resident;
    };
    return p;
}

// The converted artifact's residency: source shards minus the FP8 scale tiles
// (consumed by the requant) and the stripped vision tower. Geometry that the
// generic estimate cannot infer from tensor shapes:
//   - only the 11 DSA layers keep a per-position KV; the 34 KDA layers hold a
//     constant recurrent state, so num_attn_layers is the DSA count;
//   - a DSA layer stores K+V of [64 heads x 256 dims] FP16 per position, which
//     EstimateVram expresses as kv_dim * 2 (K,V) * 2 (bytes) = 65536 B.
VramProfile EstimateGlmFlashVram(const SafetensorsModel &model, int64_t ctx)
{
    std::vector<VramTensorInfo> inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            if (t.name.rfind(".weight_scale_inv") != std::string::npos)
                continue;
            if (classify_flash_tensor(t.name).kind == FlashTensorKind::Visual)
                continue;
            inventory.push_back({t.name, t.numel(), t.byte_size()});
        }
    using namespace frozen;
    VramGeometry vg;
    vg.hidden_dim = kHidden;
    vg.kv_dim = kHeads * kQkHeadDim;
    vg.num_attn_layers = kDsaLayerCount;
    vg.num_moe_layers = kLayers - kFirstKDense;
    vg.experts_per_layer = kNumExperts;
    vg.context_length = ctx;
    return EstimateVram(inventory, GlmFlashVramPolicy(), vg);
}

}  // namespace

arrow::Status ConvertGlm5NextModel(const std::string &input_dir,
                                   const std::string &output_root)
{
    // (1) Topology + frozen geometry gate.
    FlashTopology top = ParseFlashConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("GLM-5.3-Flash config error: ",
                                      top.error_msg);
    if (top.num_experts > 0 && top.top_k > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_experts");
    {
        std::string why;
        auto bad = [&why](const char *f, long long got, long long want) {
            why = absl::StrCat(f, ": got ", got, ", want ", want);
            return false;
        };
        using namespace frozen;
        if (!SyntheticGeometryAllowed())
        {
            if (top.hidden != kHidden) { bad("hidden_size", top.hidden, kHidden); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.layers != kLayers) { bad("num_hidden_layers", top.layers, kLayers); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.vocab != kVocab) { bad("vocab_size", top.vocab, kVocab); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.num_experts != kNumExperts) { bad("n_routed_experts", top.num_experts, kNumExperts); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.top_k != kTopK) { bad("num_experts_per_tok", top.top_k, kTopK); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.hc_count != kHcCount) { bad("hc_mult", top.hc_count, kHcCount); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.q_lora != kQLora) { bad("q_lora_rank", top.q_lora, kQLora); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.kv_lora != kKvLora) { bad("kv_lora_rank", top.kv_lora, kKvLora); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
            if (top.moe_intermediate != kMoeIntermediate) { bad("moe_intermediate_size", top.moe_intermediate, kMoeIntermediate); return arrow::Status::Invalid("frozen geometry mismatch: ", why); }
        }
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Classify the inventory: unknown names are hard errors, visual counted.
    // weight_scale_inv tensors are consumed as paired FP8 scale payloads,
    // never classified independently.
    int visual_skipped = 0;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) {
            if (t.name.rfind(".weight_scale_inv") != std::string::npos)
                continue;
            auto p = classify_flash_tensor(t.name);
            if (p.kind == FlashTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == FlashTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in glm5_next checkpoint: ", t.name);
        }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir, model));

    // Unified vault convention: <output_root>/logestix.
    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    for (const fs::path &d : {base, pinned_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    // Build a name->(shard, tensor) index; shards are streamed per pass.
    struct SrcRef {
        const SafetensorsModel::Shard *shard;
        const TensorInfo *info;
    };
    absl::flat_hash_map<std::string, SrcRef> by_name;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors) by_name[t.name] = {&shard, &t};

    auto scale_shape_of = [&](const std::string &scale_name)
        -> std::vector<int64_t> {
        auto it = by_name.find(scale_name);
        return it == by_name.end() ? std::vector<int64_t>{}
                                   : it->second.info->shape;
    };

    // Helper: fetch raw bytes of a tensor by name (empty when missing).
    auto load_raw = [&](const std::string &name) -> std::vector<uint8_t> {
        auto it = by_name.find(name);
        if (it == by_name.end()) return {};
        return read_tensor_data(input_dir, *it->second.shard,
                                *it->second.info);
    };

    // ── Globals: embed / lm_head / final norm (raw BF16). ──
    {
        std::vector<TensorRecord> global_raw;
        const char *globals[] = {
            "model.language_model.embed_tokens.weight", "lm_head.weight",
            "model.language_model.norm.weight"};
        for (const char *g : globals) {
            auto it = by_name.find(g);
            if (it == by_name.end())
                return arrow::Status::Invalid("missing global tensor: ", g);
            TensorRecord r;
            r.name = g;
            r.dtype = it->second.info->dtype;
            r.shape = it->second.info->shape;
            r.data = read_tensor_data(input_dir, *it->second.shard,
                                      *it->second.info);
            if (r.data.size() !=
                static_cast<size_t>(it->second.info->byte_size()))
                return arrow::Status::IOError("short read: ", g);
            global_raw.push_back(std::move(r));
        }
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(global_raw),
                                        pinned_dir / "global.arrow"));
    }

    // ── Experts pass (per MoE layer, bounded RAM): collect the three FP8
    // projections + F32 scale tiles per expert, dequant->requant on GPU,
    // write per-expert arrow files. ──
    for (int layer = frozen::kMoeFirstLayer; layer <= frozen::kMtpLayer;
         ++layer) {
        struct ExpertParts {
            std::vector<uint8_t> w1, w2, w3, s1, s2, s3;
            std::vector<int64_t> sh1, sh2, sh3, ssh1, ssh2, ssh3;
            bool has[6] = {};
            bool done = false;
            bool complete() const { return has[0] && has[1] && has[2]; }
        };
        absl::flat_hash_map<int, ExpertParts> experts;

        for (const auto &shard : model.shards) {
            for (const auto &t : shard.tensors) {
                auto p = classify_flash_tensor(t.name);
                if (p.layer != layer
                    || p.kind != FlashTensorKind::RoutedExpert) continue;
                if (t.name.rfind(".weight_scale_inv") != std::string::npos)
                    continue;  // scale payload, not the weight itself
                auto &ep = experts[p.expert];
                auto data = read_tensor_data(input_dir, shard, t);
                if (data.size() != static_cast<size_t>(t.byte_size()))
                    return arrow::Status::IOError("short read: ", t.name);
                const std::string scale_name =
                    t.name.substr(0, t.name.size() - strlen(".weight"))
                    + ".weight_scale_inv";
                auto scale = load_raw(scale_name);
                if (scale.empty())
                    return arrow::Status::Invalid("missing scale tile: ",
                                                  scale_name);
                auto ssh = scale_shape_of(scale_name);
                switch (p.proj_id) {
                case 1:
                    ep.w1 = std::move(data); ep.sh1 = t.shape;
                    ep.s1 = std::move(scale); ep.ssh1 = std::move(ssh);
                    ep.has[0] = true; break;
                case 2:
                    ep.w2 = std::move(data); ep.sh2 = t.shape;
                    ep.s2 = std::move(scale); ep.ssh2 = std::move(ssh);
                    ep.has[1] = true; break;
                case 3:
                    ep.w3 = std::move(data); ep.sh3 = t.shape;
                    ep.s3 = std::move(scale); ep.ssh3 = std::move(ssh);
                    ep.has[2] = true; break;
                default:
                    return arrow::Status::Invalid("bad expert proj id: ",
                                                  t.name);
                }
            }
            // Quantize + flush completed experts as the shard stream
            // provides their parts (bounded peak RAM).
            for (auto &[eid, ep] : experts) {
                if (!ep.complete() || ep.done) continue;
                PackedExpert pe;
                DenseNvfp4Record r1, r2, r3;
                ARROW_RETURN_NOT_OK(
                    quantize_fp8_nvfp4(ep.w1, ep.sh1, ep.s1, ep.ssh1,
                                       absl::StrCat("layer_", layer, ".e",
                                                    eid, ".w1"), r1));
                ARROW_RETURN_NOT_OK(FillPackedExpert(1, std::move(r1), pe));
                ARROW_RETURN_NOT_OK(
                    quantize_fp8_nvfp4(ep.w2, ep.sh2, ep.s2, ep.ssh2,
                                       absl::StrCat("layer_", layer, ".e",
                                                    eid, ".w2"), r2));
                ARROW_RETURN_NOT_OK(FillPackedExpert(2, std::move(r2), pe));
                ARROW_RETURN_NOT_OK(
                    quantize_fp8_nvfp4(ep.w3, ep.sh3, ep.s3, ep.ssh3,
                                       absl::StrCat("layer_", layer, ".e",
                                                    eid, ".w3"), r3));
                ARROW_RETURN_NOT_OK(FillPackedExpert(3, std::move(r3), pe));
                ARROW_RETURN_NOT_OK(write_expert_file(base, layer, eid,
                                                      std::move(pe)));
                ep.done = true;
                ep.w1.clear(); ep.w2.clear(); ep.w3.clear();
                ep.s1.clear(); ep.s2.clear(); ep.s3.clear();
            }
        }
        // Completeness gate: every expert of this layer flushed exactly once.
        for (const auto &[eid, ep] : experts)
            if (!ep.done)
                return arrow::Status::Invalid("incomplete expert: layer ",
                                              layer, " e", eid);
    }

    // ── Dense/attention pass (per layer): raw + NVFP4 non-expert tensors. ──
    for (int layer = 0; layer <= frozen::kMtpLayer; ++layer) {
        std::vector<TensorRecord> raw_recs;
        std::vector<DenseNvfp4Record> nvfp4_recs;

        for (const auto &shard : model.shards) {
            for (const auto &t : shard.tensors) {
                auto p = classify_flash_tensor(t.name);
                if (p.layer != layer) continue;
                if (p.kind == FlashTensorKind::Visual) continue;
                if (p.kind == FlashTensorKind::RoutedExpert) continue;
                if (t.name.rfind(".weight_scale_inv") != std::string::npos)
                    continue;  // scale payload, not the weight itself
                if (p.kind == FlashTensorKind::Embed
                    || p.kind == FlashTensorKind::LmHead
                    || p.kind == FlashTensorKind::FinalNorm)
                    continue;  // globals pass

                auto data = read_tensor_data(input_dir, shard, t);
                if (data.size() != static_cast<size_t>(t.byte_size()))
                    return arrow::Status::IOError("short read: ", t.name);

                if (flash_tensor_is_nvfp4(p.kind)) {
                    const std::string scale_name =
                        t.name.substr(0, t.name.size() - strlen(".weight"))
                        + ".weight_scale_inv";
                    DenseNvfp4Record r;
                    r.name = t.name;
                    r.shape = t.shape;
                    if (t.shape.size() != 2)
                        return arrow::Status::Invalid(
                            "NVFP4 tensor must be 2-D: ", t.name);
                    if (t.dtype == "F8_E4M3") {
                        // fp8 source with per-128x128 F32 scale tiles
                        // (experts/shared/DSA/dense-MLP projections).
                        auto scale = load_raw(scale_name);
                        if (scale.empty())
                            return arrow::Status::Invalid(
                                "missing scale tile: ", scale_name);
                        ARROW_RETURN_NOT_OK(quantize_fp8_nvfp4(
                            data, t.shape, scale, scale_shape_of(scale_name),
                            t.name, r));
                    } else {
                        // BF16 source without scale tiles (KDA q/k/v/o):
                        // quantize straight from BF16, same codec as the
                        // qwen4_exp GDN projections.
                        r.dtype = "BF16";
                        if (!gpu::quantize_tensor_gpu(
                                data.data(),
                                static_cast<int64_t>(t.shape[0]) * t.shape[1],
                                "BF16", nullptr, r.shape, {}, r.packed,
                                r.scales, r.global_scale))
                            return arrow::Status::IOError(
                                "GPU quantization failed for ", t.name);
                    }
                    nvfp4_recs.push_back(std::move(r));
                    continue;
                }

                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                raw_recs.push_back(std::move(r));
            }
        }

        const std::string l3 =
            absl::StrCat("layer_", absl::Dec(layer, absl::kZeroPad3));
        ARROW_RETURN_NOT_OK(write_batch(PinnedSchema::MakeBatch(raw_recs),
                                        pinned_dir / (l3 + ".arrow")));
        if (!nvfp4_recs.empty()) {
            ARROW_RETURN_NOT_OK(
                write_batch(DenseNvfp4Schema::MakeBatch(nvfp4_recs),
                            pinned_dir / (l3 + "_nvfp4.arrow")));
        }
    }

    lsxcommon::log::info(absl::StrCat(
        "GLM-5.3-Flash conversion complete: ", frozen::kLayers, " layers, ",
        frozen::kNumExperts, " experts, ", visual_skipped,
        " visual tensors skipped"));

    // Tokenizer: mandatory for real checkpoints (byte-level BPE, specials
    // from config: eos trio + pad).
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    if (fs::exists(tok_path)) {
        ARROW_RETURN_NOT_OK(
            integrity::BuildAndVerifyTokenizer(tok_path.string(), pinned_dir,
                                               integrity::TokenizerSpecialIds{
                                                   .bos_required = false,
                                                   .eos_required = true}));
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_dir);
    }

    // Structural manifest (index). Deterministic byte content.
    {
        using namespace frozen;
        std::ostringstream j;
        auto list = [](const int *ids, int n) {
            std::string s = "[";
            for (int i = 0; i < n; ++i) {
                if (i) s += ",";
                s += std::to_string(ids[i]);
            }
            return s + "]";
        };
        std::string moe;
        for (int l = kMoeFirstLayer; l <= kMtpLayer; ++l) {
            if (!moe.empty()) moe += ",";
            moe += std::to_string(l);
        }
        std::string dense;
        for (int l = 0; l < kMoeFirstLayer; ++l) {
            if (!dense.empty()) dense += ",";
            dense += std::to_string(l);
        }
        std::string eos;
        for (size_t i = 0; i < top.eos_ids.size(); ++i) {
            if (i) eos += ",";
            eos += std::to_string(top.eos_ids[i]);
        }
        j << "{\n"
          << "  \"format\": \"logestix-glm5next\",\n"
          << "  \"model\": \"GLM-5.3-Flash\",\n"
          << "  \"hidden_size\": " << top.hidden
          << ",\n  \"num_layers\": " << top.layers
          << ",\n  \"mtp_layer\": " << kMtpLayer
          << ",\n  \"vocab_size\": " << top.vocab
          << ",\n  \"num_attention_heads\": " << top.heads
          << ",\n  \"qk_head_dim\": " << top.qk_head_dim
          << ",\n  \"q_lora_rank\": " << top.q_lora
          << ",\n  \"kv_lora_rank\": " << top.kv_lora
          << ",\n  \"linear_conv_kernel_dim\": " << top.conv_kernel
          << ",\n  \"hc_count\": " << top.hc_count
          << ",\n  \"indexer_n_heads\": " << top.indexer_heads
          << ",\n  \"indexer_head_dim\": " << top.indexer_head_dim
          << ",\n  \"indexer_topk\": " << top.indexer_topk
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.top_k
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate
          << ",\n  \"routed_scaling_factor\": " << top.routed_scaling
          << ",\n  \"kda_layers\": " << list(kKdaLayerIds, kKdaLayerCount)
          << ",\n  \"dsa_layers\": " << list(kDsaLayerIds, kDsaLayerCount)
          << ",\n  \"moe_layers\": [" << moe << "]"
          << ",\n  \"dense_layers\": [" << dense << "]"
          << ",\n  \"eos_token_id\": [" << eos << "]"
          << ",\n  \"pad_token_id\": " << top.pad_id
          << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          // FormatVramJson renders the whole `"vram": {...}` member.
          << ",\n" 
          << FormatVramJson(EstimateGlmFlashVram(
                 model,
                 ReadContextLength((fs::path(input_dir) / "config.json").string())))
          << "\n}\n";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    return arrow::Status::OK();
}

arrow::Status RecomputeGlmFlashVram(const std::string &input_path,
                                    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    const fs::path index_path =
        fs::path(output_root) / "logestix" / "model.index.json";
    std::ifstream f(index_path);
    if (!f) return arrow::Status::IOError("cannot open ", index_path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);
    const int64_t ctx = ReadContextLength(
        (fs::path(input_path) / "config.json").string());
    const VramProfile profile = EstimateGlmFlashVram(model, ctx);
    lsxcommon::log::info(absl::StrCat(
        "GLM-5.3-Flash VRAM profile: min=",
        FormatBytesHuman(profile.min_bytes), " optimal=",
        FormatBytesHuman(profile.optimal_bytes), " max=",
        FormatBytesHuman(profile.max_bytes),
        " (ctx=", static_cast<long long>(profile.context_tokens), ")"));
    std::ofstream out(index_path, std::ios::trunc);
    if (!out)
        return arrow::Status::IOError("cannot write ", index_path.string());
    out << ReplaceVramBlock(json, profile);
    if (!out) return arrow::Status::IOError("write failed ", index_path.string());
    return arrow::Status::OK();
}

} // namespace lsxcommon::glm_moe_flash
