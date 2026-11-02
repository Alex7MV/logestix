// Mistral-Large-4-1T-A52B classifier + topology + converter (Model Ingestion
// Protocol stages 1/2). Names come from the source consolidated index; only the
// text decoder is converted (the pixtral vision tower / adapter / patch merger
// and the DSpark MTP head are stripped and counted).
//
// The source is FP8-E4M3 with F32 per-128x128 tile scales for every quantized
// linear (attention wq/wo, indexer wq_b, routed/shared experts, dense MLP) and
// BF16 for the attention KV projections, the indexer key/weights projections,
// the router, the norms and the embeddings/head. The converter dequantizes the
// FP8 side on the GPU and re-quantizes it to NVFP4; the BF16 side is copied
// verbatim.

#include "lsxcommon/mistral_moe_large4_convert.h"

#include "mistral_moe_large4_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/strip.h>

#include <simdjson.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace lsxcommon {
namespace mistral_large4 {
namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::mistral_large4::frozen;

int WeightIdFor(const std::string &proj)
{
    if (proj == "w1") return 1;
    if (proj == "w2") return 2;
    if (proj == "w3") return 3;
    return 0;
}

bool SplitLayerName(const std::string &name, int &layer, std::string &rest)
{
    const std::string kPrefix = "layers.";
    if (!absl::StartsWith(name, kPrefix)) return false;
    const size_t dot = name.find('.', kPrefix.size());
    if (dot == std::string::npos) return false;
    if (!absl::SimpleAtoi(name.substr(kPrefix.size(), dot - kPrefix.size()),
                          &layer))
        return false;
    rest = name.substr(dot + 1);
    return true;
}

// Parses "w{1,2,3}.{weight|qscale_weight}" into weight_id + companion.
bool ParseExpertProjection(const std::string &tail, int &weight_id,
                           std::string &companion)
{
    const size_t dot = tail.find('.');
    if (dot == std::string::npos) return false;
    weight_id = WeightIdFor(tail.substr(0, dot));
    if (weight_id == 0) return false;
    const std::string comp = tail.substr(dot + 1);
    if (comp == "weight") companion = "";
    else if (comp == "qscale_weight") companion = "scale";
    else return false;
    return true;
}

bool IsBallast(const std::string &name)
{
    return absl::StartsWith(name, "vision_encoder.") ||
           absl::StartsWith(name, "vision_language_adapter.") ||
           absl::StartsWith(name, "patch_merger.") ||
           name == "pre_mm_projector_norm.weight" ||
           absl::StartsWith(name, "mtp_dspark.");
}

bool IsMtp(const std::string &name)
{
    return absl::StartsWith(name, "mtp_dspark.");
}

// ── config.json (text_config) ──────────────────────────────────────────────
struct JsonText {
    std::string body;
    bool ok = false;
};

JsonText ReadFile(const fs::path &p)
{
    JsonText out;
    std::ifstream f(p);
    if (!f) return out;
    std::stringstream ss;
    ss << f.rdbuf();
    out.body = ss.str();
    out.ok = !out.body.empty();
    return out;
}

void ReadIntArray(simdjson::dom::element o, const char *key,
                  std::vector<int> &into)
{
    simdjson::dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return;
    for (auto v : arr)
    {
        int64_t x = 0;
        if (v.get_int64().get(x) == simdjson::SUCCESS)
            into.push_back(static_cast<int>(x));
    }
}

void ReadStringArrayAsBool(simdjson::dom::element o, const char *key,
                           const char *truthy, std::vector<int> &into)
{
    simdjson::dom::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return;
    for (auto v : arr)
    {
        std::string_view sv;
        if (v.get_string().get(sv) == simdjson::SUCCESS)
            into.push_back(sv == truthy ? 1 : 0);
    }
}

int GetInt(simdjson::dom::element o, const char *key, int dflt)
{
    int64_t v = 0;
    if (o[key].get_int64().get(v) == simdjson::SUCCESS)
        return static_cast<int>(v);
    return dflt;
}

} // namespace

Large4ParsedTensor mistral_large4_classify(const std::string &name)
{
    Large4ParsedTensor out;
    if (name == "tok_embeddings.weight" || name == "output.weight" ||
        name == "norm.weight" || name == "lerope.alpha")
    {
        out.kind = Large4TensorKind::Global;
        out.base = name;
        return out;
    }
    if (IsBallast(name))
    {
        out.kind = Large4TensorKind::Ballast;
        out.base = name;
        return out;
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayerName(name, layer, rest)) return out;
    out.layer = layer;
    out.base = rest;

    if (rest == "attention_norm.weight" || rest == "ffn_norm.weight" ||
        rest == "gate.weight")
    {
        out.kind = rest == "gate.weight" ? Large4TensorKind::Router
                                         : Large4TensorKind::LayerRaw;
        return out;
    }

    if (absl::StartsWith(rest, "experts."))
    {
        const size_t dot = rest.find('.', 8);
        if (dot == std::string::npos) return out;
        if (!absl::SimpleAtoi(rest.substr(8, dot - 8), &out.expert)) return out;
        int wid = 0;
        std::string comp;
        if (!ParseExpertProjection(rest.substr(dot + 1), wid, comp)) return out;
        out.kind = Large4TensorKind::RoutedExpert;
        out.weight_id = wid;
        out.companion = comp;
        out.base = absl::StrCat("experts.", out.expert, ".w", wid);
        return out;
    }
    if (absl::StartsWith(rest, "shared_experts."))
    {
        int wid = 0;
        std::string comp;
        if (!ParseExpertProjection(rest.substr(15), wid, comp)) return out;
        out.kind = Large4TensorKind::SharedExpert;
        out.weight_id = wid;
        out.companion = comp;
        out.base = absl::StrCat("shared_experts.w", wid);
        return out;
    }
    if (absl::StartsWith(rest, "feed_forward."))
    {
        int wid = 0;
        std::string comp;
        if (!ParseExpertProjection(rest.substr(13), wid, comp)) return out;
        out.kind = Large4TensorKind::DenseMlp;
        out.weight_id = wid;
        out.companion = comp;
        out.base = absl::StrCat("feed_forward.w", wid);
        return out;
    }
    if (rest == "attention.wq.weight" || rest == "attention.wq.qscale_weight" ||
        rest == "attention.wo.weight" || rest == "attention.wo.qscale_weight")
    {
        out.kind = Large4TensorKind::AttnNvfp4;
        out.companion = absl::EndsWith(rest, ".qscale_weight") ? "scale" : "";
        out.base = rest.substr(0, rest.size() -
                                      (out.companion.empty()
                                           ? std::string("weight").size()
                                           : std::string("qscale_weight").size()) -
                                      1);
        return out;
    }
    if (absl::StartsWith(rest, "attention.indexer.wq_b."))
    {
        out.kind = Large4TensorKind::IndexerNvfp4;
        out.companion =
            absl::EndsWith(rest, ".qscale_weight") ? "scale" : "";
        out.base = "attention.indexer.wq_b";
        return out;
    }
    if (rest == "attention.wkv_a_with_mqa.weight" ||
        rest == "attention.wkv_b.weight")
    {
        out.kind = Large4TensorKind::AttnRaw;
        return out;
    }
    if (rest == "attention.kv_a_norm.weight" ||
        rest == "attention.ssmax_scaling.s" ||
        rest == "attention.ssmax_scaling.b" ||
        rest == "attention.indexer.wk.weight" ||
        rest == "attention.indexer.weights_proj.weight" ||
        rest == "attention.indexer.q_input_norm.weight" ||
        rest == "attention.indexer.q_input_norm.bias" ||
        rest == "attention.indexer.k_norm.weight" ||
        rest == "attention.indexer.k_norm.bias" ||
        rest == "attention.indexer.ssmax_scaling.s" ||
        rest == "attention.indexer.ssmax_scaling.b")
    {
        out.kind = Large4TensorKind::LayerRaw;
        return out;
    }
    return out;
}

Large4Topology mistral_large4_load_topology(const std::string &config_path)
{
    Large4Topology top;
    const JsonText file = ReadFile(config_path);
    if (!file.ok)
    {
        top.error_msg = "cannot open " + config_path;
        return top;
    }
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(file.body).get(root) != simdjson::SUCCESS)
    {
        top.error_msg = "config.json is not valid JSON";
        return top;
    }
    std::string_view model_type;
    if (root["model_type"].get_string().get(model_type) != simdjson::SUCCESS ||
        model_type != "mistral_large4")
    {
        top.error_msg = "config.json model_type is not mistral_large4";
        return top;
    }
    simdjson::dom::object tc;
    if (root["text_config"].get_object().get(tc) != simdjson::SUCCESS)
    {
        top.error_msg = "config.json has no text_config";
        return top;
    }

    top.hidden = GetInt(tc, "hidden_size", 0);
    top.layers = GetInt(tc, "num_hidden_layers", 0);
    top.vocab = GetInt(tc, "vocab_size", 0);
    top.heads = GetInt(tc, "num_attention_heads", 0);
    top.kv_lora = GetInt(tc, "kv_lora_rank", 0);
    top.qk_nope = GetInt(tc, "qk_nope_head_dim", 0);
    top.qk_rope = GetInt(tc, "qk_rope_head_dim", 0);
    top.v_head = GetInt(tc, "v_head_dim", 0);
    top.num_experts = GetInt(tc, "n_routed_experts", 0);
    top.top_k = GetInt(tc, "num_experts_per_tok", 0);
    top.moe_intermediate = GetInt(tc, "moe_intermediate_size", 0);
    top.dense_intermediate = GetInt(tc, "intermediate_size", 0);
    top.index_heads = GetInt(tc, "index_n_heads", 0);
    top.index_head_dim = GetInt(tc, "index_head_dim", 0);
    top.index_topk = GetInt(tc, "index_topk", 0);
    top.context_length = GetInt(tc, "max_position_embeddings", 0);

    ReadIntArray(tc, "no_rope_layers", top.rope_layers);
    ReadIntArray(tc, "kv_source_layers", top.kv_source);
    ReadStringArrayAsBool(tc, "indexer_types", "full", top.indexer_full);
    ReadStringArrayAsBool(tc, "mlp_layer_types", "sparse", top.moe_layers);

    if (top.hidden <= 0 || top.layers <= 0 || top.vocab <= 0)
    {
        top.error_msg = "text_config missing required geometry";
        return top;
    }
    if (static_cast<int>(top.rope_layers.size()) != top.layers ||
        static_cast<int>(top.kv_source.size()) != top.layers ||
        static_cast<int>(top.indexer_full.size()) != top.layers ||
        static_cast<int>(top.moe_layers.size()) != top.layers)
    {
        top.error_msg = "text_config per-layer arrays do not match num_hidden_layers";
        return top;
    }
    for (int i = 0; i < top.layers; ++i)
    {
        if (top.rope_layers[static_cast<size_t>(i)]) continue;
        top.nope_layers.push_back(i);
    }
    for (int i = 0; i < top.layers; ++i)
        if (!top.moe_layers[static_cast<size_t>(i)])
            top.dense_layers.push_back(i);

    if (!SyntheticGeometryAllowed())
    {
        std::vector<std::string> bad;
        const auto chk = [&bad](const char *name, int got, int want) {
            if (got != want)
                bad.push_back(absl::StrFormat("%s=%d!=%d", name, got, want));
        };
        chk("hidden", top.hidden, frozen::kHidden);
        chk("layers", top.layers, frozen::kLayers);
        chk("vocab", top.vocab, frozen::kVocab);
        chk("heads", top.heads, frozen::kHeads);
        chk("kv_lora", top.kv_lora, frozen::kKvLora);
        chk("qk_nope", top.qk_nope, frozen::kQkNope);
        chk("qk_rope", top.qk_rope, frozen::kQkRope);
        chk("v_head", top.v_head, frozen::kVHead);
        chk("num_experts", top.num_experts, frozen::kNumRouted);
        chk("top_k", top.top_k, frozen::kTopK);
        chk("moe_intermediate", top.moe_intermediate, frozen::kMoeIntermediate);
        chk("dense_intermediate", top.dense_intermediate,
            frozen::kDenseIntermediate);
        chk("index_n_heads", top.index_heads, frozen::kIndexHeads);
        chk("index_head_dim", top.index_head_dim, frozen::kIndexHeadDim);
        chk("index_topk", top.index_topk, frozen::kIndexTopk);
        chk("first_k_dense", top.layers - static_cast<int>(top.dense_layers.size()),
            frozen::kFirstKDense);
        for (int i = 0; i < top.layers; ++i)
        {
            chk("rope_layers[i]", top.rope_layers[static_cast<size_t>(i)],
                frozen::kRopeLayer[static_cast<size_t>(i)]);
            chk("kv_source[i]", top.kv_source[static_cast<size_t>(i)],
                frozen::kKvSource[static_cast<size_t>(i)]);
            chk("indexer_full[i]", top.indexer_full[static_cast<size_t>(i)],
                frozen::kIndexerFull[static_cast<size_t>(i)]);
            chk("moe_layers[i]", top.moe_layers[static_cast<size_t>(i)],
                frozen::kMoeLayer[static_cast<size_t>(i)]);
        }
        if (!bad.empty())
            top.error_msg = absl::StrCat(
                "Mistral-Large-4 frozen geometry mismatch: ",
                absl::StrJoin(bad, ", "));
    }
    return top;
}

namespace {

// ── source access ──────────────────────────────────────────────────────────
struct TensorRef {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
    bool valid() const { return shard && info; }
};

std::vector<uint8_t> ReadTensorData(const std::string &input_dir,
                                    const TensorRef &r)
{
    if (!r.valid()) return {};
    std::ifstream sf((fs::path(input_dir) / r.shard->filename).string(),
                     std::ios::binary);
    if (!sf) return {};
    uint64_t hdr_len = 0;
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    sf.seekg(data_off + r.info->data_begin, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(r.info->byte_size()));
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

arrow::Status RawRecord(const std::string &input_dir, const TensorRef &r,
                        TensorRecord &out)
{
    const auto raw = ReadTensorData(input_dir, r);
    if (raw.empty() || static_cast<int64_t>(raw.size()) != r.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      r.info->name);
    out.name = r.info->name;
    out.dtype = r.info->dtype;
    out.shape = r.info->shape;
    out.data = raw;
    return arrow::Status::OK();
}

// FP8-E4M3 + F32 128x128 tile scale -> NVFP4 record.
arrow::Status QuantizeFp8Pair(const std::string &input_dir, const TensorRef &w,
                              const TensorRef &s, const std::string &name,
                              DenseNvfp4Record &out)
{
    if (!w.valid() || !s.valid())
        return arrow::Status::Invalid("missing FP8/scale pair for ", name);
    if (w.info->dtype != "F8_E4M3" || s.info->dtype != "F32")
        return arrow::Status::Invalid("unexpected dtype pair for ", name, ": ",
                                      w.info->dtype, "/", s.info->dtype);
    const auto fp8 = ReadTensorData(input_dir, w);
    const auto scale = ReadTensorData(input_dir, s);
    if (fp8.empty() || scale.empty())
        return arrow::Status::IOError("short read for ", name);
    std::vector<uint8_t> bf16;
    ARROW_RETURN_NOT_OK(gpu::DequantFp8Tile128ToBf16Gpu(
        fp8, w.info->shape, scale, s.info->shape, bf16));
    int64_t numel = 1;
    for (int64_t d : w.info->shape) numel *= d;
    out.name = name;
    out.shape = w.info->shape;
    out.dtype = "BF16";
    if (!gpu::quantize_tensor_gpu(bf16.data(), numel, "BF16", nullptr,
                                  w.info->shape, {}, out.packed, out.scales,
                                  out.global_scale))
        return arrow::Status::ExecutionError("GPU quantize failed for ", name);
    return arrow::Status::OK();
}

std::string LayerFile(int layer, const char *suffix)
{
    return absl::StrCat("layer_", layer, suffix);
}

arrow::Status WriteBatchOr(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &b,
    const fs::path &path)
{
    if (!b.ok()) return b.status();
    return WriteRecordBatchToFile(*b, path);
}

} // namespace

arrow::Status ConvertMistralLarge4Model(const std::string &input_dir,
                                        const fs::path &output_root)
{
    if (input_dir.empty() || output_root.empty())
        return arrow::Status::Invalid("input/output path not set");

    const Large4Topology top =
        mistral_large4_load_topology((fs::path(input_dir) / "config.json").string());
    if (!top.error_msg.empty())
        return arrow::Status::Invalid(top.error_msg);

    auto model = scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards in ", input_dir);

    absl::flat_hash_map<std::string, TensorRef> refs;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            refs.insert_or_assign(t.name, TensorRef{&shard, &t});

    // layer -> ordered raw (BF16) tensor names
    std::map<int, std::vector<std::string>> raw_names;
    // layer -> weight name -> {weight ref, scale ref}; order-independent
    struct PairRefs {
        TensorRef w;
        TensorRef s;
    };
    std::map<int, std::map<std::string, PairRefs>> nvfp4_pairs;
    std::map<int, std::set<int>> routed_experts;
    std::vector<std::string> global_names;
    std::vector<std::string> skipped;
    int mtp_tensors = 0;

    for (const auto &[name, ref] : refs)
    {
        const auto p = mistral_large4_classify(name);
        switch (p.kind)
        {
        case Large4TensorKind::Unknown:
            return arrow::Status::Invalid("unrecognized tensor: ", name);
        case Large4TensorKind::Ballast:
            if (IsMtp(name)) ++mtp_tensors;
            else skipped.push_back(name);
            break;
        case Large4TensorKind::Global:
            global_names.push_back(name);
            break;
        case Large4TensorKind::LayerRaw:
        case Large4TensorKind::AttnRaw:
        case Large4TensorKind::Router:
            raw_names[p.layer].push_back(name);
            break;
        case Large4TensorKind::AttnNvfp4:
        case Large4TensorKind::IndexerNvfp4:
        case Large4TensorKind::SharedExpert:
        case Large4TensorKind::DenseMlp:
        {
            auto &slot = nvfp4_pairs[p.layer][p.base];
            if (p.companion.empty()) slot.w = ref;
            else slot.s = ref;
            break;
        }
        case Large4TensorKind::RoutedExpert:
        {
            routed_experts[p.layer].insert(p.expert);
            break;
        }
        }
    }

    const fs::path base = output_root / "logestix";
    const fs::path pinned_dir = base / "pinned";
    std::error_code ec;
    fs::create_directories(pinned_dir, ec);

    // ── globals ────────────────────────────────────────────────────────────
    std::vector<TensorRecord> global_raw;
    for (const auto &name : global_names)
    {
        TensorRecord r;
        ARROW_RETURN_NOT_OK(RawRecord(input_dir, refs[name], r));
        global_raw.push_back(std::move(r));
    }
    if (global_raw.empty())
        return arrow::Status::Invalid("no globals in ", input_dir);
    ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(global_raw),
                                     pinned_dir / "global.arrow"));

    // ── per-layer pinned edges + NVFP4 projections ─────────────────────────
    int total_experts = 0;
    for (int l = 0; l < top.layers; ++l)
    {
        std::vector<TensorRecord> raw;
        for (const auto &name : raw_names[l])
        {
            TensorRecord r;
            ARROW_RETURN_NOT_OK(RawRecord(input_dir, refs[name], r));
            raw.push_back(std::move(r));
        }
        if (!raw.empty())
            ARROW_RETURN_NOT_OK(WriteBatchOr(
                PinnedSchema::MakeBatch(raw),
                pinned_dir / LayerFile(l, ".arrow")));

        std::vector<DenseNvfp4Record> dense;
        for (const auto &[base_name, pair] : nvfp4_pairs[l])
        {
            // The recorded name is the full source weight name.
            const std::string wname = absl::StrCat("layers.", l, ".", base_name,
                                                   ".weight");
            DenseNvfp4Record rec;
            ARROW_RETURN_NOT_OK(
                QuantizeFp8Pair(input_dir, pair.w, pair.s, wname, rec));
            dense.push_back(std::move(rec));
        }
        if (!dense.empty())
            ARROW_RETURN_NOT_OK(WriteBatchOr(
                DenseNvfp4Schema::MakeBatch(dense),
                pinned_dir / LayerFile(l, "_nvfp4.arrow")));

        // Routed experts: one ExpertSchema file per (layer, expert).
        if (!routed_experts[l].empty())
        {
            const fs::path dir = base / "experts" / LayerFile(l, "");
            fs::create_directories(dir, ec);
            for (const int eid : routed_experts[l])
            {
                const std::string prefix =
                    absl::StrCat("layers.", l, ".experts.", eid, ".");
                const auto pair_for = [&](const char *w)
                    -> std::optional<std::pair<TensorRef, TensorRef>> {
                    const auto it = refs.find(absl::StrCat(prefix, w, ".weight"));
                    const auto its =
                        refs.find(absl::StrCat(prefix, w, ".qscale_weight"));
                    if (it == refs.end() || its == refs.end())
                        return std::nullopt;
                    return std::make_pair(it->second, its->second);
                };
                const auto w1 = pair_for("w1");
                const auto w2 = pair_for("w2");
                const auto w3 = pair_for("w3");
                if (!w1 || !w2 || !w3)
                    return arrow::Status::Invalid(
                        "incomplete expert ", l, "/", eid);
                DenseNvfp4Record r1, r2, r3;
                ARROW_RETURN_NOT_OK(QuantizeFp8Pair(
                    input_dir, w1->first, w1->second,
                    absl::StrCat(prefix, "w1.weight"), r1));
                ARROW_RETURN_NOT_OK(QuantizeFp8Pair(
                    input_dir, w2->first, w2->second,
                    absl::StrCat(prefix, "w2.weight"), r2));
                ARROW_RETURN_NOT_OK(QuantizeFp8Pair(
                    input_dir, w3->first, w3->second,
                    absl::StrCat(prefix, "w3.weight"), r3));
                PackedExpert pe;
                pe.w1_packed = std::move(r1.packed);
                pe.w1_scales = std::move(r1.scales);
                pe.w1_global_scale = r1.global_scale;
                pe.w2_packed = std::move(r2.packed);
                pe.w2_scales = std::move(r2.scales);
                pe.w2_global_scale = r2.global_scale;
                pe.w3_packed = std::move(r3.packed);
                pe.w3_scales = std::move(r3.scales);
                pe.w3_global_scale = r3.global_scale;
                pe.w1_shape = {top.moe_intermediate, top.hidden};
                pe.w3_shape = {top.moe_intermediate, top.hidden};
                pe.w2_shape = {top.hidden, top.moe_intermediate};
                ARROW_RETURN_NOT_OK(WriteBatchOr(
                    ExpertSchema::MakeBatchView(pe),
                    dir / absl::StrCat("exp_", eid, ".arrow")));
                ++total_experts;
            }
        }
    }

    // ── tokenizer (byte-level BPE, o200k grouped digits) ───────────────────
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    if (fs::exists(tok_path))
    {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 10;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = true,
                                           .eos_required = true,
                                           .pad_required = true}, tok_policy));
    }
    else if (!SyntheticGeometryAllowed())
    {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // ── index ──────────────────────────────────────────────────────────────
    auto write_array = [](std::ofstream &f, const char *key,
                          const std::vector<int> &v, bool last) {
        f << "  \"" << key << "\": [";
        for (size_t i = 0; i < v.size(); ++i) f << (i ? "," : "") << v[i];
        f << "]" << (last ? "" : ",") << "\n";
    };

    std::ofstream f(base / "model.index.json");
    if (!f) return arrow::Status::IOError("cannot write model.index.json");
    f << "{\n";
    f << "  \"format\": \"" << frozen::kFormat << "\",\n";
    f << "  \"model\": \"" << frozen::kModelName << "\",\n";
    f << "  \"num_layers\": " << top.layers << ",\n";
    f << "  \"experts_per_layer\": " << top.num_experts << ",\n";
    f << "  \"hidden_size\": " << top.hidden << ",\n";
    f << "  \"vocab_size\": " << top.vocab << ",\n";
    f << "  \"num_attention_heads\": " << top.heads << ",\n";
    f << "  \"kv_lora_rank\": " << top.kv_lora << ",\n";
    f << "  \"qk_nope_head_dim\": " << top.qk_nope << ",\n";
    f << "  \"qk_rope_head_dim\": " << top.qk_rope << ",\n";
    f << "  \"v_head_dim\": " << top.v_head << ",\n";
    f << "  \"moe_intermediate_size\": " << top.moe_intermediate << ",\n";
    f << "  \"dense_intermediate_size\": " << top.dense_intermediate << ",\n";
    f << "  \"num_experts_per_tok\": " << top.top_k << ",\n";
    f << "  \"index_n_heads\": " << top.index_heads << ",\n";
    f << "  \"index_head_dim\": " << top.index_head_dim << ",\n";
    f << "  \"index_topk\": " << top.index_topk << ",\n";
    f << "  \"context_length\": " << top.context_length << ",\n";
    f << "  \"bos_token_id\": " << frozen::kBosId << ",\n";
    f << "  \"eos_token_id\": " << frozen::kEosId << ",\n";
    f << "  \"pad_token_id\": " << frozen::kPadId << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    write_array(f, "rope_layers", top.rope_layers, false);
    write_array(f, "kv_source_layers", top.kv_source, false);
    write_array(f, "indexer_layers", top.indexer_full, false);
    write_array(f, "moe_layers", top.moe_layers, false);
    write_array(f, "dense_layers", top.dense_layers, false);
    f << "  \"deferred\": {\"mtp_tensors\": " << mtp_tensors << "},\n";
    f << "  \"skipped_tensors\": [";
    for (size_t i = 0; i < skipped.size(); ++i)
        f << (i ? ", " : "") << "\"" << skipped[i] << "\"";
    f << "]\n";
    f << "}\n";
    f.close();

    log::info(absl::StrCat("[mistral-large4] converted ", top.layers,
                           " layers, ", total_experts, " routed experts, ",
                           skipped.size(), " vision tensors stripped, ",
                           mtp_tensors, " MTP tensors deferred"));
    return arrow::Status::OK();
}

} // namespace mistral_large4
} // namespace lsxcommon
