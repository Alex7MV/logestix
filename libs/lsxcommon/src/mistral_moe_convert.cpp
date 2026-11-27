// Mistral-Small-4-119B-2603 classifier + topology + converter
// (Model Ingestion Protocol step 1). Names are parsed straight out of the
// source consolidated index; only the text decoder is converted (the vision
// tower / patch merger / vision_language_adapter are stripped ballast).

#include "lsxcommon/mistral_moe_convert.h"

#include "mistral_moe_constants.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace lsxcommon {
namespace mistral_moe {
namespace {

namespace fs = std::filesystem;
namespace frozen = lsxcommon::mistral_moe::frozen;

int WeightIdFor(const std::string &proj)
{
    if (proj == "w1") return 1;   // gate
    if (proj == "w2") return 2;   // down
    if (proj == "w3") return 3;   // up
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

// Parses "w{1,2,3}.{weight_packed|weight_scale|weight_global_scale|
// input_global_scale}" into weight_id + companion suffix.
bool ParseExpertProjection(const std::string &tail, int &weight_id,
                           std::string &companion)
{
    const size_t dot = tail.find('.');
    if (dot == std::string::npos) return false;
    weight_id = WeightIdFor(tail.substr(0, dot));
    if (weight_id == 0) return false;
    const std::string comp = tail.substr(dot + 1);
    if (comp == "weight_packed") companion = "";
    else if (comp == "weight_scale") companion = "scale";
    else if (comp == "weight_global_scale") companion = "global_scale";
    else if (comp == "input_global_scale") companion = "input_global_scale";
    else return false;
    return true;
}

std::optional<MistralParsedTensor> ParseRoutedExpert(const std::string &rest)
{
    const std::string kP = "experts.";
    if (!absl::StartsWith(rest, kP)) return std::nullopt;
    const size_t dot = rest.find('.', kP.size());
    if (dot == std::string::npos) return std::nullopt;
    MistralParsedTensor p;
    if (!absl::SimpleAtoi(rest.substr(kP.size(), dot - kP.size()), &p.expert))
        return std::nullopt;
    int wid = 0;
    std::string comp;
    if (!ParseExpertProjection(rest.substr(dot + 1), wid, comp))
        return std::nullopt;
    p.kind = MistralTensorKind::RoutedExpert;
    p.weight_id = wid;
    p.companion = comp;
    p.base = absl::StrCat("experts.", p.expert, ".w", wid);
    return p;
}

std::optional<MistralParsedTensor> ParseSharedExpert(const std::string &rest)
{
    const std::string kP = "shared_experts.";
    if (!absl::StartsWith(rest, kP)) return std::nullopt;
    int wid = 0;
    std::string comp;
    if (!ParseExpertProjection(rest.substr(kP.size()), wid, comp))
        return std::nullopt;
    MistralParsedTensor p;
    p.kind = MistralTensorKind::SharedExpert;
    p.weight_id = wid;
    p.companion = comp;
    p.base = absl::StrCat("shared_experts.w", wid);
    return p;
}

std::optional<MistralParsedTensor> ParseAttention(const std::string &rest)
{
    const std::string kP = "attention.";
    if (!absl::StartsWith(rest, kP)) return std::nullopt;
    static const std::vector<std::string> kNorms = {"q_a_norm.weight",
                                                    "kv_a_norm.weight"};
    static const std::vector<std::string> kProjs = {
        "wq_a", "wq_b", "wkv_a_with_mqa", "wkv_b", "wo"};
    const std::string sub = rest.substr(kP.size());
    for (const auto &n : kNorms)
        if (sub == n)
        {
            MistralParsedTensor p;
            p.kind = MistralTensorKind::LayerNorm;
            p.base = rest;
            return p;
        }
    for (const auto &proj : kProjs)
        if (sub == absl::StrCat(proj, ".weight"))
        {
            MistralParsedTensor p;
            p.kind = MistralTensorKind::Attention;
            p.base = proj;
            return p;
        }
    return std::nullopt;
}

bool IsVisionPrefix(const std::string &name)
{
    return absl::StartsWith(name, "vision_encoder.") ||
           absl::StartsWith(name, "vision_language_adapter.") ||
           absl::StartsWith(name, "patch_merger.") ||
           name == "pre_mm_projector_norm.weight";
}

// ── index parsing helpers (shared with the loader's cross-check) ────────────
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

struct TensorRef {
    const SafetensorsModel::Shard *shard = nullptr;
    const TensorInfo *info = nullptr;
};

arrow::Status RawRecord(const std::string &input_dir, const TensorRef &w,
                        TensorRecord &out)
{
    const auto raw = ReadTensorData(input_dir, *w.shard, *w.info);
    if (raw.empty() || static_cast<int64_t>(raw.size()) != w.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      w.info->name);
    out.name = w.info->name;
    out.dtype = w.info->dtype;
    out.shape = w.info->shape;
    out.data = raw;
    return arrow::Status::OK();
}

// One already-quantized expert projection as read from the source shards:
// U8 E2M1 nibbles + F8_E4M3 per-16 block scales + an F32 per-tensor global.
struct QuantProjection {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    float global_scale = 1.0f;
    std::vector<int64_t> packed_shape;
    std::vector<int64_t> scale_shape;
};

arrow::Status ReadQuantProjection(const std::string &input_dir,
                                  const TensorRef &ref, QuantProjection &out)
{
    const auto raw = ReadTensorData(input_dir, *ref.shard, *ref.info);
    if (raw.empty() || static_cast<int64_t>(raw.size()) != ref.info->byte_size())
        return arrow::Status::IOError("failed to read tensor data: ",
                                      ref.info->name);
    out.packed = raw;
    out.packed_shape = ref.info->shape;
    return arrow::Status::OK();
}

} // namespace

MistralParsedTensor mistral_classify(const std::string &name)
{
    MistralParsedTensor out;
    if (name == "tok_embeddings.weight" || name == "output.weight" ||
        name == "norm.weight")
    {
        out.kind = MistralTensorKind::Global;
        out.base = name;
        return out;
    }
    if (IsVisionPrefix(name))
    {
        out.kind = MistralTensorKind::Vision;
        out.base = name;
        return out;
    }

    int layer = -1;
    std::string rest;
    if (!SplitLayerName(name, layer, rest)) return out;

    if (rest == "attention_norm.weight" || rest == "ffn_norm.weight")
    {
        out.kind = MistralTensorKind::LayerNorm;
        out.layer = layer;
        out.base = rest;
        return out;
    }
    if (rest == "gate.weight")
    {
        out.kind = MistralTensorKind::Router;
        out.layer = layer;
        out.base = rest;
        return out;
    }
    if (auto p = ParseRoutedExpert(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseSharedExpert(rest))
    {
        p->layer = layer;
        return *p;
    }
    if (auto p = ParseAttention(rest))
    {
        p->layer = layer;
        return *p;
    }
    return out;
}

VramTensorPolicy MistralVramPolicy()
{
    VramTensorPolicy policy;
    // Already-NVFP4 source: on-disk bytes == converted bytes (see the header).
    policy.is_nvfp4 = nullptr;
    policy.has_residual = nullptr;
    policy.mem_class = +[](const std::string &name) {
        return mistral_classify(name).kind == MistralTensorKind::RoutedExpert
                   ? VramTensorClass::RoutedExpert
                   : VramTensorClass::Resident;
    };
    return policy;
}

MistralTopology mistral_load_topology(const std::string &index_json_path)
{    MistralTopology top;
    std::ifstream f(index_json_path);
    if (!f)
    {
        top.error_msg = "cannot open " + index_json_path;
        return top;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    if (text.find("\"moe_layers\"") != std::string::npos)
    {
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
                const std::string t{
                    std::string(absl::StripAsciiWhitespace(part))};
                if (!t.empty())
                    into.push_back(std::strtoll(t.c_str(), nullptr, 10));
            }
        };
        top.num_layers = JsonInt(text, "num_layers", 0);
        top.experts_per_layer = JsonInt(text, "experts_per_layer", 0);
        scan_lists("moe_layers", top.moe_layers);
        scan_lists("dense_layers", top.dense_layers);
        std::sort(top.moe_layers.begin(), top.moe_layers.end());
        std::sort(top.dense_layers.begin(), top.dense_layers.end());
        // Placement profile. Optional on a tree converted before the block
        // became mandatory, so a malformed/missing block leaves the default
        // empty profile instead of failing the load.
        {
            VramProfile vp;
            std::string verr;
            if (ParseVramBlock(text, vp, verr)) top.vram = vp;
        }
        // layout is an integer; the old string tag is gone (see index_layout.h).
        {
            const size_t fp = text.find("\"layout\"");
            if (fp != std::string::npos) {
                const size_t c = text.find(':', fp);
                if (c != std::string::npos)
                    top.layout = std::atoll(text.c_str() + c + 1);
            }
        }
        return top;
    }

    // HF-style / consolidated index: walk the weight_map keys.
    std::set<long long> moe;
    long long experts = 0;
    size_t pos = 0;
    while ((pos = text.find("\"layers.", pos)) != std::string::npos)
    {
        const size_t key_start = pos + 1;
        const size_t key_end = text.find('"', key_start + 8);
        if (key_end == std::string::npos) break;
        const std::string key = text.substr(key_start, key_end - key_start);
        pos = key_end;
        const auto t = mistral_classify(key);
        if (t.kind == MistralTensorKind::RoutedExpert)
        {
            moe.insert(t.layer);
            experts = std::max(experts, static_cast<long long>(t.expert) + 1);
        }
    }
    if (moe.empty())
    {
        top.error_msg = "no decoder layers found in " + index_json_path;
        return top;
    }
    top.moe_layers.assign(moe.begin(), moe.end());
    top.experts_per_layer = experts;
    top.num_layers = static_cast<long long>(top.moe_layers.size());

    const bool relaxed = lsxcommon::SyntheticGeometryAllowed();
    if (!relaxed && (top.num_layers != frozen::kLayers ||
                     top.experts_per_layer != frozen::kNumRouted))
    {
        top.error_msg = absl::StrCat(
            "frozen geometry mismatch: layers=", top.num_layers, " (expected ",
            frozen::kLayers, "), experts=", top.experts_per_layer, " (expected ",
            frozen::kNumRouted, ")");
    }
    return top;
}

namespace {

std::string LayerFile(int layer, const char *suffix)
{
    return absl::StrCat("layer_", layer, suffix);
}

arrow::Status WriteBatchOr(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &b,
    const std::filesystem::path &path)
{
    if (!b.ok()) return b.status();
    return WriteRecordBatchToFile(*b, path);
}

// Build one NVFP4 PackedExpert from three already-quantized projections.
// w1/w3 are [I, H], w2 is [H, I] (the caller passes the logical shapes so the
// loader/kernels can size the GEMMs; the packed bytes are I*H/2 etc).
arrow::Status BuildPackedExpert(const QuantProjection &w1,
                                const QuantProjection &w2,
                                const QuantProjection &w3,
                                int64_t I, int64_t H, PackedExpert &out)
{
    out.w1_packed = w1.packed;
    out.w1_scales = w1.scales;
    out.w1_global_scale = w1.global_scale;
    out.w2_packed = w2.packed;
    out.w2_scales = w2.scales;
    out.w2_global_scale = w2.global_scale;
    out.w3_packed = w3.packed;
    out.w3_scales = w3.scales;
    out.w3_global_scale = w3.global_scale;
    out.w1_shape = {I, H};
    out.w3_shape = {I, H};
    out.w2_shape = {H, I};
    return arrow::Status::OK();
}

} // namespace

arrow::Status ConvertMistralModel(const std::string &input_dir,
                                  const std::filesystem::path &output_root)
{
    if (input_dir.empty() || output_root.empty())
        return arrow::Status::Invalid("input/output path not set");

    const auto params_path = fs::path(input_dir) / "params.json";
    if (!fs::exists(params_path))
        return arrow::Status::Invalid("params.json not found in ", input_dir);
    std::ifstream pf(params_path);
    if (!pf) return arrow::Status::IOError("cannot open ", params_path.string());
    std::string pt((std::istreambuf_iterator<char>(pf)),
                   std::istreambuf_iterator<char>());
    const long long hidden = JsonInt(pt, "dim", 0);
    const long long layers = JsonInt(pt, "n_layers", 0);
    const long long vocab = JsonInt(pt, "vocab_size", 0);
    const long long n_heads = JsonInt(pt, "n_heads", 0);
    const long long q_lora = JsonInt(pt, "q_lora_rank", 0);
    const long long kv_lora = JsonInt(pt, "kv_lora_rank", 0);
    const long long qk_nope = JsonInt(pt, "qk_nope_head_dim", 0);
    const long long qk_rope = JsonInt(pt, "qk_rope_head_dim", 0);
    const long long v_head = JsonInt(pt, "v_head_dim", 0);
    const long long num_experts = JsonInt(pt, "num_experts", 0);
    const long long top_k = JsonInt(pt, "num_experts_per_tok", 0);
    const long long expert_hidden = JsonInt(pt, "expert_hidden_dim", 0);
    if (hidden <= 0 || layers <= 0 || vocab <= 0)
        return arrow::Status::Invalid("params.json missing required geometry");

    if (!SyntheticGeometryAllowed())
    {
        std::vector<std::string> bad;
        const auto chk = [&bad](const char *name, long long got, long long want) {
            if (got != want)
                bad.push_back(absl::StrFormat("%s=%lld!=%lld", name, got, want));
        };
        chk("hidden", hidden, frozen::kHidden);
        chk("layers", layers, frozen::kLayers);
        chk("vocab", vocab, frozen::kVocab);
        chk("n_heads", n_heads, frozen::kHeads);
        chk("q_lora", q_lora, frozen::kQLoraRank);
        chk("kv_lora", kv_lora, frozen::kKvLoraRank);
        chk("qk_nope", qk_nope, frozen::kQkNopeHeadDim);
        chk("qk_rope", qk_rope, frozen::kQkRopeHeadDim);
        chk("v_head", v_head, frozen::kVHeadDim);
        chk("num_experts", num_experts, frozen::kNumRouted);
        chk("top_k", top_k, frozen::kTopK);
        chk("expert_hidden", expert_hidden, frozen::kMoeIntermediate);
        if (!bad.empty())
            return arrow::Status::Invalid(
                "Mistral frozen geometry mismatch: ", absl::StrJoin(bad, ", "),
                " (set LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 only for synthetic "
                "tests)");
    }

    auto model = scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    absl::flat_hash_map<std::string, TensorRef> refs;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors)
            refs.insert_or_assign(t.name, TensorRef{&shard, &t});

    // Bucket the tensors. Unknown text-decoder tensors are a hard error;
    // vision ballast is stripped + counted.
    struct ExpertSide {
        QuantProjection w1, w2, w3;
    };
    absl::flat_hash_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::flat_hash_map<int, std::vector<TensorRecord>> expert_raw_by_layer;
    std::vector<TensorRecord> global_raw;
    std::vector<TensorRecord> skipped;
    // What the converted tree actually holds: stripped ballast and the
    // activation-only `input_global_scale` companions are excluded here, so the
    // VRAM profile describes the artifact rather than the source checkpoint.
    std::vector<VramTensorInfo> vram_inventory;
    std::map<std::pair<int, int>, ExpertSide> routed;
    std::map<int, ExpertSide> shared;
    absl::flat_hash_map<int, int> layer_tensor_count;

    const std::string kPacked = "weight_packed";
    const std::string kScale = "weight_scale";
    const std::string kGlobal = "weight_global_scale";
    const std::string kInputGlobal = "input_global_scale";

    for (auto &shard : model.shards)
    {
        for (auto &t : shard.tensors)
        {
            const auto p = mistral_classify(t.name);
            const TensorRef ref{&shard, &t};
            switch (p.kind)
            {
            case MistralTensorKind::Unknown:
                return arrow::Status::Invalid(
                    "unrecognized text-decoder tensor: ", t.name);
            case MistralTensorKind::Vision:
                skipped.push_back(TensorRecord{t.name, t.dtype, t.shape, {}});
                break;
            case MistralTensorKind::Global:
            case MistralTensorKind::Attention:
            case MistralTensorKind::LayerNorm:
            case MistralTensorKind::Router:
            {
                TensorRecord r;
                ARROW_RETURN_NOT_OK(RawRecord(input_dir, ref, r));
                if (p.kind == MistralTensorKind::Global)
                    global_raw.push_back(std::move(r));
                else
                    raw_by_layer[p.layer].push_back(std::move(r));
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
                if (p.layer >= 0) ++layer_tensor_count[p.layer];
                break;
            }
            case MistralTensorKind::RoutedExpert:
            {
                if (p.companion == "input_global_scale") break;  // activation
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
                auto &side = routed[{p.layer, p.expert}];
                QuantProjection *q = p.weight_id == 1   ? &side.w1
                                     : p.weight_id == 2 ? &side.w2
                                                        : &side.w3;
                if (p.companion.empty())
                {
                    ARROW_RETURN_NOT_OK(ReadQuantProjection(input_dir, ref, *q));
                }
                else if (p.companion == "scale")
                {
                    const auto raw = ReadTensorData(input_dir, shard, t);
                    if (raw.empty() ||
                        static_cast<int64_t>(raw.size()) != t.byte_size())
                        return arrow::Status::IOError("short read: ", t.name);
                    q->scales = raw;
                    q->scale_shape = t.shape;
                }
                else if (p.companion == "global_scale")
                {
                    const auto raw = ReadTensorData(input_dir, shard, t);
                    if (raw.size() != 4)
                        return arrow::Status::Invalid("bad global scale: ",
                                                      t.name);
                    float gs = 0.0f;
                    std::memcpy(&gs, raw.data(), 4);
                    // compressed-tensors `weight_global_scale` is a DIVISOR:
                    // w = nibble * block_scale / weight_global_scale. The repo
                    // contour multiplies by its stored global_scale, so store
                    // the reciprocal.
                    q->global_scale = (gs != 0.0f) ? 1.0f / gs : 0.0f;
                }
                break;
            }
            case MistralTensorKind::SharedExpert:
            {
                if (p.companion == "input_global_scale") break;
                vram_inventory.push_back({t.name, t.numel(), t.byte_size()});
                auto &side = shared[p.layer];
                QuantProjection *q = p.weight_id == 1   ? &side.w1
                                     : p.weight_id == 2 ? &side.w2
                                                        : &side.w3;
                if (p.companion.empty())
                {
                    ARROW_RETURN_NOT_OK(ReadQuantProjection(input_dir, ref, *q));
                }
                else if (p.companion == "scale")
                {
                    const auto raw = ReadTensorData(input_dir, shard, t);
                    if (raw.empty() ||
                        static_cast<int64_t>(raw.size()) != t.byte_size())
                        return arrow::Status::IOError("short read: ", t.name);
                    q->scales = raw;
                    q->scale_shape = t.shape;
                }
                else if (p.companion == "global_scale")
                {
                    const auto raw = ReadTensorData(input_dir, shard, t);
                    if (raw.size() != 4)
                        return arrow::Status::Invalid("bad global scale: ",
                                                      t.name);
                    float gs = 0.0f;
                    std::memcpy(&gs, raw.data(), 4);
                    q->global_scale = (gs != 0.0f) ? 1.0f / gs : 0.0f;
                }
                break;
            }
            }
        }
    }

    int observed_layers = 0;
    for (const auto &[l, n] : layer_tensor_count)
        observed_layers = std::max(observed_layers, l + 1);
    if (observed_layers == 0)
        return arrow::Status::Invalid("no decoder layers in ", input_dir);

    const fs::path base = output_root;
    const fs::path pinned_dir = base / "pinned";
    std::error_code ec;
    fs::create_directories(pinned_dir, ec);

    // ── routed experts: one ExpertSchema file per (layer, expert) ──────────
    int total_experts = 0;
    for (auto &[key, side] : routed)
    {
        const int layer = key.first;
        const int expert = key.second;
        if (side.w1.packed.empty() || side.w2.packed.empty() ||
            side.w3.packed.empty())
            return arrow::Status::Invalid("incomplete expert ", layer, "/",
                                          expert);
        PackedExpert pe;
        ARROW_RETURN_NOT_OK(BuildPackedExpert(side.w1, side.w2, side.w3,
                                              expert_hidden, hidden, pe));
        const fs::path dir =
            base / "experts" / absl::StrCat("layer_", layer);
        fs::create_directories(dir, ec);
        ARROW_RETURN_NOT_OK(
            WriteBatchOr(ExpertSchema::MakeBatchView(pe),
                         dir / absl::StrCat("exp_", expert, ".arrow")));
        ++total_experts;
    }

    // ── shared experts: one SharedExpertSchema record per layer ────────────
    if (!shared.empty())
    {
        std::vector<PackedExpert> rows(static_cast<size_t>(observed_layers));
        for (auto &[layer, side] : shared)
        {
            ARROW_RETURN_NOT_OK(
                BuildPackedExpert(side.w1, side.w2, side.w3, expert_hidden,
                                  hidden, rows[static_cast<size_t>(layer)]));
        }
        ARROW_RETURN_NOT_OK(WriteBatchOr(
            SharedExpertSchema::MakeBatch(rows),
            pinned_dir / "shared_experts.arrow"));
    }

    // ── global + per-layer raw (attention/norms/router stay BF16) ──────────
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatchOr(PinnedSchema::MakeBatch(global_raw),
                                         pinned_dir / "global.arrow"));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatchOr(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / LayerFile(layer, ".arrow")));

    // ── tokenizer (byte-level BPE, tokenizer.json) ─────────────────────────
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    if (fs::exists(tok_path))
    {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 7;
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
    std::vector<long long> moe_layers;
    for (int l = 0; l < observed_layers; ++l) moe_layers.push_back(l);

    // Conversion-time placement profile (mandatory, see vram_estimate.h). The
    // routed experts are streamed from the host store, so min_bytes is the
    // pinned dense graph only and max_bytes adds the full routed pool. Context
    // comes from params.json: the consolidated checkpoint has no config.json
    // for ReadContextLength to read.
    VramGeometry vg;
    vg.hidden_dim = static_cast<int>(hidden);
    // MLA caches the latent, not per-head K/V: kv_lora + rope per layer.
    vg.kv_dim = static_cast<int>(kv_lora + qk_rope);
    vg.num_attn_layers = observed_layers;
    vg.num_moe_layers = observed_layers;
    vg.experts_per_layer = static_cast<int>(num_experts);
    vg.context_length = JsonInt(pt, "max_position_embeddings", 0);
    const VramProfile vram = EstimateVram(vram_inventory, MistralVramPolicy(), vg);
    lsxcommon::log::info(absl::StrFormat(
        "VRAM profile: min=%s optimal=%s max=%s (ctx=%d)",
        lsxcommon::FormatBytesHuman(vram.min_bytes),
        lsxcommon::FormatBytesHuman(vram.optimal_bytes),
        lsxcommon::FormatBytesHuman(vram.max_bytes),
        static_cast<int>(vram.context_tokens)));

    std::ofstream f(base / "model.index.json");
    if (!f) return arrow::Status::IOError("cannot write model.index.json");
    f << "{\n";
    f << "  \"layout\": 2,\n";
    // Must normalize (detect_model_type) to exactly frozen::kType, otherwise
    // path auto-detection in lsxbenchmark/lsxfabric cannot resolve the family.
    f << "  \"model\": \"Mistral-Small-4-119B\",\n";
    f << "  \"num_layers\": " << observed_layers << ",\n";
    f << "  \"experts_per_layer\": " << num_experts << ",\n";
    f << "  \"hidden_size\": " << hidden << ",\n";
    f << "  \"num_attention_heads\": " << n_heads << ",\n";
    f << "  \"q_lora_rank\": " << q_lora << ",\n";
    f << "  \"kv_lora_rank\": " << kv_lora << ",\n";
    f << "  \"qk_nope_head_dim\": " << qk_nope << ",\n";
    f << "  \"qk_rope_head_dim\": " << qk_rope << ",\n";
    f << "  \"v_head_dim\": " << v_head << ",\n";
    f << "  \"moe_intermediate_size\": " << expert_hidden << ",\n";
    f << "  \"num_experts_per_tok\": " << top_k << ",\n";
    f << "  \"context_length\": " << JsonInt(pt, "max_position_embeddings", 0)
      << ",\n";
    f << "  \"rope_theta\": " << 10000 << ",\n";
    f << "  \"yarn_factor\": " << 128 << ",\n";
    f << "  \"yarn_orig\": " << 8192 << ",\n";
    f << "  \"llama4_beta\": 0.1,\n";
    f << "  \"llama4_orig\": " << 8192 << ",\n";
    f << "  \"bos_token_id\": " << frozen::kBosId << ",\n";
    f << "  \"eos_token_id\": " << frozen::kEosId << ",\n";
    f << "  \"pad_token_id\": " << frozen::kPadId << ",\n";
    f << "  \"moe_layers\": [";
    for (size_t i = 0; i < moe_layers.size(); ++i)
        f << (i ? "," : "") << moe_layers[i];
    f << "],\n";
    f << "  \"dense_layers\": [],\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  \"skipped_tensors\": [";
    for (size_t i = 0; i < skipped.size(); ++i)
        f << (i ? ", " : "") << "\"" << skipped[i].name << "\"";
    f << "],\n";
    f << FormatVramJson(vram) << "\n";
    f << "}\n";
    f.close();

    return arrow::Status::OK();
}

} // namespace mistral_moe
} // namespace lsxcommon
