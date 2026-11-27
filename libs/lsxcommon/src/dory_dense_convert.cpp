#include "lsxcommon/dory_dense_convert.h"

#include <simdjson.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include "lsxcommon/converter.h"
#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"

namespace lsxcommon {
namespace dory {

namespace fs = std::filesystem;

// ── Classifier ──────────────────────────────────────────────────────────────

namespace {

constexpr const char *kLayerPrefix = "backbone.layers.";

struct SuffixKind {
    const char *suffix;
    TensorKind kind;
};

// Every tensor name a mixer can carry. The attention and mlp sets are
// disjoint, so one table is enough; the frozen layer pattern is what decides
// which set a given layer must have (checked by the coverage pass).
constexpr SuffixKind kMixerSuffixes[] = {
    {"mixer.q_proj.weight", TensorKind::MixerQ},
    {"mixer.k_proj.weight", TensorKind::MixerK},
    {"mixer.v_proj.weight", TensorKind::MixerV},
    {"mixer.o_proj.weight", TensorKind::MixerO},
    {"mixer.g_proj.weight", TensorKind::MixerG},
    {"mixer.gate_proj.weight", TensorKind::MixerGate},
    {"mixer.up_proj.weight", TensorKind::MixerUp},
    {"mixer.down_proj.weight", TensorKind::MixerDown},
    {"mixer.sqk", TensorKind::MixerSqk},
    {"mixer.s_gate", TensorKind::MixerSGate},
    {"mixer.suv_gate", TensorKind::MixerSuvGate},
    {"mixer.suv_up", TensorKind::MixerSuvUp},
    {"attn_alpha", TensorKind::AttnAlpha},
    {"mlp_alpha", TensorKind::MlpAlpha},
};

bool IsProjection(TensorKind kind)
{
    switch (kind) {
    case TensorKind::MixerQ:
    case TensorKind::MixerK:
    case TensorKind::MixerV:
    case TensorKind::MixerO:
    case TensorKind::MixerG:
    case TensorKind::MixerGate:
    case TensorKind::MixerUp:
    case TensorKind::MixerDown:
        return true;
    default:
        return false;
    }
}

// The expected tensor suffixes of one mixer, per the frozen layer pattern.
std::vector<std::string> ExpectedMixerTensors(int layer)
{
    if (frozen::LayerIsAttention(layer))
        return {"mixer.q_proj.weight", "mixer.k_proj.weight",
                "mixer.v_proj.weight", "mixer.o_proj.weight",
                "mixer.g_proj.weight",  "mixer.sqk",
                "mixer.s_gate",         "attn_alpha"};
    return {"mixer.gate_proj.weight", "mixer.up_proj.weight",
            "mixer.down_proj.weight", "mixer.suv_gate",
            "mixer.suv_up",           "mlp_alpha"};
}

std::vector<std::string> GenerateExpectedNames(const DoryTopology &top)
{
    std::vector<std::string> expected = {
        "backbone.embeddings.weight", "lm_head.weight", "lm_head.logit_scale"};
    for (int l = 0; l < top.num_layers; ++l)
        for (const auto &s : ExpectedMixerTensors(l))
            expected.push_back(absl::StrCat(kLayerPrefix, l, ".", s));
    return expected;
}

// ── config.json reader ─────────────────────────────────────────────────────

struct DorySourceConfig {
    DoryTopology top;
    std::vector<std::string> layer_types;
};

bool ReadDoryConfig(const std::string &path, DorySourceConfig &cfg)
{
    std::ifstream f(path);
    if (!f) return false;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return false;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return false;

    auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
        int64_t v;
        if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_d = [&](simdjson::ondemand::object &o, const char *k, double dflt) {
        simdjson::ondemand::value x;
        if (o[k].get(x) != simdjson::SUCCESS) return dflt;
        if (auto d = x.get_double(); d.error() == simdjson::SUCCESS)
            return d.value();
        if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
            return static_cast<double>(i.value());
        return dflt;
    };

    auto &t = cfg.top;
    t.hidden_size = get_i(root, "hidden_size", 0);
    t.intermediate_size = get_i(root, "intermediate_size", 0);
    t.vocab_size = get_i(root, "vocab_size", 0);
    t.num_layers = get_i(root, "num_hidden_layers", 0);
    t.num_attention_heads = get_i(root, "num_attention_heads", 0);
    t.num_key_value_heads = get_i(root, "num_key_value_heads", 0);
    t.head_dim = get_i(root, "head_dim", 0);
    if (t.head_dim == 0 && t.num_attention_heads > 0 && t.hidden_size > 0)
        t.head_dim = t.hidden_size / t.num_attention_heads;
    t.input_layers = get_i(root, "n_input_layers", 0);
    t.recurrent_layers = get_i(root, "n_recurrent_layers", 0);
    t.output_layers = get_i(root, "n_output_layers", 0);
    if (t.recurrent_layers == 0)
        t.recurrent_layers =
            t.num_layers - t.input_layers - t.output_layers;
    t.recurrent_loops = get_i(root, "n_recurrent_loops", 1);
    t.sliding_window = get_i(root, "sliding_window", 0);
    t.initializer_range = get_d(root, "initializer_range", 0.02);
    t.ngpt_alpha_init = get_d(root, "ngpt_alpha_init", 0.05);
    if (t.head_dim > 0)
        t.attention_scale = 1.0 / std::sqrt(static_cast<double>(t.head_dim));

    // RoPE: per attention block type. A missing block type is a hard error —
    // a single uniform theta silently decodes garbage.
    simdjson::ondemand::object ropes;
    if (root["rope_parameters"].get_object().get(ropes) == simdjson::SUCCESS) {
        for (auto field : ropes) {
            std::string_view name;
            if (field.unescaped_key().get(name) != simdjson::SUCCESS) continue;
            simdjson::ondemand::object r;
            if (field.value().get_object().get(r) != simdjson::SUCCESS) continue;
            const double theta = get_d(r, "rope_theta", 0.0);
            if (name == "sliding_attention" && theta > 0)
                t.rope_theta_sliding = theta;
            if (name == "full_attention" && theta > 0)
                t.rope_theta_full = theta;
        }
    }
    simdjson::ondemand::array types;
    if (root["layer_types"].get_array().get(types) == simdjson::SUCCESS) {
        for (auto v : types) {
            std::string_view s;
            if (v.get_string().get(s) != simdjson::SUCCESS) return false;
            cfg.layer_types.emplace_back(s);
        }
    }
    if (static_cast<int>(cfg.layer_types.size()) != t.num_layers) return false;
    int attn = 0;
    for (int l = 0; l < t.num_layers; ++l) {
        const std::string &kind = cfg.layer_types[static_cast<size_t>(l)];
        if (kind == "mlp") continue;
        if (kind != "sliding_attention" && kind != "full_attention") return false;
        ++attn;
    }
    t.num_attention_layers = attn;
    return true;
}

// generation_config.json supplies the special ids (config.json carries none):
// eos_token_id is a LIST ([</s>, <|im_end|>]); the canonical stop token is the
// tokenizer's eos, which the checkpoint spells <|im_end|> — the last entry.
void ReadDorySpecialIds(const std::string &dir, DoryTopology &t)
{
    t.bos_id = 1;
    t.eos_id = 11;
    t.pad_id = 11;
    std::ifstream f(fs::path(dir) / "generation_config.json");
    if (!f) return;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return;

    int64_t v = 0;
    if (root["bos_token_id"].get_int64().get(v) == simdjson::SUCCESS)
        t.bos_id = static_cast<int>(v);
    if (root["pad_token_id"].get_int64().get(v) == simdjson::SUCCESS)
        t.pad_id = static_cast<int>(v);
    if (root["eos_token_id"].get_int64().get(v) == simdjson::SUCCESS) {
        t.eos_id = static_cast<int>(v);
        return;
    }
    simdjson::ondemand::array eos;
    if (root["eos_token_id"].get_array().get(eos) == simdjson::SUCCESS) {
        int last = t.eos_id;
        for (auto e : eos) {
            int64_t id;
            if (e.get_int64().get(id) == simdjson::SUCCESS)
                last = static_cast<int>(id);
        }
        t.eos_id = last;
    }
}

// The frozen passport gate: a real checkpoint must match the constants
// exactly. Synthetic fixtures (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) skip it so
// the converter pipeline can be exercised on a tiny checkpoint.
arrow::Status CheckFrozenGeometry(const DorySourceConfig &cfg)
{
    if (SyntheticGeometryAllowed()) return arrow::Status::OK();
    const auto &t = cfg.top;
    struct Row {
        const char *name;
        long long want;
        long long got;
    };
    const Row rows[] = {
        {"hidden_size", frozen::kHiddenSize, t.hidden_size},
        {"num_hidden_layers", frozen::kNumLayers, t.num_layers},
        {"num_attention_heads", frozen::kNumHeads, t.num_attention_heads},
        {"num_key_value_heads", frozen::kNumKVHeads, t.num_key_value_heads},
        {"head_dim", frozen::kHeadDim, t.head_dim},
        {"intermediate_size", frozen::kIntermediateSize, t.intermediate_size},
        {"vocab_size", frozen::kVocabSize, t.vocab_size},
        {"n_input_layers", frozen::kInputLayers, t.input_layers},
        {"n_recurrent_layers", frozen::kRecurrentLayers, t.recurrent_layers},
        {"n_output_layers", frozen::kOutputLayers, t.output_layers},
        {"n_recurrent_loops", frozen::kRecurrentLoops, t.recurrent_loops},
        {"sliding_window", frozen::kSlidingWindow, t.sliding_window},
        {"num_attention_layers", frozen::kNumAttentionLayers,
         t.num_attention_layers},
    };
    for (const auto &r : rows)
        if (r.want != r.got)
            return arrow::Status::Invalid(
                absl::StrFormat("dory frozen geometry: %s is %d in the "
                                "checkpoint, the passport pins %d",
                                r.name, r.got, r.want));
    if (std::abs(t.rope_theta_sliding - frozen::kRopeThetaSliding) > 1e-6 ||
        std::abs(t.rope_theta_full - frozen::kRopeThetaFull) > 1e-6)
        return arrow::Status::Invalid(
            "dory frozen geometry: rope thetas differ from the passport");
    if (std::abs(t.initializer_range - frozen::kInitializerRange) > 1e-9)
        return arrow::Status::Invalid(
            "dory frozen geometry: initializer_range differs from the passport");
    // The layer pattern itself, not just the counts.
    for (int l = 0; l < t.num_layers; ++l) {
        const std::string &kind = cfg.layer_types[static_cast<size_t>(l)];
        const bool want_attn = frozen::LayerIsAttention(l);
        const bool got_attn = kind != "mlp";
        if (want_attn != got_attn)
            return arrow::Status::Invalid(absl::StrCat(
                "dory frozen geometry: layer ", l, " is '", kind,
                "' but the passport pins ",
                want_attn ? "an attention mixer" : "an mlp mixer"));
        if (!want_attn) continue;
        const bool want_full = frozen::LayerIsFullAttention(l);
        if (want_full != (kind == "full_attention"))
            return arrow::Status::Invalid(absl::StrCat(
                "dory frozen geometry: layer ", l, " attention window kind "
                "differs from the passport"));
    }
    return arrow::Status::OK();
}

// ── source tensor reading ──────────────────────────────────────────────────

struct RawTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;

    int64_t Rows() const { return shape.empty() ? 0 : shape[0]; }
    int64_t Cols() const { return shape.size() < 2 ? 0 : shape[1]; }
    int64_t Numel() const
    {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

using TensorLoc = std::pair<size_t, size_t>;  // (shard, tensor) inside the model

bool LoadRawTensor(const std::string &input_dir, const SafetensorsModel &model,
                   const absl::flat_hash_map<std::string, TensorLoc> &index,
                   const std::string &name, RawTensor &out)
{
    const auto it = index.find(name);
    if (it == index.end()) {
        log::error(absl::StrCat("[dory-convert] missing tensor: ", name));
        return false;
    }
    const auto &shard = model.shards[it->second.first];
    const auto &info = shard.tensors[it->second.second];
    out.data = ReadTensorData(input_dir, shard, info);
    if (out.data.empty()) {
        log::error(absl::StrCat("[dory-convert] unreadable tensor: ", name));
        return false;
    }
    out.name = name;
    out.dtype = info.dtype;
    out.shape = info.shape;
    return true;
}

// Row-major fusion of same-dtype [R_i, C] tensors into one [sum R_i, C]
// buffer: the decode then issues ONE GEMV for the mixer input instead of three
// (q|k|v) or two (gate|up).
bool FuseRows(const std::vector<const RawTensor *> &parts, RawTensor &out)
{
    if (parts.empty()) return false;
    const std::string &dtype = parts.front()->dtype;
    int64_t rows = 0;
    const int64_t cols = parts.front()->Cols();
    for (const RawTensor *p : parts) {
        if (p == nullptr || p->dtype != dtype || p->Cols() != cols ||
            p->shape.size() != 2)
            return false;
        rows += p->Rows();
    }
    out.data.clear();
    out.data.reserve(static_cast<size_t>(rows * cols *
                                        TensorDtypeBytes(dtype)));
    for (const RawTensor *p : parts)
        out.data.insert(out.data.end(), p->data.begin(), p->data.end());
    out.name = "fused";
    out.dtype = dtype;
    out.shape = {rows, cols};
    return true;
}

// Quantizes one source tensor to NVFP4 plus the residual plane (w - dequant).
arrow::Status QuantizeWithResidual(const RawTensor &src,
                                   std::vector<DenseNvfp4Record> &out)
{
    DenseNvfp4Record base;
    base.name = src.name;
    base.shape = src.shape;
    base.dtype = src.dtype;
    if (!gpu::quantize_tensor_gpu(src.data.data(), src.Numel(), src.dtype,
                                  nullptr, src.shape, {}, base.packed,
                                  base.scales, base.global_scale))
        return arrow::Status::IOError("NVFP4 quantization failed for ",
                                      src.name);

    DenseNvfp4Record res;
    res.name = src.name + ".res";
    res.shape = src.shape;
    res.dtype = src.dtype;
    if (!gpu::quantize_residual_tensor_gpu(
            src.data.data(), src.Numel(), src.dtype, src.shape, base.packed,
            base.scales, base.global_scale, res.packed, res.scales,
            res.global_scale))
        return arrow::Status::IOError("residual quantization failed for ",
                                      src.name);
    out.push_back(std::move(base));
    out.push_back(std::move(res));
    return arrow::Status::OK();
}

int CountKvSlots(const DoryTopology &t)
{
    int n = 0;
    for (int l = 0; l < t.num_layers; ++l) {
        if (!frozen::LayerIsAttention(l)) continue;
        const bool recurrent =
            l >= t.input_layers && l < t.input_layers + t.recurrent_layers;
        n += recurrent ? t.recurrent_loops : 1;
    }
    return n;
}

} // namespace

ParsedTensor classify_dory_tensor(const std::string &name)
{
    if (name == "backbone.embeddings.weight")
        return {TensorKind::Embed, -1};
    if (name == "lm_head.weight") return {TensorKind::LmHead, -1};
    if (name == "lm_head.logit_scale") return {TensorKind::LogitScale, -1};
    if (!absl::StartsWith(name, kLayerPrefix)) return {TensorKind::Unknown, -1};

    size_t pos = std::strlen(kLayerPrefix);
    size_t digits = pos;
    while (digits < name.size() && name[digits] >= '0' && name[digits] <= '9')
        ++digits;
    if (digits == pos || digits >= name.size() || name[digits] != '.')
        return {TensorKind::Unknown, -1};
    int layer = -1;
    if (!absl::SimpleAtoi(name.substr(pos, digits - pos), &layer) || layer < 0)
        return {TensorKind::Unknown, -1};
    const std::string_view rest(name.data() + digits + 1,
                                name.size() - digits - 1);
    for (const auto &s : kMixerSuffixes)
        if (rest == s.suffix) return {s.kind, layer};
    return {TensorKind::Unknown, -1};
}

bool dory_tensor_is_nvfp4(TensorKind kind) { return IsProjection(kind); }

bool dory_fused_kind(TensorKind kind)
{
    return kind == TensorKind::MixerQ || kind == TensorKind::MixerK ||
           kind == TensorKind::MixerV ||
           kind == TensorKind::MixerGate || kind == TensorKind::MixerUp;
}

// ── Topology ────────────────────────────────────────────────────────────────

DoryTopology load_dory_topology(const std::string &index_path)
{
    DoryTopology top;
    std::ifstream f(index_path);
    if (!f) return top;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return top;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return top;

    auto get_int = [&](const char *k, int dflt) {
        int64_t v;
        if (root[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
        return static_cast<int>(v);
    };
    auto get_dbl = [&](const char *k, double dflt) {
        if (auto x = root[k].get_double(); x.error() == simdjson::SUCCESS)
            return x.value();
        return dflt;
    };
    std::string_view format;
    if (root["format"].get_string().get(format) == simdjson::SUCCESS &&
        format != "logestix-dory-nvfp4") {
        top.error_msg = absl::StrCat("unexpected format tag: ", format);
        return top;
    }

    top.num_layers = get_int("num_layers", 0);
    top.hidden_size = get_int("hidden_size", 0);
    top.intermediate_size = get_int("intermediate_size", 0);
    top.vocab_size = get_int("vocab_size", 0);
    top.num_attention_heads = get_int("num_attention_heads", 0);
    top.num_key_value_heads = get_int("num_key_value_heads", 0);
    top.head_dim = get_int("head_dim", 0);
    top.input_layers = get_int("n_input_layers", 0);
    top.recurrent_layers = get_int("n_recurrent_layers", 0);
    top.output_layers = get_int("n_output_layers", 0);
    top.recurrent_loops = get_int("n_recurrent_loops", 1);
    top.sliding_window = get_int("sliding_window", 0);
    top.num_attention_layers = get_int("num_attention_layers", 0);
    top.rope_theta_sliding = get_dbl("rope_theta_sliding", 10000.0);
    top.rope_theta_full = get_dbl("rope_theta_full", 10000000.0);
    top.initializer_range = get_dbl("initializer_range", 0.0177);
    top.ngpt_alpha_init = get_dbl("ngpt_alpha_init", 0.05);
    top.attention_scale = get_dbl("attention_scale", 0.0625);
    top.bos_id = get_int("bos_id", -1);
    top.eos_id = get_int("eos_id", -1);
    top.pad_id = get_int("pad_id", -1);
    top.total_pinned_tensors = get_int("total_pinned_tensors", 0);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        top.error_msg = verr;
        return top;
    }
    return top;
}

arrow::Status WriteDoryModelIndex(const std::string &output_root,
                                  const DoryTopology &top)
{
    auto dir = fs::path(output_root);
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"logestix-dory-nvfp4\",\n"
      << "  \"model\": \"Dory-1.0-4B\",\n"
      << "  \"architecture\": \"looped-ngpt\",\n"
      << "  \"edges\": \"fp16\",\n"
      << "  \"residual\": true,\n"
      << "  \"kv_cache_mode\": \"per_loop\",\n"
      << "  \"num_layers\": " << top.num_layers
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"n_input_layers\": " << top.input_layers
      << ",\n  \"n_recurrent_layers\": " << top.recurrent_layers
      << ",\n  \"n_output_layers\": " << top.output_layers
      << ",\n  \"n_recurrent_loops\": " << top.recurrent_loops
      << ",\n  \"sliding_window\": " << top.sliding_window
      << ",\n  \"num_attention_layers\": " << top.num_attention_layers
      << ",\n  \"num_kv_slots\": " << CountKvSlots(top)
      << ",\n  \"rope_theta_sliding\": " << top.rope_theta_sliding
      << ",\n  \"rope_theta_full\": " << top.rope_theta_full
      << ",\n  \"initializer_range\": " << top.initializer_range
      << ",\n  \"ngpt_alpha_init\": " << top.ngpt_alpha_init
      << ",\n  \"attention_scale\": " << top.attention_scale
      << ",\n  \"bos_id\": " << top.bos_id
      << ",\n  \"eos_id\": " << top.eos_id
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram)
      << "\n}";

    std::ofstream out(dir / "model.index.json");
    if (!out) return arrow::Status::IOError("cannot write model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

// ── Converter ──────────────────────────────────────────────────────────────

arrow::Status ConvertDoryModel(const std::string &input_path,
                               const std::string &output_root)
{
    auto model = scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    const fs::path cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    DorySourceConfig cfg;
    if (!ReadDoryConfig(cfg_path.string(), cfg))
        return arrow::Status::Invalid("failed to parse ", cfg_path.string());
    ARROW_RETURN_NOT_OK(CheckFrozenGeometry(cfg));
    DoryTopology top = cfg.top;
    ReadDorySpecialIds(input_path, top);

    // Name index over every shard, and the classification pass. An
    // unrecognized tensor is an ERROR: the checkpoint has no ballast, so an
    // unknown name means the classifier is out of date.
    absl::flat_hash_map<std::string, TensorLoc> index;
    std::vector<std::string> present;
    for (size_t si = 0; si < model.shards.size(); ++si) {
        const auto &shard = model.shards[si];
        for (size_t ti = 0; ti < shard.tensors.size(); ++ti) {
            const auto &info = shard.tensors[ti];
            const ParsedTensor p = classify_dory_tensor(info.name);
            if (p.kind == TensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unclassified tensor in the Dory checkpoint: ", info.name);
            index.emplace(info.name, TensorLoc{si, ti});
            present.push_back(info.name);
        }
    }

    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));
    ARROW_RETURN_NOT_OK(integrity::CheckTensorCoverage(
        GenerateExpectedNames(top), present, "dory"));
    {
        absl::flat_hash_set<int> layers;
        for (const auto &name : present) {
            const auto p = classify_dory_tensor(name);
            if (p.layer >= 0) layers.insert(p.layer);
        }
        ARROW_RETURN_NOT_OK(
            integrity::CheckLayerContiguity(top.num_layers, layers));
    }

    const fs::path pinned_dir = fs::path(output_root) / "pinned";
    if (!fs::exists(pinned_dir)) fs::create_directories(pinned_dir);

    // Global edges + the per-vocabulary logit scale stay raw (FP16/BF16): the
    // logit scale is a nGPT normalization reference, not a linear weight.
    std::vector<TensorRecord> global_raw;
    for (const char *name : {"backbone.embeddings.weight", "lm_head.weight",
                             "lm_head.logit_scale"}) {
        RawTensor t;
        if (!LoadRawTensor(input_path, model, index, name, t))
            return arrow::Status::IOError("failed to read ", name);
        TensorRecord r;
        r.name = t.name;
        r.dtype = t.dtype;
        r.shape = t.shape;
        r.data = std::move(t.data);
        global_raw.push_back(std::move(r));
    }
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(WriteBatch(PinnedSchema::MakeBatch(global_raw),
                                       pinned_dir / "global.arrow"));

    int total_pinned = static_cast<int>(global_raw.size());
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string lp = absl::StrCat(kLayerPrefix, l, ".");
        std::vector<TensorRecord> raw_layer;
        std::vector<DenseNvfp4Record> nv_layer;

        auto raw = [&](const char *suffix) -> arrow::Status {
            RawTensor t;
            if (!LoadRawTensor(input_path, model, index,
                               absl::StrCat(lp, suffix), t))
                return arrow::Status::IOError("failed to read ", lp, suffix);
            TensorRecord r;
            r.name = t.name;
            r.dtype = t.dtype;
            r.shape = t.shape;
            r.data = std::move(t.data);
            raw_layer.push_back(std::move(r));
            return arrow::Status::OK();
        };

        if (frozen::LayerIsAttention(l)) {
            // Attention: fuse q|k|v into one record, quantize it and the
            // separate gate (g_proj) and output (o_proj) projections.
            RawTensor q, k, v, g, o;
            if (!LoadRawTensor(input_path, model, index, lp + "mixer.q_proj.weight", q) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.k_proj.weight", k) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.v_proj.weight", v) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.g_proj.weight", g) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.o_proj.weight", o))
                return arrow::Status::IOError("failed to read attention mixer ",
                                              l);
            RawTensor qkv;
            if (!FuseRows({&q, &k, &v}, qkv))
                return arrow::Status::IOError("cannot fuse q|k|v for layer ", l);
            qkv.name = absl::StrCat(lp, "mixer.qkv_proj.weight");
            g.name = absl::StrCat(lp, "mixer.g_proj.weight");
            o.name = absl::StrCat(lp, "mixer.o_proj.weight");
            ARROW_RETURN_NOT_OK(QuantizeWithResidual(qkv, nv_layer));
            ARROW_RETURN_NOT_OK(QuantizeWithResidual(g, nv_layer));
            ARROW_RETURN_NOT_OK(QuantizeWithResidual(o, nv_layer));
            ARROW_RETURN_NOT_OK(raw("mixer.sqk"));
            ARROW_RETURN_NOT_OK(raw("mixer.s_gate"));
            ARROW_RETURN_NOT_OK(raw("attn_alpha"));
        } else {
            RawTensor gate, up, down;
            if (!LoadRawTensor(input_path, model, index, lp + "mixer.gate_proj.weight", gate) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.up_proj.weight", up) ||
                !LoadRawTensor(input_path, model, index, lp + "mixer.down_proj.weight", down))
                return arrow::Status::IOError("failed to read mlp mixer ", l);
            RawTensor gate_up;
            if (!FuseRows({&gate, &up}, gate_up))
                return arrow::Status::IOError("cannot fuse gate|up for layer ",
                                              l);
            gate_up.name = absl::StrCat(lp, "mixer.gate_up_proj.weight");
            down.name = absl::StrCat(lp, "mixer.down_proj.weight");
            ARROW_RETURN_NOT_OK(QuantizeWithResidual(gate_up, nv_layer));
            ARROW_RETURN_NOT_OK(QuantizeWithResidual(down, nv_layer));
            ARROW_RETURN_NOT_OK(raw("mixer.suv_gate"));
            ARROW_RETURN_NOT_OK(raw("mixer.suv_up"));
            ARROW_RETURN_NOT_OK(raw("mlp_alpha"));
        }

        total_pinned += static_cast<int>(raw_layer.size() + nv_layer.size());
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(raw_layer),
            pinned_dir / absl::StrCat("layer_", l, ".arrow")));
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(nv_layer),
            pinned_dir / absl::StrCat("layer_", l, "_nvfp4.arrow")));
    }

    top.total_pinned_tensors = total_pinned;

    // Conversion-time VRAM profile.
    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        // Per-loop KV caching: the cache count is the slot count, not the
        // attention-layer count (a recurrent layer holds one cache per loop).
        vg.num_attn_layers = CountKvSlots(top);
        vg.context_length = ReadContextLength(cfg_path.string());

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                vram_inventory.push_back(
                    {t.name, t.numel(), t.byte_size()});

        top.vram = EstimateVram(
            vram_inventory,
            {+[](const std::string &n) {
                 return dory_tensor_is_nvfp4(
                     classify_dory_tensor(n).kind);
             },
             nullptr,
             +[](const std::string &n) {
                 return dory_tensor_is_nvfp4(
                     classify_dory_tensor(n).kind);
             }},
            vg);
        log::info(absl::StrFormat(
            "VRAM profile: min=%s optimal=%s max=%s (ctx=%lld, kv slots=%d)",
            FormatBytesHuman(top.vram.min_bytes),
            FormatBytesHuman(top.vram.optimal_bytes),
            FormatBytesHuman(top.vram.max_bytes),
            static_cast<long long>(top.vram.context_tokens),
            CountKvSlots(top)));
    }

    ARROW_RETURN_NOT_OK(WriteDoryModelIndex(output_root, top));

    const fs::path tok_path = fs::path(input_path) / "tokenizer.json";
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.pretokenizer = 7;   // o200k case runs (Nemotron/Mistral)
        // The reference tokenizer never emits BOS (`add_bos_token: false`);
        // the chat template spells <|im_start|> out as a control token.
        tok_policy.bos_id = -1;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true,
                                           .pad_required = false},
            tok_policy));
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    }
    return arrow::Status::OK();
}

arrow::Status RecomputeDoryVram(const std::string &input_path,
                                const std::string &output_root)
{
    const fs::path idx = fs::path(output_root) / "model.index.json";
    std::ifstream f(idx);
    if (!f) return arrow::Status::IOError("cannot open ", idx.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    auto model = scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);
    const fs::path cfg_path = fs::path(input_path) / "config.json";
    DorySourceConfig cfg;
    if (!ReadDoryConfig(cfg_path.string(), cfg))
        return arrow::Status::Invalid("failed to parse ", cfg_path.string());

    VramGeometry vg;
    vg.hidden_dim = cfg.top.hidden_size;
    vg.kv_dim = cfg.top.head_dim * cfg.top.num_key_value_heads;
    vg.num_attn_layers = CountKvSlots(cfg.top);
    vg.context_length = ReadContextLength(cfg_path.string());

    std::vector<VramTensorInfo> inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            inventory.push_back({t.name, t.numel(), t.byte_size()});

    const VramProfile profile = EstimateVram(
        inventory,
        {+[](const std::string &n) {
             return dory_tensor_is_nvfp4(classify_dory_tensor(n).kind);
         },
         nullptr,
         +[](const std::string &n) {
             return dory_tensor_is_nvfp4(classify_dory_tensor(n).kind);
         }},
        vg);

    std::ofstream out(idx, std::ios::trunc);
    if (!out) return arrow::Status::IOError("cannot write ", idx.string());
    out << ReplaceVramBlock(json, profile);
    return arrow::Status::OK();
}

} // namespace dory
} // namespace lsxcommon
