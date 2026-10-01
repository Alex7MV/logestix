#include "lsxcommon/phi_dense_convert.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/phi_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <sstream>

#include <absl/container/btree_map.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

namespace lsxcommon {
namespace phi {

// ── Classifier ──────────────────────────────────────────────────────────────

namespace {

std::optional<ParsedTensor> match_layer(const std::string &name,
                                        const std::regex &re, TensorKind kind)
{
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = kind;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    return p;
}

std::string layer_regex(const std::string &suffix)
{
    return R"(model\.layers\.(\d+)\.)" + suffix;
}

std::regex &re_linear(const std::string &s)
{
    static absl::flat_hash_map<std::string, std::regex> cache;
    auto it = cache.find(s);
    if (it == cache.end())
        it = cache.emplace(s, std::regex(layer_regex(s))).first;
    return it->second;
}

std::vector<std::string> generate_phi_expected_names(
    const PhiTopology &top)
{
    std::vector<std::string> expected;
    expected.push_back("model.embed_tokens.weight");
    expected.push_back("lm_head.weight");
    expected.push_back("model.norm.weight");
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string base = absl::StrCat("model.layers.", l, ".");
        expected.push_back(base + "input_layernorm.weight");
        expected.push_back(base + "post_attention_layernorm.weight");
        expected.push_back(base + "self_attn.qkv_proj.weight");
        expected.push_back(base + "self_attn.o_proj.weight");
        expected.push_back(base + "mlp.gate_up_proj.weight");
        expected.push_back(base + "mlp.down_proj.weight");
    }
    return expected;
}

void collect_phi_present(const SafetensorsModel &model,
                         std::vector<std::string> &present_names,
                         absl::flat_hash_set<int> &present_layers)
{
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_phi_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;
            present_names.push_back(t.name);
            if (p.layer >= 0) present_layers.insert(p.layer);
        }
    }
}

void phi_derive_actual_shapes(const SafetensorsModel &model, int &vocab,
                              int &hidden, int &intermediate,
                              int &tensor_layers)
{
    int max_layer = -1;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_phi_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;
            if (p.layer >= 0) max_layer = std::max(max_layer, p.layer);
            if (t.shape.size() < 2) continue;
            const int64_t r = t.shape[0];
            const int64_t c = t.shape[1];
            switch (p.kind) {
            case TensorKind::Embed:
            case TensorKind::LmHead:
                vocab = static_cast<int>(r);
                if (!hidden) hidden = static_cast<int>(c);
                break;
            case TensorKind::MlpDown:
                hidden = static_cast<int>(r);
                intermediate = static_cast<int>(c);
                break;
            default:
                break;
            }
        }
    }
    tensor_layers = max_layer + 1;
}

std::vector<uint8_t> read_tensor_data(const std::filesystem::path &input_path,
                                      const SafetensorsModel::Shard &shard,
                                      const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((input_path / shard.filename).string(), std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    return buf;
}

// ── config.json reader (flat phi3 layout) ───────────────────────────────────

bool read_phi_config(const std::string &path, PhiTopology &t)
{
    std::ifstream cf(path);
    if (!cf) return false;
    std::string cj((std::istreambuf_iterator<char>(cf)),
                   std::istreambuf_iterator<char>());
    simdjson::ondemand::parser cp;
    simdjson::padded_string pad(cj.data(), cj.size());
    simdjson::ondemand::document cd;
    if (cp.iterate(pad).get(cd) != simdjson::SUCCESS) return false;

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

    simdjson::ondemand::object root;
    if (cd.get_object().get(root) != simdjson::SUCCESS) return false;

    t.hidden_size = get_i(root, "hidden_size", t.hidden_size);
    t.intermediate_size = get_i(root, "intermediate_size", t.intermediate_size);
    t.vocab_size = get_i(root, "vocab_size", t.vocab_size);
    t.num_layers = get_i(root, "num_hidden_layers", t.num_layers);
    t.num_attention_heads =
        get_i(root, "num_attention_heads", t.num_attention_heads);
    t.num_key_value_heads =
        get_i(root, "num_key_value_heads", t.num_key_value_heads);
    t.head_dim = get_i(root, "head_dim", t.head_dim);
    if (t.head_dim == 0 && t.num_attention_heads > 0 && t.hidden_size > 0)
        t.head_dim = t.hidden_size / t.num_attention_heads;
    t.rms_norm_eps = get_d(root, "rms_norm_eps", t.rms_norm_eps);
    t.rope_theta = get_d(root, "rope_theta", t.rope_theta);
    // phi applies the textbook 1/sqrt(head_dim) attention scale; the config
    // carries no multiplier.
    if (t.head_dim > 0)
        t.attention_scale =
            1.0 / std::sqrt(static_cast<double>(t.head_dim));
    t.bos_id = get_i(root, "bos_token_id", t.bos_id);
    t.eos_id = get_i(root, "eos_token_id", t.eos_id);
    t.pad_id = get_i(root, "pad_token_id", t.pad_id);
    return true;
}

} // namespace

ParsedTensor classify_phi_tensor(const std::string &name)
{
    using K = TensorKind;
    if (name == "model.embed_tokens.weight") return {K::Embed, -1};
    if (name == "lm_head.weight") return {K::LmHead, -1};
    if (name == "model.norm.weight") return {K::FinalNorm, -1};

    auto P = [&](K kind, const std::string &suffix) -> std::optional<ParsedTensor> {
        return match_layer(name, re_linear(suffix), kind);
    };

    if (auto p = P(K::AttnQKV, R"(self_attn\.qkv_proj\.weight$)")) return *p;
    if (auto p = P(K::AttnO, R"(self_attn\.o_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpGateUp, R"(mlp\.gate_up_proj\.weight$)")) return *p;
    if (auto p = P(K::MlpDown, R"(mlp\.down_proj\.weight$)")) return *p;
    if (auto p = P(K::InputNorm, R"(input_layernorm\.weight$)")) return *p;
    if (auto p = P(K::PostNorm, R"(post_attention_layernorm\.weight$)")) return *p;

    return {K::Unknown, -1};
}

bool phi_tensor_is_nvfp4(TensorKind kind)
{
    switch (kind) {
    case TensorKind::Embed:
    case TensorKind::LmHead:
    case TensorKind::AttnQKV:
    case TensorKind::AttnO:
    case TensorKind::MlpGateUp:
    case TensorKind::MlpDown:
        return true;
    default:
        return false;
    }
}

// ── Topology ────────────────────────────────────────────────────────────────

PhiTopology load_phi_topology(const std::string &index_path)
{
    PhiTopology top;
    std::ifstream f(index_path);
    if (!f) return top;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) return top;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return top;

    auto get_int = [&](const char *k, int dflt) {
        int64_t v;
        simdjson::ondemand::value x;
        if (root[k].get(x) != simdjson::SUCCESS ||
            x.get_int64().get(v) != simdjson::SUCCESS)
            return dflt;
        return static_cast<int>(v);
    };
    auto get_dbl = [&](const char *k, double dflt) {
        if (auto x = root[k].get_double(); x.error() == simdjson::SUCCESS)
            return x.value();
        return dflt;
    };

    top.num_layers = get_int("num_layers", 0);
    top.hidden_size = get_int("hidden_size", 0);
    top.intermediate_size = get_int("intermediate_size", 0);
    top.vocab_size = get_int("vocab_size", 0);
    top.num_attention_heads = get_int("num_attention_heads", 0);
    top.num_key_value_heads = get_int("num_key_value_heads", 0);
    top.head_dim = get_int("head_dim", 0);
    top.rms_norm_eps = get_dbl("rms_norm_eps", 1e-5);
    top.rope_theta = get_dbl("rope_theta", 250000.0);
    top.attention_scale = get_dbl("attention_scale", 0.08838834764831845);
    top.bos_id = get_int("bos_id", -1);
    top.eos_id = get_int("eos_id", -1);
    top.pad_id = get_int("pad_id", -1);
    top.total_pinned_tensors = get_int("total_pinned_tensors", 0);

    std::string verr;
    if (!ParseVramBlock(json, top.vram, verr)) {
        PhiTopology failed;
        failed.error_msg = verr;
        return failed;
    }
    return top;
}

arrow::Status WritePhiModelIndex(const std::string &output_root,
                                 const PhiTopology &top)
{
    namespace fs = std::filesystem;
    auto dir = fs::path(output_root) / "logestix";
    if (!fs::exists(dir)) {
        auto st = fs::create_directories(dir);
        if (!st) return arrow::Status::IOError("cannot create ", dir.string());
    }

    std::ostringstream j;
    j << "{\n"
      << "  \"format\": \"logestix-phi\",\n"
      << "  \"model\": \"Phi-4\",\n"
      << "  \"num_layers\": " << top.num_layers
      << ",\n  \"hidden_size\": " << top.hidden_size
      << ",\n  \"intermediate_size\": " << top.intermediate_size
      << ",\n  \"vocab_size\": " << top.vocab_size
      << ",\n  \"num_attention_heads\": " << top.num_attention_heads
      << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
      << ",\n  \"head_dim\": " << top.head_dim
      << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
      << ",\n  \"rope_theta\": " << top.rope_theta
      << ",\n  \"attention_scale\": " << top.attention_scale
      << ",\n  \"bos_id\": " << top.bos_id
      << ",\n  \"eos_id\": " << top.eos_id
      << ",\n  \"pad_id\": " << top.pad_id
      << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
      << ",\n" << FormatVramJson(top.vram)
      << "\n}";

    std::ofstream out(dir / "model.index.json");
    out << j.str();
    return arrow::Status::OK();
}

// ── Converter ──────────────────────────────────────────────────────────────

arrow::Status ConvertPhiModel(const std::string &input_path,
                              const std::string &output_root)
{
    namespace fs = std::filesystem;

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    int max_layer = -1;
    for (auto &shard : model.shards)
        for (auto &t : shard.tensors) {
            auto p = classify_phi_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;
            if (p.layer >= 0) max_layer = std::max(max_layer, p.layer);
        }
    if (max_layer < 0)
        return arrow::Status::Invalid("no recognized layer tensors in ",
                                      input_path);

    PhiTopology top;
    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    if (!read_phi_config(cfg_path.string(), top))
        return arrow::Status::Invalid("failed to parse ", cfg_path.string());

    const int cfg_num_layers = top.num_layers;
    top.num_layers = max_layer + 1;

    // Source reads must be complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    auto expected = generate_phi_expected_names(top);
    std::vector<std::string> present_names;
    absl::flat_hash_set<int> present_layers;
    collect_phi_present(model, present_names, present_layers);
    ARROW_RETURN_NOT_OK(
        integrity::CheckTensorCoverage(expected, present_names, "phi"));
    ARROW_RETURN_NOT_OK(
        integrity::CheckLayerContiguity(top.num_layers, present_layers));

    int tensor_vocab = 0, tensor_hidden = 0, tensor_intermediate = 0,
        tensor_layers = 0;
    phi_derive_actual_shapes(model, tensor_vocab, tensor_hidden,
                             tensor_intermediate, tensor_layers);
    ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields({
        {"num_layers", {cfg_num_layers, tensor_layers}},
        {"hidden_size", {top.hidden_size, tensor_hidden}},
        {"vocab_size", {top.vocab_size, tensor_vocab}},
        {"intermediate_size", {top.intermediate_size, tensor_intermediate}},
    }));

    auto pinned_dir = fs::path(output_root) / "logestix" / "pinned";
    if (!fs::exists(pinned_dir)) fs::create_directories(pinned_dir);

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;
    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_phi_tensor(t.name);
            if (p.kind == TensorKind::Unknown) continue;

            auto data = read_tensor_data(input_path, shard, t);
            if (data.empty())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            const bool nvfp4 = phi_tensor_is_nvfp4(p.kind);

            if (p.kind == TensorKind::Embed || p.kind == TensorKind::LmHead) {
                if (nvfp4) {
                    DenseNvfp4Record r;
                    r.name = t.name;
                    r.shape = t.shape;
                    r.dtype = t.dtype;
                    if (!gpu::quantize_tensor_gpu(data.data(), t.numel(),
                                                  t.dtype, nullptr, t.shape,
                                                  {}, r.packed, r.scales,
                                                  r.global_scale))
                        r.packed.clear();
                    global_nvfp4.push_back(std::move(r));
                } else {
                    TensorRecord r;
                    r.name = t.name;
                    r.dtype = t.dtype;
                    r.shape = t.shape;
                    r.data = std::move(data);
                    global_raw.push_back(std::move(r));
                }
                ++total_pinned;
                continue;
            }
            if (p.kind == TensorKind::FinalNorm) {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                continue;
            }

            const int layer = p.layer;
            if (nvfp4) {
                // qkv_proj and gate_up_proj are already fused in the source,
                // so each is quantized directly into one record with a single
                // global scale (the decode layout the loader expects).
                DenseNvfp4Record r;
                r.name = t.name;
                r.shape = t.shape;
                r.dtype = t.dtype;
                if (!gpu::quantize_tensor_gpu(data.data(), t.numel(), t.dtype,
                                              nullptr, t.shape, {}, r.packed,
                                              r.scales, r.global_scale))
                    r.packed.clear();
                nvfp4_by_layer[layer].push_back(std::move(r));
            } else {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[layer].push_back(std::move(r));
            }
            ++total_pinned;
        }
    }

    if (!global_raw.empty()) {
        auto batch = PinnedSchema::MakeBatch(global_raw);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(
            WriteRecordBatchToFile(*batch, pinned_dir / "global.arrow"));
    }
    if (!global_nvfp4.empty()) {
        auto batch = DenseNvfp4Schema::MakeBatch(global_nvfp4);
        if (!batch.ok()) return batch.status();
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(
            *batch, pinned_dir / "global_nvfp4.arrow"));
    }
    for (auto &[layer, recs] : nvfp4_by_layer) {
        auto batch = DenseNvfp4Schema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        auto fname = absl::StrCat("layer_", layer, "_nvfp4.arrow");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / fname));
    }
    for (auto &[layer, recs] : raw_by_layer) {
        auto batch = PinnedSchema::MakeBatch(recs);
        if (!batch.ok()) return batch.status();
        auto fname = absl::StrCat("layer_", layer, ".arrow");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(*batch, pinned_dir / fname));
    }

    top.total_pinned_tensors = static_cast<int>(total_pinned);

    {
        VramGeometry vg;
        vg.hidden_dim = top.hidden_size;
        vg.kv_dim = top.head_dim * top.num_key_value_heads;
        vg.num_attn_layers = top.num_layers;
        vg.context_length = ReadContextLength(cfg_path.string());

        std::vector<VramTensorInfo> vram_inventory;
        for (const auto &shard : model.shards)
            for (const auto &t : shard.tensors)
                vram_inventory.push_back(
                    {t.name, t.numel(), t.byte_size()});

        top.vram = EstimateVram(
            vram_inventory,
            {+[](const std::string &n) {
                 return phi_tensor_is_nvfp4(
                     classify_phi_tensor(n).kind);
             },
             nullptr},
            vg);
        lsxcommon::log::info(absl::StrFormat(
            "VRAM profile: min=%s optimal=%s max=%s (ctx=%d)",
            lsxcommon::FormatBytesHuman(top.vram.min_bytes),
            lsxcommon::FormatBytesHuman(top.vram.optimal_bytes),
            lsxcommon::FormatBytesHuman(top.vram.max_bytes),
            top.vram.context_tokens));
    }

    auto st = WritePhiModelIndex(output_root, top);
    if (!st.ok()) return st;

    auto tok_path = fs::path(input_path) / "tokenizer.json";
    if (!fs::exists(tok_path))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tok_path.string(), pinned_dir,
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true,
                                       .pad_required = false}));

    return arrow::Status::OK();
}

arrow::Status RecomputePhiVram(const std::string &input_path,
                               const std::string &output_root)
{
    namespace fs = std::filesystem;
    auto idx = fs::path(output_root) / "logestix" / "model.index.json";
    std::ifstream f(idx);
    if (!f) return arrow::Status::IOError("cannot open ", idx.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    PhiTopology top;
    std::ifstream cf(fs::path(input_path) / "config.json");
    if (cf) {
        read_phi_config((fs::path(input_path) / "config.json").string(), top);
    }

    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.context_length =
        ReadContextLength((fs::path(input_path) / "config.json").string());

    std::vector<VramTensorInfo> vram_inventory;
    for (const auto &shard : model.shards)
        for (const auto &t : shard.tensors)
            vram_inventory.push_back({t.name, t.numel(), t.byte_size()});

    VramProfile profile = EstimateVram(
        vram_inventory,
        {+[](const std::string &n) {
             return phi_tensor_is_nvfp4(classify_phi_tensor(n).kind);
         },
         nullptr},
        vg);

    std::ofstream out(idx, std::ios::trunc);
    if (!out) return arrow::Status::IOError("cannot write ", idx.string());
    out << ReplaceVramBlock(json, profile);
    return arrow::Status::OK();
}

} // namespace phi
} // namespace lsxcommon
