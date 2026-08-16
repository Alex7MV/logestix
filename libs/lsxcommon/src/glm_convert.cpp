#include "lsxcommon/glm_convert.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/diagnostics.h"
#include "lsxcommon/schema.h"

#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <simdjson.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <string>
#include <vector>
#include <iterator>
#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

namespace lsxcommon {
// Forward declaration (defined in converter.cpp). Kept out of converter.h so
// that NVCC-translated TUs (e.g. testmodel_gpu.cu, which only needs the PIMPL
// Converter class) never parse abseil's flat_hash_map header.
arrow::Status SaveSharedExpertsToArrow(
    const absl::flat_hash_map<int, ExpertData> &shared_by_layer,
    const std::filesystem::path &output_root);

namespace glm {

namespace {

// Layers with dense MLP (experts absent) — GLM-5.2: 0,1,2.
constexpr int kDenseLayerMax = 2;

// model.layers.N.mlp.experts.E.{gate,down,up}_proj.weight
std::optional<ParsedTensor> parse_routed(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|down|up)_proj\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::RoutedExpert;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    absl::SimpleAtoi(m[2].str(), &p.expert);
    p.weight_id = (m[3] == "gate") ? 1 : (m[3] == "down") ? 2 : 3;
    p.base = m[3].str() + "_proj";
    return p;
}

// model.layers.N.mlp.shared_experts.{gate,down,up}_proj.weight
std::optional<ParsedTensor> parse_shared(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.mlp\.shared_experts\.(gate|down|up)_proj\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::SharedExpert;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    p.weight_id = (m[2] == "gate") ? 1 : (m[2] == "down") ? 2 : 3;
    p.base = m[2].str() + "_proj";
    return p;
}

// model.layers.N.mlp.gate.weight | .e_score_correction_bias | .bias
std::optional<ParsedTensor> parse_router(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias|bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::Router;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    p.base = (m[2] == "e_score_correction_bias") ? "e_score_correction_bias"
                                                 : "gate";
    return p;
}

// model.layers.N.mlp.{gate,down,up}_proj.weight  (dense layers 0-2)
std::optional<ParsedTensor> parse_dense(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.mlp\.(gate|down|up)_proj\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    int layer = 0;
    absl::SimpleAtoi(m[1].str(), &layer);
    if (layer > kDenseLayerMax) return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::DenseMlp;
    p.layer = layer;
    p.base = m[2].str() + "_proj";
    return p;
}

// model.layers.N.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//                           kv_a_layernorm,kv_b_proj,o_proj}.weight
// model.layers.78.eh_proj.weight  (MTP hidden-state projector)
std::optional<ParsedTensor> parse_attention(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.self_attn\.(q_a_proj|q_a_layernorm|q_b_proj|kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj|o_proj)\.(weight|bias))");
    static const std::regex re_eh(
        R"(model\.layers\.78\.eh_proj\.weight)");
    std::smatch m;
    if (std::regex_match(name, m, re)) {
        ParsedTensor p;
        p.kind = TensorKind::Attention;
        absl::SimpleAtoi(m[1].str(), &p.layer);
        p.base = m[2].str();
        return p;
    }
    if (std::regex_match(name, m, re_eh)) {
        ParsedTensor p;
        p.kind = TensorKind::Attention;
        p.layer = 78;
        p.base = "eh_proj";
        return p;
    }
    return std::nullopt;
}

// model.layers.N.self_attn.indexer.{wq_b,wk,weights_proj,k_norm}.{weight,bias}
std::optional<ParsedTensor> parse_indexer(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.self_attn\.indexer\.(wq_b|wk|weights_proj|k_norm)\.(weight|bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::Indexer;
    absl::SimpleAtoi(m[1].str(), &p.layer);
    p.base = m[2].str();
    return p;
}

// model.layers.N.(input_layernorm|post_attention_layernorm|enorm|hnorm).weight
// model.layers.N.shared_head.norm.weight
std::optional<ParsedTensor> parse_layer_norm(const std::string &name)
{
    static const std::regex re(
        R"(model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm|enorm|hnorm)\.weight)");
    static const std::regex re_shared_head(
        R"(model\.layers\.(\d+)\.shared_head\.norm\.weight)");
    std::smatch m;
    if (std::regex_match(name, m, re)) {
        ParsedTensor p;
        p.kind = TensorKind::LayerNorm;
        absl::SimpleAtoi(m[1].str(), &p.layer);
        p.base = m[2].str();
        return p;
    }
    if (std::regex_match(name, m, re_shared_head)) {
        ParsedTensor p;
        p.kind = TensorKind::LayerNorm;
        absl::SimpleAtoi(m[1].str(), &p.layer);
        p.base = "shared_head.norm";
        return p;
    }
    return std::nullopt;
}

// embed_tokens.weight / lm_head.weight / model.norm.weight
std::optional<ParsedTensor> parse_global(const std::string &name)
{
    std::string base;
    if (name == "model.embed_tokens.weight") base = "embed_tokens";
    else if (name == "lm_head.weight") base = "lm_head";
    else if (name == "model.norm.weight") base = "model.norm";
    else return std::nullopt;
    ParsedTensor p;
    p.kind = TensorKind::Global;
    p.base = base;
    return p;
}

std::vector<long long> read_int_array(
    simdjson::ondemand::object &root,
    const char *key,
    bool &ok)
{
    std::vector<long long> out;
    simdjson::ondemand::array arr;
    if (root[key].get_array().get(arr) != simdjson::SUCCESS) {
        ok = false;
        return out;
    }
    for (simdjson::ondemand::value v : arr) {
        int64_t x = 0;
        if (v.get_int64().get(x) != simdjson::SUCCESS) {
            ok = false;
            return out;
        }
        out.push_back(static_cast<long long>(x));
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

Topology load_topology(const std::string &index_json_path)
{
    namespace fs = std::filesystem;
    Topology topo;

    std::ifstream f(index_json_path);
    if (!f) {
        topo.error_msg = "cannot open " + index_json_path;
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "JSON parse error in " + index_json_path;
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "expected JSON object in " + index_json_path;
        return topo;
    }

    int64_t nl = 0, epl = 0, te = 0;
    root["num_layers"].get_int64().get(nl);
    root["experts_per_layer"].get_int64().get(epl);
    root["total_experts"].get_int64().get(te);
    topo.num_layers = static_cast<long long>(nl);
    topo.experts_per_layer = static_cast<long long>(epl);
    topo.total_experts = static_cast<long long>(te);

    if (topo.num_layers <= 0) {
        topo.error_msg = "missing/invalid num_layers in " + index_json_path;
        return topo;
    }

    bool ok = true;
    // Dense layers are optional (quiescent default: empty = all dense);
    // moe_layers may be empty to denote "uniform grid, all layers MoE".
    topo.dense_layers = read_int_array(root, "dense_layers", ok);
    if (!ok) {
        topo.error_msg = "invalid dense_layers in " + index_json_path;
        return topo;
    }
    topo.moe_layers = read_int_array(root, "moe_layers", ok);
    if (!ok) {
        topo.error_msg = "invalid moe_layers in " + index_json_path;
        return topo;
    }
    topo.indexer_layers = read_int_array(root, "indexer_layers", ok);
    if (!ok) {
        topo.error_msg = "invalid indexer_layers in " + index_json_path;
        return topo;
    }
    return topo;
}

ParsedTensor classify(const std::string &name)
{
    if (auto p = parse_routed(name)) return *p;
    if (auto p = parse_shared(name)) return *p;
    if (auto p = parse_router(name)) return *p;
    if (auto p = parse_dense(name)) return *p;
    if (auto p = parse_attention(name)) return *p;
    if (auto p = parse_indexer(name)) return *p;
    if (auto p = parse_layer_norm(name)) return *p;
    if (auto p = parse_global(name)) return *p;
    return {};
}

// ── Converter ──────────────────────────────────────────────────────────────

namespace {

struct GlmExpertKey {
    int layer;
    int expert;
    bool operator==(const GlmExpertKey &o) const noexcept {
        return layer == o.layer && expert == o.expert;
    }
};
struct GlmExpertKeyHash {
    std::size_t operator()(const GlmExpertKey &k) const noexcept {
        return static_cast<std::size_t>(k.layer) * 65536U
             + static_cast<std::size_t>(k.expert);
    }
};

// GLM writes RAW BF16 (no paired scales); straight into an ExpertData.
void AssignExpertWeight(ExpertData &ed, int weight_id,
                        std::vector<uint8_t> &&raw,
                        const std::vector<int64_t> &shape,
                        const std::string &dtype)
{
    switch (weight_id) {
    case 1: ed.w1_raw = std::move(raw); ed.w1_shape = shape; ed.w1_dtype = dtype; break;
    case 2: ed.w2_raw = std::move(raw); ed.w2_shape = shape; ed.w2_dtype = dtype; break;
    case 3: ed.w3_raw = std::move(raw); ed.w3_shape = shape; ed.w3_dtype = dtype; break;
    default: break;
    }
}

bool ExpertComplete(const ExpertData &ed)
{
    return !ed.w1_raw.empty() && !ed.w2_raw.empty() && !ed.w3_raw.empty();
}

// GLM-specific per-layer pinned saver. Unlike SavePinnedTensors (converter.cpp),
// GLM tensor names carry a `model.layers.N.` prefix that the "layers\.(\d+)"
// regex full-match does NOT accept (verified: regex_match("model.layers.0..."
// fails). Grouping keys off glm::classify's layer index instead.
arrow::Status SaveGlmPinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_root / "pinned");

    absl::btree_map<int, std::vector<TensorRecord>> by_layer;
    std::vector<TensorRecord> global;

    for (const auto &t : tensors) {
        auto p = classify(t.name);
        if (p.layer >= 0)
            by_layer[p.layer].push_back(t);
        else
            global.push_back(t);
    }

    if (!global.empty()) {
        auto path = output_root / "pinned" / "global.arrow";
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(global));
        diag::ValidateBatchBuffers(*batch, "SaveGlmPinnedTensors global");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    }

    for (const auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned"
            / absl::StrCat("layer_", layer, ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        diag::ValidateBatchBuffers(*batch,
            absl::StrCat("SaveGlmPinnedTensors layer_", layer).c_str());
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    }

    return arrow::Status::OK();
}

// GLM-specific IDX writer (format "logestix-glm").
void WriteGlmModelIndex(
    const std::string &model_name,
    const Topology &topo,
    int64_t total_pinned,
    int64_t total_experts,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    std::ofstream f(output_root / "model.index.json");
    if (!f) return;

    f << "{\n";
    f << "  \"format\": \"logestix-glm\",\n";
    f << "  \"model\": \"" << model_name << "\",\n";
    f << "  \"num_layers\": " << topo.num_layers << ",\n";
    f << "  \"experts_per_layer\": " << topo.experts_per_layer << ",\n";
    f << "  \"moe_layers\": [";
    for (size_t i = 0; i < topo.moe_layers.size(); ++i) {
        if (i) f << ",";
        f << topo.moe_layers[i];
    }
    f << "],\n";
    f << "  \"dense_layers\": [";
    for (size_t i = 0; i < topo.dense_layers.size(); ++i) {
        if (i) f << ",";
        f << topo.dense_layers[i];
    }
    f << "],\n";
    f << "  \"indexer_layers\": [";
    for (size_t i = 0; i < topo.indexer_layers.size(); ++i) {
        if (i) f << ",";
        f << topo.indexer_layers[i];
    }
    f << "],\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << "\n";
    f << "}\n";
    f.close();
}

} // namespace

arrow::Status ConvertGlmModel(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    if (input_path.empty())
        return arrow::Status::Invalid("input path not set");
    if (output_root.empty())
        return arrow::Status::Invalid("output path not set");

    fs::create_directories(output_root);

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                       input_path);

    // (2) config.json is mandatory; parsed for geometry cross-checks.
    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);

    int cfg_num_layers = 0;
    int cfg_hidden_size = 0;
    int cfg_vocab_size = 0;
    int cfg_experts_per_layer = 0;
    {
        std::ifstream cf(cfg_path);
        if (!cf)
            return arrow::Status::IOError("cannot open ", cfg_path.string());
        std::string cj((std::istreambuf_iterator<char>(cf)),
                       std::istreambuf_iterator<char>());
        simdjson::padded_string cpj(cj.data(), cj.size());
        simdjson::ondemand::parser cp;
        simdjson::ondemand::document cd;
        if (cp.iterate(cpj).get(cd) != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json parse failed");
        auto root = cd.get_object();
        if (root.error() != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json root not an object");
        auto get_i = [&](simdjson::ondemand::object &o, const char *k, int dflt) {
            int64_t v;
            if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
            return static_cast<int>(v);
        };
        cfg_num_layers = get_i(root.value(), "num_layers", 0);
        if (cfg_num_layers <= 0)
            cfg_num_layers = get_i(root.value(), "num_hidden_layers", 0);
        cfg_hidden_size = get_i(root.value(), "hidden_size", 0);
        cfg_vocab_size = get_i(root.value(), "vocab_size", 0);
        cfg_experts_per_layer = get_i(root.value(), "num_local_experts", 0);
        if (cfg_experts_per_layer <= 0)
            cfg_experts_per_layer = get_i(root.value(), "num_experts", 0);
    }

    // (3) Verify every tensor read is complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    absl::flat_hash_map<GlmExpertKey, ExpertData, GlmExpertKeyHash> partial;
    absl::flat_hash_map<int, ExpertData> shared_by_layer;
    std::vector<TensorRecord> pinned_tensors;

    int experts_processed = 0;
    int pinned_processed = 0;
    int shard_idx = 0;
    int max_layer = -1;
    int max_experts_per_layer = 0;
    absl::flat_hash_map<int, int> experts_per_layer;
    absl::btree_set<int> moe_layers;
    absl::btree_set<int> dense_layers;
    absl::btree_set<int> indexer_layers;

    for (const auto &shard : model.shards) {
        ++shard_idx;
        spdlog::info("[{}/{}] Scanning {} ...", shard_idx, model.shards.size(),
                     shard.filename);

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf)
            return arrow::Status::IOError("cannot open ", shard.filename);
        int64_t sz = sf.tellg();
        if (sz <= 0)
            return arrow::Status::IOError("zero/negative size: ", shard.filename);
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char*>(shard_buf.data()), sz);
        if (!sf)
            return arrow::Status::IOError("cannot read ", shard.filename);

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto parsed = classify(tensor.name);

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz)
                return arrow::Status::Invalid(
                    shard.filename, " tensor ", tensor.name,
                    " offset out of range");
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed.kind == TensorKind::RoutedExpert) {
                GlmExpertKey key{parsed.layer, parsed.expert};
                auto &exp = partial[key];
                max_layer = std::max(max_layer, parsed.layer);
                moe_layers.insert(parsed.layer);
                AssignExpertWeight(exp, parsed.weight_id, std::move(raw),
                                   tensor.shape, tensor.dtype);

                if (ExpertComplete(exp)) {
                    auto status = [&]() -> arrow::Status {
                        ARROW_ASSIGN_OR_RAISE(auto packed,
                            PackExpertNVFP4(std::move(exp)));
                        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                            packed, key.layer, key.expert, output_root));
                        return arrow::Status::OK();
                    }();
                    if (!status.ok())
                        spdlog::error("  Failed to pack expert layer={} "
                                      "expert={}: {}",
                                      key.layer, key.expert,
                                      status.ToString());
                    else {
                        ++experts_processed;
                        ++experts_per_layer[parsed.layer];
                    }
                    partial.erase(key);
                }
            } else if (parsed.kind == TensorKind::SharedExpert) {
                auto &sexp = shared_by_layer[parsed.layer];
                max_layer = std::max(max_layer, parsed.layer);
                AssignExpertWeight(sexp, parsed.weight_id, std::move(raw),
                                   tensor.shape, tensor.dtype);
            } else if (parsed.kind == TensorKind::DenseMlp) {
                dense_layers.insert(parsed.layer);
                max_layer = std::max(max_layer, parsed.layer);
                TensorRecord t;
                t.name = tensor.name; t.dtype = tensor.dtype;
                t.shape = tensor.shape; t.data = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            } else if (parsed.kind == TensorKind::Indexer) {
                indexer_layers.insert(parsed.layer);
                max_layer = std::max(max_layer, parsed.layer);
                TensorRecord t;
                t.name = tensor.name; t.dtype = tensor.dtype;
                t.shape = tensor.shape; t.data = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            } else if (parsed.kind != TensorKind::Unknown) {
                // Attention, Router, LayerNorm, Global → pinned.
                max_layer = std::max(max_layer, parsed.layer);
                TensorRecord t;
                t.name = tensor.name; t.dtype = tensor.dtype;
                t.shape = tensor.shape; t.data = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            } else {
                spdlog::warn("  Unclassified tensor: {} (skipped)",
                             tensor.name);
            }
        }
    }

    for (auto &[key, raw] : partial)
        return arrow::Status::Invalid(
            "incomplete expert layer=", key.layer,
            " expert=", key.expert,
            " (w1=", raw.w1_raw.size(), " bytes / w2=", raw.w2_raw.size(),
            " / w3=", raw.w3_raw.size(), ")");
    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    // (4) Tensor coverage: required globals + per-layer MLP/route essentials.
    {
        std::vector<std::string> present_names;
        absl::flat_hash_set<int> present_layers;
        for (auto &shard : model.shards) {
            for (auto &t : shard.tensors) {
                auto p = classify(t.name);
                if (p.kind == TensorKind::Unknown) continue;
                present_names.push_back(t.name);
                if (p.layer >= 0) present_layers.insert(p.layer);
            }
        }

        std::vector<std::string> expected;
        expected.push_back("model.embed_tokens.weight");
        expected.push_back("lm_head.weight");
        expected.push_back("model.norm.weight");
        for (int l : dense_layers) {
            std::string lb = absl::StrCat("model.layers.", l, ".");
            expected.push_back(lb + "mlp.gate_proj.weight");
            expected.push_back(lb + "mlp.up_proj.weight");
            expected.push_back(lb + "mlp.down_proj.weight");
        }
        for (int l : moe_layers) {
            std::string lb = absl::StrCat("model.layers.", l, ".");
            expected.push_back(lb + "mlp.gate.weight");
            expected.push_back(lb + "mlp.gate.e_score_correction_bias");
            expected.push_back(lb + "mlp.shared_experts.gate_proj.weight");
            expected.push_back(lb + "mlp.shared_experts.up_proj.weight");
            expected.push_back(lb + "mlp.shared_experts.down_proj.weight");
        }

        ARROW_RETURN_NOT_OK(
            integrity::CheckTensorCoverage(expected, present_names, "glm"));
    }

    // (6) Geometry: config-declared vs tensor-derived (fields with a declared
    // value only — GLM configs vary in which keys they carry).
    {
        int embed_vocab = 0, embed_hidden = 0;
        for (auto &shard : model.shards) {
            for (auto &t : shard.tensors) {
                if (t.name == "model.embed_tokens.weight" && t.shape.size() >= 2) {
                    embed_vocab = static_cast<int>(t.shape[0]);
                    embed_hidden = static_cast<int>(t.shape[1]);
                }
            }
        }
        std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> geom;
        if (cfg_num_layers > 0)
            geom.push_back({"num_layers", {cfg_num_layers, max_layer + 1}});
        if (cfg_hidden_size > 0)
            geom.push_back({"hidden_size", {cfg_hidden_size, embed_hidden}});
        if (cfg_vocab_size > 0)
            geom.push_back({"vocab_size", {cfg_vocab_size, embed_vocab}});
        if (cfg_experts_per_layer > 0)
            geom.push_back({"experts_per_layer",
                            {cfg_experts_per_layer, max_experts_per_layer}});
        if (!geom.empty())
            ARROW_RETURN_NOT_OK(integrity::CheckGeometryFields(geom));
    }

    spdlog::info("Saving {} pinned tensors...", pinned_tensors.size());
    ARROW_RETURN_NOT_OK(SaveGlmPinnedTensors(pinned_tensors, output_root));

    spdlog::info("Packing {} shared-expert layers (NVFP4)...",
                 shared_by_layer.size());
    ARROW_RETURN_NOT_OK(SaveSharedExpertsToArrow(shared_by_layer, output_root));

    Topology topo;
    topo.num_layers = (max_layer >= 0) ? max_layer + 1 : 0;
    topo.experts_per_layer = max_experts_per_layer;
    topo.total_experts = static_cast<long long>(experts_processed);
    topo.moe_layers.assign(moe_layers.begin(), moe_layers.end());
    topo.dense_layers.assign(dense_layers.begin(), dense_layers.end());
    topo.indexer_layers.assign(indexer_layers.begin(), indexer_layers.end());

    WriteGlmModelIndex(
        fs::path(input_path).filename().string(),
        topo,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        output_root);

    // (8) Build and verify the binary tokenizer (mandatory).
    auto tokenizer_json = fs::path(input_path) / "tokenizer.json";
    if (!fs::exists(tokenizer_json))
        return arrow::Status::Invalid("tokenizer.json not found in ",
                                      input_path);
    ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
        tokenizer_json.string(), output_root / "pinned",
        integrity::TokenizerSpecialIds{.bos_required = true,
                                       .eos_required = true}));

    // (9) The converted output must be a loadable, verified model.
    auto vres = integrity::VerifyConvertedModel(output_root);
    if (!vres.ok())
        return arrow::Status::Invalid(
            "output verification failed: ", vres.status().ToString());

    spdlog::info("Done — {} pinned tensors, {} experts processed, "
                 "incomplete {}",
                 pinned_processed, experts_processed, partial.size());
    return arrow::Status::OK();
}

} // namespace glm
} // namespace lsxcommon