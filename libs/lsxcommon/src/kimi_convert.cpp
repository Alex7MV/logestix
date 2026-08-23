#include "lsxcommon/kimi_convert.h"
#include "kimi_inference_kernels.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/kimi_inference_load.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/diagnostics.h"
#include "lsxcommon/schema.h"

#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <cuda_runtime.h>
#include <simdjson.h>

#include <absl/container/btree_map.h>
#include <absl/container/btree_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

namespace lsxcommon {
// Defined in converter.cpp (kept out of converter.h so NVCC TUs never parse
// abseil container headers).
arrow::Status SaveSharedExpertsToArrow(
    const absl::flat_hash_map<int, ExpertData> &shared_by_layer,
    const std::filesystem::path &output_root);

namespace kimi {
namespace {

// Strips the multimodal checkpoint wrapper: "language_model.model." or
// "language_model.". Returns the inner name.
std::string strip_language_model(const std::string &name)
{
    const std::string full = "language_model.model.";
    if (name.rfind(full, 0) == 0) return name.substr(full.size());
    const std::string head = "language_model.";
    if (name.rfind(head, 0) == 0) return name.substr(head.size());
    return name;
}

bool parse_int(const std::string &s, int &out)
{
    return absl::SimpleAtoi(s, &out);
}

// Matches the GLM/ExpertData convention: w1=gate, w2=down, w3=up.
int weight_id_for(const std::string &proj)
{
    if (proj == "gate_proj") return 1;
    if (proj == "down_proj") return 2;
    if (proj == "up_proj") return 3;
    return 0;
}

// model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight_{packed,scale}
std::optional<KimiParsedTensor> parse_routed_expert(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_(packed|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::RoutedExpert;
    parse_int(m[1].str(), p.layer);
    parse_int(m[2].str(), p.expert);
    p.weight_id = weight_id_for(m[3].str());
    p.base = m[3].str() + ".weight_" + m[4].str();
    return p;
}

// model.layers.N.mlp.shared_experts.{gate,up,down}_proj.{weight,weight_packed,
// weight_scale}
std::optional<KimiParsedTensor> parse_shared_expert(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.(weight|weight_packed|weight_scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::SharedExpert;
    parse_int(m[1].str(), p.layer);
    p.expert = 0;
    p.weight_id = weight_id_for(m[2].str());
    p.base = "shared_experts." + m[2].str() + "." + m[3].str();
    return p;
}

// model.layers.N.mlp.gate.{weight,e_score_correction_bias}
std::optional<KimiParsedTensor> parse_router(const std::string &name)
{
    static const std::regex re(R"(layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Router;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// model.0.mlp / layers.0.mlp.{gate,up,down}_proj.{weight,weight_packed,weight_scale}
// Dense MLP only exists on layer 0 (first_k_dense_replace=1).
std::optional<KimiParsedTensor> parse_dense_mlp(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(0)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|weight_packed|weight_scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::DenseMlp;
    p.layer = 0;
    p.weight_id = weight_id_for(m[2].str());
    p.base = m[2].str();
    return p;
}

// model.layers.N.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//                           kv_a_layernorm,kv_b_proj,o_proj}.{weight,bias}
std::optional<KimiParsedTensor> parse_attention(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.self_attn\.(q_a_proj|q_a_layernorm|q_b_proj|kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj|o_proj)\.(weight|bias))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Attention;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// model.layers.N.(input_layernorm|post_attention_layernorm).weight
std::optional<KimiParsedTensor> parse_layer_norm(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::LayerNorm;
    parse_int(m[1].str(), p.layer);
    p.base = m[2].str();
    return p;
}

// embed_tokens.weight / lm_head.weight / model.norm.weight (post-strip)
std::optional<KimiParsedTensor> parse_global(const std::string &name)
{
    std::string base;
    if (name == "embed_tokens.weight") base = "embed_tokens";
    else if (name == "lm_head.weight") base = "lm_head";
    else if (name == "norm.weight") base = "model.norm";
    else return std::nullopt;
    KimiParsedTensor p;
    p.kind = KimiTensorKind::Global;
    p.base = base;
    return p;
}

} // namespace

KimiParsedTensor kimi_classify(const std::string &full_name)
{
    const std::string name = strip_language_model(full_name);
    if (auto p = parse_global(name)) return *p;
    if (auto p = parse_layer_norm(name)) return *p;
    if (auto p = parse_attention(name)) return *p;
    if (auto p = parse_router(name)) return *p;
    if (auto p = parse_shared_expert(name)) return *p;
    if (auto p = parse_routed_expert(name)) return *p;
    if (auto p = parse_dense_mlp(name)) return *p;
    return {};
}

KimiTopology kimi_load_topology(const std::string &index_json_path)
{
    namespace fs = std::filesystem;
    KimiTopology topo;

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
        topo.error_msg = "simdjson iterate failed for " + index_json_path;
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "root is not an object: " + index_json_path;
        return topo;
    }
    simdjson::ondemand::object weight_map;
    if (root["weight_map"].get_object().get(weight_map) != simdjson::SUCCESS) {
        topo.error_msg = "missing weight_map in " + index_json_path;
        return topo;
    }

    std::set<long long> moe, dense;
    long long max_layer = -1;
    long long max_expert = -1;
    for (simdjson::ondemand::field field : weight_map) {
        std::string_view key_sv;
        if (field.unescaped_key().get(key_sv) != simdjson::SUCCESS) continue;
        std::string key(key_sv);
        auto parsed = kimi_classify(key);
        if (parsed.kind == KimiTensorKind::Router) {
            moe.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        } else if (parsed.kind == KimiTensorKind::RoutedExpert) {
            max_expert = std::max(max_expert,
                                  static_cast<long long>(parsed.expert));
            moe.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        } else if (parsed.kind == KimiTensorKind::DenseMlp) {
            dense.insert(parsed.layer);
            max_layer = std::max(max_layer,
                                 static_cast<long long>(parsed.layer));
        }
    }

    if (moe.empty()) {
        topo.error_msg = "no MoE layers found in " + index_json_path;
        return topo;
    }
    topo.num_layers = max_layer + 1;
    topo.experts_per_layer = max_expert + 1;
    topo.moe_layers.assign(moe.begin(), moe.end());
    topo.dense_layers.assign(dense.begin(), dense.end());
    return topo;
}

// ── Converter ──────────────────────────────────────────────────────────────

namespace {

struct KimiExpertKey {
    int layer;
    int expert;
    bool operator==(const KimiExpertKey &o) const noexcept {
        return layer == o.layer && expert == o.expert;
    }
};
struct KimiExpertKeyHash {
    std::size_t operator()(const KimiExpertKey &k) const noexcept {
        return static_cast<std::size_t>(k.layer) * 65536U
             + static_cast<std::size_t>(k.expert);
    }
};

// Accumulated INT4 parts of one routed expert: packed int32 payload + BF16
// group-32 scales for each of the three projections (w1=gate, w2=down, w3=up).
struct KimiExpertParts {
    std::vector<uint8_t> packed[4];        // index by weight_id 1..3
    std::vector<uint8_t> scale[4];         // BF16 bits
    std::vector<int64_t> logical_shape[4]; // [rows, cols] of the BF16 weight
    bool has_packed[4] = {};
    bool has_scale[4] = {};

    bool complete() const {
        for (int w = 1; w <= 3; ++w)
            if (!has_packed[w] || !has_scale[w]) return false;
        return true;
    }
};

// GPU-dequantizes one INT4 group-32 tensor into a BF16 byte buffer
// (offline conversion path; synchronization is fine here).
arrow::Status DequantInt4ToBf16Raw(const std::vector<uint8_t> &packed_raw,
                                   const std::vector<uint8_t> &scale_raw,
                                   const std::vector<int64_t> &logical_shape,
                                   std::vector<uint8_t> &out_bf16)
{
    if (logical_shape.size() != 2)
        return arrow::Status::Invalid("INT4 expert tensor must be 2-D");
    const int rows = static_cast<int>(logical_shape[0]);
    const int cols = static_cast<int>(logical_shape[1]);
    if (cols % 32 != 0)
        return arrow::Status::Invalid("INT4 expert cols not group-32 aligned");

    const size_t packed_bytes = packed_raw.size();
    const size_t scale_bytes = scale_raw.size();
    const size_t out_bytes =
        static_cast<size_t>(rows) * cols * sizeof(uint16_t);

    int *d_packed = nullptr;
    uint16_t *d_scales = nullptr;
    uint16_t *d_out = nullptr;
    cudaError_t ce = cudaMalloc(&d_packed, packed_bytes);
    if (ce == cudaSuccess) ce = cudaMalloc(&d_scales, scale_bytes);
    if (ce == cudaSuccess)
        ce = cudaMalloc(&d_out, out_bytes);
    if (ce != cudaSuccess) {
        cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_out);
        return arrow::Status::IOError("cudaMalloc failed: ",
                                      cudaGetErrorString(ce));
    }
    cudaMemcpy(d_packed, packed_raw.data(), packed_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scale_raw.data(), scale_bytes,
               cudaMemcpyHostToDevice);
    cudaStream_t stream = nullptr;   // legacy default stream (offline path)
    ce = KimiInt4DequantToBf16(d_packed, d_scales, rows, cols, d_out, stream);
    if (ce == cudaSuccess) ce = cudaDeviceSynchronize();
    if (ce == cudaSuccess) {
        out_bf16.resize(out_bytes);
        ce = cudaMemcpy(out_bf16.data(), d_out, out_bytes,
                        cudaMemcpyDeviceToHost);
    }
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_out);
    if (ce != cudaSuccess)
        return arrow::Status::IOError("KimiInt4DequantToBf16 failed: ",
                                      cudaGetErrorString(ce));
    return arrow::Status::OK();
}

// Dequantizes all three projections and produces an NVFP4-ready ExpertData.
arrow::Status BuildExpertData(KimiExpertParts &parts, ExpertData &ed)
{
    for (int w = 1; w <= 3; ++w) {
        std::vector<uint8_t> bf16;
        ARROW_RETURN_NOT_OK(DequantInt4ToBf16Raw(
            parts.packed[w], parts.scale[w], parts.logical_shape[w], bf16));
        std::vector<uint8_t> *dst;
        std::vector<int64_t> *dst_shape;
        std::string *dst_dtype;
        switch (w) {
        case 1: dst = &ed.w1_raw; dst_shape = &ed.w1_shape; dst_dtype = &ed.w1_dtype; break;
        case 2: dst = &ed.w2_raw; dst_shape = &ed.w2_shape; dst_dtype = &ed.w2_dtype; break;
        default: dst = &ed.w3_raw; dst_shape = &ed.w3_shape; dst_dtype = &ed.w3_dtype; break;
        }
        *dst = std::move(bf16);
        *dst_shape = parts.logical_shape[w];
        *dst_dtype = "BF16";
    }
    return arrow::Status::OK();
}

void AssignSharedWeight(ExpertData &ed, int weight_id,
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

bool SharedComplete(const ExpertData &ed)
{
    return !ed.w1_raw.empty() && !ed.w2_raw.empty() && !ed.w3_raw.empty();
}

// Kimi-specific per-layer pinned saver ("layers\.(\d+)" names do not pass the
// shared SavePinnedTensors regex; grouping keys off kimi_classify instead).
arrow::Status SaveKimiPinnedTensors(
    const std::vector<TensorRecord> &tensors,
    const fs::path &output_root)
{
    namespace fs = std::filesystem;
    fs::create_directories(output_root / "pinned");

    absl::btree_map<int, std::vector<TensorRecord>> by_layer;
    std::vector<TensorRecord> global;

    for (const auto &t : tensors) {
        auto p = kimi_classify(t.name);
        if (p.kind == KimiTensorKind::Global || p.layer < 0)
            global.push_back(t);
        else
            by_layer[p.layer].push_back(t);
    }

    if (!global.empty()) {
        auto path = output_root / "pinned" / "global.arrow";
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(global));
        diag::ValidateBatchBuffers(*batch, "SaveKimiPinnedTensors global");
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    }

    for (const auto &[layer, tens] : by_layer) {
        auto path = output_root / "pinned"
            / absl::StrCat("layer_", layer, ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        diag::ValidateBatchBuffers(*batch,
            absl::StrCat("SaveKimiPinnedTensors layer_", layer).c_str());
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    }

    return arrow::Status::OK();
}

// Kimi index writer (format "logestix-kimi").
void WriteKimiModelIndex(
    const std::string &model_name,
    const KimiTopology &topo,
    int64_t total_pinned,
    int64_t total_experts,
    const fs::path &output_root)
{
    namespace fs = std::filesystem;
    std::ofstream f(output_root / "model.index.json");
    if (!f) return;

    f << "{\n";
    f << "  \"format\": \"logestix-kimi\",\n";
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
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << "\n";
    f << "}\n";
    f.close();
}

} // namespace

arrow::Status ConvertKimiModel(
    const std::string &input_path,
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;

    if (input_path.empty())
        return arrow::Status::Invalid("input path not set");
    if (output_root.empty())
        return arrow::Status::Invalid("output path not set");

    // Mandatory layout: converted artifacts under <root>/logestix, format tag
    // "logestix-kimi" inside model.index.json.
    fs::path log_root = output_root / "logestix";
    fs::create_directories(log_root);

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_path);

    // config.json is mandatory; text_config values cross-check the locked
    // compile-time geometry.
    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
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
        simdjson::ondemand::object root;
        if (cd.get_object().get(root) != simdjson::SUCCESS)
            return arrow::Status::Invalid("config.json root not an object");
        simdjson::ondemand::object tc;
        if (root["text_config"].get_object().get(tc) != simdjson::SUCCESS)
            return arrow::Status::Invalid(
                "config.json missing text_config (multimodal wrapper)");
        auto get_i = [&](simdjson::ondemand::object &o, const char *k) -> int {
            int64_t v;
            if (o[k].get_int64().get(v) != simdjson::SUCCESS) return -1;
            return static_cast<int>(v);
        };
        struct GeoCheck { const char *name; int cfg; int locked; };
        // clang-format off
        GeoCheck checks[] = {
            {"hidden_size", get_i(tc, "hidden_size"), kKimiHidden},
            {"num_hidden_layers", get_i(tc, "num_hidden_layers"), kKimiNumLayers},
            {"num_attention_heads", get_i(tc, "num_attention_heads"), kKimiNumHeads},
            {"kv_lora_rank", get_i(tc, "kv_lora_rank"), kKimiKvLoraRank},
            {"q_lora_rank", get_i(tc, "q_lora_rank"), kKimiQLoraRank},
            {"qk_nope_head_dim", get_i(tc, "qk_nope_head_dim"), kKimiQkNope},
            {"qk_rope_head_dim", get_i(tc, "qk_rope_head_dim"), kKimiQkRope},
            {"v_head_dim", get_i(tc, "v_head_dim"), kKimiHeadV},
            {"moe_intermediate_size", get_i(tc, "moe_intermediate_size"), kKimiMoeIntermediate},
            {"intermediate_size", get_i(tc, "intermediate_size"), kKimiDenseIntermediate},
            {"n_routed_experts", get_i(tc, "n_routed_experts"), kKimiNumRouted},
            {"num_experts_per_tok", get_i(tc, "num_experts_per_tok"), kKimiTopK},
            {"vocab_size", get_i(tc, "vocab_size"), kKimiVocab},
        };
        // clang-format on
        for (auto &c : checks) {
            if (c.cfg >= 0 && c.cfg != c.locked)
                return arrow::Status::Invalid(
                    "config.json ", c.name, "=", c.cfg,
                    " does not match compile-time lock ", c.locked);
        }
    }

    // Verify every tensor read is complete before any output is written.
    ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_path, model));

    absl::flat_hash_map<KimiExpertKey, KimiExpertParts, KimiExpertKeyHash>
        partial_int4;
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

    for (const auto &shard : model.shards) {
        ++shard_idx;
        lsxcommon::log::info(absl::StrCat("[", shard_idx, "/",
                                          model.shards.size(),
                                          "] Scanning ", shard.filename,
                                          " ..."));

        std::ifstream sf((fs::path(input_path) / shard.filename).string(),
                         std::ios::binary | std::ios::ate);
        if (!sf)
            return arrow::Status::IOError("cannot open ", shard.filename);
        int64_t sz = sf.tellg();
        if (sz <= 0)
            return arrow::Status::IOError("zero/negative size: ",
                                          shard.filename);
        sf.seekg(0);
        std::vector<uint8_t> shard_buf(static_cast<std::size_t>(sz));
        sf.read(reinterpret_cast<char *>(shard_buf.data()), sz);
        if (!sf)
            return arrow::Status::IOError("cannot read ", shard.filename);

        uint64_t hdr_len;
        std::memcpy(&hdr_len, shard_buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            auto parsed = kimi_classify(tensor.name);

            int64_t raw_size = tensor.byte_size();
            int64_t raw_start = data_off + tensor.data_begin;
            if (raw_start < 0 || raw_start + raw_size > sz)
                return arrow::Status::Invalid(shard.filename, " tensor ",
                                              tensor.name,
                                              " offset out of range");
            const uint8_t *raw_ptr = shard_buf.data() + raw_start;
            std::vector<uint8_t> raw(raw_ptr, raw_ptr + raw_size);

            if (parsed.kind == KimiTensorKind::RoutedExpert) {
                KimiExpertKey key{parsed.layer, parsed.expert};
                auto &parts = partial_int4[key];
                max_layer = std::max(max_layer, parsed.layer);
                moe_layers.insert(parsed.layer);
                if (tensor.dtype == "I32") {
                    parts.packed[parsed.weight_id] = std::move(raw);
                    parts.has_packed[parsed.weight_id] = true;
                    if (parts.logical_shape[parsed.weight_id].empty()) {
                        // packed shape is [rows, cols/8]
                        parts.logical_shape[parsed.weight_id] = {
                            tensor.shape[0],
                            tensor.shape.size() >= 2 ? tensor.shape[1] * 8 : 0};
                    }
                } else {   // BF16 scales
                    parts.scale[parsed.weight_id] = std::move(raw);
                    parts.has_scale[parsed.weight_id] = true;
                }
                if (parts.complete()) {
                    ExpertData ed;
                    auto status = [&]() -> arrow::Status {
                        ARROW_RETURN_NOT_OK(BuildExpertData(parts, ed));
                        ARROW_ASSIGN_OR_RAISE(auto packed,
                            PackExpertNVFP4(std::move(ed)));
                        ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                            packed, key.layer, key.expert, log_root));
                        return arrow::Status::OK();
                    }();
                    if (!status.ok())
                        lsxcommon::log::error(absl::StrFormat(
                            "  Failed to pack expert layer=%d expert=%d: %s",
                            key.layer, key.expert, status.ToString()));
                    else {
                        ++experts_processed;
                        ++experts_per_layer[parsed.layer];
                    }
                    partial_int4.erase(key);
                }
            } else if (parsed.kind == KimiTensorKind::SharedExpert) {
                auto &sexp = shared_by_layer[parsed.layer];
                max_layer = std::max(max_layer, parsed.layer);
                AssignSharedWeight(sexp, parsed.weight_id, std::move(raw),
                                   tensor.shape, tensor.dtype);
            } else if (parsed.kind == KimiTensorKind::DenseMlp) {
                dense_layers.insert(parsed.layer);
                max_layer = std::max(max_layer, parsed.layer);
                TensorRecord t;
                t.name = tensor.name; t.dtype = tensor.dtype;
                t.shape = tensor.shape; t.data = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            } else if (parsed.kind != KimiTensorKind::Unknown) {
                // Attention, Router, LayerNorm, Global -> pinned.
                max_layer = std::max(max_layer, parsed.layer);
                TensorRecord t;
                t.name = tensor.name; t.dtype = tensor.dtype;
                t.shape = tensor.shape; t.data = std::move(raw);
                pinned_tensors.push_back(std::move(t));
                ++pinned_processed;
            } else {
                lsxcommon::log::warn(absl::StrCat(
                    "  Unclassified tensor: ", tensor.name, " (skipped)"));
            }
        }
    }

    for (auto &[key, parts] : partial_int4)
        return arrow::Status::Invalid("incomplete INT4 expert layer=",
                                      key.layer, " expert=", key.expert);
    for (auto &[layer, cnt] : experts_per_layer)
        max_experts_per_layer = std::max(max_experts_per_layer, cnt);

    lsxcommon::log::info(
        absl::StrCat("Saving ", pinned_tensors.size(), " pinned tensors..."));
    ARROW_RETURN_NOT_OK(SaveKimiPinnedTensors(pinned_tensors, log_root));

    lsxcommon::log::info(absl::StrCat(
        "Packing ", shared_by_layer.size(),
        " shared-expert layers (NVFP4)..."));
    ARROW_RETURN_NOT_OK(SaveSharedExpertsToArrow(shared_by_layer, log_root));

    KimiTopology topo;
    topo.num_layers = (max_layer >= 0) ? max_layer + 1 : 0;
    topo.experts_per_layer = max_experts_per_layer;
    topo.moe_layers.assign(moe_layers.begin(), moe_layers.end());
    topo.dense_layers.assign(dense_layers.begin(), dense_layers.end());

    WriteKimiModelIndex(
        fs::path(input_path).filename().string(),
        topo,
        static_cast<int64_t>(pinned_tensors.size()),
        static_cast<int64_t>(experts_processed),
        log_root);

    lsxcommon::log::info(absl::StrCat(
        "Done — ", pinned_processed, " pinned tensors, ", experts_processed,
        " experts processed, incomplete ", partial_int4.size()));
    return arrow::Status::OK();
}

} // namespace kimi
} // namespace lsxcommon
