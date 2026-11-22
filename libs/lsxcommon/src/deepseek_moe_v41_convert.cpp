#include "lsxcommon/deepseek_moe_v41_convert.h"
#include "lsxcommon/synthetic_geometry.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <arrow/api.h>
#include <simdjson.h>

#include "deepseek_moe_v41_constants.h"
#include "deepseek_moe_v41_load.h"
#include "tensor_index_parse.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace deepseek_v41 {

namespace fs = std::filesystem;

namespace {

// ── Name classification ─────────────────────────────────────────────────────

struct ExpertKey {
    int layer = 0;
    int expert = 0;
    bool operator==(const ExpertKey &o) const noexcept
    {
        return layer == o.layer && expert == o.expert;
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey &k) const noexcept
    {
        return (static_cast<std::size_t>(k.layer) << 32) |
               static_cast<std::size_t>(k.expert);
    }
};

struct ParsedExpert {
    int layer = 0;
    int expert = 0;
    int weight_id = 0;
    bool is_scale = false;
};

std::optional<ParsedExpert> ParseExpertName(const std::string &name)
{
    static const std::regex re(
        R"(layers\.(\d+)\.ffn\.experts\.(\d+)\.w(\d)\.(weight|scale))");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return std::nullopt;
    ParsedExpert p;
    // Bounded parse: \d+ can match an arbitrarily long (corrupt) number;
    // std::stoi would throw out_of_range instead of rejecting the name.
    if (!detail::ParseTensorIndex(m[1].str(), p.layer) ||
        !detail::ParseTensorIndex(m[2].str(), p.expert) ||
        !detail::ParseTensorIndex(m[3].str(), p.weight_id))
        return std::nullopt;
    p.is_scale = (m[4] == "scale");
    return p;
}

bool IsExpertTensor(const std::string &name)
{
    return ParseExpertName(name).has_value();
}

// Tensors deliberately excluded from the milestone-1 layout (counted).
std::string CutCategory(const std::string &name)
{
    if (name.rfind("vision.", 0) == 0 || name.rfind("aligner.", 0) == 0 ||
        name.rfind("image_", 0) == 0)
        return "vision";
    if (name.find(".engram.") != std::string::npos) return "engram";
    if (name.rfind("mtp.", 0) == 0) return "mtp";
    return {};
}

std::optional<int> LayerOf(const std::string &name)
{
    static const std::regex re(R"(layers\.(\d+)\.)");
    std::smatch m;
    if (!std::regex_search(name, m, re)) return std::nullopt;
    int layer = 0;
    if (!detail::ParseTensorIndex(m[1].str(), layer)) return std::nullopt;
    return layer;
}

bool IsGlobalTensor(const std::string &name)
{
    return name == "embed.weight" || name == "head.weight" ||
           name == "norm.weight";
}

// ── Config parsing ──────────────────────────────────────────────────────────

struct V41Config {
    int num_layers = 0;
    int hidden = 0;
    int vocab = 0;
    int experts = 0;
    int64_t context = 0;
    std::vector<int> compress_ratios;
    std::vector<int> kv_sources;
    std::vector<int> index_sources;
    int candidate_source = -1;
};

template <typename Obj>
int GetInt(Obj &o, const char *key, int dflt)
{
    int64_t v = 0;
    if (o[key].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

template <typename Obj>
std::vector<int> GetIntArray(Obj &o, const char *key)
{
    std::vector<int> out;
    simdjson::ondemand::array arr;
    if (o[key].get_array().get(arr) != simdjson::SUCCESS) return out;
    for (auto e : arr) {
        int64_t v = 0;
        if (e.get_int64().get(v) == simdjson::SUCCESS)
            out.push_back(static_cast<int>(v));
    }
    return out;
}

arrow::Result<V41Config> ParseConfig(const fs::path &cfg_path)
{
    std::ifstream f(cfg_path);
    if (!f)
        return arrow::Status::IOError("cannot open ", cfg_path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
        return arrow::Status::Invalid("config.json parse failed");
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
        return arrow::Status::Invalid("config.json root not an object");

    V41Config cfg;
    simdjson::ondemand::object text;
    if (root["text_config"].get_object().get(text) == simdjson::SUCCESS) {
        cfg.num_layers = GetInt(text, "num_hidden_layers", 0);
        cfg.hidden = GetInt(text, "hidden_size", 0);
        cfg.vocab = GetInt(text, "vocab_size", 0);
        cfg.experts = GetInt(text, "n_routed_experts", 0);
        cfg.compress_ratios = GetIntArray(text, "compress_ratios");
        cfg.kv_sources = GetIntArray(text, "kv_source_layer_ids");
        cfg.index_sources = GetIntArray(text, "index_source_layer_ids");
        cfg.candidate_source = GetInt(text, "candidate_source_layer_id", -1);
    } else {
        cfg.num_layers = GetInt(root, "num_hidden_layers", 0);
        cfg.hidden = GetInt(root, "hidden_size", 0);
        cfg.vocab = GetInt(root, "vocab_size", 0);
        cfg.experts = GetInt(root, "n_routed_experts", 0);
        cfg.compress_ratios = GetIntArray(root, "compress_ratios");
        cfg.kv_sources = GetIntArray(root, "kv_source_layer_ids");
        cfg.index_sources = GetIntArray(root, "index_source_layer_ids");
        cfg.candidate_source = GetInt(root, "candidate_source_layer_id", -1);
    }
    cfg.context = lsxcommon::ReadContextLength(cfg_path.string());
    return cfg;
}

bool InSet(const std::vector<int> &v, int x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

// ── VRAM estimate policy (mirror of the neutral estimator inputs) ───────────

bool V41IsNvfp4(const std::string &name)
{
    return IsExpertTensor(name) && name.size() >= 7 &&
           name.compare(name.size() - 7, 7, ".weight") == 0;
}

VramTensorClass V41MemClass(const std::string &name)
{
    return IsExpertTensor(name) ? VramTensorClass::RoutedExpert
                                : VramTensorClass::Resident;
}

// ── Index writer ────────────────────────────────────────────────────────────

// The index `model` field is the CARD name of the tree being converted:
// `gpu::detect_model_type` normalizes it (spaces -> '-', lower case) into the
// LsxModelFactory key, and the V4.1 contour serves two checkpoints of the same
// architecture (DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017). A hardcoded
// parent name would make the sibling tree resolve to the parent's kType.
// An input path whose last segment is empty or a dot entry falls back to the
// parent card name, which is the only checkpoint whose directory is not
// guaranteed to carry it.
std::string CardNameFor(const std::string &input_path)
{
    const fs::path p(input_path);
    std::string name = p.filename().string();
    if (name.empty() || name == "." || name == "..")
        return kModelName;
    return name;
}

arrow::Status WriteIndex(const V41Config &cfg, int64_t total_pinned,
                         int64_t total_experts, const VramProfile &vram,
                         const absl::flat_hash_map<std::string, int> &cuts,
                         const std::string &card_name,
                         const fs::path &log_root)
{
    std::ofstream f(log_root / "model.index.json");
    if (!f) return arrow::Status::IOError("cannot create model.index.json");

    f << "{\n";
    f << "  \"format\": \"" << kIndexFormat << "\",\n";
    f << "  \"model\": \"" << card_name << "\",\n";
    f << "  \"num_layers\": " << cfg.num_layers << ",\n";
    f << "  \"experts_per_layer\": " << cfg.experts << ",\n";
    f << "  \"total_pinned_tensors\": " << total_pinned << ",\n";
    f << "  \"total_experts\": " << total_experts << ",\n";
    f << "  " << FormatVramJson(vram) << ",\n";

    f << "  \"layers\": [\n";
    for (int l = 0; l < cfg.num_layers; ++l) {
        int ratio = (l < static_cast<int>(cfg.compress_ratios.size()))
                        ? cfg.compress_ratios[l]
                        : 0;
        f << "    {\"compress_ratio\": " << ratio
          << ", \"is_kv_source\": "
          << (InSet(cfg.kv_sources, l) ? "true" : "false")
          << ", \"is_index_source\": "
          << (InSet(cfg.index_sources, l) ? "true" : "false")
          << ", \"is_candidate_source\": "
          << (l == cfg.candidate_source ? "true" : "false") << "}";
        if (l + 1 < cfg.num_layers) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    f << "  \"deferred\": {";
    bool first = true;
    for (const char *cat : {"engram", "mtp", "vision"}) {
        if (!first) f << ", ";
        first = false;
        auto it = cuts.find(cat);
        f << "\"" << cat << "\": "
          << (it == cuts.end() ? 0 : it->second);
    }
    f << "},\n";

    f << "  \"files\": {\n";
    f << "    \"pinned\": {\n";
    f << "      \"global\": \"pinned/global.arrow\"";
    for (int i = 0; i < cfg.num_layers; ++i)
        f << ",\n      \"layer_" << absl::StrFormat("%03d", i)
          << "\": \"pinned/layer_" << absl::StrFormat("%03d", i) << ".arrow\"";
    f << "\n    },\n";
    f << "    \"experts\": {\n";
    for (int i = 0; i < cfg.num_layers; ++i) {
        f << "      \"layer_" << absl::StrFormat("%03d", i)
          << "\": \"experts/layer_" << absl::StrFormat("%03d", i) << "/\"";
        if (i + 1 < cfg.num_layers) f << ",";
        f << "\n";
    }
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    return arrow::Status::OK();
}

arrow::Status SavePinned(
    const std::vector<TensorRecord> &globals,
    const absl::flat_hash_map<int, std::vector<TensorRecord>> &by_layer,
    const fs::path &log_root)
{
    fs::create_directories(log_root / "pinned");
    if (!globals.empty()) {
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(globals));
        ARROW_RETURN_NOT_OK(
            WriteRecordBatchToFile(batch, log_root / "pinned" / "global.arrow"));
    }
    for (const auto &[layer, tens] : by_layer) {
        auto path = log_root / "pinned" /
                    (absl::StrCat("layer_", absl::StrFormat("%03d", layer)) +
                     ".arrow");
        ARROW_ASSIGN_OR_RAISE(auto batch, PinnedSchema::MakeBatch(tens));
        ARROW_RETURN_NOT_OK(WriteRecordBatchToFile(batch, path));
    }
    return arrow::Status::OK();
}

} // namespace

arrow::Status ConvertDeepSeekV41Model(const std::string &input_path,
                                      const std::string &output_path)
{
    if (input_path.empty() || output_path.empty())
        return arrow::Status::Invalid("input/output path required");

    fs::path log_root = fs::path(output_path) / "logestix";
    fs::create_directories(log_root);

    auto cfg_path = fs::path(input_path) / "config.json";
    if (!fs::exists(cfg_path))
        return arrow::Status::Invalid("config.json not found in ", input_path);
    ARROW_ASSIGN_OR_RAISE(V41Config cfg, ParseConfig(cfg_path));

    if (!SyntheticGeometryAllowed()) {
        if (cfg.num_layers != kLayers || cfg.hidden != kHidden ||
            cfg.vocab != kVocab || cfg.experts != kNumRouted)
            return arrow::Status::Invalid(absl::StrFormat(
                "config geometry (layers=%d hidden=%d vocab=%d experts=%d) "
                "does not match the frozen V4.1 passport",
                cfg.num_layers, cfg.hidden, cfg.vocab, cfg.experts));
    }
    if (cfg.num_layers <= 0)
        return arrow::Status::Invalid("config declares no layers");

    auto model = lsxcommon::scan_directory(input_path);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards in ", input_path);

    std::vector<TensorRecord> globals;
    absl::flat_hash_map<int, std::vector<TensorRecord>> by_layer;
    absl::flat_hash_map<ExpertKey, ExpertData, ExpertKeyHash> partial;
    absl::flat_hash_map<std::string, int> cuts;
    std::vector<VramTensorInfo> inventory;
    int64_t total_pinned = 0;
    int64_t total_experts = 0;

    for (const auto &shard : model.shards) {
        std::ifstream sf(fs::path(input_path) / shard.filename,
                         std::ios::binary | std::ios::ate);
        if (!sf)
            return arrow::Status::IOError("cannot open ", shard.filename);
        int64_t sz = sf.tellg();
        if (sz <= 0)
            return arrow::Status::IOError("bad shard size ", shard.filename);
        sf.seekg(0);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        sf.read(reinterpret_cast<char *>(buf.data()), sz);
        if (!sf) return arrow::Status::IOError("read failed ", shard.filename);

        uint64_t hdr_len = 0;
        std::memcpy(&hdr_len, buf.data(), 8);
        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);

        for (const auto &tensor : shard.tensors) {
            inventory.push_back(VramTensorInfo{
                tensor.name, tensor.numel(), tensor.byte_size()});

            if (const std::string cat = CutCategory(tensor.name); !cat.empty()) {
                ++cuts[cat];
                continue;
            }

            int64_t begin = data_off + tensor.data_begin;
            int64_t bytes = tensor.byte_size();
            if (begin < 0 || begin + bytes > sz)
                return arrow::Status::Invalid("tensor ", tensor.name,
                                              " out of shard bounds");
            std::vector<uint8_t> raw(buf.data() + begin,
                                     buf.data() + begin + bytes);

            if (auto parsed = ParseExpertName(tensor.name)) {
                ExpertData &exp = partial[ExpertKey{parsed->layer, parsed->expert}];
                if (parsed->is_scale) {
                    switch (parsed->weight_id) {
                    case 1: exp.w1_scale_raw = std::move(raw);
                            exp.w1_scale_shape = tensor.shape; break;
                    case 2: exp.w2_scale_raw = std::move(raw);
                            exp.w2_scale_shape = tensor.shape; break;
                    case 3: exp.w3_scale_raw = std::move(raw);
                            exp.w3_scale_shape = tensor.shape; break;
                    }
                } else {
                    switch (parsed->weight_id) {
                    case 1: exp.w1_raw = std::move(raw);
                            exp.w1_shape = tensor.shape;
                            exp.w1_dtype = tensor.dtype; break;
                    case 2: exp.w2_raw = std::move(raw);
                            exp.w2_shape = tensor.shape;
                            exp.w2_dtype = tensor.dtype; break;
                    case 3: exp.w3_raw = std::move(raw);
                            exp.w3_shape = tensor.shape;
                            exp.w3_dtype = tensor.dtype; break;
                    }
                }
                if (!exp.w1_raw.empty() && !exp.w2_raw.empty() &&
                    !exp.w3_raw.empty() && !exp.w1_scale_raw.empty() &&
                    !exp.w2_scale_raw.empty() && !exp.w3_scale_raw.empty()) {
                    ARROW_ASSIGN_OR_RAISE(auto packed,
                                          PackExpertNVFP4(std::move(exp)));
                    ARROW_RETURN_NOT_OK(SaveExpertToArrow(
                        packed, parsed->layer, parsed->expert, log_root));
                    ++total_experts;
                    partial.erase(ExpertKey{parsed->layer, parsed->expert});
                }
                continue;
            }

            TensorRecord rec;
            rec.name = tensor.name;
            rec.dtype = tensor.dtype;
            rec.shape = tensor.shape;
            rec.data = std::move(raw);

            if (IsGlobalTensor(tensor.name)) {
                globals.push_back(std::move(rec));
                ++total_pinned;
                continue;
            }
            if (auto layer = LayerOf(tensor.name)) {
                by_layer[*layer].push_back(std::move(rec));
                ++total_pinned;
                continue;
            }
            return arrow::Status::Invalid(
                "unclassified tensor (not a layer/global/cut): ",
                tensor.name);
        }
    }

    if (!partial.empty())
        return arrow::Status::Invalid(
            "incomplete routed expert records: ",
            static_cast<int64_t>(partial.size()));

    ARROW_RETURN_NOT_OK(SavePinned(globals, by_layer, log_root));

    VramGeometry geom;
    geom.hidden_dim = cfg.hidden;
    geom.kv_dim = kHeadDim;
    geom.num_attn_layers = cfg.num_layers;
    geom.num_moe_layers = cfg.num_layers;
    geom.experts_per_layer = cfg.experts;
    geom.context_length = cfg.context;
    geom.nvfp4_scale_block = 16;
    for (int l = 0; l < cfg.num_layers; ++l)
        geom.kv_compress_ratios.push_back(
            l < static_cast<int>(cfg.compress_ratios.size())
                ? cfg.compress_ratios[l]
                : 0);
    VramTensorPolicy policy;
    policy.is_nvfp4 = &V41IsNvfp4;
    policy.mem_class = &V41MemClass;
    VramProfile vram = EstimateVram(inventory, policy, geom);

    ARROW_RETURN_NOT_OK(WriteIndex(cfg, total_pinned, total_experts, vram,
                                   cuts, CardNameFor(input_path), log_root));

    // Tokenizer bridge (BPE vocab/merges); optional for synthetic fixtures.
    auto tok_json = fs::path(input_path) / "tokenizer.json";
    if (fs::exists(tok_json)) {
        ARROW_RETURN_NOT_OK(
            BuildTokenizerArrowFiles(tok_json.string(), log_root / "pinned"));
    } else {
        lsxcommon::log::warn("[v41-convert] tokenizer.json absent — skipped");
    }

    lsxcommon::log::info(absl::StrFormat(
        "[v41-convert] done: %d pinned tensors, %d experts, deferred "
        "(engram=%d mtp=%d vision=%d)",
        static_cast<int>(total_pinned), static_cast<int>(total_experts),
        cuts["engram"], cuts["mtp"], cuts["vision"]));
    return arrow::Status::OK();
}

} // namespace deepseek_v41
} // namespace lsxcommon
