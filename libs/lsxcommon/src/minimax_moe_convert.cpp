#include "lsxcommon/minimax_moe_convert.h"

#include "minimax_moe_constants.h"

#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_integrity.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/safetensors_gpu.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/synthetic_geometry.h"
#include "tensor_index_parse.h"

#include <absl/container/btree_map.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

namespace lsxcommon::minimax {
namespace {

using frozen::kType;

auto get_i = [](simdjson::ondemand::object &o, const char *k, int dflt) {
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
};
auto get_d = [](simdjson::ondemand::object &o, const char *k, double dflt) {
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return dflt;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) return d.value();
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS)
        return static_cast<double>(i.value());
    return dflt;
};

// Cross-checks the parsed config against the frozen passport. Returns false
// and fills `why` on mismatch (synthetic fixtures bypass this upstream).
bool CheckFrozenGeometry(const MinimaxTopology &top, std::string &why)
{
    using namespace frozen;
    auto bad = [&why](const char *field, int got, int want) {
        why = absl::StrFormat("%s: got %d, want %d", field, got, want);
        return false;
    };
    if (top.hidden_size != kHidden) return bad("hidden_size", top.hidden_size, kHidden);
    if (top.num_layers != kLayers) return bad("num_hidden_layers", top.num_layers, kLayers);
    if (top.num_attention_heads != kHeads)
        return bad("num_attention_heads", top.num_attention_heads, kHeads);
    if (top.num_key_value_heads != kKvHeads)
        return bad("num_key_value_heads", top.num_key_value_heads, kKvHeads);
    if (top.head_dim != kHeadDim) return bad("head_dim", top.head_dim, kHeadDim);
    if (top.rotary_dim != kRotaryDim) return bad("rotary_dim", top.rotary_dim, kRotaryDim);
    if (top.vocab_size != kVocab) return bad("vocab_size", top.vocab_size, kVocab);
    if (top.num_experts != kNumExperts)
        return bad("num_local_experts", top.num_experts, kNumExperts);
    if (top.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", top.num_experts_per_tok, kTopK);
    if (top.moe_intermediate_size != kMoeIntermediate)
        return bad("intermediate_size", top.moe_intermediate_size, kMoeIntermediate);
    if (top.shared_intermediate_size != kSharedIntermediate)
        return bad("shared_intermediate_size", top.shared_intermediate_size,
                   kSharedIntermediate);
    if (top.dense_intermediate_size != kDenseIntermediate)
        return bad("dense_intermediate_size", top.dense_intermediate_size,
                   kDenseIntermediate);
    if (top.index_n_heads != kIndexNHeads)
        return bad("sparse_num_index_heads", top.index_n_heads, kIndexNHeads);
    if (top.index_head_dim != kIndexHeadDim)
        return bad("sparse_index_dim", top.index_head_dim, kIndexHeadDim);
    if (top.index_block_size != kIndexBlockSize)
        return bad("sparse_block_size", top.index_block_size, kIndexBlockSize);
    if (top.index_topk_blocks != kIndexTopkBlocks)
        return bad("sparse_topk_blocks", top.index_topk_blocks, kIndexTopkBlocks);
    if (top.index_local_blocks != kIndexLocalBlocks)
        return bad("sparse_local_block", top.index_local_blocks, kIndexLocalBlocks);
    for (int i = 0; i < top.num_layers; ++i) {
        const size_t ui = static_cast<size_t>(i);
        const char *want_attn = IsSparseLayer(i) ? "sparse_attention" : "dense_attention";
        const char *want_mlp = IsMoeLayer(i) ? "sparse" : "dense";
        if (top.layer_types[ui] != want_attn)
            return bad("layer_types[i]", i, 0);
        if (top.mlp_types[ui] != want_mlp)
            return bad("mlp_layer_types[i]", i, 0);
    }
    return true;
}

std::vector<uint8_t> ReadTensorData(const std::filesystem::path &input_path,
                                    const SafetensorsModel::Shard &shard,
                                    const TensorInfo &info)
{
    uint64_t hdr_len = 0;
    std::ifstream sf((input_path / shard.filename).string(), std::ios::binary);
    if (!sf) return {};
    sf.read(reinterpret_cast<char *>(&hdr_len), 8);
    const int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
    const int64_t raw_start = data_off + info.data_begin;
    std::vector<uint8_t> buf(static_cast<size_t>(info.byte_size()));
    sf.seekg(raw_start, std::ios::beg);
    sf.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    if (!sf || static_cast<int64_t>(sf.gcount()) != info.byte_size()) return {};
    return buf;
}

// Quantizes a contiguous BF16/FP32 slice to NVFP4. Returns false on GPU failure.
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &r)
{
    return gpu::quantize_tensor_gpu(data, numel, dtype, nullptr, r.shape, {},
                                    r.packed, r.scales, r.global_scale);
}

arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

arrow::Status WriteExpertFile(const std::filesystem::path &base_dir, int layer,
                              int expert, PackedExpert &&pe)
{
    namespace fs = std::filesystem;
    fs::path dir = base_dir / "experts" / absl::StrCat("layer_", layer);
    std::error_code ec;
    fs::create_directories(dir, ec);
    auto batch = ExpertSchema::MakeBatch(pe);
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch,
                                  dir / absl::StrCat("exp_", expert, ".arrow"));
}

// Reads a light JSON file's "text_config"-less top level for token ids.
void ReadGenerationIds(const std::filesystem::path &path, int &bos, int &eos,
                       int &pad)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty()) return;
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad_json(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad_json).get(doc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return;
    const int b = get_i(root, "bos_token_id", -1);
    const int e = get_i(root, "eos_token_id", -1);
    const int p = get_i(root, "pad_token_id", -1);
    if (b >= 0) bos = b;
    if (e >= 0) eos = e;
    if (p >= 0) pad = p;
}

} // namespace

MinimaxTopology ParseMinimaxConfig(const std::string &input_dir)
{
    MinimaxTopology top;
    namespace fs = std::filesystem;
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

    // text_config preferred; fall back to flat root (some fixtures).
    simdjson::ondemand::object src = root;
    if (auto tc = root["text_config"].get_object();
        tc.error() == simdjson::SUCCESS)
        src = tc.value();

    top.hidden_size = get_i(src, "hidden_size", 0);
    top.num_layers = get_i(src, "num_hidden_layers", 0);
    top.vocab_size = get_i(src, "vocab_size", 0);
    top.num_attention_heads = get_i(src, "num_attention_heads", 0);
    top.num_key_value_heads = get_i(src, "num_key_value_heads", 0);
    top.head_dim = get_i(src, "head_dim", 0);
    top.rotary_dim = get_i(src, "rotary_dim", -1);
    top.dense_intermediate_size = get_i(src, "dense_intermediate_size", 0);
    top.moe_intermediate_size = get_i(src, "intermediate_size", 0);
    top.shared_intermediate_size = get_i(src, "shared_intermediate_size", 0);
    top.num_experts = get_i(src, "num_local_experts", 0);
    top.num_experts_per_tok = get_i(src, "num_experts_per_tok", 0);
    top.rms_norm_eps = get_d(src, "rms_norm_eps", 1e-6);
    top.rope_theta = get_d(src, "rope_theta", 5000000.0);
    top.routed_scaling_factor = get_d(src, "routed_scaling_factor", 2.0);
    top.swiglu_alpha = get_d(src, "swiglu_alpha", 1.702);
    top.swiglu_limit = get_d(src, "swiglu_limit", 7.0);
    top.eos_token_id = get_i(src, "eos_token_id", -1);
    top.bos_token_id = get_i(src, "bos_token_id", -1);
    top.pad_token_id = get_i(src, "pad_token_id", -1);
    // rotary_dim falls back to head_dim * partial_rotary_factor.
    if (top.rotary_dim < 0) {
        const double prf = get_d(src, "partial_rotary_factor", 1.0);
        top.rotary_dim = static_cast<int>(top.head_dim * prf);
    }

    // MiniMax Sparse Attention block (legacy flat names).
    if (auto sac = src["sparse_attention_config"].get_object();
        sac.error() == simdjson::SUCCESS) {
        auto s = sac.value();
        top.index_n_heads = get_i(s, "sparse_num_index_heads", 0);
        top.index_head_dim = get_i(s, "sparse_index_dim", 0);
        top.index_block_size = get_i(s, "sparse_block_size", 0);
        top.index_topk_blocks = get_i(s, "sparse_topk_blocks", 0);
        top.index_local_blocks = get_i(s, "sparse_local_block", 1);
        if (auto freq = s["sparse_attention_freq"].get_array();
            freq.error() == simdjson::SUCCESS) {
            for (auto e : freq.value()) {
                int64_t v = 0;
                if (e.get_int64().get(v) != simdjson::SUCCESS) continue;
                top.layer_types.push_back(v ? "sparse_attention"
                                            : "dense_attention");
            }
        }
    }
    if (auto moe = src["moe_layer_freq"].get_array();
        moe.error() == simdjson::SUCCESS) {
        for (auto e : moe.value()) {
            int64_t v = 0;
            if (e.get_int64().get(v) != simdjson::SUCCESS) continue;
            top.mlp_types.push_back(v ? "sparse" : "dense");
        }
    }

    // Token ids: text_config first, then generation_config.json (this
    // checkpoint puts them there, NOT in text_config).
    if (top.bos_token_id < 0 || top.eos_token_id < 0 || top.pad_token_id < 0)
        ReadGenerationIds(fs::path(input_dir) / "generation_config.json",
                          top.bos_token_id, top.eos_token_id,
                          top.pad_token_id);

    if (top.hidden_size <= 0 || top.num_layers <= 0 ||
        top.num_experts <= 0 || top.num_experts_per_tok <= 0 ||
        top.moe_intermediate_size <= 0 || top.head_dim <= 0 ||
        top.index_n_heads <= 0 || top.index_block_size <= 0) {
        top.error_msg = "config.json missing required MiniMax-M3 fields";
        return top;
    }
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("sparse_attention_freq size ",
                                     top.layer_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    if (static_cast<int>(top.mlp_types.size()) != top.num_layers) {
        top.error_msg = absl::StrCat("moe_layer_freq size ",
                                     top.mlp_types.size(),
                                     " != num_hidden_layers ", top.num_layers);
        return top;
    }
    return top;
}

ParsedMinimaxTensor classify_minimax_tensor(const std::string &name)
{
    ParsedMinimaxTensor r;
    if (name.rfind("vision_tower.", 0) == 0 ||
        name.rfind("multi_modal_projector.", 0) == 0 ||
        name.rfind("patch_merge_mlp.", 0) == 0 ||
        name.rfind("mtp.", 0) == 0) {
        r.kind = MinimaxTensorKind::Visual;
        return r;
    }
    if (name == "language_model.lm_head.weight") {
        r.kind = MinimaxTensorKind::LmHead;
        return r;
    }
    const std::string kPrefix = "language_model.model.";
    if (name.rfind(kPrefix, 0) != 0) return r;
    const std::string rest = name.substr(kPrefix.size());
    if (rest == "embed_tokens.weight") {
        r.kind = MinimaxTensorKind::Embed;
        return r;
    }
    if (rest == "norm.weight") {
        r.kind = MinimaxTensorKind::FinalNorm;
        return r;
    }

    std::string body = rest;
    const std::string kLayers = "layers.";
    if (rest.rfind(kLayers, 0) == 0) {
        const auto digits_start = kLayers.size();
        const auto digits_end = rest.find('.', digits_start);
        if (digits_end == std::string::npos) return r;
        int layer = 0;
        if (!detail::ParseTensorIndex(
                std::string_view(rest).substr(digits_start,
                                              digits_end - digits_start),
                layer))
            return r;
        r.layer = layer;
        body = rest.substr(digits_end + 1);
    } else {
        return r;
    }

    // Routed experts carry an expert index in the path.
    const std::string kExpert = "block_sparse_moe.experts.";
    if (body.rfind(kExpert, 0) == 0) {
        const auto digits_start = kExpert.size();
        const auto digits_end = body.find('.', digits_start);
        if (digits_end == std::string::npos) return r;
        int expert = 0;
        if (!detail::ParseTensorIndex(
                std::string_view(body).substr(digits_start,
                                              digits_end - digits_start),
                expert))
            return r;
        r.expert = expert;
        const std::string leaf = body.substr(digits_end + 1);
        if (leaf == "w1.weight") r.kind = MinimaxTensorKind::ExpertW1;
        else if (leaf == "w2.weight") r.kind = MinimaxTensorKind::ExpertW2;
        else if (leaf == "w3.weight") r.kind = MinimaxTensorKind::ExpertW3;
        return r;
    }

    using K = MinimaxTensorKind;
    static const std::pair<const char *, K> kMap[] = {
        {"input_layernorm.weight", K::InputNorm},
        {"post_attention_layernorm.weight", K::PostNorm},
        {"self_attn.q_proj.weight", K::AttnQ},
        {"self_attn.k_proj.weight", K::AttnK},
        {"self_attn.v_proj.weight", K::AttnV},
        {"self_attn.o_proj.weight", K::AttnO},
        {"self_attn.q_norm.weight", K::AttnQNorm},
        {"self_attn.k_norm.weight", K::AttnKNorm},
        {"self_attn.index_q_proj.weight", K::IndexQ},
        {"self_attn.index_k_proj.weight", K::IndexK},
        {"self_attn.index_q_norm.weight", K::IndexQNorm},
        {"self_attn.index_k_norm.weight", K::IndexKNorm},
        {"mlp.gate_proj.weight", K::DenseMlpGate},
        {"mlp.up_proj.weight", K::DenseMlpUp},
        {"mlp.down_proj.weight", K::DenseMlpDown},
        {"block_sparse_moe.gate.weight", K::RouterGate},
        {"block_sparse_moe.e_score_correction_bias", K::RouterBias},
        {"block_sparse_moe.shared_experts.gate_proj.weight", K::SharedGateProj},
        {"block_sparse_moe.shared_experts.up_proj.weight", K::SharedUpProj},
        {"block_sparse_moe.shared_experts.down_proj.weight", K::SharedDownProj},
    };
    for (const auto &[suffix, kind] : kMap) {
        if (body == suffix) {
            r.kind = kind;
            return r;
        }
    }
    return r;
}

bool minimax_tensor_is_nvfp4(MinimaxTensorKind kind)
{
    switch (kind) {
    case MinimaxTensorKind::Embed:
    case MinimaxTensorKind::LmHead:
    case MinimaxTensorKind::AttnQ:
    case MinimaxTensorKind::AttnK:
    case MinimaxTensorKind::AttnV:
    case MinimaxTensorKind::AttnO:
    case MinimaxTensorKind::IndexQ:
    case MinimaxTensorKind::IndexK:
    case MinimaxTensorKind::DenseMlpGate:
    case MinimaxTensorKind::DenseMlpUp:
    case MinimaxTensorKind::DenseMlpDown:
    case MinimaxTensorKind::ExpertW1:
    case MinimaxTensorKind::ExpertW2:
    case MinimaxTensorKind::ExpertW3:
    case MinimaxTensorKind::SharedGateProj:
    case MinimaxTensorKind::SharedUpProj:
    case MinimaxTensorKind::SharedDownProj:
        return true;
    default:
        return false;
    }
}

bool minimax_tensor_is_routed_expert(MinimaxTensorKind kind)
{
    return kind == MinimaxTensorKind::ExpertW1 ||
           kind == MinimaxTensorKind::ExpertW2 ||
           kind == MinimaxTensorKind::ExpertW3;
}

arrow::Status ConvertMinimaxMoeModel(const std::string &input_dir,
                                     const std::string &output_root)
{
    namespace fs = std::filesystem;

    // (1) Topology: config.json is mandatory and authoritative.
    MinimaxTopology top = ParseMinimaxConfig(input_dir);
    if (!top.error_msg.empty())
        return arrow::Status::Invalid("MiniMax-M3 config error: ", top.error_msg);
    if (top.num_experts_per_tok > top.num_experts)
        return arrow::Status::Invalid("num_experts_per_tok > num_local_experts");
    if (!SyntheticGeometryAllowed()) {
        std::string why;
        if (!CheckFrozenGeometry(top, why))
            return arrow::Status::Invalid("MiniMax-M3 frozen geometry mismatch: ",
                                          why);
    }

    auto model = lsxcommon::scan_directory(input_dir);
    if (model.shards.empty())
        return arrow::Status::Invalid("no safetensors shards found in ",
                                      input_dir);

    // Classify the inventory up front: unknown names are hard errors, vision
    // ballast is counted, routed experts are indexed by name for the
    // expert-major second pass (they are ~98% of the bytes and must never be
    // held in host RAM as a whole).
    struct ExpertLoc {
        const SafetensorsModel::Shard *shard = nullptr;
        const TensorInfo *info = nullptr;
    };
    absl::btree_map<std::string, ExpertLoc> expert_index;
    absl::btree_map<int, int> layer_kind_count;
    int visual_skipped = 0;
    for (const auto &shard : model.shards) {
        for (const auto &t : shard.tensors) {
            auto p = classify_minimax_tensor(t.name);
            if (p.kind == MinimaxTensorKind::Visual) {
                ++visual_skipped;
                continue;
            }
            if (p.kind == MinimaxTensorKind::Unknown)
                return arrow::Status::Invalid(
                    "unrecognized tensor in MiniMax-M3 checkpoint: ", t.name);
            if (minimax_tensor_is_routed_expert(p.kind)) {
                if (p.layer < 0 || p.expert < 0)
                    return arrow::Status::Invalid("expert tensor without index: ",
                                                  t.name);
                if (top.mlp_types[static_cast<size_t>(p.layer)] != "sparse")
                    return arrow::Status::Invalid(
                        "expert tensor on a dense-MLP layer: ", t.name);
                expert_index.emplace(t.name, ExpertLoc{&shard, &t});
            }
            if (p.layer >= 0) ++layer_kind_count[p.layer];
        }
    }
    for (int l = 0; l < top.num_layers; ++l)
        if (!layer_kind_count.contains(l))
            return arrow::Status::Invalid("layer ", l, " has no tensors");

    // (2) Source reads must be complete before any output is written. Skip the
    // routed-expert payloads here: they are re-read (and verified) by the
    // expert pass below, and a full pass would read the 800 GB tree twice.
    {
        SafetensorsModel resident_only;
        for (const auto &shard : model.shards) {
            SafetensorsModel::Shard s;
            s.filename = shard.filename;
            s.file_size = shard.file_size;
            for (const auto &t : shard.tensors) {
                auto p = classify_minimax_tensor(t.name);
                if (!minimax_tensor_is_routed_expert(p.kind))
                    s.tensors.push_back(t);
            }
            if (!s.tensors.empty()) resident_only.shards.push_back(std::move(s));
        }
        ARROW_RETURN_NOT_OK(integrity::VerifyAllTensorReads(input_dir,
                                                            resident_only));
    }

    const fs::path base = fs::path(output_root) / "logestix";
    const fs::path pinned_dir = base / "pinned";
    const fs::path router_dir = base / "router";
    const fs::path shared_dir = base / "shared";
    for (const fs::path &d : {base, pinned_dir, router_dir, shared_dir}) {
        std::error_code ec;
        fs::create_directories(d, ec);
    }

    absl::btree_map<int, std::vector<DenseNvfp4Record>> nvfp4_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> raw_by_layer;
    absl::btree_map<int, std::vector<TensorRecord>> router_by_layer;
    absl::btree_map<int, std::vector<DenseNvfp4Record>> shared_nvfp4_by_layer;
    std::vector<DenseNvfp4Record> global_nvfp4;
    std::vector<TensorRecord> global_raw;

    int64_t total_pinned = 0;

    auto add_nvfp4 = [&](const TensorInfo &t, std::vector<uint8_t> &data,
                         std::vector<DenseNvfp4Record> &dst) -> arrow::Status {
        DenseNvfp4Record r;
        r.name = t.name;
        r.shape = t.shape;
        r.dtype = t.dtype;
        if (!QuantizeSlice(data.data(), t.numel(), t.dtype, r))
            return arrow::Status::IOError("GPU quantization failed for ", t.name);
        dst.push_back(std::move(r));
        ++total_pinned;
        return arrow::Status::OK();
    };

    for (auto &shard : model.shards) {
        for (auto &t : shard.tensors) {
            auto p = classify_minimax_tensor(t.name);
            if (p.kind == MinimaxTensorKind::Visual ||
                minimax_tensor_is_routed_expert(p.kind))
                continue;

            auto data = ReadTensorData(input_dir, shard, t);
            if (data.empty() || static_cast<int64_t>(data.size()) != t.byte_size())
                return arrow::Status::IOError("failed to read tensor data: ",
                                              t.name);

            switch (p.kind) {
            case MinimaxTensorKind::Embed:
            case MinimaxTensorKind::LmHead:
                ARROW_RETURN_NOT_OK(add_nvfp4(t, data, global_nvfp4));
                break;
            case MinimaxTensorKind::FinalNorm: {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                global_raw.push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::InputNorm:
            case MinimaxTensorKind::PostNorm:
            case MinimaxTensorKind::AttnQNorm:
            case MinimaxTensorKind::AttnKNorm:
            case MinimaxTensorKind::IndexQNorm:
            case MinimaxTensorKind::IndexKNorm: {
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                raw_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::RouterGate:
            case MinimaxTensorKind::RouterBias: {
                // High precision on purpose: the routing bias moves the
                // top-k SELECTION, so a 4-bit router swaps the expert set.
                TensorRecord r;
                r.name = t.name;
                r.dtype = t.dtype;
                r.shape = t.shape;
                r.data = std::move(data);
                router_by_layer[p.layer].push_back(std::move(r));
                ++total_pinned;
                break;
            }
            case MinimaxTensorKind::SharedGateProj:
            case MinimaxTensorKind::SharedUpProj:
            case MinimaxTensorKind::SharedDownProj:
                ARROW_RETURN_NOT_OK(
                    add_nvfp4(t, data, shared_nvfp4_by_layer[p.layer]));
                break;
            case MinimaxTensorKind::AttnQ:
            case MinimaxTensorKind::AttnK:
            case MinimaxTensorKind::AttnV:
            case MinimaxTensorKind::AttnO:
            case MinimaxTensorKind::IndexQ:
            case MinimaxTensorKind::IndexK:
            case MinimaxTensorKind::DenseMlpGate:
            case MinimaxTensorKind::DenseMlpUp:
            case MinimaxTensorKind::DenseMlpDown:
                ARROW_RETURN_NOT_OK(add_nvfp4(t, data, nvfp4_by_layer[p.layer]));
                break;
            case MinimaxTensorKind::ExpertW1:
            case MinimaxTensorKind::ExpertW2:
            case MinimaxTensorKind::ExpertW3:
                break;  // handled by the expert pass
            default:
                return arrow::Status::Invalid("unhandled tensor kind: ", t.name);
            }
        }
    }

    // (3) Routed experts: expert-major second pass. Reads w1/w2/w3 (BF16,
    // <= 38 MB each) straight from their shards, quantizes, and writes one
    // ExpertSchema file per (layer, expert). No host-resident expert pool.
    const int H = top.hidden_size;
    const int I = top.moe_intermediate_size;
    for (int l = 0; l < top.num_layers; ++l) {
        if (top.mlp_types[static_cast<size_t>(l)] != "sparse") continue;
        for (int e = 0; e < top.num_experts; ++e) {
            const std::string stem = absl::StrCat(
                "language_model.model.layers.", l,
                ".block_sparse_moe.experts.", e, ".");
            const char *kLeaf[3] = {"w1.weight", "w2.weight", "w3.weight"};
            DenseNvfp4Record recs[3];
            for (int k = 0; k < 3; ++k) {
                const std::string name = stem + kLeaf[k];
                auto it = expert_index.find(name);
                if (it == expert_index.end())
                    return arrow::Status::Invalid("missing expert tensor: ", name);
                auto data = ReadTensorData(input_dir, *it->second.shard,
                                           *it->second.info);
                if (data.empty() ||
                    static_cast<int64_t>(data.size()) != it->second.info->byte_size())
                    return arrow::Status::IOError("failed to read expert tensor: ",
                                                  name);
                const auto &shape = it->second.info->shape;
                if (shape.size() != 2)
                    return arrow::Status::Invalid("expert tensor must be 2-D: ",
                                                  name);
                if (k == 1) {
                    if (shape[0] != H || shape[1] != I)
                        return arrow::Status::Invalid("w2 shape != [H,I]: ", name);
                } else if (shape[0] != I || shape[1] != H) {
                    return arrow::Status::Invalid("w1/w3 shape != [I,H]: ", name);
                }
                recs[k].name = name;
                recs[k].shape = shape;
                recs[k].dtype = it->second.info->dtype;
                if (!QuantizeSlice(data.data(), it->second.info->numel(),
                                   recs[k].dtype, recs[k]))
                    return arrow::Status::IOError("quantize failed: ", name);
            }
            PackedExpert pe;
            pe.w1_packed = std::move(recs[0].packed);
            pe.w1_scales = std::move(recs[0].scales);
            pe.w1_shape = recs[0].shape;
            pe.w1_global_scale = recs[0].global_scale;
            pe.w2_packed = std::move(recs[1].packed);
            pe.w2_scales = std::move(recs[1].scales);
            pe.w2_shape = recs[1].shape;
            pe.w2_global_scale = recs[1].global_scale;
            pe.w3_packed = std::move(recs[2].packed);
            pe.w3_scales = std::move(recs[2].scales);
            pe.w3_shape = recs[2].shape;
            pe.w3_global_scale = recs[2].global_scale;
            ARROW_RETURN_NOT_OK(WriteExpertFile(base, l, e, std::move(pe)));
        }
    }

    // (4) Dense-side artifacts.
    if (!global_raw.empty())
        ARROW_RETURN_NOT_OK(
            WriteBatch(PinnedSchema::MakeBatch(global_raw),
                       pinned_dir / "global.arrow"));
    if (!global_nvfp4.empty())
        ARROW_RETURN_NOT_OK(
            WriteBatch(DenseNvfp4Schema::MakeBatch(global_nvfp4),
                       pinned_dir / "global_nvfp4.arrow"));
    for (auto &[layer, recs] : nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));
    for (auto &[layer, recs] : raw_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            pinned_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : router_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            PinnedSchema::MakeBatch(recs),
            router_dir / absl::StrCat("layer_", layer, ".arrow")));
    for (auto &[layer, recs] : shared_nvfp4_by_layer)
        ARROW_RETURN_NOT_OK(WriteBatch(
            DenseNvfp4Schema::MakeBatch(recs),
            shared_dir / absl::StrCat("layer_", layer, "_nvfp4.arrow")));

    top.total_pinned_tensors = static_cast<int>(total_pinned);
    top.skipped_visual_tensors = visual_skipped;

    // (5) Tokenizer: mandatory for real checkpoints; synthetic fixtures may
    // omit it (recorded in the index). The chat template spells `]~b]` out as
    // text, so the reference tokenizer must NOT auto-prepend BOS.
    const fs::path tok_path = fs::path(input_dir) / "tokenizer.json";
    bool tokenizer_ok = false;
    if (fs::exists(tok_path)) {
        TokenizerSpecialPolicy tok_policy;
        tok_policy.bos_id = -1;
        if (top.eos_token_id >= 0) tok_policy.eos_id = top.eos_token_id;
        if (top.pad_token_id >= 0) tok_policy.pad_id = top.pad_token_id;
        ARROW_RETURN_NOT_OK(integrity::BuildAndVerifyTokenizer(
            tok_path.string(), pinned_dir,
            integrity::TokenizerSpecialIds{.bos_required = false,
                                           .eos_required = true,
                                           .pad_required = true},
            tok_policy));
        tokenizer_ok = true;
    } else if (!SyntheticGeometryAllowed()) {
        return arrow::Status::Invalid("tokenizer.json not found in ", input_dir);
    }

    // (6) Conversion-time VRAM profile.
    VramGeometry vg;
    vg.hidden_dim = top.hidden_size;
    vg.kv_dim = top.head_dim * top.num_key_value_heads;
    vg.num_attn_layers = top.num_layers;
    vg.num_moe_layers = top.num_layers - frozen::kFirstMoeLayer;
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
             return minimax_tensor_is_nvfp4(classify_minimax_tensor(n).kind);
         },
         +[](const std::string &n) -> VramTensorClass {
             return minimax_tensor_is_routed_expert(
                        classify_minimax_tensor(n).kind)
                        ? VramTensorClass::RoutedExpert
                        : VramTensorClass::Resident;
         },
         nullptr},
        vg);

    // (7) Structural manifest (deterministic byte content).
    {
        std::ostringstream j;
        j << "{\n  \"format\": \"" << frozen::kFormatTag << "\",\n"
          << "  \"model\": \"" << frozen::kModelName << "\",\n"
          << "  \"model_type\": \"" << frozen::kType << "\",\n"
          << "  \"num_layers\": " << top.num_layers << ",\n  \"layer_types\": [";
        for (size_t i = 0; i < top.layer_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.layer_types[i] << "\"";
        }
        j << "],\n  \"mlp_layer_types\": [";
        for (size_t i = 0; i < top.mlp_types.size(); ++i) {
            if (i) j << ",";
            j << "\"" << top.mlp_types[i] << "\"";
        }
        j << "],\n  \"hidden_size\": " << top.hidden_size
          << ",\n  \"vocab_size\": " << top.vocab_size
          << ",\n  \"num_attention_heads\": " << top.num_attention_heads
          << ",\n  \"num_key_value_heads\": " << top.num_key_value_heads
          << ",\n  \"head_dim\": " << top.head_dim
          << ",\n  \"rotary_dim\": " << top.rotary_dim
          << ",\n  \"rms_norm_eps\": " << top.rms_norm_eps
          << ",\n  \"rope_theta\": " << top.rope_theta
          << ",\n  \"num_experts\": " << top.num_experts
          << ",\n  \"num_experts_per_tok\": " << top.num_experts_per_tok
          << ",\n  \"moe_intermediate_size\": " << top.moe_intermediate_size
          << ",\n  \"dense_intermediate_size\": " << top.dense_intermediate_size
          << ",\n  \"shared_intermediate_size\": "
          << top.shared_intermediate_size
          << ",\n  \"routed_scaling_factor\": " << top.routed_scaling_factor
          << ",\n  \"swiglu_alpha\": " << top.swiglu_alpha
          << ",\n  \"swiglu_limit\": " << top.swiglu_limit
          << ",\n  \"index_n_heads\": " << top.index_n_heads
          << ",\n  \"index_head_dim\": " << top.index_head_dim
          << ",\n  \"index_block_size\": " << top.index_block_size
          << ",\n  \"index_topk_blocks\": " << top.index_topk_blocks
          << ",\n  \"index_local_blocks\": " << top.index_local_blocks
          << ",\n  \"first_moe_layer\": " << frozen::kFirstMoeLayer
          << ",\n  \"eos_token_id\": " << top.eos_token_id
          << ",\n  \"bos_token_id\": " << top.bos_token_id
          << ",\n  \"pad_token_id\": " << top.pad_token_id
          << ",\n  \"vram\": {\n    \"min_bytes\": " << vram.min_bytes
          << ",\n    \"optimal_bytes\": " << vram.optimal_bytes
          << ",\n    \"max_bytes\": " << vram.max_bytes
          << ",\n    \"context_tokens\": " << vram.context_tokens << "\n  }"
          << ",\n  \"skipped_visual_tensors\": " << visual_skipped
          << ",\n  \"tokenizer\": \"" << (tokenizer_ok ? "binary" : "fixture")
          << "\""
          << ",\n  \"total_pinned_tensors\": " << top.total_pinned_tensors
          << ",\n  \"total_experts\": " << top.num_experts * (top.num_layers -
                                                             frozen::kFirstMoeLayer)
          << "\n}";
        std::ofstream out(base / "model.index.json");
        out << j.str();
    }

    // (8) Final verification: every produced arrow file must re-open cleanly.
    for (const fs::path &dir : {pinned_dir, router_dir, shared_dir}) {
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
             it.increment(ec)) {
            if (!it->is_regular_file()) continue;
            auto res = ReadRecordBatchFromFile(it->path());
            if (!res.ok())
                return arrow::Status::IOError("unreadable output arrow: ",
                                              it->path().string());
        }
        if (ec) return arrow::Status::IOError("cannot list ", dir.string());
    }

    lsxcommon::log::info(absl::StrFormat(
        "MiniMax-M3 conversion: %d layers (%d MoE), %d experts/layer, "
        "%d visual tensors skipped, %d pinned tensors",
        top.num_layers, top.num_layers - frozen::kFirstMoeLayer, top.num_experts,
        visual_skipped, top.total_pinned_tensors));
    return arrow::Status::OK();
}

} // namespace lsxcommon::minimax
