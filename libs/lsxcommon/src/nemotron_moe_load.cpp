#include "lsxcommon/nemotron_moe_load.h"
#include "nemotron_moe_constants.h"

#include "lsxcommon/model_integrity.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/schema.h"

#include <simdjson.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace lsxcommon::nemotron_moe {
namespace {

constexpr const char *kFormatTag = "logestix-nemotron-nvfp4";

bool synthetic_geometry_allowed()
{
    const char *env = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    return env && env[0] == '1';
}

bool get_i(simdjson::ondemand::object &o, const char *k, int &out)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return false;
    out = static_cast<int>(v);
    return true;
}

} // namespace

bool LoadNemotronIndex(const std::string &model_path, NemotronIndex &out,
                       std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path idx_path =
        fs::path(model_path) / "logestix" / "model.index.json";
    std::error_code ec;
    if (!fs::exists(idx_path, ec)) {
        error = "model.index.json not found: " + idx_path.string();
        return false;
    }
    std::ifstream f(idx_path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        error = "model.index.json parse error";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        error = "model.index.json root is not an object";
        return false;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) != simdjson::SUCCESS) {
        error = "index missing 'format' tag";
        return false;
    }
    out.format = std::string(fmt);
    if (out.format != kFormatTag) {
        error = absl::StrCat("unknown format tag: ", out.format,
                             " (want ", kFormatTag, ")");
        return false;
    }
    std::string_view model;
    if (root["model"].get_string().get(model) == simdjson::SUCCESS)
        out.model = std::string(model);

    get_i(root, "num_layers", out.num_layers);
    get_i(root, "hidden_size", out.hidden_size);
    get_i(root, "vocab_size", out.vocab_size);
    get_i(root, "mamba_num_heads", out.mamba_num_heads);
    get_i(root, "mamba_head_dim", out.mamba_head_dim);
    get_i(root, "ssm_state_size", out.ssm_state_size);
    get_i(root, "n_groups", out.n_groups);
    get_i(root, "conv_kernel", out.conv_kernel);
    get_i(root, "num_attention_heads", out.num_attention_heads);
    get_i(root, "num_key_value_heads", out.num_key_value_heads);
    get_i(root, "head_dim", out.head_dim);
    get_i(root, "num_experts", out.num_experts);
    get_i(root, "num_experts_per_tok", out.num_experts_per_tok);
    get_i(root, "moe_intermediate_size", out.moe_intermediate_size);
    get_i(root, "shared_expert_intermediate_size",
          out.shared_expert_intermediate_size);
    get_i(root, "bos_token_id", out.bos_token_id);
    get_i(root, "eos_token_id", out.eos_token_id);
    get_i(root, "pad_token_id", out.pad_token_id);
    get_i(root, "total_pinned_tensors", out.total_pinned_tensors);

    double rd;
    if (root["rope_theta"].get_double().get(rd) == simdjson::SUCCESS)
        out.rope_theta = rd;
    if (root["rms_norm_eps"].get_double().get(rd) == simdjson::SUCCESS)
        out.rms_norm_eps = rd;
    bool b;
    if (root["experts_no_gate"].get_bool().get(b) == simdjson::SUCCESS)
        out.experts_no_gate = b;

    if (auto lt = root["layer_types"].get_array();
        lt.error() == simdjson::SUCCESS) {
        for (auto e : lt.value()) {
            std::string_view sv;
            if (e.get(sv) == simdjson::SUCCESS)
                out.layer_types.emplace_back(sv);
        }
    }
    if (auto ai = root["attn_layer_idx"].get_array();
        ai.error() == simdjson::SUCCESS) {
        for (auto e : ai.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                out.attn_layer_idx.push_back(static_cast<int>(v));
        }
    }
    if (auto def = root["deferred"].get_object();
        def.error() == simdjson::SUCCESS) {
        get_i(def.value(), "mtp_tensors", out.mtp_tensors);
    }
    if (auto vr = root["vram"].get_object(); vr.error() == simdjson::SUCCESS) {
        uint64_t u;
        auto vo = vr.value();
        if (vo["min_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.min_bytes = u;
        if (vo["optimal_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.optimal_bytes = u;
        if (vo["max_bytes"].get_uint64().get(u) == simdjson::SUCCESS)
            out.vram.max_bytes = u;
        int64_t c;
        if (vo["context_tokens"].get_int64().get(c) == simdjson::SUCCESS)
            out.vram.context_tokens = c;
    }

    if (out.num_layers <= 0 || out.hidden_size <= 0 || out.vocab_size <= 0 ||
        static_cast<int>(out.layer_types.size()) != out.num_layers) {
        error = "index missing required NemotronH fields";
        return false;
    }
    return true;
}

bool ValidateNemotronGeometry(const NemotronIndex &idx, std::string &error)
{
    using namespace frozen;
    auto bad = [&error](const char *field, long long got, long long want) {
        error = absl::StrFormat("%s: got %lld, want %lld", field, got, want);
        return false;
    };
    if (synthetic_geometry_allowed()) return true;

    if (idx.num_layers != kLayers)
        return bad("num_layers", idx.num_layers, kLayers);
    if (idx.hidden_size != kHidden)
        return bad("hidden_size", idx.hidden_size, kHidden);
    if (idx.vocab_size != kVocab)
        return bad("vocab_size", idx.vocab_size, kVocab);
    if (idx.mamba_num_heads != kMambaHeads)
        return bad("mamba_num_heads", idx.mamba_num_heads, kMambaHeads);
    if (idx.mamba_head_dim != kMambaHeadDim)
        return bad("mamba_head_dim", idx.mamba_head_dim, kMambaHeadDim);
    if (idx.ssm_state_size != kSsmStateSize)
        return bad("ssm_state_size", idx.ssm_state_size, kSsmStateSize);
    if (idx.n_groups != kSsmGroups)
        return bad("n_groups", idx.n_groups, kSsmGroups);
    if (idx.conv_kernel != kMambaConvKernel)
        return bad("conv_kernel", idx.conv_kernel, kMambaConvKernel);
    if (idx.num_attention_heads != kHeads)
        return bad("num_attention_heads", idx.num_attention_heads, kHeads);
    if (idx.num_key_value_heads != kKVHeads)
        return bad("num_key_value_heads", idx.num_key_value_heads, kKVHeads);
    if (idx.head_dim != kHeadDim)
        return bad("head_dim", idx.head_dim, kHeadDim);
    if (idx.num_experts != kNumRouted)
        return bad("num_experts", idx.num_experts, kNumRouted);
    if (idx.num_experts_per_tok != kTopK)
        return bad("num_experts_per_tok", idx.num_experts_per_tok, kTopK);
    if (idx.moe_intermediate_size != kMoeIntermediate)
        return bad("moe_intermediate_size", idx.moe_intermediate_size,
                   kMoeIntermediate);
    if (idx.shared_expert_intermediate_size != kSharedIntermediate)
        return bad("shared_expert_intermediate_size",
                   idx.shared_expert_intermediate_size, kSharedIntermediate);
    if (!idx.experts_no_gate)
        return bad("experts_no_gate", 0, 1);

    int mamba = 0, moe = 0, attn = 0;
    for (const auto &t : idx.layer_types) {
        if (t == "mamba") ++mamba;
        else if (t == "moe") ++moe;
        else if (t == "attention") ++attn;
        else return bad("layer_types", 0, 0);
    }
    if (mamba != kMambaLayers || moe != kMoeLayers || attn != kAttnLayers)
        return bad("layer mix", mamba * 10000 + moe * 100 + attn,
                   kMambaLayers * 10000 + kMoeLayers * 100 + kAttnLayers);

    if (static_cast<int>(idx.attn_layer_idx.size()) != kAttnLayers)
        return bad("attn_layer_idx size", (long long)idx.attn_layer_idx.size(),
                   kAttnLayers);
    for (int i = 0; i < kAttnLayers; ++i)
        if (idx.attn_layer_idx[i] != kAttnLayerIdx[i])
            return bad("attn_layer_idx", idx.attn_layer_idx[i],
                       kAttnLayerIdx[i]);
    return true;
}

} // namespace lsxcommon::nemotron_moe
