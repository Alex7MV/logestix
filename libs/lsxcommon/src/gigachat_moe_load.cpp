// GigaChat3.5 loader-stage passport (index parse + frozen-geometry gate).
#include "lsxcommon/gigachat_moe_load.h"

#include "gigachat_moe_constants.h"
#include "lsxcommon/log.h"

#include <simdjson.h>
#include <absl/strings/str_format.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace lsxcommon::gigachat {
namespace {

bool SyntheticGeometryAllowed()
{
    const char *v = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    return v && *v && v[0] != '0';
}

} // namespace

bool LoadGigaChatIndex(const std::string &model_path, GigaChatIndex &out,
                       std::string &error)
{
    namespace fs = std::filesystem;
    const fs::path index_path =
        fs::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(index_path, std::ios::binary);
    if (!f)
    {
        error = "model.index.json not found in " + model_path;
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty())
    {
        error = "index file is empty";
        return false;
    }

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
    {
        error = "cannot parse model.index.json";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS)
    {
        error = "index root is not an object";
        return false;
    }

    auto as_string = [](simdjson::ondemand::value v, std::string &s) -> bool {
        std::string_view sv;
        if (v.get_string().get(sv) != simdjson::SUCCESS) return false;
        s.assign(sv);
        return true;
    };
    auto as_num = [](simdjson::ondemand::value v, double &d) -> bool {
        if (v.get_double().get(d) == simdjson::SUCCESS) return true;
        int64_t i = 0;
        if (v.get_int64().get(i) == simdjson::SUCCESS)
        {
            d = static_cast<double>(i);
            return true;
        }
        return false;
    };

    bool have_format = false;
    for (auto field : root)
    {
        std::string_view key;
        if (field.unescaped_key().get(key) != simdjson::SUCCESS) continue;
        if (key == "format")
        {
            have_format = as_string(field.value(), out.format);
            continue;
        }
        if (key == "model")
        {
            as_string(field.value(), out.model);
            continue;
        }
        if (key == "full_attention_layers")
        {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto e : arr)
                {
                    int64_t v = 0;
                    if (e.get_int64().get(v) == simdjson::SUCCESS)
                        out.full_attention_layers.push_back(static_cast<int>(v));
                }
            continue;
        }
        if (key == "layer_types")
        {
            simdjson::ondemand::array arr;
            if (field.value().get_array().get(arr) == simdjson::SUCCESS)
                for (auto e : arr)
                {
                    std::string_view sv;
                    if (e.get_string().get(sv) == simdjson::SUCCESS)
                        out.layer_types.emplace_back(sv);
                }
            continue;
        }
        if (key == "vram")
        {
            // Rendered as an object; ParseVramBlock is applied to the raw text
            // below (simdjson ondemand object re-reads are order-sensitive).
            continue;
        }
        double d = 0.0;
        if (!as_num(field.value(), d)) continue;
        const int di = static_cast<int>(d);
        if (key == "num_layers") out.num_layers = di;
        else if (key == "hidden_size") out.hidden_size = di;
        else if (key == "vocab_size") out.vocab_size = di;
        else if (key == "num_attention_heads") out.num_attention_heads = di;
        else if (key == "num_key_value_heads") out.num_key_value_heads = di;
        else if (key == "num_experts") out.num_experts = di;
        else if (key == "num_experts_per_tok") out.num_experts_per_tok = di;
        else if (key == "moe_intermediate_size") out.moe_intermediate_size = di;
        else if (key == "first_k_dense_replace") out.first_k_dense_replace = di;
        else if (key == "rms_norm_eps") out.rms_norm_eps = d;
        else if (key == "router_scaling_factor") out.router_scaling_factor = d;
        else if (key == "swiglu_limit") out.swiglu_limit = d;
        else if (key == "bos_token_id") out.bos_token_id = di;
        else if (key == "eos_token_id") out.eos_token_id = di;
        else if (key == "pad_token_id") out.pad_token_id = di;
        else if (key == "total_pinned_tensors") out.total_pinned_tensors = di;
    }

    if (!have_format)
    {
        error = "index missing 'format' tag";
        return false;
    }
    if (out.format != "logestix-gigachat-nvfp4")
    {
        error = "unknown format tag: " + out.format;
        return false;
    }
    ParseVramBlock(json, out.vram, error);
    if (!error.empty())
    {
        // A missing vram block is tolerated by the loader (older converters).
        error.clear();
    }
    if (out.num_layers <= 0 || out.hidden_size <= 0)
    {
        error = "index missing required geometry";
        return false;
    }
    return true;
}

bool ValidateGigaChatGeometry(const GigaChatIndex &idx, std::string &error)
{
    if (SyntheticGeometryAllowed()) return true;
    using namespace lsxcommon::gigachat::frozen;
    auto fail = [&](const char *field, long long expected, long long actual) {
        error = absl::StrFormat("frozen geometry mismatch: %s config=%lld tensor=%lld",
                                field, expected, actual);
        return false;
    };
    if (idx.num_layers != kLayers) return fail("num_layers", kLayers, idx.num_layers);
    if (idx.hidden_size != kHidden) return fail("hidden_size", kHidden, idx.hidden_size);
    if (idx.vocab_size != kVocab) return fail("vocab_size", kVocab, idx.vocab_size);
    if (idx.num_attention_heads != kHeads)
        return fail("num_attention_heads", kHeads, idx.num_attention_heads);
    if (idx.num_key_value_heads != kKVHeads)
        return fail("num_key_value_heads", kKVHeads, idx.num_key_value_heads);
    if (idx.num_experts != kNumRouted) return fail("num_experts", kNumRouted, idx.num_experts);
    if (idx.num_experts_per_tok != kTopK) return fail("num_experts_per_tok", kTopK, idx.num_experts_per_tok);
    if (idx.moe_intermediate_size != kMoeIntermediate)
        return fail("moe_intermediate_size", kMoeIntermediate, idx.moe_intermediate_size);
    if (idx.first_k_dense_replace != kFirstKDense)
        return fail("first_k_dense_replace", kFirstKDense, idx.first_k_dense_replace);
    if (static_cast<int>(idx.full_attention_layers.size()) != kNumFullAttn)
        return fail("full_attention_layers", kNumFullAttn,
                    static_cast<long long>(idx.full_attention_layers.size()));
    for (int i = 0; i < kNumFullAttn; ++i)
        if (idx.full_attention_layers[static_cast<size_t>(i)] != kFullAttnLayers[i])
            return fail("full_attention_layers[i]", kFullAttnLayers[i],
                        idx.full_attention_layers[static_cast<size_t>(i)]);
    error.clear();
    return true;
}

} // namespace lsxcommon::gigachat
