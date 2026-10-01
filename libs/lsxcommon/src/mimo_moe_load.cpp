#include "lsxcommon/mimo_moe_load.h"

#include "mimo_moe_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/synthetic_geometry.h"

#include <simdjson.h>
#include <absl/strings/str_cat.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace lsxcommon {
namespace mimo {

arrow::Status ValidateMiMoGeometry(const MiMoTopology &t)
{
    auto bad = [](const char *field, long long got, long long want) {
        return arrow::Status::Invalid(
            "MiMo frozen geometry mismatch: ", field, " = ", got,
            " (expected ", want, ")");
    };
    if (t.hidden != frozen::kHidden) return bad("hidden_size", t.hidden, frozen::kHidden);
    if (t.layers != frozen::kLayers) return bad("num_layers", t.layers, frozen::kLayers);
    if (t.vocab != frozen::kVocab) return bad("vocab_size", t.vocab, frozen::kVocab);
    if (t.heads != frozen::kHeads) return bad("num_attention_heads", t.heads, frozen::kHeads);
    if (t.kv_heads_ga != frozen::kKvHeadsGa)
        return bad("num_key_value_heads", t.kv_heads_ga, frozen::kKvHeadsGa);
    if (t.kv_heads_swa != frozen::kKvHeadsSwa)
        return bad("num_key_value_heads_swa", t.kv_heads_swa, frozen::kKvHeadsSwa);
    if (t.head_dim != frozen::kQkHeadDim) return bad("head_dim", t.head_dim, frozen::kQkHeadDim);
    if (t.v_head_dim != frozen::kVHeadDim)
        return bad("v_head_dim", t.v_head_dim, frozen::kVHeadDim);
    if (t.rope_dim != frozen::kRopeDim)
        return bad("rope_dim", t.rope_dim, frozen::kRopeDim);
    if (t.num_experts != frozen::kNumRouted)
        return bad("num_experts", t.num_experts, frozen::kNumRouted);
    if (t.top_k != frozen::kTopK) return bad("num_experts_per_tok", t.top_k, frozen::kTopK);
    if (t.moe_intermediate != frozen::kMoeIntermediate)
        return bad("moe_intermediate_size", t.moe_intermediate, frozen::kMoeIntermediate);
    if (t.dense_intermediate != frozen::kDenseIntermediate)
        return bad("dense_intermediate_size", t.dense_intermediate,
                   frozen::kDenseIntermediate);
    return arrow::Status::OK();
}

namespace {

std::string LayerFileName(int layer, const char *suffix)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "layer_%03d%s", layer, suffix);
    return buf;
}

} // namespace

arrow::Result<MiMoModelFiles> LoadMiMoLayout(const std::string &root)
{
    namespace fs = std::filesystem;
    const fs::path logestix = fs::path(root) / "logestix";
    const fs::path index = logestix / "model.index.json";
    std::ifstream f(index);
    if (!f) return arrow::Status::IOError("cannot open ", index.string());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS)
        return arrow::Status::Invalid("cannot parse ", index.string());
    simdjson::ondemand::object obj;
    if (doc.get_object().get(obj) != simdjson::SUCCESS)
        return arrow::Status::Invalid("index root is not an object");

    std::string_view format;
    if (obj["format"].get_string().get(format) != simdjson::SUCCESS)
        return arrow::Status::Invalid("index has no format tag");
    if (format != "logestix-mimo-nvfp4")
        return arrow::Status::Invalid("unsupported format tag: ", format);

    MiMoTopology t;
    auto geti = [&](const char *k, int &out) {
        int64_t v;
        if (obj[k].get_int64().get(v) == simdjson::SUCCESS)
            out = static_cast<int>(v);
    };
    geti("hidden_size", t.hidden);
    geti("num_layers", t.layers);
    geti("vocab_size", t.vocab);
    geti("num_attention_heads", t.heads);
    geti("num_key_value_heads", t.kv_heads_ga);
    geti("num_key_value_heads_swa", t.kv_heads_swa);
    geti("head_dim", t.head_dim);
    geti("v_head_dim", t.v_head_dim);
    geti("rope_dim", t.rope_dim);
    geti("num_experts", t.num_experts);
    geti("num_experts_per_tok", t.top_k);
    geti("moe_intermediate_size", t.moe_intermediate);
    geti("dense_intermediate_size", t.dense_intermediate);
    geti("sliding_window", t.sliding_window);

    if (auto arr = obj["full_attention_layers"].get_array();
        arr.error() == simdjson::SUCCESS)
        for (auto e : arr.value()) {
            int64_t v;
            if (e.get_int64().get(v) == simdjson::SUCCESS)
                t.full_attn_layers.push_back(static_cast<int>(v));
        }
    t.moe_layers.clear();
    for (int i = 1; i < t.layers; ++i) t.moe_layers.push_back(i);

    if (!SyntheticGeometryAllowed()) {
        ARROW_RETURN_NOT_OK(ValidateMiMoGeometry(t));
    }

    MiMoModelFiles files;
    files.root = logestix.string();
    files.num_layers = t.layers;
    files.topo = t;
    files.experts_root = (logestix / "experts").string();

    const fs::path global = logestix / "pinned" / "global.arrow";
    if (!fs::exists(global))
        return arrow::Status::IOError("missing global pinned file: ",
                                      global.string());
    files.global_file = global.string();

    for (int l = 0; l < t.layers; ++l) {
        const fs::path raw = logestix / "pinned" / LayerFileName(l, ".arrow");
        const fs::path nv = logestix / "pinned" / LayerFileName(l, "_nvfp4.arrow");
        if (!fs::exists(raw))
            return arrow::Status::IOError("missing layer pinned file: ",
                                          raw.string());
        files.layer_raw.push_back(raw.string());
        files.layer_nvfp4.push_back(fs::exists(nv) ? nv.string() : "");
    }
    return files;
}

} // namespace mimo
} // namespace lsxcommon
