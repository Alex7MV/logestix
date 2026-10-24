#include "deepseek_moe_v41_load.h"

#include <filesystem>
#include <fstream>
#include <iterator>

#include <simdjson.h>

#include "lsxcommon/path_registry.h"

namespace lsxcommon {
namespace deepseek_v41 {

ModelTopology LoadTopology(const std::string &model_path)
{
    ModelTopology topo;
    auto ipath =
        std::filesystem::path(model_path) / "logestix" / "model.index.json";
    std::ifstream f(ipath);
    if (!f) {
        topo.error_msg = "Cannot open " + ipath.string();
        return topo;
    }
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        topo.error_msg = "JSON parse error in " + ipath.string();
        return topo;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        topo.error_msg = "Expected JSON object in " + ipath.string();
        return topo;
    }

    std::string verr;
    if (!lsxcommon::ParseVramBlock(json, topo.vram, verr)) {
        ModelTopology failed;
        failed.error_msg = verr + " in " + ipath.string();
        return failed;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) == simdjson::SUCCESS)
        topo.format.assign(fmt);
    std::string_view mdl;
    if (root["model"].get_string().get(mdl) == simdjson::SUCCESS)
        topo.model.assign(mdl);

    int64_t nl = 0, epl = 0, te = 0;
    if (root["num_layers"].get_int64().get(nl) != simdjson::SUCCESS) nl = 0;
    if (root["experts_per_layer"].get_int64().get(epl) != simdjson::SUCCESS)
        epl = 0;
    if (root["total_experts"].get_int64().get(te) != simdjson::SUCCESS) te = 0;
    topo.num_layers = static_cast<int>(nl);
    topo.experts_per_layer = static_cast<int>(epl);
    topo.total_experts = te;

    // Per-layer CED metadata, when present, is the source of truth for the
    // runtime; otherwise the frozen passport fills it in.
    for (int l = 0; l < kLayers; ++l) topo.layers[l] = LayerMetaAt(l);
    simdjson::ondemand::array arr;
    if (root["layers"].get_array().get(arr) == simdjson::SUCCESS) {
        int l = 0;
        for (auto elem : arr) {
            if (l >= kLayers) break;
            simdjson::ondemand::object lo;
            if (elem.get_object().get(lo) != simdjson::SUCCESS) break;
            int64_t cr = 0;
            bool kv = false, ix = false, cand = false;
            if (lo["compress_ratio"].get_int64().get(cr) != simdjson::SUCCESS)
                cr = 0;
            if (lo["is_kv_source"].get_bool().get(kv) != simdjson::SUCCESS)
                kv = false;
            if (lo["is_index_source"].get_bool().get(ix) != simdjson::SUCCESS)
                ix = false;
            if (lo["is_candidate_source"].get_bool().get(cand) !=
                simdjson::SUCCESS)
                cand = false;
            topo.layers[l] =
                LayerMeta{static_cast<int>(cr), kv, ix, cand};
            ++l;
        }
    }

    if (topo.format != kIndexFormat) {
        ModelTopology failed;
        failed.error_msg = "unsupported index format '" +
                           (topo.format.empty() ? std::string("<missing>")
                                                : topo.format) +
                           "' in " + ipath.string() + " (expected " +
                           std::string(kIndexFormat) + ")";
        return failed;
    }
    return topo;
}

bool ValidateGeometry(const ModelTopology &topo, std::string *error)
{
    auto fail = [error](const std::string &m) {
        if (error) *error = m;
        return false;
    };
    if (topo.num_layers != kLayers)
        return fail("layers " + std::to_string(topo.num_layers) + " != " +
                    std::to_string(kLayers));
    if (topo.experts_per_layer != kNumRouted)
        return fail("experts " + std::to_string(topo.experts_per_layer) +
                    " != " + std::to_string(kNumRouted));
    if (topo.total_experts != static_cast<int64_t>(kLayers) * kNumRouted)
        return fail("total_experts " + std::to_string(topo.total_experts) +
                    " != " +
                    std::to_string(static_cast<int64_t>(kLayers) * kNumRouted));
    return true;
}

} // namespace deepseek_v41
} // namespace lsxcommon
