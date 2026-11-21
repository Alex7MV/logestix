// config_json.h — neutral simdjson ondemand field readers shared by the
// family loaders/converters (dedup phase 5). The bodies were byte-identical
// per-family copies (get_d in alice/inkling×2/kimi-k3 load.cpp, the get_i
// lambda in hy/minimax convert.cpp, parse_layer_body in alice/kimi-k3
// convert.cpp). Host-only; never throws.
#pragma once

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace lsxcommon {
namespace config {

// Reads a numeric field as double: a JSON double passes through, an int64 is
// widened; anything else (string/bool/missing) returns false.
inline bool GetDouble(simdjson::ondemand::object &o, const char *k, double &out)
{
    simdjson::ondemand::value x;
    if (o[k].get(x) != simdjson::SUCCESS) return false;
    if (auto d = x.get_double(); d.error() == simdjson::SUCCESS) {
        out = d.value();
        return true;
    }
    if (auto i = x.get_int64(); i.error() == simdjson::SUCCESS) {
        out = static_cast<double>(i.value());
        return true;
    }
    return false;
}

// Reads an int64 field as int, returning @p dflt when absent or non-integer.
inline int GetInt(simdjson::ondemand::object &o, const char *k, int dflt)
{
    int64_t v;
    if (o[k].get_int64().get(v) != simdjson::SUCCESS) return dflt;
    return static_cast<int>(v);
}

// Merges generation_config.json's eos_token_id array into @p eos_ids (the
// gemma dense/moe converter copy; dedup phase 5). Missing file/field is a
// no-op; ids already present are not duplicated.
inline void MergeGenerationConfigEos(const std::string &dir,
                                     std::vector<int> &eos_ids)
{
    std::ifstream gf(std::filesystem::path(dir) / "generation_config.json");
    if (!gf) return;
    std::string gjson((std::istreambuf_iterator<char>(gf)),
                      std::istreambuf_iterator<char>());
    simdjson::ondemand::parser gp;
    simdjson::padded_string gpad(gjson.data(), gjson.size());
    simdjson::ondemand::document gdoc;
    if (gp.iterate(gpad).get(gdoc) != simdjson::SUCCESS) return;
    simdjson::ondemand::object go;
    if (gdoc.get_object().get(go) != simdjson::SUCCESS) return;
    if (auto arr = go["eos_token_id"].get_array();
        arr.error() == simdjson::SUCCESS) {
        for (auto x : arr.value()) {
            int64_t e;
            if (x.get_int64().get(e) != simdjson::SUCCESS) continue;
            const int id = static_cast<int>(e);
            if (std::find(eos_ids.begin(), eos_ids.end(), id) == eos_ids.end())
                eos_ids.push_back(id);
        }
    }
}

// Splits "model.layers.<N>.<body>" into the layer index and the remainder.
// Returns false on any other shape (non-digit index, missing dot/body).
inline bool ParseLayerBody(const std::string &name, int &layer,
                           std::string &body)
{
    const std::string prefix = "model.layers.";
    if (name.rfind(prefix, 0) != 0) return false;
    const size_t digits_start = prefix.size();
    const size_t dot = name.find('.', digits_start);
    if (dot == std::string::npos) return false;
    int v = 0;
    for (size_t i = digits_start; i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
        v = v * 10 + (name[i] - '0');
    }
    layer = v;
    body = name.substr(dot + 1);
    return true;
}

} // namespace config
} // namespace lsxcommon