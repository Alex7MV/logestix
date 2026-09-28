#include "config.h"

#include <simdjson.h>

#include <fstream>
#include <sstream>

namespace taskmosaic {
namespace {

bool GetString(simdjson::dom::element obj, std::string_view key, std::string& out) {
    auto v = obj[key].get_string();
    if (v.error()) return false;
    out.assign(v.value_unsafe());
    return true;
}

bool GetInt(simdjson::dom::element obj, std::string_view key, int& out) {
    int64_t v = 0;
    if (obj[key].get_int64().get(v)) return false;
    out = static_cast<int>(v);
    return true;
}

}  // namespace

bool ParseBridgeConfig(std::string_view json, BridgeConfig& out, std::string& err) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root)) {
        err = "invalid JSON";
        return false;
    }
    BridgeConfig cfg;

    simdjson::dom::element listen;
    if (!root["listen"].get(listen)) {
        GetString(listen, "host", cfg.host);
        int port = 0;
        if (!GetInt(listen, "port", port) || port <= 0 || port > 65535) {
            err = "listen.port must be 1..65535";
            return false;
        }
        cfg.port = port;
    }
    GetString(root, "default_model", cfg.default_model);
    GetInt(root, "queue_depth", cfg.queue_depth);
    GetInt(root, "request_timeout_ms", cfg.request_timeout_ms);
    GetInt(root, "default_max_tokens", cfg.default_max_tokens);
    if (cfg.queue_depth < 1) cfg.queue_depth = 1;

    simdjson::dom::array models;
    if (root["models"].get(models)) {
        err = "models must be a non-empty array";
        return false;
    }
    for (simdjson::dom::element m : models) {
        ModelConfig mc;
        if (!GetString(m, "id", mc.id) || mc.id.empty()) {
            err = "each model needs a non-empty id";
            return false;
        }
        if (!GetString(m, "model_path", mc.model_path) || mc.model_path.empty()) {
            err = "model '" + mc.id + "' needs model_path";
            return false;
        }
        simdjson::dom::element be;
        if (m["backend"].get(be) != simdjson::SUCCESS ||
            !GetString(be, "endpoint", mc.backend.endpoint) ||
            mc.backend.endpoint.empty()) {
            err = "model '" + mc.id + "' needs backend.endpoint";
            return false;
        }
        GetString(be, "cancel_endpoint", mc.backend.cancel_endpoint);
        cfg.models.push_back(std::move(mc));
    }
    if (cfg.models.empty()) {
        err = "models must be a non-empty array";
        return false;
    }
    out = std::move(cfg);
    return true;
}

bool LoadBridgeConfigFile(const std::string& path, BridgeConfig& out,
                          std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open config file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ParseBridgeConfig(ss.str(), out, err);
}

}  // namespace taskmosaic
