#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace taskmosaic {

struct BackendConfig {
    std::string endpoint;
    std::string cancel_endpoint;
};

struct ModelConfig {
    std::string id;
    std::string model_path;
    BackendConfig backend;
};

struct BridgeConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string default_model;
    int queue_depth = 4;
    int request_timeout_ms = 300000;
    int default_max_tokens = 64;
    std::vector<ModelConfig> models;
};

/// Parses a bridge JSON document. On failure returns false and fills @p err.
bool ParseBridgeConfig(std::string_view json, BridgeConfig& out, std::string& err);

/// Reads and parses @p path. On failure returns false and fills @p err.
bool LoadBridgeConfigFile(const std::string& path, BridgeConfig& out,
                          std::string& err);

}  // namespace taskmosaic
