#include <catch2/catch_test_macros.hpp>

#include "config.h"

#include <filesystem>
#include <fstream>
#include <string>

using namespace taskmosaic;

TEST_CASE("config: parses a full document", "[taskmosaic][config]") {
    const std::string json = R"({
      "listen": { "host": "0.0.0.0", "port": 9090 },
      "default_model": "m1",
      "queue_depth": 3,
      "request_timeout_ms": 12345,
      "default_max_tokens": 128,
      "models": [
        { "id": "m1", "model_path": "/vault/m1",
          "backend": { "endpoint": "tcp://127.0.0.1:5555",
                       "cancel_endpoint": "tcp://127.0.0.1:5556" } }
      ]
    })";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(ParseBridgeConfig(json, cfg, err));
    CHECK(cfg.host == "0.0.0.0");
    CHECK(cfg.port == 9090);
    CHECK(cfg.default_model == "m1");
    CHECK(cfg.queue_depth == 3);
    CHECK(cfg.request_timeout_ms == 12345);
    CHECK(cfg.default_max_tokens == 128);
    REQUIRE(cfg.models.size() == 1);
    CHECK(cfg.models[0].id == "m1");
    CHECK(cfg.models[0].model_path == "/vault/m1");
    CHECK(cfg.models[0].backend.endpoint == "tcp://127.0.0.1:5555");
    CHECK(cfg.models[0].backend.cancel_endpoint == "tcp://127.0.0.1:5556");
}

TEST_CASE("config: defaults when optional fields are absent",
          "[taskmosaic][config]") {
    const std::string json =
        R"({"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(ParseBridgeConfig(json, cfg, err));
    CHECK(cfg.host == "127.0.0.1");
    CHECK(cfg.port == 8080);
    CHECK(cfg.queue_depth == 4);
    CHECK(cfg.default_max_tokens == 64);
    CHECK(cfg.models[0].backend.cancel_endpoint.empty());
}

TEST_CASE("config: rejects missing models and bad port", "[taskmosaic][config]") {
    BridgeConfig cfg;
    std::string err;
    CHECK_FALSE(ParseBridgeConfig(R"({"models":[]})", cfg, err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(ParseBridgeConfig(
        R"({"listen":{"port":70000},"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})",
        cfg, err));
}

TEST_CASE("config: loads from file", "[taskmosaic][config]") {
    const auto path =
        std::filesystem::temp_directory_path() / "lsx_bridge_cfg.json";
    std::ofstream(path)
        << R"({"models":[{"id":"m","model_path":"/m","backend":{"endpoint":"tcp://x:1"}}]})";
    BridgeConfig cfg;
    std::string err;
    REQUIRE(LoadBridgeConfigFile(path.string(), cfg, err));
    CHECK(cfg.models.size() == 1);
    std::filesystem::remove(path);
}
