#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/core.h"
#include "validator.h"
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace lsxfabric;

TEST_CASE("ValidateNVFP4Data valid", "[validator]") {
    uint8_t data[] = {0x00, 0x01, 0x0F, 0xAB, 0xCD, 0xFF};
    REQUIRE(ValidateNVFP4Data(data, 6));
}

TEST_CASE("ValidateNVFP4Data empty", "[validator]") {
    REQUIRE(ValidateNVFP4Data(nullptr, 0));
}

TEST_CASE("ValidateE4M3Scales valid", "[validator]") {
    uint8_t scales[] = {0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
                        0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70};
    REQUIRE(ValidateE4M3Scales(scales, 14));
}

TEST_CASE("ValidateE4M3Scales negative sign", "[validator]") {
    uint8_t bad = 0x88;
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales zero exponent", "[validator]") {
    uint8_t bad = 0x00;
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales non-zero mantissa", "[validator]") {
    uint8_t bad = 0x0B;
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales empty", "[validator]") {
    REQUIRE(ValidateE4M3Scales(nullptr, 0));
}

TEST_CASE("ValidateGlobalScale valid", "[validator]") {
    REQUIRE(ValidateGlobalScale(1.0f));
    REQUIRE(ValidateGlobalScale(0.001f));
    REQUIRE(ValidateGlobalScale(1000.0f));
}

TEST_CASE("ValidateGlobalScale invalid", "[validator]") {
    REQUIRE_FALSE(ValidateGlobalScale(0.0f));
    REQUIRE_FALSE(ValidateGlobalScale(-1.0f));
    REQUIRE_FALSE(ValidateGlobalScale(NAN));
    REQUIRE_FALSE(ValidateGlobalScale(INFINITY));
}

TEST_CASE("LoadModelIndex valid json", "[validator]") {
    auto tmp = std::filesystem::temp_directory_path() / "test_model_index.json";
    std::ofstream(tmp.string()) << R"({
        "format": "logestix-v1",
        "model": "test-model",
        "num_layers": 3,
        "experts_per_layer": 64,
        "total_pinned_tensors": 42,
        "total_experts": 192
    })";
    auto info = LoadModelIndex(tmp.string());
    REQUIRE(info.format == "logestix-v1");
    REQUIRE(info.model_name == "test-model");
    REQUIRE(info.num_layers == 3);
    REQUIRE(info.experts_per_layer == 64);
    REQUIRE(info.total_pinned == 42);
    REQUIRE(info.total_experts == 192);
    std::filesystem::remove(tmp);
}

TEST_CASE("LoadModelIndex missing file", "[validator]") {
    lsxcommon::Core::Initialize("test_validator_missing.log");
    auto info = LoadModelIndex("nonexistent.json");
    REQUIRE(info.num_layers == 0);
    lsxcommon::Core::Shutdown();
    std::remove("test_validator_missing.log");
}
