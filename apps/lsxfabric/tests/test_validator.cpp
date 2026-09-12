#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/core.h"
#include "validator.h"
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
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

TEST_CASE("ValidatePinnedTensors counts rows not columns", "[validator]") {
    namespace fs = std::filesystem;
    lsxcommon::Core::Initialize("test_validator_rows.log");
    auto tmp = fs::temp_directory_path() / "lsx_validator_rows_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    auto sink = arrow::io::FileOutputStream::Open((tmp / "l0.arrow").string())
                    .ValueOrDie();
    auto schema = arrow::schema({arrow::field("a", arrow::int64()),
                                 arrow::field("b", arrow::int64()),
                                 arrow::field("c", arrow::int64())});
    auto builder = arrow::Int64Builder();
    builder.AppendValues({1, 2});
    auto a = builder.Finish().ValueOrDie();
    builder.AppendValues({3, 4});
    auto b = builder.Finish().ValueOrDie();
    builder.AppendValues({5, 6});
    auto c = builder.Finish().ValueOrDie();
    auto batch = arrow::RecordBatch::Make(schema, 2, {a, b, c});
    auto writer = arrow::ipc::MakeFileWriter(sink, schema).ValueOrDie();
    REQUIRE(writer->WriteRecordBatch(*batch).ok());
    REQUIRE(writer->Close().ok());
    REQUIRE(sink->Close().ok());

    auto tok_sink =
        arrow::io::FileOutputStream::Open(
            (tmp / "tokenizer_vocab.arrow").string()).ValueOrDie();
    auto tok_batch = arrow::RecordBatch::Make(schema, 5, {a, b, c});
    auto tok_writer = arrow::ipc::MakeFileWriter(tok_sink, schema).ValueOrDie();
    REQUIRE(tok_writer->WriteRecordBatch(*tok_batch).ok());
    REQUIRE(tok_writer->Close().ok());
    REQUIRE(tok_sink->Close().ok());

    auto res = ValidatePinnedTensors(tmp.string(), 2);

    REQUIRE(res.ok);
    REQUIRE(res.errors == 0);
    REQUIRE(res.warnings == 0);

    fs::remove_all(tmp);
    lsxcommon::Core::Shutdown();
    std::remove("test_validator_rows.log");
}

TEST_CASE("RunValidator resolves expert exp_N naming", "[validator]") {
    namespace fs = std::filesystem;
    lsxcommon::Core::Initialize("test_validator_expert.log");
    auto tmp = fs::temp_directory_path() / "lsx_validator_expert_test";
    fs::remove_all(tmp);
    auto logestix = tmp / "logestix";
    fs::create_directories(logestix / "pinned");
    fs::create_directories(logestix / "experts" / "layer_0");
    std::ofstream(logestix / "model.index.json") << R"({
        "format": "logestix-v1",
        "model": "expert-layout-test",
        "num_layers": 1,
        "experts_per_layer": 1,
        "total_pinned_tensors": 0,
        "total_experts": 1
    })";
    { std::ofstream f(logestix / "experts" / "layer_0" / "exp_0.arrow"); }

    auto res = RunValidator(tmp.string());

    REQUIRE_FALSE(res.ok);
    REQUIRE(res.errors == 1);
    REQUIRE(res.warnings == 0);

    fs::remove_all(tmp);
    lsxcommon::Core::Shutdown();
    std::remove("test_validator_expert.log");
}

TEST_CASE("RunValidator resolves pinned under logestix/ layout", "[validator]") {
    namespace fs = std::filesystem;
    lsxcommon::Core::Initialize("test_validator_layout.log");
    auto tmp = fs::temp_directory_path() /
               "lsx_validator_layout_test";
    fs::remove_all(tmp);
    auto logestix = tmp / "logestix";
    fs::create_directories(logestix / "pinned");
    std::ofstream(logestix / "model.index.json") << R"({
        "format": "logestix-qwen",
        "model": "layout-test",
        "num_layers": 1,
        "experts_per_layer": 0,
        "total_pinned_tensors": 0,
        "total_experts": 0
    })";

    auto res = RunValidator(tmp.string());

    REQUIRE(res.ok);
    REQUIRE(res.errors == 0);
    REQUIRE(res.warnings == 0);

    fs::remove_all(tmp);
    lsxcommon::Core::Shutdown();
    std::remove("test_validator_layout.log");
}
