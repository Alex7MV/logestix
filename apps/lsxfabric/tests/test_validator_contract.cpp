#include <catch2/catch_test_macros.hpp>
#include "validator.h"
#include "lsxcommon/core.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <arrow/array/concatenate.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lsxfabric;
namespace fs = std::filesystem;

namespace {

struct CoreScope {
    CoreScope() { lsxcommon::Core::Initialize("test_validator_contract.log"); }
    ~CoreScope() {
        lsxcommon::Core::Shutdown();
        std::remove("test_validator_contract.log");
    }
};

std::shared_ptr<arrow::RecordBatch> MakeExpertBatch(float w1_global = 1.0f,
                                                     uint8_t w1_scale = 0x08) {
    lsxcommon::PackedExpert e;
    e.w1_packed = {0x01, 0x23, 0x45, 0x67};
    e.w2_packed = {0x89, 0xAB};
    e.w3_packed = {0xCD, 0xEF};
    e.w1_scales = {w1_scale};
    e.w2_scales = {0x10};
    e.w3_scales = {0x18};
    e.w1_global_scale = w1_global;
    e.w2_global_scale = 1.0f;
    e.w3_global_scale = 1.0f;
    return lsxcommon::ExpertSchema::MakeBatch(e).ValueOrDie();
}

std::shared_ptr<arrow::RecordBatch> MakeDenseBatch(float gs = 1.0f) {
    lsxcommon::DenseNvfp4Record r;
    r.name = "w";
    r.shape = {2, 2};
    r.dtype = "NVFP4";
    r.packed = {0x01, 0x23};
    r.scales = {0x08};
    r.global_scale = gs;
    return lsxcommon::DenseNvfp4Schema::MakeBatch({r}).ValueOrDie();
}

std::shared_ptr<arrow::RecordBatch>
ConcatRows(const std::shared_ptr<arrow::RecordBatch> &a,
           const std::shared_ptr<arrow::RecordBatch> &b) {
    std::vector<std::shared_ptr<arrow::Array>> cols;
    for (int i = 0; i < a->num_columns(); ++i)
        cols.push_back(arrow::Concatenate({a->column(i), b->column(i)}).ValueOrDie());
    return arrow::RecordBatch::Make(a->schema(), a->num_rows() + b->num_rows(), cols);
}

fs::path WriteTemp(const std::string &name,
                   const std::shared_ptr<arrow::RecordBatch> &batch) {
    const fs::path p = fs::temp_directory_path() / name;
    fs::remove(p);
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, p).ok());
    return p;
}

} // namespace

TEST_CASE("validator: ValidateNVFP4Data accepts every byte value",
          "[validator]") {
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i) all[i] = static_cast<uint8_t>(i);
    REQUIRE(ValidateNVFP4Data(all.data(), static_cast<int64_t>(all.size())));
    REQUIRE(ValidateNVFP4Data(nullptr, 0));
}

TEST_CASE("validator: ValidateE4M3Scales accepts E4M3 P2 codes only",
          "[validator]") {
    uint8_t ok[] = {0x08, 0x10, 0x18, 0x20, 0x70};
    REQUIRE(ValidateE4M3Scales(ok, 5));
    REQUIRE(ValidateE4M3Scales(nullptr, 0));
    uint8_t sign = 0x88, zero_exp = 0x00, too_high = 0x78, mantissa = 0x09;
    REQUIRE_FALSE(ValidateE4M3Scales(&sign, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&zero_exp, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&too_high, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&mantissa, 1));
}

TEST_CASE("validator: ValidateGlobalScale requires finite positive",
          "[validator]") {
    REQUIRE(ValidateGlobalScale(1.0f));
    REQUIRE(ValidateGlobalScale(0.001f));
    REQUIRE_FALSE(ValidateGlobalScale(0.0f));
    REQUIRE_FALSE(ValidateGlobalScale(-1.0f));
    REQUIRE_FALSE(ValidateGlobalScale(NAN));
    REQUIRE_FALSE(ValidateGlobalScale(INFINITY));
}

TEST_CASE("validator: LoadModelIndex parses a full index", "[validator]") {
    const fs::path p = fs::temp_directory_path() / "lsxfabric_idx_full.json";
    std::ofstream(p) << R"({
        "format": "logestix-v1-nvfp4",
        "model": "ds",
        "num_layers": 43,
        "experts_per_layer": 256,
        "total_pinned_tensors": 2881,
        "total_experts": 11008
    })";
    const ModelInfo info = LoadModelIndex(p.string());
    REQUIRE(info.format == "logestix-v1-nvfp4");
    REQUIRE(info.model_name == "ds");
    REQUIRE(info.num_layers == 43);
    REQUIRE(info.experts_per_layer == 256);
    REQUIRE(info.total_pinned == 2881);
    REQUIRE(info.total_experts == 11008);
    fs::remove(p);
}

TEST_CASE("validator: LoadModelIndex defaults absent fields", "[validator]") {
    CoreScope scope;
    const fs::path p = fs::temp_directory_path() / "lsxfabric_idx_sparse.json";
    std::ofstream(p) << R"({"format": "logestix-granite"})";
    const ModelInfo info = LoadModelIndex(p.string());
    REQUIRE(info.format == "logestix-granite");
    REQUIRE(info.model_name.empty());
    REQUIRE(info.num_layers == 0);
    REQUIRE(info.experts_per_layer == 0);
    REQUIRE(info.total_pinned == 0);
    REQUIRE(info.total_experts == 0);
    fs::remove(p);
}

TEST_CASE("validator: LoadModelIndex defaults on malformed input",
          "[validator]") {
    CoreScope scope;
    const fs::path bad = fs::temp_directory_path() / "lsxfabric_idx_bad.json";
    std::ofstream(bad) << "{ not json";
    REQUIRE(LoadModelIndex(bad.string()).num_layers == 0);

    const fs::path arr = fs::temp_directory_path() / "lsxfabric_idx_arr.json";
    std::ofstream(arr) << R"([1, 2, 3])";
    REQUIRE(LoadModelIndex(arr.string()).num_layers == 0);

    const fs::path typed = fs::temp_directory_path() / "lsxfabric_idx_typed.json";
    std::ofstream(typed) << R"({"num_layers": "43"})";
    REQUIRE(LoadModelIndex(typed.string()).num_layers == 0);

    REQUIRE(LoadModelIndex("lsxfabric_no_such_index.json").num_layers == 0);
    fs::remove(bad);
    fs::remove(arr);
    fs::remove(typed);
}

namespace lsxfabric {
// Declared in validator.cpp only; redeclared here to test it directly.
ValidationResult ValidateDenseNvfp4Schema(const std::string &arrow_path);
} // namespace lsxfabric

TEST_CASE("validator: ValidateExpert accepts a valid 9-column batch",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_ok.arrow", MakeExpertBatch());
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert rejects a wrong column count",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_6col.arrow", MakeDenseBatch());
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert checks every row", "[validator]") {
    CoreScope scope;
    auto bad = MakeExpertBatch(1.0f, 0x09); // mantissa != 0 -> invalid scale
    auto batch = ConcatRows(MakeExpertBatch(), bad);
    const fs::path p = WriteTemp("lsxfabric_expert_2row.arrow", batch);
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert rejects an invalid global scale",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_gs.arrow", MakeExpertBatch(0.0f));
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert errors on an unreadable file",
          "[validator]") {
    CoreScope scope;
    const ValidationResult r = ValidateExpert("lsxfabric_missing_expert.arrow");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
}

TEST_CASE("validator: ValidateDenseNvfp4Schema accepts 6 columns",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_dense_ok.arrow", MakeDenseBatch());
    const ValidationResult r = ValidateDenseNvfp4Schema(p.string());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    fs::remove(p);
}

TEST_CASE("validator: ValidateDenseNvfp4Schema rejects a wrong column count",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_dense_9col.arrow", MakeExpertBatch());
    const ValidationResult r = ValidateDenseNvfp4Schema(p.string());
    REQUIRE_FALSE(r.ok);
    fs::remove(p);
}

TEST_CASE("validator: ValidatePinnedTensors skips tokenizer and shared experts",
          "[validator]") {
    CoreScope scope;
    const fs::path dir = fs::temp_directory_path() / "lsxfabric_pinned_skip";
    fs::remove_all(dir);
    fs::create_directories(dir);
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "global.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "layer_0.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "tokenizer_vocab.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "shared_experts.arrow").ok());
    // Only global.arrow + layer_0.arrow are counted => 2 rows.
    const ValidationResult r = ValidatePinnedTensors(dir.string(), 2);
    REQUIRE(r.ok);
    REQUIRE(r.warnings == 0);
    fs::remove_all(dir);
}
