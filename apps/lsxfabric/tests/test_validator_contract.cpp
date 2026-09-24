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
