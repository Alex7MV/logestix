#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {

struct ExpertData {
    std::vector<uint8_t> w1_raw;
    std::vector<uint8_t> w2_raw;
    std::vector<uint8_t> w3_raw;
    std::vector<int64_t> w1_shape;
    std::vector<int64_t> w2_shape;
    std::vector<int64_t> w3_shape;
    std::string w1_dtype;
    std::string w2_dtype;
    std::string w3_dtype;
};

struct PackedExpert {
    std::vector<uint8_t> w1_packed;
    std::vector<uint8_t> w2_packed;
    std::vector<uint8_t> w3_packed;
    std::vector<float> w1_scales;
    std::vector<float> w2_scales;
    std::vector<float> w3_scales;
    std::vector<int64_t> w1_shape;
    std::vector<int64_t> w2_shape;
    std::vector<int64_t> w3_shape;
};

class ExpertSchema {
public:
    static constexpr int kNumFields = 6;

    static constexpr int kIdxW1Weight = 0;
    static constexpr int kIdxW2Weight = 1;
    static constexpr int kIdxW3Weight = 2;
    static constexpr int kIdxW1Scale  = 3;
    static constexpr int kIdxW2Scale  = 4;
    static constexpr int kIdxW3Scale  = 5;

    static std::shared_ptr<arrow::Schema> Make();

    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const PackedExpert &expert);

    static arrow::Result<PackedExpert>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

constexpr int kArrowAlignment = 256;
constexpr int kNVFP4BlockSize = 16;

} // namespace lsxcommon
