#pragma once

#include <arrow/api.h>
#include <array>
#include <cstdint>
#include <filesystem>
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
    std::vector<uint8_t> w1_scales;
    std::vector<uint8_t> w2_scales;
    std::vector<uint8_t> w3_scales;
    std::vector<int64_t> w1_shape;
    std::vector<int64_t> w2_shape;
    std::vector<int64_t> w3_shape;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
};

struct TensorRecord {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

class ExpertSchema {
public:
    static constexpr int kNumFields = 9;

    static constexpr int kIdxW1Weight      = 0;
    static constexpr int kIdxW2Weight      = 1;
    static constexpr int kIdxW3Weight      = 2;
    static constexpr int kIdxW1Scale       = 3;
    static constexpr int kIdxW2Scale       = 4;
    static constexpr int kIdxW3Scale       = 5;
    static constexpr int kIdxW1GlobalScale = 6;
    static constexpr int kIdxW2GlobalScale = 7;
    static constexpr int kIdxW3GlobalScale = 8;

    static std::shared_ptr<arrow::Schema> Make();

    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const PackedExpert &expert);

    static arrow::Result<PackedExpert>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

class PinnedSchema {
public:
    static constexpr int kIdxName  = 0;
    static constexpr int kIdxShape = 1;
    static constexpr int kIdxDtype = 2;
    static constexpr int kIdxData  = 3;

    static std::shared_ptr<arrow::Schema> Make();

    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<TensorRecord> &tensors);

    static arrow::Result<std::vector<TensorRecord>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

// Shared expert weights per layer, NVFP4-packed (same 9-column layout as
// ExpertSchema). One record per layer → single pinned/shared_experts.arrow
// file covering every layer.
class SharedExpertSchema {
public:
    static constexpr int kNumFields = 9;

    static constexpr int kIdxW1Weight      = 0;
    static constexpr int kIdxW2Weight      = 1;
    static constexpr int kIdxW3Weight      = 2;
    static constexpr int kIdxW1Scale       = 3;
    static constexpr int kIdxW2Scale       = 4;
    static constexpr int kIdxW3Scale       = 5;
    static constexpr int kIdxW1GlobalScale = 6;
    static constexpr int kIdxW2GlobalScale = 7;
    static constexpr int kIdxW3GlobalScale = 8;

    static std::shared_ptr<arrow::Schema> Make();

    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<PackedExpert> &layers);

    static arrow::Result<std::vector<PackedExpert>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

arrow::Status WriteRecordBatchToFile(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::filesystem::path &path);

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ReadRecordBatchFromFile(const std::filesystem::path &path);

constexpr int kArrowAlignment = 256;
constexpr int kNVFP4BlockSize = 16;

uint8_t float_to_e4m3_p2(float s);
float e4m3_p2_to_float(uint8_t e4m3);

struct VocabRow {
    int32_t token_id = 0;
    std::vector<uint8_t> token_bytes;
    int8_t is_special = 0;
};

struct TokenizerVocabMeta {
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    int32_t vocab_size = 0;
    std::array<int32_t, 256> byte_to_token_id{};
};

class TokenizerVocabSchema {
public:
    static constexpr int kIdxTokenId = 0;
    static constexpr int kIdxTokenBytes = 1;
    static constexpr int kIdxIsSpecial = 2;

    static std::shared_ptr<arrow::Schema> Make(const TokenizerVocabMeta &meta);
    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<VocabRow> &rows, const TokenizerVocabMeta &meta);
    static arrow::Result<std::vector<VocabRow>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

struct MergeRow {
    int32_t merge_left = 0;
    int32_t merge_right = 0;
    int32_t merge_rank = 0;
    int32_t merge_token_id = 0;
};

class TokenizerMergesSchema {
public:
    static constexpr int kIdxMergeLeft = 0;
    static constexpr int kIdxMergeRight = 1;
    static constexpr int kIdxMergeRank = 2;
    static constexpr int kIdxMergeTokenId = 3;

    static std::shared_ptr<arrow::Schema> Make(int32_t merge_count);
    static arrow::Result<std::shared_ptr<arrow::RecordBatch>>
    MakeBatch(const std::vector<MergeRow> &rows, int32_t merge_count);
    static arrow::Result<std::vector<MergeRow>>
    UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch);
};

} // namespace lsxcommon
