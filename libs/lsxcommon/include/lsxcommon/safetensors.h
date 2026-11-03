#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {

// One safetensors header entry. The offsets are zero-initialized on purpose:
// a header that omits them (or carries only one) must never yield a descriptor
// with indeterminate bytes — the parser validates before storing, and these
// defaults make the invalid case deterministic when a caller builds one by
// hand.
struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t data_begin = 0;
    int64_t data_end = 0;

    // Untrusted descriptors must go through ValidateTensorInfo first: these
    // two are plain products and can overflow on a hostile header.
    int64_t numel() const;
    int64_t byte_size() const;
};

// ── Tensor descriptor validation ────────────────────────────────────────────

enum class TensorInfoError {
    kOk = 0,
    kEmptyName,
    kUnknownDtype,
    kNegativeDimension,
    kNumelOverflow,
    kByteSizeOverflow,
    kNegativeOffset,
    kReversedOffsets,
    kSizeMismatch,  // data_end - data_begin != numel * dtype width
};

const char *TensorInfoErrorMessage(TensorInfoError err);

// Element width in bytes for a safetensors dtype string; 0 when unknown.
// Covers every dtype the spec defines (the converters previously knew only
// the numeric subset and silently sized the rest as 0).
int64_t TensorDtypeBytes(std::string_view dtype);

// Structural check against the safetensors layout rules (name, dtype width,
// non-negative dimensions, offset ordering, exact byte size). It cannot see
// the shard, so file bounds are checked by ReadTensorDataChecked. The first
// failing check wins in this order: name, dtype, offsets, shape, size — so the
// reported reason is stable when several fields are wrong.
TensorInfoError ValidateTensorInfo(const TensorInfo &info);

// Parses a safetensors header object. Entries that fail ValidateTensorInfo
// are dropped and counted; the returned descriptors are always valid.
struct TensorInfoParseReport {
    std::vector<TensorInfo> tensors;
    int64_t rejected = 0;
};

TensorInfoParseReport ParseTensorInfos(std::string_view header_json);

// ── Shard model ────────────────────────────────────────────────────────────

struct SafetensorsModel {
    struct Shard {
        std::string filename;
        int64_t file_size;
        std::vector<TensorInfo> tensors;
    };
    std::vector<Shard> shards;
};

// ── Validated tensor payload reader ─────────────────────────────────────────
//
// Opens <input_dir>/<shard.filename>, validates the descriptor, re-reads the
// 8-byte header length from the file itself and reads exactly byte_size()
// bytes at 8 + header_length + data_begin. Bounds come from the actual file
// size, not from shard.file_size, so a stale descriptor cannot widen the
// window. This is the only implementation converters may use.

enum class TensorReadError {
    kOk = 0,
    kInvalidDescriptor,  // ValidateTensorInfo failed
    kOpenFailed,
    kShortHeader,     // file shorter than 8 bytes / length read failed
    kHeaderTooLarge,  // header length does not fit the file
    kOutOfBounds,     // tensor range exceeds the payload section
    kShortRead,       // the file shrank between the size check and the read
};

const char *TensorReadErrorMessage(TensorReadError err);

struct TensorReadResult {
    TensorReadError error = TensorReadError::kOk;
    std::vector<uint8_t> bytes;
    std::string detail;  // "file: tensor 'name': reason" on failure

    bool ok() const noexcept { return error == TensorReadError::kOk; }
};

TensorReadResult ReadTensorDataChecked(const std::filesystem::path &input_dir,
                                       const SafetensorsModel::Shard &shard,
                                       const TensorInfo &info);

// Compatibility wrapper with the converters' contract: zeros bytes on any
// failure (callers treat an empty vector as a hard read error) and logs the
// reason from ReadTensorDataChecked.
std::vector<uint8_t> ReadTensorData(const std::filesystem::path &input_dir,
                                    const SafetensorsModel::Shard &shard,
                                    const TensorInfo &info);

// ── Scanning ───────────────────────────────────────────────────────────────

SafetensorsModel::Shard read_safetensors_file(const std::string &path);
SafetensorsModel scan_directory(const std::string &path);

void print_shard(const SafetensorsModel::Shard &shard);
void print_model(const SafetensorsModel &model);

} // namespace lsxcommon
