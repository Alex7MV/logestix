// arrow_validate.h — shared validation for Arrow IPC payloads.
//
// Every owning UnpackBatch path and every zero-copy view loader publishes or
// copies bytes described by Arrow list/binary views. A corrupted, truncated or
// foreign .arrow file must fail with a status, never read past its buffer, and
// never hand a pointer into unmapped memory to the upload path.
//
// Host-only, header-only, std::-only (safe to include from .cu/.cuh TUs; no
// abseil containers). The arithmetic is shared by schema.cpp (owning and view
// paths), expert_store.h (DecodeEntry) and shared_experts.cu.
#pragma once

#include <arrow/array/array_base.h>
#include <arrow/array/array_primitive.h>
#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

namespace lsxcommon {
namespace arrow_check {

// A list column is addressed by an Int32Array of offsets. Reject a slice that
// is reversed or leaves the value array before the caller indexes it.
inline arrow::Status CheckListSlice(const arrow::Int32Array &offsets,
                                    int64_t row,
                                    int64_t values_length,
                                    std::string_view what,
                                    std::string_view name)
{
    if (row < 0 || row + 1 >= offsets.length())
        return arrow::Status::Invalid(what, " for '", name,
                                      "' has no offset pair for row ", row,
                                      " (rows=", offsets.length() - 1, ")");
    const int64_t begin = offsets.Value(row);
    const int64_t end = offsets.Value(row + 1);
    if (begin < 0 || end < begin || end > values_length)
        return arrow::Status::Invalid(what, " for '", name,
                                      "' has an out-of-range list slice [",
                                      begin, ", ", end, ") of ", values_length);
    return arrow::Status::OK();
}

// Zero-copy views publish pointers into Arrow buffers. A corrupt/foreign batch
// (or a broken offset table) can produce a view whose extent crosses the
// backing buffer; reject it here so callers never dereference past a mapping.
inline arrow::Status CheckViewRange(const void *p,
                                    int64_t bytes,
                                    const std::shared_ptr<arrow::Buffer> &buf,
                                    std::string_view what,
                                    std::string_view name)
{
    if (bytes < 0)
        return arrow::Status::Invalid(what, " for '", name,
                                      "' has a negative size");
    if (!buf)
        return bytes == 0 ? arrow::Status::OK()
                          : arrow::Status::Invalid(what, " for '", name,
                                                   "' has no backing buffer");
    const uintptr_t base = reinterpret_cast<uintptr_t>(buf->data());
    const uintptr_t end = base + static_cast<uintptr_t>(buf->size());
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    if (addr < base || addr > end ||
        static_cast<uintptr_t>(bytes) > end - addr)
        return arrow::Status::Invalid(
            what, " for '", name, "' lies outside its buffer (offset=",
            static_cast<int64_t>(addr - base), ", bytes=", bytes,
            ", buffer=", buf->size(), ")");
    return arrow::Status::OK();
}

// Byte width of the raw dtypes the converters emit. NVFP4 (E2M1) is a packed
// format and is validated by CheckNvfp4Size, not here.
inline arrow::Result<int64_t> DtypeByteSize(std::string_view dtype)
{
    if (dtype == "BF16" || dtype == "F16" || dtype == "I16" || dtype == "U16")
        return int64_t{2};
    if (dtype == "F32" || dtype == "I32" || dtype == "U32")
        return int64_t{4};
    if (dtype == "F64" || dtype == "I64" || dtype == "U64")
        return int64_t{8};
    if (dtype == "I8" || dtype == "U8" || dtype == "F8_E4M3" ||
        dtype == "F8_E5M2" || dtype == "F8_E8M0" || dtype == "BOOL")
        return int64_t{1};
    return arrow::Status::TypeError("unknown tensor dtype '", dtype, "'");
}

// product(shape) with overflow detection; a negative or absurd rank is an
// error rather than a silently wrapped element count.
inline arrow::Result<int64_t> ShapeNumel(const int64_t *shape, int64_t rank)
{
    if (!shape && rank != 0)
        return arrow::Status::Invalid("null shape with rank ", rank);
    if (rank < 0 || rank > 64)
        return arrow::Status::Invalid("bad tensor rank ", rank);
    int64_t numel = 1;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t d = shape[i];
        if (d < 0)
            return arrow::Status::Invalid("negative tensor dimension ", d,
                                          " at axis ", i);
        if (d != 0 &&
            numel > std::numeric_limits<int64_t>::max() / d)
            return arrow::Status::Invalid("tensor shape overflows int64");
        numel *= d;
    }
    return numel;
}

// R-A5: a raw tensor's payload must be exactly product(shape) * dtype_size.
inline arrow::Status CheckTensorSize(const int64_t *shape,
                                     int64_t rank,
                                     int64_t bytes,
                                     std::string_view dtype,
                                     std::string_view name)
{
    if (bytes < 0)
        return arrow::Status::Invalid("tensor '", name, "' has negative bytes");
    ARROW_ASSIGN_OR_RAISE(int64_t width, DtypeByteSize(dtype));
    ARROW_ASSIGN_OR_RAISE(int64_t numel, ShapeNumel(shape, rank));
    if (numel != 0 &&
        width > std::numeric_limits<int64_t>::max() / numel)
        return arrow::Status::Invalid("tensor '", name,
                                      "' byte size overflows int64");
    const int64_t expected = numel * width;
    if (bytes != expected)
        return arrow::Status::Invalid(
            "tensor '", name, "' payload is ", bytes, " bytes, expected ",
            expected, " (shape numel=", numel, ", dtype=", dtype, ")");
    return arrow::Status::OK();
}

inline bool EndsWith(std::string_view s, std::string_view suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Gemma stores an NVFP4 dense weight as three Pinned records whose `shape` is
// the LOGICAL tensor shape: "<name>.packed" (E2M1 nibbles), "<name>.scales"
// (one E4M3 per 16 elements) and "<name>.global_scale" (F32, shape {1}).
inline arrow::Status CheckNvfp4PackedSize(const int64_t *shape, int64_t rank,
                                          int64_t bytes, std::string_view name)
{
    ARROW_ASSIGN_OR_RAISE(int64_t numel, ShapeNumel(shape, rank));
    const int64_t want = (numel + 1) / 2;
    if (bytes != want)
        return arrow::Status::Invalid(
            "nvfp4 packed record '", name, "' is ", bytes,
            " bytes, expected ", want, " (numel=", numel, ")");
    return arrow::Status::OK();
}

inline arrow::Status CheckNvfp4ScaleSize(const int64_t *shape, int64_t rank,
                                         int64_t bytes, std::string_view name)
{
    ARROW_ASSIGN_OR_RAISE(int64_t numel, ShapeNumel(shape, rank));
    const int64_t want = (numel + 15) / 16;
    if (bytes != want)
        return arrow::Status::Invalid(
            "nvfp4 scale record '", name, "' is ", bytes,
            " bytes, expected ", want, " (numel=", numel, ")");
    return arrow::Status::OK();
}

// Size check for a PinnedSchema record: the NVFP4 triple is packed/scales
// (logical shape, packed payload), everything else is shape x dtype.
inline arrow::Status CheckPinnedTensor(std::string_view name,
                                       std::string_view dtype,
                                       const int64_t *shape, int64_t rank,
                                       int64_t bytes)
{
    if (EndsWith(name, ".packed"))
        return CheckNvfp4PackedSize(shape, rank, bytes, name);
    if (EndsWith(name, ".scales"))
        return CheckNvfp4ScaleSize(shape, rank, bytes, name);
    return CheckTensorSize(shape, rank, bytes, dtype, name);
}

// R-A5 for NVFP4: E2M1 nibbles (2 per byte) plus one E4M3 scale per 16
// elements and a finite, positive per-tensor global scale.
inline arrow::Status CheckNvfp4Size(const int64_t *shape,
                                    int64_t rank,
                                    int64_t packed_bytes,
                                    int64_t scale_bytes,
                                    float global_scale,
                                    std::string_view name)
{
    ARROW_ASSIGN_OR_RAISE(int64_t numel, ShapeNumel(shape, rank));
    const int64_t want_packed = (numel + 1) / 2;
    const int64_t want_scales = (numel + 15) / 16;
    if (packed_bytes != want_packed)
        return arrow::Status::Invalid(
            "nvfp4 tensor '", name, "' packed payload is ", packed_bytes,
            " bytes, expected ", want_packed, " (numel=", numel, ")");
    if (scale_bytes != want_scales)
        return arrow::Status::Invalid(
            "nvfp4 tensor '", name, "' scale payload is ", scale_bytes,
            " bytes, expected ", want_scales, " (numel=", numel, ")");
    if (!std::isfinite(global_scale) || global_scale <= 0.0f)
        return arrow::Status::Invalid(
            "nvfp4 tensor '", name, "' has an invalid global_scale ",
            global_scale);
    return arrow::Status::OK();
}

} // namespace arrow_check
} // namespace lsxcommon
