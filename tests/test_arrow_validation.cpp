// TDD — shared Arrow IPC validators (arrow_validate.h).
//
// Every owning UnpackBatch path and every zero-copy view loader publishes or
// copies bytes described by Arrow list/binary views. A corrupted, truncated or
// foreign .arrow file must fail with a status, never read past its buffer, and
// never hand a pointer into unmapped memory to the upload path.
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <arrow/array/concatenate.h>
#include <arrow/array/data.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#include "lsxcommon/arrow_validate.h"

namespace av = lsxcommon::arrow_check;

namespace {

std::shared_ptr<arrow::Int32Array> MakeI32(std::vector<int32_t> values)
{
    arrow::Int32Builder b;
    REQUIRE(b.AppendValues(values).ok());
    auto arr = b.Finish().ValueOrDie();
    return std::static_pointer_cast<arrow::Int32Array>(arr);
}

} // namespace

TEST_CASE("CheckListSlice accepts a valid slice and rejects bad ones",
          "[arrow][validate]")
{
    const auto good = MakeI32({0, 4, 8});
    CHECK(av::CheckListSlice(*good, 0, 8, "scale", "t").ok());
    CHECK(av::CheckListSlice(*good, 1, 8, "scale", "t").ok());

    // Reversed slice: end < begin.
    const auto reversed = MakeI32({0, 5, 3});
    CHECK_FALSE(av::CheckListSlice(*reversed, 1, 8, "scale", "t").ok());

    // Negative begin.
    const auto negative = MakeI32({-1, 2});
    CHECK_FALSE(av::CheckListSlice(*negative, 0, 8, "scale", "t").ok());

    // End past the value array.
    const auto past = MakeI32({0, 99});
    CHECK_FALSE(av::CheckListSlice(*past, 0, 8, "scale", "t").ok());
}

TEST_CASE("CheckViewRange rejects a view crossing or preceding its buffer",
          "[arrow][validate]")
{
    auto buf = arrow::Buffer::FromString(std::string(4, 'x'));
    CHECK(av::CheckViewRange(buf->data(), 4, buf, "data", "t").ok());
    CHECK(av::CheckViewRange(buf->data(), 0, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data(), 5, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data() - 1, 1, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data(), -1, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data(), 1, nullptr, "data", "t").ok());
}

TEST_CASE("DtypeByteSize knows the on-disk dtypes and rejects the rest",
          "[arrow][validate]")
{
    CHECK(av::DtypeByteSize("BF16").ValueOrDie() == 2);
    CHECK(av::DtypeByteSize("F16").ValueOrDie() == 2);
    CHECK(av::DtypeByteSize("F32").ValueOrDie() == 4);
    CHECK(av::DtypeByteSize("I8").ValueOrDie() == 1);
    CHECK(av::DtypeByteSize("U8").ValueOrDie() == 1);
    CHECK(av::DtypeByteSize("F8_E4M3").ValueOrDie() == 1);
    CHECK_FALSE(av::DtypeByteSize("XYZ").ok());
    CHECK_FALSE(av::DtypeByteSize("E2M1").ok());
}

TEST_CASE("CheckTensorSize enforces shape x dtype == payload bytes",
          "[arrow][validate]")
{
    const int64_t shape[2] = {8, 4};
    CHECK(av::CheckTensorSize(shape, 2, 8 * 4 * 2, "BF16", "w").ok());
    CHECK(av::CheckTensorSize(shape, 2, 8 * 4 * 4, "F32", "w").ok());
    CHECK_FALSE(av::CheckTensorSize(shape, 2, 8 * 4 * 2 - 1, "BF16", "w").ok());
    CHECK_FALSE(av::CheckTensorSize(shape, 2, 8 * 4 * 2, "XYZ", "w").ok());
    CHECK_FALSE(av::CheckTensorSize(shape, 2, -1, "BF16", "w").ok());
}

TEST_CASE("CheckNvfp4Size enforces packed=ceil(n/2) and scales=ceil(n/16)",
          "[arrow][validate]")
{
    const int64_t shape[2] = {128, 256};
    const int64_t n = 128 * 256;
    CHECK(av::CheckNvfp4Size(shape, 2, n / 2, n / 16, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2 - 1, n / 16, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2, n / 16 - 1, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2, n / 16, 0.0f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2, n / 16,
                                   std::numeric_limits<float>::quiet_NaN(),
                                   "w").ok());
}

TEST_CASE("CheckNvfp4Size rounds odd element counts up", "[arrow][validate]")
{
    const int64_t shape[1] = {17};
    CHECK(av::CheckNvfp4Size(shape, 1, 9, 2, 1.0f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 1, 8, 2, 1.0f, "w").ok());
}

// ── Owning UnpackBatch paths: malformed batches must return a status ───────

#include <memory>
#include <string_view>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/arrow_tree_verify.h"
#include "lsxcommon/schema.h"

namespace {

namespace lc = lsxcommon;

std::shared_ptr<arrow::Buffer> OffsetsBuf(const std::vector<int32_t> &offs)
{
    return arrow::Buffer::FromString(std::string(
        reinterpret_cast<const char *>(offs.data()),
        offs.size() * sizeof(int32_t)));
}

std::shared_ptr<arrow::Buffer> Offsets64Buf(const std::vector<int64_t> &offs)
{
    return arrow::Buffer::FromString(std::string(
        reinterpret_cast<const char *>(offs.data()),
        offs.size() * sizeof(int64_t)));
}

std::shared_ptr<arrow::Array> BinaryWithOffsets(std::vector<int64_t> offs,
                                                std::string values)
{
    const int64_t rows = static_cast<int64_t>(offs.size()) - 1;
    auto vb = arrow::Buffer::FromString(std::move(values));
    return arrow::MakeArray(arrow::ArrayData::Make(
        arrow::large_binary(), rows,
        {nullptr, Offsets64Buf(offs), vb}, /*child_data=*/{}));
}

std::shared_ptr<arrow::Array> ListU8WithOffsets(std::vector<int32_t> offs,
                                                std::string values)
{
    const int64_t rows = static_cast<int64_t>(offs.size()) - 1;
    const int64_t n = static_cast<int64_t>(values.size());
    auto vb = arrow::Buffer::FromString(std::move(values));
    auto vals = arrow::MakeArray(arrow::ArrayData::Make(
        arrow::uint8(), n, {nullptr, vb}, /*child_data=*/{}));
    return arrow::MakeArray(arrow::ArrayData::Make(
        arrow::list(arrow::uint8()), rows, {nullptr, OffsetsBuf(offs)},
        {vals->data()}));
}

std::shared_ptr<arrow::Array> ListI64WithOffsets(std::vector<int32_t> offs,
                                                std::vector<int64_t> values)
{
    const int64_t rows = static_cast<int64_t>(offs.size()) - 1;
    auto vb = arrow::Buffer::FromString(std::string(
        reinterpret_cast<const char *>(values.data()),
        values.size() * sizeof(int64_t)));
    auto vals = arrow::MakeArray(arrow::ArrayData::Make(
        arrow::int64(), static_cast<int64_t>(values.size()), {nullptr, vb},
        /*child_data=*/{}));
    return arrow::MakeArray(arrow::ArrayData::Make(
        arrow::list(arrow::int64()), rows, {nullptr, OffsetsBuf(offs)},
        {vals->data()}));
}

std::shared_ptr<arrow::Array> ListI16WithOffsets(std::vector<int32_t> offs,
                                                std::vector<int16_t> values)
{
    const int64_t rows = static_cast<int64_t>(offs.size()) - 1;
    auto vb = arrow::Buffer::FromString(std::string(
        reinterpret_cast<const char *>(values.data()),
        values.size() * sizeof(int16_t)));
    auto vals = arrow::MakeArray(arrow::ArrayData::Make(
        arrow::int16(), static_cast<int64_t>(values.size()), {nullptr, vb},
        /*child_data=*/{}));
    return arrow::MakeArray(arrow::ArrayData::Make(
        arrow::list(arrow::int16()), rows, {nullptr, OffsetsBuf(offs)},
        {vals->data()}));
}

std::shared_ptr<arrow::RecordBatch> ReplaceColumn(
    const std::shared_ptr<arrow::RecordBatch> &src, int idx,
    std::shared_ptr<arrow::Array> col)
{
    std::vector<std::shared_ptr<arrow::Array>> cols;
    for (int c = 0; c < src->num_columns(); ++c)
        cols.push_back(c == idx ? std::move(col) : src->column(c));
    return arrow::RecordBatch::Make(src->schema(), src->num_rows(),
                                    std::move(cols));
}

lc::PackedExpert SampleExpert()
{
    lc::PackedExpert pe;
    pe.w1_packed.assign(8, 0x11u);
    pe.w2_packed.assign(8, 0x22u);
    pe.w3_packed.assign(8, 0x33u);
    pe.w1_scales.assign(1, 0x81u);
    pe.w2_scales.assign(1, 0x82u);
    pe.w3_scales.assign(1, 0x83u);
    pe.w1_global_scale = 0.5f;
    pe.w2_global_scale = 0.25f;
    pe.w3_global_scale = 0.125f;
    return pe;
}

} // namespace

TEST_CASE("ExpertSchema::UnpackBatch rejects a zero-row batch",
          "[arrow][unpack][expert]")
{
    auto valid = lc::ExpertSchema::MakeBatch(SampleExpert());
    REQUIRE(valid.ok());
    auto empty = (*valid)->Slice(0, 0);
    CHECK_FALSE(lc::ExpertSchema::UnpackBatch(empty).ok());
}

TEST_CASE("ExpertSchema::UnpackBatch rejects a multi-row batch",
          "[arrow][unpack][expert]")
{
    auto valid = lc::ExpertSchema::MakeBatch(SampleExpert());
    REQUIRE(valid.ok());
    std::vector<std::shared_ptr<arrow::Array>> cols;
    for (int c = 0; c < valid->get()->num_columns(); ++c) {
        auto both = arrow::Concatenate({(*valid)->column(c), (*valid)->column(c)});
        REQUIRE(both.ok());
        cols.push_back(*both);
    }
    auto two = arrow::RecordBatch::Make((*valid)->schema(), 2, std::move(cols));
    CHECK_FALSE(lc::ExpertSchema::UnpackBatch(two).ok());
}

TEST_CASE("ExpertSchema::UnpackBatch rejects out-of-range scale offsets",
          "[arrow][unpack][expert]")
{
    auto valid = lc::ExpertSchema::MakeBatch(SampleExpert());
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW1Scale,
                             ListU8WithOffsets({0, 5}, std::string(4, 'x')));
    CHECK_FALSE(lc::ExpertSchema::UnpackBatch(bad).ok());
    auto past_end = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW2Scale,
                                  ListU8WithOffsets({0, 99}, std::string(8, 'x')));
    CHECK_FALSE(lc::ExpertSchema::UnpackBatch(past_end).ok());
}

TEST_CASE("SharedExpertSchema::UnpackBatch rejects a non-uint8 scale list",
          "[arrow][unpack][shared]")
{
    std::vector<lc::PackedExpert> layers(2, SampleExpert());
    auto valid = lc::SharedExpertSchema::MakeBatch(layers);
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW1Scale,
                             ListI16WithOffsets({0, 2, 4}, {1, 2, 3, 4}));
    CHECK_FALSE(lc::SharedExpertSchema::UnpackBatch(bad).ok());
}

TEST_CASE("SharedExpertSchema::UnpackBatch rejects out-of-range scale offsets",
          "[arrow][unpack][shared]")
{
    std::vector<lc::PackedExpert> layers(2, SampleExpert());
    auto valid = lc::SharedExpertSchema::MakeBatch(layers);
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW3Scale,
                             ListU8WithOffsets({0, 3, 99}, std::string(8, 'x')));
    CHECK_FALSE(lc::SharedExpertSchema::UnpackBatch(bad).ok());
}

TEST_CASE("DenseNvfp4Schema::UnpackBatch rejects bad shape/scale offsets",
          "[arrow][unpack][nvfp4]")
{
    lc::DenseNvfp4Record rec;
    rec.name = "w";
    rec.dtype = "E2M1";
    rec.shape = {16, 16};
    rec.packed.assign(128, 0x01u);
    rec.scales.assign(16, 0x81u);
    rec.global_scale = 0.5f;
    auto valid = lc::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(valid.ok());

    auto bad_shape = ReplaceColumn(*valid, lc::DenseNvfp4Schema::kIdxShape,
                                   ListI64WithOffsets({0, 3}, {16, 16, 9}));
    CHECK_FALSE(lc::DenseNvfp4Schema::UnpackBatch(bad_shape).ok());

    auto bad_scales = ReplaceColumn(*valid, lc::DenseNvfp4Schema::kIdxScales,
                                    ListU8WithOffsets({0, 99}, std::string(4, 'x')));
    CHECK_FALSE(lc::DenseNvfp4Schema::UnpackBatch(bad_scales).ok());
}

TEST_CASE("DenseNvfp4Schema::UnpackBatch rejects a truncated packed payload",
          "[arrow][unpack][nvfp4]")
{
    lc::DenseNvfp4Record rec;
    rec.name = "w";
    rec.dtype = "E2M1";
    rec.shape = {16, 16};
    rec.packed.assign(128, 0x01u);
    rec.scales.assign(16, 0x81u);
    rec.global_scale = 0.5f;
    auto valid = lc::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::DenseNvfp4Schema::kIdxPacked,
                             BinaryWithOffsets({0, 64}, std::string(64, 'x')));
    CHECK_FALSE(lc::DenseNvfp4Schema::UnpackBatch(bad).ok());
}

TEST_CASE("DenseNvfp4 payloads are NVFP4 regardless of the source dtype",
          "[arrow][unpack][nvfp4]")
{
    lc::DenseNvfp4Record rec;
    rec.name = "w";
    rec.dtype = "BF16";              // source dtype metadata, not the payload
    rec.shape = {16, 16};
    rec.packed.assign(128, 0x01u);   // 256 elems / 2
    rec.scales.assign(16, 0x81u);    // 256 elems / 16
    rec.global_scale = 0.5f;
    auto batch = lc::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(batch.ok());
    auto unpacked = lc::DenseNvfp4Schema::UnpackBatch(*batch);
    REQUIRE(unpacked.ok());
    CHECK((*unpacked)[0].dtype == "BF16");
}

TEST_CASE("DenseNvfp4Schema::UnpackBatch rejects a non-finite global scale",
          "[arrow][unpack][nvfp4]")
{
    lc::DenseNvfp4Record rec;
    rec.name = "w";
    rec.dtype = "E2M1";
    rec.shape = {16, 16};
    rec.packed.assign(128, 0x01u);
    rec.scales.assign(16, 0x81u);
    rec.global_scale = std::numeric_limits<float>::quiet_NaN();
    auto valid = lc::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(valid.ok());
    CHECK_FALSE(lc::DenseNvfp4Schema::UnpackBatch(*valid).ok());
}

TEST_CASE("PinnedSchema::UnpackBatch rejects bad shape slices and short data",
          "[arrow][unpack][pinned]")
{
    lc::TensorRecord t;
    t.name = "model.norm.weight";
    t.dtype = "F32";
    t.shape = {8};
    t.data.assign(8 * 4, 0x00u);
    auto valid = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(valid.ok());

    auto bad_shape = ReplaceColumn(*valid, lc::PinnedSchema::kIdxShape,
                                   ListI64WithOffsets({0, 99}, {8}));
    CHECK_FALSE(lc::PinnedSchema::UnpackBatch(bad_shape).ok());

    auto short_data = ReplaceColumn(*valid, lc::PinnedSchema::kIdxData,
                                    BinaryWithOffsets({0, 4}, std::string(4, 'x')));
    CHECK_FALSE(lc::PinnedSchema::UnpackBatch(short_data).ok());
}

namespace {

struct ProbeEntry {
    const uint8_t *host_w1 = nullptr, *host_w2 = nullptr, *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr, *host_w2_scales = nullptr,
                  *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f, w2_global_scale = 1.0f,
          w3_global_scale = 1.0f;
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    int64_t w1_scales_bytes = 0, w2_scales_bytes = 0, w3_scales_bytes = 0;
};

using ProbeStore = lc::gpu::expert_store::ExpertStore<ProbeEntry>;

} // namespace

TEST_CASE("ExpertStore::DecodeEntry rejects a multi-row or bad-offset batch",
          "[arrow][unpack][store]")
{
    auto valid = lc::ExpertSchema::MakeBatch(SampleExpert());
    REQUIRE(valid.ok());

    ProbeEntry e;
    CHECK(ProbeStore::DecodeEntry(*valid, e).ok());
    CHECK(e.host_w1 != nullptr);
    CHECK(e.w1_bytes == 8);

    auto empty = (*valid)->Slice(0, 0);
    ProbeEntry e0;
    CHECK_FALSE(ProbeStore::DecodeEntry(empty, e0).ok());

    auto bad = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW1Scale,
                             ListU8WithOffsets({0, 99}, std::string(4, 'x')));
    ProbeEntry e1;
    CHECK_FALSE(ProbeStore::DecodeEntry(bad, e1).ok());
    CHECK(e1.host_w1 == nullptr);
}

// ── View loaders: payload must be consistent with shape/dtype (A-02/A-03) ──

#include <filesystem>

namespace {

namespace fs = std::filesystem;

fs::path TempDir(const std::string &tag)
{
    const fs::path p = fs::temp_directory_path() /
                       ("lsx_arrow_validation_" + tag + "_" +
                        std::to_string(::getpid()));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

} // namespace

TEST_CASE("LoadTensorBatchViews rejects a payload shorter than shape x dtype",
          "[arrow][views][size]")
{
    const auto dir = TempDir("pinned_short");
    lc::TensorRecord t;
    t.name = "model.layers.0.mlp.weight";
    t.dtype = "BF16";
    t.shape = {8, 4};
    t.data.assign(8 * 4 * 2, 0x00u);
    auto valid = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::PinnedSchema::kIdxData,
                             BinaryWithOffsets({0, 7}, std::string(7, 'x')));
    const auto path = dir / "short.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(bad, path).ok());

    const auto views = lc::LoadTensorBatchViews(path);
    CHECK_FALSE(views.ok());
    fs::remove_all(dir);
}

TEST_CASE("LoadTensorBatchViews rejects an out-of-range shape list",
          "[arrow][views][size]")
{
    const auto dir = TempDir("pinned_shape");
    lc::TensorRecord t;
    t.name = "model.norm.weight";
    t.dtype = "F32";
    t.shape = {8};
    t.data.assign(8 * 4, 0x00u);
    auto valid = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::PinnedSchema::kIdxShape,
                             ListI64WithOffsets({0, 99}, {8}));
    const auto path = dir / "shape.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(bad, path).ok());

    const auto views = lc::LoadTensorBatchViews(path);
    CHECK_FALSE(views.ok());
    fs::remove_all(dir);
}

TEST_CASE("LoadDenseNvfp4BatchViews enforces the NVFP4 size invariant",
          "[arrow][views][nvfp4]")
{
    const auto dir = TempDir("nvfp4_size");
    lc::DenseNvfp4Record rec;
    rec.name = "w";
    rec.dtype = "E2M1";
    rec.shape = {16, 16};
    rec.packed.assign(128, 0x01u);
    rec.scales.assign(16, 0x81u);
    rec.global_scale = 0.5f;
    auto valid = lc::DenseNvfp4Schema::MakeBatch({rec});
    REQUIRE(valid.ok());

    const auto good_path = dir / "good.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(*valid, good_path).ok());
    {
        auto views = lc::LoadDenseNvfp4BatchViews(good_path);
        CHECK(views.ok());
    }

    const auto short_packed = dir / "short_packed.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(
                ReplaceColumn(*valid, lc::DenseNvfp4Schema::kIdxPacked,
                              BinaryWithOffsets({0, 64}, std::string(64, 'x'))),
                short_packed).ok());
    CHECK_FALSE(lc::LoadDenseNvfp4BatchViews(short_packed).ok());

    const auto short_scales = dir / "short_scales.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(
                ReplaceColumn(*valid, lc::DenseNvfp4Schema::kIdxScales,
                              ListU8WithOffsets({0, 15}, std::string(15, 'x'))),
                short_scales).ok());
    CHECK_FALSE(lc::LoadDenseNvfp4BatchViews(short_scales).ok());

    fs::remove_all(dir);
}

// ── shared_experts.cu: reversed scale offsets must not become a negative size ─

#include "lsxcommon/shared_experts.h"

TEST_CASE("load_shared_experts_metadata rejects reversed scale offsets",
          "[arrow][shared-experts][offsets]")
{
    const auto dir = TempDir("shared_offsets");
    std::vector<lc::PackedExpert> layers(2, SampleExpert());
    auto valid = lc::SharedExpertSchema::MakeBatch(layers);
    REQUIRE(valid.ok());
    auto bad = ReplaceColumn(*valid, lc::ExpertSchema::kIdxW1Scale,
                             ListU8WithOffsets({0, 3, 2}, std::string(8, 'x')));
    const auto path = dir / "shared_bad.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(bad, path).ok());

    auto mmap = arrow::io::MemoryMappedFile::Open(path.string(),
                                                  arrow::io::FileMode::READ);
    REQUIRE(mmap.ok());
    lc::gpu::SharedExpertDb db;
    const auto st = lc::gpu::load_shared_experts_metadata(*mmap, db);
    CHECK_FALSE(st.ok());
    fs::remove_all(dir);
}

// ── Atomic artifact writes (A-05) ──────────────────────────────────────────

namespace {

void SetEnv(const char *key, const char *value)
{
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value)
        ::setenv(key, value, 1);
    else
        ::unsetenv(key);
#endif
}

std::vector<fs::path> TempLeftovers(const fs::path &dir,
                                    const std::string &stem)
{
    std::vector<fs::path> out;
    for (const auto &e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        if (name.rfind(stem + ".tmp.", 0) == 0)
            out.push_back(e.path());
    }
    return out;
}

} // namespace

TEST_CASE("WriteRecordBatchToFile leaves no destination on an injected failure",
          "[arrow][write][atomic]")
{
    const auto dir = TempDir("write_fail");
    lc::TensorRecord t;
    t.name = "model.norm.weight";
    t.dtype = "F32";
    t.shape = {4};
    t.data.assign(16, 0x00u);
    auto batch = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(batch.ok());
    const auto path = dir / "artifact.arrow";

    SetEnv("LSX_TEST_ARROW_WRITE_FAIL", "1");
    const auto st = lc::WriteRecordBatchToFile(*batch, path);
    SetEnv("LSX_TEST_ARROW_WRITE_FAIL", nullptr);

    CHECK_FALSE(st.ok());
    CHECK_FALSE(fs::exists(path));
    CHECK(TempLeftovers(dir, "artifact.arrow").empty());
    fs::remove_all(dir);
}

TEST_CASE("WriteRecordBatchToFile replaces an existing artifact atomically",
          "[arrow][write][atomic]")
{
    const auto dir = TempDir("write_replace");
    const auto path = dir / "artifact.arrow";
    lc::TensorRecord first;
    first.name = "model.a";
    first.dtype = "F32";
    first.shape = {4};
    first.data.assign(16, 0x11u);
    auto b1 = lc::PinnedSchema::MakeBatch({first});
    REQUIRE(b1.ok());
    REQUIRE(lc::WriteRecordBatchToFile(*b1, path).ok());
    const auto size1 = fs::file_size(path);

    lc::TensorRecord second = first;
    second.name = "model.b";
    second.shape = {16};
    second.data.assign(64, 0x22u);
    auto b2 = lc::PinnedSchema::MakeBatch({second});
    REQUIRE(b2.ok());
    REQUIRE(lc::WriteRecordBatchToFile(*b2, path).ok());

    auto stored = lc::ReadRecordBatchFromFile(path);
    REQUIRE(stored.ok());
    auto reopened = lc::PinnedSchema::UnpackBatch(*stored);
    REQUIRE(reopened.ok());
    REQUIRE(reopened->size() == 1);
    CHECK((*reopened)[0].name == "model.b");
    CHECK((*reopened)[0].data.size() == 64);
    CHECK(fs::file_size(path) != size1);
    CHECK(TempLeftovers(dir, "artifact.arrow").empty());
    fs::remove_all(dir);
}

TEST_CASE("Pinned NVFP4 triple records validate against the logical shape",
          "[arrow][unpack][pinned][nvfp4]")
{
    lc::TensorRecord pk;
    pk.name = "model.layers.0.mlp.down_proj.weight.packed";
    pk.dtype = "U8";
    pk.shape = {16, 16};
    pk.data.assign(128, 0x01u);          // 256 elems / 2
    lc::TensorRecord sc;
    sc.name = "model.layers.0.mlp.down_proj.weight.scales";
    sc.dtype = "U8";
    sc.shape = {16, 16};
    sc.data.assign(16, 0x81u);           // 256 elems / 16
    lc::TensorRecord gs;
    gs.name = "model.layers.0.mlp.down_proj.weight.global_scale";
    gs.dtype = "F32";
    gs.shape = {1};
    gs.data.assign(4, 0x00u);

    auto good = lc::PinnedSchema::MakeBatch({pk, sc, gs});
    REQUIRE(good.ok());
    CHECK(lc::PinnedSchema::UnpackBatch(*good).ok());

    auto short_packed = pk;
    short_packed.data.resize(127);
    auto bad_packed = lc::PinnedSchema::MakeBatch({short_packed, sc, gs});
    REQUIRE(bad_packed.ok());
    CHECK_FALSE(lc::PinnedSchema::UnpackBatch(*bad_packed).ok());

    auto short_scales = sc;
    short_scales.data.resize(15);
    auto bad_scales = lc::PinnedSchema::MakeBatch({pk, short_scales, gs});
    REQUIRE(bad_scales.ok());
    CHECK_FALSE(lc::PinnedSchema::UnpackBatch(*bad_scales).ok());
}

// ── Write-path allocation budget (no redundant copies / growth reallocs) ───
//
// The converters write the whole expert tree through these MakeBatch calls.
// A batch must allocate each payload once: a redundant staging copy or a
// geometrically growing builder doubles the host bytes copied and the peak
// RSS, which is measurable on an 800 GB tree. A tracking pool makes that
// budget deterministic (no timing flakiness).

#include <arrow/memory_pool.h>

namespace {

class TrackedPool final : public arrow::MemoryPool {
public:
    arrow::Status Allocate(int64_t size, int64_t alignment,
                           uint8_t **out) override
    {
        auto st = arrow::default_memory_pool()->Allocate(size, alignment, out);
        if (st.ok()) {
            ++num_allocations_;
            total_allocated_ += size;
            Account(size);
        }
        return st;
    }
    arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                             int64_t alignment, uint8_t **ptr) override
    {
        auto st = arrow::default_memory_pool()->Reallocate(
            old_size, new_size, alignment, ptr);
        if (st.ok()) {
            if (new_size > old_size) {
                ++num_growing_reallocations_;
                total_allocated_ += new_size - old_size;
            }
            Account(new_size - old_size);
        }
        return st;
    }
    void Free(uint8_t *buffer, int64_t size, int64_t alignment) override
    {
        arrow::default_memory_pool()->Free(buffer, size, alignment);
        current_ -= size;
    }
    int64_t bytes_allocated() const override { return current_; }
    int64_t max_memory() const override
    {
        return arrow::default_memory_pool()->max_memory();
    }
    int64_t total_bytes_allocated() const override { return total_allocated_; }
    int64_t num_allocations() const override { return num_allocations_; }
    std::string backend_name() const override { return "tracked"; }

    int64_t peak() const { return peak_; }
    int64_t num_growing_reallocations() const
    {
        return num_growing_reallocations_;
    }

private:
    void Account(int64_t delta)
    {
        current_ += delta;
        if (current_ > peak_) peak_ = current_;
    }
    int64_t current_ = 0;
    int64_t peak_ = 0;
    int64_t total_allocated_ = 0;
    int64_t num_allocations_ = 0;
    int64_t num_growing_reallocations_ = 0;
};

} // namespace

TEST_CASE("ExpertSchema::MakeBatch allocates each payload once",
          "[arrow][write][budget]")
{
    constexpr int64_t kMiB = 1024 * 1024;
    lc::PackedExpert pe;
    pe.w1_packed.assign(static_cast<size_t>(8 * kMiB), 0x11u);
    pe.w2_packed.assign(static_cast<size_t>(8 * kMiB), 0x22u);
    pe.w3_packed.assign(static_cast<size_t>(4 * kMiB), 0x33u);
    pe.w1_scales.assign(4096, 0x81u);
    pe.w2_scales.assign(4096, 0x82u);
    pe.w3_scales.assign(2048, 0x83u);
    const int64_t payload = 8 * kMiB + 8 * kMiB + 4 * kMiB;

    TrackedPool pool;
    auto batch = lc::ExpertSchema::MakeBatch(pe, &pool);
    REQUIRE(batch.ok());
    INFO("peak=" << pool.peak() << " payload=" << payload
                 << " allocated=" << pool.bytes_allocated());
    // One values allocation per weight plus the small offsets/scales buffers.
    CHECK(pool.peak() <= payload + kMiB);
    CHECK(pool.bytes_allocated() <= payload + kMiB);
}

TEST_CASE("PinnedSchema::MakeBatch reserves the values buffer",
          "[arrow][write][budget]")
{
    constexpr int64_t kMiB = 1024 * 1024;
    std::vector<lc::TensorRecord> tensors;
    for (int i = 0; i < 32; ++i) {
        lc::TensorRecord t;
        t.name = "tensor." + std::to_string(i);
        t.dtype = "BF16";
        t.shape = {256 * 1024};
        t.data.assign(static_cast<size_t>(kMiB), static_cast<uint8_t>(i));
        tensors.push_back(std::move(t));
    }
    const int64_t payload = 32 * kMiB;

    TrackedPool pool;
    auto batch = lc::PinnedSchema::MakeBatch(tensors, &pool);
    REQUIRE(batch.ok());
    INFO("peak=" << pool.peak() << " payload=" << payload
                 << " allocated=" << pool.bytes_allocated()
                 << " reallocs=" << pool.num_growing_reallocations());
    CHECK(pool.num_growing_reallocations() == 0);
    CHECK(pool.peak() <= payload + kMiB);
    CHECK(pool.bytes_allocated() <= payload + kMiB);
}

TEST_CASE("DenseNvfp4Schema::MakeBatch reserves packed and scale buffers",
          "[arrow][write][budget]")
{
    constexpr int64_t kMiB = 1024 * 1024;
    std::vector<lc::DenseNvfp4Record> recs;
    for (int i = 0; i < 32; ++i) {
        lc::DenseNvfp4Record r;
        r.name = "w." + std::to_string(i);
        r.dtype = "BF16";
        r.shape = {2048, 512};               // 1 048 576 elements
        r.packed.assign(static_cast<size_t>(kMiB / 2), 0x01u);
        r.scales.assign(static_cast<size_t>(kMiB / 32), 0x81u);
        r.global_scale = 0.5f;
        recs.push_back(std::move(r));
    }
    const int64_t payload = 32 * (kMiB / 2) + 32 * (kMiB / 32);

    TrackedPool pool;
    auto batch = lc::DenseNvfp4Schema::MakeBatch(recs, &pool);
    REQUIRE(batch.ok());
    INFO("peak=" << pool.peak() << " payload=" << payload
                 << " allocated=" << pool.bytes_allocated()
                 << " reallocs=" << pool.num_growing_reallocations());
    CHECK(pool.num_growing_reallocations() == 0);
    CHECK(pool.peak() <= payload + kMiB);
    CHECK(pool.bytes_allocated() <= payload + kMiB);
}

TEST_CASE("SharedExpertSchema::MakeBatch reserves all payload buffers",
          "[arrow][write][budget]")
{
    constexpr int64_t kMiB = 1024 * 1024;
    std::vector<lc::PackedExpert> layers;
    for (int i = 0; i < 16; ++i) {
        lc::PackedExpert e;
        e.w1_packed.assign(static_cast<size_t>(kMiB), 0x11u);
        e.w2_packed.assign(static_cast<size_t>(kMiB), 0x22u);
        e.w3_packed.assign(static_cast<size_t>(kMiB), 0x33u);
        e.w1_scales.assign(1024, 0x81u);
        e.w2_scales.assign(1024, 0x82u);
        e.w3_scales.assign(1024, 0x83u);
        e.w1_global_scale = 0.5f;
        e.w2_global_scale = 0.25f;
        e.w3_global_scale = 0.125f;
        layers.push_back(std::move(e));
    }
    const int64_t payload = 16 * 3 * kMiB + 16 * 3 * 1024;

    TrackedPool pool;
    auto batch = lc::SharedExpertSchema::MakeBatch(layers, &pool);
    REQUIRE(batch.ok());
    INFO("peak=" << pool.peak() << " payload=" << payload
                 << " allocated=" << pool.bytes_allocated()
                 << " reallocs=" << pool.num_growing_reallocations());
    CHECK(pool.num_growing_reallocations() == 0);
    CHECK(pool.peak() <= payload + kMiB);
    CHECK(pool.bytes_allocated() <= payload + kMiB);
}

// ── Exact routed-expert scale count (measured: every converted tree is
//    per-16 across packs) ────────────────────────────────────────────────

TEST_CASE("CheckExpertScaleCount enforces one scale per NVFP4 block",
          "[arrow][expert][scale]")
{
    CHECK(av::CheckExpertScaleCount(16, 2, 16, "w").ok());   // 32 elems
    CHECK_FALSE(av::CheckExpertScaleCount(16, 1, 16, "w").ok());
    CHECK_FALSE(av::CheckExpertScaleCount(16, 3, 16, "w").ok());
    CHECK(av::CheckExpertScaleCount(16, 1, 32, "w").ok());   // pre-repack
    CHECK(av::CheckExpertScaleCount(9, 2, 16, "w").ok());    // odd tail (17)
    CHECK(av::CheckExpertScaleCount(0, 0, 16, "w").ok());    // inactive slot
    CHECK_FALSE(av::CheckExpertScaleCount(16, 0, 16, "w").ok());
    CHECK_FALSE(av::CheckExpertScaleCount(0, 1, 16, "w").ok());
    CHECK_FALSE(av::CheckExpertScaleCount(16, 2, 0, "w").ok());
}

TEST_CASE("ExpertSchema::UnpackBatch checks the scale count when asked",
          "[arrow][expert][scale]")
{
    lc::PackedExpert e;
    e.w1_packed.assign(16, 0x11u);   // 32 elements
    e.w2_packed.assign(16, 0x22u);
    e.w3_packed.assign(16, 0x33u);
    e.w1_scales.assign(1, 0x81u);    // 32 / 32: a pre-repack (block-32) record
    e.w2_scales.assign(1, 0x82u);
    e.w3_scales.assign(1, 0x83u);
    e.w1_global_scale = e.w2_global_scale = e.w3_global_scale = 1.0f;
    auto batch = lc::ExpertSchema::MakeBatch(e);
    REQUIRE(batch.ok());

    // Default keeps legacy/pre-repack layouts readable.
    CHECK(lc::ExpertSchema::UnpackBatch(*batch).ok());
    CHECK_FALSE(lc::ExpertSchema::UnpackBatch(*batch, 16).ok());
    CHECK(lc::ExpertSchema::UnpackBatch(*batch, 32).ok());
}

TEST_CASE("ExpertStore::DecodeEntry enforces the converted-tree block size",
          "[arrow][expert][scale][store]")
{
    lc::PackedExpert e;
    e.w1_packed.assign(16, 0x11u);
    e.w2_packed.assign(16, 0x22u);
    e.w3_packed.assign(16, 0x33u);
    e.w1_scales.assign(2, 0x81u);    // block 16
    e.w2_scales.assign(2, 0x82u);
    e.w3_scales.assign(2, 0x83u);
    e.w1_global_scale = e.w2_global_scale = e.w3_global_scale = 1.0f;
    auto batch = lc::ExpertSchema::MakeBatch(e);
    REQUIRE(batch.ok());

    ProbeEntry entry;
    CHECK(ProbeStore::DecodeEntry(*batch, entry).ok());
    CHECK(entry.w1_bytes == 16);

    auto bad = e;
    bad.w2_scales.assign(1, 0x82u);
    auto bad_batch = lc::ExpertSchema::MakeBatch(bad);
    REQUIRE(bad_batch.ok());
    ProbeEntry bad_entry;
    CHECK_FALSE(ProbeStore::DecodeEntry(*bad_batch, bad_entry).ok());
}

TEST_CASE("ReadRecordBatchFromFile rejects an empty or wrong-width batch",
          "[arrow][read][schema]")
{
    const auto dir = TempDir("read_batch");
    lc::TensorRecord t;
    t.name = "model.norm.weight";
    t.dtype = "F32";
    t.shape = {4};
    t.data.assign(16, 0x00u);
    auto batch = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(batch.ok());

    const auto good = dir / "good.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(*batch, good).ok());
    CHECK(lc::ReadRecordBatchFromFile(good).ok());
    CHECK(lc::ReadRecordBatchFromFile(good, 4).ok());
    CHECK_FALSE(lc::ReadRecordBatchFromFile(good, 5).ok());

    const auto empty = dir / "empty.arrow";
    REQUIRE(lc::WriteRecordBatchToFile((*batch)->Slice(0, 0), empty).ok());
    CHECK_FALSE(lc::ReadRecordBatchFromFile(empty).ok());
    fs::remove_all(dir);
}

// ── Zero-copy expert batch (MakeBatchView) ────────────────────────────────

TEST_CASE("MakeBatchView wraps the expert payload without copying",
          "[arrow][write][view][budget]")
{
    constexpr int64_t kMiB = 1024 * 1024;
    lc::PackedExpert pe;
    pe.w1_packed.assign(static_cast<size_t>(8 * kMiB), 0x11u);
    pe.w2_packed.assign(static_cast<size_t>(8 * kMiB), 0x22u);
    pe.w3_packed.assign(static_cast<size_t>(4 * kMiB), 0x33u);
    pe.w1_scales.assign(8 * kMiB / 8, 0x81u);   // payload*2 elems / 16
    pe.w2_scales.assign(8 * kMiB / 8, 0x82u);
    pe.w3_scales.assign(4 * kMiB / 8, 0x83u);
    pe.w1_global_scale = pe.w2_global_scale = pe.w3_global_scale = 1.0f;
    const int64_t payload = 20 * kMiB;

    TrackedPool pool;
    auto batch = lc::ExpertSchema::MakeBatchView(pe, &pool);
    REQUIRE(batch.ok());
    INFO("peak=" << pool.peak() << " payload=" << payload);
    // The three payloads are wrapped, not allocated: only the (small) scales,
    // offsets and global-scale floats touch the pool.
    CHECK(pool.peak() <= 4 * kMiB);

    // Zero-copy proof: mutating the source is visible through the batch.
    auto w1 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        (*batch)->column(lc::ExpertSchema::kIdxW1Weight));
    REQUIRE(w1 != nullptr);
    REQUIRE(w1->Value(0).size() == 8 * kMiB);
    pe.w1_packed[123] = 0xABu;
    CHECK(static_cast<uint8_t>(w1->Value(0)[123]) == 0xABu);
    pe.w1_packed[123] = 0x11u;

    // And it round-trips through a write/read (the source outlives the batch).
    const auto dir = TempDir("make_batch_view");
    const auto path = dir / "expert.arrow";
    REQUIRE(lc::WriteRecordBatchToFile(*batch, path).ok());
    auto stored = lc::ReadRecordBatchFromFile(path, lc::ExpertSchema::kNumFields);
    REQUIRE(stored.ok());
    auto back = lc::ExpertSchema::UnpackBatch(*stored, 16);
    REQUIRE(back.ok());
    CHECK(back->w1_packed == pe.w1_packed);
    CHECK(back->w2_packed == pe.w2_packed);
    CHECK(back->w3_packed == pe.w3_packed);
    CHECK(back->w1_scales == pe.w1_scales);
    fs::remove_all(dir);
}

// ── Shared tree verifier (also the `lsxforge --verify` body) ──────────────

TEST_CASE("VerifyArrowTree reports missing and corrupt artifacts",
          "[arrow][tree][verify]")
{
    const auto dir = TempDir("verify_tree");
    {
        const auto missing = av::VerifyArrowTree(dir / "absent");
        CHECK_FALSE(missing.problems.empty());
    }

    lc::TensorRecord t;
    t.name = "model.norm.weight";
    t.dtype = "F32";
    t.shape = {4};
    t.data.assign(16, 0x00u);
    auto batch = lc::PinnedSchema::MakeBatch({t});
    REQUIRE(batch.ok());
    const auto path = dir / "pinned" / "global.arrow";
    fs::create_directories(path.parent_path());
    REQUIRE(lc::WriteRecordBatchToFile(*batch, path).ok());

    {
        const auto ok = av::VerifyArrowTree(dir);
        INFO((ok.problems.empty() ? std::string() : ok.problems.front()));
        CHECK(ok.problems.empty());
        CHECK(ok.files == 1);
        CHECK(ok.footer_ok == 1);
        CHECK(ok.kinds.at(av::TreeFileKind::kPinned) == 1);
    }

    REQUIRE(lc::WriteRecordBatchToFile((*batch)->Slice(0, 0), path).ok());
    {
        const auto bad = av::VerifyArrowTree(dir);
        CHECK_FALSE(bad.problems.empty());
    }
    fs::remove_all(dir);
}

// ── GH-54936: a malformed map array from IPC must be rejected, not crash ───
//
// Arrow's writer always serializes a consistent FieldNode.null_count, so the
// malformed metadata the hardened reader guards against cannot be produced by
// building an invalid batch in memory: it has to be patched into the file.
namespace {

// FieldNode = {int64 length, int64 null_count}. A one-column map batch of
// `rows` rows serializes four nodes with identical values (map, entries
// struct, keys, items), i.e. one 64-byte run. Patch the second node (the
// entries struct) to advertise an unknown null count and a length far larger
// than the validity buffer Arrow wrote for it -- the exact shape GH-54936
// describes: `GetNullCount()` then counts set bits past the buffer.
bool PatchMapChildNullCount(const fs::path &path, int64_t rows)
{
    std::vector<char> bytes;
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        bytes.assign(std::istreambuf_iterator<char>(in), {});
    }

    std::array<char, 64> pattern{};
    for (std::size_t i = 0; i < pattern.size(); i += 16)
        std::memcpy(pattern.data() + i, &rows, sizeof(rows));

    const auto it = std::search(bytes.begin(), bytes.end(),
                                pattern.begin(), pattern.end());
    if (it == bytes.end()) return false;

    const int64_t unknown = arrow::kUnknownNullCount;
    const int64_t huge = int64_t{1} << 40;
    std::memcpy(&*(it + 16 + 0), &huge, sizeof(huge));
    std::memcpy(&*(it + 16 + 8), &unknown, sizeof(unknown));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

} // namespace

TEST_CASE("ReadRecordBatchFromFile rejects a malformed map array instead of crashing",
          "[arrow][ipc][regression]")
{
    const auto dir = TempDir("ipc_map");
    const auto path = dir / "map.arrow";
    constexpr int64_t kRows = 2;

    auto keys = std::make_shared<arrow::StringBuilder>();
    auto items = std::make_shared<arrow::Int32Builder>();
    arrow::MapBuilder mb(arrow::default_memory_pool(), keys, items);
    REQUIRE(mb.Append().ok());
    REQUIRE(keys->Append("a").ok());
    REQUIRE(items->Append(1).ok());
    REQUIRE(mb.Append().ok());
    REQUIRE(keys->Append("b").ok());
    REQUIRE(items->Append(2).ok());
    auto map_array = mb.Finish().ValueOrDie();
    REQUIRE(map_array->length() == kRows);

    auto schema = arrow::schema({arrow::field("m", map_array->type())});
    auto batch = arrow::RecordBatch::Make(schema, kRows, {map_array});

    const auto write_file = [&] {
        auto sink = arrow::io::FileOutputStream::Open(path.string()).ValueOrDie();
        auto writer =
            arrow::ipc::MakeFileWriter(sink.get(), schema).ValueOrDie();
        REQUIRE(writer->WriteRecordBatch(*batch).ok());
        REQUIRE(writer->Close().ok());
        REQUIRE(sink->Close().ok());
    };

    SECTION("a well-formed map file round-trips")
    {
        write_file();
        auto stored = lc::ReadRecordBatchFromFile(path, 1);
        REQUIRE(stored.ok());
        CHECK((*stored)->num_rows() == kRows);
        CHECK((*stored)->schema()->field(0)->type()->Equals(*map_array->type()));
    }

    SECTION("a map child claiming an unknown null count is rejected")
    {
        write_file();
        REQUIRE(PatchMapChildNullCount(path, kRows));
        auto stored = lc::ReadRecordBatchFromFile(path, 1);
        REQUIRE_FALSE(stored.ok());
        CHECK(stored.status().IsInvalid());
        CHECK(stored.status().ToString().find("Map array") != std::string::npos);
    }

    fs::remove_all(dir);
}
