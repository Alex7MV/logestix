#include "lsxcommon/schema.h"
#include "lsxcommon/arrow_validate.h"
#include "lsxcommon/diagnostics.h"
#include "lsxcommon/log.h"
#include "nvfp4_scale.h"
#include <arrow/type_fwd.h>
#include <arrow/builder.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <system_error>

namespace lsxcommon {

// E4M3 scale buffers are allocated by Arrow's default memory pool (64-byte aligned).
// Arrow IPC serialization also pads each buffer to 8-byte boundaries, ensuring
// that out-of-bounds vector reads (e.g., uint4 loads in CUDA kernels) stay within
// the allocation. The `raw_values()` pointer from UInt8Array is at least 64-byte aligned.

// Full unsigned-E4M3 (UE4M3) scalar codec for the per-block NVFP4 quantizer.
// Implementation lives in the shared nvfp4_scale.h (single source for host
// converter and device act-quant kernels).
uint8_t float_to_e4m3(float s)
{
    return nvfp4::FloatToE4m3(s);
}

float e4m3_to_float(uint8_t e4m3)
{
    return nvfp4::E4m3ToFloat(e4m3);
}

namespace {

arrow::Result<std::shared_ptr<arrow::Array>>
make_scales_u8(const std::vector<uint8_t> &scales, arrow::MemoryPool *pool)
{
    auto lb_val = std::make_shared<arrow::UInt8Builder>(pool);
    arrow::ListBuilder lb(pool, lb_val);
    ARROW_RETURN_NOT_OK(lb.Reserve(1));
    ARROW_RETURN_NOT_OK(lb.Append());
    auto *vb = static_cast<arrow::UInt8Builder *>(lb.value_builder());
    ARROW_RETURN_NOT_OK(vb->AppendValues(scales.data(), scales.size()));
    return lb.Finish();
}

} // anonymous namespace

std::shared_ptr<arrow::Schema> ExpertSchema::Make()
{
    auto binary_type = arrow::large_binary();
    auto list_u8     = arrow::list(arrow::uint8());

    return arrow::schema({
        arrow::field("w1_weight", binary_type),
        arrow::field("w2_weight", binary_type),
        arrow::field("w3_weight", binary_type),
        arrow::field("w1_scale",       list_u8),
        arrow::field("w2_scale",       list_u8),
        arrow::field("w3_scale",       list_u8),
        arrow::field("w1_global_scale", arrow::float32()),
        arrow::field("w2_global_scale", arrow::float32()),
        arrow::field("w3_global_scale", arrow::float32()),
    });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ExpertSchema::MakeBatch(const PackedExpert &expert, arrow::MemoryPool *pool)
{
    auto schema = Make();
    arrow::MemoryPool *mp = pool ? pool : arrow::default_memory_pool();

    // The IPC writer copies each payload into the file stream; there is no
    // reason to stage it in a second host buffer first. Append straight from
    // the PackedExpert vectors (one copy instead of two).
    auto make_binary = [mp](const uint8_t *data, int64_t size)
        -> arrow::Result<std::shared_ptr<arrow::Array>>
    {
        arrow::LargeBinaryBuilder bb(mp);
        ARROW_RETURN_NOT_OK(bb.Reserve(1));
        ARROW_RETURN_NOT_OK(bb.ReserveData(size));
        ARROW_RETURN_NOT_OK(bb.Append(data, size));
        return bb.Finish();
    };
    ARROW_ASSIGN_OR_RAISE(auto w1_arr, make_binary(
        expert.w1_packed.data(), static_cast<int64_t>(expert.w1_packed.size())));
    ARROW_ASSIGN_OR_RAISE(auto w2_arr, make_binary(
        expert.w2_packed.data(), static_cast<int64_t>(expert.w2_packed.size())));
    ARROW_ASSIGN_OR_RAISE(auto w3_arr, make_binary(
        expert.w3_packed.data(), static_cast<int64_t>(expert.w3_packed.size())));

    ARROW_ASSIGN_OR_RAISE(auto s1, make_scales_u8(expert.w1_scales, mp));
    ARROW_ASSIGN_OR_RAISE(auto s2, make_scales_u8(expert.w2_scales, mp));
    ARROW_ASSIGN_OR_RAISE(auto s3, make_scales_u8(expert.w3_scales, mp));

    auto make_gs = [mp](float val) -> arrow::Result<std::shared_ptr<arrow::Array>> {
        arrow::FloatBuilder fb(mp);
        ARROW_RETURN_NOT_OK(fb.Reserve(1));
        ARROW_RETURN_NOT_OK(fb.Append(val));
        return fb.Finish();
    };
    ARROW_ASSIGN_OR_RAISE(auto gs1, make_gs(expert.w1_global_scale));
    ARROW_ASSIGN_OR_RAISE(auto gs2, make_gs(expert.w2_global_scale));
    ARROW_ASSIGN_OR_RAISE(auto gs3, make_gs(expert.w3_global_scale));

    auto batch = arrow::RecordBatch::Make(schema, 1,
        {w1_arr, w2_arr, w3_arr, s1, s2, s3, gs1, gs2, gs3});
    diag::ValidateBatchBuffers(*batch, "ExpertSchema::MakeBatch result");
    return batch;
}

arrow::Result<PackedExpert>
ExpertSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch,
                          int32_t scale_block)
{
    if (!batch || batch->num_columns() != kNumFields)
        return arrow::Status::Invalid("Expected 9-column expert RecordBatch");
    if (batch->num_rows() != 1)
        return arrow::Status::Invalid(
            "ExpertSchema expects exactly one row, got ", batch->num_rows());

    diag::ValidateBatchBuffers(*batch, "ExpertSchema::UnpackBatch input");

    PackedExpert pe;

    auto extract_binary = [&](int idx, std::vector<uint8_t> &out,
                              std::vector<int64_t> &shape)
        -> arrow::Status
    {
        auto arr = batch->column(idx);
        auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(arr);
        if (!bin)
            return arrow::Status::TypeError("Column is not large_binary");

        auto view = bin->GetView(0);
        ARROW_RETURN_NOT_OK(arrow_check::CheckViewRange(
            view.data(), static_cast<int64_t>(view.size()),
            bin->value_data(), "expert weight column",
            std::to_string(idx)));
        out.assign(view.begin(), view.end());

        return arrow::Status::OK();
    };

    auto extract_scales = [&](int idx, std::vector<uint8_t> &scales)
        -> arrow::Status
    {
        auto s_arr = batch->column(idx);
        auto lst = std::dynamic_pointer_cast<arrow::ListArray>(s_arr);
        if (!lst)
            return arrow::Status::TypeError("Scale column is not list");

        auto flat = lst->values();
        auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(flat);
        if (!u8)
            return arrow::Status::TypeError("Scale list is not uint8");

        auto offsets = lst->offsets();
        auto offset_arr = std::dynamic_pointer_cast<arrow::Int32Array>(offsets);
        if (!offset_arr)
            return arrow::Status::TypeError("Offsets array is not int32");
        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *offset_arr, 0, u8->length(), "expert scale list",
            std::to_string(idx)));
        int64_t begin = offset_arr->Value(0);
        int64_t end   = offset_arr->Value(1);
        scales.assign(u8->raw_values() + begin,
                      u8->raw_values() + end);

        return arrow::Status::OK();
    };

    ARROW_RETURN_NOT_OK(extract_binary(kIdxW1Weight, pe.w1_packed, pe.w1_shape));
    ARROW_RETURN_NOT_OK(extract_binary(kIdxW2Weight, pe.w2_packed, pe.w2_shape));
    ARROW_RETURN_NOT_OK(extract_binary(kIdxW3Weight, pe.w3_packed, pe.w3_shape));

    ARROW_RETURN_NOT_OK(extract_scales(kIdxW1Scale, pe.w1_scales));
    ARROW_RETURN_NOT_OK(extract_scales(kIdxW2Scale, pe.w2_scales));
    ARROW_RETURN_NOT_OK(extract_scales(kIdxW3Scale, pe.w3_scales));

    auto read_gs = [&](int idx, float &out) -> arrow::Status {
        auto arr = batch->column(idx);
        auto f = std::dynamic_pointer_cast<arrow::FloatArray>(arr);
        if (!f) return arrow::Status::TypeError("global_scale column not float32");
        out = f->Value(0);
        return arrow::Status::OK();
    };
    ARROW_RETURN_NOT_OK(read_gs(kIdxW1GlobalScale, pe.w1_global_scale));
    ARROW_RETURN_NOT_OK(read_gs(kIdxW2GlobalScale, pe.w2_global_scale));
    ARROW_RETURN_NOT_OK(read_gs(kIdxW3GlobalScale, pe.w3_global_scale));

    // The routed-expert BLOCK SIZE is model-dependent (per-16 E4M3 for the
    // fresh-NVFP4 families, per-32/per-128 E8M0 for DeepSeek passthrough), so
    // only the encoding-independent invariants are enforced here: every
    // payload is present and every global scale is finite and positive.
    // An all-empty payload encodes an inactive expert slot (GLM writes
    // all-inactive defaults); a half-empty one is malformed.
    const auto check_expert = [scale_block](const std::vector<uint8_t> &packed,
                                            const std::vector<uint8_t> &scales,
                                            float gs,
                                            const char *what) -> arrow::Status {
        if (packed.empty() != scales.empty())
            return arrow::Status::Invalid(
                "expert ", what, " has a partial payload (packed=",
                packed.size(), ", scales=", scales.size(), ")");
        if (packed.empty())
            return arrow::Status::OK();
        if (!std::isfinite(gs) || gs <= 0.0f)
            return arrow::Status::Invalid(
                "expert ", what, " has an invalid global_scale ", gs);
        if (scale_block > 0)
            return arrow_check::CheckExpertScaleCount(
                static_cast<int64_t>(packed.size()),
                static_cast<int64_t>(scales.size()), scale_block, what);
        return arrow::Status::OK();
    };
    ARROW_RETURN_NOT_OK(check_expert(pe.w1_packed, pe.w1_scales,
                                     pe.w1_global_scale, "w1"));
    ARROW_RETURN_NOT_OK(check_expert(pe.w2_packed, pe.w2_scales,
                                     pe.w2_global_scale, "w2"));
    ARROW_RETURN_NOT_OK(check_expert(pe.w3_packed, pe.w3_scales,
                                     pe.w3_global_scale, "w3"));

    return pe;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ExpertSchema::MakeBatchView(const PackedExpert &expert, arrow::MemoryPool *pool)
{
    auto schema = Make();
    arrow::MemoryPool *mp = pool ? pool : arrow::default_memory_pool();

    // LargeBinaryArray arrow data: {validity, offsets, values}. The values
    // buffer wraps the caller's vector, so nothing is copied here.
    auto wrap_binary = [](const std::vector<uint8_t> &bytes)
        -> arrow::Result<std::shared_ptr<arrow::Array>> {
        const int64_t size = static_cast<int64_t>(bytes.size());
        const int64_t offsets[2] = {0, size};
        auto off = arrow::Buffer::FromString(
            std::string(reinterpret_cast<const char *>(offsets),
                        sizeof(offsets)));
        auto vals = arrow::Buffer::Wrap(bytes);
        return arrow::MakeArray(arrow::ArrayData::Make(
            arrow::large_binary(), 1, {nullptr, off, vals},
            /*child_data=*/{}));
    };
    auto make_gs = [mp](float val)
        -> arrow::Result<std::shared_ptr<arrow::Array>> {
        arrow::FloatBuilder fb(mp);
        ARROW_RETURN_NOT_OK(fb.Reserve(1));
        ARROW_RETURN_NOT_OK(fb.Append(val));
        return fb.Finish();
    };

    ARROW_ASSIGN_OR_RAISE(auto w1_arr, wrap_binary(expert.w1_packed));
    ARROW_ASSIGN_OR_RAISE(auto w2_arr, wrap_binary(expert.w2_packed));
    ARROW_ASSIGN_OR_RAISE(auto w3_arr, wrap_binary(expert.w3_packed));
    ARROW_ASSIGN_OR_RAISE(auto s1, make_scales_u8(expert.w1_scales, mp));
    ARROW_ASSIGN_OR_RAISE(auto s2, make_scales_u8(expert.w2_scales, mp));
    ARROW_ASSIGN_OR_RAISE(auto s3, make_scales_u8(expert.w3_scales, mp));
    ARROW_ASSIGN_OR_RAISE(auto gs1, make_gs(expert.w1_global_scale));
    ARROW_ASSIGN_OR_RAISE(auto gs2, make_gs(expert.w2_global_scale));
    ARROW_ASSIGN_OR_RAISE(auto gs3, make_gs(expert.w3_global_scale));

    auto batch = arrow::RecordBatch::Make(schema, 1,
        {w1_arr, w2_arr, w3_arr, s1, s2, s3, gs1, gs2, gs3});
    diag::ValidateBatchBuffers(*batch, "ExpertSchema::MakeBatchView result");
    return batch;
}

// ── SharedExpertSchema ───────────────────────────────────────────────────────

std::shared_ptr<arrow::Schema> SharedExpertSchema::Make()
{
    auto binary_type = arrow::large_binary();
    auto list_u8     = arrow::list(arrow::uint8());

    return arrow::schema({
        arrow::field("w1_weight", binary_type),
        arrow::field("w2_weight", binary_type),
        arrow::field("w3_weight", binary_type),
        arrow::field("w1_scale",       list_u8),
        arrow::field("w2_scale",       list_u8),
        arrow::field("w3_scale",       list_u8),
        arrow::field("w1_global_scale", arrow::float32()),
        arrow::field("w2_global_scale", arrow::float32()),
        arrow::field("w3_global_scale", arrow::float32()),
    });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
SharedExpertSchema::MakeBatch(const std::vector<PackedExpert> &layers,
                              arrow::MemoryPool *pool)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(layers.size());
    arrow::MemoryPool *mp = pool ? pool : arrow::default_memory_pool();

    // Pre-size every values buffer: a geometrically growing builder
    // reallocates and re-copies its contents (up to ~2x the final size).
    int64_t bytes_w1 = 0, bytes_w2 = 0, bytes_w3 = 0;
    int64_t scales_w1 = 0, scales_w2 = 0, scales_w3 = 0;
    for (const auto &e : layers) {
        bytes_w1 += static_cast<int64_t>(e.w1_packed.size());
        bytes_w2 += static_cast<int64_t>(e.w2_packed.size());
        bytes_w3 += static_cast<int64_t>(e.w3_packed.size());
        scales_w1 += static_cast<int64_t>(e.w1_scales.size());
        scales_w2 += static_cast<int64_t>(e.w2_scales.size());
        scales_w3 += static_cast<int64_t>(e.w3_scales.size());
    }

    auto w1_builder = std::make_shared<arrow::LargeBinaryBuilder>(mp);
    auto w2_builder = std::make_shared<arrow::LargeBinaryBuilder>(mp);
    auto w3_builder = std::make_shared<arrow::LargeBinaryBuilder>(mp);
    ARROW_RETURN_NOT_OK(w1_builder->Reserve(n));
    ARROW_RETURN_NOT_OK(w2_builder->Reserve(n));
    ARROW_RETURN_NOT_OK(w3_builder->Reserve(n));
    ARROW_RETURN_NOT_OK(w1_builder->ReserveData(bytes_w1));
    ARROW_RETURN_NOT_OK(w2_builder->ReserveData(bytes_w2));
    ARROW_RETURN_NOT_OK(w3_builder->ReserveData(bytes_w3));

    auto s1_val = std::make_shared<arrow::UInt8Builder>(mp);
    auto s2_val = std::make_shared<arrow::UInt8Builder>(mp);
    auto s3_val = std::make_shared<arrow::UInt8Builder>(mp);
    arrow::ListBuilder s1_builder(mp, s1_val);
    arrow::ListBuilder s2_builder(mp, s2_val);
    arrow::ListBuilder s3_builder(mp, s3_val);
    ARROW_RETURN_NOT_OK(s1_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(s2_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(s3_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(s1_val->Reserve(scales_w1));
    ARROW_RETURN_NOT_OK(s2_val->Reserve(scales_w2));
    ARROW_RETURN_NOT_OK(s3_val->Reserve(scales_w3));

    arrow::FloatBuilder gs1_builder(mp), gs2_builder(mp), gs3_builder(mp);
    ARROW_RETURN_NOT_OK(gs1_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(gs2_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(gs3_builder.Reserve(n));

    for (const auto &e : layers) {
        ARROW_RETURN_NOT_OK(w1_builder->Append(e.w1_packed.data(), e.w1_packed.size()));
        ARROW_RETURN_NOT_OK(w2_builder->Append(e.w2_packed.data(), e.w2_packed.size()));
        ARROW_RETURN_NOT_OK(w3_builder->Append(e.w3_packed.data(), e.w3_packed.size()));

        ARROW_RETURN_NOT_OK(s1_builder.Append());
        ARROW_RETURN_NOT_OK(s1_val->AppendValues(e.w1_scales.data(), e.w1_scales.size()));
        ARROW_RETURN_NOT_OK(s2_builder.Append());
        ARROW_RETURN_NOT_OK(s2_val->AppendValues(e.w2_scales.data(), e.w2_scales.size()));
        ARROW_RETURN_NOT_OK(s3_builder.Append());
        ARROW_RETURN_NOT_OK(s3_val->AppendValues(e.w3_scales.data(), e.w3_scales.size()));

        ARROW_RETURN_NOT_OK(gs1_builder.Append(e.w1_global_scale));
        ARROW_RETURN_NOT_OK(gs2_builder.Append(e.w2_global_scale));
        ARROW_RETURN_NOT_OK(gs3_builder.Append(e.w3_global_scale));
    }

    ARROW_ASSIGN_OR_RAISE(auto w1_arr, w1_builder->Finish());
    ARROW_ASSIGN_OR_RAISE(auto w2_arr, w2_builder->Finish());
    ARROW_ASSIGN_OR_RAISE(auto w3_arr, w3_builder->Finish());
    ARROW_ASSIGN_OR_RAISE(auto s1_arr, s1_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto s2_arr, s2_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto s3_arr, s3_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto gs1_arr, gs1_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto gs2_arr, gs2_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto gs3_arr, gs3_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n,
        {w1_arr, w2_arr, w3_arr, s1_arr, s2_arr, s3_arr,
         gs1_arr, gs2_arr, gs3_arr});
    diag::ValidateBatchBuffers(*batch, "SharedExpertSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<PackedExpert>>
SharedExpertSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != kNumFields)
        return arrow::Status::Invalid("Expected 9-column shared-expert RecordBatch");

    diag::ValidateBatchBuffers(*batch, "SharedExpertSchema::UnpackBatch input");

    int64_t n = batch->num_rows();
    std::vector<PackedExpert> layers;
    layers.resize(static_cast<size_t>(n));

    auto w1 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxW1Weight));
    auto w2 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxW2Weight));
    auto w3 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxW3Weight));
    auto s1 = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxW1Scale));
    auto s2 = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxW2Scale));
    auto s3 = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxW3Scale));
    auto gs1 = std::dynamic_pointer_cast<arrow::FloatArray>(batch->column(kIdxW1GlobalScale));
    auto gs2 = std::dynamic_pointer_cast<arrow::FloatArray>(batch->column(kIdxW2GlobalScale));
    auto gs3 = std::dynamic_pointer_cast<arrow::FloatArray>(batch->column(kIdxW3GlobalScale));

    if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3 || !gs1 || !gs2 || !gs3)
        return arrow::Status::Invalid("Shared-expert batch column type mismatch");

    // The scale columns are list<uint8>; casting the values/offsets lazily
    // (as the old lambda did) dereferenced null on a foreign schema.
    auto extract_scales = [](const arrow::ListArray *lst, int64_t row,
                             std::vector<uint8_t> &out) -> arrow::Status {
        auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
        auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
        if (!u8 || !offs)
            return arrow::Status::TypeError(
                "Shared-expert scale column is not list<uint8>");
        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *offs, row, u8->length(), "shared-expert scale list",
            std::to_string(row)));
        const int64_t begin = offs->Value(row);
        const int64_t end = offs->Value(row + 1);
        out.assign(u8->raw_values() + begin, u8->raw_values() + end);
        return arrow::Status::OK();
    };
    auto extract_binary = [](const arrow::LargeBinaryArray *arr, int64_t row,
                             std::vector<uint8_t> &out) -> arrow::Status {
        auto v = arr->GetView(row);
        ARROW_RETURN_NOT_OK(arrow_check::CheckViewRange(
            v.data(), static_cast<int64_t>(v.size()), arr->value_data(),
            "shared-expert weight", std::to_string(row)));
        out.assign(v.begin(), v.end());
        return arrow::Status::OK();
    };

    for (int64_t i = 0; i < n; ++i) {
        auto &e = layers[static_cast<size_t>(i)];
        ARROW_RETURN_NOT_OK(extract_binary(w1.get(), i, e.w1_packed));
        ARROW_RETURN_NOT_OK(extract_binary(w2.get(), i, e.w2_packed));
        ARROW_RETURN_NOT_OK(extract_binary(w3.get(), i, e.w3_packed));
        ARROW_RETURN_NOT_OK(extract_scales(s1.get(), i, e.w1_scales));
        ARROW_RETURN_NOT_OK(extract_scales(s2.get(), i, e.w2_scales));
        ARROW_RETURN_NOT_OK(extract_scales(s3.get(), i, e.w3_scales));
        // Shared-expert encoding is family-specific (DeepSeek stores raw
        // F8_E4M3 with E8M0 128x128 tile scales; the NVFP4 families use
        // per-16 E4M3), so the schema cannot infer a block size. Offsets and
        // ranges are checked above; the size contract belongs to the loader.
        e.w1_global_scale = gs1->Value(i);
        e.w2_global_scale = gs2->Value(i);
        e.w3_global_scale = gs3->Value(i);
    }

    return layers;
}

// ── DenseNvfp4Schema ──────────────────────────────────────────────────────────

std::shared_ptr<arrow::Schema> DenseNvfp4Schema::Make()
{
    return arrow::schema({
        arrow::field("name",         arrow::utf8()),
        arrow::field("shape",        arrow::list(arrow::int64())),
        arrow::field("dtype",        arrow::utf8()),
        arrow::field("packed",       arrow::large_binary()),
        arrow::field("scales",       arrow::list(arrow::uint8())),
        arrow::field("global_scale", arrow::float32()),
    });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
DenseNvfp4Schema::MakeBatch(const std::vector<DenseNvfp4Record> &records,
                            arrow::MemoryPool *pool)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(records.size());
    arrow::MemoryPool *mp = pool ? pool : arrow::default_memory_pool();

    int64_t bytes_packed = 0, bytes_scales = 0, shape_rank = 0;
    int64_t name_bytes = 0, dtype_bytes = 0;
    for (const auto &r : records) {
        bytes_packed += static_cast<int64_t>(r.packed.size());
        bytes_scales += static_cast<int64_t>(r.scales.size());
        shape_rank += static_cast<int64_t>(r.shape.size());
        name_bytes += static_cast<int64_t>(r.name.size());
        dtype_bytes += static_cast<int64_t>(r.dtype.size());
    }

    arrow::StringBuilder name_builder(mp);
    arrow::StringBuilder dtype_builder(mp);
    arrow::LargeBinaryBuilder packed_builder(mp);
    arrow::FloatBuilder gs_builder(mp);

    auto shape_val_builder = std::make_shared<arrow::Int64Builder>(mp);
    arrow::ListBuilder shape_builder(mp, shape_val_builder);
    auto scale_val_builder = std::make_shared<arrow::UInt8Builder>(mp);
    arrow::ListBuilder scale_builder(mp, scale_val_builder);

    ARROW_RETURN_NOT_OK(name_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(dtype_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(packed_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(gs_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(shape_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(scale_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(name_builder.ReserveData(name_bytes));
    ARROW_RETURN_NOT_OK(dtype_builder.ReserveData(dtype_bytes));
    ARROW_RETURN_NOT_OK(packed_builder.ReserveData(bytes_packed));
    ARROW_RETURN_NOT_OK(shape_val_builder->Reserve(shape_rank));
    ARROW_RETURN_NOT_OK(scale_val_builder->Reserve(bytes_scales));

    for (auto &r : records) {
        ARROW_RETURN_NOT_OK(name_builder.Append(r.name));
        ARROW_RETURN_NOT_OK(dtype_builder.Append(r.dtype));
        ARROW_RETURN_NOT_OK(packed_builder.Append(r.packed.data(), r.packed.size()));
        ARROW_RETURN_NOT_OK(gs_builder.Append(r.global_scale));
        ARROW_RETURN_NOT_OK(shape_builder.Append());
        for (int64_t d : r.shape)
            ARROW_RETURN_NOT_OK(shape_val_builder->Append(d));
        ARROW_RETURN_NOT_OK(scale_builder.Append());
        for (uint8_t s : r.scales)
            ARROW_RETURN_NOT_OK(scale_val_builder->Append(s));
    }

    ARROW_ASSIGN_OR_RAISE(auto name_arr, name_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto shape_arr, shape_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto dtype_arr, dtype_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto packed_arr, packed_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto scales_arr, scale_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto gs_arr, gs_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n,
        {name_arr, shape_arr, dtype_arr, packed_arr, scales_arr, gs_arr});
    diag::ValidateBatchBuffers(*batch, "DenseNvfp4Schema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<DenseNvfp4Record>>
DenseNvfp4Schema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 6)
        return arrow::Status::Invalid("Expected 6-column DenseNvfp4 RecordBatch");

    diag::ValidateBatchBuffers(*batch, "DenseNvfp4Schema::UnpackBatch input");

    auto names = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxName));
    auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxDtype));
    auto packeds = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxPacked));
    auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxShape));
    auto scales = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxScales));
    auto gs_arr = std::dynamic_pointer_cast<arrow::FloatArray>(batch->column(kIdxGlobalScale));

    if (!names || !dtypes || !packeds || !shapes || !scales || !gs_arr)
        return arrow::Status::Invalid("DenseNvfp4 batch column type mismatch");

    int64_t n = batch->num_rows();
    std::vector<DenseNvfp4Record> records;
    records.reserve(static_cast<size_t>(n));

    auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(shapes->values());
    auto flat_scales = std::dynamic_pointer_cast<arrow::UInt8Array>(scales->values());
    if (!flat_shapes || !flat_scales)
        return arrow::Status::Invalid("DenseNvfp4 list value array type mismatch");

    auto shape_offsets = std::dynamic_pointer_cast<arrow::Int32Array>(shapes->offsets());
    auto scale_offsets = std::dynamic_pointer_cast<arrow::Int32Array>(scales->offsets());
    if (!shape_offsets || !scale_offsets)
        return arrow::Status::Invalid("DenseNvfp4 list offsets not int32");

    for (int64_t i = 0; i < n; ++i) {
        DenseNvfp4Record rec;
        rec.name = names->GetString(i);
        rec.dtype = dtypes->GetString(i);
        rec.global_scale = gs_arr->Value(i);

        auto pv = packeds->GetView(i);
        ARROW_RETURN_NOT_OK(arrow_check::CheckViewRange(
            pv.data(), static_cast<int64_t>(pv.size()), packeds->value_data(),
            "packed payload", rec.name));
        rec.packed.assign(pv.begin(), pv.end());

        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *shape_offsets, i, flat_shapes->length(), "shape", rec.name));
        int64_t sb = shape_offsets->Value(i);
        int64_t se = shape_offsets->Value(i + 1);
        for (int64_t j = sb; j < se; ++j)
            rec.shape.push_back(flat_shapes->Value(j));

        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *scale_offsets, i, flat_scales->length(), "scales", rec.name));
        int64_t cb = scale_offsets->Value(i);
        int64_t ce = scale_offsets->Value(i + 1);
        const uint8_t *sraw = flat_scales->raw_values();
        rec.scales.insert(rec.scales.end(), sraw + cb, sraw + ce);

        // The payload of a DenseNvfp4 record is always NVFP4; `dtype` names
        // the SOURCE tensor dtype (BF16/F8_E4M3/...), never the payload
        // encoding. Validate the packed/scale/global-scale invariant.
        ARROW_RETURN_NOT_OK(arrow_check::CheckNvfp4Size(
            rec.shape.data(), static_cast<int64_t>(rec.shape.size()),
            static_cast<int64_t>(rec.packed.size()),
            static_cast<int64_t>(rec.scales.size()), rec.global_scale,
            rec.name));

        records.push_back(std::move(rec));
    }
    return records;
}

// ── PinnedSchema ────────────────────────────────────────────────────────────

std::shared_ptr<arrow::Schema> PinnedSchema::Make()
{
    return arrow::schema({
        arrow::field("name",  arrow::utf8()),
        arrow::field("shape", arrow::list(arrow::int64())),
        arrow::field("dtype", arrow::utf8()),
        arrow::field("data",  arrow::large_binary()),
    });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
PinnedSchema::MakeBatch(const std::vector<TensorRecord> &tensors,
                        arrow::MemoryPool *pool)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(tensors.size());
    arrow::MemoryPool *mp = pool ? pool : arrow::default_memory_pool();

    int64_t bytes_data = 0, shape_rank = 0;
    int64_t name_bytes = 0, dtype_bytes = 0;
    for (const auto &t : tensors) {
        bytes_data += static_cast<int64_t>(t.data.size());
        shape_rank += static_cast<int64_t>(t.shape.size());
        name_bytes += static_cast<int64_t>(t.name.size());
        dtype_bytes += static_cast<int64_t>(t.dtype.size());
    }

    arrow::StringBuilder name_builder(mp);
    arrow::StringBuilder dtype_builder(mp);
    arrow::LargeBinaryBuilder data_builder(mp);

    auto shape_val_builder = std::make_shared<arrow::Int64Builder>(mp);
    arrow::ListBuilder shape_builder(mp, shape_val_builder);

    ARROW_RETURN_NOT_OK(name_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(dtype_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(data_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(shape_builder.Reserve(n + 1));
    ARROW_RETURN_NOT_OK(name_builder.ReserveData(name_bytes));
    ARROW_RETURN_NOT_OK(dtype_builder.ReserveData(dtype_bytes));
    ARROW_RETURN_NOT_OK(data_builder.ReserveData(bytes_data));
    ARROW_RETURN_NOT_OK(shape_val_builder->Reserve(shape_rank));

    for (auto &t : tensors) {
        ARROW_RETURN_NOT_OK(name_builder.Append(t.name));
        ARROW_RETURN_NOT_OK(dtype_builder.Append(t.dtype));
        ARROW_RETURN_NOT_OK(data_builder.Append(t.data.data(), t.data.size()));
        ARROW_RETURN_NOT_OK(shape_builder.Append());
        for (int64_t d : t.shape)
            ARROW_RETURN_NOT_OK(shape_val_builder->Append(d));
    }

    ARROW_ASSIGN_OR_RAISE(auto name_arr, name_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto shape_arr, shape_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto dtype_arr, dtype_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto data_arr, data_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n,
        {name_arr, shape_arr, dtype_arr, data_arr});
    diag::ValidateBatchBuffers(*batch, "PinnedSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<TensorRecord>>
PinnedSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 4)
        return arrow::Status::Invalid("Expected 4-column pinned RecordBatch");

    diag::ValidateBatchBuffers(*batch, "PinnedSchema::UnpackBatch input");

    auto names = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxName));
    auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxDtype));
    auto datas = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxData));
    auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(kIdxShape));

    if (!names || !dtypes || !datas || !shapes)
        return arrow::Status::Invalid("Pinned batch column type mismatch");

    auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(shapes->values());
    if (!flat_shapes)
        return arrow::Status::Invalid("Shape array is not int64");
    auto offset_arr = std::dynamic_pointer_cast<arrow::Int32Array>(shapes->offsets());
    if (!offset_arr)
        return arrow::Status::Invalid("Shape offsets not int32");

    int64_t n = batch->num_rows();
    std::vector<TensorRecord> tensors;
    tensors.reserve(static_cast<size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        TensorRecord t;
        t.name = names->GetString(i);
        t.dtype = dtypes->GetString(i);
        auto view = datas->GetView(i);
        ARROW_RETURN_NOT_OK(arrow_check::CheckViewRange(
            view.data(), static_cast<int64_t>(view.size()), datas->value_data(),
            "tensor data", t.name));
        t.data.assign(view.begin(), view.end());

        ARROW_RETURN_NOT_OK(arrow_check::CheckListSlice(
            *offset_arr, i, flat_shapes->length(), "shape", t.name));
        const int64_t begin = offset_arr->Value(i);
        const int64_t end = offset_arr->Value(i + 1);
        for (int64_t j = begin; j < end; ++j)
            t.shape.push_back(flat_shapes->Value(j));

        ARROW_RETURN_NOT_OK(arrow_check::CheckPinnedTensor(
            t.name, t.dtype, t.shape.data(),
            static_cast<int64_t>(t.shape.size()),
            static_cast<int64_t>(t.data.size())));

        tensors.push_back(std::move(t));
    }

    return tensors;
}

// ── Tokenizer schemas ───────────────────────────────────────────────────────

namespace {

std::string SerializeByteToTokenId(const std::array<int32_t, 256> &b)
{
    return absl::StrJoin(b, ",");
}

std::array<int32_t, 256> DeserializeByteToTokenId(const std::string &s)
{
    std::array<int32_t, 256> out{};
    size_t i = 0;
    for (absl::string_view part : absl::StrSplit(s, ',')) {
        int32_t v = 0;
        if (!absl::SimpleAtoi(part, &v)) break;
        if (i >= out.size()) break;
        out[i++] = v;
    }
    return out;
}

} // namespace

std::shared_ptr<arrow::Schema> TokenizerVocabSchema::Make(const TokenizerVocabMeta &meta)
{
    auto kv = arrow::KeyValueMetadata::Make({
        "bos_id", "eos_id", "pad_id", "vocab_size", "byte_to_token_id",
        "tokenizer_kind", "base_vocab_size", "unk_id", "byte_fallback",
        "add_dummy_prefix", "escape_whitespaces", "drop_unmapped_bytes",
        "normalize_nfc"
    }, {
        absl::StrCat(meta.bos_id),
        absl::StrCat(meta.eos_id),
        absl::StrCat(meta.pad_id),
        absl::StrCat(meta.vocab_size),
        SerializeByteToTokenId(meta.byte_to_token_id),
        absl::StrCat(meta.tokenizer_kind),
        absl::StrCat(meta.base_vocab_size),
        absl::StrCat(meta.unk_id),
        absl::StrCat(meta.byte_fallback),
        absl::StrCat(meta.add_dummy_prefix),
        absl::StrCat(meta.escape_whitespaces),
        absl::StrCat(meta.drop_unmapped_bytes),
        absl::StrCat(meta.normalize_nfc)
    });
    return arrow::schema({
        arrow::field("token_id", arrow::int32()),
        arrow::field("token_bytes", arrow::large_binary()),
        arrow::field("is_special", arrow::int8()),
    }, kv);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
TokenizerVocabSchema::MakeBatch(const std::vector<VocabRow> &rows,
                                const TokenizerVocabMeta &meta)
{
    auto schema = Make(meta);
    const int64_t n = static_cast<int64_t>(rows.size());

    arrow::Int32Builder id_builder;
    arrow::LargeBinaryBuilder bytes_builder;
    arrow::Int8Builder spec_builder;
    ARROW_RETURN_NOT_OK(id_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(bytes_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(spec_builder.Reserve(n));

    for (const auto &r : rows) {
        ARROW_RETURN_NOT_OK(id_builder.Append(r.token_id));
        ARROW_RETURN_NOT_OK(bytes_builder.Append(
            r.token_bytes.data(), static_cast<int64_t>(r.token_bytes.size())));
        ARROW_RETURN_NOT_OK(spec_builder.Append(r.is_special));
    }

    ARROW_ASSIGN_OR_RAISE(auto id_arr, id_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto bytes_arr, bytes_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto spec_arr, spec_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n, {id_arr, bytes_arr, spec_arr});
    diag::ValidateBatchBuffers(*batch, "TokenizerVocabSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<VocabRow>>
TokenizerVocabSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 3)
        return arrow::Status::Invalid("Expected 3-column tokenizer vocab RecordBatch");

    auto ids = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxTokenId));
    auto bytes = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(batch->column(kIdxTokenBytes));
    auto specs = std::dynamic_pointer_cast<arrow::Int8Array>(batch->column(kIdxIsSpecial));
    if (!ids || !bytes || !specs)
        return arrow::Status::TypeError("tokenizer vocab column type mismatch");

    const int64_t n = batch->num_rows();
    std::vector<VocabRow> rows;
    rows.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto &r = rows[static_cast<size_t>(i)];
        r.token_id = ids->Value(i);
        r.is_special = specs->Value(i);
        auto v = bytes->GetView(i);
        r.token_bytes.assign(v.begin(), v.end());
    }
    return rows;
}

std::shared_ptr<arrow::Schema> TokenizerMergesSchema::Make(int32_t merge_count)
{
    auto kv = arrow::KeyValueMetadata::Make({"merge_count"}, {absl::StrCat(merge_count)});
    return arrow::schema({
        arrow::field("merge_left", arrow::int32()),
        arrow::field("merge_right", arrow::int32()),
        arrow::field("merge_rank", arrow::int32()),
        arrow::field("merge_token_id", arrow::int32()),
    }, kv);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
TokenizerMergesSchema::MakeBatch(const std::vector<MergeRow> &rows, int32_t merge_count)
{
    auto schema = Make(merge_count);
    const int64_t n = static_cast<int64_t>(rows.size());

    arrow::Int32Builder left_builder, right_builder, rank_builder, token_builder;
    ARROW_RETURN_NOT_OK(left_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(right_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(rank_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(token_builder.Reserve(n));

    for (const auto &r : rows) {
        ARROW_RETURN_NOT_OK(left_builder.Append(r.merge_left));
        ARROW_RETURN_NOT_OK(right_builder.Append(r.merge_right));
        ARROW_RETURN_NOT_OK(rank_builder.Append(r.merge_rank));
        ARROW_RETURN_NOT_OK(token_builder.Append(r.merge_token_id));
    }

    ARROW_ASSIGN_OR_RAISE(auto left_arr, left_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto right_arr, right_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto rank_arr, rank_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto token_arr, token_builder.Finish());

    auto batch = arrow::RecordBatch::Make(schema, n,
        {left_arr, right_arr, rank_arr, token_arr});
    diag::ValidateBatchBuffers(*batch, "TokenizerMergesSchema::MakeBatch result");
    return batch;
}

arrow::Result<std::vector<MergeRow>>
TokenizerMergesSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 4)
        return arrow::Status::Invalid("Expected 4-column tokenizer merges RecordBatch");

    auto left = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeLeft));
    auto right = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeRight));
    auto rank = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeRank));
    auto token = std::dynamic_pointer_cast<arrow::Int32Array>(batch->column(kIdxMergeTokenId));
    if (!left || !right || !rank || !token)
        return arrow::Status::TypeError("tokenizer merges column type mismatch");

    const int64_t n = batch->num_rows();
    std::vector<MergeRow> rows;
    rows.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto &r = rows[static_cast<size_t>(i)];
        r.merge_left = left->Value(i);
        r.merge_right = right->Value(i);
        r.merge_rank = rank->Value(i);
        r.merge_token_id = token->Value(i);
    }
    return rows;
}

// ── File I/O helpers ────────────────────────────────────────────────────────

namespace {

// Removes the partially written temp artifact unless it was committed.
struct TempFileGuard {
    std::filesystem::path path;
    bool armed = true;
    ~TempFileGuard()
    {
        if (!armed) return;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

// Fault-injection seam for the atomicity regression test: the writer still
// writes the temp artifact. LSX_TEST_* is the established test-only env class.
bool ArrowWriteFailureInjected()
{
#if defined(_WIN32)
    char *value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, "LSX_TEST_ARROW_WRITE_FAIL") == 0 && value) {
        const bool on = value[0] == '1';
        std::free(value);
        return on;
    }
    return false;
#else
    const char *value = std::getenv("LSX_TEST_ARROW_WRITE_FAIL");
    return value && value[0] == '1';
#endif
}

} // namespace

arrow::Status WriteRecordBatchToFile(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::filesystem::path &path)
{
    if (!batch)
        return arrow::Status::Invalid("WriteRecordBatchToFile: null batch");
    const auto parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent))
        return arrow::Status::IOError(
            "output directory does not exist: ", parent.string());

    diag::LogArrowSystemState("WriteRecordBatchToFile");
    diag::ValidateBatchBuffers(*batch, ("Write " + path.filename().string()).c_str());

    // Write to a sibling temp file and rename on success, so a failure (or a
    // crash) can never leave a partially written artifact at the final path.
    static std::atomic<uint64_t> write_seq{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path tmp = path;
    tmp += ".tmp." + std::to_string(stamp) + "." +
           std::to_string(write_seq.fetch_add(1));
    TempFileGuard guard{tmp};

    {
        ARROW_ASSIGN_OR_RAISE(auto outfile,
            arrow::io::FileOutputStream::Open(tmp.string()));
        ARROW_ASSIGN_OR_RAISE(auto writer,
            arrow::ipc::MakeFileWriter(outfile.get(), batch->schema()));
        ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
        ARROW_RETURN_NOT_OK(writer->Close());
        ARROW_RETURN_NOT_OK(outfile->Close());
    }
    if (ArrowWriteFailureInjected())
        return arrow::Status::IOError("injected Arrow write failure for ",
                                      path.string());

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
#if defined(_WIN32)
    // Windows rename cannot replace an existing file.
    if (ec) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
#endif
    if (ec)
        return arrow::Status::IOError("cannot commit artifact ", path.string(),
                                      ": ", ec.message());
    guard.armed = false;
    lsxcommon::log::debug(absl::StrCat(
        "[DIAG] wrote batch to ", path.string(), " (", batch->num_columns(),
        " columns, ", batch->num_rows(), " rows)"));
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ReadRecordBatchFromFile(const std::filesystem::path &path,
                        int expected_columns)
{
    ARROW_ASSIGN_OR_RAISE(auto infile,
        arrow::io::MemoryMappedFile::Open(path.string(),
            arrow::io::FileMode::READ));
    ARROW_ASSIGN_OR_RAISE(auto reader,
        arrow::ipc::RecordBatchFileReader::Open(infile.get()));
    if (reader->num_record_batches() == 0)
        return arrow::Status::Invalid("No record batches in ", path.string());
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadRecordBatch(0));
    if (batch->num_rows() <= 0)
        return arrow::Status::Invalid(
            "artifact has no rows: ", path.string());
    if (expected_columns > 0 && batch->num_columns() != expected_columns)
        return arrow::Status::Invalid(
            "artifact ", path.string(), " has ", batch->num_columns(),
            " columns, expected ", expected_columns);
    diag::ValidateBatchBuffers(*batch, ("Read " + path.filename().string()).c_str());
    return batch;
}

// ── Zero-copy fast-load views ───────────────────────────────────────────────

namespace {

// Kicks an async kernel-side readahead for the whole mapping so cold-start
// disk reads overlap with the GPU upload of previously opened files instead of
// stalling the first page faults of each mmap view. Arrow's WillNeed is the
// portable form of the madvise hint (Windows included); the fadvise call it
// replaces was Linux-only.
void KickWholeFileReadahead(const std::shared_ptr<arrow::io::MemoryMappedFile> &mmap)
{
    if (!mmap) return;
    const auto size = mmap->GetSize();
    if (!size.ok() || *size <= 0) return;
    (void)mmap->WillNeed({{0, *size}});
}

} // namespace

namespace {

// Holds the mmap input stream + IPC reader + record batch so every buffer
// slice handed out through the views outlives them.
struct BatchKeepalive {
    std::shared_ptr<arrow::io::MemoryMappedFile> infile;
    std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
    std::shared_ptr<arrow::RecordBatch> batch;
};

// Order matters on destruction: batch → reader → infile.
struct KeepaliveBox {
    std::shared_ptr<BatchKeepalive> k;
};

} // namespace

namespace {

const int64_t *ShapePtr(const arrow::ListArray &shapes, int64_t i,
                        const arrow::Int64Array &flat, int64_t *rank)
{
    auto offsets = std::dynamic_pointer_cast<arrow::Int32Array>(shapes.offsets());
    if (!offsets) return nullptr;
    if (!arrow_check::CheckListSlice(*offsets, i, flat.length(),
                                     "shape", "view").ok())
        return nullptr;
    *rank = offsets->Value(i + 1) - offsets->Value(i);
    return flat.raw_values() + offsets->Value(i);
}

// Zero-copy views publish pointers into Arrow buffers. A corrupt/foreign
// batch (or a broken offset table) can produce a view whose extent crosses the
// backing buffer; reject it here so callers never mmap-dereference past the
// mapping (audit X15).
arrow::Status CheckViewRange(const void *p, int64_t bytes,
                             const std::shared_ptr<arrow::Buffer> &buf,
                             std::string_view what, std::string_view name)
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

arrow::Status CheckListSlice(const arrow::Int32Array &offsets, int64_t row,
                             int64_t values_length, std::string_view what,
                             std::string_view name)
{
    const int64_t begin = offsets.Value(row);
    const int64_t end = offsets.Value(row + 1);
    if (begin < 0 || end < begin || end > values_length)
        return arrow::Status::Invalid(what, " for '", name,
                                      "' has an out-of-range list slice [",
                                      begin, ", ", end, ") of ", values_length);
    return arrow::Status::OK();
}

} // namespace

arrow::Result<TensorBatchViews>
LoadTensorBatchViews(const std::filesystem::path &path)
{
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
    KickWholeFileReadahead(k->infile);
    ARROW_ASSIGN_OR_RAISE(
        k->reader, arrow::ipc::RecordBatchFileReader::Open(k->infile.get()));
    if (k->reader->num_record_batches() == 0)
        return arrow::Status::Invalid("No record batches in ", path.string());
    ARROW_ASSIGN_OR_RAISE(k->batch, k->reader->ReadRecordBatch(0));

    auto names = std::dynamic_pointer_cast<arrow::StringArray>(
        k->batch->column(PinnedSchema::kIdxName));
    auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(
        k->batch->column(PinnedSchema::kIdxDtype));
    auto datas = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        k->batch->column(PinnedSchema::kIdxData));
    auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(PinnedSchema::kIdxShape));
    if (!names || !dtypes || !datas || !shapes)
        return arrow::Status::Invalid("Pinned batch column type mismatch");
    auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(shapes->values());
    if (!flat_shapes)
        return arrow::Status::Invalid("Shape array is not int64");

    TensorBatchViews out;
    out.keepalive = std::make_shared<KeepaliveBox>(KeepaliveBox{k});
    const int64_t n = k->batch->num_rows();
    out.tensors.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        TensorView &v = out.tensors[static_cast<size_t>(i)];
        v.name = names->GetView(i);
        v.dtype = dtypes->GetView(i);
        v.shape = ShapePtr(*shapes, i, *flat_shapes, &v.rank);
        if (!v.shape) return arrow::Status::Invalid("Shape offsets not int32");
        auto view = datas->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            view.data(), static_cast<int64_t>(view.size()),
            datas->value_data(), "tensor data", v.name));
        v.data = reinterpret_cast<const uint8_t *>(view.data());
        v.bytes = static_cast<size_t>(view.size());
        ARROW_RETURN_NOT_OK(arrow_check::CheckPinnedTensor(
            v.name, v.dtype, v.shape, v.rank,
            static_cast<int64_t>(v.bytes)));
    }
    return out;
}

arrow::Result<DenseNvfp4BatchViews>
LoadDenseNvfp4BatchViews(const std::filesystem::path &path)
{
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
    KickWholeFileReadahead(k->infile);
    ARROW_ASSIGN_OR_RAISE(
        k->reader, arrow::ipc::RecordBatchFileReader::Open(k->infile.get()));
    if (k->reader->num_record_batches() == 0)
        return arrow::Status::Invalid("No record batches in ", path.string());
    ARROW_ASSIGN_OR_RAISE(k->batch, k->reader->ReadRecordBatch(0));

    auto names = std::dynamic_pointer_cast<arrow::StringArray>(
        k->batch->column(DenseNvfp4Schema::kIdxName));
    auto shapes = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(DenseNvfp4Schema::kIdxShape));
    auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(
        k->batch->column(DenseNvfp4Schema::kIdxDtype));
    auto packeds = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        k->batch->column(DenseNvfp4Schema::kIdxPacked));
    auto scales = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(DenseNvfp4Schema::kIdxScales));
    auto gs_arr = std::dynamic_pointer_cast<arrow::FloatArray>(
        k->batch->column(DenseNvfp4Schema::kIdxGlobalScale));
    if (!names || !dtypes || !packeds || !shapes || !scales || !gs_arr)
        return arrow::Status::Invalid("DenseNvfp4 batch column type mismatch");
    auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(shapes->values());
    auto flat_scales = std::dynamic_pointer_cast<arrow::UInt8Array>(scales->values());
    if (!flat_shapes || !flat_scales)
        return arrow::Status::Invalid("DenseNvfp4 list value array type mismatch");
    {
        auto shape_offsets =
            std::dynamic_pointer_cast<arrow::Int32Array>(shapes->offsets());
        auto scale_offsets =
            std::dynamic_pointer_cast<arrow::Int32Array>(scales->offsets());
        if (!shape_offsets || !scale_offsets)
            return arrow::Status::Invalid("DenseNvfp4 list offsets not int32");
    }

    DenseNvfp4BatchViews out;
    out.keepalive = std::make_shared<KeepaliveBox>(KeepaliveBox{k});
    const int64_t n = k->batch->num_rows();
    out.records.resize(static_cast<size_t>(n));

    auto scale_offsets =
        std::dynamic_pointer_cast<arrow::Int32Array>(scales->offsets());
    for (int64_t i = 0; i < n; ++i) {
        DenseNvfp4View &v = out.records[static_cast<size_t>(i)];
        v.name = names->GetView(i);
        v.dtype = dtypes->GetView(i);
        v.global_scale = gs_arr->Value(i);
        v.shape = ShapePtr(*shapes, i, *flat_shapes, &v.rank);
        if (!v.shape) return arrow::Status::Invalid("Shape offsets not int32");
        auto pv = packeds->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            pv.data(), static_cast<int64_t>(pv.size()), packeds->value_data(),
            "packed data", v.name));
        v.packed = reinterpret_cast<const uint8_t *>(pv.data());
        v.packed_bytes = static_cast<int64_t>(pv.size());
        ARROW_RETURN_NOT_OK(CheckListSlice(*scale_offsets, i,
                                           flat_scales->length(),
                                           "scales", v.name));
        const uint8_t *sraw = flat_scales->raw_values();
        v.scales = sraw + scale_offsets->Value(i);
        v.scale_bytes = static_cast<int64_t>(scale_offsets->Value(i + 1) -
                                             scale_offsets->Value(i));
        ARROW_RETURN_NOT_OK(CheckViewRange(
            v.scales, v.scale_bytes, flat_scales->data()->buffers[1],
            "scales", v.name));
        // See DenseNvfp4Schema::UnpackBatch: `dtype` is the source dtype; the
        // payload is always NVFP4.
        ARROW_RETURN_NOT_OK(arrow_check::CheckNvfp4Size(
            v.shape, v.rank, v.packed_bytes, v.scale_bytes,
            v.global_scale, v.name));
    }
    return out;
}

arrow::Result<SharedExpertBatchViews>
LoadSharedExpertBatchViews(const std::filesystem::path &path)
{
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
    KickWholeFileReadahead(k->infile);
    ARROW_ASSIGN_OR_RAISE(
        k->reader, arrow::ipc::RecordBatchFileReader::Open(k->infile.get()));
    if (k->reader->num_record_batches() == 0)
        return arrow::Status::Invalid("No record batches in ", path.string());
    ARROW_ASSIGN_OR_RAISE(k->batch, k->reader->ReadRecordBatch(0));

    auto w1 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        k->batch->column(SharedExpertSchema::kIdxW1Weight));
    auto w2 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        k->batch->column(SharedExpertSchema::kIdxW2Weight));
    auto w3 = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(
        k->batch->column(SharedExpertSchema::kIdxW3Weight));
    auto s1 = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(SharedExpertSchema::kIdxW1Scale));
    auto s2 = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(SharedExpertSchema::kIdxW2Scale));
    auto s3 = std::dynamic_pointer_cast<arrow::ListArray>(
        k->batch->column(SharedExpertSchema::kIdxW3Scale));
    auto gs1 = std::dynamic_pointer_cast<arrow::FloatArray>(
        k->batch->column(SharedExpertSchema::kIdxW1GlobalScale));
    auto gs2 = std::dynamic_pointer_cast<arrow::FloatArray>(
        k->batch->column(SharedExpertSchema::kIdxW2GlobalScale));
    auto gs3 = std::dynamic_pointer_cast<arrow::FloatArray>(
        k->batch->column(SharedExpertSchema::kIdxW3GlobalScale));
    if (!w1 || !w2 || !w3 || !s1 || !s2 || !s3 || !gs1 || !gs2 || !gs3)
        return arrow::Status::Invalid("Shared-expert batch column type mismatch");
    auto u1 = std::dynamic_pointer_cast<arrow::UInt8Array>(s1->values());
    auto u2 = std::dynamic_pointer_cast<arrow::UInt8Array>(s2->values());
    auto u3 = std::dynamic_pointer_cast<arrow::UInt8Array>(s3->values());
    auto o1 = std::dynamic_pointer_cast<arrow::Int32Array>(s1->offsets());
    auto o2 = std::dynamic_pointer_cast<arrow::Int32Array>(s2->offsets());
    auto o3 = std::dynamic_pointer_cast<arrow::Int32Array>(s3->offsets());
    if (!u1 || !u2 || !u3 || !o1 || !o2 || !o3)
        return arrow::Status::Invalid("Shared-expert list value/offset mismatch");

    SharedExpertBatchViews out;
    out.keepalive = std::make_shared<KeepaliveBox>(KeepaliveBox{k});
    const int64_t n = k->batch->num_rows();
    out.layers.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        PackedExpertView &v = out.layers[static_cast<size_t>(i)];
        const std::string name =
            "shared_experts[" + std::to_string(i) + "]";
        auto pv1 = w1->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            pv1.data(), static_cast<int64_t>(pv1.size()), w1->value_data(),
            "w1", name));
        v.w1_packed = reinterpret_cast<const uint8_t *>(pv1.data());
        v.w1_packed_bytes = static_cast<int64_t>(pv1.size());
        auto pv2 = w2->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            pv2.data(), static_cast<int64_t>(pv2.size()), w2->value_data(),
            "w2", name));
        v.w2_packed = reinterpret_cast<const uint8_t *>(pv2.data());
        v.w2_packed_bytes = static_cast<int64_t>(pv2.size());
        auto pv3 = w3->GetView(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            pv3.data(), static_cast<int64_t>(pv3.size()), w3->value_data(),
            "w3", name));
        v.w3_packed = reinterpret_cast<const uint8_t *>(pv3.data());
        v.w3_packed_bytes = static_cast<int64_t>(pv3.size());
        ARROW_RETURN_NOT_OK(CheckListSlice(*o1, i, u1->length(), "w1 scales",
                                           name));
        ARROW_RETURN_NOT_OK(CheckListSlice(*o2, i, u2->length(), "w2 scales",
                                           name));
        ARROW_RETURN_NOT_OK(CheckListSlice(*o3, i, u3->length(), "w3 scales",
                                           name));
        v.w1_scales = u1->raw_values() + o1->Value(i);
        v.w1_scale_bytes = o1->Value(i + 1) - o1->Value(i);
        v.w2_scales = u2->raw_values() + o2->Value(i);
        v.w2_scale_bytes = o2->Value(i + 1) - o2->Value(i);
        v.w3_scales = u3->raw_values() + o3->Value(i);
        v.w3_scale_bytes = o3->Value(i + 1) - o3->Value(i);
        ARROW_RETURN_NOT_OK(CheckViewRange(
            v.w1_scales, v.w1_scale_bytes, u1->data()->buffers[1], "w1 scales",
            name));
        ARROW_RETURN_NOT_OK(CheckViewRange(
            v.w2_scales, v.w2_scale_bytes, u2->data()->buffers[1], "w2 scales",
            name));
        ARROW_RETURN_NOT_OK(CheckViewRange(
            v.w3_scales, v.w3_scale_bytes, u3->data()->buffers[1], "w3 scales",
            name));
        v.w1_global_scale = gs1->Value(i);
        v.w2_global_scale = gs2->Value(i);
        v.w3_global_scale = gs3->Value(i);
    }
    return out;
}

} // namespace lsxcommon