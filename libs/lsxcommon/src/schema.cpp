#include "lsxcommon/schema.h"
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
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
make_scales_u8(const std::vector<uint8_t> &scales)
{
    arrow::UInt8Builder ub;
    ARROW_RETURN_NOT_OK(ub.Reserve(scales.size()));
    ARROW_RETURN_NOT_OK(ub.AppendValues(scales.data(), scales.size()));
    ARROW_ASSIGN_OR_RAISE(auto flat, ub.Finish());
    auto lb_val = std::make_shared<arrow::UInt8Builder>(arrow::default_memory_pool());
    arrow::ListBuilder lb(arrow::default_memory_pool(), lb_val);
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
ExpertSchema::MakeBatch(const PackedExpert &expert)
{
    auto schema = Make();

    ARROW_ASSIGN_OR_RAISE(auto w1_buf,
        arrow::AllocateBuffer(expert.w1_packed.size(), kArrowAlignment));
    ARROW_ASSIGN_OR_RAISE(auto w2_buf,
        arrow::AllocateBuffer(expert.w2_packed.size(), kArrowAlignment));
    ARROW_ASSIGN_OR_RAISE(auto w3_buf,
        arrow::AllocateBuffer(expert.w3_packed.size(), kArrowAlignment));

    diag::LogBuffer("ExpertSchema::MakeBatch w1", *w1_buf);
    diag::LogBuffer("ExpertSchema::MakeBatch w2", *w2_buf);
    diag::LogBuffer("ExpertSchema::MakeBatch w3", *w3_buf);
    diag::ValidateCpuBuffer(*w1_buf, "ExpertSchema::MakeBatch w1 mutable_data");
    diag::ValidateCpuBuffer(*w2_buf, "ExpertSchema::MakeBatch w2 mutable_data");
    diag::ValidateCpuBuffer(*w3_buf, "ExpertSchema::MakeBatch w3 mutable_data");

    std::memcpy(w1_buf->mutable_data(), expert.w1_packed.data(),
                expert.w1_packed.size());
    std::memcpy(w2_buf->mutable_data(), expert.w2_packed.data(),
                expert.w2_packed.size());
    std::memcpy(w3_buf->mutable_data(), expert.w3_packed.data(),
                expert.w3_packed.size());

    auto make_binary = [](const void *data, int64_t size)
        -> arrow::Result<std::shared_ptr<arrow::Array>>
    {
        arrow::LargeBinaryBuilder bb;
        ARROW_RETURN_NOT_OK(bb.Reserve(1));
        ARROW_RETURN_NOT_OK(bb.Append(
            static_cast<const uint8_t *>(data), size));
        return bb.Finish();
    };
    ARROW_ASSIGN_OR_RAISE(auto w1_arr, make_binary(w1_buf->data(), w1_buf->size()));
    ARROW_ASSIGN_OR_RAISE(auto w2_arr, make_binary(w2_buf->data(), w2_buf->size()));
    ARROW_ASSIGN_OR_RAISE(auto w3_arr, make_binary(w3_buf->data(), w3_buf->size()));

    ARROW_ASSIGN_OR_RAISE(auto s1, make_scales_u8(expert.w1_scales));
    ARROW_ASSIGN_OR_RAISE(auto s2, make_scales_u8(expert.w2_scales));
    ARROW_ASSIGN_OR_RAISE(auto s3, make_scales_u8(expert.w3_scales));

    auto make_gs = [](float val) -> arrow::Result<std::shared_ptr<arrow::Array>> {
        arrow::FloatBuilder fb;
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
ExpertSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != kNumFields)
        return arrow::Status::Invalid("Expected 9-column expert RecordBatch");

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

    return pe;
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
SharedExpertSchema::MakeBatch(const std::vector<PackedExpert> &layers)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(layers.size());

    auto w1_builder = std::make_shared<arrow::LargeBinaryBuilder>();
    auto w2_builder = std::make_shared<arrow::LargeBinaryBuilder>();
    auto w3_builder = std::make_shared<arrow::LargeBinaryBuilder>();
    ARROW_RETURN_NOT_OK(w1_builder->Reserve(n));
    ARROW_RETURN_NOT_OK(w2_builder->Reserve(n));
    ARROW_RETURN_NOT_OK(w3_builder->Reserve(n));

    auto s1_val = std::make_shared<arrow::UInt8Builder>(arrow::default_memory_pool());
    auto s2_val = std::make_shared<arrow::UInt8Builder>(arrow::default_memory_pool());
    auto s3_val = std::make_shared<arrow::UInt8Builder>(arrow::default_memory_pool());
    arrow::ListBuilder s1_builder(arrow::default_memory_pool(), s1_val);
    arrow::ListBuilder s2_builder(arrow::default_memory_pool(), s2_val);
    arrow::ListBuilder s3_builder(arrow::default_memory_pool(), s3_val);
    ARROW_RETURN_NOT_OK(s1_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(s2_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(s3_builder.Reserve(n));

    arrow::FloatBuilder gs1_builder, gs2_builder, gs3_builder;
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

    auto extract_scales = [](const arrow::ListArray *lst, int64_t row,
                             std::vector<uint8_t> &out) -> void {
        auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
        auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
        int64_t begin = offs->Value(row);
        int64_t end   = offs->Value(row + 1);
        out.assign(u8->raw_values() + begin, u8->raw_values() + end);
    };

    for (int64_t i = 0; i < n; ++i) {
        auto &e = layers[static_cast<size_t>(i)];
        {
            auto v = w1->GetView(i);
            e.w1_packed.assign(v.begin(), v.end());
        }
        {
            auto v = w2->GetView(i);
            e.w2_packed.assign(v.begin(), v.end());
        }
        {
            auto v = w3->GetView(i);
            e.w3_packed.assign(v.begin(), v.end());
        }
        extract_scales(s1.get(), i, e.w1_scales);
        extract_scales(s2.get(), i, e.w2_scales);
        extract_scales(s3.get(), i, e.w3_scales);
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
DenseNvfp4Schema::MakeBatch(const std::vector<DenseNvfp4Record> &records)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(records.size());

    arrow::StringBuilder name_builder;
    arrow::StringBuilder dtype_builder;
    arrow::LargeBinaryBuilder packed_builder;
    arrow::FloatBuilder gs_builder;

    auto shape_val_builder = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder shape_builder(arrow::default_memory_pool(), shape_val_builder);
    auto scale_val_builder = std::make_shared<arrow::UInt8Builder>();
    arrow::ListBuilder scale_builder(arrow::default_memory_pool(), scale_val_builder);

    ARROW_RETURN_NOT_OK(name_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(dtype_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(packed_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(gs_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(shape_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(scale_builder.Reserve(n));

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
        rec.packed.assign(pv.begin(), pv.end());

        int64_t sb = shape_offsets->Value(i);
        int64_t se = shape_offsets->Value(i + 1);
        for (int64_t j = sb; j < se; ++j)
            rec.shape.push_back(flat_shapes->Value(j));

        int64_t cb = scale_offsets->Value(i);
        int64_t ce = scale_offsets->Value(i + 1);
        const uint8_t *sraw = flat_scales->raw_values();
        rec.scales.insert(rec.scales.end(), sraw + cb, sraw + ce);

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
PinnedSchema::MakeBatch(const std::vector<TensorRecord> &tensors)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(tensors.size());

    arrow::StringBuilder name_builder;
    arrow::StringBuilder dtype_builder;
    arrow::LargeBinaryBuilder data_builder;

    auto shape_val_builder = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder shape_builder(arrow::default_memory_pool(), shape_val_builder);

    ARROW_RETURN_NOT_OK(name_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(dtype_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(data_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(shape_builder.Reserve(n));

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

    int64_t n = batch->num_rows();
    std::vector<TensorRecord> tensors;
    tensors.reserve(static_cast<size_t>(n));

    for (int64_t i = 0; i < n; ++i) {
        TensorRecord t;
        t.name = names->GetString(i);
        t.dtype = dtypes->GetString(i);
        auto view = datas->GetView(i);
        t.data.assign(view.begin(), view.end());

        auto flat_shapes = std::dynamic_pointer_cast<arrow::Int64Array>(shapes->values());
        if (!flat_shapes)
            return arrow::Status::Invalid("Shape array is not int64");

        auto offsets = shapes->offsets();
        auto offset_arr = std::dynamic_pointer_cast<arrow::Int32Array>(offsets);
        if (!offset_arr)
            return arrow::Status::Invalid("Shape offsets not int32");

        int64_t begin = offset_arr->Value(i);
        int64_t end = offset_arr->Value(i + 1);
        for (int64_t j = begin; j < end; ++j)
            t.shape.push_back(flat_shapes->Value(j));

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

arrow::Status WriteRecordBatchToFile(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::filesystem::path &path)
{
    diag::LogArrowSystemState("WriteRecordBatchToFile");
    diag::ValidateBatchBuffers(*batch, ("Write " + path.filename().string()).c_str());
    ARROW_ASSIGN_OR_RAISE(auto outfile,
        arrow::io::FileOutputStream::Open(path.string()));
    ARROW_ASSIGN_OR_RAISE(auto writer,
        arrow::ipc::MakeFileWriter(outfile.get(), batch->schema()));
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    ARROW_RETURN_NOT_OK(outfile->Close());
    lsxcommon::log::debug(absl::StrCat(
        "[DIAG] wrote batch to ", path.string(), " (", batch->num_columns(),
        " columns, ", batch->num_rows(), " rows)"));
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
ReadRecordBatchFromFile(const std::filesystem::path &path)
{
    ARROW_ASSIGN_OR_RAISE(auto infile,
        arrow::io::MemoryMappedFile::Open(path.string(),
            arrow::io::FileMode::READ));
    ARROW_ASSIGN_OR_RAISE(auto reader,
        arrow::ipc::RecordBatchFileReader::Open(infile.get()));
    if (reader->num_record_batches() == 0)
        return arrow::Status::Invalid("No record batches in ", path.string());
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadRecordBatch(0));
    diag::ValidateBatchBuffers(*batch, ("Read " + path.filename().string()).c_str());
    return batch;
}

// ── Zero-copy fast-load views ───────────────────────────────────────────────

namespace {

// Kicks an async kernel-side readahead for the whole file so cold-start disk
// reads overlap with the GPU upload of previously opened files instead of
// stalling the first page faults of each mmap view.
void KickWholeFileReadahead(const std::filesystem::path &path)
{
#if defined(__linux__)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    struct stat st {};
    if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode))
        (void)::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_WILLNEED);
    ::close(fd);
#endif
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
    KickWholeFileReadahead(path);
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
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
    }
    return out;
}

arrow::Result<DenseNvfp4BatchViews>
LoadDenseNvfp4BatchViews(const std::filesystem::path &path)
{
    KickWholeFileReadahead(path);
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
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
    }
    return out;
}

arrow::Result<SharedExpertBatchViews>
LoadSharedExpertBatchViews(const std::filesystem::path &path)
{
    KickWholeFileReadahead(path);
    auto k = std::make_shared<BatchKeepalive>();
    ARROW_ASSIGN_OR_RAISE(
        k->infile, arrow::io::MemoryMappedFile::Open(path.string(),
                                                     arrow::io::FileMode::READ));
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