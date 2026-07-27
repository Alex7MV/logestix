#include "lsxcommon/schema.h"
#include <arrow/type_fwd.h>
#include <arrow/builder.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <cstring>

namespace lsxcommon {

std::shared_ptr<arrow::Schema> ExpertSchema::Make()
{
    auto binary_type = arrow::binary();
    auto list_float  = arrow::list(arrow::float32());

    return arrow::schema({
        arrow::field("w1_weight", binary_type),
        arrow::field("w2_weight", binary_type),
        arrow::field("w3_weight", binary_type),
        arrow::field("w1_scale",  list_float),
        arrow::field("w2_scale",  list_float),
        arrow::field("w3_scale",  list_float),
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

    std::memcpy(w1_buf->mutable_data(), expert.w1_packed.data(),
                expert.w1_packed.size());
    std::memcpy(w2_buf->mutable_data(), expert.w2_packed.data(),
                expert.w2_packed.size());
    std::memcpy(w3_buf->mutable_data(), expert.w3_packed.data(),
                expert.w3_packed.size());

    auto make_binary = [](const void *data, int64_t size)
        -> arrow::Result<std::shared_ptr<arrow::Array>>
    {
        arrow::BinaryBuilder bb;
        ARROW_RETURN_NOT_OK(bb.Reserve(1));
        ARROW_RETURN_NOT_OK(bb.Append(
            static_cast<const uint8_t *>(data),
            static_cast<int32_t>(size)));
        return bb.Finish();
    };
    ARROW_ASSIGN_OR_RAISE(auto w1_arr, make_binary(w1_buf->data(), w1_buf->size()));
    ARROW_ASSIGN_OR_RAISE(auto w2_arr, make_binary(w2_buf->data(), w2_buf->size()));
    ARROW_ASSIGN_OR_RAISE(auto w3_arr, make_binary(w3_buf->data(), w3_buf->size()));

    auto make_scales = [](const std::vector<float> &scales)
        -> arrow::Result<std::shared_ptr<arrow::Array>>
    {
        arrow::FloatBuilder fb;
        ARROW_RETURN_NOT_OK(fb.Reserve(scales.size()));
        ARROW_RETURN_NOT_OK(fb.AppendValues(scales.data(), scales.size()));
        ARROW_ASSIGN_OR_RAISE(auto flat, fb.Finish());

        auto lb_val = std::make_shared<arrow::FloatBuilder>(arrow::default_memory_pool());
        arrow::ListBuilder lb(arrow::default_memory_pool(), lb_val);
        ARROW_RETURN_NOT_OK(lb.Reserve(1));
        ARROW_RETURN_NOT_OK(lb.Append());
        auto *vb = static_cast<arrow::FloatBuilder *>(lb.value_builder());
        ARROW_RETURN_NOT_OK(vb->AppendValues(scales.data(), scales.size()));
        return lb.Finish();
    };

    ARROW_ASSIGN_OR_RAISE(auto s1, make_scales(expert.w1_scales));
    ARROW_ASSIGN_OR_RAISE(auto s2, make_scales(expert.w2_scales));
    ARROW_ASSIGN_OR_RAISE(auto s3, make_scales(expert.w3_scales));

    return arrow::RecordBatch::Make(schema, 1,
        {w1_arr, w2_arr, w3_arr, s1, s2, s3});
}

arrow::Result<PackedExpert>
ExpertSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != kNumFields)
        return arrow::Status::Invalid("Expected 6-column expert RecordBatch");

    PackedExpert pe;

    auto extract_binary = [&](int idx, std::vector<uint8_t> &out,
                              std::vector<int64_t> &shape,
                              std::vector<float> &scales,
                              int scale_idx)
        -> arrow::Status
    {
        auto arr = batch->column(idx);
        auto bin = std::dynamic_pointer_cast<arrow::BinaryArray>(arr);
        if (!bin)
            return arrow::Status::TypeError("Column is not binary");

        auto view = bin->GetView(0);
        out.assign(view.begin(), view.end());

        auto s_arr = batch->column(scale_idx);
        auto lst = std::dynamic_pointer_cast<arrow::ListArray>(s_arr);
        if (!lst)
            return arrow::Status::TypeError("Scale column is not list");

        auto flat = lst->values();
        auto flt = std::dynamic_pointer_cast<arrow::FloatArray>(flat);
        if (!flt)
            return arrow::Status::TypeError("Scale list is not float32");

        auto offsets = lst->offsets();
        auto offset_arr = std::dynamic_pointer_cast<arrow::Int32Array>(offsets);
        if (!offset_arr)
            return arrow::Status::TypeError("Offsets array is not int32");
        int64_t begin = offset_arr->Value(0);
        int64_t end   = offset_arr->Value(1);
        scales.assign(flt->raw_values() + begin,
                      flt->raw_values() + end);

        return arrow::Status::OK();
    };

    ARROW_RETURN_NOT_OK(extract_binary(kIdxW1Weight, pe.w1_packed,
                                        pe.w1_shape, pe.w1_scales, kIdxW1Scale));
    ARROW_RETURN_NOT_OK(extract_binary(kIdxW2Weight, pe.w2_packed,
                                        pe.w2_shape, pe.w2_scales, kIdxW2Scale));
    ARROW_RETURN_NOT_OK(extract_binary(kIdxW3Weight, pe.w3_packed,
                                        pe.w3_shape, pe.w3_scales, kIdxW3Scale));

    return pe;
}

// ── PinnedSchema ────────────────────────────────────────────────────────────

std::shared_ptr<arrow::Schema> PinnedSchema::Make()
{
    return arrow::schema({
        arrow::field("name",  arrow::utf8()),
        arrow::field("shape", arrow::list(arrow::int64())),
        arrow::field("dtype", arrow::utf8()),
        arrow::field("data",  arrow::binary()),
    });
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
PinnedSchema::MakeBatch(const std::vector<TensorRecord> &tensors)
{
    auto schema = Make();
    int64_t n = static_cast<int64_t>(tensors.size());

    arrow::StringBuilder name_builder;
    arrow::StringBuilder dtype_builder;
    arrow::BinaryBuilder data_builder;

    auto shape_val_builder = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder shape_builder(arrow::default_memory_pool(), shape_val_builder);

    ARROW_RETURN_NOT_OK(name_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(dtype_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(data_builder.Reserve(n));
    ARROW_RETURN_NOT_OK(shape_builder.Reserve(n));

    for (auto &t : tensors) {
        ARROW_RETURN_NOT_OK(name_builder.Append(t.name));
        ARROW_RETURN_NOT_OK(dtype_builder.Append(t.dtype));
        ARROW_RETURN_NOT_OK(data_builder.Append(t.data.data(),
            static_cast<int32_t>(t.data.size())));
        ARROW_RETURN_NOT_OK(shape_builder.Append());
        for (int64_t d : t.shape)
            ARROW_RETURN_NOT_OK(shape_val_builder->Append(d));
    }

    ARROW_ASSIGN_OR_RAISE(auto name_arr, name_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto shape_arr, shape_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto dtype_arr, dtype_builder.Finish());
    ARROW_ASSIGN_OR_RAISE(auto data_arr, data_builder.Finish());

    return arrow::RecordBatch::Make(schema, n,
        {name_arr, shape_arr, dtype_arr, data_arr});
}

arrow::Result<std::vector<TensorRecord>>
PinnedSchema::UnpackBatch(const std::shared_ptr<arrow::RecordBatch> &batch)
{
    if (!batch || batch->num_columns() != 4)
        return arrow::Status::Invalid("Expected 4-column pinned RecordBatch");

    auto names = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxName));
    auto dtypes = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(kIdxDtype));
    auto datas = std::dynamic_pointer_cast<arrow::BinaryArray>(batch->column(kIdxData));
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

// ── File I/O helpers ────────────────────────────────────────────────────────

arrow::Status WriteRecordBatchToFile(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const std::filesystem::path &path)
{
    ARROW_ASSIGN_OR_RAISE(auto outfile,
        arrow::io::FileOutputStream::Open(path.string()));
    ARROW_ASSIGN_OR_RAISE(auto writer,
        arrow::ipc::MakeFileWriter(outfile.get(), batch->schema()));
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
    ARROW_RETURN_NOT_OK(writer->Close());
    ARROW_RETURN_NOT_OK(outfile->Close());
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
    return reader->ReadRecordBatch(0);
}

} // namespace lsxcommon