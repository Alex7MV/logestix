#include "lsxcommon/schema.h"
#include "lsxcommon/diagnostics.h"
#include <arrow/type_fwd.h>
#include <arrow/builder.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <cmath>
#include <cstring>

namespace lsxcommon {

// E4M3 scale buffers are allocated by Arrow's default memory pool (64-byte aligned).
// Arrow IPC serialization also pads each buffer to 8-byte boundaries, ensuring
// that out-of-bounds vector reads (e.g., uint4 loads in CUDA kernels) stay within
// the allocation. The `raw_values()` pointer from UInt8Array is at least 64-byte aligned.

uint8_t float_to_e4m3_p2(float s)
{
    if (s == 0.0f) return 0;
    uint32_t bits;
    std::memcpy(&bits, &s, sizeof(bits));
    int exp8 = (bits >> 23) & 0xFF;
    // Subnormals (exp8==0) or tiny values that underflow E4M3: clamp to min E4M3 normal
    // Returning 0 would cause div-by-zero downstream (e4m3_p2_to_float(0)=0.0f → inf in quantizer)
    if (exp8 == 0) return 0x08;
    uint32_t mant = bits & 0x7FFFFF;
    int p2_exp = exp8 - 127;
    if (mant != 0) ++p2_exp;
    int e4 = p2_exp + 7;
    if (e4 < 1) e4 = 1;
    if (e4 > 14) e4 = 14;
    return static_cast<uint8_t>(static_cast<uint8_t>(e4) << 3);
}

float e4m3_p2_to_float(uint8_t e4m3)
{
    if (e4m3 == 0) return 0.0f;
    int e4_stored = (e4m3 >> 3) & 0x0F;
    int exp_unb = e4_stored - 7;
    uint32_t bits = static_cast<uint32_t>(exp_unb + 127) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
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
    std::fprintf(stderr, "[DIAG] wrote batch to %s (%d columns, %lld rows)\n",
                 path.string().c_str(),
                 batch->num_columns(),
                 static_cast<long long>(batch->num_rows()));
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

} // namespace lsxcommon