#include "lsxcommon/schema.h"
#include <arrow/type_fwd.h>
#include <arrow/builder.h>
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

} // namespace lsxcommon
