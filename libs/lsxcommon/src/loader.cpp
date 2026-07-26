#include "lsxcommon/loader.h"
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <spdlog/spdlog.h>
#include <cstring>

#ifdef LSXCOMMON_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lsxcommon {

namespace {

arrow::Result<LoadedExpert> extract_from_batch(
    std::shared_ptr<arrow::RecordBatch> batch,
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap)
{
    LoadedExpert exp;
    exp.batch = std::move(batch);
    exp.mmap  = std::move(mmap);

    auto get_binary = [&](int idx, const uint8_t *&ptr, int64_t &sz)
        -> arrow::Status
    {
        auto arr = exp.batch->column(idx);
        auto bin = std::dynamic_pointer_cast<arrow::BinaryArray>(arr);
        if (!bin)
            return arrow::Status::TypeError("Column ", idx, " is not binary");

        auto view = bin->GetView(0);
        ptr = reinterpret_cast<const uint8_t *>(view.data());
        sz  = view.size();
        return arrow::Status::OK();
    };

    auto get_scales = [&](int idx, const float *&ptr, int64_t &cnt)
        -> arrow::Status
    {
        auto arr = exp.batch->column(idx);
        auto lst = std::dynamic_pointer_cast<arrow::ListArray>(arr);
        if (!lst)
            return arrow::Status::TypeError("Column ", idx, " is not list");

        cnt = lst->raw_value_offsets()[1] - lst->raw_value_offsets()[0];

        auto flat = lst->values();
        auto flt = std::dynamic_pointer_cast<arrow::FloatArray>(flat);
        if (!flt)
            return arrow::Status::TypeError("Scale list is not float32");

        ptr = flt->raw_values() + lst->raw_value_offsets()[0];
        return arrow::Status::OK();
    };

    {
        auto st = get_binary(ExpertSchema::kIdxW1Weight,
                             exp.w1_weight, exp.w1_weight_size);
        if (!st.ok()) return st;
    }
    {
        auto st = get_binary(ExpertSchema::kIdxW2Weight,
                             exp.w2_weight, exp.w2_weight_size);
        if (!st.ok()) return st;
    }
    {
        auto st = get_binary(ExpertSchema::kIdxW3Weight,
                             exp.w3_weight, exp.w3_weight_size);
        if (!st.ok()) return st;
    }
    {
        auto st = get_scales(ExpertSchema::kIdxW1Scale,
                             exp.w1_scale, exp.w1_scale_count);
        if (!st.ok()) return st;
    }
    {
        auto st = get_scales(ExpertSchema::kIdxW2Scale,
                             exp.w2_scale, exp.w2_scale_count);
        if (!st.ok()) return st;
    }
    {
        auto st = get_scales(ExpertSchema::kIdxW3Scale,
                             exp.w3_scale, exp.w3_scale_count);
        if (!st.ok()) return st;
    }

    return exp;
}

} // anonymous namespace

Loader::~Loader() = default;

arrow::Result<LoadedExpert>
Loader::LoadExpertFromArrow(const std::string &arrow_path)
{
    std::shared_ptr<arrow::io::MemoryMappedFile> mmap;
    {
        auto result = arrow::io::MemoryMappedFile::Open(arrow_path,
                                                         arrow::io::FileMode::READ);
        if (!result.ok())
            return result.status();
        mmap = std::move(result).ValueOrDie();
    }

    std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
    {
        auto result = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
        if (!result.ok())
            return result.status();
        reader = std::move(result).ValueOrDie();
    }

    if (reader->num_record_batches() == 0)
        return arrow::Status::Invalid("no record batches in ", arrow_path);

    std::shared_ptr<arrow::RecordBatch> batch;
    {
        auto result = reader->ReadRecordBatch(0);
        if (!result.ok())
            return result.status();
        batch = std::move(result).ValueOrDie();
    }

    return extract_from_batch(std::move(batch), std::move(mmap));
}

arrow::Status Loader::RegisterCudaPinned(LoadedExpert &expert)
{
#ifdef LSXCOMMON_HAS_CUDA
    auto reg = [](void *ptr, std::size_t size) -> cudaError_t {
        if (!ptr || size == 0) return cudaSuccess;
        return cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    };

    // Each Arrow buffer within the RecordBatch is already page-aligned
    // (Arrow guarantees 64-byte alignment by default)
    auto batch = expert.batch;
    for (int i = 0; i < batch->num_columns(); ++i) {
        auto arr = batch->column(i);
        for (int b = 0; b < arr->num_buffers(); ++b) {
            auto buf = arr->data()->buffers[b];
            if (buf && buf->size() > 0) {
                cudaError_t e = reg(buf->mutable_data(), buf->size());
                if (e != cudaSuccess) {
                    UnregisterCudaPinned(expert);
                    return arrow::Status::IOError(
                        "cudaHostRegister failed: ", cudaGetErrorString(e));
                }
            }
        }
    }

    expert.cuda_registered = true;
    spdlog::debug("Registered {} buffers as CUDA pinned memory",
                  batch->num_columns());

#else
    (void)expert;
    return arrow::Status::NotImplemented(
        "lsxcommon built without CUDA support");
#endif
    return arrow::Status::OK();
}

void Loader::UnregisterCudaPinned(LoadedExpert &expert)
{
#ifdef LSXCOMMON_HAS_CUDA
    if (!expert.cuda_registered) return;

    auto batch = expert.batch;
    for (int i = 0; i < batch->num_columns(); ++i) {
        auto arr = batch->column(i);
        for (int b = 0; b < arr->num_buffers(); ++b) {
            auto buf = arr->data()->buffers[b];
            if (buf && buf->size() > 0) {
                cudaError_t e = cudaHostUnregister(buf->mutable_data());
                if (e != cudaSuccess) {
                    spdlog::warn("cudaHostUnregister failed: {}",
                                  cudaGetErrorString(e));
                }
            }
        }
    }
    expert.cuda_registered = false;
#else
    (void)expert;
#endif
}

} // namespace lsxcommon
