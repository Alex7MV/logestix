#pragma once

#include <arrow/array/array_base.h>
#include <arrow/buffer.h>
#include <arrow/device.h>
#include <arrow/device_allocation_type_set.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <cstdio>
#include <string>
#include <array>

namespace lsxcommon {
namespace diag {

inline std::string MemoryManagerDesc(const std::shared_ptr<arrow::MemoryManager> &mm)
{
    if (!mm) return "nullptr";
    auto dev = mm->device();
    if (!dev) return "no-device";
    return std::string(arrow::DeviceAllocationTypeToCStr(dev->device_type()))
        + "(" + dev->ToString() + ")";
}

inline void LogToStderr(const char *msg)
{
    std::fprintf(stderr, "[DIAG] %s\n", msg);
    std::fflush(stderr);
}

inline void LogBuffer(const char *label, const arrow::Buffer &buf)
{
    auto mm = buf.memory_manager();
    std::fprintf(stderr,
        "[DIAG] buffer %s: size=%lld is_cpu=%d device=%s\n",
        label,
        static_cast<long long>(buf.size()),
        static_cast<int>(buf.is_cpu()),
        MemoryManagerDesc(mm).c_str());
    std::fflush(stderr);
}

inline void ValidateCpuBuffer(const arrow::Buffer &buf, const char *context)
{
    if (buf.is_cpu()) return;
    auto mm = buf.memory_manager();
    std::fprintf(stderr,
        "[DIAG] BUFFER DEVICE MISMATCH [%s]: expected CPU buffer, got %s "
        "(size=%lld, data=%p)\n",
        context,
        MemoryManagerDesc(mm).c_str(),
        static_cast<long long>(buf.size()),
        static_cast<const void*>(buf.data()));
    std::fflush(stderr);
}

inline void ValidateBatchBuffers(const arrow::RecordBatch &batch, const char *context)
{
    std::fprintf(stderr, "[DIAG] ValidateBatch [%s]: %d columns, %d rows\n",
        context, batch.num_columns(), static_cast<int>(batch.num_rows()));
    for (int col = 0; col < batch.num_columns(); ++col) {
        auto arr = batch.column(col);
        auto data = arr->data();
        for (std::size_t b = 0; b < data->buffers.size(); ++b) {
            auto buf = data->buffers[b];
            if (buf) {
                ValidateCpuBuffer(*buf, context);
            }
        }
    }
    std::fflush(stderr);
}

inline void LogMemoryPoolState(const char *context)
{
    auto *pool = arrow::default_memory_pool();
    std::fprintf(stderr,
        "[DIAG] memory pool [%s]: backend=%s allocated=%lld max_memory=%lld\n",
        context,
        pool->backend_name().c_str(),
        static_cast<long long>(pool->bytes_allocated()),
        static_cast<long long>(pool->max_memory()));
    std::fflush(stderr);
}

inline void LogArrowSystemState(const char *context)
{
    std::fprintf(stderr, "[DIAG] === Arrow system state [%s] ===\n", context);
    LogMemoryPoolState(context);
    auto cpu_mm = arrow::default_cpu_memory_manager();
    std::fprintf(stderr, "[DIAG] default CPU memory manager: %s\n",
        MemoryManagerDesc(cpu_mm).c_str());
    std::fprintf(stderr, "[DIAG] ===================================\n");
    std::fflush(stderr);
}

} // namespace diag
} // namespace lsxcommon
