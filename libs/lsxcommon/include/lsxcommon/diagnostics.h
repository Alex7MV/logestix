#pragma once

#include <arrow/array/array_base.h>
#include <arrow/buffer.h>
#include <arrow/device.h>
#include <arrow/device_allocation_type_set.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include "lsxcommon/log.h"
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
    log::debug(std::string("[DIAG] ") + msg);
}

inline void LogBuffer(const char *label, const arrow::Buffer &buf)
{
    auto mm = buf.memory_manager();
    log::debug(std::string("[DIAG] buffer ") + label
        + ": size=" + std::to_string(buf.size())
        + " is_cpu=" + std::to_string(static_cast<int>(buf.is_cpu()))
        + " device=" + MemoryManagerDesc(mm));
}

inline void ValidateCpuBuffer(const arrow::Buffer &buf, const char *context)
{
    if (buf.is_cpu()) return;
    auto mm = buf.memory_manager();
    log::warn(std::string("[DIAG] BUFFER DEVICE MISMATCH [") + context
        + "]: expected CPU buffer, got " + MemoryManagerDesc(mm)
        + " (size=" + std::to_string(buf.size())
        + ", data=" + std::to_string(reinterpret_cast<uintptr_t>(buf.data())) + ")");
}

inline void ValidateBatchBuffers(const arrow::RecordBatch &batch, const char *context)
{
    log::debug(std::string("[DIAG] ValidateBatch [") + context + "]: "
        + std::to_string(batch.num_columns()) + " columns, "
        + std::to_string(static_cast<int>(batch.num_rows())) + " rows");
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
}

inline void LogMemoryPoolState(const char *context)
{
    auto *pool = arrow::default_memory_pool();
    log::debug(std::string("[DIAG] memory pool [") + context + "]: backend="
        + pool->backend_name() + " allocated="
        + std::to_string(pool->bytes_allocated()) + " max_memory="
        + std::to_string(pool->max_memory()));
}

inline void LogArrowSystemState(const char *context)
{
    log::debug(std::string("[DIAG] === Arrow system state [") + context + "] ===");
    LogMemoryPoolState(context);
    auto cpu_mm = arrow::default_cpu_memory_manager();
    log::debug(std::string("[DIAG] default CPU memory manager: ")
        + MemoryManagerDesc(cpu_mm));
    log::debug("[DIAG] ===================================");
}

} // namespace diag
} // namespace lsxcommon
