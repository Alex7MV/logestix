#pragma once

// Per-stream, per-tag grow-only device scratch.
//
// Several decode hot paths (split-KV attention, router logits) need a small
// temporary every layer/token. cudaMallocAsync/cudaFreeAsync per call is pure
// dispatch overhead, while one process-global buffer is unsafe when two
// streams run concurrently. This keeps one buffer per (stream, tag): a stream
// reuses its own region, two streams never alias, and growth is rare.
//
// The buffer lives until Reset()/process exit. Growth releases the old buffer
// only after cudaFree's implicit device synchronization, so no in-flight kernel
// can still be reading it.

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace lsxcommon {
namespace cuda {

template <typename Tag>
class StreamScratch
{
public:
    static cudaError_t Get(cudaStream_t stream, size_t bytes, void **out)
    {
        if (out == nullptr) return cudaErrorInvalidValue;
        std::lock_guard<std::mutex> lk(Mutex());
        auto &slot = Map()[stream];
        if (slot.second < bytes)
        {
            void *p = nullptr;
            const cudaError_t e = cudaMalloc(&p, bytes == 0 ? 1 : bytes);
            if (e != cudaSuccess) return e;
            if (slot.first) cudaFree(slot.first);
            slot = {p, bytes};
        }
        else if (slot.first == nullptr)
        {
            const cudaError_t e = cudaMalloc(&slot.first, 1);
            if (e != cudaSuccess) return e;
        }
        *out = slot.first;
        return cudaSuccess;
    }

    static void Reset()
    {
        std::lock_guard<std::mutex> lk(Mutex());
        for (auto &[stream, slot] : Map())
        {
            (void)stream;
            if (slot.first) cudaFree(slot.first);
        }
        Map().clear();
    }

private:
    static std::mutex &Mutex()
    {
        static std::mutex m;
        return m;
    }
    static std::unordered_map<cudaStream_t, std::pair<void *, size_t>> &Map()
    {
        static std::unordered_map<cudaStream_t, std::pair<void *, size_t>> map;
        return map;
    }
};

} // namespace cuda
} // namespace lsxcommon
