#pragma once

// Bounded, error-checking wait for a CUDA event.
//
// The MoE router handshake used to spin on cudaEventQuery and discard the
// result (`(void)q`), so a failed kernel was followed by a read of stale
// router slots, and a truly stuck event spun forever. This helper returns the
// first non-NotReady status and a timeout instead (audit X10).

#include <chrono>
#include <thread>

#include <cuda_runtime.h>

namespace lsxcommon {
namespace cuda {

inline cudaError_t WaitEventChecked(cudaEvent_t event,
                                    int timeout_ms = 30000)
{
    if (event == nullptr) return cudaErrorInvalidValue;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const cudaError_t st = cudaEventQuery(event);
        if (st == cudaSuccess) return cudaSuccess;
        if (st != cudaErrorNotReady) return st;
        if (std::chrono::steady_clock::now() >= deadline)
            return cudaErrorLaunchTimeout;
        // `yield` keeps this a spin, not a sleep: the event usually retires in
        // microseconds, and a sleep would add scheduler latency to every layer.
        std::this_thread::yield();
    }
}

} // namespace cuda
} // namespace lsxcommon
