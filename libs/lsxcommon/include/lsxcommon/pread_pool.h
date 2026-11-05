// pread_pool.h — persistent parallel pread pool for pinned expert staging.
//
// Why a pool and not one thread per call: the expert staging ring fills its
// pinned slots every time it stages an expert (1472 experts/token on a 2.8T
// KimiLinear tree), so spawning threads per fill would pay the creation cost
// 1472 times per token. A single-threaded pread loop is worse still — one NVMe
// stream is latency-bound at ~2.5 GB/s while the array does ~39.8 GB/s at 32
// concurrent cold reads, and a 4-thread pool measured 13.9 GB/s on the real
// expert files (/tmp probe, 2026-11-05).
//
// Host-only: no CUDA, no Arrow, no absl (this header is reachable from .cu TUs,
// where <absl/container/*> breaks NVCC). Plain std:: and POSIX pread.
//
// The pool is NOT internally synchronized for concurrent Run() calls: one
// consumer per instance, same contract as the ExpertStore LRU it serves.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace lsxcommon {
namespace io {

// One copy: read `len` bytes from `fd` at `offset` into `dst`.
// `dst` must stay valid and must not alias another job's destination.
struct PreadJob {
    int fd = -1;
    int64_t offset = 0;
    void *dst = nullptr;
    size_t len = 0;
};

// Fills buffers from file descriptors with N persistent worker threads.
class PreadPool {
public:
    // `threads <= 0` means the documented default (4), never 0 worker threads:
    // a zero-worker pool would silently complete nothing.
    explicit PreadPool(int threads);
    ~PreadPool();

    PreadPool(const PreadPool &) = delete;
    PreadPool &operator=(const PreadPool &) = delete;

    // Fills every job's dst from its fd, blocking until all jobs are done.
    // Returns the number of jobs that read their full `len`. A short read
    // (EOF, truncate) or an error leaves that job's count out and its tail
    // untouched — the caller must treat anything below jobs.size() as "do not
    // use this buffer", because a half-filled staging slot DMA'd to the device
    // is silent garbage.
    size_t Run(const std::vector<PreadJob> &jobs);

    int Threads() const { return threads_; }

private:
    void WorkerLoop();
    // Reads the whole job, retrying EINTR. True only on a full read.
    static bool ReadWhole(const PreadJob &job);

    int threads_ = 4;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    const std::vector<PreadJob> *jobs_ = nullptr;
    std::atomic<size_t> next_{0};
    std::atomic<size_t> done_{0};
    size_t generation_ = 0;     // bumped by each Run to release the workers
    // Workers that have finished the current generation. Run waits for ALL of
    // them, not for "any": a single worker bumping a completion flag would let
    // Run return while its siblings still write into the caller's buffers.
    size_t workers_finished_ = 0;
    bool stop_ = false;
};

}  // namespace io
}  // namespace lsxcommon
