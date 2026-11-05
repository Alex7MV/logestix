// pread_pool.cpp — see pread_pool.h for why the pool exists.
#include "lsxcommon/pread_pool.h"

#include <cerrno>

namespace lsxcommon {
namespace io {

PreadPool::PreadPool(int threads)
    : threads_(threads > 0 ? threads : 4)
{
    workers_.reserve(static_cast<size_t>(threads_));
    for (int i = 0; i < threads_; ++i)
        workers_.emplace_back([this] { WorkerLoop(); });
}

PreadPool::~PreadPool()
{
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        ++generation_;
    }
    cv_start_.notify_all();
    for (auto &w : workers_)
        if (w.joinable()) w.join();
}

bool PreadPool::ReadWhole(const PreadJob &job)
{
    if (job.fd < 0 || job.dst == nullptr) return false;
    auto *p = static_cast<uint8_t *>(job.dst);
    size_t got = 0;
    while (got < job.len) {
        // pread may return less than asked even on a regular file (EINTR, or a
        // concurrent truncate), so one call is not enough — a single call here
        // is how a staging slot ends up half full.
        const ssize_t n = ::pread(job.fd, p + got, job.len - got,
                                  static_cast<off_t>(job.offset +
                                                     static_cast<int64_t>(got)));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  // EOF before len: short read
        got += static_cast<size_t>(n);
    }
    return true;
}

void PreadPool::WorkerLoop()
{
    size_t seen_gen = 0;
    for (;;) {
        const std::vector<PreadJob> *jobs = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_start_.wait(lock, [&] { return stop_ || generation_ != seen_gen; });
            if (stop_ && generation_ == seen_gen) return;
            seen_gen = generation_;
            jobs = jobs_;
        }
        if (jobs != nullptr) {
            for (;;) {
                const size_t i = next_.fetch_add(1);
                if (i >= jobs->size()) break;
                if (ReadWhole((*jobs)[i])) done_.fetch_add(1);
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            ++workers_finished_;
        }
        cv_done_.notify_all();
    }
}

size_t PreadPool::Run(const std::vector<PreadJob> &jobs)
{
    if (jobs.empty()) return 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        if (workers_.empty()) {
            // No workers (construction was asked for nothing valid): do the
            // work here rather than reporting zero, so the pool never silently
            // completes nothing.
            size_t ok = 0;
            for (const auto &j : jobs)
                if (ReadWhole(j)) ++ok;
            return ok;
        }
        jobs_ = &jobs;
        next_.store(0);
        done_.store(0);
        ++generation_;
        workers_finished_ = 0;
    }
    cv_start_.notify_all();

    std::unique_lock<std::mutex> lock(m_);
    const size_t want = workers_.size();
    cv_done_.wait(lock, [&] { return workers_finished_ >= want; });
    jobs_ = nullptr;
    return done_.load();
}

}  // namespace io
}  // namespace lsxcommon
