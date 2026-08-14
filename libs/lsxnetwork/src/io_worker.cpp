#include "lsxnetwork/io_worker.h"
#include "lsxnetwork/affinity.h"

#include <spdlog/spdlog.h>

namespace lsxnetwork {

IoWorker::IoWorker(std::function<void()> loop)
    : IoWorker(std::move(loop), std::string{}) {}

IoWorker::IoWorker(std::function<void()> loop, std::string passport_path)
    : loop_(std::move(loop)), passport_path_(std::move(passport_path)) {}

IoWorker::~IoWorker() noexcept {
    Stop();
}

int IoWorker::Start() {
    if (thread_.joinable()) {
        return -1;  // already started; reassigning a joinable std::thread terminates
    }
    running_.store(true, std::memory_order_release);
    pin_result_ = std::promise<int>{};
    auto future = pin_result_.get_future();
    thread_ = std::thread(&IoWorker::Entry, this);
    return future.get();
}

void IoWorker::Stop() noexcept {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void IoWorker::Entry() {
    const int rc = CpuAffinity::PinToCores0To3();
    pinned_.store(rc == 0, std::memory_order_release);

    // Phase 4.5: the worker's private translator is Load()ed natively on the
    // pinned thread. Load maps + validates the passport (CreateFileMapping /
    // mmap) with zero heap allocation on the success path, so the translation
    // burst that follows stays allocation-free. A failed load leaves the
    // translator invalid (all lookups fall back to the passport's genuine unk
    // id), which is a soft degradation, not a hard abort.
    if (!passport_path_.empty()) {
        const int lrc = translator_.Load(passport_path_.c_str());
        if (lrc != 0) {
            spdlog::error("[io_worker] passport '{}' could not be loaded: {}",
                          passport_path_, lrc);
        }
    }

    // Signal readiness AFTER the translator is loaded so a caller's Start()
    // returning guarantees the per-thread mapping is fully established (no
    // race on Translator().Valid() from the observing thread).
    pin_result_.set_value(rc);

    if (running_.load(std::memory_order_acquire)) {
        loop_();
    }
}

}  // namespace lsxnetwork
